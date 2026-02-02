import os
import logging
import copy
import glob
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.reconstruct_dataset import get_reconstruct_dataloader
from models.Reconstruction.model import ReconstructionModel
from models.MIL.model import MILModel

logger = logging.getLogger(__name__)

def train_reconstruction_module(
    mil_model: MILModel,
    recon_model: ReconstructionModel,
    clinical_file: str,
    feature_dir: str,
    output_dir: str,
    n_epochs: int = 100,
    lr: float = 1e-3
):
    logger.info("=== Starting Stage 2: Training Reconstruction Module ===")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs_recon'))
    device = next(recon_model.parameters()).device

    # 1. setup model
    mil_model.eval() #freeze mil
    for param in mil_model.parameters():
        param.requires_grad = False 

    recon_model.train()

    # 2. Loader
    train_dataloader = get_reconstruct_dataloader(clinical_file, feature_dir, split='train')
    val_dataloader = get_reconstruct_dataloader(clinical_file, feature_dir, split='val', shuffle=False)

    # 3. Optimizer & Load
    optimizer = optim.AdamW(recon_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss(reduction='none')

    best_val_loss = float('inf')
    patience_limit = 20
    patience_counter = 0

    for epoch in range(n_epochs):
        #TRAINING
        recon_model.train()
        running_loss = 0.0
        train_steps = 0

        for data in train_dataloader:
            wsi_bag = [f.to(device) for f in data['wsi_bag']]
            ct_bag = [f.to(device) for f in data['ct_bag']]
            mri_bag = [f.to(device) for f in data['mri_bag']]
            cli_feat = data['cli_feat'].to(device)

            masks = torch.tensor([
                data['wsi_mask'], data['ct_mask'], data['mri_mask'], data['cli_mask']
            ]).float().to(device)

            # Step 1: MIL (Freeze)
            with torch.no_grad():
                # WSI
                _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, modality='WSI', add_noise=False)
                # CT
                _, ct_best, _ = mil_model.forward_single_bag(ct_bag, modality='CT', add_noise=False)
                # MRI
                _, mri_best, _ = mil_model.forward_single_bag(mri_bag, modality='MRI', add_noise=False)
                
                # Đảm bảo shape (1, 768)
                if wsi_best.dim() == 1: 
                    wsi_best = wsi_best.unsqueeze(0)
                if ct_best.dim() == 1: 
                    ct_best = ct_best.unsqueeze(0)
                if mri_best.dim() == 1: 
                    mri_best = mri_best.unsqueeze(0)
                if cli_feat.dim() == 1: 
                    cli_feat = cli_feat.unsqueeze(0)

                # Nếu modality bị thiếu (mask=0), thay thế vector tốt nhất bằng 0 
                # để input vào Recon đúng chuẩn
                if data['wsi_mask'] == 0: 
                    wsi_best = torch.zeros_like(wsi_best)
                if data['ct_mask'] == 0:  
                    ct_best  = torch.zeros_like(ct_best)
                if data['mri_mask'] == 0: 
                    mri_best = torch.zeros_like(mri_best)
                if data['cli_mask'] == 0: 
                    cli_feat = torch.zeros_like(cli_feat)

            # Step 2: Reconstruction
            optimizer.zero_grad(set_to_none=True)

            rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(wsi_best, ct_best, mri_best, cli_feat)

            # Step 3: Loss calc
            loss_wsi = (criterion(rec_wsi, wsi_best).mean(dim=1) * masks[0]).mean()
            loss_ct  = (criterion(rec_ct, ct_best).mean(dim=1)  * masks[1]).mean()
            loss_mri = (criterion(rec_mri, mri_best).mean(dim=1) * masks[2]).mean()
            loss_cli = (criterion(rec_cli, cli_feat).mean(dim=1) * masks[3]).mean()

            loss = loss_wsi + loss_ct + loss_mri + loss_cli

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1

        avg_train_loss = running_loss / train_steps if train_steps > 0 else 0

        # VALIDATION
        recon_model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for data in val_dataloader:
                wsi_bag = [f.to(device) for f in data['wsi_bag']]
                ct_bag  = [f.to(device) for f in data['ct_bag']]
                mri_bag = [f.to(device) for f in data['mri_bag']]
                cli_feat = data['cli_feat'].to(device)
                
                masks = torch.tensor([data['wsi_mask'], data['ct_mask'], data['mri_mask'], data['cli_mask']]).float().to(device)

                # MIL Select (Frozen)
                _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, modality='WSI', add_noise=False)
                _, ct_best, _ = mil_model.forward_single_bag(ct_bag, modality='CT', add_noise=False)
                _, mri_best, _ = mil_model.forward_single_bag(mri_bag, modality='MRI', add_noise=False)

                if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
                if ct_best.dim() == 1: ct_best = ct_best.unsqueeze(0)
                if mri_best.dim() == 1: mri_best = mri_best.unsqueeze(0)
                if cli_feat.dim() == 1: cli_feat = cli_feat.unsqueeze(0)
                
                # Zero out missing for input
                if data['wsi_mask'] == 0: wsi_best = torch.zeros_like(wsi_best)
                if data['ct_mask'] == 0:  ct_best  = torch.zeros_like(ct_best)
                if data['mri_mask'] == 0: mri_best = torch.zeros_like(mri_best)
                if data['cli_mask'] == 0: cli_feat = torch.zeros_like(cli_feat)

                # Recon Forward (training=False -> No random drop, just reconstruct missing)
                rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(wsi_best, ct_best, mri_best, cli_feat)

                loss_wsi = (criterion(rec_wsi, wsi_best).mean(dim=1) * masks[0]).mean()
                loss_ct  = (criterion(rec_ct, ct_best).mean(dim=1)  * masks[1]).mean()
                loss_mri = (criterion(rec_mri, mri_best).mean(dim=1) * masks[2]).mean()
                loss_cli = (criterion(rec_cli, cli_feat).mean(dim=1) * masks[3]).mean()

                val_loss += (loss_wsi + loss_ct + loss_mri + loss_cli).item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else  0

        logger.info(f"Epoch {epoch+1}/{n_epochs} | Recon Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Reconstruction/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Reconstruction/Val_Loss', avg_val_loss, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(recon_model.state_dict(), os.path.join(output_dir, 'best_reconstruction.pth'))
            logger.info("  -> Best Reconstruction Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info("Early stopping triggered.")
                break

    writer.close()
    logger.info("Reconstruction Training Complete.")
