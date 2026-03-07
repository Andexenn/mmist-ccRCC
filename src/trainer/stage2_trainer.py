import os
import logging
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.Fusion.model import Fusion
from models.MIL.model import MILModel
from models.Reconstruction.model import ReconstructionModel
from dataset.reconstruct_dataset import get_reconstruct_dataloader

logger = logging.getLogger(__name__)

def train_pipeline(
    mil_model: MILModel,
    recon_model: ReconstructionModel,
    fusion_model: Fusion,
    clinical_file: str,
    feature_dir: str,
    output_dir: str,
    bacc_mods: List = [1.0, 1.0, 1.0, 1.0],
    epochs: int = 100,
    lr: float = 1e-5,
    death_weight: float = 1.0
):
    """
    Stage 2: Finetune the entire pipeline end-to-end.
    Unfreeze all modules and train with a single optimizer.
    """
    logger.info("=== Starting Stage 2: Finetuning Pipeline End-to-End ===")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs_pipeline'))
    
    device = next(fusion_model.parameters()).device

    # Unfreeze all models
    mil_model.train()
    for param in mil_model.parameters():
        param.requires_grad = True

    recon_model.train()
    for param in recon_model.parameters():
        param.requires_grad = True

    fusion_model.train()
    for param in fusion_model.parameters():
        param.requires_grad = True

    # Dataloaders
    train_loader = get_reconstruct_dataloader(clinical_file, feature_dir, split='train')
    val_loader = get_reconstruct_dataloader(clinical_file, feature_dir, split='val', shuffle=False)

    # Gather all parameters
    all_params = list(mil_model.parameters()) + list(recon_model.parameters()) + list(fusion_model.parameters())

    # Optimizer, Scheduler, Loss (as per specs: AdamW lr=1e-5, patience=5)
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.BCELoss(reduction='none')

    # Early stopping (patience=20)
    best_val_loss = float('inf')
    patience_limit = 20
    patience_counter = 0

    for epoch in range(epochs):
        mil_model.train()
        recon_model.train()
        fusion_model.train()

        running_loss = 0.0
        train_steps = 0

        for data in train_loader:
            wsi_bag = [f.to(device) for f in data['wsi_bag']]
            ct_bag  = [f.to(device) for f in data['ct_bag']]
            mri_bag = [f.to(device) for f in data['mri_bag']]
            cli_feat = data['cli_feat'].to(device)
            
            if 'label' in data:
                label = data['label'].float().to(device)
            else:
                continue

            masks = [
                data['wsi_mask'].to(device),
                data['ct_mask'].to(device),
                data['mri_mask'].to(device),
                data['cli_mask'].to(device)
            ]

            optimizer.zero_grad(set_to_none=True)

            # === MIL Selection ===
            _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, modality='WSI', add_noise=True)
            _, ct_best, _ = mil_model.forward_single_bag(ct_bag, modality='CT', add_noise=True)
            _, mri_best, _ = mil_model.forward_single_bag(mri_bag, modality='MRI', add_noise=True)

            if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
            if ct_best.dim() == 1: ct_best = ct_best.unsqueeze(0)
            if mri_best.dim() == 1: mri_best = mri_best.unsqueeze(0)
            if cli_feat.dim() == 1: cli_feat = cli_feat.unsqueeze(0)

            in_wsi = wsi_best.clone() * masks[0]
            in_ct  = ct_best.clone()  * masks[1]
            in_mri = mri_best.clone() * masks[2]
            in_cli = cli_feat.clone() * masks[3]

            # === Reconstruction ===
            rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(in_wsi, in_ct, in_mri, in_cli)

            final_wsi = masks[0] * wsi_best + (1 - masks[0]) * rec_wsi
            final_ct  = masks[1] * ct_best  + (1 - masks[1]) * rec_ct
            final_mri = masks[2] * mri_best + (1 - masks[2]) * rec_mri
            final_cli = masks[3] * cli_feat + (1 - masks[3]) * rec_cli

            feature_list = [final_wsi, final_ct, final_mri, final_cli]
            full_masks = [torch.ones_like(m) for m in masks]

            # === Fusion ===
            prob = fusion_model(feature_list, full_masks, bacc_mods)

            prob = prob.squeeze()
            label = label.squeeze()

            loss_unreduced = criterion(prob, label)
            weight = death_weight if label.item() == 0 else 1.0
            loss = (loss_unreduced * weight).mean()

            # End-to-end backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1

        avg_train_loss = running_loss / train_steps if train_steps > 0 else 0.0

        # === Validation ===
        mil_model.eval()
        recon_model.eval()
        fusion_model.eval()

        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for data in val_loader:
                wsi_bag = [f.to(device) for f in data['wsi_bag']]
                ct_bag  = [f.to(device) for f in data['ct_bag']]
                mri_bag = [f.to(device) for f in data['mri_bag']]
                cli_feat = data['cli_feat'].to(device)
                
                if 'label' in data:
                    label = data['label'].float().to(device)
                else:
                    continue

                masks = [
                    data['wsi_mask'].to(device),
                    data['ct_mask'].to(device),
                    data['mri_mask'].to(device),
                    data['cli_mask'].to(device)
                ]

                _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, 'WSI', add_noise=False)
                _, ct_best, _  = mil_model.forward_single_bag(ct_bag, 'CT', add_noise=False)
                _, mri_best, _ = mil_model.forward_single_bag(mri_bag, 'MRI', add_noise=False)

                if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
                if ct_best.dim() == 1: ct_best = ct_best.unsqueeze(0)
                if mri_best.dim() == 1: mri_best = mri_best.unsqueeze(0)
                if cli_feat.dim() == 1: cli_feat = cli_feat.unsqueeze(0)

                in_wsi = wsi_best * masks[0]
                in_ct  = ct_best  * masks[1]
                in_mri = mri_best * masks[2]
                in_cli = cli_feat * masks[3]

                rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(in_wsi, in_ct, in_mri, in_cli)

                final_wsi = masks[0] * wsi_best + (1 - masks[0]) * rec_wsi
                final_ct  = masks[1] * ct_best  + (1 - masks[1]) * rec_ct
                final_mri = masks[2] * mri_best + (1 - masks[2]) * rec_mri
                final_cli = masks[3] * cli_feat + (1 - masks[3]) * rec_cli

                feature_list = [final_wsi, final_ct, final_mri, final_cli]
                full_masks = [torch.ones_like(m) for m in masks]

                prob = fusion_model(feature_list, full_masks, bacc_mods)
                loss = criterion(prob.squeeze(), label.squeeze())
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0

        logger.info(f"Epoch {epoch+1}/{epochs} | Pipeline Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Pipeline/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Pipeline/Val_Loss', avg_val_loss, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save the final best ensemble
            torch.save({
                'mil_state_dict': mil_model.state_dict(),
                'recon_state_dict': recon_model.state_dict(),
                'fusion_state_dict': fusion_model.state_dict()
            }, os.path.join(output_dir, 'best_pipeline.pth'))
            
            logger.info("  -> Best Pipeline Checkpoint Saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info("Early stopping triggered in pipeline finetuning.")
                break

    writer.close()
    logger.info("Pipeline Finetuning Complete.")
