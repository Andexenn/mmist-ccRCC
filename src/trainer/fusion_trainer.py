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

def train_fuse_module(
    mil_model: MILModel,
    recon_model: ReconstructionModel,
    fusion_model: Fusion,
    clinical_file: str,
    feature_dir: str,
    output_dir: str,
    fusion_strategy: str = 'early_mean',
    epochs: int = 100,
    lr: float = 1e-3,
    bacc_mods: List = [1.0, 1.0, 1.0, 1.0],
    pos_weight = 1
):
    """ Train fuse module """
    logger.info("=== Starting Stage 3: Training Fusion Module ===")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs_recon'))
    device = next(fusion_model.parameters()).device

    #1. Freeze MIL and reconstruction module
    mil_model.eval()
    for param in mil_model.parameters():
        param.to(device)

    recon_model.eval()
    for param in recon_model.parameters():
        param.to(device)

    fusion_model.train()

    #2. dataloader
    train_loader = get_reconstruct_dataloader(clinical_file, feature_dir, split='train')
    val_loader = get_reconstruct_dataloader(clinical_file, feature_dir, 'val', shuffle=False)

    #3. optimizer
    optimizer = optim.AdamW(fusion_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.BCELoss(reduction='none')

    #4. training loop
    best_val_loss = float('inf')
    patience_limit = 20
    patience_counter = 0

    for epoch in range(epochs):
        fusion_model.train()
        running_loss = 0
        train_steps = 0
        for batch_idx, data in train_loader:
            wsi_bag = [f.to(device) for f in data['wsi_bag']]
            ct_bag  = [f.to(device) for f in data['ct_bag']]
            mri_bag = [f.to(device) for f in data['mri_bag']]
            cli_feat = data['cli_feat'].to(device) 
            label = torch.tensor([1.0]).to(device) 

            if 'label' in data:
                label = data['label'].float().to(device)
            else:
                pass

            masks = [
                data['wsi_mask'].to(device),
                data['ct_mask'].to(device),
                data['mri_mask'].to(device),
                data['cli_mask'].to(device)
            ]

            optimizer.zero_grad()

            # === BƯỚC 1: TRÍCH XUẤT FEATURE (Frozen) ===
            with torch.no_grad():
                # MIL Selection
                _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, modality='WSI', add_noise=False)
                _, ct_best, _ = mil_model.forward_single_bag(ct_bag, modality='CT', add_noise=False)
                _, mri_best, _ = mil_model.forward_single_bag(mri_bag, modality='MRI', add_noise=False)

                # Chuẩn hóa shape (1, Dim)
                if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
                if ct_best.dim() == 1: ct_best = ct_best.unsqueeze(0)
                if mri_best.dim() == 1: mri_best = mri_best.unsqueeze(0)
                if cli_feat.dim() == 1: cli_feat = cli_feat.unsqueeze(0)

                in_wsi = wsi_best.clone() * masks[0]
                in_ct  = ct_best.clone()  * masks[1]
                in_mri = mri_best.clone() * masks[2]
                in_cli = cli_feat.clone() * masks[3]

                # Reconstruction Forward
                # Model Recon sẽ trả về feature đã tái tạo cho TẤT CẢ modality
                rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(in_wsi, in_ct, in_mri, in_cli)

            # === BƯỚC 2: CHUẨN BỊ INPUT CHO FUSION ===
            final_wsi = masks[0] * wsi_best + (1 - masks[0]) * rec_wsi
            final_ct  = masks[1] * ct_best  + (1 - masks[1]) * rec_ct
            final_mri = masks[2] * mri_best + (1 - masks[2]) * rec_mri
            final_cli = masks[3] * cli_feat + (1 - masks[3]) * rec_cli

            feature_list = [final_wsi, final_ct, final_mri, final_cli]

            full_masks = [torch.ones_like(m) for m in masks]

            # === BƯỚC 3: FUSION & LOSS ===
            # Forward pass (Gradient bắt đầu được tính từ đây)
            prob = fusion_model(feature_list, full_masks, bacc_mods)
            
            # Đảm bảo prob shape [1, 1] -> [1]
            prob = prob.squeeze()
            label = label.squeeze()

            loss = criterion(prob, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1

        avg_train_loss = running_loss / train_steps if train_steps > 0 else 0.0

        # === VALIDATION ===
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

                masks = [data['wsi_mask'].to(device), data['ct_mask'].to(device), 
                         data['mri_mask'].to(device), data['cli_mask'].to(device)]

                # 1. MIL (Best select)
                _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, 'WSI')
                _, ct_best, _  = mil_model.forward_single_bag(ct_bag, 'CT')
                _, mri_best, _ = mil_model.forward_single_bag(mri_bag, 'MRI')
                
                if wsi_best.dim()==1: wsi_best = wsi_best.unsqueeze(0)
                if ct_best.dim()==1: ct_best = ct_best.unsqueeze(0)
                if mri_best.dim()==1: mri_best = mri_best.unsqueeze(0)
                if cli_feat.dim()==1: cli_feat = cli_feat.unsqueeze(0)

                # 2. Recon (Fill missing)
                in_wsi = wsi_best * masks[0]
                in_ct  = ct_best  * masks[1]
                in_mri = mri_best * masks[2]
                in_cli = cli_feat * masks[3]
                
                rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(in_wsi, in_ct, in_mri, in_cli)
                
                # 3. Combine
                final_wsi = masks[0] * wsi_best + (1 - masks[0]) * rec_wsi
                final_ct  = masks[1] * ct_best  + (1 - masks[1]) * rec_ct
                final_mri = masks[2] * mri_best + (1 - masks[2]) * rec_mri
                final_cli = masks[3] * cli_feat + (1 - masks[3]) * rec_cli
                
                feature_list = [final_wsi, final_ct, final_mri, final_cli]
                full_masks = [torch.ones_like(m) for m in masks]

                # 4. Predict
                prob = fusion_model(feature_list, full_masks, bacc_mods)
                loss = criterion(prob.squeeze(), label.squeeze())
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0

        logger.info(f"Epoch {epoch+1}/{n_epochs} | Fusion Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Fusion/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Fusion/Val_Loss', avg_val_loss, epoch)

        # Scheduler Step
        scheduler.step(avg_val_loss)

        # Early Stopping & Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(fusion_model.state_dict(), os.path.join(output_dir, 'best_fusion_model.pth'))
            logger.info("  -> Best Fusion Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info("Early stopping triggered.")
                break

    writer.close()
    logger.info("Fusion Training Complete.")

