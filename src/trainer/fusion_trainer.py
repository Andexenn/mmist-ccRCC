"""
Fusion Trainer — Stage 1, Step 3
==================================
Trains the multi-modal fusion module (early or late fusion).
MIL and Reconstruction models are frozen.
Uses BCE loss with weighted minority class.
"""

import os
import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from configs.logging_config import get_logger
from configs.paths import TB_FUSION_DIR, CHECKPOINT_DIR, CKPT_FUSION
from models.Fusion.model import Fusion
from models.MIL.model import MILModel
from models.Reconstruction.model import ReconstructionModel
from dataset.reconstruct_dataset import get_reconstruct_dataloader

logger = get_logger('fusion_trainer')


def train_fuse_module(
    mil_model: MILModel,
    recon_model: ReconstructionModel,
    fusion_model: Fusion,
    clinical_file: str,
    feature_dir: str,
    fusion_strategy: str = 'early_mean',
    epochs: int = 100,
    lr: float = 1e-3,
    bacc_mods: List = [1.0, 1.0, 1.0, 1.0],
    death_weight: float = 1.0
):
    """Train the fusion module with MIL and Reconstruction frozen."""

    logger.info("=" * 60)
    logger.info("STAGE 1 — STEP 3/3: Fusion Training")
    logger.info("  strategy=%s | epochs=%d | lr=%.2e | death_weight=%.1f",
                fusion_strategy, epochs, lr, death_weight)
    logger.info("=" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TB_FUSION_DIR)
    device = next(fusion_model.parameters()).device

    # ─── Freeze MIL and Reconstruction ───────────────────────────
    mil_model.to(device)
    mil_model.eval()
    for param in mil_model.parameters():
        param.requires_grad = False
    logger.info("MIL model frozen (%d params)", sum(p.numel() for p in mil_model.parameters()))

    recon_model.to(device)
    recon_model.eval()
    for param in recon_model.parameters():
        param.requires_grad = False
    logger.info("Reconstruction model frozen (%d params)", sum(p.numel() for p in recon_model.parameters()))

    fusion_model.to(device)
    fusion_model.train()
    trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    logger.info("Fusion model trainable params: %d", trainable_params)

    # ─── DataLoaders ─────────────────────────────────────────────
    train_loader = get_reconstruct_dataloader(clinical_file, feature_dir, split='train')
    val_loader = get_reconstruct_dataloader(clinical_file, feature_dir, split='val', shuffle=False)

    # ─── Optimizer & Scheduler ───────────────────────────────────
    optimizer = optim.AdamW(fusion_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.BCELoss(reduction='none')
    logger.info("Optimizer: AdamW(lr=%.2e) | Scheduler: ReduceLROnPlateau(factor=0.5, patience=5)", lr)

    # ─── Training Loop ───────────────────────────────────────────
    best_val_loss = float('inf')
    patience_limit = 20
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        fusion_model.train()
        running_loss = 0.0
        train_steps = 0

        for data in train_loader:
            wsi_bag = [f.to(device) for f in data['wsi_bag']]
            ct_bag  = [f.to(device) for f in data['ct_bag']]
            mri_bag = [f.to(device) for f in data['mri_bag']]
            cli_feat = data['cli_feat'].to(device)
            label = torch.tensor([1.0]).to(device)

            if 'label' in data:
                label = data['label'].float().to(device)
            else:
                logger.debug("Sample missing 'label' key, using default=1.0")

            # Masks: convert to tensors properly (fix for plain int from dataset)
            masks = [
                torch.tensor([[data['wsi_mask']]]).float().to(device),
                torch.tensor([[data['ct_mask']]]).float().to(device),
                torch.tensor([[data['mri_mask']]]).float().to(device),
                torch.tensor([[data['cli_mask']]]).float().to(device),
            ]

            optimizer.zero_grad()

            # ─── Step 1: Feature extraction (frozen) ─────────────
            with torch.no_grad():
                _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, modality='WSI', add_noise=False)
                _, ct_best, _  = mil_model.forward_single_bag(ct_bag, modality='CT', add_noise=False)
                _, mri_best, _ = mil_model.forward_single_bag(mri_bag, modality='MRI', add_noise=False)

                if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
                if ct_best.dim() == 1:  ct_best = ct_best.unsqueeze(0)
                if mri_best.dim() == 1: mri_best = mri_best.unsqueeze(0)
                if cli_feat.dim() == 1: cli_feat = cli_feat.unsqueeze(0)

                in_wsi = wsi_best.clone() * masks[0]
                in_ct  = ct_best.clone()  * masks[1]
                in_mri = mri_best.clone() * masks[2]
                in_cli = cli_feat.clone() * masks[3]

                # Reconstruction forward (fill missing)
                rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(in_wsi, in_ct, in_mri, in_cli)

            # ─── Step 2: Prepare fusion input ─────────────────────
            final_wsi = masks[0] * wsi_best + (1 - masks[0]) * rec_wsi
            final_ct  = masks[1] * ct_best  + (1 - masks[1]) * rec_ct
            final_mri = masks[2] * mri_best + (1 - masks[2]) * rec_mri
            final_cli = masks[3] * cli_feat + (1 - masks[3]) * rec_cli

            feature_list = [final_wsi, final_ct, final_mri, final_cli]
            full_masks = [torch.ones_like(m) for m in masks]

            # ─── Step 3: Fusion & Loss ────────────────────────────
            prob = fusion_model(feature_list, full_masks, bacc_mods)
            prob = prob.squeeze()
            label = label.squeeze()

            loss_unreduced = criterion(prob, label)
            weight = death_weight if label.item() == 0 else 1.0
            loss = (loss_unreduced * weight).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1

        avg_train_loss = running_loss / train_steps if train_steps > 0 else 0.0

        # ═══ VALIDATION ══════════════════════════════════════════
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
                    torch.tensor([[data['wsi_mask']]]).float().to(device),
                    torch.tensor([[data['ct_mask']]]).float().to(device),
                    torch.tensor([[data['mri_mask']]]).float().to(device),
                    torch.tensor([[data['cli_mask']]]).float().to(device),
                ]

                _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, 'WSI')
                _, ct_best, _  = mil_model.forward_single_bag(ct_bag, 'CT')
                _, mri_best, _ = mil_model.forward_single_bag(mri_bag, 'MRI')

                if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
                if ct_best.dim() == 1:  ct_best = ct_best.unsqueeze(0)
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
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # ─── Logging ────────────────────────────────────────────
        logger.info(
            "[Epoch %03d/%03d] [Fusion] train_loss=%.4f | val_loss=%.4f | best_val=%.4f | "
            "lr=%.2e | patience=%d/%d | time=%.1fs",
            epoch + 1, epochs,
            avg_train_loss, avg_val_loss, best_val_loss,
            current_lr, patience_counter, patience_limit, elapsed
        )
        writer.add_scalar('Fusion/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Fusion/Val_Loss', avg_val_loss, epoch)
        writer.add_scalar('Fusion/LR', current_lr, epoch)

        scheduler.step(avg_val_loss)

        # ─── Checkpointing ──────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(CHECKPOINT_DIR, CKPT_FUSION)
            torch.save(fusion_model.state_dict(), save_path)
            logger.info("  >> Checkpoint saved: %s (val_loss=%.4f)", save_path, best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.warning(
                    "[Fusion] Early stopping at epoch %d — no improvement for %d epochs "
                    "(best_val=%.4f, current_val=%.4f)",
                    epoch + 1, patience_limit, best_val_loss, avg_val_loss
                )
                break

    writer.close()
    logger.info("STAGE 1 — STEP 3/3: Fusion Training COMPLETE (best_val=%.4f)", best_val_loss)
