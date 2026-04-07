"""
Reconstruction Trainer — Stage 1, Step 2
=========================================
Trains the encoder-decoder reconstruction module to recover missing modalities.
MIL model is frozen; only reconstruction parameters are updated.
Uses MSE loss, CosineAnnealingLR scheduler, and 6x minority oversampling.
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from configs.logging_config import get_logger
from configs.paths import TB_RECON_DIR, CHECKPOINT_DIR, CKPT_RECON
from dataset.reconstruct_dataset import get_reconstruct_dataloader
from models.Reconstruction.model import ReconstructionModel
from models.MIL.model import MILModel

logger = get_logger('reconstruction_trainer')


def train_reconstruction_module(
    mil_model: MILModel,
    recon_model: ReconstructionModel,
    clinical_file: str,
    feature_dir: str,
    n_epochs: int = 600,
    lr: float = 1e-3,
    batch_size: int = 14,
    train_split: str = 'train',
    val_split: str = 'val'
):
    """
    Train the missing modality reconstruction module.

    The MIL model is frozen — its selected features are used as input/target
    for the reconstruction encoder-decoder.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1 — STEP 2/3: Reconstruction Training")
    logger.info("  epochs=%d | lr=%.2e | batch_size=%d", n_epochs, lr, batch_size)
    logger.info("=" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TB_RECON_DIR)
    device = next(recon_model.parameters()).device

    # ─── Freeze MIL ──────────────────────────────────────────────
    mil_model.eval()
    for param in mil_model.parameters():
        param.requires_grad = False
    logger.info("MIL model frozen (%d params)", sum(p.numel() for p in mil_model.parameters()))

    recon_model.train()
    trainable_params = sum(p.numel() for p in recon_model.parameters() if p.requires_grad)
    logger.info("Reconstruction model trainable params: %d", trainable_params)

    # ─── DataLoaders ─────────────────────────────────────────────
    train_dataloader = get_reconstruct_dataloader(
        clinical_file, feature_dir, split=train_split, batch_size=batch_size
    )
    val_dataloader = get_reconstruct_dataloader(
        clinical_file, feature_dir, split=val_split, shuffle=False, batch_size=batch_size
    )
    logger.info("DataLoaders ready — train: %d batches, val: %d batches",
                len(train_dataloader), len(val_dataloader))

    # ─── Optimizer & Scheduler (Table A.3) ───────────────────────
    optimizer = optim.AdamW(recon_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss(reduction='none')
    logger.info("Optimizer: AdamW(lr=%.2e, wd=1e-5) | Scheduler: CosineAnnealingLR(T_max=%d)", lr, n_epochs)

    best_val_loss = float('inf')
    patience_limit = 20
    patience_counter = 0

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # ═══ TRAINING ════════════════════════════════════════════
        recon_model.train()
        running_loss = 0.0
        train_steps = 0

        for batch_data in train_dataloader:
            samples = [batch_data] if isinstance(batch_data, dict) else batch_data

            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0

            for data in samples:
                wsi_bag = [f.to(device) for f in data['wsi_bag']]
                ct_bag = [f.to(device) for f in data['ct_bag']]
                mri_bag = [f.to(device) for f in data['mri_bag']]
                cli_feat = data['cli_feat'].to(device)

                masks = torch.tensor([
                    data['wsi_mask'], data['ct_mask'], data['mri_mask'], data['cli_mask']
                ]).float().to(device)

                # Step 1: MIL feature selection (frozen)
                with torch.no_grad():
                    _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, modality='WSI', add_noise=False)
                    _, ct_best, _ = mil_model.forward_single_bag(ct_bag, modality='CT', add_noise=False)
                    _, mri_best, _ = mil_model.forward_single_bag(mri_bag, modality='MRI', add_noise=False)

                    if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
                    if ct_best.dim() == 1:  ct_best = ct_best.unsqueeze(0)
                    if mri_best.dim() == 1: mri_best = mri_best.unsqueeze(0)
                    if cli_feat.dim() == 1: cli_feat = cli_feat.unsqueeze(0)

                    if data['wsi_mask'] == 0: wsi_best = torch.zeros_like(wsi_best)
                    if data['ct_mask'] == 0:  ct_best = torch.zeros_like(ct_best)
                    if data['mri_mask'] == 0: mri_best = torch.zeros_like(mri_best)
                    if data['cli_mask'] == 0: cli_feat = torch.zeros_like(cli_feat)

                # Step 2: Reconstruction forward
                rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(wsi_best, ct_best, mri_best, cli_feat)

                # Step 3: MSE loss (only on available modalities via mask)
                loss_wsi = (criterion(rec_wsi, wsi_best).mean(dim=1) * masks[0]).mean()
                loss_ct  = (criterion(rec_ct, ct_best).mean(dim=1)  * masks[1]).mean()
                loss_mri = (criterion(rec_mri, mri_best).mean(dim=1) * masks[2]).mean()
                loss_cli = (criterion(rec_cli, cli_feat).mean(dim=1) * masks[3]).mean()

                sample_loss = (loss_wsi + loss_ct + loss_mri + loss_cli) / len(samples)
                sample_loss.backward()
                batch_loss += sample_loss.item()

            optimizer.step()
            running_loss += batch_loss
            train_steps += 1

        avg_train_loss = running_loss / train_steps if train_steps > 0 else 0

        # ═══ VALIDATION ══════════════════════════════════════════
        recon_model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch_data in val_dataloader:
                samples = [batch_data] if isinstance(batch_data, dict) else batch_data

                for data in samples:
                    wsi_bag = [f.to(device) for f in data['wsi_bag']]
                    ct_bag  = [f.to(device) for f in data['ct_bag']]
                    mri_bag = [f.to(device) for f in data['mri_bag']]
                    cli_feat = data['cli_feat'].to(device)

                    masks = torch.tensor([
                        data['wsi_mask'], data['ct_mask'], data['mri_mask'], data['cli_mask']
                    ]).float().to(device)

                    _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, modality='WSI', add_noise=False)
                    _, ct_best, _ = mil_model.forward_single_bag(ct_bag, modality='CT', add_noise=False)
                    _, mri_best, _ = mil_model.forward_single_bag(mri_bag, modality='MRI', add_noise=False)

                    if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
                    if ct_best.dim() == 1:  ct_best = ct_best.unsqueeze(0)
                    if mri_best.dim() == 1: mri_best = mri_best.unsqueeze(0)
                    if cli_feat.dim() == 1: cli_feat = cli_feat.unsqueeze(0)

                    if data['wsi_mask'] == 0: wsi_best = torch.zeros_like(wsi_best)
                    if data['ct_mask'] == 0:  ct_best = torch.zeros_like(ct_best)
                    if data['mri_mask'] == 0: mri_best = torch.zeros_like(mri_best)
                    if data['cli_mask'] == 0: cli_feat = torch.zeros_like(cli_feat)

                    rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(wsi_best, ct_best, mri_best, cli_feat)

                    loss_wsi = (criterion(rec_wsi, wsi_best).mean(dim=1) * masks[0]).mean()
                    loss_ct  = (criterion(rec_ct, ct_best).mean(dim=1)  * masks[1]).mean()
                    loss_mri = (criterion(rec_mri, mri_best).mean(dim=1) * masks[2]).mean()
                    loss_cli = (criterion(rec_cli, cli_feat).mean(dim=1) * masks[3]).mean()

                    val_loss += (loss_wsi + loss_ct + loss_mri + loss_cli).item()
                    val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # ─── Logging ────────────────────────────────────────────
        logger.info(
            "[Epoch %03d/%03d] [Recon] train_loss=%.6f | val_loss=%.6f | best_val=%.6f | "
            "lr=%.2e | patience=%d/%d | time=%.1fs",
            epoch + 1, n_epochs,
            avg_train_loss, avg_val_loss, best_val_loss,
            current_lr, patience_counter, patience_limit, elapsed
        )
        writer.add_scalar('Reconstruction/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Reconstruction/Val_Loss', avg_val_loss, epoch)
        writer.add_scalar('Reconstruction/LR', current_lr, epoch)

        scheduler.step()

        # ─── Checkpointing ──────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(CHECKPOINT_DIR, CKPT_RECON)
            torch.save(recon_model.state_dict(), save_path)
            logger.info("  >> Checkpoint saved: %s (val_loss=%.6f)", save_path, best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.warning(
                    "[Recon] Early stopping at epoch %d — no improvement for %d epochs "
                    "(best_val=%.6f, current_val=%.6f)",
                    epoch + 1, patience_limit, best_val_loss, avg_val_loss
                )
                break

    writer.close()
    logger.info("STAGE 1 — STEP 2/3: Reconstruction Training COMPLETE (best_val=%.6f)", best_val_loss)
