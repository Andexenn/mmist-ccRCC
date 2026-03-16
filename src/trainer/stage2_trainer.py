"""
Pipeline (Stage 2) Trainer — End-to-End Finetuning
====================================================
After Stage 1 trains each module separately, Stage 2 unfreezes all parameters
and finetunes the entire pipeline (MIL → Reconstruction → Fusion) jointly
with a low learning rate (lr=1e-5).
"""

import os
import time
from typing import List

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from configs.logging_config import get_logger
from configs.paths import CHECKPOINT_DIR, get_pipeline_tb_dir, get_pipeline_ckpt_name
from models.Fusion.model import Fusion
from models.MIL.model import MILModel
from models.Reconstruction.model import ReconstructionModel
from dataset.reconstruct_dataset import get_reconstruct_dataloader
from utils.metrics import calc_bacc, calc_macro_f1

logger = get_logger('stage2_trainer')


def train_pipeline(
    mil_model: MILModel,
    recon_model: ReconstructionModel,
    fusion_model: Fusion,
    clinical_file: str,
    feature_dir: str,
    fusion_strategy: str = 'early_mean',
    bacc_mods: List = [1.0, 1.0, 1.0, 1.0],
    epochs: int = 100,
    lr: float = 1e-5,
    death_weight: float = 1.0
):
    """
    Stage 2: Finetune the entire pipeline end-to-end.
    All modules are unfrozen and trained jointly with a single optimizer.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: End-to-End Pipeline Finetuning [%s]", fusion_strategy)
    logger.info("  epochs=%d | lr=%.2e | death_weight=%.1f", epochs, lr, death_weight)
    logger.info("=" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tb_dir = get_pipeline_tb_dir(fusion_strategy)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    device = next(fusion_model.parameters()).device

    # ─── Unfreeze ALL models ─────────────────────────────────────
    for model, name in [(mil_model, 'MIL'), (recon_model, 'Reconstruction'), (fusion_model, 'Fusion')]:
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("%s unfrozen (%d params)", name, n_params)

    total_params = (sum(p.numel() for p in mil_model.parameters()) +
                    sum(p.numel() for p in recon_model.parameters()) +
                    sum(p.numel() for p in fusion_model.parameters()))
    logger.info("Total trainable params: %d", total_params)

    # ─── DataLoaders ─────────────────────────────────────────────
    train_loader = get_reconstruct_dataloader(clinical_file, feature_dir, split='train')
    val_loader = get_reconstruct_dataloader(clinical_file, feature_dir, split='val', shuffle=False)

    # ─── Optimizer & Scheduler ───────────────────────────────────
    all_params = list(mil_model.parameters()) + list(recon_model.parameters()) + list(fusion_model.parameters())
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.BCELoss(reduction='none')
    logger.info("Optimizer: AdamW(lr=%.2e, wd=1e-5) | Scheduler: ReduceLROnPlateau(factor=0.5, patience=5)", lr)

    # ─── Training Loop ───────────────────────────────────────────
    best_val_loss = float('inf')
    best_val_bacc = 0.0
    best_val_f1 = 0.0
    patience_limit = 20
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        mil_model.train()
        recon_model.train()
        fusion_model.train()

        running_loss = 0.0
        train_steps = 0
        all_train_preds = []
        all_train_labels = []

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
                torch.tensor([[data['wsi_mask']]]).float().to(device),
                torch.tensor([[data['ct_mask']]]).float().to(device),
                torch.tensor([[data['mri_mask']]]).float().to(device),
                torch.tensor([[data['cli_mask']]]).float().to(device),
            ]

            optimizer.zero_grad(set_to_none=True)

            # ─── MIL Selection (with gradient) ───────────────────
            _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, modality='WSI', add_noise=True)
            _, ct_best, _  = mil_model.forward_single_bag(ct_bag, modality='CT', add_noise=True)
            _, mri_best, _ = mil_model.forward_single_bag(mri_bag, modality='MRI', add_noise=True)

            if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
            if ct_best.dim() == 1:  ct_best = ct_best.unsqueeze(0)
            if mri_best.dim() == 1: mri_best = mri_best.unsqueeze(0)
            if cli_feat.dim() == 1: cli_feat = cli_feat.unsqueeze(0)

            in_wsi = wsi_best.clone() * masks[0]
            in_ct  = ct_best.clone()  * masks[1]
            in_mri = mri_best.clone() * masks[2]
            in_cli = cli_feat.clone() * masks[3]

            # ─── Reconstruction (with gradient) ──────────────────
            rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(in_wsi, in_ct, in_mri, in_cli)

            final_wsi = masks[0] * wsi_best + (1 - masks[0]) * rec_wsi
            final_ct  = masks[1] * ct_best  + (1 - masks[1]) * rec_ct
            final_mri = masks[2] * mri_best + (1 - masks[2]) * rec_mri
            final_cli = masks[3] * cli_feat + (1 - masks[3]) * rec_cli

            feature_list = [final_wsi, final_ct, final_mri, final_cli]
            full_masks = [torch.ones_like(m) for m in masks]

            # ─── Fusion (with gradient) ──────────────────────────
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
            all_train_preds.append((prob > 0.5).int().cpu().unsqueeze(0) if prob.dim() == 0 else (prob > 0.5).int().cpu())
            all_train_labels.append(label.int().cpu().unsqueeze(0) if label.dim() == 0 else label.int().cpu())

        avg_train_loss = running_loss / train_steps if train_steps > 0 else 0.0

        # ═══ VALIDATION ══════════════════════════════════════════
        mil_model.eval()
        recon_model.eval()
        fusion_model.eval()

        val_loss = 0.0
        val_steps = 0
        all_val_preds = []
        all_val_labels = []

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

                _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, 'WSI', add_noise=False)
                _, ct_best, _  = mil_model.forward_single_bag(ct_bag, 'CT', add_noise=False)
                _, mri_best, _ = mil_model.forward_single_bag(mri_bag, 'MRI', add_noise=False)

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
                all_val_preds.append((prob.squeeze() > 0.5).int().cpu().unsqueeze(0) if prob.squeeze().dim() == 0 else (prob.squeeze() > 0.5).int().cpu())
                all_val_labels.append(label.squeeze().int().cpu().unsqueeze(0) if label.squeeze().dim() == 0 else label.squeeze().int().cpu())

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # ─── Compute Metrics ─────────────────────────────────────
        if all_train_preds:
            train_preds_cat = torch.cat(all_train_preds)
            train_labels_cat = torch.cat(all_train_labels)
            train_bacc = calc_bacc(train_labels_cat, train_preds_cat)
            train_f1 = calc_macro_f1(train_labels_cat, train_preds_cat)
        else:
            train_bacc, train_f1 = 0.0, 0.0

        if all_val_preds:
            val_preds_cat = torch.cat(all_val_preds)
            val_labels_cat = torch.cat(all_val_labels)
            val_bacc = calc_bacc(val_labels_cat, val_preds_cat)
            val_f1 = calc_macro_f1(val_labels_cat, val_preds_cat)
        else:
            val_bacc, val_f1 = 0.0, 0.0

        # ─── Logging ────────────────────────────────────────────
        logger.info(
            "[Epoch %03d/%03d] [Pipeline] train_loss=%.4f | val_loss=%.4f | best_val=%.4f | "
            "train_bacc=%.4f | train_f1=%.4f | val_bacc=%.4f | val_f1=%.4f | "
            "lr=%.2e | patience=%d/%d | time=%.1fs",
            epoch + 1, epochs,
            avg_train_loss, avg_val_loss, best_val_loss,
            train_bacc, train_f1, val_bacc, val_f1,
            current_lr, patience_counter, patience_limit, elapsed
        )
        writer.add_scalar('Pipeline/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Pipeline/Val_Loss', avg_val_loss, epoch)
        writer.add_scalar('Pipeline/Train_BAcc', train_bacc, epoch)
        writer.add_scalar('Pipeline/Train_MacroF1', train_f1, epoch)
        writer.add_scalar('Pipeline/Val_BAcc', val_bacc, epoch)
        writer.add_scalar('Pipeline/Val_MacroF1', val_f1, epoch)
        writer.add_scalar('Pipeline/LR', current_lr, epoch)

        scheduler.step(avg_val_loss)

        # ─── Checkpointing ──────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_bacc = val_bacc
            best_val_f1 = val_f1
            patience_counter = 0
            ckpt_name = get_pipeline_ckpt_name(fusion_strategy)
            save_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
            torch.save({
                'mil_state_dict': mil_model.state_dict(),
                'recon_state_dict': recon_model.state_dict(),
                'fusion_state_dict': fusion_model.state_dict()
            }, save_path)
            logger.info("  >> Pipeline checkpoint saved: %s (val_loss=%.4f)", save_path, best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.warning(
                    "[Pipeline] Early stopping at epoch %d — no improvement for %d epochs "
                    "(best_val=%.4f, current_val=%.4f)",
                    epoch + 1, patience_limit, best_val_loss, avg_val_loss
                )
                break

    writer.close()
    logger.info("STAGE 2: Pipeline Finetuning COMPLETE [%s] (best_val_loss=%.4f, best_val_bacc=%.4f, best_val_f1=%.4f)",
                fusion_strategy, best_val_loss, best_val_bacc, best_val_f1)

    return {
        'strategy': fusion_strategy,
        'best_val_loss': best_val_loss,
        'best_val_bacc': best_val_bacc,
        'best_val_f1': best_val_f1,
    }
