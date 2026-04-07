"""
MIL Trainer — Stage 1, Step 1
==============================
Trains three independent MIL models (WSI, CT, MRI) for 12-month survival prediction.
Each modality uses per-modality optimizer, scheduler, and epoch count as per Table A.1.
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from configs.logging_config import get_logger
from configs.paths import TB_MIL_DIR, CHECKPOINT_DIR, CKPT_MIL_FORMAT
from utils.metrics import calc_bacc, calc_macro_f1

logger = get_logger('MIL_trainer')


def train_single_modality(
    mil_model,
    modality: str,
    train_loader,
    val_loader,
    device,
    lr: float = 1e-3,
    n_epochs: int = 100,
    death_weight: float = 1.0
):
    """Train a single modality (WSI, CT, or MRI) MIL sub-model."""

    logger.info("=" * 60)
    logger.info("[MIL/%s] Starting training — %d epochs, lr=%.2e", modality, n_epochs, lr)
    logger.info("=" * 60)

    writer = SummaryWriter(log_dir=os.path.join(TB_MIL_DIR, modality))

    # Select sub-model
    sub_model_map = {'WSI': mil_model.wsi_mil, 'CT': mil_model.ct_mil, 'MRI': mil_model.mri_mil}
    if modality not in sub_model_map:
        logger.error("[MIL/%s] Invalid modality. Expected one of %s", modality, list(sub_model_map.keys()))
        raise ValueError(f"Unknown modality: {modality}")
    sub_model = sub_model_map[modality]

    # Per-modality optimizer (Table A.1)
    if modality == 'CT':
        optimizer = optim.SGD(sub_model.parameters(), lr=lr)
        logger.info("[MIL/%s] Optimizer: SGD(lr=%.2e)", modality, lr)
    elif modality == 'MRI':
        optimizer = optim.Adam(sub_model.parameters(), lr=lr)
        logger.info("[MIL/%s] Optimizer: Adam(lr=%.2e)", modality, lr)
    else:  # WSI
        optimizer = optim.AdamW(sub_model.parameters(), lr=lr, weight_decay=1e-5)
        logger.info("[MIL/%s] Optimizer: AdamW(lr=%.2e, wd=1e-5)", modality, lr)

    # LR Scheduler: StepLR (step_size=30, gamma=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1e-2)
    logger.info("[MIL/%s] Scheduler: StepLR(step=30, gamma=0.01)", modality)

    criterion = nn.BCELoss(reduction='none')

    # Early Stopping
    patience_limit = 20
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # ─── Training Phase ─────────────────────────────────────
        mil_model.train()
        running_loss = 0.0
        train_batches = 0
        all_train_preds = []
        all_train_labels = []

        for patient_id, feature_list, label, mask in train_loader:
            if mask.item() == 0:
                continue

            feature_list = [f.to(device) for f in feature_list]
            label = label.float().to(device).unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)

            prob, _, _ = mil_model.forward_single_bag(
                feature_list, modality=modality, add_noise=True
            )

            loss_unreduced = criterion(prob, label)
            weight = death_weight if label.item() == 0 else 1.0
            loss = (loss_unreduced * weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sub_model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            train_batches += 1
            all_train_preds.append((prob > 0.5).int().cpu())
            all_train_labels.append(label.int().cpu())

        avg_train_loss = running_loss / train_batches if train_batches > 0 else 0.0

        # ─── Validation Phase ───────────────────────────────────
        mil_model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for patient_id, feature_list, label, mask in val_loader:
                if mask.item() == 0:
                    continue

                feature_list = [f.to(device) for f in feature_list]
                label = label.float().to(device).unsqueeze(0)

                prob, _, _ = mil_model.forward_single_bag(
                    feature_list, modality=modality, add_noise=False
                )

                loss = criterion(prob, label)
                val_loss += loss.item()
                val_batches += 1
                all_val_preds.append((prob > 0.5).int().cpu())
                all_val_labels.append(label.int().cpu())

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
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
            "[Epoch %03d/%03d] [%s] train_loss=%.4f | val_loss=%.4f | best_val=%.4f | "
            "train_bacc=%.4f | train_f1=%.4f | val_bacc=%.4f | val_f1=%.4f | "
            "lr=%.2e | patience=%d/%d | time=%.1fs",
            epoch + 1, n_epochs, modality,
            avg_train_loss, avg_val_loss, best_val_loss,
            train_bacc, train_f1, val_bacc, val_f1,
            current_lr, patience_counter, patience_limit, elapsed
        )
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metrics/Train_BAcc', train_bacc, epoch)
        writer.add_scalar('Metrics/Train_MacroF1', train_f1, epoch)
        writer.add_scalar('Metrics/Val_BAcc', val_bacc, epoch)
        writer.add_scalar('Metrics/Val_MacroF1', val_f1, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        scheduler.step()

        # ─── Checkpointing ──────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(sub_model.state_dict())

            save_path = os.path.join(CHECKPOINT_DIR, CKPT_MIL_FORMAT.format(modality=modality))
            torch.save(mil_model.state_dict(), save_path)
            logger.info(
                "  >> Checkpoint saved: %s (val_loss=%.4f)", save_path, best_val_loss
            )
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.warning(
                    "[MIL/%s] Early stopping at epoch %d — no improvement for %d epochs "
                    "(best_val=%.4f, current_val=%.4f)",
                    modality, epoch + 1, patience_limit, best_val_loss, avg_val_loss
                )
                break

    writer.close()

    if best_model_state is not None:
        sub_model.load_state_dict(best_model_state)
        logger.info("[MIL/%s] Restored best weights (val_loss=%.4f)", modality, best_val_loss)
    else:
        logger.warning("[MIL/%s] No improvement was observed during training!", modality)


def train_mil_survival(mil_model, n_epochs: int = 100, lr: float = 1e-3,
                       train_split: str = 'train', val_split: str = 'test'):
    """
    Orchestrator: train WSI, CT, MRI MIL models sequentially.
    Uses per-modality epoch counts from Table A.1.

    Args:
        train_split: Split to use for training ('train' or 'train_val')
        val_split: Split to use for validation ('test' or 'test' when using train_val)
    """
    logger.info("=" * 60)
    logger.info("STAGE 1 — STEP 1/3: MIL Selection Training")
    logger.info("  train_split=%s | val_split=%s", train_split, val_split)
    logger.info("=" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loaders = mil_model._get_dataloader(split=train_split, shuffle=True)
    val_loaders = mil_model._get_dataloader(split=val_split, shuffle=False)

    loaders = {
        'WSI': {'train': train_loaders[0], 'val': val_loaders[0]},
        'CT':  {'train': train_loaders[1], 'val': val_loaders[1]},
        'MRI': {'train': train_loaders[2], 'val': val_loaders[2]}
    }

    device = next(mil_model.parameters()).device
    logger.info("Device: %s", device)

    # Per-modality epochs (Table A.1): CT=60, MRI=60, WSI=100
    modality_epochs = {'WSI': 100, 'CT': 60, 'MRI': 60}

    for i, modality in enumerate(['WSI', 'CT', 'MRI'], 1):
        logger.info("─── Training modality %d/3: %s (%d epochs) ───",
                     i, modality, modality_epochs[modality])
        train_single_modality(
            mil_model=mil_model,
            modality=modality,
            train_loader=loaders[modality]['train'],
            val_loader=loaders[modality]['val'],
            n_epochs=modality_epochs[modality],
            device=device,
            lr=lr
        )

    logger.info("=" * 60)
    logger.info("STAGE 1 — STEP 1/3: MIL Training COMPLETE")
    logger.info("=" * 60)
