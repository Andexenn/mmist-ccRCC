"""
Test Set Evaluator — Inference on Test Split
==============================================
Loads the best checkpoints and runs the full pipeline
(MIL → Reconstruction → Fusion) on the 'test' split.
No training, no optimizer, no checkpointing — purely evaluation.
"""

import os
import time
from typing import List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from configs.logging_config import get_logger
from configs.paths import get_test_tb_dir
from models.Fusion.model import Fusion
from models.MIL.model import MILModel
from models.Reconstruction.model import ReconstructionModel
from dataset.reconstruct_dataset import get_reconstruct_dataloader
from utils.metrics import calc_bacc, calc_macro_f1

logger = get_logger('test_evaluator')


def evaluate_test_set(
    mil_model: MILModel,
    recon_model: ReconstructionModel,
    fusion_model: Fusion,
    clinical_file: str,
    feature_dir: str,
    fusion_strategy: str = 'early_mean',
    bacc_mods: List = [1.0, 1.0, 1.0, 1.0],
):
    """
    Evaluate the full pipeline on the test split.

    All models are set to eval mode. No gradients, no training.
    Results are logged to TensorBoard and returned as a dict.
    """
    logger.info("=" * 60)
    logger.info("TEST EVALUATION: %s", fusion_strategy)
    logger.info("=" * 60)

    tb_dir = get_test_tb_dir(fusion_strategy)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    device = next(fusion_model.parameters()).device

    # ─── Set all models to eval mode ─────────────────────────────
    mil_model.eval()
    recon_model.eval()
    fusion_model.eval()

    # ─── Test DataLoader ─────────────────────────────────────────
    test_loader = get_reconstruct_dataloader(
        clinical_file, feature_dir, split='test', shuffle=False
    )

    if len(test_loader.dataset) == 0:
        logger.warning("Test dataset is empty — skipping evaluation for [%s]", fusion_strategy)
        writer.close()
        return {
            'strategy': fusion_strategy,
            'test_loss': 0.0,
            'test_bacc': 0.0,
            'test_f1': 0.0,
            'test_samples': 0,
        }

    logger.info("Test samples: %d", len(test_loader.dataset))

    criterion = nn.BCELoss(reduction='none')

    # ─── Inference Loop ──────────────────────────────────────────
    test_loss = 0.0
    test_steps = 0
    all_test_preds = []
    all_test_labels = []

    eval_start = time.time()

    with torch.no_grad():
        for data in test_loader:
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

            # ─── MIL Selection ───────────────────────────────────
            _, wsi_best, _ = mil_model.forward_single_bag(wsi_bag, 'WSI', add_noise=False)
            _, ct_best, _  = mil_model.forward_single_bag(ct_bag, 'CT', add_noise=False)
            _, mri_best, _ = mil_model.forward_single_bag(mri_bag, 'MRI', add_noise=False)

            if wsi_best.dim() == 1: wsi_best = wsi_best.unsqueeze(0)
            if ct_best.dim() == 1:  ct_best = ct_best.unsqueeze(0)
            if mri_best.dim() == 1: mri_best = mri_best.unsqueeze(0)
            if cli_feat.dim() == 1: cli_feat = cli_feat.unsqueeze(0)

            # ─── Reconstruction ──────────────────────────────────
            in_wsi = wsi_best * masks[0]
            in_ct  = ct_best  * masks[1]
            in_mri = mri_best * masks[2]
            in_cli = cli_feat * masks[3]

            rec_wsi, rec_ct, rec_mri, rec_cli = recon_model(in_wsi, in_ct, in_mri, in_cli)

            final_wsi = masks[0] * wsi_best + (1 - masks[0]) * rec_wsi
            final_ct  = masks[1] * ct_best  + (1 - masks[1]) * rec_ct
            final_mri = masks[2] * mri_best + (1 - masks[2]) * rec_mri
            final_cli = masks[3] * cli_feat + (1 - masks[3]) * rec_cli

            # ─── Fusion ──────────────────────────────────────────
            feature_list = [final_wsi, final_ct, final_mri, final_cli]
            full_masks = [torch.ones_like(m) for m in masks]

            prob = fusion_model(feature_list, full_masks, bacc_mods)
            loss = criterion(prob.squeeze(), label.squeeze())
            test_loss += loss.item()
            test_steps += 1

            all_test_preds.append(
                (prob.squeeze() > 0.5).int().cpu().unsqueeze(0)
                if prob.squeeze().dim() == 0
                else (prob.squeeze() > 0.5).int().cpu()
            )
            all_test_labels.append(
                label.squeeze().int().cpu().unsqueeze(0)
                if label.squeeze().dim() == 0
                else label.squeeze().int().cpu()
            )

    elapsed = time.time() - eval_start
    avg_test_loss = test_loss / test_steps if test_steps > 0 else 0.0

    # ─── Compute Metrics ─────────────────────────────────────────
    if all_test_preds:
        test_preds_cat = torch.cat(all_test_preds)
        test_labels_cat = torch.cat(all_test_labels)
        test_bacc = calc_bacc(test_labels_cat, test_preds_cat)
        test_f1 = calc_macro_f1(test_labels_cat, test_preds_cat)
    else:
        test_bacc, test_f1 = 0.0, 0.0

    # ─── Logging ─────────────────────────────────────────────────
    logger.info(
        "[TEST] [%s] test_loss=%.4f | test_bacc=%.4f | test_f1=%.4f | "
        "samples=%d | time=%.1fs",
        fusion_strategy, avg_test_loss, test_bacc, test_f1,
        test_steps, elapsed
    )
    writer.add_scalar('Test/Loss', avg_test_loss, 0)
    writer.add_scalar('Test/BAcc', test_bacc, 0)
    writer.add_scalar('Test/MacroF1', test_f1, 0)
    writer.close()

    logger.info("TEST EVALUATION COMPLETE [%s]", fusion_strategy)

    return {
        'strategy': fusion_strategy,
        'test_loss': avg_test_loss,
        'test_bacc': test_bacc,
        'test_f1': test_f1,
        'test_samples': test_steps,
    }
