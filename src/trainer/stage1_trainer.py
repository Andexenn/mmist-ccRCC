"""
Stage 1 Orchestrator — Train Each Module Separately
=====================================================
Runs the three Stage 1 steps in order:
  1. MIL Selection      (train_mil_survival)
  2. Reconstruction     (train_reconstruction_module)
  3. Fusion             (train_fuse_module)

Each step loads the best checkpoint from the previous step and freezes it.
"""

import os
import torch

from configs.logging_config import get_logger
from configs.paths import CHECKPOINT_DIR, CKPT_MIL_FORMAT, CKPT_RECON, ensure_dirs

from models.MIL.model import MILModel
from models.Reconstruction.model import ReconstructionModel
from models.Fusion.model import Fusion

from trainer.MIL_trainer import train_mil_survival
from trainer.reconstruction_trainer import train_reconstruction_module
from trainer.fusion_trainer import train_fuse_module

logger = get_logger('stage1')


def train_stage1(
    feature_dir: str,
    clinical_file: str,
    device: str = 'cuda',
    dim: int = 768,
    fusion_strategy: str = 'early_mean',
    mil_lr: float = 1e-3,
    recon_lr: float = 1e-3,
    recon_epochs: int = 600,
    recon_batch_size: int = 14,
    fusion_lr: float = 1e-3,
    fusion_epochs: int = 100,
):
    """
    Full Stage 1 pipeline: MIL → Reconstruction → Fusion.

    Args:
        feature_dir:      Path to extracted features directory
        clinical_file:    Path to clinical CSV file
        device:           'cuda' or 'cpu'
        dim:              Feature dimension (default 768)
        fusion_strategy:  'early_mean', 'early_cat', 'late_ws', 'late_lw'
        mil_lr:           Learning rate for MIL training
        recon_lr:         Learning rate for Reconstruction training
        recon_epochs:     Number of epochs for Reconstruction
        recon_batch_size: Batch size for Reconstruction
        fusion_lr:        Learning rate for Fusion training
        fusion_epochs:    Number of epochs for Fusion training
    """
    ensure_dirs()

    logger.info("=" * 60)
    logger.info("STAGE 1: Training Modules Separately")
    logger.info("  feature_dir   = %s", feature_dir)
    logger.info("  clinical_file = %s", clinical_file)
    logger.info("  device        = %s", device)
    logger.info("  fusion_strategy = %s", fusion_strategy)
    logger.info("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # STEP 1/3: MIL Selection
    # ═══════════════════════════════════════════════════════════════
    logger.info("─── Initializing MIL Model ───")
    mil_model = MILModel(
        feature_dir=feature_dir,
        clinical_dir=clinical_file,
        device=device,
        dim=dim
    )

    train_mil_survival(mil_model, lr=mil_lr)

    # Load best MIL weights (use MRI as the reference — it's loaded last)
    best_mil_path = os.path.join(CHECKPOINT_DIR, CKPT_MIL_FORMAT.format(modality='MRI'))
    if os.path.exists(best_mil_path):
        mil_model.load_state_dict(torch.load(best_mil_path, map_location=device))
        logger.info("Loaded best MIL checkpoint: %s", best_mil_path)
    else:
        logger.warning("MIL checkpoint not found at %s — using last training state", best_mil_path)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2/3: Missing Modality Reconstruction
    # ═══════════════════════════════════════════════════════════════
    logger.info("─── Initializing Reconstruction Model ───")
    recon_model = ReconstructionModel(feature_dim=dim, hidden_dim=128).to(device)

    train_reconstruction_module(
        mil_model=mil_model,
        recon_model=recon_model,
        clinical_file=clinical_file,
        feature_dir=feature_dir,
        n_epochs=recon_epochs,
        lr=recon_lr,
        batch_size=recon_batch_size
    )

    # Load best Reconstruction weights
    best_recon_path = os.path.join(CHECKPOINT_DIR, CKPT_RECON)
    if os.path.exists(best_recon_path):
        recon_model.load_state_dict(torch.load(best_recon_path, map_location=device))
        logger.info("Loaded best Reconstruction checkpoint: %s", best_recon_path)
    else:
        logger.warning("Reconstruction checkpoint not found at %s", best_recon_path)

    # ═══════════════════════════════════════════════════════════════
    # STEP 3/3: Multi-modal Fusion
    # ═══════════════════════════════════════════════════════════════
    logger.info("─── Initializing Fusion Model (strategy=%s) ───", fusion_strategy)
    fusion_model = Fusion(
        fusion_strategy=fusion_strategy,
        input_dims=[dim] * 4,
        num_modalities=4
    ).to(device)

    train_fuse_module(
        mil_model=mil_model,
        recon_model=recon_model,
        fusion_model=fusion_model,
        clinical_file=clinical_file,
        feature_dir=feature_dir,
        fusion_strategy=fusion_strategy,
        epochs=fusion_epochs,
        lr=fusion_lr
    )

    logger.info("=" * 60)
    logger.info("STAGE 1 COMPLETE — All modules trained and saved.")
    logger.info("  Checkpoints in: %s", os.path.abspath(CHECKPOINT_DIR))
    logger.info("=" * 60)

    return mil_model, recon_model, fusion_model
