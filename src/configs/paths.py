"""
Centralized Path Configuration for MMIST-ccRCC Pipeline
========================================================
All log directories, checkpoint paths, and TensorBoard runs are defined here.
Change these paths ONCE and all trainers will use them automatically.

TensorBoard Usage:
    tensorboard --logdir=./runs
    Then open http://localhost:6006 in your browser.
"""

import os

# ─── Base directories (relative to src/) ───
TENSORBOARD_ROOT = './runs'         # TensorBoard event logs
CHECKPOINT_DIR   = './checkpoints'  # Model checkpoint .pth files
FILE_LOG_DIR     = './logs'         # File-based text logs (.log)

# ─── TensorBoard sub-directories for each module ───
TB_MIL_DIR       = os.path.join(TENSORBOARD_ROOT, 'stage1_mil')
TB_RECON_DIR     = os.path.join(TENSORBOARD_ROOT, 'stage1_reconstruction')
TB_FUSION_DIR    = os.path.join(TENSORBOARD_ROOT, 'stage1_fusion')
TB_PIPELINE_DIR  = os.path.join(TENSORBOARD_ROOT, 'stage2_pipeline')

# ─── Checkpoint filenames ───
CKPT_MIL_FORMAT      = 'best_mil_{modality}.pth'    # e.g. best_mil_WSI.pth
CKPT_RECON           = 'best_reconstruction.pth'
CKPT_FUSION          = 'best_fusion_model.pth'
CKPT_PIPELINE        = 'best_pipeline.pth'


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [TENSORBOARD_ROOT, CHECKPOINT_DIR, FILE_LOG_DIR,
              TB_MIL_DIR, TB_RECON_DIR, TB_FUSION_DIR, TB_PIPELINE_DIR]:
        os.makedirs(d, exist_ok=True)
