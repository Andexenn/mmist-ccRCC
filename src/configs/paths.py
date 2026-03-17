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
TENSORBOARD_ROOT = '/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/runs/v1.2'         # TensorBoard event logs
CHECKPOINT_DIR   = '/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/checkpoint/v1.2'  # Model checkpoint .pth files
FILE_LOG_DIR     = '/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/logs/v1.2'         # File-based text logs (.log)

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

# ─── All fusion strategies ───
ALL_FUSION_STRATEGIES = ['early_mean', 'early_cat', 'late_ws', 'late_lw']


def get_fusion_tb_dir(strategy: str) -> str:
    """Return strategy-specific TensorBoard dir, e.g. runs/stage1_fusion_early_mean/"""
    return os.path.join(TENSORBOARD_ROOT, f'stage1_fusion_{strategy}')


def get_pipeline_tb_dir(strategy: str) -> str:
    """Return strategy-specific TensorBoard dir, e.g. runs/stage2_pipeline_early_mean/"""
    return os.path.join(TENSORBOARD_ROOT, f'stage2_pipeline_{strategy}')


def get_test_tb_dir(strategy: str) -> str:
    """Return strategy-specific TensorBoard dir for test evaluation, e.g. runs/test_early_mean/"""
    return os.path.join(TENSORBOARD_ROOT, f'test_{strategy}')


def get_fusion_ckpt_name(strategy: str) -> str:
    """Return strategy-specific checkpoint filename, e.g. best_fusion_early_mean.pth"""
    return f'best_fusion_{strategy}.pth'


def get_pipeline_ckpt_name(strategy: str) -> str:
    """Return strategy-specific checkpoint filename, e.g. best_pipeline_early_mean.pth"""
    return f'best_pipeline_{strategy}.pth'


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [TENSORBOARD_ROOT, CHECKPOINT_DIR, FILE_LOG_DIR,
              TB_MIL_DIR, TB_RECON_DIR, TB_FUSION_DIR, TB_PIPELINE_DIR]:
        os.makedirs(d, exist_ok=True)
