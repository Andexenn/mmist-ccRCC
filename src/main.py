"""
MMIST-ccRCC Training Pipeline — Main Entry Point
==================================================
Usage:
    # Stage 1 only (train each module separately)
    python main.py --stage 1 --feature_dir ./data/features --clinical_file ./data/clinical.csv

    # Stage 2 only (end-to-end finetuning, requires Stage 1 checkpoints)
    python main.py --stage 2 --feature_dir ./data/features --clinical_file ./data/clinical.csv

    # Both stages
    python main.py --stage all --feature_dir ./data/features --clinical_file ./data/clinical.csv

    # View TensorBoard logs
    tensorboard --logdir=./runs
"""

import argparse
import os
import sys

import torch

from configs.logging_config import setup_logger
from configs.paths import (
    CHECKPOINT_DIR, CKPT_MIL_FORMAT, CKPT_RECON, CKPT_FUSION, CKPT_PIPELINE,
    ensure_dirs
)
from models.MIL.model import MILModel
from models.Reconstruction.model import ReconstructionModel
from models.Fusion.model import Fusion

from trainer.stage1_trainer import train_stage1
from trainer.stage2_trainer import train_pipeline
from prepare_data import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMIST-ccRCC Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage 1 --feature_dir ./data/features --clinical_file ./data/clinical.csv
  python main.py --stage 2 --feature_dir ./data/features --clinical_file ./data/clinical.csv
  python main.py --stage all --feature_dir ./data/features --clinical_file ./data/clinical.csv
  
TensorBoard:
  tensorboard --logdir=./runs
        """
    )

    # Required
    parser.add_argument('--stage', type=str, required=True, choices=['prepare', '1', '2', 'all'],
                        help='Training stage: prepare (generate CSV), 1 (modules), 2 (finetune), or all')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='Path to extracted features directory')
    parser.add_argument('--clinical_file', type=str, required=True,
                        help='Path to clinical CSV file')

    # Optional
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--dim', type=int, default=768,
                        help='Feature dimension (default: 768)')
    parser.add_argument('--fusion_strategy', type=str, default='early_mean',
                        choices=['early_mean', 'early_cat', 'late_ws', 'late_lw'],
                        help='Fusion strategy (default: early_mean)')

    # Stage 1 hyperparams
    parser.add_argument('--mil_lr', type=float, default=1e-3,
                        help='MIL learning rate (default: 1e-3)')
    parser.add_argument('--recon_lr', type=float, default=1e-3,
                        help='Reconstruction learning rate (default: 1e-3)')
    parser.add_argument('--recon_epochs', type=int, default=600,
                        help='Reconstruction epochs (default: 600)')
    parser.add_argument('--recon_batch_size', type=int, default=14,
                        help='Reconstruction batch size (default: 14)')
    parser.add_argument('--fusion_lr', type=float, default=1e-3,
                        help='Fusion learning rate (default: 1e-3)')
    parser.add_argument('--fusion_epochs', type=int, default=100,
                        help='Fusion epochs (default: 100)')

    # Stage 2 hyperparams
    parser.add_argument('--pipeline_lr', type=float, default=1e-5,
                        help='Pipeline finetuning learning rate (default: 1e-5)')
    parser.add_argument('--pipeline_epochs', type=int, default=100,
                        help='Pipeline finetuning epochs (default: 100)')

    return parser.parse_args()


def run_stage2(args, mil_model=None, recon_model=None, fusion_model=None):
    """Run Stage 2 end-to-end finetuning."""
    logger = setup_logger('mmist')
    device = args.device

    if mil_model is None or recon_model is None or fusion_model is None:
        logger.info("─── Loading Stage 1 checkpoints for Stage 2 ───")

        # Initialize models
        mil_model = MILModel(
            feature_dir=args.feature_dir,
            clinical_dir=args.clinical_file,
            device=device,
            dim=args.dim
        )
        recon_model = ReconstructionModel(feature_dim=args.dim, hidden_dim=128).to(device)
        fusion_model = Fusion(
            fusion_strategy=args.fusion_strategy,
            input_dims=[args.dim] * 4,
            num_modalities=4
        ).to(device)

        # Load Stage 1 checkpoints
        mil_ckpt = os.path.join(CHECKPOINT_DIR, CKPT_MIL_FORMAT.format(modality='MRI'))
        recon_ckpt = os.path.join(CHECKPOINT_DIR, CKPT_RECON)
        fusion_ckpt = os.path.join(CHECKPOINT_DIR, CKPT_FUSION)

        for path, model, name in [
            (mil_ckpt, mil_model, 'MIL'),
            (recon_ckpt, recon_model, 'Reconstruction'),
            (fusion_ckpt, fusion_model, 'Fusion')
        ]:
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                logger.info("Loaded %s checkpoint: %s", name, path)
            else:
                logger.error(
                    "Missing %s checkpoint: %s — Run Stage 1 first! "
                    "(python main.py --stage 1 ...)", name, path
                )
                sys.exit(1)

    train_pipeline(
        mil_model=mil_model,
        recon_model=recon_model,
        fusion_model=fusion_model,
        clinical_file=args.clinical_file,
        feature_dir=args.feature_dir,
        epochs=args.pipeline_epochs,
        lr=args.pipeline_lr
    )


def main():
    args = parse_args()

    # Setup root logger
    logger = setup_logger('mmist')

    # Ensure all directories exist
    ensure_dirs()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        args.device = 'cpu'

    logger.info("=" * 60)
    logger.info("MMIST-ccRCC Training Pipeline")
    logger.info("  Stage:          %s", args.stage)
    logger.info("  Feature Dir:    %s", args.feature_dir)
    logger.info("  Clinical File:  %s", args.clinical_file)
    logger.info("  Device:         %s", args.device)
    logger.info("  Fusion:         %s", args.fusion_strategy)
    logger.info("=" * 60)

    # ─── Data Preparation ────────────────────────────────────────
    if args.stage == 'prepare':
        output_csv = os.path.join(args.feature_dir, 'master_dataset.csv')
        prepare_dataset(args.feature_dir, output_csv)
        logger.info("Data preparation complete. Master CSV: %s", output_csv)
        return

    # Auto-generate master CSV if it doesn't exist
    master_csv = os.path.join(args.feature_dir, 'master_dataset.csv')
    if not os.path.exists(master_csv):
        logger.info("Master CSV not found — generating from feature directory...")
        prepare_dataset(args.feature_dir, master_csv)

    # Use master CSV as the clinical file for training
    if os.path.exists(master_csv):
        logger.info("Using master CSV: %s", master_csv)
        args.clinical_file = master_csv

    # ─── Stage 1 ─────────────────────────────────────────────────
    mil_model, recon_model, fusion_model = None, None, None

    if args.stage in ['1', 'all']:
        mil_model, recon_model, fusion_model = train_stage1(
            feature_dir=args.feature_dir,
            clinical_file=args.clinical_file,
            device=args.device,
            dim=args.dim,
            fusion_strategy=args.fusion_strategy,
            mil_lr=args.mil_lr,
            recon_lr=args.recon_lr,
            recon_epochs=args.recon_epochs,
            recon_batch_size=args.recon_batch_size,
            fusion_lr=args.fusion_lr,
            fusion_epochs=args.fusion_epochs,
        )

    # ─── Stage 2 ─────────────────────────────────────────────────
    if args.stage in ['2', 'all']:
        run_stage2(args, mil_model, recon_model, fusion_model)

    logger.info("=" * 60)
    logger.info("ALL TRAINING COMPLETE")
    logger.info("  Checkpoints: %s", os.path.abspath(CHECKPOINT_DIR))
    logger.info("  TensorBoard: tensorboard --logdir=./runs")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
