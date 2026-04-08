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
    CHECKPOINT_DIR, CKPT_MIL_FORMAT, CKPT_RECON,
    ALL_FUSION_STRATEGIES, get_fusion_ckpt_name, get_pipeline_ckpt_name,
    ensure_dirs
)
from models.MIL.model import MILModel
from models.Reconstruction.model import ReconstructionModel
from models.Fusion.model import Fusion

from trainer.stage1_trainer import train_stage1
from trainer.stage2_trainer import train_pipeline
from trainer.test_evaluator import evaluate_test_set, evaluate_individual_mil
from trainer.MIL_trainer import train_mil_survival
from prepare_data import prepare_dataset

import itertools
from configs.paths import get_recon_ckpt_name_ablation, get_fusion_ckpt_name_ablation, get_pipeline_ckpt_name_ablation, get_ablation_suffix


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
    parser.add_argument('--stage', type=str, required=True, choices=['prepare', '1', '2', 'all', 'test', 'ablation'],
                        help='Training stage: prepare (generate CSV), 1 (modules), 2 (finetune), all, test (eval), or ablation')
    parser.add_argument('--ablation_mode', type=str, choices=['individual', '2_mod', '3_mod', 'all'],
                        help='Which ablation to run. Only used if --stage ablation')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='Path to extracted features directory')
    parser.add_argument('--clinical_file', type=str, required=True,
                        help='Path to clinical CSV file')

    # Optional
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--dim', type=int, default=768,
                        help='Feature dimension (default: 768)')
    parser.add_argument('--fusion_strategy', type=str, default='all',
                        choices=['early_mean', 'early_cat', 'late_ws', 'late_lw', 'all'],
                        help='Fusion strategy: one of the 4 strategies, or "all" to train all (default: all)')

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


def _load_base_models(args, logger):
    """Load MIL and Reconstruction models from Stage 1 checkpoints."""
    device = args.device

    mil_model = MILModel(
        feature_dir=args.feature_dir,
        clinical_dir=args.clinical_file,
        device=device,
        dim=args.dim
    )
    recon_model = ReconstructionModel(feature_dim=args.dim, hidden_dim=128).to(device)

    mil_ckpt = os.path.join(CHECKPOINT_DIR, CKPT_MIL_FORMAT.format(modality='MRI'))
    recon_ckpt = os.path.join(CHECKPOINT_DIR, CKPT_RECON)

    for path, model, name in [
        (mil_ckpt, mil_model, 'MIL'),
        (recon_ckpt, recon_model, 'Reconstruction'),
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

    return mil_model, recon_model


def run_stage2(args, mil_model=None, recon_model=None,
               train_split='train', val_split='val'):
    """Run Stage 2 end-to-end finetuning for all fusion strategies."""
    logger = setup_logger('mmist')
    device = args.device

    # Determine which strategies to train
    if args.fusion_strategy == 'all':
        strategies = ALL_FUSION_STRATEGIES
    else:
        strategies = [args.fusion_strategy]

    # Load base models if not provided
    if mil_model is None or recon_model is None:
        logger.info("─── Loading Stage 1 checkpoints for Stage 2 ───")
        mil_model, recon_model = _load_base_models(args, logger)

    pipeline_results = []

    for strategy in strategies:
        logger.info("─── Stage 2: Training pipeline with fusion strategy=%s ───", strategy)

        # Load the fusion checkpoint for this strategy
        fusion_ckpt = os.path.join(CHECKPOINT_DIR, get_fusion_ckpt_name(strategy))
        fusion_model = Fusion(
            fusion_strategy=strategy,
            input_dims=[args.dim] * 4,
            num_modalities=4
        ).to(device)

        if os.path.exists(fusion_ckpt):
            fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
            logger.info("Loaded Fusion checkpoint for [%s]: %s", strategy, fusion_ckpt)
        else:
            logger.warning("Fusion checkpoint not found for [%s] at %s — training from scratch", strategy, fusion_ckpt)

        result = train_pipeline(
            mil_model=mil_model,
            recon_model=recon_model,
            fusion_model=fusion_model,
            clinical_file=args.clinical_file,
            feature_dir=args.feature_dir,
            fusion_strategy=strategy,
            epochs=args.pipeline_epochs,
            lr=args.pipeline_lr,
            train_split=train_split,
            val_split=val_split
        )
        pipeline_results.append(result)

    return pipeline_results


def run_test(args):
    """Run --stage test: train with combined train+val data, then evaluate on test split."""
    logger = setup_logger('mmist')
    device = args.device

    logger.info("═" * 60)
    logger.info("TEST MODE: Training on train+val, evaluating on test")
    logger.info("═" * 60)

    # ─── Stage 1: Train with combined train+val ──────────────────
    mil_model, recon_model, fusion_results = train_stage1(
        feature_dir=args.feature_dir,
        clinical_file=args.clinical_file,
        device=device,
        dim=args.dim,
        fusion_strategy=args.fusion_strategy,
        mil_lr=args.mil_lr,
        recon_lr=args.recon_lr,
        recon_epochs=args.recon_epochs,
        recon_batch_size=args.recon_batch_size,
        fusion_lr=args.fusion_lr,
        fusion_epochs=args.fusion_epochs,
        train_split='train_val',
        val_split='test',
    )

    # ─── Stage 2: Finetune with combined train+val ───────────────
    pipeline_results = run_stage2(
        args, mil_model, recon_model,
        train_split='train_val', val_split='test'
    )

    # ─── Final Evaluation on Test Split ──────────────────────────
    logger.info("─── Final test set evaluation ───")

    # Determine which strategies to evaluate
    if args.fusion_strategy == 'all':
        strategies = ALL_FUSION_STRATEGIES
    else:
        strategies = [args.fusion_strategy]

    test_results = []

    for strategy in strategies:
        logger.info("─── Test: Evaluating pipeline with fusion strategy=%s ───", strategy)

        fusion_model = Fusion(
            fusion_strategy=strategy,
            input_dims=[args.dim] * 4,
            num_modalities=4
        ).to(device)

        # Load the best Stage 2 pipeline checkpoint
        pipeline_ckpt = os.path.join(CHECKPOINT_DIR, get_pipeline_ckpt_name(strategy))
        if os.path.exists(pipeline_ckpt):
            ckpt = torch.load(pipeline_ckpt, map_location=device)
            mil_model.load_state_dict(ckpt['mil_state_dict'])
            recon_model.load_state_dict(ckpt['recon_state_dict'])
            fusion_model.load_state_dict(ckpt['fusion_state_dict'])
            logger.info("Loaded Pipeline checkpoint for [%s]: %s", strategy, pipeline_ckpt)
        else:
            # Fallback: load Stage 1 fusion checkpoint
            fusion_ckpt = os.path.join(CHECKPOINT_DIR, get_fusion_ckpt_name(strategy))
            if os.path.exists(fusion_ckpt):
                fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
                logger.info("Loaded Fusion checkpoint for [%s]: %s (no pipeline ckpt found)", strategy, fusion_ckpt)
            else:
                logger.warning(
                    "No checkpoint found for [%s] — evaluating with random weights!", strategy
                )

        result = evaluate_test_set(
            mil_model=mil_model,
            recon_model=recon_model,
            fusion_model=fusion_model,
            clinical_file=args.clinical_file,
            feature_dir=args.feature_dir,
            fusion_strategy=strategy,
        )
        test_results.append(result)

    return test_results, fusion_results, pipeline_results


def run_ablation(args):
    """Run ablation studies directly matching the test flow: train on train+val, infer on test."""
    logger = setup_logger('mmist')
    device = args.device

    logger.info("═" * 60)
    logger.info("ABLATION MODE: Training on train+val, evaluating on test. Mode: %s", args.ablation_mode)
    logger.info("═" * 60)

    modes = []
    if args.ablation_mode == 'individual':
        modes = ['individual']
    elif args.ablation_mode == '2_mod':
        modes = ['2_mod']
    elif args.ablation_mode == '3_mod':
        modes = ['3_mod']
    elif args.ablation_mode == 'all':
        modes = ['individual', '2_mod', '3_mod', '4_mod']
        
    mil_model = MILModel(
        feature_dir=args.feature_dir,
        clinical_dir=args.clinical_file,
        device=device,
        dim=args.dim
    )
    
    # Train MIL once using combined data
    logger.info("─── Training MIL Models for Ablation Base ───")
    train_mil_survival(mil_model, lr=args.mil_lr, train_split='train_val', val_split='test')
    
    all_ablation_results = []

    if 'individual' in modes:
        logger.info("=== Running Individual Modality Ablation (No Recon/Fusion) ===")
        for mod in ['WSI', 'CT', 'MRI']:
            ckpt_path = os.path.join(CHECKPOINT_DIR, CKPT_MIL_FORMAT.format(modality=mod))
            if os.path.exists(ckpt_path):
                mil_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            res = evaluate_individual_mil(mil_model, args.clinical_file, args.feature_dir, mod, split='test')
            all_ablation_results.append(res)
            
    combos = []
    base_mods = ['WSI', 'CT', 'MRI', 'Clinical']
    if '2_mod' in modes:
        combos.extend(list(itertools.combinations(base_mods, 2)))
    if '3_mod' in modes:
        combos.extend(list(itertools.combinations(base_mods, 3)))
    if '4_mod' in modes:
        combos.extend(list(itertools.combinations(base_mods, 4)))
        
    for combo in combos:
        active_modalities = list(combo)
        logger.info("=" * 60)
        logger.info("=== Running Ablation for Combination: %s ===", active_modalities)
        logger.info("=" * 60)
        
        # Need to load reference MRI again for MIL baseline
        ref_mil_path = os.path.join(CHECKPOINT_DIR, CKPT_MIL_FORMAT.format(modality='MRI'))
        if os.path.exists(ref_mil_path):
            mil_model.load_state_dict(torch.load(ref_mil_path, map_location=device))
            
        recon_model = ReconstructionModel(feature_dim=args.dim, hidden_dim=128).to(device)
        from trainer.reconstruction_trainer import train_reconstruction_module
        train_reconstruction_module(
            mil_model=mil_model, recon_model=recon_model,
            clinical_file=args.clinical_file, feature_dir=args.feature_dir,
            n_epochs=args.recon_epochs, lr=args.recon_lr, batch_size=args.recon_batch_size,
            train_split='train_val', val_split='test', active_modalities=active_modalities
        )
        
        recon_ckpt_path = os.path.join(CHECKPOINT_DIR, get_recon_ckpt_name_ablation(active_modalities))
        if os.path.exists(recon_ckpt_path):
            recon_model.load_state_dict(torch.load(recon_ckpt_path, map_location=device))
            
        fusion_strategies = ALL_FUSION_STRATEGIES if args.fusion_strategy == 'all' else [args.fusion_strategy]
        from trainer.fusion_trainer import train_fuse_module
        
        for strategy in fusion_strategies:
            fusion_model = Fusion(fusion_strategy=strategy, input_dims=[args.dim]*4, num_modalities=4).to(device)
            train_fuse_module(
                mil_model, recon_model, fusion_model, args.clinical_file, args.feature_dir,
                strategy, args.fusion_epochs, args.fusion_lr, train_split='train_val', val_split='test',
                active_modalities=active_modalities
            )
            
            fusion_ckpt_path = os.path.join(CHECKPOINT_DIR, get_fusion_ckpt_name_ablation(strategy, active_modalities))
            if os.path.exists(fusion_ckpt_path):
                fusion_model.load_state_dict(torch.load(fusion_ckpt_path, map_location=device))
                
            from trainer.stage2_trainer import train_pipeline
            train_pipeline(
                mil_model, recon_model, fusion_model, args.clinical_file, args.feature_dir,
                strategy, epochs=args.pipeline_epochs, lr=args.pipeline_lr, train_split='train_val', val_split='test',
                active_modalities=active_modalities
            )
            
            pipeline_ckpt_path = os.path.join(CHECKPOINT_DIR, get_pipeline_ckpt_name_ablation(strategy, active_modalities))
            if os.path.exists(pipeline_ckpt_path):
                ckpt = torch.load(pipeline_ckpt_path, map_location=device)
                mil_model.load_state_dict(ckpt['mil_state_dict'])
                recon_model.load_state_dict(ckpt['recon_state_dict'])
                fusion_model.load_state_dict(ckpt['fusion_state_dict'])
            
            res = evaluate_test_set(
                mil_model, recon_model, fusion_model, args.clinical_file, args.feature_dir,
                strategy, active_modalities=active_modalities
            )
            res['strategy'] = f"{strategy}_{get_ablation_suffix(active_modalities)}"
            all_ablation_results.append(res)
            
    return all_ablation_results


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
    mil_model, recon_model = None, None
    fusion_results = []
    pipeline_results = []

    if args.stage in ['1', 'all']:
        mil_model, recon_model, fusion_results = train_stage1(
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
        pipeline_results = run_stage2(args, mil_model, recon_model)

    # ─── Test Evaluation ─────────────────────────────────────────
    test_results = []
    if args.stage == 'test':
        test_results, fusion_results, pipeline_results = run_test(args)

    # ─── Ablation ────────────────────────────────────────────────
    ablation_results = []
    if args.stage == 'ablation':
        if not args.ablation_mode:
            logger.error("--ablation_mode is required when --stage ablation is used!")
            sys.exit(1)
        ablation_results = run_ablation(args)

    # ─── Final Summary ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ALL TRAINING COMPLETE")
    logger.info("  Checkpoints: %s", os.path.abspath(CHECKPOINT_DIR))
    logger.info("  TensorBoard: tensorboard --logdir=./runs")

    if fusion_results:
        logger.info("─" * 60)
        logger.info("  STAGE 1 FUSION — Best Results per Strategy:")
        logger.info("  %-15s | %-10s | %-10s | %-10s", "Strategy", "Val Loss", "Val BAcc", "Val F1")
        logger.info("  " + "-" * 55)
        for r in fusion_results:
            logger.info("  %-15s | %-10.4f | %-10.4f | %-10.4f",
                        r['strategy'], r['best_val_loss'], r['best_val_bacc'], r['best_val_f1'])

    if pipeline_results:
        logger.info("─" * 60)
        logger.info("  STAGE 2 PIPELINE — Best Results per Strategy:")
        logger.info("  %-15s | %-10s | %-10s | %-10s", "Strategy", "Val Loss", "Val BAcc", "Val F1")
        logger.info("  " + "-" * 55)
        for r in pipeline_results:
            logger.info("  %-15s | %-10.4f | %-10.4f | %-10.4f",
                        r['strategy'], r['best_val_loss'], r['best_val_bacc'], r['best_val_f1'])

    if test_results:
        logger.info("─" * 60)
        logger.info("  TEST SET — Results per Strategy:")
        logger.info("  %-15s | %-10s | %-10s | %-10s | %-8s", "Strategy", "Test Loss", "Test BAcc", "Test F1", "Samples")
        logger.info("  " + "-" * 65)
        for r in test_results:
            logger.info("  %-15s | %-10.4f | %-10.4f | %-10.4f | %-8d",
                        r['strategy'], r['test_loss'], r['test_bacc'], r['test_f1'], r['test_samples'])
                        
    if ablation_results:
        logger.info("─" * 60)
        logger.info("  ABLATION SET — Results:")
        logger.info("  %-35s | %-10s | %-10s | %-10s | %-8s", "Ablation Run", "Test Loss", "Test BAcc", "Test F1", "Samples")
        logger.info("  " + "-" * 85)
        for r in ablation_results:
            logger.info("  %-35s | %-10.4f | %-10.4f | %-10.4f | %-8d",
                        r['strategy'], r['test_loss'], r['test_bacc'], r['test_f1'], r['test_samples'])

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
