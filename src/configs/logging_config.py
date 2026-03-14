"""
Professional Logging Configuration for MMIST-ccRCC Pipeline
============================================================
Provides structured, scannable log output for both console and file.
Uses centralized paths from configs.paths.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from configs.paths import FILE_LOG_DIR


def setup_logger(name: str = 'mmist'):
    """
    Set up a professional logger with console + file handlers.

    Args:
        name (str): Logger name (default: 'mmist').

    Returns:
        logging.Logger: Configured logger instance.

    Usage:
        from configs.logging_config import setup_logger
        logger = setup_logger()          # root project logger
        logger = setup_logger('trainer') # module-specific logger
    """
    os.makedirs(FILE_LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ─── Console Handler (INFO+) ─────────────────────────────────
    # Compact format for quick scanning during training
    console_fmt = logging.Formatter(
        fmt='%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)

    # ─── File Handler (DEBUG+) ───────────────────────────────────
    # Verbose format with function name and line number for debugging
    file_fmt = logging.Formatter(
        fmt='%(asctime)s │ %(levelname)-8s │ %(name)s.%(funcName)s:%(lineno)d │ %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = RotatingFileHandler(
        filename=os.path.join(FILE_LOG_DIR, 'mmist.log'),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a child logger under the main 'mmist' hierarchy.
    This ensures all modules share the same handlers (console + shared file).
    Additionally, each module gets its **own** log file for easy debugging.

    Args:
        module_name (str): e.g. 'MIL_trainer', 'reconstruction_trainer'

    Returns:
        logging.Logger: Child logger (e.g. mmist.MIL_trainer)

    Log files produced:
        logs/mmist.log              — combined log (all modules)
        logs/mil_trainer.log        — MIL training only
        logs/reconstruction_trainer.log
        logs/fusion_trainer.log
        logs/stage1.log
        logs/stage2_trainer.log
        logs/prepare_data.log
    """
    # Ensure the root logger is configured
    setup_logger('mmist')

    child_logger = logging.getLogger(f'mmist.{module_name}')

    # Only add the per-module file handler once
    handler_tag = f'_file_{module_name}'
    if not any(getattr(h, '_tag', None) == handler_tag for h in child_logger.handlers):
        os.makedirs(FILE_LOG_DIR, exist_ok=True)

        file_fmt = logging.Formatter(
            fmt='%(asctime)s │ %(levelname)-8s │ %(name)s.%(funcName)s:%(lineno)d │ %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        module_file_handler = RotatingFileHandler(
            filename=os.path.join(FILE_LOG_DIR, f'{module_name}.log'),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding='utf-8'
        )
        module_file_handler.setLevel(logging.DEBUG)
        module_file_handler.setFormatter(file_fmt)
        module_file_handler._tag = handler_tag  # mark to prevent duplicates
        child_logger.addHandler(module_file_handler)

    return child_logger
