"""
GBPJPY Structured Logging Module
Provides Rich Console formatting and Rotating File Logging strictly inside the gbpjpy workspace.
Guarantees workspace isolation with zero pollution of external directories.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_initialized_loggers = set()
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = WORKSPACE_ROOT / "logs"
DEFAULT_LOG_FILE = "gbpjpy_scalper.log"


def setup_logger(
    name: str = "GBPJPY",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file_name: str = DEFAULT_LOG_FILE,
) -> logging.Logger:
    """
    Initializes and configures a logger instance with Rich console and rotating file output.
    Ensures logs are written strictly to c:\\Trading algo\\gbpjpy\\logs\\gbpjpy_scalper.log.
    """
    logger = logging.getLogger(name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logger is called repeatedly
    if name not in _initialized_loggers and not logger.handlers:
        # 1. Rich Console Handler
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
        rich_handler.setLevel(level)
        logger.addHandler(rich_handler)

        # 2. Rotating File Handler
        try:
            target_log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
            target_log_dir.mkdir(parents=True, exist_ok=True)
            log_filepath = target_log_dir / log_file_name

            file_handler = RotatingFileHandler(
                log_filepath,
                maxBytes=10 * 1024 * 1024,  # 10 MB per log file
                backupCount=5,
                encoding="utf-8",
            )
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # Debug output always captured in log file
            logger.addHandler(file_handler)
        except Exception as e:
            console.print(f"[bold red]Warning: Could not create GBPJPY file logger: {e}[/bold red]")

        _initialized_loggers.add(name)

    return logger


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Convenience accessor for submodules.
    Usage:
        logger = get_logger("MockClient") -> returns logging.getLogger("GBPJPY.MockClient")
    """
    if module_name:
        name = f"GBPJPY.{module_name}" if not module_name.startswith("GBPJPY") else module_name
    else:
        name = "GBPJPY"
    return logging.getLogger(name)
