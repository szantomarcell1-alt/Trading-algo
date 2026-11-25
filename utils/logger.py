"""
Structured Logging Module with Rich Console formatting and File Logging.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler

console = Console()

_initialized = False


def setup_logger(
    name: str = "TitanScalper",
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_file_name: str = "titan_scalper.log",
) -> logging.Logger:
    global _initialized
    logger = logging.getLogger(name)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        # Rich Console Handler
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
        rich_handler.setLevel(level)
        logger.addHandler(rich_handler)

        # File Handler (Rotating)
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path / log_file_name,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # Always log debug to file
            logger.addHandler(file_handler)
        except Exception as e:
            console.print(f"[bold red]Warning: Could not create file logger: {e}[/bold red]")

    return logger
