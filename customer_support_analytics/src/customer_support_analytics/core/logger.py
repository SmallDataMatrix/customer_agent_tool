"""
Logging module for Customer Support Analytics.

Provides centralized logging configuration with structured output
for pipeline execution tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m',      # Reset
    }
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class SupportIQLogger:
    """
    Custom logger for SupportIQ with colored console output and file logging.
    """
    
    def __init__(self, name: str = "supportiq", log_dir: Optional[Path] = None):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files. If None, no file logging.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s │ %(levelname)-8s │ %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler without colors
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"supportiq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error message, optionally with exception details."""
        if exception:
            self.logger.error(f"{message}: {type(exception).__name__}: {exception}")
        else:
            self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_step(self, current: int, total: int, description: str):
        """Log a pipeline step progress."""
        self.logger.info(f"[Step {current}/{total}] {description}")
    
    def log_data_loaded(self, filename: str, rows: int, cols: int):
        """Log data loading completion."""
        self.logger.info(f"Loaded {filename}: {rows:,} rows × {cols} columns")
    
    def log_metric(self, metric_name: str, value: str, unit: str = ""):
        """Log a metric value."""
        if unit:
            self.logger.info(f"  📊 {metric_name}: {value} {unit}")
        else:
            self.logger.info(f"  📊 {metric_name}: {value}")
    
    def log_failure(self, step_name: str, error_msg: str):
        """Log a failed step."""
        self.logger.error(f"FAILED: {step_name} - {error_msg}")


# Global logger instance
logger = SupportIQLogger()


def log_pipeline_start(pipeline_name: str):
    """Log the start of a pipeline execution."""
    logger.info(f"{'=' * 60}")
    logger.info(f"PIPELINE START: {pipeline_name}")
    logger.info(f"{'=' * 60}")


def log_pipeline_complete(pipeline_name: str, status: str = "SUCCESS"):
    """Log the completion of a pipeline execution."""
    logger.info(f"{'=' * 60}")
    logger.info(f"PIPELINE COMPLETE: {pipeline_name} [{status}]")
    logger.info(f"{'=' * 60}")


def log_dataframe_info(df, name: str = "DataFrame"):
    """Log information about a pandas DataFrame."""
    logger.info(f"{name} Info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    if len(df) > 0:
        logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
