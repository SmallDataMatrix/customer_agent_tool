"""
Compatibility module for SupportIQ Web imports.

This module provides backward compatibility for scripts that import from supportiq_web.
The package was renamed to customer_support_analytics.
"""

from customer_support_analytics import __version__, __author__

# Import path_manager from core
from customer_support_analytics.core.path import path_manager

# Import logger from core
from customer_support_analytics.core.logger import logger, log_pipeline_start, log_pipeline_complete, log_dataframe_info

__all__ = [
    '__version__',
    '__author__',
    'path_manager',
    'logger',
    'log_pipeline_start',
    'log_pipeline_complete',
    'log_dataframe_info',
]
