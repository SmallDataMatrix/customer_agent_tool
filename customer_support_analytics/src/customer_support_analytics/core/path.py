"""
Path management module for Customer Support Analytics.

Provides centralized path management for data, models, and output directories.
"""

from pathlib import Path
import os


class PathManager:
    """
    Centralized path management for the application.
    
    This class resolves all file paths relative to the project root,
    ensuring consistent location of data, models, and output files
    across different execution contexts.
    """
    
    def __init__(self, project_root=None):
        """
        Initialize PathManager with the project root directory.
        
        Args:
            project_root: Path to project root. If None, auto-detects from this file's location.
        """
        if project_root is None:
            # Auto-detect: src/customer_support_analytics/core -> src/customer_support_analytics -> src -> project root
            self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        # Define standard directory names
        self.DATA_DIR = self.project_root / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.TRAINED_MODEL_DIR = self.project_root / "trained_model"
        self.SQL_DIR = self.project_root / "src" / "customer_support_analytics" / "sql"
        self.STEPS_DIR = self.project_root / "src" / "customer_support_analytics" / "steps"
        self.OUTPUT_DIR = self.project_root / "output"
        
        # Aliases for backward compatibility
        self.BASE_DIR = self.project_root  # Alias for project_root
        self.MODELS_DIR = self.TRAINED_MODEL_DIR  # Alias for TRAINED_MODEL_DIR
    
    def _create_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.TRAINED_MODEL_DIR,
            self.OUTPUT_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_raw_data_path(self, filename: str) -> Path:
        """Get path to a raw data file."""
        return self.RAW_DATA_DIR / filename
    
    def get_processed_data_path(self, filename: str) -> Path:
        """Get path to a processed data file."""
        return self.PROCESSED_DATA_DIR / filename
    
    def get_model_path(self, filename: str) -> Path:
        """Get path to a trained model file."""
        # Ensure directory exists
        self.TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return self.TRAINED_MODEL_DIR / filename
    
    def get_sql_path(self, filename: str) -> Path:
        """Get path to a SQL script file."""
        return self.SQL_DIR / filename
    
    def get_step_path(self, step_name: str) -> Path:
        """Get path to a pipeline step script."""
        return self.STEPS_DIR / step_name
    
    def get_output_path(self, filename: str) -> Path:
        """Get path to an output file."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return self.OUTPUT_DIR / filename
    
    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return self.project_root


# Global path manager instance
path_manager = PathManager()
