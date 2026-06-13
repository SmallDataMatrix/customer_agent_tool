"""
Download Kaggle Customer Support Ticket Dataset using kagglehub
This script downloads the real customer support ticket dataset without requiring Kaggle login.

Dataset: https://www.kaggle.com/suraj520/customer-support-ticket-dataset

Requirements:
- pip install kagglehub pandas

kagglehub allows downloading public Kaggle datasets without authentication.
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Import from SupportIQ package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from supportiq_web import path_manager, logger, log_pipeline_start, log_pipeline_complete


def download_kagglehub_dataset(dataset_name: str, download_path: Path) -> bool:
    """Download dataset from Kaggle using kagglehub (no login required)."""
    import kagglehub
    
    logger.info(f"Downloading dataset: {dataset_name}")
    try:
        # Download dataset using kagglehub
        path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Dataset downloaded to: {path}")
        
        # Copy files to our raw data directory
        source_dir = Path(path)
        dest_dir = download_path
        
        if source_dir.exists() and source_dir.is_dir():
            for file in source_dir.iterdir():
                if file.is_file():
                    shutil.copy(file, dest_dir / file.name)
                    logger.info(f"Copied: {file.name}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to download with kagglehub", exception=e)
        return False


def download_fallback_data() -> bool:
    """Download fallback data from public GitHub sources."""
    import requests
    
    logger.info("\nDownloading fallback dataset from public GitHub sources...")
    
    # URLs for public customer support datasets
    urls = [
        ("https://raw.githubusercontent.com/GoogleCloudPlatform/training-data-analyst/master/courses/data-analyst/intro-to-sql/data/support_tickets.csv", "support_tickets.csv"),
        ("https://raw.githubusercontent.com/databricks/LearningSparkV2/master/chapter4/python/data/sf-fire-calls.csv", "sf_fire_calls.csv")
    ]
    
    success_count = 0
    for url, filename in urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            file_path = path_manager.get_raw_data_path(filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded: {filename}")
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to download {filename}", exception=e)
    
    return success_count > 0


def create_sample_dataset() -> pd.DataFrame:
    """Create a comprehensive sample dataset based on real Kaggle data structure."""
    logger.info("\nCreating realistic sample dataset based on Kaggle Customer Support Ticket structure...")
    logger.info("This generates 5000 ticket records with realistic distributions...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate realistic data based on typical support ticket patterns
    n_records = 5000
    
    logger.debug(f"Generating {n_records} records with random seed 42")
    
    data = {
        'ticket_id': [f'TKT-{i:06d}' for i in range(1, n_records + 1)],
        'ticket_created_at': pd.date_range(start='2024-01-01', end='2024-01-31', periods=n_records),
        'channel': np.random.choice(
            ['Email', 'Chat', 'Phone', 'Social Media'], 
            size=n_records,
            p=[0.45, 0.30, 0.15, 0.10]
        ),
        'priority': np.random.choice(
            ['Low', 'Medium', 'High', 'Critical'],
            size=n_records,
            p=[0.30, 0.45, 0.20, 0.05]
        ),
        'issue_type': np.random.choice(
            ['Shipping', 'Payment', 'Account', 'Returns', 'Technical', 'Product'],
            size=n_records,
            p=[0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        ),
        'sub_issue': np.random.choice([
            'Delivery delay', 'Lost package', 'Wrong item',
            'Refund request', 'Payment failed', 'Chargeback',
            'Login issue', 'Account blocked', 'Password reset',
            'Return request', 'Refund status', 'Exchange',
            'App crash', 'Website error', 'API issue',
            'Product info', 'Pricing', 'Availability'
        ], size=n_records),
        'customer_id': [f'CUST-{i:06d}' for i in range(1, n_records + 1)],
        'customer_name': [f'Customer {i}' for i in range(1, n_records + 1)],
        'customer_email': [f'customer{i}@example.com' for i in range(1, n_records + 1)],
        'agent_id': np.random.choice([f'AGT-{i:03d}' for i in range(1, 21)], size=n_records),
        'agent_name': np.random.choice([f'Agent {i}' for i in range(1, 21)], size=n_records),
        'status': np.random.choice(
            ['Open', 'Pending', 'Solved', 'Closed'],
            size=n_records,
            p=[0.10, 0.15, 0.60, 0.15]
        ),
        'first_response_time_min': [max(1, round(np.random.normal(30, 20))) for _ in range(n_records)],
        'resolution_time_min': [max(1, round(np.random.normal(240, 180))) for _ in range(n_records)],
        'csat_score': np.random.choice([1, 2, 3, 4, 5], size=n_records, p=[0.05, 0.08, 0.15, 0.32, 0.4]),
        'ticket_text': [f"This is a support ticket regarding {issue}. The customer needs assistance with their request." 
                       for issue in np.random.choice(['shipping', 'payment', 'account', 'returns', 'technical', 'product'], size=n_records)],
        'tags': np.random.choice(['urgent', 'refund', 'login', 'shipping', 'technical', 'question'], size=n_records),
        'solved_by_agent': np.random.choice([True, False], size=n_records, p=[0.85, 0.15]),
        'reopened': np.random.choice([True, False], size=n_records, p=[0.08, 0.92]),
        'queue_name': np.random.choice([
            'General Support', 'Shipping Team', 'Payment Team', 'Technical Support', 'Account Management'
        ], size=n_records),
        'language': np.random.choice(['English', 'Spanish', 'French', 'German'], size=n_records, p=[0.80, 0.10, 0.06, 0.04]),
        'timezone': np.random.choice(['UTC', 'EST', 'CET', 'PST'], size=n_records, p=[0.40, 0.30, 0.20, 0.10]),
        'escalated': np.random.choice([True, False], size=n_records, p=[0.05, 0.95]),
        'handling_time_min': [max(1, round(np.random.normal(120, 90))) for _ in range(n_records)],
        'wait_time_min': [max(0, round(np.random.normal(15, 10))) for _ in range(n_records)],
    }
    
    df = pd.DataFrame(data)
    
    # Ensure all numeric fields are positive
    df['first_response_time_min'] = df['first_response_time_min'].clip(lower=1)
    df['resolution_time_min'] = df['resolution_time_min'].clip(lower=1)
    df['handling_time_min'] = df['handling_time_min'].clip(lower=1)
    df['wait_time_min'] = df['wait_time_min'].clip(lower=0)
    
    # Save to CSV using path manager
    output_path = path_manager.get_raw_data_path('customer_support_tickets.csv')
    df.to_csv(output_path, index=False)
    
    logger.log_data_loaded('customer_support_tickets.csv', len(df), len(df.columns))
    logger.info(f"Created sample dataset: {output_path}")
    
    return df


def main():
    import time
    start_time = time.time()
    
    log_pipeline_start("Kaggle Data Downloader")
    
    # Use path manager to ensure directories exist
    path_manager._create_directories()
    logger.info(f"Base directory: {path_manager.BASE_DIR}")
    
    # Check if raw data already exists
    raw_data_path = path_manager.get_raw_data_path('customer_support_tickets.csv')
    if raw_data_path.exists():
        logger.info(f"Raw data already exists at: {raw_data_path}")
        logger.info("Skipping data download. To re-download, delete the existing file first.")
        duration = time.time() - start_time
        log_pipeline_complete("Kaggle Data Downloader", duration)
        return
    
    logger.info("Attempting to download dataset...")
    
    # Try kagglehub first
    logger.info("\n[Step 1/2] Trying to download from Kaggle using kagglehub...")
    kagglehub_success = download_kagglehub_dataset(
        "suraj520/customer-support-ticket-dataset",
        path_manager.RAW_DATA_DIR
    )
    
    if not kagglehub_success:
        # Try fallback sources
        logger.info("\n[Step 2/2] Kaggle download failed. Trying fallback sources...")
        fallback_success = download_fallback_data()
        
        if not fallback_success:
            # All download methods failed - raise error
            error_msg = (
                "Failed to download dataset from all sources.\n"
                "Please check your internet connection and try again.\n"
                "Or manually place the 'customer_support_tickets.csv' file in:\n"
                f"  {path_manager.RAW_DATA_DIR}"
            )
            logger.log_failure("Data Download", error_msg)
            raise RuntimeError(error_msg)
    
    # Verify the data file exists after download
    if not raw_data_path.exists():
        error_msg = (
            f"Download completed but data file not found at: {raw_data_path}\n"
            "Please check the download process and try again."
        )
        logger.log_failure("Data Verification", error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load and display statistics
    df = pd.read_csv(raw_data_path)
    
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"  Total records: {len(df):,}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  File location: {raw_data_path}")
    
    duration = time.time() - start_time
    log_pipeline_complete("Kaggle Data Downloader", duration)


if __name__ == "__main__":
    main()