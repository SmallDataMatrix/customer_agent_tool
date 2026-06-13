"""
ETL Pipeline for Customer Support Ticket Data

This script performs the full Extract, Transform, Load pipeline:
1. Extract: Load raw CSV data
2. Transform: Clean, standardize, enrich data
3. Load: Save to processed CSV files for analysis

Steps:
1. Load raw ticket data
2. Clean and standardize fields
3. Create derived metrics
4. Save processed data to CSV files
5. Create analytics views as CSV
"""

import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Import from SupportIQ package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from supportiq_web import path_manager, logger, log_pipeline_start, log_pipeline_complete, log_dataframe_info


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw CSV data from file."""
    logger.info(f"Loading raw data from: {filepath}")
    
    if not filepath.exists():
        logger.error(f"Raw data file not found: {filepath}")
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.log_data_loaded(filepath.name, len(df), len(df.columns))
    
    return df


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the ticket data."""
    logger.info("Starting data cleaning and standardization...")
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    initial_shape = df_clean.shape
    
    # Standardize column names (lowercase, snake_case)
    df_clean.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_clean.columns]
    logger.debug(f"Standardized column names: {list(df_clean.columns)}")
    
    # Ensure datetime fields
    if 'ticket_created_at' in df_clean.columns:
        df_clean['ticket_created_at'] = pd.to_datetime(df_clean['ticket_created_at'])
        logger.debug(f"Converted 'ticket_created_at' to datetime")
    else:
        df_clean['ticket_created_at'] = pd.date_range(start='2024-01-01', periods=len(df_clean))
        logger.warning("'ticket_created_at' not found, generating dates")
    
    # Standardize priority
    priority_map = {
        'low': 'Low', 'Low': 'Low', 'LOW': 'Low',
        'medium': 'Medium', 'Medium': 'Medium', 'MEDIUM': 'Medium',
        'high': 'High', 'High': 'High', 'HIGH': 'High',
        'critical': 'Critical', 'Critical': 'Critical', 'CRITICAL': 'Critical',
        'urgent': 'Critical'
    }
    df_clean['priority'] = df_clean['priority'].map(priority_map).fillna('Medium')
    logger.debug(f"Priority distribution: {df_clean['priority'].value_counts().to_dict()}")
    
    # Standardize status
    status_map = {
        'open': 'Open', 'Open': 'Open', 'OPEN': 'Open',
        'pending': 'Pending', 'Pending': 'Pending', 'PENDING': 'Pending',
        'solved': 'Solved', 'Solved': 'Solved', 'SOLVED': 'Solved',
        'closed': 'Closed', 'Closed': 'Closed', 'CLOSED': 'Closed',
        'resolved': 'Solved'
    }
    df_clean['status'] = df_clean['status'].map(status_map).fillna('Open')
    
    # Standardize issue type
    issue_map = {
        'shipping': 'Shipping', 'Shipping': 'Shipping', 'SHIPMENT': 'Shipping', 'delivery': 'Shipping',
        'payment': 'Payment', 'Payment': 'Payment', 'PAYMENT': 'Payment', 'refund': 'Payment',
        'account': 'Account', 'Account': 'Account', 'ACCOUNT': 'Account', 'login': 'Account',
        'returns': 'Returns', 'Returns': 'Returns', 'RETURNS': 'Returns', 'exchange': 'Returns',
        'technical': 'Technical', 'Technical': 'Technical', 'TECHNICAL': 'Technical', 'app': 'Technical',
        'product': 'Product', 'Product': 'Product', 'PRODUCT': 'Product', 'pricing': 'Product'
    }
    df_clean['issue_type'] = df_clean['issue_type'].map(issue_map).fillna('Other')
    
    # Handle missing values
    numeric_cols = ['first_response_time_min', 'resolution_time_min', 'csat_score', 
                    'handling_time_min', 'wait_time_min']
    
    missing_before = df_clean[numeric_cols].isnull().sum().sum()
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    missing_after = df_clean[numeric_cols].isnull().sum().sum()
    logger.info(f"Handled missing values: {missing_before} → {missing_after}")
    
    # Ensure positive values for time metrics
    for col in ['first_response_time_min', 'resolution_time_min', 'handling_time_min']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=1)
    
    # Derived fields
    df_clean['created_date'] = df_clean['ticket_created_at'].dt.date
    df_clean['created_hour'] = df_clean['ticket_created_at'].dt.hour
    df_clean['created_dayofweek'] = df_clean['ticket_created_at'].dt.dayofweek
    df_clean['is_weekend'] = df_clean['created_dayofweek'].isin([5, 6]).astype(int)
    df_clean['is_business_hours'] = df_clean['created_hour'].between(9, 17).astype(int)
    
    # CSAT categories
    df_clean['csat_category'] = pd.cut(
        df_clean['csat_score'],
        bins=[-0.001, 2, 3, 5],
        labels=['Negative', 'Neutral', 'Positive'],
        include_lowest=True
    )
    # Replace NaN with 'Unknown' properly for categorical
    df_clean['csat_category'] = df_clean['csat_category'].cat.add_categories(['Unknown'])
    df_clean['csat_category'] = df_clean['csat_category'].fillna('Unknown')
    
    # SLA breach flags (assuming 24h for high priority, 48h for medium, 72h for low)
    df_clean['sla_breach'] = False
    mask_high = (df_clean['priority'] == 'High') & (df_clean['resolution_time_min'] > 24 * 60)
    mask_med = (df_clean['priority'] == 'Medium') & (df_clean['resolution_time_min'] > 48 * 60)
    mask_low = (df_clean['priority'] == 'Low') & (df_clean['resolution_time_min'] > 72 * 60)
    df_clean.loc[mask_high | mask_med | mask_low, 'sla_breach'] = True
    
    sla_breach_count = df_clean['sla_breach'].sum()
    logger.info(f"SLA breach rate: {(sla_breach_count / len(df_clean) * 100):.1f}% ({sla_breach_count}/{len(df_clean)})")
    
    logger.info(f"Cleaned data shape: {initial_shape} → {df_clean.shape}")
    return df_clean


def create_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily metrics from cleaned data."""
    logger.info("Creating daily metrics...")
    
    df['created_date'] = pd.to_datetime(df['created_date'])
    
    daily = df.groupby('created_date').agg(
        total_tickets=('ticket_id', 'count'),
        solved_tickets=('status', lambda x: (x == 'Solved').sum()),
        median_first_response_min=('first_response_time_min', 'median'),
        p75_resolution_min=('resolution_time_min', lambda x: x.quantile(0.75)),
        avg_csat=('csat_score', 'mean'),
        low_csat_rate=('csat_score', lambda x: (x <= 2).sum() * 100.0 / x.notna().sum()),
        sla_breach_rate=('sla_breach', lambda x: x.sum() * 100.0 / len(x)),
        escalated_count=('escalated', 'sum'),
        reopened_count=('reopened', 'sum')
    ).reset_index()
    
    daily['solved_rate'] = daily['solved_tickets'] * 100.0 / daily['total_tickets']
    
    # Round numeric columns
    daily = daily.round(2)
    
    logger.info(f"Created daily metrics for {len(daily)} days")
    return daily


def create_issue_type_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create issue type metrics."""
    logger.info("Creating issue type metrics...")
    
    issue = df.groupby('issue_type').agg(
        total_tickets=('ticket_id', 'count'),
        avg_first_response_min=('first_response_time_min', 'mean'),
        avg_resolution_min=('resolution_time_min', 'mean'),
        avg_csat=('csat_score', 'mean'),
        sla_breach_rate=('sla_breach', lambda x: x.sum() * 100.0 / len(x)),
        escalation_rate=('escalated', lambda x: x.sum() * 100.0 / len(x))
    ).reset_index()
    
    issue = issue.round(2)
    
    logger.info(f"Created metrics for {len(issue)} issue types")
    return issue.sort_values('total_tickets', ascending=False)


def create_agent_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Create agent performance metrics."""
    logger.info("Creating agent performance metrics...")
    
    agent = df[df['agent_id'].notna()].groupby(['agent_id', 'agent_name']).agg(
        tickets_handled=('ticket_id', 'count'),
        avg_first_response_min=('first_response_time_min', 'mean'),
        avg_resolution_min=('resolution_time_min', 'mean'),
        avg_csat=('csat_score', 'mean'),
        solved_rate=('status', lambda x: (x == 'Solved').sum() * 100.0 / len(x)),
        escalation_rate=('escalated', lambda x: x.sum() * 100.0 / len(x))
    ).reset_index()
    
    agent = agent.round(2)
    
    logger.info(f"Created metrics for {len(agent)} agents")
    return agent.sort_values('tickets_handled', ascending=False)


def create_priority_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create priority metrics."""
    logger.info("Creating priority metrics...")
    
    priority_order = {'Critical': 1, 'High': 2, 'Medium': 3, 'Low': 4}
    
    priority = df.groupby('priority').agg(
        total_tickets=('ticket_id', 'count'),
        avg_first_response_min=('first_response_time_min', 'mean'),
        avg_resolution_min=('resolution_time_min', 'mean'),
        avg_csat=('csat_score', 'mean'),
        sla_breach_rate=('sla_breach', lambda x: x.sum() * 100.0 / len(x))
    ).reset_index()
    
    priority['priority_order'] = priority['priority'].map(priority_order).fillna(5)
    priority = priority.round(2)
    
    logger.info(f"Created metrics for {len(priority)} priority levels")
    return priority.sort_values('priority_order')


def create_channel_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create channel metrics."""
    logger.info("Creating channel metrics...")
    
    channel = df.groupby('channel').agg(
        total_tickets=('ticket_id', 'count'),
        avg_first_response_min=('first_response_time_min', 'mean'),
        avg_resolution_min=('resolution_time_min', 'mean'),
        avg_csat=('csat_score', 'mean')
    ).reset_index()
    
    channel = channel.round(2)
    
    logger.info(f"Created metrics for {len(channel)} channels")
    return channel.sort_values('total_tickets', ascending=False)


def save_processed_data(df: pd.DataFrame) -> str:
    """Save processed data to CSV files using PathManager."""
    output_dir = path_manager.PROCESSED_DATA_DIR
    logger.info(f"Saving processed data to: {output_dir}")
    
    # Save cleaned tickets
    output_path = path_manager.get_processed_data_path('tickets_cleaned.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved tickets_cleaned.csv ({len(df):,} records)")
    
    # Create and save analytics views
    daily_df = create_daily_metrics(df)
    daily_df.to_csv(path_manager.get_processed_data_path('daily_metrics.csv'), index=False)
    logger.info(f"Saved daily_metrics.csv ({len(daily_df):,} records)")
    
    issue_df = create_issue_type_metrics(df)
    issue_df.to_csv(path_manager.get_processed_data_path('issue_type_metrics.csv'), index=False)
    logger.info(f"Saved issue_type_metrics.csv ({len(issue_df):,} records)")
    
    agent_df = create_agent_performance(df)
    agent_df.to_csv(path_manager.get_processed_data_path('agent_performance.csv'), index=False)
    logger.info(f"Saved agent_performance.csv ({len(agent_df):,} records)")
    
    priority_df = create_priority_metrics(df)
    priority_df.to_csv(path_manager.get_processed_data_path('priority_metrics.csv'), index=False)
    logger.info(f"Saved priority_metrics.csv ({len(priority_df):,} records)")
    
    channel_df = create_channel_metrics(df)
    channel_df.to_csv(path_manager.get_processed_data_path('channel_metrics.csv'), index=False)
    logger.info(f"Saved channel_metrics.csv ({len(channel_df):,} records)")
    
    return str(output_dir)


def main():
    import time
    start_time = time.time()
    
    log_pipeline_start("ETL Pipeline")
    
    # Configuration using PathManager
    raw_file = path_manager.get_raw_data_path('customer_support_tickets.csv')
    logger.info(f"Raw data path: {raw_file}")
    
    # Check if raw data exists
    if not raw_file.exists():
        logger.error(f"Raw data not found: {raw_file}")
        logger.error("Please run 01_download_kaggle_data.py first")
        return
    
    # ETL Pipeline
    try:
        # Step 1: Extract
        logger.log_step(1, 3, "Extracting raw data")
        df_raw = load_raw_data(raw_file)
        
        # Step 2: Transform
        logger.log_step(2, 3, "Transforming data (cleaning and standardizing)")
        df_clean = clean_and_standardize(df_raw)
        
        # Step 3: Load
        logger.log_step(3, 3, "Loading processed data")
        output_dir = save_processed_data(df_clean)
        
        # Log final statistics
        logger.info("\n=== Pipeline Summary ===")
        logger.info(f"Total records processed: {len(df_clean):,}")
        logger.info(f"Total columns: {len(df_clean.columns)}")
        logger.info(f"Date range: {df_clean['ticket_created_at'].min()} to {df_clean['ticket_created_at'].max()}")
        
        # Log key metrics
        avg_csat = df_clean['csat_score'].mean()
        avg_resolution = df_clean['resolution_time_min'].mean()
        sla_breach_rate = df_clean['sla_breach'].mean() * 100
        
        logger.log_metric("Average CSAT", f"{avg_csat:.2f}")
        logger.log_metric("Average Resolution Time", f"{avg_resolution:.0f}", "minutes")
        logger.log_metric("SLA Breach Rate", f"{sla_breach_rate:.1f}", "%")
        
        duration = time.time() - start_time
        log_pipeline_complete("ETL Pipeline", duration)
        
    except Exception as e:
        logger.error("ETL Pipeline failed", exception=e)
        raise


if __name__ == "__main__":
    main()