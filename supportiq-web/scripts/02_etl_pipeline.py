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

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data from file."""
    print(f"Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    return df

def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the ticket data."""
    print("Cleaning and standardizing data...")
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Standardize column names (lowercase, snake_case)
    df_clean.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_clean.columns]
    
    # Ensure datetime fields
    if 'ticket_created_at' in df_clean.columns:
        df_clean['ticket_created_at'] = pd.to_datetime(df_clean['ticket_created_at'])
    else:
        df_clean['ticket_created_at'] = pd.date_range(start='2024-01-01', periods=len(df_clean))
    
    # Standardize priority
    priority_map = {
        'low': 'Low', 'Low': 'Low', 'LOW': 'Low',
        'medium': 'Medium', 'Medium': 'Medium', 'MEDIUM': 'Medium',
        'high': 'High', 'High': 'High', 'HIGH': 'High',
        'critical': 'Critical', 'Critical': 'Critical', 'CRITICAL': 'Critical',
        'urgent': 'Critical'
    }
    df_clean['priority'] = df_clean['priority'].map(priority_map).fillna('Medium')
    
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
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
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
        bins=[0, 2, 3, 5],
        labels=['Negative', 'Neutral', 'Positive']
    ).fillna('Unknown')
    
    # SLA breach flags (assuming 24h for high priority, 48h for medium, 72h for low)
    df_clean['sla_breach'] = False
    mask_high = (df_clean['priority'] == 'High') & (df_clean['resolution_time_min'] > 24 * 60)
    mask_med = (df_clean['priority'] == 'Medium') & (df_clean['resolution_time_min'] > 48 * 60)
    mask_low = (df_clean['priority'] == 'Low') & (df_clean['resolution_time_min'] > 72 * 60)
    df_clean.loc[mask_high | mask_med | mask_low, 'sla_breach'] = True
    
    print(f"Cleaned data shape: {df_clean.shape}")
    return df_clean

def create_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily metrics from cleaned data."""
    print("Creating daily metrics...")
    
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
    
    return daily

def create_issue_type_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create issue type metrics."""
    print("Creating issue type metrics...")
    
    issue = df.groupby('issue_type').agg(
        total_tickets=('ticket_id', 'count'),
        avg_first_response_min=('first_response_time_min', 'mean'),
        avg_resolution_min=('resolution_time_min', 'mean'),
        avg_csat=('csat_score', 'mean'),
        sla_breach_rate=('sla_breach', lambda x: x.sum() * 100.0 / len(x)),
        escalation_rate=('escalated', lambda x: x.sum() * 100.0 / len(x))
    ).reset_index()
    
    issue = issue.round(2)
    return issue.sort_values('total_tickets', ascending=False)

def create_agent_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Create agent performance metrics."""
    print("Creating agent performance metrics...")
    
    agent = df[df['agent_id'].notna()].groupby(['agent_id', 'agent_name']).agg(
        tickets_handled=('ticket_id', 'count'),
        avg_first_response_min=('first_response_time_min', 'mean'),
        avg_resolution_min=('resolution_time_min', 'mean'),
        avg_csat=('csat_score', 'mean'),
        solved_rate=('status', lambda x: (x == 'Solved').sum() * 100.0 / len(x)),
        escalation_rate=('escalated', lambda x: x.sum() * 100.0 / len(x))
    ).reset_index()
    
    agent = agent.round(2)
    return agent.sort_values('tickets_handled', ascending=False)

def create_priority_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create priority metrics."""
    print("Creating priority metrics...")
    
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
    return priority.sort_values('priority_order')

def create_channel_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create channel metrics."""
    print("Creating channel metrics...")
    
    channel = df.groupby('channel').agg(
        total_tickets=('ticket_id', 'count'),
        avg_first_response_min=('first_response_time_min', 'mean'),
        avg_resolution_min=('resolution_time_min', 'mean'),
        avg_csat=('csat_score', 'mean')
    ).reset_index()
    
    channel = channel.round(2)
    return channel.sort_values('total_tickets', ascending=False)

def save_processed_data(df: pd.DataFrame, output_dir: str = 'data/processed'):
    """Save processed data to CSV files."""
    print(f"Saving processed data to: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save cleaned tickets
    df.to_csv(f'{output_dir}/tickets_cleaned.csv', index=False)
    print(f"[OK] Saved tickets_cleaned.csv")
    
    # Create and save analytics views
    daily_df = create_daily_metrics(df)
    daily_df.to_csv(f'{output_dir}/daily_metrics.csv', index=False)
    print(f"[OK] Saved daily_metrics.csv")
    
    issue_df = create_issue_type_metrics(df)
    issue_df.to_csv(f'{output_dir}/issue_type_metrics.csv', index=False)
    print(f"[OK] Saved issue_type_metrics.csv")
    
    agent_df = create_agent_performance(df)
    agent_df.to_csv(f'{output_dir}/agent_performance.csv', index=False)
    print(f"[OK] Saved agent_performance.csv")
    
    priority_df = create_priority_metrics(df)
    priority_df.to_csv(f'{output_dir}/priority_metrics.csv', index=False)
    print(f"[OK] Saved priority_metrics.csv")
    
    channel_df = create_channel_metrics(df)
    channel_df.to_csv(f'{output_dir}/channel_metrics.csv', index=False)
    print(f"[OK] Saved channel_metrics.csv")
    
    return output_dir

def main():
    print("=" * 60)
    print("ETL Pipeline - Customer Support Ticket Data")
    print("=" * 60)
    
    # Configuration
    raw_file = 'data/raw/customer_support_tickets.csv'
    output_dir = 'data/processed'
    
    # Check if raw data exists
    if not os.path.exists(raw_file):
        print(f"Raw data not found: {raw_file}")
        print("Please run 01_download_kaggle_data.py first")
        return
    
    # ETL Pipeline
    try:
        df_raw = load_raw_data(raw_file)
        df_clean = clean_and_standardize(df_raw)
        save_processed_data(df_clean, output_dir)
        
        print("\n" + "=" * 60)
        print("ETL Pipeline completed successfully!")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Records processed: {len(df_clean)}")
        print(f"Columns: {list(df_clean.columns)}")
        
    except Exception as e:
        print(f"ETL Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
