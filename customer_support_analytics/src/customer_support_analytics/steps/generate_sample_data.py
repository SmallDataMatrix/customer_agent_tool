"""Generate synthetic sample data for the Customer Support Analytics platform.

This script creates realistic-looking synthetic support ticket data and the derived
analytics CSV files used by the Streamlit dashboard. It is intended for --launch-only
mode when no real Kaggle data has been downloaded.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from customer_support_analytics.core.path import path_manager

np.random.seed(42)

def generate_tickets(n=2000):
    """Generate a synthetic support-ticket dataset with realistic distributions."""
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-12-31")
    total_days = (end_date - start_date).days

    ticket_ids = np.arange(1, n + 1)
    created_offsets = np.random.randint(0, total_days, n)
    created_times = [start_date + pd.Timedelta(days=int(d), hours=int(np.random.randint(7, 22))) for d in created_offsets]
    ticket_created_at = pd.Series(created_times)

    priorities = np.random.choice(["Low", "Medium", "High", "Critical"], n, p=[0.35, 0.40, 0.18, 0.07])
    statuses = np.random.choice(["Solved", "Closed", "Open", "Pending"], n, p=[0.55, 0.25, 0.12, 0.08])
    issue_types = np.random.choice(
        ["Shipping", "Payment", "Account", "Returns", "Technical", "Product", "Other"],
        n, p=[0.22, 0.18, 0.15, 0.12, 0.15, 0.10, 0.08],
    )
    channels = np.random.choice(["Email", "Phone", "Chat", "Web", "Social"], n, p=[0.30, 0.22, 0.25, 0.15, 0.08])

    first_response = np.random.exponential(45, n).clip(1, 480)
    resolution = (first_response + np.random.exponential(120, n)).clip(1, 2880)
    handling = np.random.exponential(30, n).clip(1, 240)
    wait_time = np.random.exponential(15, n).clip(1, 240)

    csat_raw = np.clip(np.random.normal(3.8, 1.0, n), 1, 5).round(1)
    csat = np.where(np.random.rand(n) < 0.15, np.nan, csat_raw)

    agent_ids = np.random.choice(np.arange(1001, 1031), n)
    agent_names = [f"Agent_{aid}" for aid in agent_ids]
    escalated = np.random.choice([0, 1], n, p=[0.82, 0.18])
    reopened = np.random.choice([0, 1], n, p=[0.90, 0.10])

    df = pd.DataFrame({
        "ticket_id": ticket_ids,
        "ticket_created_at": ticket_created_at,
        "priority": priorities,
        "status": statuses,
        "issue_type": issue_types,
        "channel": channels,
        "first_response_time_min": np.round(first_response, 1),
        "resolution_time_min": np.round(resolution, 1),
        "handling_time_min": np.round(handling, 1),
        "wait_time_min": np.round(wait_time, 1),
        "csat_score": csat,
        "agent_id": agent_ids,
        "agent_name": agent_names,
        "escalated": escalated,
        "reopened": reopened,
    })

    df["created_date"] = pd.to_datetime(df["ticket_created_at"].dt.date)
    df["created_hour"] = df["ticket_created_at"].dt.hour
    df["created_dayofweek"] = df["ticket_created_at"].dt.dayofweek
    df["is_weekend"] = df["created_dayofweek"].isin([5, 6]).astype(int)
    df["is_business_hours"] = df["created_hour"].between(9, 17).astype(int)

    df["csat_category"] = pd.cut(
        df["csat_score"].fillna(3),
        bins=[-0.001, 2, 3, 5],
        labels=["Negative", "Neutral", "Positive"],
        include_lowest=True,
    )
    df["csat_category"] = df["csat_category"].cat.add_categories(["Unknown"])
    df.loc[df["csat_score"].isna(), "csat_category"] = "Unknown"

    df["sla_breach"] = False
    sla_map = {"High": 24 * 60, "Medium": 48 * 60, "Low": 72 * 60, "Critical": 12 * 60}
    for prio, limit in sla_map.items():
        mask = (df["priority"] == prio) & (df["resolution_time_min"] > limit)
        df.loc[mask, "sla_breach"] = True

    return df


def build_analytics(df):
    """Build the analytics views expected by the Streamlit dashboard."""
    df = df.copy()
    df["created_date"] = pd.to_datetime(df["created_date"])

    daily = df.groupby("created_date").agg(
        total_tickets=("ticket_id", "count"),
        solved_tickets=("status", lambda x: (x == "Solved").sum()),
        median_first_response_min=("first_response_time_min", "median"),
        p75_resolution_min=("resolution_time_min", lambda x: x.quantile(0.75)),
        avg_csat=("csat_score", "mean"),
        low_csat_rate=("csat_score", lambda x: (x <= 2).sum() * 100.0 / x.notna().sum()),
        sla_breach_rate=("sla_breach", lambda x: x.sum() * 100.0 / len(x)),
        escalated_count=("escalated", "sum"),
        reopened_count=("reopened", "sum"),
    ).reset_index()
    daily["solved_rate"] = daily["solved_tickets"] * 100.0 / daily["total_tickets"]
    daily = daily.round(2)

    issue = df.groupby("issue_type").agg(
        total_tickets=("ticket_id", "count"),
        avg_first_response_min=("first_response_time_min", "mean"),
        avg_resolution_min=("resolution_time_min", "mean"),
        avg_csat=("csat_score", "mean"),
        sla_breach_rate=("sla_breach", lambda x: x.sum() * 100.0 / len(x)),
        escalation_rate=("escalated", lambda x: x.sum() * 100.0 / len(x)),
    ).reset_index().round(2).sort_values("total_tickets", ascending=False)

    agent = df.groupby(["agent_id", "agent_name"]).agg(
        tickets_handled=("ticket_id", "count"),
        avg_first_response_min=("first_response_time_min", "mean"),
        avg_resolution_min=("resolution_time_min", "mean"),
        avg_csat=("csat_score", "mean"),
        solved_rate=("status", lambda x: (x == "Solved").sum() * 100.0 / len(x)),
        escalation_rate=("escalated", lambda x: x.sum() * 100.0 / len(x)),
    ).reset_index().round(2).sort_values("tickets_handled", ascending=False)

    priority_order = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
    priority = df.groupby("priority").agg(
        total_tickets=("ticket_id", "count"),
        avg_first_response_min=("first_response_time_min", "mean"),
        avg_resolution_min=("resolution_time_min", "mean"),
        avg_csat=("csat_score", "mean"),
        sla_breach_rate=("sla_breach", lambda x: x.sum() * 100.0 / len(x)),
    ).reset_index()
    priority["priority_order"] = priority["priority"].map(priority_order).fillna(5)
    priority = priority.round(2).sort_values("priority_order")

    channel = df.groupby("channel").agg(
        total_tickets=("ticket_id", "count"),
        avg_first_response_min=("first_response_time_min", "mean"),
        avg_resolution_min=("resolution_time_min", "mean"),
        avg_csat=("csat_score", "mean"),
    ).reset_index().round(2).sort_values("total_tickets", ascending=False)

    return {
        "tickets_cleaned.csv": df,
        "daily_metrics.csv": daily,
        "issue_type_metrics.csv": issue,
        "agent_performance.csv": agent,
        "priority_metrics.csv": priority,
        "channel_metrics.csv": channel,
    }


def main():
    print("Generating sample support ticket data...")
    df = generate_tickets(n=2500)
    tables = build_analytics(df)

    out_dir = path_manager.PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path_manager.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, table in tables.items():
        path = out_dir / filename
        table.to_csv(path, index=False)
        print(f"  Saved {filename}: {len(table):,} rows  ->  {path}")

    raw_path = path_manager.RAW_DATA_DIR / "customer_support_tickets.csv"
    df.to_csv(raw_path, index=False)
    print(f"  Raw copy: {raw_path}")

    print(f"\nDone. Sample data for {len(df):,} tickets written to {out_dir}")


if __name__ == "__main__":
    main()
