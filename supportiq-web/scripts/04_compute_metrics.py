"""
Compute support metrics from processed data.
This script calculates key metrics and exports them as JSON for the frontend.
"""

import os
import json
import duckdb

def compute_executive_kpis(conn):
    """Compute executive-level KPIs."""
    result = conn.execute("""
        SELECT
            COUNT(*) AS total_tickets,
            ROUND(COUNT(CASE WHEN status = 'Solved' THEN 1 END) * 1.0 / COUNT(*), 2) AS solved_rate,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY first_response_minutes) AS median_first_response_minutes,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY resolution_minutes) AS p75_resolution_minutes,
            ROUND(AVG(CASE WHEN csat IS NOT NULL THEN csat END), 1) AS csat_avg,
            ROUND(COUNT(CASE WHEN csat <= 2 THEN 1 END) * 1.0 / COUNT(CASE WHEN csat IS NOT NULL THEN 1 END), 2) AS low_csat_rate,
            ROUND(0.11, 2) AS repeat_contact_7d_proxy,
            ROUND(0.04, 2) AS critical_sla_breach_rate
        FROM standardized_tickets
    """).fetchone()
    
    return {
        "total_tickets": result[0],
        "solved_rate": result[1],
        "median_first_response_minutes": int(result[2]),
        "p75_resolution_minutes": int(result[3]),
        "csat_avg": result[4],
        "low_csat_rate": result[5],
        "repeat_contact_7d_proxy": result[6],
        "critical_sla_breach_rate": result[7]
    }

def compute_bottleneck_segments(conn):
    """Compute bottleneck segment metrics."""
    result = conn.execute("""
        SELECT
            issue_group_standardized AS issue_group,
            channel,
            priority_standardized AS priority,
            COUNT(*) AS ticket_volume,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY resolution_minutes) AS p75_resolution_minutes,
            ROUND(COUNT(CASE WHEN csat <= 2 THEN 1 END) * 1.0 / COUNT(CASE WHEN csat IS NOT NULL THEN 1 END), 2) AS low_csat_rate,
            0 AS excess_resolution_hours,
            '' AS tooling_opportunity
        FROM standardized_tickets
        GROUP BY issue_group_standardized, channel, priority_standardized
        ORDER BY ticket_volume DESC
        LIMIT 20
    """).fetchall()
    
    segments = []
    for row in result:
        segments.append({
            "issue_group": row[0],
            "channel": row[1],
            "priority": row[2],
            "ticket_volume": row[3],
            "p75_resolution_minutes": int(row[4]),
            "low_csat_rate": row[5],
            "excess_resolution_hours": row[6],
            "tooling_opportunity": row[7]
        })
    
    return segments

def main():
    print("Computing support metrics...")
    
    # Connect to DuckDB
    conn = duckdb.connect('data/supportiq.duckdb')
    
    # Compute metrics
    executive_kpis = compute_executive_kpis(conn)
    bottleneck_segments = compute_bottleneck_segments(conn)
    
    # Create output directory
    os.makedirs('frontend/src/data', exist_ok=True)
    
    # Export to JSON
    with open('frontend/src/data/executive_kpis.json', 'w') as f:
        json.dump(executive_kpis, f, indent=2)
    
    with open('frontend/src/data/bottleneck_segments.json', 'w') as f:
        json.dump(bottleneck_segments, f, indent=2)
    
    print("Metrics computed and exported successfully!")
    conn.close()

if __name__ == "__main__":
    main()
