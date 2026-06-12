-- Calculate core ticket metrics
-- This script computes key support metrics for analysis

-- Create daily ticket metrics
CREATE TABLE IF NOT EXISTS daily_ticket_metrics AS
SELECT
    DATE(created_at) AS date,
    issue_group_standardized AS issue_group,
    priority_standardized AS priority,
    channel,
    COUNT(*) AS ticket_volume,
    COUNT(CASE WHEN status = 'Solved' THEN 1 END) AS solved_count,
    AVG(CASE WHEN csat IS NOT NULL THEN csat END) AS csat_avg,
    COUNT(CASE WHEN csat <= 2 THEN 1 END) AS low_csat_count,
    COUNT(CASE WHEN priority_standardized = 'Critical' THEN 1 END) AS critical_count,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_minutes) AS resolution_minutes_p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY resolution_minutes) AS resolution_minutes_p75,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY resolution_minutes) AS resolution_minutes_p90,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY first_response_minutes) AS first_response_minutes_p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY first_response_minutes) AS first_response_minutes_p75
FROM standardized_tickets
GROUP BY 1, 2, 3, 4;

-- Create overall daily metrics
CREATE TABLE IF NOT EXISTS daily_overall_metrics AS
SELECT
    DATE(created_at) AS date,
    COUNT(*) AS total_tickets,
    COUNT(CASE WHEN status = 'Solved' THEN 1 END) AS solved_count,
    ROUND(COUNT(CASE WHEN status = 'Solved' THEN 1 END) * 100.0 / COUNT(*), 2) AS solved_rate,
    AVG(CASE WHEN csat IS NOT NULL THEN csat END) AS csat_avg,
    ROUND(COUNT(CASE WHEN csat <= 2 THEN 1 END) * 100.0 / COUNT(CASE WHEN csat IS NOT NULL THEN 1 END), 2) AS low_csat_rate,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_minutes) AS resolution_minutes_p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY resolution_minutes) AS resolution_minutes_p75,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY first_response_minutes) AS first_response_minutes_p50,
    ROUND(COUNT(CASE WHEN priority_standardized = 'Critical' AND sla_breached THEN 1 END) * 100.0 / COUNT(CASE WHEN priority_standardized = 'Critical' THEN 1 END), 2) AS critical_sla_breach_rate
FROM standardized_tickets
GROUP BY 1;
