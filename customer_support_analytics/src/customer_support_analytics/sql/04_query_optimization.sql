-- Query Optimization and Performance Indexes
-- This script adds optimized indexes and materialized views for improved query performance

-- ============================================================================
-- PART 1: PERFORMANCE INDEXES
-- ============================================================================

-- Primary indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_created_at_date
ON cleaned_tickets(DATE(created_at));

CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_status
ON cleaned_tickets(status);

CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_priority_standardized
ON cleaned_tickets(priority_standardized);

CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_issue_group_standardized
ON cleaned_tickets(issue_group_standardized);

CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_channel
ON cleaned_tickets(channel);

CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_csat
ON cleaned_tickets(csat)
WHERE csat IS NOT NULL;

-- Composite indexes for common multi-column queries
CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_date_priority
ON cleaned_tickets(DATE(created_at), priority_standardized);

CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_date_status
ON cleaned_tickets(DATE(created_at), status);

CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_date_issue
ON cleaned_tickets(DATE(created_at), issue_group_standardized);

CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_priority_status
ON cleaned_tickets(priority_standardized, status);

-- Index for CSAT analysis (commonly queried with priority)
CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_priority_csat
ON cleaned_tickets(priority_standardized, csat)
WHERE csat IS NOT NULL;

-- Index for time-based SLA calculations
CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_resolution_time
ON cleaned_tickets(resolution_minutes);

CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_first_response_time
ON cleaned_tickets(first_response_minutes);

-- ============================================================================
-- PART 2: MATERIALIZED VIEWS FOR FREQUENTLY ACCESSED DATA
-- ============================================================================

-- Materialized view: Agent performance summary (refresh hourly)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_agent_performance AS
SELECT
    agent_id,
    COUNT(*) AS total_tickets,
    COUNT(CASE WHEN status = 'Solved' THEN 1 END) AS solved_tickets,
    ROUND(COUNT(CASE WHEN status = 'Solved' THEN 1 END) * 100.0 / COUNT(*), 2) AS solved_rate,
    AVG(CASE WHEN csat IS NOT NULL THEN csat END) AS avg_csat,
    AVG(first_response_minutes) AS avg_first_response_minutes,
    AVG(resolution_minutes) AS avg_resolution_minutes,
    COUNT(CASE WHEN priority_standardized = 'Critical' THEN 1 END) AS critical_tickets,
    COUNT(CASE WHEN sla_breached THEN 1 END) AS sla_breaches
FROM cleaned_tickets
WHERE agent_id IS NOT NULL
GROUP BY agent_id;

CREATE INDEX IF NOT EXISTS idx_mv_agent_performance_agent
ON mv_agent_performance(agent_id);

-- Materialized view: Daily metrics snapshot (refresh hourly)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_metrics AS
SELECT
    DATE(created_at) AS date,
    COUNT(*) AS total_tickets,
    COUNT(CASE WHEN status = 'Solved' THEN 1 END) AS solved_count,
    ROUND(COUNT(CASE WHEN status = 'Solved' THEN 1 END) * 100.0 / COUNT(*), 2) AS solved_rate,
    AVG(CASE WHEN csat IS NOT NULL THEN csat END) AS avg_csat,
    ROUND(COUNT(CASE WHEN csat <= 2 THEN 1 END) * 100.0 /
          COUNT(CASE WHEN csat IS NOT NULL THEN 1 END), 2) AS low_csat_rate,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY first_response_minutes) AS median_first_response_minutes,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_minutes) AS median_resolution_minutes,
    ROUND(COUNT(CASE WHEN sla_breached THEN 1 END) * 100.0 / COUNT(*), 2) AS sla_breach_rate
FROM cleaned_tickets
GROUP BY 1;

CREATE INDEX IF NOT EXISTS idx_mv_daily_metrics_date
ON mv_daily_metrics(date);

-- Materialized view: Issue type performance (refresh daily)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_issue_performance AS
SELECT
    issue_group_standardized AS issue_group,
    priority_standardized AS priority,
    channel,
    COUNT(*) AS ticket_volume,
    ROUND(COUNT(CASE WHEN status = 'Solved' THEN 1 END) * 100.0 / COUNT(*), 2) AS solved_rate,
    AVG(CASE WHEN csat IS NOT NULL THEN csat END) AS avg_csat,
    AVG(first_response_minutes) AS avg_first_response_minutes,
    AVG(resolution_minutes) AS avg_resolution_minutes,
    COUNT(CASE WHEN sla_breached THEN 1 END) AS sla_breaches
FROM cleaned_tickets
GROUP BY 1, 2, 3;

CREATE INDEX IF NOT EXISTS idx_mv_issue_performance_issue
ON mv_issue_performance(issue_group);

-- Materialized view: Hourly volume patterns (refresh daily)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_hourly_patterns AS
SELECT
    EXTRACT(HOUR FROM created_at) AS hour_of_day,
    EXTRACT(DOW FROM created_at) AS day_of_week,
    issue_group_standardized AS issue_group,
    COUNT(*) AS ticket_volume,
    AVG(CASE WHEN csat IS NOT NULL THEN csat END) AS avg_csat
FROM cleaned_tickets
GROUP BY 1, 2, 3;

CREATE INDEX IF NOT EXISTS idx_mv_hourly_patterns_hour
ON mv_hourly_patterns(hour_of_day);

-- ============================================================================
-- PART 3: OPTIMIZED AGGREGATION QUERIES
-- ============================================================================

-- Pre-computed summary for executive dashboard (uses materialized view)
CREATE OR REPLACE FUNCTION fn_get_executive_dashboard(date_start DATE, date_end DATE)
RETURNS TABLE (
    total_tickets BIGINT,
    solved_rate DECIMAL,
    avg_csat DECIMAL,
    median_first_response DECIMAL,
    median_resolution DECIMAL,
    sla_breach_rate DECIMAL,
    low_csat_rate DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        SUM(total_tickets)::BIGINT AS total_tickets,
        ROUND(AVG(solved_rate), 2) AS solved_rate,
        ROUND(AVG(avg_csat), 2) AS avg_csat,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY median_first_response_minutes) AS median_first_response,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY median_resolution_minutes) AS median_resolution,
        ROUND(AVG(sla_breach_rate), 2) AS sla_breach_rate,
        ROUND(AVG(low_csat_rate), 2) AS low_csat_rate
    FROM mv_daily_metrics
    WHERE date BETWEEN date_start AND date_end;
END;
$$ LANGUAGE plpgsql;

-- Optimized trend query using window functions
CREATE OR REPLACE FUNCTION fn_get_metric_trends(metric_name TEXT, date_start DATE, date_end DATE)
RETURNS TABLE (
    date DATE,
    metric_value DECIMAL,
    moving_avg_7d DECIMAL,
    moving_avg_30d DECIMAL,
    trend_direction TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH daily_data AS (
        SELECT
            date,
            CASE metric_name
                WHEN 'tickets' THEN total_tickets::DECIMAL
                WHEN 'csat' THEN avg_csat::DECIMAL
                WHEN 'solved_rate' THEN solved_rate::DECIMAL
                WHEN 'sla_breach' THEN sla_breach_rate::DECIMAL
                ELSE NULL
            END AS value
        FROM mv_daily_metrics
        WHERE date BETWEEN date_start AND date_end
    )
    SELECT
        d.date,
        d.value,
        AVG(d.value) OVER (ORDER BY d.date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS moving_avg_7d,
        AVG(d.value) OVER (ORDER BY d.date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS moving_avg_30d,
        CASE
            WHEN d.value > AVG(d.value) OVER (ORDER BY d.date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
            THEN 'increasing'
            WHEN d.value < AVG(d.value) OVER (ORDER BY d.date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
            THEN 'decreasing'
            ELSE 'stable'
        END AS trend_direction
    FROM daily_data d
    ORDER BY d.date;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PART 4: QUERY PERFORMANCE TIPS
-- ============================================================================

-- For optimal performance, use these query patterns:

-- GOOD: Use date range on indexed column
-- SELECT * FROM cleaned_tickets WHERE DATE(created_at) BETWEEN '2024-01-01' AND '2024-01-31';

-- GOOD: Use materialized views for dashboards
-- SELECT * FROM mv_daily_metrics WHERE date >= CURRENT_DATE - INTERVAL '30 days';

-- GOOD: Use IN for small sets, EXISTS for large sets
-- SELECT * FROM cleaned_tickets t WHERE EXISTS (SELECT 1 FROM priority_customers c WHERE c.ticket_id = t.ticket_id);

-- AVOID: Function on indexed column in WHERE clause
-- SELECT * FROM cleaned_tickets WHERE DATE_TRUNC('month', created_at) = '2024-01-01'; -- SLOW

-- AVOID: SELECT * when you only need specific columns
-- SELECT ticket_id, status, csat FROM cleaned_tickets; -- FASTER

-- ============================================================================
-- PART 5: REFRESH FUNCTIONS FOR MATERIALIZED VIEWS
-- ============================================================================

-- Refresh materialized views (run periodically via scheduler)
CREATE OR REPLACE FUNCTION refresh_analytics_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_agent_performance;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_metrics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_issue_performance;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_hourly_patterns;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PART 6: ANALYZE TABLES FOR QUERY OPTIMIZER
-- ============================================================================

-- Run ANALYZE to update statistics for query planner
ANALYZE cleaned_tickets;
ANALYZE standardized_tickets;
