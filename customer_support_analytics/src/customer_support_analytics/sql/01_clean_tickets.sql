-- Clean and standardize ticket data
-- This script prepares the raw ticket data for analysis

-- Create cleaned tickets table
CREATE TABLE IF NOT EXISTS cleaned_tickets AS
SELECT
    ticket_id,
    TRIM(subject) AS subject,
    TRIM(body) AS body,
    issue_group,
    priority,
    channel,
    language,
    status,
    created_at,
    csat,
    -- Derived fields
    LENGTH(body) AS text_length,
    CASE 
        WHEN DAYOFWEEK(created_at) IN (1, 7) THEN 1 
        ELSE 0 
    END AS is_weekend,
    -- Standardize priority
    CASE priority
        WHEN 'Critical' THEN 'Critical'
        WHEN 'High' THEN 'High'
        WHEN 'Medium' THEN 'Medium'
        WHEN 'Low' THEN 'Low'
        ELSE 'Unknown'
    END AS priority_standardized,
    -- Issue group taxonomy
    CASE issue_group
        WHEN 'Shipping' THEN 'Shipping / Delivery'
        WHEN 'Delivery' THEN 'Shipping / Delivery'
        WHEN 'Payment' THEN 'Payment / Refund'
        WHEN 'Refund' THEN 'Payment / Refund'
        WHEN 'Account' THEN 'Account Access'
        WHEN 'Returns' THEN 'Returns / Disputes'
        WHEN 'Dispute' THEN 'Returns / Disputes'
        WHEN 'Technical' THEN 'Technical Issues'
        WHEN 'Product' THEN 'Product Questions'
        WHEN 'Verification' THEN 'Verification'
        WHEN 'Fraud' THEN 'Fraud / Safety'
        ELSE issue_group
    END AS issue_group_standardized
FROM raw_tickets
WHERE ticket_id IS NOT NULL
    AND created_at IS NOT NULL;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_created_at ON cleaned_tickets(created_at);
CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_issue_group ON cleaned_tickets(issue_group_standardized);
CREATE INDEX IF NOT EXISTS idx_cleaned_tickets_priority ON cleaned_tickets(priority_standardized);
