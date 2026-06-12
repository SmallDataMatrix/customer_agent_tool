-- Standardize issue taxonomy for consistent analysis
-- This script creates a standardized issue group taxonomy

-- Create issue taxonomy lookup table
CREATE TABLE IF NOT EXISTS issue_taxonomy (
    original_issue_group TEXT,
    standardized_issue_group TEXT,
    issue_category TEXT,
    risk_level TEXT,
    requires_human_review BOOLEAN
);

-- Populate issue taxonomy
INSERT INTO issue_taxonomy VALUES
    ('Shipping / Delivery', 'Shipping / Delivery', 'Operations', 'Medium', FALSE),
    ('Payment / Refund', 'Payment / Refund', 'Financial', 'High', TRUE),
    ('Account Access', 'Account Access', 'Security', 'High', TRUE),
    ('Returns / Disputes', 'Returns / Disputes', 'Operations', 'Medium', FALSE),
    ('Technical Issues', 'Technical Issues', 'Technical', 'Medium', FALSE),
    ('Product Questions', 'Product Questions', 'Information', 'Low', FALSE),
    ('Verification', 'Verification', 'Security', 'High', TRUE),
    ('Fraud / Safety', 'Fraud / Safety', 'Security', 'Critical', TRUE);

-- Create standardized tickets view
CREATE VIEW IF NOT EXISTS standardized_tickets AS
SELECT
    t.*,
    it.issue_category,
    it.risk_level,
    it.requires_human_review
FROM cleaned_tickets t
LEFT JOIN issue_taxonomy it 
    ON t.issue_group_standardized = it.standardized_issue_group;
