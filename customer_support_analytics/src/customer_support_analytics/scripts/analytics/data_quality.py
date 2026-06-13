"""
Data Quality Monitoring Framework for SupportIQ

This module provides comprehensive data quality assessment capabilities including:
- Completeness checks (missing values, nulls)
- Consistency validation (duplicates, outliers)
- Accuracy verification (range checks, pattern matching)
- Timeliness assessment (data freshness, gaps)
- Data lineage tracking

Use this module to ensure data integrity before analysis
and provide confidence in analytical results.

Reference: https://vinted.com - Support Agent Tooling team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json

from supportiq_web import path_manager, logger


class DataQualityChecker:
    """
    Data Quality Monitoring Suite for Support Metrics.

    Provides methods for:
    - Completeness checks (missing values, nulls)
    - Consistency validation (duplicates, outliers)
    - Accuracy verification (range checks, patterns)
    - Timeliness assessment (freshness, gaps)
    - Data quality scoring
    """

    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Initialize Data Quality Checker.

        Args:
            thresholds: Dictionary of quality thresholds
                - missing_rate_max: Max allowed missing value rate (default 0.1)
                - duplicate_rate_max: Max allowed duplicate rate (default 0.05)
                - outlier_rate_max: Max allowed outlier rate (default 0.01)
                - freshness_max_hours: Max data age in hours (default 24)
        """
        self.thresholds = thresholds or {
            'missing_rate_max': 0.10,
            'duplicate_rate_max': 0.05,
            'outlier_rate_max': 0.01,
            'freshness_max_hours': 24,
            'completeness_min': 0.90,
            'consistency_min': 0.95,
            'accuracy_min': 0.95,
            'timeliness_min': 0.90
        }

    def check_completeness(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> dict:
        """
        Check data completeness (missing values, nulls).

        Args:
            df: DataFrame to check
            columns: List of columns to check (None = all)

        Returns:
            Dictionary with completeness check results
        """
        if columns is None:
            columns = df.columns.tolist()

        total_rows = len(df)
        results = {
            'column_results': {},
            'overall_missing_rate': 0.0,
            'complete_columns': [],
            'incomplete_columns': [],
            'critical_columns': [],
            'score': 0.0
        }

        total_missing = 0
        total_cells = total_rows * len(columns)

        for col in columns:
            missing_count = df[col].isnull().sum()
            missing_rate = missing_count / total_rows if total_rows > 0 else 1.0
            total_missing += missing_count

            # Identify empty strings and placeholders as missing
            if df[col].dtype == 'object':
                empty_strings = (df[col] == '').sum()
                placeholder_values = df[col].isin(['N/A', 'NA', 'null', 'None', '-', 'Unknown']).sum()
                additional_missing = empty_strings + placeholder_values
                missing_count += additional_missing
                missing_rate = missing_count / total_rows if total_rows > 0 else 1.0
            else:
                additional_missing = 0
                placeholder_values = 0

            col_result = {
                'missing_count': int(missing_count),
                'missing_rate': round(missing_rate, 4),
                'non_missing_count': int(total_rows - missing_count),
                'completeness_rate': round(1 - missing_rate, 4),
                'is_complete': missing_rate <= self.thresholds['missing_rate_max'],
                'empty_strings': int(empty_strings) if df[col].dtype == 'object' else 0,
                'placeholder_count': int(placeholder_values) if df[col].dtype == 'object' else 0
            }

            results['column_results'][col] = col_result

            if missing_rate <= self.thresholds['missing_rate_max']:
                results['complete_columns'].append(col)
            else:
                results['incomplete_columns'].append(col)
                if missing_rate > 0.5:
                    results['critical_columns'].append(col)

        results['overall_missing_rate'] = round(total_missing / total_cells, 4) if total_cells > 0 else 1.0
        results['score'] = round(1 - results['overall_missing_rate'], 4)
        results['pass'] = results['score'] >= self.thresholds['completeness_min']

        logger.info(f"Completeness check: score={results['score']:.2%}, {len(results['incomplete_columns'])} incomplete columns")
        return results

    def check_consistency(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        id_column: str = 'ticket_id'
    ) -> dict:
        """
        Check data consistency (duplicates, outliers).

        Args:
            df: DataFrame to check
            columns: Columns to check for duplicates (None = use id_column)
            id_column: Primary key column for duplicate detection

        Returns:
            Dictionary with consistency check results
        """
        results = {
            'duplicate_rows': 0,
            'duplicate_rate': 0.0,
            'duplicate_ids': [],
            'outlier_results': {},
            'overall_outlier_rate': 0.0,
            'score': 0.0
        }

        total_rows = len(df)

        # Check for duplicate rows
        if id_column and id_column in df.columns:
            duplicates = df[id_column].duplicated()
            duplicate_count = duplicates.sum()
            results['duplicate_rows'] = int(duplicate_count)
            results['duplicate_rate'] = round(duplicate_count / total_rows, 4) if total_rows > 0 else 0
            results['duplicate_ids'] = df[duplicates][id_column].head(100).tolist()

        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        total_outliers = 0
        total_numeric_cells = 0

        for col in numeric_cols:
            outlier_mask = self._detect_outliers_iqr(df[col])
            outlier_count = outlier_mask.sum()
            outlier_rate = outlier_count / total_rows if total_rows > 0 else 0
            total_outliers += outlier_count
            total_numeric_cells += total_rows

            col_outliers = df[col][outlier_mask]
            outlier_values = col_outliers.head(100).tolist()

            results['outlier_results'][col] = {
                'outlier_count': int(outlier_count),
                'outlier_rate': round(outlier_rate, 4),
                'outlier_values': [round(v, 4) for v in outlier_values],
                'q1': round(df[col].quantile(0.25), 4),
                'q3': round(df[col].quantile(0.75), 4),
                'iqr': round(df[col].quantile(0.75) - df[col].quantile(0.25), 4),
                'min': round(df[col].min(), 4),
                'max': round(df[col].max(), 4),
                'is_consistent': outlier_rate <= self.thresholds['outlier_rate_max']
            }

        results['overall_outlier_rate'] = round(total_outliers / total_numeric_cells, 4) if total_numeric_cells > 0 else 0

        # Consistency score
        duplicate_penalty = results['duplicate_rate'] * 0.5
        outlier_penalty = results['overall_outlier_rate'] * 0.5
        results['score'] = round(max(0, 1 - duplicate_penalty - outlier_penalty), 4)
        results['pass'] = results['score'] >= self.thresholds['consistency_min']

        logger.info(f"Consistency check: score={results['score']:.2%}, {results['duplicate_rows']} duplicates, {total_outliers} outliers")
        return results

    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (series < lower_bound) | (series > upper_bound)

    def check_accuracy(
        self,
        df: pd.DataFrame,
        validation_rules: Dict[str, Dict] = None
    ) -> dict:
        """
        Check data accuracy using validation rules.

        Args:
            df: DataFrame to check
            validation_rules: Dictionary of column validation rules
                e.g., {'priority': {'allowed_values': ['Low', 'Medium', 'High', 'Critical']},
                       'csat_score': {'min': 1, 'max': 5}}

        Returns:
            Dictionary with accuracy check results
        """
        if validation_rules is None:
            # Default validation rules based on data types
            validation_rules = self._infer_validation_rules(df)

        results = {
            'column_results': {},
            'total_violations': 0,
            'overall_accuracy': 0.0,
            'score': 0.0
        }

        total_checks = 0
        total_violations = 0

        for col, rules in validation_rules.items():
            if col not in df.columns:
                continue

            violations = 0
            col_checks = 0

            # Range validation
            if 'min' in rules or 'max' in rules:
                min_val = rules.get('min', -np.inf)
                max_val = rules.get('max', np.inf)
                valid_mask = (df[col] >= min_val) & (df[col] <= max_val)
                violations += (~valid_mask).sum()
                col_checks += len(df)

            # Allowed values validation
            if 'allowed_values' in rules:
                valid_mask = df[col].isin(rules['allowed_values'])
                violations += (~valid_mask).sum()
                col_checks += len(df)

            # Pattern validation (regex)
            if 'pattern' in rules:
                import re
                valid_mask = df[col].astype(str).str.match(rules['pattern'])
                violations += (~valid_mask).sum()
                col_checks += len(df)

            # Date format validation
            if 'date_format' in rules:
                try:
                    parsed = pd.to_datetime(df[col], format=rules['date_format'], errors='coerce')
                    violations += parsed.isnull().sum()
                    col_checks += len(df)
                except:
                    pass

            accuracy = 1 - (violations / col_checks) if col_checks > 0 else 0
            total_violations += violations
            total_checks += col_checks

            results['column_results'][col] = {
                'violations': int(violations),
                'checks': int(col_checks),
                'accuracy': round(accuracy, 4),
                'is_accurate': accuracy >= self.thresholds['accuracy_min']
            }

        results['total_violations'] = int(total_violations)
        results['overall_accuracy'] = round(1 - (total_violations / total_checks), 4) if total_checks > 0 else 0
        results['score'] = results['overall_accuracy']
        results['pass'] = results['score'] >= self.thresholds['accuracy_min']

        logger.info(f"Accuracy check: score={results['score']:.2%}, {total_violations} violations")
        return results

    def _infer_validation_rules(self, df: pd.DataFrame) -> Dict:
        """Infer validation rules from data characteristics."""
        rules = {}

        # Priority should be categorical
        if 'priority' in df.columns:
            rules['priority'] = {'allowed_values': ['Low', 'Medium', 'High', 'Critical']}

        # CSAT scores typically 1-5
        if 'csat_score' in df.columns:
            rules['csat_score'] = {'min': 1, 'max': 5}

        # Status should be categorical
        if 'status' in df.columns:
            rules['status'] = {'allowed_values': ['Open', 'Pending', 'Solved', 'Closed']}

        return rules

    def check_timeliness(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'created_at',
        expected_freq: str = 'D'
    ) -> dict:
        """
        Check data timeliness and freshness.

        Args:
            df: DataFrame to check
            timestamp_col: Name of timestamp column
            expected_freq: Expected data frequency ('H'=hourly, 'D'=daily, 'W'=weekly)

        Returns:
            Dictionary with timeliness check results
        """
        results = {
            'freshness_score': 0.0,
            'gap_count': 0,
            'gaps': [],
            'last_update': None,
            'hours_since_update': None,
            'expected_frequency': expected_freq,
            'score': 0.0
        }

        if timestamp_col not in df.columns:
            results['interpretation'] = f"Timestamp column '{timestamp_col}' not found"
            return results

        # Convert to datetime
        timestamps = pd.to_datetime(df[timestamp_col])

        # Calculate freshness
        last_update = timestamps.max()
        now = datetime.now()
        hours_since_update = (now - last_update).total_seconds() / 3600

        results['last_update'] = last_update.isoformat()
        results['hours_since_update'] = round(hours_since_update, 2)
        results['freshness_score'] = max(0, 1 - hours_since_update / self.thresholds['freshness_max_hours'])

        # Check for data gaps
        expected_freq_delta = {
            'H': timedelta(hours=1),
            'D': timedelta(days=1),
            'W': timedelta(weeks=1)
        }.get(expected_freq, timedelta(days=1))

        expected_interval = expected_freq_delta * 1.5  # Allow 50% tolerance
        time_diffs = timestamps.sort_values().diff()

        gap_mask = time_diffs > expected_interval
        gap_indices = gap_mask[gap_mask].index.tolist()

        results['gap_count'] = len(gap_indices)
        results['gaps'] = []

        for idx in gap_indices[:10]:  # Report first 10 gaps
            gap_start = timestamps.loc[idx - 1] if idx > 0 else None
            gap_end = timestamps.loc[idx]
            gap_duration = (gap_end - gap_start).total_seconds() / 3600 if gap_start else 0
            results['gaps'].append({
                'gap_start': gap_start.isoformat() if gap_start else None,
                'gap_end': gap_end.isoformat(),
                'gap_hours': round(gap_duration, 2)
            })

        results['score'] = results['freshness_score']
        results['pass'] = results['score'] >= self.thresholds['timeliness_min']
        results['interpretation'] = f"Data is {results['freshness_score']:.0%} fresh ({hours_since_update:.1f}h old)"

        logger.info(f"Timeliness check: score={results['score']:.2%}, {results['gap_count']} gaps detected")
        return results

    def run_full_quality_assessment(
        self,
        df: pd.DataFrame,
        id_column: str = 'ticket_id',
        timestamp_col: str = 'created_at'
    ) -> dict:
        """
        Run complete data quality assessment.

        Args:
            df: DataFrame to assess
            id_column: Primary key column
            timestamp_col: Timestamp column for timeliness

        Returns:
            Dictionary with complete quality assessment
        """
        logger.info(f"Starting full data quality assessment for {len(df)} rows")

        # Run all checks
        completeness = self.check_completeness(df)
        consistency = self.check_consistency(df, id_column=id_column)
        accuracy = self.check_accuracy(df)
        timeliness = self.check_timeliness(df, timestamp_col=timestamp_col)

        # Calculate overall score
        overall_score = (
            completeness['score'] * 0.30 +
            consistency['score'] * 0.25 +
            accuracy['score'] * 0.30 +
            timeliness['score'] * 0.15
        )

        # Determine overall pass/fail
        all_pass = (
            completeness['pass'] and
            consistency['pass'] and
            accuracy['pass'] and
            timeliness['pass']
        )

        # Identify issues
        issues = []
        if not completeness['pass']:
            issues.append(f"Completeness below threshold: {completeness['score']:.2%}")
        if not consistency['pass']:
            issues.append(f"Consistency below threshold: {consistency['score']:.2%}")
        if not accuracy['pass']:
            issues.append(f"Accuracy below threshold: {accuracy['score']:.2%}")
        if not timeliness['pass']:
            issues.append(f"Timeliness below threshold: {timeliness['score']:.2%}")

        result = {
            'overall_score': round(overall_score, 4),
            'overall_pass': all_pass,
            'completeness': completeness,
            'consistency': consistency,
            'accuracy': accuracy,
            'timeliness': timeliness,
            'issues': issues,
            'recommendations': self._generate_recommendations(completeness, consistency, accuracy, timeliness)
        }

        logger.info(f"Quality assessment complete: score={overall_score:.2%}, pass={all_pass}")
        return result

    def _generate_recommendations(
        self,
        completeness: dict,
        consistency: dict,
        accuracy: dict,
        timeliness: dict
    ) -> List[str]:
        """Generate actionable recommendations based on quality issues."""
        recommendations = []

        # Completeness recommendations
        if completeness['critical_columns']:
            recommendations.append(
                f"Critical: Address missing data in columns: {', '.join(completeness['critical_columns'])}. "
                f"Consider removing or imputing these fields."
            )
        if completeness['incomplete_columns']:
            recommendations.append(
                f"Improve completeness by adding default values or enabling required field validation for: "
                f"{', '.join(completeness['incomplete_columns'][:5])}"
            )

        # Consistency recommendations
        if consistency['duplicate_rows'] > 0:
            recommendations.append(
                f"Remove {consistency['duplicate_rows']} duplicate records to ensure data uniqueness."
            )
        high_outlier_cols = [
            col for col, res in consistency['outlier_results'].items()
            if res['outlier_rate'] > self.thresholds['outlier_rate_max']
        ]
        if high_outlier_cols:
            recommendations.append(
                f"Investigate outliers in columns: {', '.join(high_outlier_cols)}. "
                f"These may indicate data entry errors or genuine anomalies."
            )

        # Accuracy recommendations
        violated_cols = [
            col for col, res in accuracy['column_results'].items()
            if not res['is_accurate']
        ]
        if violated_cols:
            recommendations.append(
                f"Fix accuracy issues in columns: {', '.join(violated_cols)}. "
                f"Review validation rules and data entry processes."
            )

        # Timeliness recommendations
        if not timeliness['pass']:
            recommendations.append(
                f"Data is {timeliness['hours_since_update']:.1f} hours old. "
                f"Implement more frequent data refresh cycles."
            )
        if timeliness['gap_count'] > 0:
            recommendations.append(
                f"Detected {timeliness['gap_count']} data gaps. "
                f"Review data pipeline for interruptions or ingestion issues."
            )

        if not recommendations:
            recommendations.append("Data quality is satisfactory. Continue monitoring for regressions.")

        return recommendations

    def create_quality_report(
        self,
        quality_result: dict,
        format: str = 'dict'
    ) -> Union[dict, str]:
        """
        Create formatted quality report.

        Args:
            quality_result: Result from run_full_quality_assessment
            format: Output format ('dict', 'json', 'markdown')

        Returns:
            Formatted report
        """
        if format == 'dict':
            return quality_result

        elif format == 'json':
            return json.dumps(quality_result, indent=2, default=str)

        elif format == 'markdown':
            lines = [
                "# Data Quality Report",
                "",
                f"**Overall Score**: {quality_result['overall_score']:.2%}",
                f"**Status**: {'✅ PASS' if quality_result['overall_pass'] else '❌ FAIL'}",
                "",
                "## Dimension Scores",
                "",
                f"| Dimension | Score | Status |",
                f"|-----------|-------|--------|",
                f"| Completeness | {quality_result['completeness']['score']:.2%} | {'✅' if quality_result['completeness']['pass'] else '❌'} |",
                f"| Consistency | {quality_result['consistency']['score']:.2%} | {'✅' if quality_result['consistency']['pass'] else '❌'} |",
                f"| Accuracy | {quality_result['accuracy']['score']:.2%} | {'✅' if quality_result['accuracy']['pass'] else '❌'} |",
                f"| Timeliness | {quality_result['timeliness']['score']:.2%} | {'✅' if quality_result['timeliness']['pass'] else '❌'} |",
                "",
                "## Issues"
            ]

            for issue in quality_result['issues']:
                lines.append(f"- {issue}")

            lines.extend(["", "## Recommendations", ""])
            for i, rec in enumerate(quality_result['recommendations'], 1):
                lines.append(f"{i}. {rec}")

            return "\n".join(lines)

        return str(quality_result)


# Convenience function
def assess_data_quality(
    df: pd.DataFrame,
    id_column: str = 'ticket_id',
    timestamp_col: str = 'created_at'
) -> dict:
    """
    Quick function to assess data quality.

    Args:
        df: DataFrame to assess
        id_column: Primary key column
        timestamp_col: Timestamp column

    Returns:
        Quality assessment results
    """
    checker = DataQualityChecker()
    return checker.run_full_quality_assessment(df, id_column, timestamp_col)
