"""
A/B Test Analysis Module for SupportIQ

This module provides comprehensive A/B testing capabilities including:
- Sample size calculation
- Statistical significance testing (t-test, chi-square)
- Confidence intervals
- Power analysis
- Results visualization

Use this module to design, analyze, and interpret experiments
for validating product changes and tooling enhancements.

Reference: https://vinted.com - Support Agent Tooling team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu, norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

from supportiq_web import path_manager, logger, log_pipeline_start, log_pipeline_complete


class ABTestAnalyzer:
    """
    A/B Test Analyzer for Support Metrics.
    
    Provides statistical methods for experiment analysis including:
    - Sample size estimation
    - Effect size calculation
    - Statistical significance testing
    - Confidence interval estimation
    - Power analysis
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.80):
        """
        Initialize A/B Test Analyzer.
        
        Args:
            alpha: Significance level (default 0.05 = 95% confidence)
            power: Statistical power (default 0.80 = 80% chance of detecting true effect)
        """
        self.alpha = alpha
        self.power = power
        self.z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed
        self.z_beta = norm.ppf(power)
    
    def calculate_sample_size(
        self, 
        baseline_rate: float, 
        minimum_detectable_effect: float,
        control_size: float = None,
        variant_size: float = None
    ) -> dict:
        """
        Calculate required sample size for A/B test.
        
        Args:
            baseline_rate: Baseline conversion rate (e.g., 0.10 for 10%)
            minimum_detectable_effect: Minimum relative effect to detect (e.g., 0.05 for 5% lift)
            control_size: Optional fixed control group size
            variant_size: Optional fixed variant group size
            
        Returns:
            Dictionary with sample size calculations
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        # Pooled proportion under null hypothesis
        p_pooled = (p1 + p2) / 2
        
        # Standard error components
        se_null = np.sqrt(2 * p_pooled * (1 - p_pooled))
        se_alt = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        
        # Required sample size per group (for equal groups)
        n_per_group = ((self.z_alpha * se_null + self.z_beta * se_alt) / (p2 - p1)) ** 2
        
        total_sample = int(np.ceil(2 * n_per_group))
        
        result = {
            'baseline_rate': p1,
            'target_rate': p2,
            'minimum_detectable_effect': minimum_detectable_effect,
            'relative_lift': f"{(p2/p1 - 1)*100:.1f}%",
            'n_per_group': int(np.ceil(n_per_group)),
            'total_sample_needed': total_sample,
            'duration_days_estimate': self._estimate_duration(total_sample, baseline_rate)
        }
        
        logger.info(f"Sample size calculation: Need {total_sample} total samples")
        return result
    
    def _estimate_duration(self, sample_size: int, daily_rate: float) -> int:
        """Estimate test duration in days."""
        daily_visitors = sample_size / 30  # Assume 30-day test
        return int(np.ceil(sample_size / max(1, daily_visitors)))
    
    def run_ttest(
        self, 
        control: np.ndarray, 
        variant: np.ndarray,
        equal_var: bool = False
    ) -> dict:
        """
        Perform independent samples t-test.
        
        Args:
            control: Control group data
            variant: Variant group data
            equal_var: Use Welch's t-test if False (unequal variances)
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        control = control[~np.isnan(control)]
        variant = variant[~np.isnan(variant)]
        
        n_control = len(control)
        n_variant = len(variant)
        
        # Calculate statistics
        mean_control = np.mean(control)
        mean_variant = np.mean(variant)
        std_control = np.std(control, ddof=1)
        std_variant = np.std(variant, ddof=1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n_control - 1) * std_control**2 + (n_variant - 1) * std_variant**2) 
                            / (n_control + n_variant - 2))
        cohens_d = (mean_variant - mean_control) / pooled_std if pooled_std > 0 else 0
        
        # Perform t-test
        t_stat, p_value = ttest_ind(control, variant, equal_var=equal_var)
        
        # Welch-Satterthwaite degrees of freedom
        if not equal_var:
            df = ((std_control**2/n_control + std_variant**2/n_variant)**2 / 
                  ((std_control**2/n_control)**2/(n_control-1) + 
                   (std_variant**2/n_variant)**2/(n_variant-1)))
        else:
            df = n_control + n_variant - 2
        
        # Confidence interval for difference in means
        se_diff = np.sqrt(std_control**2/n_control + std_variant**2/n_variant)
        ci_lower = (mean_variant - mean_control) - self.z_alpha * se_diff
        ci_upper = (mean_variant - mean_control) + self.z_alpha * se_diff
        
        # Relative lift
        relative_lift = (mean_variant - mean_control) / mean_control * 100 if mean_control != 0 else 0
        
        result = {
            'test_type': 'Independent Samples t-test' + (' (Welch)' if not equal_var else ''),
            'n_control': n_control,
            'n_variant': n_variant,
            'mean_control': round(mean_control, 4),
            'mean_variant': round(mean_variant, 4),
            'std_control': round(std_control, 4),
            'std_variant': round(std_variant, 4),
            'mean_difference': round(mean_variant - mean_control, 4),
            'relative_lift_pct': round(relative_lift, 2),
            't_statistic': round(t_stat, 4),
            'degrees_of_freedom': round(df, 2),
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'cohens_d': round(cohens_d, 4),
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
            'ci_95_lower': round(ci_lower, 4),
            'ci_95_upper': round(ci_upper, 4),
            'confidence_level': f"{(1-self.alpha)*100}%"
        }
        
        logger.info(f"t-test result: p={p_value:.4f}, significant={p_value < self.alpha}")
        return result
    
    def run_mann_whitney(self, control: np.ndarray, variant: np.ndarray) -> dict:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Use when data is not normally distributed.
        """
        control = control[~np.isnan(control)]
        variant = variant[~np.isnan(variant)]
        
        statistic, p_value = mannwhitneyu(control, variant, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(control), len(variant)
        r = 1 - (2 * statistic) / (n1 * n2)
        
        result = {
            'test_type': 'Mann-Whitney U Test',
            'n_control': n1,
            'n_variant': n2,
            'u_statistic': statistic,
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'rank_biserial_r': round(r, 4),
            'effect_size_interpretation': 'Small' if abs(r) < 0.3 else ('Medium' if abs(r) < 0.5 else 'Large')
        }
        
        logger.info(f"Mann-Whitney result: p={p_value:.4f}")
        return result
    
    def run_chi_square(self, control_success: int, control_total: int,
                       variant_success: int, variant_total: int) -> dict:
        """
        Perform chi-square test for proportions.
        
        Use for comparing conversion rates, CSAT rates, SLA breach rates.
        """
        # Create contingency table
        observed = np.array([
            [control_success, control_total - control_success],
            [variant_success, variant_total - variant_success]
        ])
        
        chi2, p_value, dof, expected = chi2_contingency(observed)
        
        # Proportions
        p_control = control_success / control_total if control_total > 0 else 0
        p_variant = variant_success / variant_total if variant_total > 0 else 0
        
        # Relative lift
        relative_lift = (p_variant - p_control) / p_control * 100 if p_control > 0 else 0
        
        # Pooled proportion
        p_pooled = (control_success + variant_success) / (control_total + variant_total)
        
        # Standard error for difference in proportions
        se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/variant_total))
        
        # Confidence interval
        ci_lower = (p_variant - p_control) - self.z_alpha * se_diff
        ci_upper = (p_variant - p_control) + self.z_alpha * se_diff
        
        # Cramér's V (effect size)
        n = control_total + variant_total
        cramers_v = np.sqrt(chi2 / (n * (min(observed.shape) - 1))) if n > 0 else 0
        
        result = {
            'test_type': 'Chi-Square Test',
            'control_conversion_rate': round(p_control * 100, 2),
            'variant_conversion_rate': round(p_variant * 100, 2),
            'absolute_difference': round((p_variant - p_control) * 100, 2),
            'relative_lift_pct': round(relative_lift, 2),
            'chi_square_statistic': round(chi2, 4),
            'degrees_of_freedom': dof,
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'cramers_v': round(cramers_v, 4),
            'ci_95_lower': round(ci_lower * 100, 2),
            'ci_95_upper': round(ci_upper * 100, 2),
            'contingency_table': observed.tolist()
        }
        
        logger.info(f"Chi-square result: χ²={chi2:.2f}, p={p_value:.4f}")
        return result
    
    def calculate_confidence_interval(
        self, 
        data: np.ndarray, 
        metric_type: str = 'mean'
    ) -> dict:
        """
        Calculate confidence interval for a metric.
        
        Args:
            data: Array of metric values
            metric_type: 'mean', 'proportion', or 'median'
        """
        data = data[~np.isnan(data)]
        n = len(data)
        
        if metric_type == 'mean':
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            se = std / np.sqrt(n)
            
            # t-distribution for small samples
            if n < 30:
                t_val = stats.t.ppf(1 - self.alpha/2, n-1)
            else:
                t_val = self.z_alpha
            
            ci_lower = mean - t_val * se
            ci_upper = mean + t_val * se
            
            return {
                'metric_type': 'mean',
                'point_estimate': round(mean, 4),
                'std_error': round(se, 4),
                'ci_95_lower': round(ci_lower, 4),
                'ci_95_upper': round(ci_upper, 4),
                'confidence_level': f"{(1-self.alpha)*100}%",
                'sample_size': n
            }
        
        elif metric_type == 'proportion':
            p = np.sum(data) / n
            se = np.sqrt(p * (1 - p) / n)
            
            ci_lower = p - self.z_alpha * se
            ci_upper = p + self.z_alpha * se
            
            return {
                'metric_type': 'proportion',
                'point_estimate': round(p * 100, 2),
                'std_error': round(se * 100, 2),
                'ci_95_lower': round(max(0, ci_lower) * 100, 2),
                'ci_95_upper': round(min(1, ci_upper) * 100, 2),
                'confidence_level': f"{(1-self.alpha)*100}%",
                'sample_size': n
            }
        
        elif metric_type == 'median':
            sorted_data = np.sort(data)
            # Bootstrap CI for median
            n_bootstrap = 1000
            medians = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=n, replace=True)
                medians.append(np.median(sample))
            
            ci_lower = np.percentile(medians, (1-self.alpha/2)*100)
            ci_upper = np.percentile(medians, (self.alpha/2)*100)
            
            return {
                'metric_type': 'median',
                'point_estimate': round(np.median(data), 4),
                'ci_95_lower': round(ci_lower, 4),
                'ci_95_upper': round(ci_upper, 4),
                'confidence_level': f"{(1-self.alpha)*100}%",
                'sample_size': n,
                'method': 'Bootstrap'
            }
    
    def power_analysis(
        self,
        baseline_rate: float,
        effect_size: float,
        sample_size: int
    ) -> dict:
        """
        Calculate achieved statistical power.
        
        Args:
            baseline_rate: Baseline conversion rate
            effect_size: Expected effect (absolute difference)
            sample_size: Available sample size per group
        """
        p1 = baseline_rate
        p2 = p1 + effect_size
        
        # Calculate effect size (Cohen's h for proportions)
        import math
        h = 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))
        
        # Standard error under alternative
        se_alt = np.sqrt(p1*(1-p1)/sample_size + p2*(1-p2)/sample_size)
        
        # Non-centrality parameter
        delta = abs(p2 - p1) / se_alt
        
        # Achieved power
        achieved_power = 1 - norm.cdf(self.z_alpha - delta)
        
        # Sample size needed for target power
        n_needed = ((self.z_alpha + self.z_beta) ** 2 * (p1*(1-p1) + p2*(1-p2))) / effect_size**2
        
        return {
            'baseline_rate': p1,
            'target_rate': p2,
            'effect_size': effect_size,
            'sample_size_available': sample_size,
            'achieved_power': round(achieved_power * 100, 1),
            'power_interpretation': 'Adequate' if achieved_power >= 0.80 else 'Underpowered',
            'sample_size_needed_for_80_power': int(np.ceil(n_needed)),
            'recommendation': 'Continue test' if achieved_power < 0.80 else 'Test adequately powered'
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'Negligible'
        elif abs_d < 0.5:
            return 'Small'
        elif abs_d < 0.8:
            return 'Medium'
        else:
            return 'Large'


def generate_ab_test_data(
    n_control: int = 5000,
    n_variant: int = 5000,
    baseline_csat: float = 4.2,
    variant_csat: float = 4.4,
    baseline_resolution: int = 240,
    variant_resolution: int = 220
) -> pd.DataFrame:
    """
    Generate simulated A/B test data for demonstration.
    
    Args:
        n_control: Control group size
        n_variant: Variant group size
        baseline_csat: Control group CSAT mean
        variant_csat: Variant group CSAT mean
        baseline_resolution: Control resolution time (min)
        variant_resolution: Variant resolution time (min)
    """
    np.random.seed(42)
    
    control = pd.DataFrame({
        'experiment_group': 'control',
        'ticket_id': [f'CTRL-{i:05d}' for i in range(1, n_control + 1)],
        'csat_score': np.random.normal(baseline_csat, 0.8, n_control).clip(1, 5),
        'resolution_time_min': np.random.normal(baseline_resolution, 60, n_control).clip(10, 600),
        'first_response_min': np.random.normal(30, 15, n_control).clip(1, 120),
        'solved': np.random.binomial(1, 0.75, n_control),
        'sla_breach': np.random.binomial(1, 0.12, n_control),
        'escalated': np.random.binomial(1, 0.05, n_control),
    })
    
    variant = pd.DataFrame({
        'experiment_group': 'variant',
        'ticket_id': [f'VARI-{i:05d}' for i in range(1, n_variant + 1)],
        'csat_score': np.random.normal(variant_csat, 0.8, n_variant).clip(1, 5),
        'resolution_time_min': np.random.normal(variant_resolution, 60, n_variant).clip(10, 600),
        'first_response_min': np.random.normal(28, 15, n_variant).clip(1, 120),
        'solved': np.random.binomial(1, 0.78, n_variant),
        'sla_breach': np.random.binomial(1, 0.10, n_variant),
        'escalated': np.random.binomial(1, 0.04, n_variant),
    })
    
    df = pd.concat([control, variant], ignore_index=True)
    df['created_date'] = pd.date_range(start='2024-02-01', periods=len(df), freq='10min')
    
    return df


def create_results_visualization(results: dict, metric_name: str) -> go.Figure:
    """
    Create visualization for A/B test results.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Control vs Variant Distribution',
            'Confidence Interval',
            'Effect Size',
            'Statistical Power'
        ),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Distribution visualization (placeholder - would need raw data)
    fig.add_trace(
        go.Bar(x=['Control', 'Variant'], 
               y=[results['mean_control'], results['mean_variant']],
               name='Mean',
               marker_color=['#1f77b4', '#2ca02c']),
        row=1, col=1
    )
    
    # CI visualization
    fig.add_trace(
        go.Bar(x=['Difference'],
               y=[results['mean_difference']],
               error_y=dict(
                   type='data',
                   array=[[results['ci_95_upper'] - results['mean_difference']]],
                   arrayminus=[[results['mean_difference'] - results['ci_95_lower']]],
                   color='#E74C3C'
               ),
               marker_color='#3498DB',
               name='Effect'),
        row=1, col=2
    )
    
    # Effect size indicator
    fig.add_trace(
        go.Indicator(
            mode='number',
            value=results['cohens_d'],
            title={'text': "Cohen's d"},
            domain={'row': 0, 'column': 0}
        ),
        row=2, col=1
    )
    
    # P-value indicator
    fig.update_layout(
        title=f'A/B Test Results: {metric_name}',
        height=600,
        showlegend=False
    )
    
    return fig


def main():
    """Run A/B test analysis demonstration."""
    import time
    start_time = time.time()
    
    log_pipeline_start("A/B Test Analysis")
    
    # Initialize analyzer
    analyzer = ABTestAnalyzer(alpha=0.05, power=0.80)
    
    # Generate sample data
    logger.info("Generating sample A/B test data...")
    df = generate_ab_test_data()
    
    # Save test data
    output_path = path_manager.get_processed_data_path('ab_test_data.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved test data to: {output_path}")
    
    # Split by group
    control = df[df['experiment_group'] == 'control']
    variant = df[df['experiment_group'] == 'variant']
    
    # Analysis 1: CSAT Scores (t-test)
    logger.info("\n=== Analysis 1: CSAT Score Comparison ===")
    csat_results = analyzer.run_ttest(
        control['csat_score'].values,
        variant['csat_score'].values
    )
    logger.info(f"CSAT - Control: {csat_results['mean_control']}, Variant: {csat_results['mean_variant']}")
    logger.info(f"CSAT - Relative lift: {csat_results['relative_lift_pct']}%")
    logger.info(f"CSAT - Significant: {csat_results['significant']}")
    
    # Analysis 2: Resolution Time (t-test)
    logger.info("\n=== Analysis 2: Resolution Time Comparison ===")
    resolution_results = analyzer.run_ttest(
        control['resolution_time_min'].values,
        variant['resolution_time_min'].values
    )
    logger.info(f"Resolution - Control: {resolution_results['mean_control']} min, Variant: {resolution_results['mean_variant']} min")
    logger.info(f"Resolution - Time saved: {abs(resolution_results['mean_difference']):.1f} min")
    
    # Analysis 3: SLA Breach Rate (chi-square)
    logger.info("\n=== Analysis 3: SLA Breach Rate Comparison ===")
    breach_results = analyzer.run_chi_square(
        control_success=len(control) - control['sla_breach'].sum(),
        control_total=len(control),
        variant_success=len(variant) - variant['sla_breach'].sum(),
        variant_total=len(variant)
    )
    logger.info(f"SLA Breach - Control: {breach_results['control_conversion_rate']}%, Variant: {breach_results['variant_conversion_rate']}%")
    logger.info(f"SLA Breach - Improvement: {breach_results['relative_lift_pct']}%")
    
    # Analysis 4: Sample Size Calculation
    logger.info("\n=== Analysis 4: Sample Size Calculation ===")
    sample_calc = analyzer.calculate_sample_size(
        baseline_rate=0.75,  # 75% baseline CSAT rate
        minimum_detectable_effect=0.05  # Detect 5% relative improvement
    )
    logger.info(f"Sample size needed: {sample_calc['total_sample_needed']}")
    logger.info(f"Estimated duration: {sample_calc['duration_days_estimate']} days")
    
    # Analysis 5: Confidence Intervals
    logger.info("\n=== Analysis 5: Confidence Intervals ===")
    ci_csat = analyzer.calculate_confidence_interval(
        variant['csat_score'].values,
        metric_type='mean'
    )
    logger.info(f"Variant CSAT CI: [{ci_csat['ci_95_lower']:.2f}, {ci_csat['ci_95_upper']:.2f}]")
    
    # Save results
    results_summary = {
        'analysis_date': datetime.now().isoformat(),
        'sample_size': {
            'control': len(control),
            'variant': len(variant),
            'total': len(df)
        },
        'csat_analysis': csat_results,
        'resolution_analysis': resolution_results,
        'sla_breach_analysis': breach_results,
        'sample_size_calculation': sample_calc
    }
    
    results_path = path_manager.get_processed_data_path('ab_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Saved analysis results to: {results_path}")
    
    duration = time.time() - start_time
    log_pipeline_complete("A/B Test Analysis", duration)
    
    return results_summary


if __name__ == "__main__":
    results = main()
