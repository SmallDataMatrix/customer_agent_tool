"""
Statistical Tests Module for SupportIQ

This module provides comprehensive statistical testing capabilities including:
- T-tests (independent, paired, one-sample)
- Chi-square tests (goodness of fit, independence)
- ANOVA (one-way, two-way)
- Correlation analysis (Pearson, Spearman)
- Non-parametric tests (Wilcoxon, Kruskal-Wallis)
- Effect size calculations
- Normality tests

Reference: https://vinted.com - Support Agent Tooling team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, ttest_1samp,
    chi2_contingency, chi2,
    f_oneway, kruskal,
    pearsonr, spearmanr,
    shapiro, normaltest, levene,
    mannwhitneyu, wilcoxon,
    ks_2samp
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json

from supportiq_web import path_manager, logger, log_pipeline_start, log_pipeline_complete


class StatisticalAnalyzer:
    """
    Comprehensive Statistical Analysis Suite.
    
    Provides methods for:
    - Descriptive statistics
    - Hypothesis testing
    - Correlation analysis
    - Effect size estimation
    - Normality assessment
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize Statistical Analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
    
    def descriptive_stats(self, data: np.ndarray) -> dict:
        """
        Calculate comprehensive descriptive statistics.
        """
        data = data[~np.isnan(data)]
        n = len(data)
        
        result = {
            'n': n,
            'mean': round(np.mean(data), 4),
            'median': round(np.median(data), 4),
            'mode': round(stats.mode(data, keepdims=True).mode[0], 4),
            'std': round(np.std(data, ddof=1), 4),
            'variance': round(np.var(data, ddof=1), 4),
            'min': round(np.min(data), 4),
            'max': round(np.max(data), 4),
            'range': round(np.max(data) - np.min(data), 4),
            'q1': round(np.percentile(data, 25), 4),
            'q3': round(np.percentile(data, 75), 4),
            'iqr': round(np.percentile(data, 75) - np.percentile(data, 25), 4),
            'skewness': round(stats.skew(data), 4),
            'kurtosis': round(stats.kurtosis(data), 4),
            'cv': round(np.std(data, ddof=1) / np.mean(data) * 100, 2) if np.mean(data) != 0 else None,
            'sem': round(stats.sem(data), 4),  # Standard error of mean
        }
        
        # Percentiles
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            result[f'p{p}'] = round(np.percentile(data, p), 4)
        
        return result
    
    def normality_tests(self, data: np.ndarray) -> dict:
        """
        Test for normality using multiple methods.
        """
        data = data[~np.isnan(data)]
        n = len(data)
        
        results = {'sample_size': n}
        
        # Shapiro-Wilk test (best for n < 5000)
        if n >= 3 and n <= 5000:
            stat_shapiro, p_shapiro = shapiro(data)
            results['shapiro_wilk'] = {
                'statistic': round(stat_shapiro, 4),
                'p_value': p_shapiro,
                'is_normal': p_shapiro > self.alpha
            }
        
        # D'Agostino-Pearson test (best for n >= 20)
        if n >= 20:
            stat_dagostino, p_dagostino = normaltest(data)
            results['dagostino_pearson'] = {
                'statistic': round(stat_dagostino, 4),
                'p_value': p_dagostino,
                'is_normal': p_dagostino > self.alpha
            }
        
        # Kolmogorov-Smirnov test
        stat_ks, p_ks = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        results['kolmogorov_smirnov'] = {
            'statistic': round(stat_ks, 4),
            'p_value': p_ks,
            'is_normal': p_ks > self.alpha
        }
        
        # Overall assessment
        all_tests = [r.get('is_normal', True) for r in results.values() if isinstance(r, dict)]
        results['is_normal'] = sum(all_tests) / len(all_tests) >= 0.5 if all_tests else True
        results['recommendation'] = (
            'Data appears normally distributed' if results['is_normal'] 
            else 'Data deviates from normality - consider non-parametric tests'
        )
        
        return results
    
    def ttest_independent(self, group1: np.ndarray, group2: np.ndarray) -> dict:
        """
        Independent samples t-test (Welch's by default).
        """
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Welch's t-test (does not assume equal variances)
        stat, p_value = ttest_ind(group1, group2, equal_var=False)
        
        # Degrees of freedom (Welch-Satterthwaite)
        df = ((std1**2/n1 + std2**2/n2)**2 / 
              ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1)))
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
        
        # 95% CI for difference
        se_diff = np.sqrt(std1**2/n1 + std2**2/n2)
        ci_lower = (mean2 - mean1) - 1.96 * se_diff
        ci_upper = (mean2 - mean1) + 1.96 * se_diff
        
        return {
            'test': 'Independent Samples t-test (Welch)',
            'n1': n1, 'n2': n2,
            'mean1': round(mean1, 4), 'mean2': round(mean2, 4),
            'std1': round(std1, 4), 'std2': round(std2, 4),
            'difference': round(mean2 - mean1, 4),
            't_statistic': round(stat, 4),
            'df': round(df, 2),
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'cohens_d': round(cohens_d, 4),
            'effect_interpretation': self._cohens_interpretation(cohens_d),
            'ci_95': [round(ci_lower, 4), round(ci_upper, 4)],
            'assumption_check': {
                'levene_test': self._levene_test(group1, group2)
            }
        }
    
    def ttest_paired(self, before: np.ndarray, after: np.ndarray) -> dict:
        """
        Paired samples t-test (for pre/post comparisons).
        """
        # Remove pairs with any NaN
        mask = ~(np.isnan(before) | np.isnan(after))
        before = before[mask]
        after = after[mask]
        
        n = len(before)
        diff = after - before
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        stat, p_value = ttest_rel(before, after)
        
        # Cohen's d for paired samples
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # 95% CI
        se_diff = std_diff / np.sqrt(n)
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        return {
            'test': 'Paired Samples t-test',
            'n_pairs': n,
            'mean_before': round(np.mean(before), 4),
            'mean_after': round(np.mean(after), 4),
            'mean_difference': round(mean_diff, 4),
            'std_difference': round(std_diff, 4),
            't_statistic': round(stat, 4),
            'df': n - 1,
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'cohens_d': round(cohens_d, 4),
            'effect_interpretation': self._cohens_interpretation(cohens_d),
            'ci_95': [round(ci_lower, 4), round(ci_upper, 4)],
            'percent_change': round((mean_diff / np.mean(before)) * 100, 2) if np.mean(before) != 0 else None
        }
    
    def oneway_anova(self, *groups) -> dict:
        """
        One-way ANOVA for comparing multiple groups.
        """
        groups_clean = [g[~np.isnan(g)] for g in groups]
        n_groups = len(groups_clean)
        all_data = np.concatenate(groups_clean)
        
        # Calculate group statistics
        group_stats = []
        for i, g in enumerate(groups_clean):
            group_stats.append({
                'group': f'Group {i+1}',
                'n': len(g),
                'mean': round(np.mean(g), 4),
                'std': round(np.std(g, ddof=1), 4)
            })
        
        # ANOVA
        stat, p_value = f_oneway(*groups_clean)
        
        # Effect size (eta-squared)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups_clean)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Omega-squared (less biased)
        k = n_groups
        n = len(all_data)
        ms_within = (ss_total - ss_between) / (n - k)
        omega_squared = (ss_between - (k-1)*ms_within) / (ss_total + ms_within) if (ss_total + ms_within) != 0 else 0
        
        # Degrees of freedom
        df_between = k - 1
        df_within = n - k
        df_total = n - 1
        
        # ANOVA table
        ss_within = ss_total - ss_between
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_stat = ms_between / ms_within if ms_within > 0 else 0
        
        return {
            'test': 'One-Way ANOVA',
            'n_groups': n_groups,
            'total_n': len(all_data),
            'group_statistics': group_stats,
            'f_statistic': round(f_stat, 4),
            'df_between': df_between,
            'df_within': df_within,
            'df_total': df_total,
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'eta_squared': round(eta_squared, 4),
            'omega_squared': round(max(0, omega_squared), 4),
            'effect_interpretation': self._eta_interpretation(eta_squared),
            'anova_table': {
                'Source': ['Between Groups', 'Within Groups', 'Total'],
                'SS': [round(ss_between, 2), round(ss_within, 2), round(ss_total, 2)],
                'df': [df_between, df_within, df_total],
                'MS': [round(ms_between, 2), round(ms_within, 2), ''],
                'F': [round(f_stat, 2), '', '']
            }
        }
    
    def kruskal_wallis(self, *groups) -> dict:
        """
        Kruskal-Wallis H test (non-parametric alternative to ANOVA).
        """
        groups_clean = [g[~np.isnan(g)] for g in groups]
        n_groups = len(groups_clean)
        all_data = np.concatenate(groups_clean)
        n_total = len(all_data)
        
        stat, p_value = kruskal(*groups_clean)
        
        # Effect size (epsilon-squared)
        h_stat = stat
        k = n_groups
        n = n_total
        epsilon_squared = (h_stat - k + 1) / (n - k) if (n - k) > 0 else 0
        
        return {
            'test': 'Kruskal-Wallis H Test',
            'n_groups': k,
            'total_n': n,
            'h_statistic': round(h_stat, 4),
            'df': k - 1,
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'epsilon_squared': round(max(0, epsilon_squared), 4),
            'effect_interpretation': 'Small' if epsilon_squared < 0.06 else ('Medium' if epsilon_squared < 0.14 else 'Large'),
            'note': 'Non-parametric alternative to ANOVA - does not assume normality'
        }
    
    def chi_square_test(self, observed: np.ndarray, expected: np.ndarray = None) -> dict:
        """
        Chi-square goodness of fit test.
        """
        if expected is None:
            expected = np.full_like(observed, np.mean(observed), dtype=float)
        
        # Normalize to expected proportions
        expected_props = expected / np.sum(expected)
        expected_counts = expected_props * np.sum(observed)
        
        chi2_stat = np.sum((observed - expected_counts)**2 / expected_counts)
        df = len(observed) - 1
        p_value = 1 - chi2.cdf(chi2_stat, df)
        
        # Cramér's V (effect size)
        n = np.sum(observed)
        min_dim = min(len(observed), 1) - 1
        cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if (n * min_dim) > 0 else 0
        
        return {
            'test': 'Chi-Square Goodness of Fit',
            'observed': observed.tolist(),
            'expected': expected_counts.tolist(),
            'chi_square': round(chi2_stat, 4),
            'df': df,
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'cramers_v': round(cramers_v, 4),
            'residuals': ((observed - expected_counts) / np.sqrt(expected_counts)).tolist()
        }
    
    def chi_square_independence(self, contingency_table: np.ndarray) -> dict:
        """
        Chi-square test of independence (for categorical variables).
        """
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Cramér's V
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
        cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if (n * min_dim) > 0 else 0
        
        # Phi coefficient (for 2x2)
        if contingency_table.shape == (2, 2):
            phi = chi2_stat / n if n > 0 else 0
        else:
            phi = None
        
        return {
            'test': 'Chi-Square Test of Independence',
            'contingency_table': contingency_table.tolist(),
            'chi_square': round(chi2_stat, 4),
            'df': dof,
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'cramers_v': round(cramers_v, 4),
            'phi_coefficient': round(phi, 4) if phi is not None else None,
            'expected_frequencies': expected.tolist(),
            'effect_interpretation': self._cramers_interpretation(cramers_v)
        }
    
    def correlation_analysis(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Comprehensive correlation analysis (Pearson + Spearman).
        """
        # Remove NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        n = len(x)
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(x, y)
        
        # Spearman correlation (rank-based, robust to outliers)
        spearman_r, spearman_p = spearmanr(x, y)
        
        # Coefficient of determination
        r_squared = pearson_r ** 2
        
        # Confidence interval for Pearson r (Fisher z transformation)
        z = np.arctanh(pearson_r)
        se_z = 1 / np.sqrt(n - 3)
        z_lower = z - 1.96 * se_z
        z_upper = z + 1.96 * se_z
        r_ci_lower = np.tanh(z_lower)
        r_ci_upper = np.tanh(z_upper)
        
        return {
            'n': n,
            'pearson': {
                'r': round(pearson_r, 4),
                'r_squared': round(r_squared, 4),
                'p_value': pearson_p,
                'p_value_formatted': f"{'{:e}'.format(pearson_p)}",
                'significant': pearson_p < self.alpha,
                'ci_95': [round(r_ci_lower, 4), round(r_ci_upper, 4)]
            },
            'spearman': {
                'rho': round(spearman_r, 4),
                'p_value': spearman_p,
                'p_value_formatted': f"{'{:e}'.format(spearman_p)}",
                'significant': spearman_p < self.alpha
            },
            'interpretation': self._correlation_interpretation(pearson_r),
            'recommendation': (
                'Use Pearson (linear relationship)' if abs(pearson_r - spearman_r) < 0.1 
                else 'Consider Spearman (monotonic relationship)'
            )
        }
    
    def mann_whitney_test(self, group1: np.ndarray, group2: np.ndarray) -> dict:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).
        """
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        n1, n2 = len(group1), len(group2)
        
        # Effect size (rank-biserial correlation)
        r = 1 - (2 * stat) / (n1 * n2)
        
        # Common Language Effect Size (probability of superiority)
        cles = stat / (n1 * n2) if (n1 * n2) > 0 else 0.5
        
        return {
            'test': 'Mann-Whitney U Test',
            'n1': n1, 'n2': n2,
            'u_statistic': stat,
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'rank_biserial_r': round(r, 4),
            'effect_interpretation': self._cohens_interpretation(abs(r)),
            'common_language_effect_size': round(cles, 4),
            'note': 'Non-parametric - does not assume normality'
        }
    
    def wilcoxon_test(self, before: np.ndarray, after: np.ndarray) -> dict:
        """
        Wilcoxon signed-rank test (non-parametric paired test).
        """
        mask = ~(np.isnan(before) | np.isnan(after))
        before, after = before[mask], after[mask]
        
        stat, p_value = wilcoxon(before, after)
        
        n = len(before)
        
        # Effect size (r = Z / sqrt(N))
        z_score = stats.norm.ppf(p_value / 2)
        r_effect = abs(z_score) / np.sqrt(n) if n > 0 else 0
        
        return {
            'test': 'Wilcoxon Signed-Rank Test',
            'n_pairs': n,
            'w_statistic': stat,
            'p_value': p_value,
            'p_value_formatted': f"{'{:e}'.format(p_value)}",
            'significant': p_value < self.alpha,
            'effect_size_r': round(r_effect, 4),
            'effect_interpretation': 'Small' if r_effect < 0.3 else ('Medium' if r_effect < 0.5 else 'Large'),
            'note': 'Non-parametric alternative to paired t-test'
        }
    
    # Helper methods
    def _levene_test(self, group1: np.ndarray, group2: np.ndarray) -> dict:
        """Test for homogeneity of variances."""
        stat, p_value = levene(group1, group2)
        return {
            'statistic': round(stat, 4),
            'p_value': p_value,
            'equal_variances': p_value > self.alpha
        }
    
    def _cohens_interpretation(self, d: float) -> str:
        d = abs(d)
        if d < 0.2: return 'Negligible'
        elif d < 0.5: return 'Small'
        elif d < 0.8: return 'Medium'
        else: return 'Large'
    
    def _eta_interpretation(self, eta: float) -> str:
        if eta < 0.01: return 'Negligible'
        elif eta < 0.06: return 'Small'
        elif eta < 0.14: return 'Medium'
        else: return 'Large'
    
    def _cramers_interpretation(self, v: float) -> str:
        v = abs(v)
        if v < 0.1: return 'Negligible'
        elif v < 0.3: return 'Small'
        elif v < 0.5: return 'Medium'
        else: return 'Large'
    
    def _correlation_interpretation(self, r: float) -> str:
        r = abs(r)
        if r < 0.1: return 'Negligible'
        elif r < 0.3: return 'Weak'
        elif r < 0.5: return 'Moderate'
        elif r < 0.7: return 'Strong'
        else: return 'Very Strong'


def main():
    """Run comprehensive statistical analysis demonstration."""
    import time
    start_time = time.time()
    
    log_pipeline_start("Statistical Analysis")
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(alpha=0.05)
    
    # Load processed data
    tickets_path = path_manager.get_processed_data_path('tickets_cleaned.csv')
    
    if not tickets_path.exists():
        logger.warning(f"No processed data found at {tickets_path}")
        logger.info("Generating sample data for demonstration...")
        np.random.seed(42)
        n = 1000
        
        # Simulate ticket data with different channels
        channel_a = pd.DataFrame({
            'channel': ['Email'] * n,
            'csat_score': np.random.normal(4.0, 0.8, n).clip(1, 5),
            'resolution_time_min': np.random.normal(200, 60, n).clip(10, 600)
        })
        
        channel_b = pd.DataFrame({
            'channel': ['Chat'] * n,
            'csat_score': np.random.normal(4.3, 0.8, n).clip(1, 5),
            'resolution_time_min': np.random.normal(150, 50, n).clip(10, 600)
        })
        
        channel_c = pd.DataFrame({
            'channel': ['Phone'] * n,
            'csat_score': np.random.normal(3.8, 0.9, n).clip(1, 5),
            'resolution_time_min': np.random.normal(180, 70, n).clip(10, 600)
        })
        
        df = pd.concat([channel_a, channel_b, channel_c], ignore_index=True)
    else:
        df = pd.read_csv(tickets_path)
        logger.info(f"Loaded data: {len(df)} records")
    
    results = {}
    
    # Analysis 1: Descriptive Statistics
    logger.info("\n=== Descriptive Statistics: CSAT Score ===")
    csat_stats = analyzer.descriptive_stats(df['csat_score'].values)
    results['csat_descriptive'] = csat_stats
    logger.info(f"Mean: {csat_stats['mean']}, Std: {csat_stats['std']}, Median: {csat_stats['median']}")
    
    # Analysis 2: Normality Tests
    logger.info("\n=== Normality Tests: CSAT Score ===")
    normality = analyzer.normality_tests(df['csat_score'].values)
    results['normality_tests'] = normality
    logger.info(f"Is Normal: {normality['is_normal']}")
    
    # Analysis 3: Compare CSAT by Channel (ANOVA)
    logger.info("\n=== One-Way ANOVA: CSAT by Channel ===")
    email_csat = df[df['channel'] == 'Email']['csat_score'].dropna().values
    chat_csat = df[df['channel'] == 'Chat']['csat_score'].dropna().values
    phone_csat = df[df['channel'] == 'Phone']['csat_score'].dropna().values
    
    if len(email_csat) > 0 and len(chat_csat) > 0 and len(phone_csat) > 0:
        anova_result = analyzer.oneway_anova(email_csat, chat_csat, phone_csat)
        results['anova_csat_by_channel'] = anova_result
        logger.info(f"F={anova_result['f_statistic']:.2f}, p={anova_result['p_value']:.4f}")
        logger.info(f"Effect (η²): {anova_result['eta_squared']} ({anova_result['effect_interpretation']})")
    
    # Analysis 4: Non-parametric alternative (Kruskal-Wallis)
    logger.info("\n=== Kruskal-Wallis Test: CSAT by Channel ===")
    kruskal_result = analyzer.kruskal_wallis(email_csat, chat_csat, phone_csat)
    results['kruskal_csat_by_channel'] = kruskal_result
    logger.info(f"H={kruskal_result['h_statistic']:.2f}, p={kruskal_result['p_value']:.4f}")
    
    # Analysis 5: Correlation Analysis
    logger.info("\n=== Correlation: Resolution Time vs CSAT ===")
    corr_data = df[['resolution_time_min', 'csat_score']].dropna()
    if len(corr_data) > 0:
        corr_result = analyzer.correlation_analysis(
            corr_data['resolution_time_min'].values,
            corr_data['csat_score'].values
        )
        results['correlation_resolution_csat'] = corr_result
        logger.info(f"Pearson r={corr_result['pearson']['r']:.3f}, p={corr_result['pearson']['p_value']:.4f}")
        logger.info(f"Spearman ρ={corr_result['spearman']['rho']:.3f}")
    
    # Analysis 6: Compare Resolution Time by Priority
    logger.info("\n=== t-test: Resolution Time by Priority ===")
    if 'priority' in df.columns:
        high_priority = df[df['priority'] == 'High']['resolution_time_min'].dropna().values
        low_priority = df[df['priority'] == 'Low']['resolution_time_min'].dropna().values
        
        if len(high_priority) > 0 and len(low_priority) > 0:
            ttest_result = analyzer.ttest_independent(high_priority, low_priority)
            results['ttest_resolution_by_priority'] = ttest_result
            logger.info(f"High: {ttest_result['mean1']:.1f} min, Low: {ttest_result['mean2']:.1f} min")
            logger.info(f"t={ttest_result['t_statistic']:.2f}, p={ttest_result['p_value']:.4f}")
            logger.info(f"Effect (d={ttest_result['cohens_d']:.3f}): {ttest_result['effect_interpretation']}")
    
    # Save results
    output_path = path_manager.get_processed_data_path('statistical_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nSaved analysis results to: {output_path}")
    
    duration = time.time() - start_time
    log_pipeline_complete("Statistical Analysis", duration)
    
    return results


if __name__ == "__main__":
    results = main()
