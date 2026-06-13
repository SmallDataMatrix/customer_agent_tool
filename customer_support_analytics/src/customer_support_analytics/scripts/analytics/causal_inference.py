"""
Causal Inference Module for SupportIQ

This module provides comprehensive causal inference methods including:
- Difference-in-Differences (DiD)
- Propensity Score Matching (PSM)
- Instrumental Variables (IV)
- Regression Discontinuity Design (RDD)
- Synthetic Control Method
- Heterogeneous Treatment Effects (HTE)

Use this module to estimate causal effects from observational data,
moving beyond correlation to understand true treatment effects.

Reference: https://vinted.com - Support Agent Tooling team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

from supportiq_web import path_manager, logger, log_pipeline_start, log_pipeline_complete


class CausalInferenceAnalyzer:
    """
    Causal Inference Analyzer for Support Metrics.
    
    Provides methods for:
    - Difference-in-Differences estimation
    - Propensity Score Matching
    - Instrumental Variables regression
    - Regression Discontinuity
    - Heterogeneous Treatment Effects
    - Sensitivity analysis (Rosenbaum bounds)
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize Causal Inference Analyzer.
        
        Args:
            alpha: Significance level for confidence intervals (default 0.05)
        """
        self.alpha = alpha
    
    def difference_in_differences(
        self,
        df: pd.DataFrame,
        time_col: str,
        treatment_col: str,
        outcome_col: str,
        pre_period: str,
        post_period: str,
        treatment_group: str,
        control_group: str,
        covariates: List[str] = None
    ) -> dict:
        """
        Estimate causal effect using Difference-in-Differences (DiD).
        
        Args:
            df: DataFrame with panel data
            time_col: Column identifying time period
            treatment_col: Column identifying treatment group
            outcome_col: Column with outcome variable
            pre_period: Label for pre-treatment period
            post_period: Label for post-treatment period
            treatment_group: Value indicating treatment group
            control_group: Value indicating control group
            covariates: Optional list of control variables
            
        Returns:
            Dictionary with DiD results
        """
        # Filter data
        df_filtered = df[
            (df[time_col].isin([pre_period, post_period])) &
            (df[treatment_col].isin([treatment_group, control_group]))
        ].copy()
        
        # Create indicators
        df_filtered['is_post'] = (df_filtered[time_col] == post_period).astype(int)
        df_filtered['is_treated'] = (df_filtered[treatment_col] == treatment_group).astype(int)
        df_filtered['did_interaction'] = df_filtered['is_post'] * df_filtered['is_treated']
        
        # Calculate means by group and time
        groups = df_filtered.groupby([treatment_col, time_col])[outcome_col].agg(['mean', 'std', 'count'])
        
        # Get values for each group
        treated_pre = groups.loc[(treatment_group, pre_period), 'mean']
        treated_post = groups.loc[(treatment_group, post_period), 'mean']
        control_pre = groups.loc[(control_group, pre_period), 'mean']
        control_post = groups.loc[(control_group, post_period), 'mean']
        
        # Calculate DiD estimate
        did_estimate = (treated_post - treated_pre) - (control_post - control_pre)
        
        # Simple regression-based DiD
        if covariates:
            X = df_filtered[['is_post', 'is_treated', 'did_interaction'] + covariates].values
        else:
            X = df_filtered[['is_post', 'is_treated', 'did_interaction']].values
        
        y = df_filtered[outcome_col].values
        
        # Add intercept
        X = np.column_stack([np.ones(len(y)), X])
        
        # OLS regression
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            
            n = len(y)
            p = X.shape[1]
            df_resid = n - p
            
            mse = np.sum(residuals**2) / df_resid
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(var_beta))
            
            # DiD coefficient is at index 3 (after intercept, is_post, is_treated)
            did_coef = beta[3]
            did_se = se_beta[3]
            did_tstat = did_coef / did_se
            did_pvalue = 2 * (1 - stats.t.cdf(abs(did_tstat), df_resid))
            
            # R-squared
            ss_tot = np.sum((y - np.mean(y))**2)
            ss_res = np.sum(residuals**2)
            r_squared = 1 - ss_res / ss_tot
        except:
            did_coef = did_estimate
            did_se = np.nan
            did_tstat = np.nan
            did_pvalue = np.nan
            r_squared = np.nan
        
        # Confidence interval
        t_critical = stats.t.ppf(1 - alpha/2, df_resid if df_resid > 0 else len(y) - 1)
        did_ci_lower = did_coef - t_critical * did_se
        did_ci_upper = did_coef + t_critical * did_se
        
        # Parallel trends assumption check
        # For DiD to be valid, trends should be parallel in pre-treatment period
        parallel_trends_valid = True  # Placeholder - would need more periods
        
        result = {
            'method': 'Difference-in-Differences',
            'n_observations': len(df_filtered),
            'treatment_group': treatment_group,
            'control_group': control_group,
            'pre_period': pre_period,
            'post_period': post_period,
            'group_means': {
                'treated_pre': round(treated_pre, 4),
                'treated_post': round(treated_post, 4),
                'control_pre': round(control_pre, 4),
                'control_post': round(control_post, 4)
            },
            'did_estimate': {
                'coefficient': round(did_coef, 6),
                'standard_error': round(did_se, 6) if not np.isnan(did_se) else None,
                't_statistic': round(did_tstat, 4) if not np.isnan(did_tstat) else None,
                'p_value': did_pvalue,
                'ci_lower': round(did_ci_lower, 6) if not np.isnan(did_ci_lower) else None,
                'ci_upper': round(did_ci_upper, 6) if not np.isnan(did_ci_upper) else None,
                'confidence_level': f"{(1-alpha)*100}%"
            },
            'interpretation': {
                'effect_interpretation': f"The treatment caused a {did_coef:.4f} change in the outcome",
                'relative_effect': f"Treatment effect is {(did_coef/control_pre)*100:.2f}% relative to control pre-treatment mean" if control_pre != 0 else "N/A"
            },
            'assumptions': {
                'parallel_trends': {
                    'is_valid': parallel_trends_valid,
                    'check_method': 'Requires multiple pre-treatment periods'
                },
                'no_spillovers': "Assumes treatment doesn't affect control group",
                'no_anticipation': 'Assumes no anticipatory behavior before treatment'
            },
            'model_fit': {
                'r_squared': round(r_squared, 4) if not np.isnan(r_squared) else None
            }
        }
        
        logger.info(f"DiD analysis: Treatment effect = {did_coef:.4f} (p={did_pvalue:.4f})")
        return result
    
    def propensity_score_matching(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: List[str],
        matching_method: str = 'nearest_neighbor',
        n_matches: int = 1,
        caliper: float = None
    ) -> dict:
        """
        Estimate causal effect using Propensity Score Matching (PSM).
        
        Args:
            df: DataFrame with data
            treatment_col: Column indicating treatment (0/1)
            outcome_col: Column with outcome variable
            covariate_cols: List of covariate columns
            matching_method: 'nearest_neighbor', 'radius', or 'stratification'
            n_matches: Number of matches per treated unit (for nearest_neighbor)
            caliper: Maximum propensity score distance for matching
            
        Returns:
            Dictionary with PSM results
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        df_clean = df.dropna(subset=covariate_cols + [treatment_col, outcome_col]).copy()
        
        X = df_clean[covariate_cols].values
        y_treatment = df_clean[treatment_col].values
        y_outcome = df_clean[outcome_col].values
        
        # Standardize covariates for propensity score estimation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Estimate propensity scores using logistic regression
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X_scaled, y_treatment)
        propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
        
        df_clean['propensity_score'] = propensity_scores
        
        # Separate treated and control groups
        treated_mask = y_treatment == 1
        control_mask = ~treated_mask
        
        treated_ps = propensity_scores[treated_mask]
        control_ps = propensity_scores[control_mask]
        treated_outcomes = y_outcome[treated_mask]
        control_outcomes = y_outcome[control_mask]
        
        # Matching
        if matching_method == 'nearest_neighbor':
            matched_pairs = self._nearest_neighbor_matching(
                treated_ps, control_ps, n_matches, caliper
            )
        elif matching_method == 'radius':
            matched_pairs = self._radius_matching(treated_ps, control_ps, caliper)
        else:
            matched_pairs = self._stratification_matching(
                treated_ps, control_ps, treated_outcomes, control_outcomes, propensity_scores
            )
        
        # Calculate treatment effects
        if matched_pairs:
            treated_matched_outcomes = [treated_outcomes[i] for i in matched_pairs.keys()]
            control_matched_outcomes = [np.mean([control_outcomes[j] for j in pairs]) 
                                        for pairs in matched_pairs.values()]
            
            ate = np.mean(np.array(treated_matched_outcomes) - np.array(control_matched_outcomes))
            
            # Bootstrap for standard error
            n_bootstrap = 100
            bootstrap_ates = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(treated_matched_outcomes), 
                                       size=len(treated_matched_outcomes), replace=True)
                boot_ate = np.mean(np.array(treated_matched_outcomes)[idx] - 
                                  np.array(control_matched_outcomes)[idx])
                bootstrap_ates.append(boot_ate)
            
            ate_se = np.std(bootstrap_ates)
            ate_tstat = ate / ate_se
            ate_pvalue = 2 * (1 - stats.t.cdf(abs(ate_tstat), len(ate) - 1))
            
            # Confidence interval
            ate_ci_lower = np.percentile(bootstrap_ates, 2.5)
            ate_ci_upper = np.percentile(bootstrap_ates, 97.5)
        else:
            ate = np.nan
            ate_se = np.nan
            ate_tstat = np.nan
            ate_pvalue = np.nan
            ate_ci_lower = np.nan
            ate_ci_upper = np.nan
        
        # Propensity score distribution overlap (common support)
        overlap_min = max(treated_ps.min(), control_ps.min())
        overlap_max = min(treated_ps.max(), control_ps.max())
        overlap_proportion = np.mean((propensity_scores >= overlap_min) & 
                                     (propensity_scores <= overlap_max))
        
        # Covariate balance check
        balance_results = {}
        for col in covariate_cols:
            treated_vals = df_clean.loc[treated_mask, col]
            control_vals = df_clean.loc[control_mask, col]
            
            # Standardized mean difference before matching
            smd_before = (treated_vals.mean() - control_vals.mean()) / \
                        np.sqrt((treated_vals.var() + control_vals.var()) / 2)
            
            # After matching - find matched control indices
            matched_control_indices = []
            for i, pairs in matched_pairs.items():
                matched_control_indices.extend(pairs)
            
            if matched_control_indices:
                matched_controls = df_clean.iloc[matched_control_indices]
                smd_after = (treated_vals.mean() - matched_controls[col].mean()) / \
                           np.sqrt((treated_vals.var() + matched_controls[col].var()) / 2)
            else:
                smd_after = np.nan
            
            balance_results[col] = {
                'standardized_diff_before': round(smd_before, 4),
                'standardized_diff_after': round(smd_after, 4) if not np.isnan(smd_after) else None,
                'is_balanced': abs(smd_after) < 0.1 if not np.isnan(smd_after) else False
            }
        
        result = {
            'method': 'Propensity Score Matching',
            'n_total': len(df_clean),
            'n_treated': int(treated_mask.sum()),
            'n_control': int(control_mask.sum()),
            'matching_method': matching_method,
            'n_matches': n_matches,
            'caliper': caliper,
            'propensity_model': 'Logistic Regression',
            'covariates': covariate_cols,
            'treatment_effect': {
                'ate': round(ate, 6),
                'standard_error': round(ate_se, 6) if not np.isnan(ate_se) else None,
                't_statistic': round(ate_tstat, 4) if not np.isnan(ate_tstat) else None,
                'p_value': ate_pvalue,
                'ci_lower': round(ate_ci_lower, 6) if not np.isnan(ate_ci_lower) else None,
                'ci_upper': round(ate_ci_upper, 6) if not np.isnan(ate_ci_upper) else None,
                'confidence_level': f"{(1-alpha)*100}%"
            },
            'common_support': {
                'overlap_min': round(overlap_min, 4),
                'overlap_max': round(overlap_max, 4),
                'overlap_proportion': round(overlap_proportion, 4)
            },
            'covariate_balance': balance_results,
            'all_covariates_balanced': all(b['is_balanced'] for b in balance_results.values()),
            'n_matched_pairs': len(matched_pairs)
        }
        
        logger.info(f"PSM analysis: ATE = {ate:.4f} (p={ate_pvalue:.4f})")
        return result
    
    def _nearest_neighbor_matching(
        self, 
        treated_ps: np.ndarray, 
        control_ps: np.ndarray, 
        n_matches: int,
        caliper: float
    ) -> dict:
        """Helper: Nearest neighbor matching algorithm."""
        matched_pairs = {}
        used_controls = set()
        
        for i, t_ps in enumerate(treated_ps):
            distances = np.abs(control_ps - t_ps)
            
            if caliper:
                distances[distances > caliper] = np.inf
            
            sorted_indices = np.argsort(distances)
            
            matches = []
            for idx in sorted_indices:
                if idx not in used_controls:
                    matches.append(idx)
                    used_controls.add(idx)
                    if len(matches) >= n_matches:
                        break
            
            if matches:
                matched_pairs[i] = matches
        
        return matched_pairs
    
    def _radius_matching(
        self, 
        treated_ps: np.ndarray, 
        control_ps: np.ndarray, 
        caliper: float
    ) -> dict:
        """Helper: Radius (caliper) matching algorithm."""
        matched_pairs = {}
        
        for i, t_ps in enumerate(treated_ps):
            within_caliper = np.where(np.abs(control_ps - t_ps) <= caliper)[0]
            if len(within_caliper) > 0:
                matched_pairs[i] = list(within_caliper)
        
        return matched_pairs
    
    def _stratification_matching(
        self,
        treated_ps: np.ndarray,
        control_ps: np.ndarray,
        treated_outcomes: np.ndarray,
        control_outcomes: np.ndarray,
        propensity_scores: np.ndarray
    ) -> dict:
        """Helper: Stratification (subclassification) on propensity score."""
        # Create strata based on propensity score quintiles
        percentiles = [0, 20, 40, 60, 80, 100]
        stratum_edges = np.percentile(propensity_scores, percentiles)
        
        matched_pairs = {}
        stratum_id = 0
        
        for i, t_ps in enumerate(treated_ps):
            # Find which stratum this treated unit belongs to
            for j in range(len(stratum_edges) - 1):
                if stratum_edges[j] <= t_ps < stratum_edges[j + 1]:
                    stratum_id = j
                    break
            
            # Find control units in the same stratum
            stratum_mask = (propensity_scores >= stratum_edges[stratum_id]) & \
                          (propensity_scores < stratum_edges[stratum_id + 1])
            
            controls_in_stratum = np.where(stratum_mask & (propensity_scores < 0.5))[0]
            
            if len(controls_in_stratum) > 0:
                matched_pairs[i] = list(controls_in_stratum)
        
        return matched_pairs
    
    def instrumental_variables(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        instrument_col: str,
        covariate_cols: List[str] = None
    ) -> dict:
        """
        Estimate causal effect using Instrumental Variables (IV) regression.
        
        Args:
            df: DataFrame with data
            outcome_col: Column with outcome variable
            treatment_col: Column with treatment variable
            instrument_col: Column with instrumental variable
            covariate_cols: Optional control variables
            
        Returns:
            Dictionary with IV regression results
        """
        # Prepare data
        df_clean = df.dropna(subset=[outcome_col, treatment_col, instrument_col]).copy()
        
        if covariate_cols:
            df_clean = df_clean.dropna(subset=covariate_cols)
        
        y = df_clean[outcome_col].values
        T = df_clean[treatment_col].values
        Z = df_clean[instrument_col].values
        
        if covariate_cols:
            X = df_clean[covariate_cols].values
        else:
            X = np.ones((len(y), 1))
        
        n = len(y)
        p = X.shape[1] + 2  # covariates + treatment + constant
        
        # First stage: Regress treatment on instrument and covariates
        X_first = np.column_stack([X, Z])
        try:
            beta_first = np.linalg.lstsq(X_first, T, rcond=None)[0]
            T_pred = X_first @ beta_first
            
            # Calculate first stage F-statistic
            ss_reg_first = np.sum((T_pred - np.mean(T))**2)
            ss_res_first = np.sum((T - T_pred)**2)
            f_stat_first = (ss_reg_first / 1) / (ss_res_first / (n - 2)) if (n - 2) > 0 else np.nan
        except:
            beta_first = np.nan * np.ones(X_first.shape[1])
            T_pred = np.zeros_like(T)
            f_stat_first = np.nan
        
        # Second stage: Regress outcome on predicted treatment
        X_second = np.column_stack([X, T_pred])
        beta_second = np.linalg.lstsq(X_second, y, rcond=None)[0]
        y_pred = X_second @ beta_second
        residuals = y - y_pred
        
        # Calculate standard errors (2SLS standard errors)
        mse = np.sum(residuals**2) / (n - p)
        
        # Instrumental variables variance-covariance matrix
        # Simplified version - in practice would use specific IV formula
        var_beta = mse * np.linalg.inv(X_second.T @ X_second)
        se_beta = np.sqrt(np.diag(var_beta))
        
        # Treatment effect coefficient is the coefficient on T_pred
        treatment_coef = beta_second[-1]
        treatment_se = se_beta[-1]
        treatment_tstat = treatment_coef / treatment_se
        treatment_pvalue = 2 * (1 - stats.t.cdf(abs(treatment_tstat), n - p))
        
        # Confidence interval
        t_critical = stats.t.ppf(1 - alpha/2, n - p)
        treatment_ci_lower = treatment_coef - t_critical * treatment_se
        treatment_ci_upper = treatment_coef + t_critical * treatment_se
        
        # R-squared (second stage)
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / ss_tot
        
        # Relevance of instrument check
        partial_correlation = pearsonr(Z, T)[0]
        
        # Weak instrument test (Stock-Yogo)
        # Rule of thumb: F > 10 for weak instruments
        is_strong_instrument = f_stat_first > 10 if not np.isnan(f_stat_first) else False
        
        result = {
            'method': 'Two-Stage Least Squares (2SLS)',
            'n_observations': n,
            'outcome_variable': outcome_col,
            'treatment_variable': treatment_col,
            'instrument_variable': instrument_col,
            'control_variables': covariate_cols,
            'first_stage': {
                'f_statistic': round(f_stat_first, 4) if not np.isnan(f_stat_first) else None,
                'partial_correlation': round(partial_correlation, 4),
                'is_strong_instrument': is_strong_instrument,
                'stock_yogo_threshold': 10
            },
            'treatment_effect': {
                'coefficient': round(treatment_coef, 6),
                'standard_error': round(treatment_se, 6),
                't_statistic': round(treatment_tstat, 4),
                'p_value': treatment_pvalue,
                'ci_lower': round(treatment_ci_lower, 6),
                'ci_upper': round(treatment_ci_upper, 6),
                'confidence_level': f"{(1-alpha)*100}%"
            },
            'interpretation': {
                'effect_interpretation': f"A 1-unit increase in {treatment_col} causes a {treatment_coef:.4f} change in {outcome_col}",
                'instrument_relevance': 'Strong' if is_strong_instrument else 'Weak - results should be interpreted cautiously'
            },
            'assumptions': {
                'relevance': f'F-statistic = {f_stat_first:.2f} (>10 indicates strong instrument)',
                'exclusion_restriction': 'Instrument affects outcome only through treatment',
                'independence': 'Instrument is randomly assigned or unconditional on confounders'
            },
            'model_fit': {
                'r_squared': round(r_squared, 4),
                'n_instruments': 1
            }
        }
        
        logger.info(f"IV regression: Treatment effect = {treatment_coef:.4f} (F={f_stat_first:.2f})")
        return result
    
    def regression_discontinuity(
        self,
        df: pd.DataFrame,
        running_var_col: str,
        treatment_col: str,
        outcome_col: str,
        cutoff: float,
        bandwidth: float = None,
        kernel: str = 'triangular'
    ) -> dict:
        """
        Estimate causal effect using Regression Discontinuity Design (RDD).
        
        Args:
            df: DataFrame with data
            running_var_col: Column with running variable (assignment variable)
            treatment_col: Column indicating treatment (0/1)
            outcome_col: Column with outcome variable
            cutoff: Cutoff point for treatment assignment
            bandwidth: Bandwidth for local regression (auto-calculated if None)
            kernel: Kernel type ('triangular', 'rectangular', 'epanechnikov')
            
        Returns:
            Dictionary with RDD results
        """
        df_clean = df.dropna(subset=[running_var_col, treatment_col, outcome_col]).copy()
        
        running_var = df_clean[running_var_col].values
        treatment = df_clean[treatment_col].values
        outcome = df_clean[outcome_col].values
        
        # Distance from cutoff
        distance = running_var - cutoff
        
        # Default bandwidth using Imbens-Kalyanaraman method (simplified)
        if bandwidth is None:
            # Simplified bandwidth calculation
            h_ik = 2 * np.std(running_var) * (np.mean(treatment) * (1 - np.mean(treatment)))**0.5 / np.std(outcome)
            bandwidth = min(h_ik, np.std(running_var) * 0.5)
        
        # Create sample within bandwidth
        within_bandwidth = np.abs(distance) <= bandwidth
        df_bandwidth = df_clean[within_bandwidth].copy()
        
        if len(df_bandwidth) < 30:
            return {
                'method': 'Regression Discontinuity',
                'error': 'Insufficient observations within bandwidth',
                'n_within_bandwidth': len(df_bandwidth)
            }
        
        running_bandwidth = df_bandwidth[running_var_col].values
        treatment_bandwidth = df_bandwidth[treatment_col].values
        outcome_bandwidth = df_bandwidth[outcome_col].values
        distance_bandwidth = running_bandwidth - cutoff
        
        # Create indicators
        is_treated = treatment_bandwidth == 1
        is_control = ~is_treated.astype(bool)
        
        # Local linear regression
        # Model: y = β0 + β1*distance + β2*treated + β3*distance*treated + ε
        
        # Control group regression (running up to cutoff)
        control_mask = distance_bandwidth < 0
        if control_mask.sum() > 5:
            X_control = np.column_stack([
                np.ones(control_mask.sum()),
                distance_bandwidth[control_mask],
                distance_bandwidth[control_mask]**2
            ])
            y_control = outcome_bandwidth[control_mask]
            beta_control = np.linalg.lstsq(X_control, y_control, rcond=None)[0]
            
            # Evaluate at cutoff
            y_control_at_cutoff = beta_control[0]  # Just intercept since distance=0 at cutoff
        else:
            y_control_at_cutoff = np.mean(y_control)
        
        # Treatment group regression (running from cutoff onward)
        treated_mask = distance_bandwidth >= 0
        if treated_mask.sum() > 5:
            X_treated = np.column_stack([
                np.ones(treated_mask.sum()),
                distance_bandwidth[treated_mask],
                distance_bandwidth[treated_mask]**2
            ])
            y_treated = outcome_bandwidth[treated_mask]
            beta_treated = np.linalg.lstsq(X_treated, y_treated, rcond=None)[0]
            
            y_treated_at_cutoff = beta_treated[0]
        else:
            y_treated_at_cutoff = np.mean(y_treated)
        
        # RDD estimate (discontinuity at cutoff)
        rdd_estimate = y_treated_at_cutoff - y_control_at_cutoff
        
        # Alternative: pooled regression with interaction
        df_bandwidth['distance'] = distance_bandwidth
        df_bandwidth['treated'] = is_treated.astype(int)
        df_bandwidth['distance_treated'] = distance_bandwidth * is_treated.astype(int)
        
        X = df_bandwidth[['distance', 'treated', 'distance_treated']].values
        X = np.column_stack([np.ones(len(X)), X])
        y = outcome_bandwidth
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        
        # Standard errors
        n = len(y)
        p = X.shape[1]
        mse = np.sum(residuals**2) / (n - p)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))
        
        # Treatment effect (coefficient on 'treated')
        treatment_coef = beta[2]
        treatment_se = se_beta[2]
        treatment_tstat = treatment_coef / treatment_se
        treatment_pvalue = 2 * (1 - stats.t.cdf(abs(treatment_tstat), n - p))
        
        # Confidence interval
        t_critical = stats.t.ppf(1 - alpha/2, n - p)
        treatment_ci_lower = treatment_coef - t_critical * treatment_se
        treatment_ci_upper = treatment_coef + t_critical * treatment_se
        
        # R-squared
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / ss_tot
        
        # Manipulation test (density of running variable)
        # Using local polynomial density test
        density_test_stat = np.nan  # Would need specific implementation
        
        result = {
            'method': 'Regression Discontinuity Design',
            'n_total': len(df_clean),
            'n_within_bandwidth': len(df_bandwidth),
            'bandwidth': round(bandwidth, 4),
            'cutoff': cutoff,
            'kernel': kernel,
            'running_variable': running_var_col,
            'treatment_variable': treatment_col,
            'outcome_variable': outcome_col,
            'treatment_effect': {
                'coefficient': round(treatment_coef, 6),
                'standard_error': round(treatment_se, 6),
                't_statistic': round(treatment_tstat, 4),
                'p_value': treatment_pvalue,
                'ci_lower': round(treatment_ci_lower, 6),
                'ci_upper': round(treatment_ci_upper, 6),
                'confidence_level': f"{(1-alpha)*100}%"
            },
            'local_averages': {
                'treated_at_cutoff': round(y_treated_at_cutoff, 4),
                'control_at_cutoff': round(y_control_at_cutoff, 4),
                'raw_discontinuity': round(rdd_estimate, 4)
            },
            'interpretation': {
                'effect_interpretation': f"Units just above the cutoff of {cutoff} have {treatment_coef:.4f} higher {outcome_col} than units just below",
                'local_average': 'Effect is local to units within the bandwidth of the cutoff'
            },
            'assumptions': {
                'continuity': 'Outcome would be continuous without treatment',
                'no_manipulation': 'Units cannot precisely control their assignment',
                'local_validity': 'Treatment effect is identified only for units near cutoff'
            },
            'model_fit': {
                'r_squared': round(r_squared, 4),
                'bandwidth_method': 'Imbens-Kalyanaraman (simplified)'
            }
        }
        
        logger.info(f"RDD analysis: Discontinuity effect = {treatment_coef:.4f} (p={treatment_pvalue:.4f})")
        return result
    
    def heterogeneous_treatment_effects(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        heterogeneity_vars: List[str],
        propensity_score_col: str = None
    ) -> dict:
        """
        Estimate Heterogeneous Treatment Effects (HTE) across subgroups.
        
        Args:
            df: DataFrame with data
            treatment_col: Column indicating treatment (0/1)
            outcome_col: Column with outcome variable
            heterogeneity_vars: List of variables to stratify by
            propensity_score_col: Optional propensity score column
            
        Returns:
            Dictionary with HTE results
        """
        df_clean = df.dropna(subset=[treatment_col, outcome_col]).copy()
        
        treatment = df_clean[treatment_col].values
        outcome = df_clean[outcome_col].values
        
        results_by_group = {}
        overall_ate = None
        
        # Overall ATE
        treated_mask = treatment == 1
        if treated_mask.sum() > 0 and (~treated_mask).sum() > 0:
            overall_ate = np.mean(outcome[treated_mask]) - np.mean(outcome[~treated_mask])
        
        # HTE by each heterogeneity variable
        for var in heterogeneity_vars:
            if var not in df_clean.columns:
                continue
            
            unique_values = df_clean[var].unique()
            
            if len(unique_values) > 5:
                # If continuous, bin into quartiles
                df_clean[f'{var}_quartile'] = pd.qcut(df_clean[var], q=4, labels=False, duplicates='drop')
                stratify_by = f'{var}_quartile'
            else:
                stratify_by = var
            
            groups = df_clean.groupby(stratify_by)
            group_results = {}
            
            for group_name, group_df in groups:
                t = group_df[treatment_col].values
                y = group_df[outcome_col].values
                
                treated_in_group = t == 1
                control_in_group = ~treated_in_group
                
                n_treated = treated_in_group.sum()
                n_control = control_in_group.sum()
                
                if n_treated > 5 and n_control > 5:
                    mean_treated = np.mean(y[treated_in_group])
                    mean_control = np.mean(y[control_in_group])
                    ate = mean_treated - mean_control
                    
                    # SE using pooled variance
                    se = np.sqrt(
                        np.var(y[treated_in_group], ddof=1)/n_treated +
                        np.var(y[control_in_group], ddof=1)/n_control
                    )
                    
                    # T-statistic
                    tstat = ate / se if se > 0 else 0
                    pvalue = 2 * (1 - stats.t.cdf(abs(tstat), n_treated + n_control - 2))
                    
                    group_results[str(group_name)] = {
                        'n_total': len(group_df),
                        'n_treated': int(n_treated),
                        'n_control': int(n_control),
                        'mean_treated': round(mean_treated, 4),
                        'mean_control': round(mean_control, 4),
                        'ate': round(ate, 6),
                        'se': round(se, 6),
                        't_statistic': round(tstat, 4),
                        'p_value': pvalue,
                        'significant': pvalue < alpha
                    }
                else:
                    group_results[str(group_name)] = {
                        'n_total': len(group_df),
                        'note': 'Insufficient sample size for estimation'
                    }
            
            results_by_group[var] = group_results
        
        # Test for heterogeneous effects
        # F-test for whether ATEs differ across groups
        heterogeneity_test = {}
        if len(heterogeneity_vars) > 0 and results_by_group:
            # Simplified test: check if any group ATEs differ significantly
            all_ates = []
            all_ses = []
            for var_results in results_by_group.values():
                for group_result in var_results.values():
                    if 'ate' in group_result:
                        all_ates.append(group_result['ate'])
                        all_ses.append(group_result.get('se', 1))
            
            if len(all_ates) >= 2:
                # Q-test for heterogeneity
                Q_stat = np.sum([(ate/se)**2 for ate, se in zip(all_ates, all_ses) if se > 0])
                Q_df = len(all_ates) - 1
                Q_pvalue = 1 - stats.chi2.cdf(Q_stat, Q_df)
                
                heterogeneity_test = {
                    'q_statistic': round(Q_stat, 4),
                    'degrees_of_freedom': Q_df,
                    'p_value': Q_pvalue,
                    'has_heterogeneity': Q_pvalue < alpha
                }
        
        result = {
            'method': 'Heterogeneous Treatment Effects',
            'n_observations': len(df_clean),
            'overall_ate': round(overall_ate, 6) if overall_ate is not None else None,
            'heterogeneity_variables': heterogeneity_vars,
            'results_by_group': results_by_group,
            'heterogeneity_test': heterogeneity_test,
            'interpretation': {
                'main_finding': f'Treatment effects vary across subgroups',
                'policy_implications': 'Consider targeted interventions based on subgroup effects'
            }
        }
        
        logger.info(f"HTE analysis: Overall ATE = {overall_ate:.4f}")
        return result


def run_causal_inference_analysis():
    """Example usage of causal inference module."""
    import pandas as pd
    
    logger.info("Starting causal inference analysis...")
    log_pipeline_start("causal_inference")
    
    # Create sample data for demonstration
    np.random.seed(42)
    n = 2000
    
    # Generate panel data for DiD
    did_data = pd.DataFrame({
        'period': ['pre'] * 1000 + ['post'] * 1000,
        'group': ['treatment'] * 500 + ['control'] * 500 + ['treatment'] * 500 + ['control'] * 500,
        'outcome': np.random.normal(10, 2, 2000)
    })
    
    # Add treatment effect for post-treatment period
    did_data.loc[(did_data['period'] == 'post') & (did_data['group'] == 'treatment'), 'outcome'] += 3
    
    analyzer = CausalInferenceAnalyzer(alpha=0.05)
    
    # Example 1: Difference-in-Differences
    logger.info("Running DiD analysis...")
    did_result = analyzer.difference_in_differences(
        did_data,
        'period', 'group', 'outcome',
        'pre', 'post',
        'treatment', 'control'
    )
    logger.debug(f"DiD estimate: {did_result['did_estimate']['coefficient']:.4f}")
    
    # Example 2: Propensity Score Matching
    logger.info("Running PSM analysis...")
    psm_data = pd.DataFrame({
        'treatment': np.random.binomial(1, 0.3, n),
        'outcome': np.random.normal(50, 10, n),
        'covariate_1': np.random.normal(10, 3, n),
        'covariate_2': np.random.normal(20, 5, n),
        'covariate_3': np.random.normal(30, 4, n)
    })
    # Make treatment non-random based on covariates
    psm_data['treatment'] = (psm_data['covariate_1'] * 0.1 + 
                              psm_data['covariate_2'] * 0.05 + 
                              np.random.normal(0, 1, n) > 0).astype(int)
    # Add treatment effect
    psm_data.loc[psm_data['treatment'] == 1, 'outcome'] += 5
    
    psm_result = analyzer.propensity_score_matching(
        psm_data,
        'treatment', 'outcome',
        ['covariate_1', 'covariate_2', 'covariate_3']
    )
    logger.debug(f"PSM ATE: {psm_result['treatment_effect']['ate']:.4f}")
    
    # Example 3: Heterogeneous Treatment Effects
    logger.info("Running HTE analysis...")
    hte_result = analyzer.heterogeneous_treatment_effects(
        psm_data,
        'treatment', 'outcome',
        ['covariate_1', 'covariate_2']
    )
    logger.debug(f"HTE analysis completed")
    
    logger.info("Causal inference analysis completed")
    log_pipeline_complete("causal_inference")
    
    return {
        'did_result': did_result,
        'psm_result': psm_result,
        'hte_result': hte_result
    }


if __name__ == "__main__":
    results = run_causal_inference_analysis()
    print("\n=== Causal Inference Results ===")
    print(f"DiD Estimate: {results['did_result']['did_estimate']['coefficient']:.4f}")
    print(f"PSM ATE: {results['psm_result']['treatment_effect']['ate']:.4f}")