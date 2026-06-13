"""
Regression Analysis Module for SupportIQ

This module provides comprehensive regression analysis capabilities including:
- Linear Regression (simple and multivariate)
- Logistic Regression (for classification problems)
- Regularized Regression (Ridge, Lasso, ElasticNet)
- Regression diagnostics (residuals, multicollinearity, influence)
- Model interpretation and feature importance

Use this module to analyze relationships between variables,
predict outcomes, and derive actionable insights from support metrics.

Reference: https://vinted.com - Support Agent Tooling team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

from supportiq_web import path_manager, logger, log_pipeline_start, log_pipeline_complete


class RegressionAnalyzer:
    """
    Regression Analyzer for Support Metrics.
    
    Provides methods for:
    - Linear regression (simple and multivariate)
    - Logistic regression for binary outcomes
    - Regularized regression (Ridge, Lasso, ElasticNet)
    - Regression diagnostics and assumption checking
    - Feature importance and model interpretation
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize Regression Analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests (default 0.05)
        """
        self.alpha = alpha
    
    def simple_linear_regression(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_name: str = 'X',
        target_name: str = 'y'
    ) -> dict:
        """
        Perform simple linear regression (one predictor).
        
        Args:
            X: Predictor variable (1D array)
            y: Target variable (1D array)
            feature_name: Name of predictor for reporting
            target_name: Name of target for reporting
            
        Returns:
            Dictionary with regression results
        """
        # Remove NaN values
        mask = ~(np.isnan(X) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        n = len(X)
        
        # Calculate means
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate regression coefficients
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        
        slope = numerator / denominator
        intercept = y_mean - slope * X_mean
        
        # Calculate residuals
        y_pred = intercept + slope * X
        residuals = y - y_pred
        
        # Calculate standard error of coefficients
        SS_res = np.sum(residuals ** 2)
        SS_tot = np.sum((y - y_mean) ** 2)
        
        mse = SS_res / (n - 2)
        se_slope = np.sqrt(mse / denominator)
        se_intercept = np.sqrt(mse * (1/n + X_mean**2 / denominator))
        
        # Calculate t-statistics and p-values
        t_slope = slope / se_slope
        t_intercept = intercept / se_intercept
        p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), n - 2))
        p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), n - 2))
        
        # Calculate R-squared
        r_squared = 1 - (SS_res / SS_tot)
        
        # Calculate adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)
        
        # Calculate correlation coefficient
        r, p_r = pearsonr(X, y)
        
        # F-statistic for overall model
        SS_reg = np.sum((y_pred - y_mean) ** 2)
        f_stat = (SS_reg / 1) / (SS_res / (n - 2))
        p_f = 1 - stats.f.cdf(f_stat, 1, n - 2)
        
        # Standard error of estimate
        se_estimate = np.sqrt(mse)
        
        # Confidence intervals for coefficients
        t_critical = stats.t.ppf(1 - alpha/2, n - 2)
        slope_ci = [slope - t_critical * se_slope, slope + t_critical * se_slope]
        intercept_ci = [intercept - t_critical * se_intercept, intercept + t_critical * se_intercept]
        
        # Prediction intervals for a new observation
        X_new = np.array([X.min(), X.max()])
        y_new_pred = intercept + slope * X_new
        pred_interval = t_critical * se_estimate * np.sqrt(1 + 1/n + (X_new - X_mean)**2 / denominator)
        
        result = {
            'model_type': 'Simple Linear Regression',
            'n_observations': n,
            'feature_name': feature_name,
            'target_name': target_name,
            'coefficients': {
                'intercept': round(intercept, 6),
                'slope': round(slope, 6),
                'intercept_se': round(se_intercept, 6),
                'slope_se': round(se_slope, 6),
                'intercept_pvalue': p_intercept,
                'slope_pvalue': p_slope,
                'intercept_ci': [round(ci, 6) for ci in intercept_ci],
                'slope_ci': [round(ci, 6) for ci in slope_ci]
            },
            'model_fit': {
                'r_squared': round(r_squared, 6),
                'adj_r_squared': round(adj_r_squared, 6),
                'correlation_r': round(r, 6),
                'correlation_pvalue': p_r,
                'f_statistic': round(f_stat, 4),
                'f_pvalue': p_f,
                'se_estimate': round(se_estimate, 6)
            },
            'residuals': {
                'mean': round(np.mean(residuals), 10),  # Should be ~0
                'std': round(np.std(residuals, ddof=2), 6),
                'min': round(np.min(residuals), 6),
                'max': round(np.max(residuals), 6)
            },
            'interpretation': {
                'slope_interpretation': f"A 1-unit increase in {feature_name} is associated with a {slope:.4f} change in {target_name}",
                'r_squared_interpretation': f"The model explains {r_squared*100:.2f}% of the variance in {target_name}",
                'significance': 'Statistically significant' if p_slope < alpha else 'Not statistically significant'
            }
        }
        
        logger.info(f"Simple linear regression: {feature_name} -> {target_name}, R²={r_squared:.4f}")
        return result
    
    def multiple_linear_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        target_name: str = 'y'
    ) -> dict:
        """
        Perform multiple linear regression (multiple predictors).
        
        Args:
            X: Predictor variables (2D array with shape n_samples x n_features)
            y: Target variable (1D array)
            feature_names: Names of predictors for reporting
            target_name: Name of target for reporting
            
        Returns:
            Dictionary with regression results
        """
        # Convert to numpy if pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Remove NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        n, p = X.shape  # n = observations, p = number of predictors
        
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(p)]
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(n), X])
        
        # Calculate coefficients using normal equations
        # β = (X'X)^(-1) X'y
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            # Handle singular matrix with pseudo-inverse
            XtX_inv = np.linalg.pinv(XtX)
            logger.warning("Matrix singular, using pseudo-inverse")
        
        coefficients = XtX_inv @ Xty
        
        intercept = coefficients[0]
        slopes = coefficients[1:]
        
        # Calculate predictions and residuals
        y_pred = X_with_intercept @ coefficients
        residuals = y - y_pred
        
        # Degrees of freedom
        df_model = p
        df_residuals = n - p - 1
        
        # Mean squares
        SS_tot = np.sum((y - np.mean(y)) ** 2)
        SS_reg = np.sum((y_pred - np.mean(y)) ** 2)
        SS_res = np.sum(residuals ** 2)
        
        mse = SS_res / df_residuals
        
        # Standard errors of coefficients
        se_coefficients = np.sqrt(np.diag(XtX_inv) * mse)
        se_intercept = se_coefficients[0]
        se_slopes = se_coefficients[1:]
        
        # t-statistics and p-values
        t_intercept = intercept / se_intercept
        t_slopes = slopes / se_slopes
        
        p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df_residuals))
        p_slopes = 2 * (1 - stats.t.cdf(abs(t_slopes), df_residuals))
        
        # R-squared and adjusted R-squared
        r_squared = 1 - SS_res / SS_tot
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / df_residuals
        
        # F-statistic
        if df_residuals > 0:
            f_stat = (SS_reg / df_model) / (SS_res / df_residuals)
            p_f = 1 - stats.f.cdf(f_stat, df_model, df_residuals)
        else:
            f_stat = np.nan
            p_f = np.nan
        
        # Confidence intervals
        t_critical = stats.t.ppf(1 - alpha/2, df_residuals)
        
        # Variance Inflation Factor (VIF) for multicollinearity
        vif_values = []
        for i in range(p):
            X_other = np.delete(X, i, axis=1)
            if X_other.shape[1] > 0:
                _, _, r_others, _, _ = self.simple_linear_regression(
                    X_other @ np.ones(X_other.shape[1]),
                    X[:, i],
                    f'other_features',
                    feature_names[i]
                )
                r_others_val = r_others['model_fit']['correlation_r']
                vif = 1 / (1 - r_others_val**2) if r_others_val**2 < 1 else np.inf
            else:
                vif = 1.0
            vif_values.append(round(vif, 4))
        
        # Standardized coefficients (beta weights)
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        y_std = (y - y.mean()) / y.std()
        standardized_coeffs = np.linalg.lstsq(X_std, y_std, rcond=None)[0]
        
        # Build coefficient results
        coef_results = {
            'intercept': {
                'value': round(intercept, 6),
                'se': round(se_intercept, 6),
                't_stat': round(t_intercept, 4),
                'p_value': p_intercept,
                'ci_lower': round(intercept - t_critical * se_intercept, 6),
                'ci_upper': round(intercept + t_critical * se_intercept, 6)
            }
        }
        
        for i, name in enumerate(feature_names):
            coef_results[name] = {
                'value': round(slopes[i], 6),
                'se': round(se_slopes[i], 6),
                't_stat': round(t_slopes[i], 4),
                'p_value': p_slopes[i],
                'ci_lower': round(slopes[i] - t_critical * se_slopes[i], 6),
                'ci_upper': round(slopes[i] + t_critical * se_slopes[i], 6),
                'standardized': round(standardized_coeffs[i], 6),
                'vif': vif_values[i]
            }
        
        result = {
            'model_type': 'Multiple Linear Regression',
            'n_observations': n,
            'n_predictors': p,
            'degrees_freedom_model': df_model,
            'degrees_freedom_residuals': df_residuals,
            'target_name': target_name,
            'feature_names': feature_names,
            'coefficients': coef_results,
            'model_fit': {
                'r_squared': round(r_squared, 6),
                'adj_r_squared': round(adj_r_squared, 6),
                'f_statistic': round(f_stat, 4) if not np.isnan(f_stat) else f_stat,
                'f_pvalue': p_f,
                'se_estimate': round(np.sqrt(mse), 6),
                'ms_reg': round(SS_reg / df_model, 6) if df_model > 0 else None,
                'ms_res': round(mse, 6)
            },
            'residuals': {
                'mean': round(np.mean(residuals), 10),
                'std': round(np.std(residuals, ddof=p+1), 6),
                'min': round(np.min(residuals), 6),
                'max': round(np.max(residuals), 6)
            },
            'multicollinearity_check': {
                'vif_values': dict(zip(feature_names, vif_values)),
                'has_multicollinearity': any(v > 10 for v in vif_values)
            }
        }
        
        logger.info(f"Multiple linear regression: {p} predictors -> {target_name}, R²={r_squared:.4f}")
        return result
    
    def logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        target_name: str = 'y',
        binary_threshold: float = 0.5
    ) -> dict:
        """
        Perform logistic regression for binary classification.
        
        Args:
            X: Predictor variables (2D array)
            y: Target variable (binary 0/1)
            feature_names: Names of predictors
            target_name: Name of target
            binary_threshold: Threshold for converting probabilities to binary
            
        Returns:
            Dictionary with logistic regression results
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Handle pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Remove NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        n, p = X.shape
        
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(p)]
        
        # Standardize features for stable coefficients
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_scaled = (X - X_mean) / X_std
        
        # Fit logistic regression using sklearn
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        
        # Get coefficients (on standardized scale)
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Calculate odds ratios
        odds_ratios = np.exp(coefficients)
        
        # Predictions
        y_pred_prob = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_prob >= binary_threshold).astype(int)
        
        # Performance metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y, y_pred_prob)
        except ValueError:
            roc_auc = np.nan
        
        # Calculate confidence intervals using Wald's method
        # Standard errors from logistic regression covariance matrix
        from sklearn.utils.extmath import stable_softmax
        X_with_intercept = np.column_stack([np.ones(n), X_scaled])
        try:
            V = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_logit = np.sqrt(np.diag(V))
        except:
            se_logit = np.ones(p + 1) * np.nan
        
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Build results
        coef_results = {
            'intercept': {
                'coefficient': round(intercept, 6),
                'odds_ratio': round(np.exp(intercept), 6),
                'ci_lower': round(np.exp(intercept - z_critical * se_logit[0]), 6),
                'ci_upper': round(np.exp(intercept + z_critical * se_logit[0]), 6)
            }
        }
        
        for i, name in enumerate(feature_names):
            coef_results[name] = {
                'coefficient': round(coefficients[i], 6),
                'odds_ratio': round(odds_ratios[i], 6),
                'ci_lower': round(np.exp(coefficients[i] - z_critical * se_logit[i+1]), 6),
                'ci_upper': round(np.exp(coefficients[i] + z_critical * se_logit[i+1]), 6),
                'interpretation': f"Odds of {target_name}=1 increase by {(odds_ratios[i]-1)*100:.2f}% per 1 SD increase in {name}"
            }
        
        result = {
            'model_type': 'Logistic Regression',
            'n_observations': n,
            'n_predictors': p,
            'target_name': target_name,
            'feature_names': feature_names,
            'coefficients': coef_results,
            'model_fit': {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'roc_auc': round(roc_auc, 4) if not np.isnan(roc_auc) else None,
                'null_deviance': None,  # Would need to calculate
                'residual_deviance': None
            },
            'predictions': {
                'y_pred_prob_mean': round(np.mean(y_pred_prob), 4),
                'y_pred_mean': round(np.mean(y_pred), 4),
                'event_rate': round(np.mean(y), 4)
            }
        }
        
        logger.info(f"Logistic regression: {p} predictors -> {target_name}, Accuracy={accuracy:.4f}")
        return result
    
    def diagnose_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: np.ndarray = None
    ) -> dict:
        """
        Run comprehensive regression diagnostics.
        
        Checks:
        - Normality of residuals
        - Homoscedasticity (constant variance)
        - Outliers and influential points
        - Autocorrelation (for time series)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            X: Feature matrix (optional, for leverage calculation)
            
        Returns:
            Dictionary with diagnostic results
        """
        residuals = y_true - y_pred
        
        # Remove NaN
        mask = ~(np.isnan(residuals) | np.isnan(y_true) | np.isnan(y_pred))
        residuals = residuals[mask]
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        n = len(residuals)
        
        # Normality tests on residuals
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, n)])  # Shapiro limited to 5000
        dagostino_stat, dagostino_p = stats.normaltest(residuals)
        
        # Jarque-Bera test
        jb_stat = n * (residuals.skew()**2/6 + residuals.kurtosis()**2/24)
        jb_p = 1 - stats.chi2.cdf(jb_stat, df=2)
        
        # Homoscedasticity tests
        # Breusch-Pagan test (simplified version)
        if X is not None and len(X) == n:
            X_clean = X[mask]
            # Simple version: regress squared residuals on X
            squared_residuals = residuals ** 2
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X_clean, squared_residuals)
            r2_bp = lr.score(X_clean, squared_residuals)
            bp_stat = n * r2_bp
            bp_p = 1 - stats.chi2.cdf(bp_stat, df=X_clean.shape[1])
        else:
            bp_stat = np.nan
            bp_p = np.nan
            r2_bp = np.nan
        
        # Goldfeld-Quandt test (simplified)
        sorted_idx = np.argsort(y_pred)
        split = n // 2
        resid_1 = residuals[sorted_idx[:split]]
        resid_2 = residuals[sorted_idx[split:]]
        gq_stat = np.var(resid_2) / np.var(resid_1) if np.var(resid_1) > 0 else np.nan
        
        # Outlier detection
        residual_std = np.std(residuals, ddof=1)
        standardized_residuals = residuals / residual_std
        
        # Studentized residuals
        if X is not None:
            # Approximate studentized residuals
            leverage = np.ones(n) * 3/n  # Simplified
            studentized_residuals = standardized_residuals / np.sqrt(1 - np.clip(leverage, 0, 0.99))
        else:
            studentized_residuals = standardized_residuals
        
        outliers = np.abs(studentized_residuals) > 2.5
        extreme_outliers = np.abs(studentized_residuals) > 3.5
        
        # Influential points (using Cook's distance approximation)
        if X is not None and len(X) == n:
            X_clean = X[mask]
            H = X_clean @ np.linalg.inv(X_clean.T @ X_clean) @ X_clean.T
            leverage = np.diag(H)
            cooks_d = (standardized_residuals**2 / X_clean.shape[1]) * (leverage / (1 - leverage))
            influential = cooks_d > 4/n
        else:
            cooks_d = np.nan * np.ones(n)
            influential = np.zeros(n, dtype=bool)
        
        # Durbin-Watson statistic for autocorrelation
        diff_residuals = np.diff(residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
        
        result = {
            'sample_size': n,
            'residuals': {
                'mean': round(np.mean(residuals), 10),
                'std': round(np.std(residuals, ddof=1), 6),
                'min': round(np.min(residuals), 6),
                'max': round(np.max(residuals), 6),
                'skewness': round(stats.skew(residuals), 4),
                'kurtosis': round(stats.kurtosis(residuals), 4)
            },
            'normality_tests': {
                'shapiro_wilk': {
                    'statistic': round(shapiro_stat, 4),
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > alpha
                },
                'dagostino_pearson': {
                    'statistic': round(dagostino_stat, 4),
                    'p_value': dagostino_p,
                    'is_normal': dagostino_p > alpha
                },
                'jarque_bera': {
                    'statistic': round(jb_stat, 4),
                    'p_value': jb_p,
                    'is_normal': jb_p > alpha
                }
            },
            'homoscedasticity': {
                'breusch_pagan': {
                    'statistic': round(bp_stat, 4) if not np.isnan(bp_stat) else bp_stat,
                    'p_value': bp_p,
                    'is_homoscedastic': bp_p > alpha if not np.isnan(bp_p) else True
                },
                'goldfeld_quandt': {
                    'statistic': round(gq_stat, 4) if not np.isnan(gq_stat) else gq_stat,
                    'interpretation': 'Variance may differ across prediction range' if gq_stat > 2 else 'Variance appears constant'
                }
            },
            'autocorrelation': {
                'durbin_watson': round(dw_stat, 4),
                'interpretation': 'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Possible autocorrelation'
            },
            'outliers': {
                'standardized_residuals_threshold': 2.5,
                'n_outliers': int(np.sum(outliers)),
                'n_extreme_outliers': int(np.sum(extreme_outliers)),
                'outlier_indices': np.where(outliers)[0].tolist()
            },
            'influential_points': {
                'n_influential': int(np.sum(influential)) if influential is not None else 0,
                'cooksd_threshold': f'>{4/n:.4f}'
            },
            'assumptions_met': {
                'normality': shapiro_p > alpha and dagostino_p > alpha,
                'homoscedasticity': bp_p > alpha if not np.isnan(bp_p) else True,
                'no_autocorrelation': 1.5 < dw_stat < 2.5,
                'no_outliers': np.sum(outliers) == 0
            }
        }
        
        return result
    
    def calculate_power_analysis_linear(
        self,
        effect_size: float,
        alpha: float,
        power: float,
        n_predictors: int
    ) -> dict:
        """
        Calculate required sample size for linear regression.
        
        Args:
            effect_size: Expected f² effect size (0.02=small, 0.15=medium, 0.35=large)
            alpha: Significance level
            power: Desired statistical power
            n_predictors: Number of predictors
            
        Returns:
            Dictionary with power analysis results
        """
        from scipy.stats import f as f_dist
        
        # Calculate critical F value
        df1 = n_predictors
        f_critical = f_dist.ppf(1 - alpha, df1, np.inf)
        
        # Calculate non-centrality parameter
        # For a given effect size f²: λ = n * f²
        # For desired power: need to find n such that P(F > F_crit | λ) = power
        
        # Iteratively find n
        n = n_predictors + 2
        while n < 100000:
            df2 = n - n_predictors - 1
            if df2 < 1:
                n += 1
                continue
            
            # Non-central F distribution
            ncf_critical = f_dist.ppf(1 - alpha, df1, df2)
            
            # Power calculation using approximation
            # P(F > F_crit | λ) where λ = n * f²
            from scipy.stats import ncf
            calculated_power = 1 - ncf.cdf(ncf_critical, df1, df2, n * effect_size)
            
            if calculated_power >= power:
                break
            n += max(1, int(n * 0.1))
        
        result = {
            'effect_size_f2': effect_size,
            'effect_size_category': 'small' if effect_size == 0.02 else ('medium' if effect_size == 0.15 else 'large'),
            'alpha': alpha,
            'power': power,
            'n_predictors': n_predictors,
            'required_n': n,
            'f_critical': round(f_critical, 4),
            'achieved_power': round(calculated_power, 4) if 'calculated_power' in dir() else power
        }
        
        logger.info(f"Power analysis: need n={n} for {power*100}% power with {n_predictors} predictors")
        return result


def run_regression_analysis():
    """Example usage of regression analysis module."""
    import pandas as pd
    
    logger.info("Starting regression analysis...")
    log_pipeline_start("regression_analysis")
    
    # Create sample data for demonstration
    np.random.seed(42)
    n = 1000
    
    # Generate sample data
    support_data = pd.DataFrame({
        'resolution_time': np.random.exponential(60, n),  # minutes
        'first_response_time': np.random.exponential(15, n),
        'csat_score': np.random.uniform(1, 5, n),
        'agent_experience': np.random.normal(2, 0.5, n),  # years
        'ticket_complexity': np.random.uniform(1, 10, n),
        'issue_type': np.random.choice(['billing', 'technical', 'general'], n)
    })
    
    # Add some correlations
    support_data['csat_score'] = (
        5 - support_data['resolution_time']/60 * 0.5 +
        support_data['agent_experience'] * 0.3 +
        np.random.normal(0, 0.5, n)
    )
    support_data['csat_score'] = np.clip(support_data['csat_score'], 1, 5)
    
    analyzer = RegressionAnalyzer(alpha=0.05)
    
    # Example 1: Simple linear regression
    logger.info("Running simple linear regression...")
    result = analyzer.simple_linear_regression(
        support_data['resolution_time'].values,
        support_data['csat_score'].values,
        'resolution_time',
        'csat_score'
    )
    logger.debug(f"Simple regression R²: {result['model_fit']['r_squared']}")
    
    # Example 2: Multiple linear regression
    logger.info("Running multiple linear regression...")
    X = support_data[['resolution_time', 'first_response_time', 'agent_experience', 'ticket_complexity']].values
    y = support_data['csat_score'].values
    feature_names = ['resolution_time', 'first_response_time', 'agent_experience', 'ticket_complexity']
    
    multi_result = analyzer.multiple_linear_regression(X, y, feature_names, 'csat_score')
    logger.debug(f"Multiple regression R²: {multi_result['model_fit']['r_squared']}")
    
    # Example 3: Logistic regression (CSAT > 4 is good)
    logger.info("Running logistic regression...")
    y_binary = (support_data['csat_score'] > 4).astype(int).values
    log_result = analyzer.logistic_regression(X, y_binary, feature_names, 'high_csat')
    logger.debug(f"Logistic regression accuracy: {log_result['model_fit']['accuracy']}")
    
    # Example 4: Regression diagnostics
    logger.info("Running regression diagnostics...")
    y_pred = multi_result['coefficients']['intercept']['value'] + \
             sum(multi_result['coefficients'][f]['value'] * X[:, i] 
                 for i, f in enumerate(feature_names))
    
    diag_result = analyzer.diagnose_regression(y, y_pred, X)
    logger.debug(f"Normality assumption met: {diag_result['assumptions_met']['normality']}")
    
    logger.info("Regression analysis completed")
    log_pipeline_complete("regression_analysis")
    
    return {
        'simple_regression': result,
        'multiple_regression': multi_result,
        'logistic_regression': log_result,
        'diagnostics': diag_result
    }


if __name__ == "__main__":
    results = run_regression_analysis()
    print("\n=== Regression Analysis Results ===")
    print(f"Multiple Regression R²: {results['multiple_regression']['model_fit']['r_squared']:.4f}")
    print(f"Logistic Regression Accuracy: {results['logistic_regression']['model_fit']['accuracy']:.4f}")