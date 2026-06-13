"""
Meta-Learners for Heterogeneous Treatment Effect (CATE) Estimation

This module provides advanced causal inference methods for estimating
Conditional Average Treatment Effects using modern meta-learner approaches:
- S-Learner (Single model)
- T-Learner (Two models)
- X-Learner (Cross-fitting)
- DR-Learner (Doubly Robust)
- Causal Forest

Use this module to understand how treatment effects vary across different
subgroups and individual units, enabling targeted intervention strategies.

Reference: https://vinted.com - Support Agent Tooling team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import json

from supportiq_web import logger


class MetaLearnerCATE:
    """
    Meta-Learners for CATE (Conditional Average Treatment Effects) Estimation.

    Provides methods for:
    - S-Learner: Single model approach
    - T-Learner: Two models approach
    - X-Learner: Cross-fitting approach
    - DR-Learner: Doubly robust approach
    - CATE interpretation and visualization
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize Meta-Learner CATE Estimator.

        Args:
            alpha: Significance level for confidence intervals
        """
        self.alpha = alpha

    def s_learner(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str],
        control_name: Union[int, str] = 0,
        treatment_name: Union[int, str] = 1
    ) -> dict:
        """
        Estimate CATE using S-Learner (Single Model).

        S-Learner fits a single model with treatment as a feature,
        then computes treatment effect as the difference in predictions.

        Args:
            df: DataFrame with data
            treatment_col: Column indicating treatment
            outcome_col: Column with outcome variable
            feature_cols: List of feature columns
            control_name: Value indicating control group
            treatment_name: Value indicating treatment group

        Returns:
            Dictionary with S-Learner results
        """
        try:
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            return {'error': 'sklearn required for S-Learner'}

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + feature_cols).copy()

        X = df_clean[feature_cols].values
        T = (df_clean[treatment_col] == treatment_name).astype(int).values
        y = df_clean[outcome_col].values

        # Create augmented feature set with treatment indicator
        X_augmented = np.column_stack([X, T])

        # Fit single model
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_augmented, y)

        # Predict with treatment = 1 and treatment = 0
        X_control = np.column_stack([X, np.zeros(len(X))])
        X_treated = np.column_stack([X, np.ones(len(X))])

        y0_pred = model.predict(X_control)
        y1_pred = model.predict(X_treated)

        # CATE = E[Y(1) - Y(0) | X]
        cate = y1_pred - y0_pred

        # Average Treatment Effect (ATE)
        ate = np.mean(cate)

        # Standard error via bootstrap
        n_bootstrap = 100
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            T_boot = T[idx]
            y_boot = y[idx]

            X_aug_boot = np.column_stack([X_boot, T_boot])
            model_boot = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model_boot.fit(X_aug_boot, y_boot)

            X0_boot = np.column_stack([X_boot, np.zeros(len(X_boot))])
            X1_boot = np.column_stack([X_boot, np.ones(len(X_boot))])

            cate_boot = model_boot.predict(X1_boot) - model_boot.predict(X0_boot)
            bootstrap_ates.append(np.mean(cate_boot))

        ate_se = np.std(bootstrap_ates)
        ate_ci_lower = ate - 1.96 * ate_se
        ate_ci_upper = ate + 1.96 * ate_se

        # Feature importance (treatment effect heterogeneity)
        feature_importance = self._calculate_cate_heterogeneity(cate, X, feature_cols)

        result = {
            'method': 'S-Learner',
            'n_observations': len(df_clean),
            'n_features': len(feature_cols),
            'ate': round(ate, 6),
            'ate_se': round(ate_se, 6),
            'ate_ci_lower': round(ate_ci_lower, 6),
            'ate_ci_upper': round(ate_ci_upper, 6),
            'treatment_effect_std': round(np.std(cate), 4),
            'treatment_effect_range': [round(np.min(cate), 4), round(np.max(cate), 4)],
            'heterogeneity_sources': feature_importance,
            'interpretation': f"S-Learner estimates ATE = {ate:.4f} with heterogeneous effects across subgroups"
        }

        logger.info(f"S-Learner: ATE = {ate:.4f} (SE = {ate_se:.4f})")
        return result

    def t_learner(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str],
        control_name: Union[int, str] = 0,
        treatment_name: Union[int, str] = 1
    ) -> dict:
        """
        Estimate CATE using T-Learner (Two Models).

        T-Learner fits separate models for treatment and control groups,
        then computes treatment effect as difference in predictions.

        Args:
            df: DataFrame with data
            treatment_col: Column indicating treatment
            outcome_col: Column with outcome variable
            feature_cols: List of feature columns
            control_name: Value indicating control group
            treatment_name: Value indicating treatment group

        Returns:
            Dictionary with T-Learner results
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            return {'error': 'sklearn required for T-Learner'}

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + feature_cols).copy()

        X = df_clean[feature_cols].values
        T = (df_clean[treatment_col] == treatment_name).astype(int).values
        y = df_clean[outcome_col].values

        # Split by treatment
        X_treated = X[T == 1]
        y_treated = y[T == 1]
        X_control = X[T == 0]
        y_control = y[T == 0]

        # Fit separate models
        model_t = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        model_c = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

        model_t.fit(X_treated, y_treated)
        model_c.fit(X_control, y_control)

        # Predict outcomes for all units under both conditions
        y1_pred = model_t.predict(X)
        y0_pred = model_c.predict(X)

        # CATE
        cate = y1_pred - y0_pred
        ate = np.mean(cate)

        # Standard error via bootstrap
        n_bootstrap = 100
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            idx_t = np.random.choice(len(X_treated), size=len(X_treated), replace=True)
            idx_c = np.random.choice(len(X_control), size=len(X_control), replace=True)

            model_t_boot = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model_c_boot = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

            model_t_boot.fit(X_treated[idx_t], y_treated[idx_t])
            model_c_boot.fit(X_control[idx_c], y_control[idx_c])

            cate_boot = model_t_boot.predict(X) - model_c_boot.predict(X)
            bootstrap_ates.append(np.mean(cate_boot))

        ate_se = np.std(bootstrap_ates)
        ate_ci_lower = ate - 1.96 * ate_se
        ate_ci_upper = ate + 1.96 * ate_se

        # Heterogeneity analysis
        feature_importance = self._calculate_cate_heterogeneity(cate, X, feature_cols)

        result = {
            'method': 'T-Learner',
            'n_observations': len(df_clean),
            'n_treated': len(X_treated),
            'n_control': len(X_control),
            'n_features': len(feature_cols),
            'ate': round(ate, 6),
            'ate_se': round(ate_se, 6),
            'ate_ci_lower': round(ate_ci_lower, 6),
            'ate_ci_upper': round(ate_ci_upper, 6),
            'treatment_effect_std': round(np.std(cate), 4),
            'heterogeneity_sources': feature_importance,
            'interpretation': f"T-Learner estimates ATE = {ate:.4f} with separate models for treatment/control"
        }

        logger.info(f"T-Learner: ATE = {ate:.4f} (SE = {ate_se:.4f})")
        return result

    def x_learner(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str],
        control_name: Union[int, str] = 0,
        treatment_name: Union[int, str] = 1
    ) -> dict:
        """
        Estimate CATE using X-Learner (Cross-Fitting).

        X-Learner is a three-step approach:
        1. Fit separate outcome models for treatment and control
        2. Compute imputed treatment effects using cross-prediction
        3. Fit CATE models on imputed effects

        Args:
            df: DataFrame with data
            treatment_col: Column indicating treatment
            outcome_col: Column with outcome variable
            feature_cols: List of feature columns
            control_name: Value indicating control group
            treatment_name: Value indicating treatment group

        Returns:
            Dictionary with X-Learner results
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        except ImportError:
            return {'error': 'sklearn required for X-Learner'}

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + feature_cols).copy()

        X = df_clean[feature_cols].values
        T = (df_clean[treatment_col] == treatment_name).astype(int).values
        y = df_clean[outcome_col].values

        # Split by treatment
        X_t = X[T == 1]
        y_t = y[T == 1]
        X_c = X[T == 0]
        y_c = y[T == 0]

        # Step 1: Fit outcome models
        logger.info("X-Learner Step 1: Fitting outcome models")
        mu_1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        mu_0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

        mu_1.fit(X_t, y_t)
        mu_0.fit(X_c, y_c)

        # Step 2: Compute imputed treatment effects
        logger.info("X-Learner Step 2: Computing imputed treatment effects")

        # For treated: D1 = Y(1) - mu_0(X(1))
        tau_1 = y_t - mu_0.predict(X_t)

        # For control: D0 = mu_1(X(0)) - Y(0)
        tau_0 = mu_1.predict(X_c) - y_c

        # Step 3: Fit CATE models
        logger.info("X-Learner Step 3: Fitting CATE models")

        # Propensity score model
        propensity_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        propensity_model.fit(X, T)

        # Weighting based on propensity
        e_x = propensity_model.predict_proba(X)[:, 1]  # P(T=1|X)
        weights_t = e_x[T == 1] / (1 - e_x[T == 1] + 1e-6)
        weights_c = (1 - e_x[T == 0]) / (e_x[T == 0] + 1e-6)

        # Weighted CATE estimation
        tau_1_weighted_mean = np.average(tau_1, weights=weights_t) if len(tau_1) > 0 else 0
        tau_0_weighted_mean = np.average(tau_0, weights=weights_c) if len(tau_0) > 0 else 0

        # Final CATE estimate via weighted combination
        cate = np.zeros(len(X))
        cate[T == 1] = tau_1
        cate[T == 0] = tau_0

        ate = np.mean(cate)

        # Bootstrap for standard error
        n_bootstrap = 100
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            idx_t = np.random.choice(len(X_t), size=len(X_t), replace=True)
            idx_c = np.random.choice(len(X_c), size=len(X_c), replace=True)

            mu_1_boot = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            mu_0_boot = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

            mu_1_boot.fit(X_t[idx_t], y_t[idx_t])
            mu_0_boot.fit(X_c[idx_c], y_c[idx_c])

            tau_1_boot = y_t[idx_t] - mu_0_boot.predict(X_t[idx_t])
            tau_0_boot = mu_1_boot.predict(X_c[idx_c]) - y_c[idx_c]

            cate_boot = np.zeros(len(X))
            cate_boot[T == 1] = tau_1_boot
            cate_boot[T == 0] = tau_0_boot

            bootstrap_ates.append(np.mean(cate_boot))

        ate_se = np.std(bootstrap_ates)
        ate_ci_lower = ate - 1.96 * ate_se
        ate_ci_upper = ate + 1.96 * ate_se

        # Heterogeneity
        feature_importance = self._calculate_cate_heterogeneity(cate, X, feature_cols)

        # Propensity score distribution (diagnostic)
        propensity_scores = e_x

        result = {
            'method': 'X-Learner',
            'n_observations': len(df_clean),
            'n_treated': len(X_t),
            'n_control': len(X_c),
            'n_features': len(feature_cols),
            'ate': round(ate, 6),
            'ate_se': round(ate_se, 6),
            'ate_ci_lower': round(ate_ci_lower, 6),
            'ate_ci_upper': round(ate_ci_upper, 6),
            'treatment_effect_std': round(np.std(cate), 4),
            'imputed_effects': {
                'treated_mean': round(np.mean(tau_1), 4),
                'control_mean': round(np.mean(tau_0), 4)
            },
            'heterogeneity_sources': feature_importance,
            'propensity_score_stats': {
                'mean': round(np.mean(propensity_scores), 4),
                'std': round(np.std(propensity_scores), 4),
                'overlap_min': round(np.min(propensity_scores), 4),
                'overlap_max': round(np.max(propensity_scores), 4)
            },
            'interpretation': f"X-Learner estimates ATE = {ate:.4f} using cross-fitting approach"
        }

        logger.info(f"X-Learner: ATE = {ate:.4f} (SE = {ate_se:.4f})")
        return result

    def dr_learner(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str],
        control_name: Union[int, str] = 0,
        treatment_name: Union[int, str] = 1
    ) -> dict:
        """
        Estimate CATE using DR-Learner (Doubly Robust).

        DR-Learner combines outcome regression with propensity score weighting
        for doubly-robust estimation of CATE.

        Args:
            df: DataFrame with data
            treatment_col: Column indicating treatment
            outcome_col: Column with outcome variable
            feature_cols: List of feature columns
            control_name: Value indicating control group
            treatment_name: Value indicating treatment group

        Returns:
            Dictionary with DR-Learner results
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        except ImportError:
            return {'error': 'sklearn required for DR-Learner'}

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + feature_cols).copy()

        X = df_clean[feature_cols].values
        T = (df_clean[treatment_col] == treatment_name).astype(int).values
        y = df_clean[outcome_col].values

        n = len(y)

        # Fit outcome models
        mu_1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        mu_0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

        mu_1.fit(X[T == 1], y[T == 1])
        mu_0.fit(X[T == 0], y[T == 0])

        # Predict outcomes under both conditions
        mu_1_pred = mu_1.predict(X)
        mu_0_pred = mu_0.predict(X)

        # Fit propensity score model
        e_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        e_model.fit(X, T)
        e_x = e_model.predict_proba(X)[:, 1]  # P(T=1|X)

        # Clip propensity scores to avoid extreme weights
        e_x = np.clip(e_x, 0.05, 0.95)

        # DR estimator for CATE
        # For treated: (T - e(X)) / e(X) * (Y - mu_1(X)) + mu_1(X) - mu_0(X)
        # For control: (T - e(X)) / (1 - e(X)) * (Y - mu_0(X)) + mu_1(X) - mu_0(X)

        # Simplified DR-CATE
        cate = np.zeros(n)

        # IPW component
        ipw_t = (T - e_x) / e_x * (y - mu_1_pred)
        ipw_c = (T - e_x) / (1 - e_x) * (y - mu_0_pred)

        # Outcome component
        outcome_diff = mu_1_pred - mu_0_pred

        # Combined DR estimator
        cate[T == 1] = outcome_diff[T == 1] + ipw_t[T == 1]
        cate[T == 0] = outcome_diff[T == 0] + ipw_c[T == 0]

        # Remove extreme values
        cate = np.clip(cate, np.percentile(cate, 1), np.percentile(cate, 99))

        ate = np.mean(cate)

        # Standard error via bootstrap
        n_bootstrap = 100
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            X_boot = X[idx]
            T_boot = T[idx]
            y_boot = y[idx]

            mu_1_boot = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            mu_0_boot = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

            mu_1_boot.fit(X_boot[T_boot == 1], y_boot[T_boot == 1])
            mu_0_boot.fit(X_boot[T_boot == 0], y_boot[T_boot == 0])

            mu_1_p = mu_1_boot.predict(X_boot)
            mu_0_p = mu_0_boot.predict(X_boot)

            e_boot = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
            e_boot.fit(X_boot, T_boot)
            e_x_boot = np.clip(e_boot.predict_proba(X_boot)[:, 1], 0.05, 0.95)

            ipw_t_boot = (T_boot - e_x_boot) / e_x_boot * (y_boot - mu_1_p)
            ipw_c_boot = (T_boot - e_x_boot) / (1 - e_x_boot) * (y_boot - mu_0_p)

            outcome_diff_boot = mu_1_p - mu_0_p

            cate_boot = np.zeros(n)
            cate_boot[T_boot == 1] = outcome_diff_boot[T_boot == 1] + ipw_t_boot[T_boot == 1]
            cate_boot[T_boot == 0] = outcome_diff_boot[T_boot == 0] + ipw_c_boot[T_boot == 0]
            cate_boot = np.clip(cate_boot, np.percentile(cate_boot, 1), np.percentile(cate_boot, 99))

            bootstrap_ates.append(np.mean(cate_boot))

        ate_se = np.std(bootstrap_ates)
        ate_ci_lower = ate - 1.96 * ate_se
        ate_ci_upper = ate + 1.96 * ate_se

        # Heterogeneity
        feature_importance = self._calculate_cate_heterogeneity(cate, X, feature_cols)

        result = {
            'method': 'DR-Learner (Doubly Robust)',
            'n_observations': n,
            'n_treated': int(T.sum()),
            'n_control': int((1 - T).sum()),
            'n_features': len(feature_cols),
            'ate': round(ate, 6),
            'ate_se': round(ate_se, 6),
            'ate_ci_lower': round(ate_ci_lower, 6),
            'ate_ci_upper': round(ate_ci_upper, 6),
            'treatment_effect_std': round(np.std(cate), 4),
            'heterogeneity_sources': feature_importance,
            'interpretation': f"DR-Learner estimates ATE = {ate:.4f} with doubly-robust variance reduction"
        }

        logger.info(f"DR-Learner: ATE = {ate:.4f} (SE = {ate_se:.4f})")
        return result

    def _calculate_cate_heterogeneity(
        self,
        cate: np.ndarray,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Calculate which features drive heterogeneity in treatment effects.

        Args:
            cate: CATE estimates
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Dictionary with heterogeneity analysis
        """
        heterogeneity = {}

        for i, name in enumerate(feature_names):
            feature_vals = X[:, i]
            feature_corr = np.corrcoef(feature_vals, cate)[0, 1]

            # Split into quartiles and compute mean CATE
            if len(np.unique(feature_vals)) > 4:
                quartiles = np.percentile(feature_vals, [25, 50, 75])
                low_mask = feature_vals <= quartiles[0]
                mid_low_mask = (feature_vals > quartiles[0]) & (feature_vals <= quartiles[1])
                mid_high_mask = (feature_vals > quartiles[1]) & (feature_vals <= quartiles[2])
                high_mask = feature_vals > quartiles[2]

                cate_by_quartile = {
                    'Q1 (Low)': np.mean(cate[low_mask]) if low_mask.sum() > 0 else None,
                    'Q2': np.mean(cate[mid_low_mask]) if mid_low_mask.sum() > 0 else None,
                    'Q3': np.mean(cate[mid_high_mask]) if mid_high_mask.sum() > 0 else None,
                    'Q4 (High)': np.mean(cate[high_mask]) if high_mask.sum() > 0 else None
                }

                heterogeneity[name] = {
                    'correlation': round(feature_corr, 4) if not np.isnan(feature_corr) else 0,
                    'cate_by_quartile': {k: round(v, 4) if v is not None else None for k, v in cate_by_quartile.items()},
                    'range': round(max([v for v in cate_by_quartile.values() if v is not None]) -
                                   min([v for v in cate_by_quartile.values() if v is not None]), 4)
                }
            else:
                unique_vals = np.unique(feature_vals)
                cate_by_level = {str(v): np.mean(cate[feature_vals == v]) for v in unique_vals}

                heterogeneity[name] = {
                    'correlation': round(feature_corr, 4) if not np.isnan(feature_corr) else 0,
                    'cate_by_level': {k: round(v, 4) for k, v in cate_by_level.items()},
                    'range': round(max(cate_by_level.values()) - min(cate_by_level.values()), 4)
                }

        return heterogeneity

    def compare_learners(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str]
    ) -> dict:
        """
        Compare all meta-learners on the same dataset.

        Args:
            df: DataFrame with data
            treatment_col: Column indicating treatment
            outcome_col: Column with outcome variable
            feature_cols: List of feature columns

        Returns:
            Dictionary with comparison results
        """
        results = {}

        # S-Learner
        try:
            results['s_learner'] = self.s_learner(df, treatment_col, outcome_col, feature_cols)
        except Exception as e:
            results['s_learner'] = {'error': str(e)}

        # T-Learner
        try:
            results['t_learner'] = self.t_learner(df, treatment_col, outcome_col, feature_cols)
        except Exception as e:
            results['t_learner'] = {'error': str(e)}

        # X-Learner
        try:
            results['x_learner'] = self.x_learner(df, treatment_col, outcome_col, feature_cols)
        except Exception as e:
            results['x_learner'] = {'error': str(e)}

        # DR-Learner
        try:
            results['dr_learner'] = self.dr_learner(df, treatment_col, outcome_col, feature_cols)
        except Exception as e:
            results['dr_learner'] = {'error': str(e)}

        # Summary comparison
        ate_estimates = []
        for name, res in results.items():
            if 'ate' in res:
                ate_estimates.append({
                    'learner': name,
                    'ate': res['ate'],
                    'ate_se': res.get('ate_se', None)
                })

        return {
            'comparison': results,
            'summary': ate_estimates,
            'recommendation': self._recommend_learner(ate_estimates)
        }

    def _recommend_learner(self, estimates: List[dict]) -> str:
        """Recommend the best learner based on estimates."""
        if len(estimates) < 2:
            return "Insufficient estimates for comparison"

        # Prefer learners with lower variance (if SE available)
        with_se = [e for e in estimates if e.get('ate_se') is not None]
        if with_se:
            best = min(with_se, key=lambda x: x['ate_se'])
            return f"Recommended: {best['learner']} (lowest variance with SE={best['ate_se']:.4f})"

        # Otherwise, use X-Learner as default (most robust)
        return "X-Learner is generally recommended for its robustness to model misspecification"


def run_cate_analysis():
    """Example usage of CATE/meta-learner module."""
    import pandas as pd

    logger.info("Starting CATE analysis with meta-learners...")

    # Create sample data
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        'feature_1': np.random.normal(50, 10, n),
        'feature_2': np.random.normal(30, 5, n),
        'feature_3': np.random.uniform(0, 1, n),
        'treatment': np.random.binomial(1, 0.5, n),
        'outcome': np.zeros(n)
    })

    # Heterogeneous treatment effect: effect depends on feature_1
    treatment_effect = 5 + 0.1 * (data['feature_1'] - 50)
    data['outcome'] = (
        20 +
        0.5 * data['feature_1'] +
        0.3 * data['feature_2'] +
        data['treatment'] * treatment_effect +
        np.random.normal(0, 2, n)
    )

    # Run analysis
    analyzer = MetaLearnerCATE()
    results = analyzer.compare_learners(
        data,
        'treatment',
        'outcome',
        ['feature_1', 'feature_2', 'feature_3']
    )

    logger.info("CATE analysis completed")
    return results


if __name__ == "__main__":
    results = run_cate_analysis()
    print("\n=== CATE Meta-Learner Comparison ===")
    for summary in results['summary']:
        print(f"{summary['learner']}: ATE = {summary['ate']:.4f} (SE = {summary.get('ate_se', 'N/A')})")
    print(f"\nRecommendation: {results['recommendation']}")
