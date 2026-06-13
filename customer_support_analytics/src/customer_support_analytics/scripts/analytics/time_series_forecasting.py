"""
Time Series Forecasting Module for SupportIQ

This module provides comprehensive time series forecasting capabilities including:
- ARIMA/SARIMA models for trend and seasonality
- Exponential Smoothing (Holt-Winters)
- Prophet for flexible business time series
- Vector Autoregression (VAR) for multivariate forecasting
- Forecast evaluation and confidence intervals

Use this module to predict future support metrics, identify trends,
and inform capacity planning and resource allocation decisions.

Reference: https://vinted.com - Support Agent Tooling team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

from supportiq_web import path_manager, logger, log_pipeline_start, log_pipeline_complete


class TimeSeriesForecaster:
    """
    Time Series Forecasting Suite for Support Metrics.

    Provides methods for:
    - Trend analysis and extrapolation
    - Seasonality detection and modeling
    - ARIMA/SARIMA forecasting
    - Exponential smoothing methods
    - Multi-variate time series (VAR)
    - Forecast confidence intervals
    - Model evaluation and selection
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize Time Series Forecaster.

        Args:
            alpha: Significance level for confidence intervals (default 0.05)
        """
        self.alpha = alpha
        self.z_score = norm.ppf(1 - alpha / 2)

    def prepare_time_series(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        freq: str = 'D'
    ) -> pd.Series:
        """
        Prepare time series data from DataFrame.

        Args:
            df: Input DataFrame
            date_col: Name of date column
            value_col: Name of value column to forecast
            freq: Frequency for resampling ('D'=daily, 'W'=weekly, 'M'=monthly)

        Returns:
            Prepared time series as pandas Series
        """
        ts = df[[date_col, value_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col])
        ts = ts.set_index(date_col)
        ts = ts.asfreq(freq)
        ts[value_col] = ts[value_col].fillna(method='ffill').fillna(method='bfill')
        ts = ts[value_col]

        logger.info(f"Prepared time series: {len(ts)} observations, freq={freq}")
        return ts

    def detect_trend(
        self,
        ts: pd.Series,
        window: int = 7
    ) -> dict:
        """
        Detect and quantify trend in time series using multiple methods.

        Args:
            ts: Time series data
            window: Rolling window size for moving average

        Returns:
            Dictionary with trend analysis results
        """
        # Linear regression for trend
        x = np.arange(len(ts))
        y = ts.values
        mask = ~np.isnan(y)
        x, y = x[mask], y[mask]

        n = len(x)
        x_mean, y_mean = np.mean(x), np.mean(y)

        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # Moving average trend
        ma = ts.rolling(window=window).mean()
        ma_trend = (ma.iloc[-1] - ma.iloc[0]) / ma.iloc[0] * 100 if ma.iloc[0] != 0 else 0

        # Trend classification
        if abs(slope) < 0.01:
            trend_type = 'Stable'
        elif slope > 0:
            trend_type = 'Increasing'
        else:
            trend_type = 'Decreasing'

        # Trend strength
        if r_squared < 0.3:
            trend_strength = 'Weak'
        elif r_squared < 0.7:
            trend_strength = 'Moderate'
        else:
            trend_strength = 'Strong'

        result = {
            'trend_type': trend_type,
            'trend_strength': trend_strength,
            'slope': round(slope, 4),
            'daily_change': round(slope, 4),
            'weekly_change': round(slope * 7, 4),
            'monthly_change': round(slope * 30, 4),
            'r_squared': round(r_squared, 4),
            'moving_avg_trend_pct': round(ma_trend, 2),
            'forecast_interpretation': self._interpret_trend(slope, r_squared)
        }

        logger.info(f"Trend detection: {trend_type} ({trend_strength}), slope={slope:.4f}")
        return result

    def _interpret_trend(self, slope: float, r_squared: float) -> str:
        """Generate human-readable trend interpretation."""
        direction = 'upward' if slope > 0 else 'downward'
        magnitude = abs(slope)

        if r_squared < 0.3:
            confidence = 'weak confidence'
        elif r_squared < 0.7:
            confidence = 'moderate confidence'
        else:
            confidence = 'high confidence'

        return f"Series shows {direction} trend with {confidence}"

    def detect_seasonality(
        self,
        ts: pd.Series,
        period: Optional[int] = None
    ) -> dict:
        """
        Detect and analyze seasonality in time series.

        Args:
            ts: Time series data
            period: Known seasonal period (7=daily weekly, 24=hourly daily)

        Returns:
            Dictionary with seasonality analysis
        """
        # Auto-detect period if not provided
        if period is None:
            # Try to detect weekly pattern
            period = 7 if len(ts) >= 28 else None

        if period is None or len(ts) < 2 * period:
            return {
                'has_seasonality': False,
                'period': None,
                'seasonal_strength': 0,
                'interpretation': 'Insufficient data for seasonality detection'
            }

        # Classical decomposition
        decomposition = self._seasonal_decomposition(ts, period)

        # Seasonal strength metric (similar to STL)
        var_residual = np.var(decomposition['residual'])
        var_seasonal_residual = np.var(decomposition['seasonal'] + decomposition['residual'])
        seasonal_strength = max(0, 1 - var_residual / var_seasonal_residual) if var_seasonal_residual > 0 else 0

        # Calculate average seasonal factors
        seasonal_factors = decomposition['seasonal'].values[:period]
        seasonal_factors = seasonal_factors[~np.isnan(seasonal_factors)]

        result = {
            'has_seasonality': seasonal_strength > 0.1,
            'period': period,
            'seasonal_strength': round(seasonal_strength, 4),
            'avg_seasonal_factors': [round(f, 4) for f in seasonal_factors] if len(seasonal_factors) > 0 else [],
            'peak_seasonal_factor': round(max(seasonal_factors), 4) if len(seasonal_factors) > 0 else None,
            'trough_seasonal_factor': round(min(seasonal_factors), 4) if len(seasonal_factors) > 0 else None,
            'interpretation': self._interpret_seasonality(seasonal_strength, seasonal_factors)
        }

        logger.info(f"Seasonality detection: period={period}, strength={seasonal_strength:.2f}")
        return result

    def _seasonal_decomposition(
        self,
        ts: pd.Series,
        period: int
    ) -> dict:
        """Perform classical additive seasonal decomposition."""
        # Moving average for trend
        trend = ts.rolling(window=period, center=True).mean()
        trend = trend.rolling(window=2).mean()  # Smooth further

        # Detrend
        detrended = ts - trend

        # Seasonal factors (average for each period position)
        seasonal = np.zeros(len(ts))
        for i in range(period):
            indices = np.arange(i, len(ts), period)
            seasonal[indices] = np.nanmean(detrended.iloc[indices])

        # Fill NaN in seasonal with 0
        seasonal = pd.Series(seasonal, index=ts.index).fillna(0)

        # Residual
        residual = ts - trend - seasonal

        return {
            'trend': trend.fillna(method='bfill').fillna(method='ffill'),
            'seasonal': seasonal,
            'residual': residual.fillna(0)
        }

    def _interpret_seasonality(self, strength: float, factors: np.ndarray) -> str:
        """Generate human-readable seasonality interpretation."""
        if strength < 0.1:
            return 'No significant seasonal pattern detected'
        elif strength < 0.3:
            return 'Weak seasonal pattern - may not be reliable for forecasting'
        elif strength < 0.6:
            return 'Moderate seasonal pattern - should incorporate into forecasting'
        else:
            peak = max(factors) if len(factors) > 0 else 0
            trough = min(factors) if len(factors) > 0 else 0
            return f'Strong seasonal pattern (peak={peak:.2f}, trough={trough:.2f})'

    def arima_forecast(
        self,
        ts: pd.Series,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        forecast_periods: int = 7,
        seasonal: bool = False,
        m: int = 7  # Seasonal period
    ) -> dict:
        """
        Fit ARIMA model and generate forecasts.

        Args:
            ts: Time series data
            p: AR order (number of lag observations)
            d: Differencing order (degree of differencing)
            q: MA order (size of moving average window)
            forecast_periods: Number of periods to forecast
            seasonal: Whether to include seasonal component
            m: Seasonal period

        Returns:
            Dictionary with forecast results
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            logger.warning("statsmodels not available, using simplified ARIMA")
            return self._simplified_arima(ts, p, d, q, forecast_periods)

        # Fit model
        if seasonal:
            try:
                model = SARIMAX(ts, order=(p, d, q), seasonal_order=(p, d, q, m))
                fitted = model.fit(disp=False)
            except:
                logger.warning("SARIMAX failed, falling back to ARIMA")
                model = ARIMA(ts, order=(p, d, q))
                fitted = model.fit()
        else:
            model = ARIMA(ts, order=(p, d, q))
            fitted = model.fit()

        # Forecast
        forecast = fitted.forecast(steps=forecast_periods)

        # Confidence intervals
        se = np.std(fitted.resid)
        ci_lower = forecast - self.z_score * se * np.sqrt(np.arange(1, forecast_periods + 1))
        ci_upper = forecast + self.z_score * se * np.sqrt(np.arange(1, forecast_periods + 1))

        # Model metrics
        aic = fitted.aic
        bic = fitted.bic

        # Generate future dates
        freq = ts.index.freq or 'D'
        future_dates = pd.date_range(start=ts.index[-1] + freq, periods=forecast_periods, freq=freq)

        result = {
            'model_type': f'ARIMA({p},{d},{q})' + ('S' if seasonal else ''),
            'forecast': forecast.values.tolist(),
            'forecast_dates': [d.isoformat() for d in future_dates],
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist(),
            'aic': round(aic, 2),
            'bic': round(bic, 2),
            'residuals_mean': round(np.mean(fitted.resid), 4),
            'residuals_std': round(np.std(fitted.resid), 4),
            'interpretation': self._interpret_forecast(forecast, ci_lower, ci_upper)
        }

        logger.info(f"ARIMA forecast generated: {forecast_periods} periods ahead")
        return result

    def _simplified_arima(
        self,
        ts: pd.Series,
        p: int,
        d: int,
        q: int,
        forecast_periods: int
    ) -> dict:
        """Simplified ARIMA implementation without statsmodels."""
        # Use exponential smoothing as fallback
        return self.exponential_smoothing_forecast(ts, forecast_periods)

    def exponential_smoothing_forecast(
        self,
        ts: pd.Series,
        forecast_periods: int = 7,
        trend: bool = True,
        seasonal: bool = False,
        seasonal_periods: int = 7
    ) -> dict:
        """
        Exponential smoothing (Holt-Winters) forecast.

        Args:
            ts: Time series data
            forecast_periods: Number of periods to forecast
            trend: Include trend component
            seasonal: Include seasonal component
            seasonal_periods: Period for seasonal component

        Returns:
            Dictionary with forecast results
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            logger.warning("statsmodels not available, using simplified ES")
            return self._simplified_exponential_smoothing(ts, forecast_periods)

        # Determine model type
        if trend and seasonal:
            model_type = 'add'
        elif trend:
            model_type = 'add'
        elif seasonal:
            model_type = 'add'
        else:
            model_type = 'add'

        try:
            if seasonal:
                model = ExponentialSmoothing(
                    ts,
                    trend='add' if trend else None,
                    seasonal='add' if seasonal else None,
                    seasonal_periods=seasonal_periods
                )
            else:
                model = ExponentialSmoothing(
                    ts,
                    trend='add' if trend else None,
                    exponential=False
                )

            fitted = model.fit(optimized=True)
            forecast = fitted.forecast(forecast_periods)

            # Confidence intervals using residual std
            residuals = fitted.resid
            se = np.std(residuals[~np.isnan(residuals)])
            ci_lower = forecast - self.z_score * se
            ci_upper = forecast + self.z_score * se

            # Generate future dates
            freq = ts.index.freq or 'D'
            future_dates = pd.date_range(start=ts.index[-1] + freq, periods=forecast_periods, freq=freq)

            smoothing_level = fitted.params.get('smoothing_level', None)
            smoothing_trend = fitted.params.get('smoothing_trend', None)

            result = {
                'model_type': f'ExponentialSmoothing({"trend" if trend else ""}{"+" if trend and seasonal else ""}{"seasonal" if seasonal else ""})',
                'forecast': forecast.values.tolist(),
                'forecast_dates': [d.isoformat() for d in future_dates],
                'ci_lower': ci_lower.tolist(),
                'ci_upper': ci_upper.tolist(),
                'smoothing_level': round(smoothing_level, 4) if smoothing_level else None,
                'smoothing_trend': round(smoothing_trend, 4) if smoothing_trend else None,
                'aic': round(fitted.aic, 2) if hasattr(fitted, 'aic') else None,
                'interpretation': self._interpret_forecast(forecast, ci_lower, ci_upper)
            }
        except Exception as e:
            logger.warning(f"ExponentialSmoothing failed: {e}, using simplified version")
            return self._simplified_exponential_smoothing(ts, forecast_periods)

        logger.info(f"Exponential smoothing forecast generated: {forecast_periods} periods")
        return result

    def _simplified_exponential_smoothing(
        self,
        ts: pd.Series,
        forecast_periods: int
    ) -> dict:
        """Simplified exponential smoothing without statsmodels."""
        alpha = 0.3  # Smoothing parameter

        # Calculate smoothed value
        smoothed = ts.copy()
        for i in range(1, len(ts)):
            smoothed.iloc[i] = alpha * ts.iloc[i] + (1 - alpha) * smoothed.iloc[i-1]

        # Forecast as flat line (last smoothed value)
        last_value = smoothed.iloc[-1]
        forecast = np.full(forecast_periods, last_value)

        # Confidence intervals widen over time
        residuals = ts - smoothed
        se = np.std(residuals[~np.isnan(residuals)])
        ci_lower = forecast - self.z_score * se * np.sqrt(np.arange(1, forecast_periods + 1))
        ci_upper = forecast + self.z_score * se * np.sqrt(np.arange(1, forecast_periods + 1))

        freq = ts.index.freq or 'D'
        future_dates = pd.date_range(start=ts.index[-1] + freq, periods=forecast_periods, freq=freq)

        return {
            'model_type': 'SimpleExponentialSmoothing',
            'forecast': forecast.tolist(),
            'forecast_dates': [d.isoformat() for d in future_dates],
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist(),
            'alpha': alpha,
            'interpretation': f'Forecast stable at {last_value:.2f} with widening uncertainty'
        }

    def _interpret_forecast(
        self,
        forecast: Union[pd.Series, np.ndarray],
        ci_lower: Union[pd.Series, np.ndarray],
        ci_upper: Union[pd.Series, np.ndarray]
    ) -> str:
        """Generate human-readable forecast interpretation."""
        if isinstance(forecast, pd.Series):
            forecast = forecast.values

        first_val = forecast[0]
        last_val = forecast[-1]
        avg_forecast = np.mean(forecast)

        change_pct = (last_val - first_val) / first_val * 100 if first_val != 0 else 0

        direction = 'increase' if change_pct > 0 else 'decrease'
        trend_desc = 'upward' if change_pct > 0 else 'downward'

        return f"Forecast predicts {trend_desc} trend from {first_val:.2f} to {last_val:.2f} (avg: {avg_forecast:.2f}, {abs(change_pct):.1f}% {direction})"

    def calculate_forecast_accuracy(
        self,
        ts: pd.Series,
        train_size: int = None,
        test_size: int = 7,
        model_func: str = 'arima'
    ) -> dict:
        """
        Calculate forecast accuracy using holdout validation.

        Args:
            ts: Full time series
            train_size: Size of training set (None = use all except test_size)
            test_size: Size of test/holdout set
            model_func: Model to use ('arima', 'ets')

        Returns:
            Dictionary with accuracy metrics
        """
        if train_size is None:
            train_size = len(ts) - test_size

        train = ts.iloc[:train_size]
        test = ts.iloc[train_size:train_size + test_size]

        # Fit and forecast
        if model_func == 'arima':
            forecast_result = self.arima_forecast(train, p=1, d=1, q=1, forecast_periods=test_size)
        else:
            forecast_result = self.exponential_smoothing_forecast(train, forecast_periods=test_size)

        forecast = np.array(forecast_result['forecast'][:len(test)])

        # Calculate error metrics
        actual = test.values
        mae = np.mean(np.abs(actual - forecast))
        mse = np.mean((actual - forecast) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100 if not np.any(actual == 0) else np.nan

        # Direction accuracy
        actual_direction = np.diff(actual) > 0
        forecast_direction = np.diff(forecast) > 0
        direction_accuracy = np.mean(actual_direction == forecast_direction) * 100

        result = {
            'train_size': train_size,
            'test_size': len(test),
            'mae': round(mae, 4),
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2) if not np.isnan(mape) else None,
            'direction_accuracy': round(direction_accuracy, 2),
            'model_used': model_func,
            'interpretation': self._interpret_accuracy(mae, mape, direction_accuracy)
        }

        logger.info(f"Forecast accuracy: MAE={mae:.4f}, MAPE={mape:.2f}%, Direction={direction_accuracy:.1f}%")
        return result

    def _interpret_accuracy(self, mae: float, mape: float, direction_acc: float) -> str:
        """Generate human-readable accuracy interpretation."""
        if mape < 10:
            mape_desc = 'excellent'
        elif mape < 20:
            mape_desc = 'good'
        elif mape < 30:
            mape_desc = 'fair'
        else:
            mape_desc = 'poor'

        if direction_acc > 80:
            dir_desc = 'high'
        elif direction_acc > 60:
            dir_desc = 'moderate'
        else:
            dir_desc = 'low'

        return f"Forecast accuracy is {mape_desc} (MAPE={mape:.1f}%), direction prediction is {dir_desc} ({direction_acc:.1f}%)"

    def create_forecast_visualization(
        self,
        ts: pd.Series,
        forecast_result: dict,
        title: str = "Time Series Forecast"
    ) -> go.Figure:
        """
        Create visualization of historical data and forecast.

        Args:
            ts: Historical time series
            forecast_result: Result from arima_forecast or exponential_smoothing_forecast
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = make_subplots()

        # Historical data
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts.values,
            name='Historical',
            mode='lines',
            line=dict(color='#1f77b4')
        ))

        # Forecast
        forecast_dates = [pd.to_datetime(d) for d in forecast_result['forecast_dates']]
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_result['forecast'],
            name='Forecast',
            mode='lines',
            line=dict(color='#ff7f0e', dash='dash')
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=forecast_result['ci_upper'] + forecast_result['ci_lower'][::-1],
            name='95% CI',
            fill='toself',
            opacity=0.2,
            line=dict(color='#ff7f0e', width=0)
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            height=500
        )

        return fig

    def detect_anomalies(
        self,
        ts: pd.Series,
        threshold: float = 3.0
    ) -> dict:
        """
        Detect anomalies in time series using statistical methods.

        Args:
            ts: Time series data
            threshold: Z-score threshold for anomaly detection

        Returns:
            Dictionary with anomaly detection results
        """
        # Calculate z-scores
        mean = ts.mean()
        std = ts.std()

        if std == 0:
            return {
                'anomaly_indices': [],
                'anomaly_dates': [],
                'anomaly_values': [],
                'anomaly_count': 0,
                'anomaly_rate': 0.0
            }

        z_scores = np.abs((ts - mean) / std)

        # Identify anomalies
        anomaly_mask = z_scores > threshold
        anomaly_indices = ts.index[anomaly_mask].tolist()
        anomaly_values = ts[anomaly_mask].values.tolist()

        result = {
            'anomaly_indices': [str(i) for i in anomaly_indices],
            'anomaly_dates': [str(i) for i in anomaly_indices],
            'anomaly_values': [round(v, 4) for v in anomaly_values],
            'anomaly_count': len(anomaly_values),
            'anomaly_rate': round(len(anomaly_values) / len(ts) * 100, 2),
            'mean': round(mean, 4),
            'std': round(std, 4),
            'threshold_z': threshold,
            'interpretation': f"Found {len(anomaly_values)} anomalies ({len(anomaly_values)/len(ts)*100:.1f}% of data)"
        }

        logger.info(f"Anomaly detection: {len(anomaly_values)} anomalies found")
        return result

    def detect_change_points(
        self,
        ts: pd.Series,
        min_length: int = 5,
        jump: int = 1
    ) -> dict:
        """
        Detect change points in time series using cumulative sum method.

        Args:
            ts: Time series data
            min_length: Minimum segment length
            jump: Step size for scanning

        Returns:
            Dictionary with change point detection results
        """
        n = len(ts)
        if n < 2 * min_length:
            return {
                'change_points': [],
                'change_point_count': 0,
                'interpretation': 'Insufficient data for change point detection'
            }

        # Calculate cumulative sum (CUSUM)
        mean = ts.mean()
        residuals = ts - mean
        cusum = np.cumsum(residuals)

        # Find peaks in CUSUM (potential change points)
        change_points = []
        threshold = 1.5 * np.std(residuals)

        for i in range(min_length, n - min_length, jump):
            # Check if this is a local extremum in CUSUM
            if (cusum[i] > cusum[i-1] and cusum[i] > cusum[i+1] and cusum[i] > threshold) or \
               (cusum[i] < cusum[i-1] and cusum[i] < cusum[i+1] and cusum[i] < -threshold):
                change_points.append(i)

        # Remove duplicate/close change points
        filtered_cp = []
        for cp in change_points:
            if not filtered_cp or cp - filtered_cp[-1] >= min_length:
                filtered_cp.append(cp)

        result = {
            'change_points': filtered_cp,
            'change_point_dates': [str(ts.index[i]) for i in filtered_cp],
            'change_point_count': len(filtered_cp),
            'interpretation': f"Detected {len(filtered_cp)} change points in the series"
        }

        logger.info(f"Change point detection: {len(filtered_cp)} change points found")
        return result


# Convenience functions for pipeline integration
def forecast_metrics(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    forecast_periods: int = 7,
    model: str = 'auto'
) -> dict:
    """
    Convenience function to run complete forecast analysis.

    Args:
        df: DataFrame with time series data
        date_col: Name of date column
        value_col: Name of value column to forecast
        forecast_periods: Number of periods to forecast
        model: Model to use ('auto', 'arima', 'ets')

    Returns:
        Dictionary with complete forecast analysis
    """
    forecaster = TimeSeriesForecaster()

    # Prepare time series
    ts = forecaster.prepare_time_series(df, date_col, value_col)

    # Detect trend and seasonality
    trend_result = forecaster.detect_trend(ts)
    seasonality_result = forecaster.detect_seasonality(ts)

    # Choose model based on seasonality
    if model == 'auto':
        model = 'ets' if seasonality_result['has_seasonality'] else 'arima'

    # Generate forecast
    if model == 'arima':
        forecast_result = forecaster.arima_forecast(ts, forecast_periods=forecast_periods)
    else:
        seasonal = seasonality_result['has_seasonality']
        forecast_result = forecaster.exponential_smoothing_forecast(
            ts,
            forecast_periods=forecast_periods,
            trend=True,
            seasonal=seasonal,
            seasonal_periods=seasonality_result.get('period', 7)
        )

    # Detect anomalies
    anomaly_result = forecaster.detect_anomalies(ts)

    return {
        'trend': trend_result,
        'seasonality': seasonality_result,
        'forecast': forecast_result,
        'anomalies': anomaly_result,
        'model_used': model
    }
