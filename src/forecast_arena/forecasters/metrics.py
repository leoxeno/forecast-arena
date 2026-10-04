"""
forecasters/metrics.py - Forecast evaluation metrics.

Implements:
- Standard: MAE, RMSE, MAPE, sMAPE
- Scaled: MASE, RMSSE (M5 competition)
- Competition: OWA (M4)
- Statistical: Diebold-Mariano test
- Uncertainty: Block bootstrap confidence intervals

References:
- Hyndman & Koehler (2006): "Another look at measures of forecast accuracy"
- Makridakis et al. (2020): M5 competition
- Lopez de Prado (2018): "Advances in Financial Machine Learning"
- Diebold & Mariano (1995): "Comparing Predictive Accuracy"
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


# =============================================================================
# Point Forecast Metrics
# =============================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    MAE = mean(|y_true - y_pred|)

    Robust to outliers, interpretable in original units.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.

    MSE = mean((y_true - y_pred)^2)

    Penalizes large errors more heavily than MAE.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt(MSE)

    Same units as target, penalizes large errors.
    """
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.

    MAPE = 100 * mean(|y_true - y_pred| / |y_true|)

    WARNING: Undefined when y_true contains zeros.
    Consider sMAPE or MASE for series with zeros.

    Args:
        epsilon: Small value to avoid division by zero
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if np.any(np.abs(y_true) < epsilon):
        # Warn but compute with epsilon protection
        import warnings
        warnings.warn(
            "MAPE is unreliable when y_true contains values near zero. "
            "Consider using sMAPE or MASE instead."
        )
    return 100.0 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon))


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    sMAPE = 200 * mean(|y_true - y_pred| / (|y_true| + |y_pred|))

    Bounded between 0 and 200, symmetric in over/under prediction.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return 200.0 * np.mean(numerator / denominator)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = MAE / MAE_naive

    Where naive is seasonal naive (y[t-seasonality]) on training data.

    Scale-free metric, <1 means better than naive.

    Args:
        y_train: Training series (for computing naive denominator)
        seasonality: Seasonal period (1 for non-seasonal)

    Reference:
        Hyndman & Koehler (2006)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    y_train = np.asarray(y_train).flatten()

    # Naive forecast error on training set
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale < 1e-10:
        # Series is constant, MASE undefined
        return np.nan

    forecast_mae = np.mean(np.abs(y_true - y_pred))
    return forecast_mae / scale


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Root Mean Squared Scaled Error (M5 competition metric).

    RMSSE = sqrt(MSE / MSE_naive)

    Official metric for the M5 Forecasting Competition.

    Args:
        y_train: Training series (for computing naive denominator)
        seasonality: Seasonal period (1 for non-seasonal)

    Reference:
        Makridakis et al. (2020) - M5 Competition
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    y_train = np.asarray(y_train).flatten()

    # Naive forecast MSE on training set
    naive_errors = (y_train[seasonality:] - y_train[:-seasonality]) ** 2
    scale = np.mean(naive_errors)

    if scale < 1e-10:
        return np.nan

    forecast_mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(forecast_mse / scale)


# =============================================================================
# Competition Metrics
# =============================================================================

def owa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    naive2_smape: float,
    naive2_mase: float,
    seasonality: int = 1
) -> float:
    """
    Overall Weighted Average (M4 competition metric).

    OWA = 0.5 * (sMAPE/sMAPE_naive2 + MASE/MASE_naive2)

    The M4 competition used Naive2 (seasonal naive with deseasoning)
    as reference. Lower OWA is better; OWA < 1 beats the benchmark.

    Args:
        naive2_smape: sMAPE of Naive2 method (provided by competition)
        naive2_mase: MASE of Naive2 method (provided by competition)
        seasonality: Seasonal period

    Reference:
        Makridakis et al. (2018) - M4 Competition
    """
    model_smape = smape(y_true, y_pred)
    model_mase = mase(y_true, y_pred, y_train, seasonality)

    if naive2_smape < 1e-10 or naive2_mase < 1e-10:
        return np.nan

    return 0.5 * (model_smape / naive2_smape + model_mase / naive2_mase)


# =============================================================================
# Statistical Tests
# =============================================================================

@dataclass
class DMTestResult:
    """Results from Diebold-Mariano test."""
    statistic: float
    pvalue: float
    significant: bool
    better_model: str  # '1', '2', or 'neither'


def diebold_mariano_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    loss: str = 'mse',
    h: int = 1,
    alpha: float = 0.05
) -> DMTestResult:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: E[L(e1)] = E[L(e2)] (equal predictive accuracy)
    vs H1: E[L(e1)] != E[L(e2)] (different accuracy)

    Uses HAC (Newey-West) standard errors for serial correlation.

    Args:
        y_true: Actual values
        pred1: Forecasts from model 1
        pred2: Forecasts from model 2
        loss: Loss function ('mse', 'mae', 'mape')
        h: Forecast horizon (for HAC bandwidth)
        alpha: Significance level

    Returns:
        DMTestResult with statistic, p-value, significance, and better model

    Reference:
        Diebold & Mariano (1995), Harvey, Leybourne & Newbold (1997)
    """
    y_true, pred1 = _validate_inputs(y_true, pred1)
    _, pred2 = _validate_inputs(y_true, pred2)

    # Compute loss differentials
    e1 = y_true - pred1
    e2 = y_true - pred2

    if loss == 'mse':
        d = e1**2 - e2**2
    elif loss == 'mae':
        d = np.abs(e1) - np.abs(e2)
    elif loss == 'mape':
        d = np.abs(e1 / y_true) - np.abs(e2 / y_true)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    d_mean = np.mean(d)

    # HAC variance estimator (Newey-West)
    # Bandwidth selection: h - 1 for h-step ahead forecasts
    bandwidth = h - 1

    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0

    for k in range(1, bandwidth + 1):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * (1 - k / (bandwidth + 1)) * gamma_k

    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        # Degenerate case
        return DMTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            significant=False,
            better_model='neither'
        )

    # DM statistic
    dm_stat = d_mean / np.sqrt(var_d)

    # Harvey, Leybourne, Newbold small-sample correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * correction

    # Two-sided test using t-distribution (more conservative for small n)
    pvalue = 2 * stats.t.sf(np.abs(dm_stat_corrected), df=n - 1)

    significant = pvalue < alpha
    if not significant:
        better = 'neither'
    elif dm_stat_corrected < 0:
        better = '1'  # Model 1 has lower loss
    else:
        better = '2'  # Model 2 has lower loss

    return DMTestResult(
        statistic=dm_stat_corrected,
        pvalue=pvalue,
        significant=significant,
        better_model=better
    )


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

@dataclass
class BootstrapResult:
    """Results from bootstrap confidence interval estimation."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int


def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    block_size: Optional[int] = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> BootstrapResult:
    """
    Block bootstrap confidence intervals for forecast metrics.

    Uses moving block bootstrap to preserve temporal dependence.
    Critical for honest uncertainty quantification in time series.

    Args:
        y_true: Actual values
        y_pred: Forecast values
        metric_fn: Function(y_true, y_pred) -> float
        block_size: Bootstrap block size (default: sqrt(n))
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        BootstrapResult with point estimate, CI bounds, and std error

    Reference:
        Politis & Romano (1994): "The Stationary Bootstrap"
        Lopez de Prado (2018): Ch. 12 - Backtesting
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    n = len(y_true)

    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    rng = np.random.default_rng(seed)

    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap
    bootstrap_metrics = []
    n_blocks = int(np.ceil(n / block_size))

    for _ in range(n_bootstrap):
        # Sample block start indices
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)

        # Build bootstrap sample
        indices = []
        for start in starts:
            indices.extend(range(start, min(start + block_size, n)))
        indices = np.array(indices[:n])  # Trim to exact length

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        bootstrap_metrics.append(metric_fn(y_true_boot, y_pred_boot))

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Percentile method for CI
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=np.std(bootstrap_metrics),
        n_bootstrap=n_bootstrap
    )


# =============================================================================
# Multi-Metric Evaluation
# =============================================================================

def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 1,
    include_scaled: bool = True
) -> Dict[str, float]:
    """
    Compute all relevant metrics for a forecast.

    Args:
        y_true: Actual test values
        y_pred: Forecast values
        y_train: Training data (required for scaled metrics)
        seasonality: Seasonal period for MASE/RMSSE
        include_scaled: Whether to include MASE/RMSSE (requires y_train)

    Returns:
        Dictionary of metric names to values
    """
    results = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
    }

    if include_scaled and y_train is not None:
        results['MASE'] = mase(y_true, y_pred, y_train, seasonality)
        results['RMSSE'] = rmsse(y_true, y_pred, y_train, seasonality)

    return results


# =============================================================================
# Helpers
# =============================================================================

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs to numpy arrays."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")

    return y_true, y_pred


# Public API
__all__ = [
    # Standard metrics
    'mae', 'mse', 'rmse', 'mape', 'smape',
    # Scaled metrics
    'mase', 'rmsse',
    # Competition metrics
    'owa',
    # Statistical tests
    'diebold_mariano_test', 'DMTestResult',
    # Bootstrap
    'block_bootstrap_ci', 'BootstrapResult',
    # Multi-metric
    'evaluate_forecast',
]
