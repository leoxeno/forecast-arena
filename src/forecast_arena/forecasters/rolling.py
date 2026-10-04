"""
forecasters/rolling.py - Rolling origin forecast evaluation.

Implements rolling forecast with configurable refit frequency,
the gold standard for time series model evaluation.

Rolling forecast simulates real-world deployment:
1. Train on available data up to time t
2. Forecast h steps ahead
3. Move forward, optionally refit
4. Repeat until end of series

References:
- Tashman (2000): "Out-of-sample tests of forecasting accuracy"
- Hyndman & Athanasopoulos (2021): "Forecasting: Principles and Practice"
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class RollingResult:
    """Results from rolling forecast evaluation."""
    forecasts: np.ndarray  # All point forecasts
    actuals: np.ndarray    # Corresponding actuals
    forecast_origins: np.ndarray  # Index of each forecast origin
    horizons: np.ndarray   # Horizon of each forecast
    n_refits: int          # Number of times model was refit

    def get_horizon_forecasts(self, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get forecasts and actuals for a specific horizon."""
        mask = self.horizons == h
        return self.forecasts[mask], self.actuals[mask]

    def get_metrics(
        self,
        metric_fn: Callable,
        by_horizon: bool = False
    ) -> Union[float, Dict[int, float]]:
        """
        Compute metric on rolling forecasts.

        Args:
            metric_fn: Function(y_true, y_pred) -> float
            by_horizon: If True, return dict of metrics by horizon

        Returns:
            Overall metric or dict of metrics by horizon
        """
        if not by_horizon:
            return metric_fn(self.actuals, self.forecasts)

        unique_horizons = np.unique(self.horizons)
        return {
            int(h): metric_fn(*self.get_horizon_forecasts(h))
            for h in unique_horizons
        }


def rolling_forecast(
    forecaster,
    y: np.ndarray,
    horizon: int,
    initial_train: Optional[int] = None,
    step: int = 1,
    refit_every: int = 1,
    X: Optional[np.ndarray] = None,
    verbose: bool = False
) -> RollingResult:
    """
    Rolling origin forecast evaluation.

    Simulates real forecasting: train on past, forecast future, move forward.

    Args:
        forecaster: Forecaster instance (will be cloned)
        y: Full time series
        horizon: Forecast horizon (h-step ahead)
        initial_train: Minimum training size (default: 2 * horizon)
        step: How many steps to move forward each iteration
        refit_every: Refit model every N iterations (1 = always refit)
        X: Exogenous features (optional)
        verbose: Print progress

    Returns:
        RollingResult with all forecasts and actuals

    Example:
        # Multi-step rolling forecast
        result = rolling_forecast(model, y, horizon=12, step=1, refit_every=12)

        # Evaluate by horizon
        from forecasters.metrics import mae
        metrics = result.get_metrics(mae, by_horizon=True)
        # {1: 0.5, 2: 0.6, ..., 12: 0.9}
    """
    y = np.asarray(y).flatten()
    n = len(y)

    if initial_train is None:
        initial_train = max(2 * horizon, 10)

    if initial_train + horizon > n:
        raise ValueError(
            f"Not enough data: n={n}, initial_train={initial_train}, horizon={horizon}. "
            f"Need at least initial_train + horizon = {initial_train + horizon} observations."
        )

    forecasts_list = []
    actuals_list = []
    origins_list = []
    horizons_list = []

    model = forecaster.clone()
    n_refits = 0
    iteration = 0

    # Start from initial_train, end when we can't forecast horizon steps
    origin = initial_train
    while origin + horizon <= n:
        # Refit if needed
        should_refit = (iteration % refit_every == 0) or (iteration == 0)

        if should_refit:
            train_y = y[:origin]
            train_X = X[:origin] if X is not None else None

            model = forecaster.clone()
            model.fit(train_y, X=train_X)
            n_refits += 1

            if verbose:
                print(f"Origin {origin}: Refitting on {len(train_y)} observations")

        # Forecast
        future_X = X[origin:origin + horizon] if X is not None else None
        preds = model.predict(horizon, X=future_X)
        actuals = y[origin:origin + horizon]

        # Store results for each horizon
        for h in range(horizon):
            forecasts_list.append(preds[h])
            actuals_list.append(actuals[h])
            origins_list.append(origin)
            horizons_list.append(h + 1)  # 1-indexed horizon

        # Move forward
        origin += step
        iteration += 1

    if len(forecasts_list) == 0:
        raise ValueError("No forecasts generated. Check parameters.")

    return RollingResult(
        forecasts=np.array(forecasts_list),
        actuals=np.array(actuals_list),
        forecast_origins=np.array(origins_list),
        horizons=np.array(horizons_list),
        n_refits=n_refits
    )


def expanding_rolling_forecast(
    forecaster,
    y: np.ndarray,
    horizon: int,
    initial_train: Optional[int] = None,
    step: int = 1,
    X: Optional[np.ndarray] = None,
    verbose: bool = False
) -> RollingResult:
    """
    Expanding window (always refit) rolling forecast.

    Convenience wrapper for rolling_forecast with refit_every=1.
    Training window grows each iteration.

    Args:
        forecaster: Forecaster instance
        y: Full time series
        horizon: Forecast horizon
        initial_train: Minimum training size
        step: Steps to move forward
        X: Exogenous features
        verbose: Print progress

    Returns:
        RollingResult
    """
    return rolling_forecast(
        forecaster=forecaster,
        y=y,
        horizon=horizon,
        initial_train=initial_train,
        step=step,
        refit_every=1,
        X=X,
        verbose=verbose
    )


def fixed_window_rolling_forecast(
    forecaster,
    y: np.ndarray,
    horizon: int,
    train_size: int,
    step: int = 1,
    refit_every: int = 1,
    X: Optional[np.ndarray] = None,
    verbose: bool = False
) -> RollingResult:
    """
    Fixed-size sliding window rolling forecast.

    Training window is fixed size, slides forward.
    Useful when older data becomes less relevant.

    Args:
        forecaster: Forecaster instance
        y: Full time series
        horizon: Forecast horizon
        train_size: Fixed training window size
        step: Steps to move forward
        refit_every: Refit frequency
        X: Exogenous features
        verbose: Print progress

    Returns:
        RollingResult
    """
    y = np.asarray(y).flatten()
    n = len(y)

    if train_size + horizon > n:
        raise ValueError(
            f"Not enough data: n={n}, train_size={train_size}, horizon={horizon}"
        )

    forecasts_list = []
    actuals_list = []
    origins_list = []
    horizons_list = []

    model = forecaster.clone()
    n_refits = 0
    iteration = 0

    origin = train_size
    while origin + horizon <= n:
        should_refit = (iteration % refit_every == 0) or (iteration == 0)

        if should_refit:
            # Fixed window: only use last train_size observations
            train_start = origin - train_size
            train_y = y[train_start:origin]
            train_X = X[train_start:origin] if X is not None else None

            model = forecaster.clone()
            model.fit(train_y, X=train_X)
            n_refits += 1

            if verbose:
                print(f"Origin {origin}: Refitting on [{train_start}:{origin}]")

        # Forecast
        future_X = X[origin:origin + horizon] if X is not None else None
        preds = model.predict(horizon, X=future_X)
        actuals = y[origin:origin + horizon]

        for h in range(horizon):
            forecasts_list.append(preds[h])
            actuals_list.append(actuals[h])
            origins_list.append(origin)
            horizons_list.append(h + 1)

        origin += step
        iteration += 1

    return RollingResult(
        forecasts=np.array(forecasts_list),
        actuals=np.array(actuals_list),
        forecast_origins=np.array(origins_list),
        horizons=np.array(horizons_list),
        n_refits=n_refits
    )


def compare_rolling_forecasts(
    forecasters: Dict[str, 'Forecaster'],
    y: np.ndarray,
    horizon: int,
    metric_fn: Callable,
    initial_train: Optional[int] = None,
    step: int = 1,
    refit_every: int = 1,
    X: Optional[np.ndarray] = None,
    by_horizon: bool = False,
    verbose: bool = False
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compare multiple forecasters using rolling forecast.

    Args:
        forecasters: Dict of {name: forecaster}
        y: Time series
        horizon: Forecast horizon
        metric_fn: Evaluation metric
        initial_train: Minimum training size
        step: Step size
        refit_every: Refit frequency
        X: Exogenous features
        by_horizon: Return metrics by horizon
        verbose: Print progress

    Returns:
        Dict of {name: metric} or {name: {horizon: metric}}
    """
    results = {}

    for name, forecaster in forecasters.items():
        if verbose:
            print(f"\nEvaluating: {name}")

        result = rolling_forecast(
            forecaster=forecaster,
            y=y,
            horizon=horizon,
            initial_train=initial_train,
            step=step,
            refit_every=refit_every,
            X=X,
            verbose=verbose
        )

        results[name] = result.get_metrics(metric_fn, by_horizon=by_horizon)

    return results


# Public API
__all__ = [
    'RollingResult',
    'rolling_forecast',
    'expanding_rolling_forecast',
    'fixed_window_rolling_forecast',
    'compare_rolling_forecasts',
]
