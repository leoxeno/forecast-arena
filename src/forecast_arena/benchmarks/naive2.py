"""Naive2 (seasonally adjusted naive) — official M3/M4 baseline.

Naive2 is the reference model for OWA computation. It applies classical
multiplicative seasonal decomposition, then produces a naive forecast on the
deseasonalized series, and re-seasonalizes.

For non-seasonal data (seasonality <= 1): simple last-value naive.

Reference: Makridakis & Hibon (2000), M3 competition paper.
"""

import numpy as np


def naive2_forecast(y_train: np.ndarray, horizon: int, seasonality: int = 1) -> np.ndarray:
    """Produce Naive2 (seasonally adjusted naive) forecast.

    Parameters
    ----------
    y_train : np.ndarray
        Historical values (1D).
    horizon : int
        Number of steps to forecast.
    seasonality : int
        Seasonal period (1 = non-seasonal).

    Returns
    -------
    np.ndarray
        Point forecasts of length `horizon`.
    """
    y = np.asarray(y_train, dtype=np.float64)

    # Non-seasonal or too short for decomposition → simple naive
    if seasonality <= 1 or len(y) < 2 * seasonality:
        return np.full(horizon, y[-1])

    # Classical multiplicative decomposition
    # Step 1: Centered moving average (trend-cycle)
    if seasonality % 2 == 0:
        # Even period: 2×m MA (average of two m-length MAs offset by 1)
        weights = np.ones(seasonality + 1)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights /= seasonality
    else:
        weights = np.ones(seasonality) / seasonality

    trend = np.convolve(y, weights, mode="valid")
    # Align: trend starts at index offset
    offset = (len(weights) - 1) // 2

    # Step 2: Seasonal indices (multiplicative)
    seasonal_component = y[offset : offset + len(trend)] / trend
    # Average seasonal index per position within the season
    n_full = len(seasonal_component)
    indices = np.zeros(seasonality)
    counts = np.zeros(seasonality)
    for i in range(n_full):
        pos = (offset + i) % seasonality
        indices[pos] += seasonal_component[i]
        counts[pos] += 1

    counts[counts == 0] = 1  # avoid division by zero
    indices /= counts
    # Normalize so indices average to 1.0
    indices *= seasonality / indices.sum()

    # Step 3: Deseasonalize
    season_factors = np.array([indices[i % seasonality] for i in range(len(y))])
    # Guard against zero seasonal factors
    season_factors[season_factors == 0] = 1.0
    deseasonalized = y / season_factors

    # Step 4: Naive forecast on deseasonalized series
    last_deseas = deseasonalized[-1]

    # Step 5: Re-seasonalize
    forecast = np.empty(horizon)
    for h in range(horizon):
        pos = (len(y) + h) % seasonality
        forecast[h] = last_deseas * indices[pos]

    return forecast
