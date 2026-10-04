"""
forecasters/wrappers/tsfresh_wrapper.py - tsfresh feature extraction.

Uses tsfresh for automated feature extraction + ML for forecasting.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


@register_model
class TSFreshForecaster(Forecaster):
    """
    TSFresh-based forecaster.

    Extracts features with tsfresh, then uses ML model.

    Args:
        n_lags: Number of historical windows for feature extraction
        regressor: Sklearn-compatible regressor (default: GradientBoosting)
    """

    def __init__(
        self,
        n_lags: int = 10,
        **kwargs
    ):
        super().__init__(n_lags=n_lags, **kwargs)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'TSFreshForecaster':
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import MinimalFCParameters
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            raise ImportError(
                "tsfresh not installed. Install with: pip install tsfresh"
            )

        y = self._validate_y(y)
        self._train_y = y

        n_lags = self.params.get('n_lags', 10)
        self._n_lags = n_lags

        # Single-output targets: each window predicts the next value
        n_samples = len(y) - n_lags
        if n_samples < 1:
            raise ValueError(
                f"Series too short ({len(y)}) for n_lags={n_lags}. "
                f"Need at least {n_lags + 1} points."
            )

        X_data = []
        y_targets = []

        for i in range(n_samples):
            window = y[i:i + n_lags]
            df_window = pd.DataFrame({
                'id': [i] * n_lags,
                'time': range(n_lags),
                'value': window
            })
            X_data.append(df_window)
            y_targets.append(y[i + n_lags])

        df_all = pd.concat(X_data, ignore_index=True)

        features = extract_features(
            df_all,
            column_id='id',
            column_sort='time',
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
        )

        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        self._feature_names = features.columns.tolist()
        self._X_train_features = features.values
        self._regressor = GradientBoostingRegressor(random_state=42)
        self._regressor.fit(self._X_train_features, np.array(y_targets))

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters

        # Recursive: predict 1 step, shift window, repeat
        last_values = list(self._train_y[-self._n_lags:])
        predictions = []

        for _ in range(horizon):
            window = np.array(last_values[-self._n_lags:])
            df_window = pd.DataFrame({
                'id': [0] * self._n_lags,
                'time': range(self._n_lags),
                'value': window
            })

            features = extract_features(
                df_window,
                column_id='id',
                column_sort='time',
                default_fc_parameters=MinimalFCParameters(),
                disable_progressbar=True,
            )

            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
            pred = self._regressor.predict(features.values)[0]
            predictions.append(pred)
            last_values.append(pred)

        return np.array(predictions)

    def _compute_fitted_values(self):
        if not hasattr(self, '_X_train_features') or self._regressor is None:
            return None
        preds = self._regressor.predict(self._X_train_features)
        fitted = np.full(len(self._train_y), np.nan)
        fitted[self._n_lags:self._n_lags + len(preds)] = preds
        return fitted

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TSFreshForecaster",
            category=ModelCategory.ML,
            library="tsfresh",
            year=2018,
            paper="Christ et al. (2018)",
            github_url="https://github.com/blue-yonder/tsfresh",
        )
