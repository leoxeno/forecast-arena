"""
forecasters/wrappers/tslearn_wrapper.py - tslearn time series ML.

Uses tslearn for k-NN based forecasting with DTW distance.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


@register_model
class TSLearnKNN(Forecaster):
    """
    K-NN forecaster with DTW distance.

    Finds similar historical patterns using Dynamic Time Warping.

    Args:
        n_neighbors: Number of neighbors
        metric: Distance metric ('dtw', 'euclidean', 'softdtw')
        window_size: Size of pattern windows
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = 'dtw',
        window_size: int = 10,
        **kwargs
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            metric=metric,
            window_size=window_size,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'TSLearnKNN':
        try:
            from tslearn.neighbors import KNeighborsTimeSeriesRegressor
        except ImportError:
            raise ImportError(
                "tslearn not installed. Install with: pip install tslearn"
            )

        y = self._validate_y(y)
        self._train_y = y

        window_size = self.params.get('window_size', 10)
        self._window_size = window_size

        # Single-output targets: each window predicts the next value
        n_samples = len(y) - window_size
        if n_samples < 1:
            raise ValueError(
                f"Series too short ({len(y)}) for window_size={window_size}. "
                f"Need at least {window_size + 1} points."
            )

        X_train = []
        y_train = []

        for i in range(n_samples):
            X_train.append(y[i:i + window_size])
            y_train.append(y[i + window_size])

        self._X_train_windows = np.array(X_train).reshape(-1, window_size, 1)
        y_train = np.array(y_train)

        self._model = KNeighborsTimeSeriesRegressor(
            n_neighbors=self.params.get('n_neighbors', 5),
            metric=self.params.get('metric', 'dtw'),
        )
        self._model.fit(self._X_train_windows, y_train)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        # Recursive: predict 1 step, shift window, repeat
        last_values = list(self._train_y[-self._window_size:])
        predictions = []

        for _ in range(horizon):
            window = np.array(last_values[-self._window_size:]).reshape(1, -1, 1)
            pred = self._model.predict(window)[0]
            predictions.append(pred)
            last_values.append(pred)

        return np.array(predictions)

    def _compute_fitted_values(self):
        if self._model is None or not hasattr(self, '_X_train_windows'):
            return None
        preds = self._model.predict(self._X_train_windows)
        fitted = np.full(len(self._train_y), np.nan)
        fitted[self._window_size:self._window_size + len(preds)] = preds
        return fitted

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TSLearnKNN",
            category=ModelCategory.ML,
            library="tslearn",
            year=2017,
            paper="Tavenard et al. (2017)",
            github_url="https://github.com/tslearn-team/tslearn",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_neighbors", "int", low=1, high=20, default=5),
            ParamSpace("window_size", "int", low=5, high=50, default=10),
            ParamSpace("metric", "categorical", choices=['dtw', 'euclidean', 'softdtw']),
        ]
