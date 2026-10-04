"""Reference baselines in pure NumPy: Naive, SeasonalNaive, Drift and Mean.

They need no optional dependency, so every environment can run the arena, and
they are the floor every other model has to clear.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .. import register_model
from ..base import Forecaster, ModelCategory, ModelMetadata, ParamSpace

_SEASONS = {"D": 7, "W": 52, "ME": 12, "MS": 12, "M": 12, "QE": 4, "Q": 4, "h": 24, "H": 24}


class _NumpyBaseline(Forecaster):
    def fit(self, y, X=None, freq: Optional[str] = None, **kwargs) -> "Forecaster":
        self._train_y = self._validate_y(y)
        self._freq = freq
        self._is_fitted = True
        return self

    def _compute_fitted_values(self):
        return None


@register_model
class Naive(_NumpyBaseline):
    """Repeat the last observation."""

    def predict(self, horizon: int, X=None, **kwargs) -> np.ndarray:
        self._check_fitted()
        return np.repeat(self._train_y[-1], horizon)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(name="Naive", category=ModelCategory.CLASSICAL, library="numpy",
                             year=1900, paper="Random walk forecast")


@register_model
class SeasonalNaive(_NumpyBaseline):
    """Repeat the observation one season back; the season comes from ``freq`` or ``m``."""

    def predict(self, horizon: int, X=None, **kwargs) -> np.ndarray:
        self._check_fitted()
        m = int(self.params.get("m") or _SEASONS.get(self._freq or "", 1))
        y = self._train_y
        if m <= 1 or len(y) < m:
            return np.repeat(y[-1], horizon)
        last = y[-m:]
        return np.array([last[i % m] for i in range(horizon)])

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(name="SeasonalNaive", category=ModelCategory.CLASSICAL, library="numpy",
                             year=1900, paper="Seasonal random walk forecast")

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [ParamSpace("m", "categorical", choices=[1, 4, 7, 12, 24, 52], default=None)]


@register_model
class Drift(_NumpyBaseline):
    """Extend the straight line from the first to the last observation."""

    def predict(self, horizon: int, X=None, **kwargs) -> np.ndarray:
        self._check_fitted()
        y = self._train_y
        slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
        return y[-1] + slope * np.arange(1, horizon + 1)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(name="Drift", category=ModelCategory.CLASSICAL, library="numpy",
                             year=1900, paper="Drift method, Hyndman and Athanasopoulos (2021)")


@register_model
class Mean(_NumpyBaseline):
    """Forecast the training mean."""

    def predict(self, horizon: int, X=None, **kwargs) -> np.ndarray:
        self._check_fitted()
        return np.repeat(float(np.mean(self._train_y)), horizon)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(name="Mean", category=ModelCategory.CLASSICAL, library="numpy",
                             year=1900, paper="Historical mean forecast")


__all__ = ["Naive", "SeasonalNaive", "Drift", "Mean"]
