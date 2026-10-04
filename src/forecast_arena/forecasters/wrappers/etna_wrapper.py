"""
forecasters/wrappers/etna_wrapper.py - Tinkoff ETNA framework.

Wraps ETNA models (Tinkoff's time series library):
- ProphetModel
- CatBoostModel
- AutoARIMA
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


# NOTE: ETNA is quarantined on PyPI (security issue) - models disabled
# See: https://github.com/tinkoff-ai/etna/issues
# @register_model  # DISABLED - etna quarantined on PyPI
class ETNACatBoost(Forecaster):
    """
    ETNA CatBoost model.

    Gradient boosting with lag features.

    Args:
        lags: List of lag values
        iterations: Number of boosting iterations
    """

    def __init__(
        self,
        lags: List[int] = [1, 2, 3, 7, 14],
        iterations: int = 100,
        **kwargs
    ):
        super().__init__(lags=lags, iterations=iterations, **kwargs)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'ETNACatBoost':
        try:
            from etna.datasets import TSDataset
            from etna.models import CatBoostMultiSegmentModel
            from etna.transforms import LagTransform
            from etna.pipeline import Pipeline
        except ImportError:
            raise ImportError("etna not installed. Install with: pip install etna")

        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        df = pd.DataFrame({
            'timestamp': safe_date_range(start='2000-01-01', periods=len(y), freq=self._freq),
            'segment': 'series',
            'target': y
        })

        ts = TSDataset(TSDataset.to_dataset(df), freq=self._freq)

        lags = self.params.get('lags', [1, 2, 3, 7, 14])
        transforms = [LagTransform(in_column='target', lags=lags)]

        model = CatBoostMultiSegmentModel(iterations=self.params.get('iterations', 100))

        self._pipeline = Pipeline(model=model, transforms=transforms, horizon=12)
        self._pipeline.fit(ts)
        self._ts = ts

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        forecast = self._pipeline.forecast()
        return forecast.to_pandas()['series']['target'].values[:horizon]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ETNACatBoost",
            category=ModelCategory.ML,
            library="etna",
            year=2021,
            paper="N/A",
            github_url="https://github.com/tinkoff-ai/etna",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("iterations", "int", low=50, high=500, default=100),
        ]


class _BaseETNA(Forecaster):
    """Base class for ETNA models."""

    def _prepare_data(self, y: np.ndarray, freq: str):
        """Prepare data in ETNA format."""
        from etna.datasets import TSDataset

        df = pd.DataFrame({
            'timestamp': safe_date_range(start='2000-01-01', periods=len(y), freq=freq),
            'segment': 'series',
            'target': y
        })

        return TSDataset(TSDataset.to_dataset(df), freq=freq)

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        forecast = self._pipeline.forecast()
        return forecast.to_pandas()['series']['target'].values[:horizon]


# @register_model  # DISABLED - etna quarantined on PyPI
class ETNAProphet(_BaseETNA):
    """
    ETNA Prophet model.

    Facebook Prophet via ETNA framework.
    Handles trend changes, seasonality, and holidays.

    Args:
        growth: Trend type ('linear', 'logistic')
        changepoint_prior_scale: Flexibility of trend changes
        seasonality_mode: Seasonality type ('additive', 'multiplicative')
    """

    def __init__(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = 'additive',
        **kwargs
    ):
        super().__init__(
            growth=growth,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'ETNAProphet':
        try:
            from etna.models import ProphetModel
            from etna.pipeline import Pipeline
        except ImportError:
            raise ImportError("etna not installed. Install with: pip install etna[prophet]")

        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        self._ts = self._prepare_data(y, self._freq)

        model = ProphetModel(
            growth=self.params.get('growth', 'linear'),
            changepoint_prior_scale=self.params.get('changepoint_prior_scale', 0.05),
            seasonality_mode=self.params.get('seasonality_mode', 'additive'),
        )

        self._pipeline = Pipeline(model=model, transforms=[], horizon=12)
        self._pipeline.fit(self._ts)

        self._is_fitted = True
        return self

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ETNAProphet",
            category=ModelCategory.CLASSICAL,
            library="etna",
            year=2017,
            paper="Taylor & Letham (2017) - Forecasting at Scale",
            github_url="https://github.com/tinkoff-ai/etna",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("growth", "categorical", choices=['linear', 'logistic'], default='linear'),
            ParamSpace("changepoint_prior_scale", "log_float", low=0.001, high=0.5, default=0.05),
            ParamSpace("seasonality_mode", "categorical", choices=['additive', 'multiplicative'], default='additive'),
        ]


# @register_model  # DISABLED - etna quarantined on PyPI
class ETNASARIMAX(_BaseETNA):
    """
    ETNA SARIMAX model.

    Seasonal ARIMA with exogenous variables via ETNA.

    Args:
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, m)
    """

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 7),
        **kwargs
    ):
        super().__init__(
            order=order,
            seasonal_order=seasonal_order,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'ETNASARIMAX':
        try:
            from etna.models import SARIMAXModel
            from etna.pipeline import Pipeline
        except ImportError:
            raise ImportError("etna not installed. Install with: pip install etna")

        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        self._ts = self._prepare_data(y, self._freq)

        order = self.params.get('order', (1, 1, 1))
        seasonal_order = self.params.get('seasonal_order', (1, 1, 1, 7))

        model = SARIMAXModel(
            order=order,
            seasonal_order=seasonal_order,
        )

        self._pipeline = Pipeline(model=model, transforms=[], horizon=12)
        self._pipeline.fit(self._ts)

        self._is_fitted = True
        return self

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ETNASARIMAX",
            category=ModelCategory.CLASSICAL,
            library="etna",
            year=1970,
            paper="Box & Jenkins (1970) - Time Series Analysis",
            github_url="https://github.com/tinkoff-ai/etna",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("order", "categorical", choices=[(0, 1, 1), (1, 1, 1), (1, 1, 2), (2, 1, 2)], default=(1, 1, 1)),
        ]


# @register_model  # DISABLED - etna quarantined on PyPI
class ETNAHoltWinters(_BaseETNA):
    """
    ETNA Holt-Winters model.

    Triple exponential smoothing via ETNA.
    Handles trend and seasonal components.

    Args:
        trend: Trend type ('add', 'mul', None)
        seasonal: Seasonal type ('add', 'mul', None)
        seasonal_periods: Number of periods in a season
    """

    def __init__(
        self,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = 'add',
        seasonal_periods: int = 7,
        **kwargs
    ):
        super().__init__(
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'ETNAHoltWinters':
        try:
            from etna.models import HoltWintersModel
            from etna.pipeline import Pipeline
        except ImportError:
            raise ImportError("etna not installed. Install with: pip install etna")

        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        self._ts = self._prepare_data(y, self._freq)

        model = HoltWintersModel(
            trend=self.params.get('trend', 'add'),
            seasonal=self.params.get('seasonal', 'add'),
            seasonal_periods=self.params.get('seasonal_periods', 7),
        )

        self._pipeline = Pipeline(model=model, transforms=[], horizon=12)
        self._pipeline.fit(self._ts)

        self._is_fitted = True
        return self

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ETNAHoltWinters",
            category=ModelCategory.CLASSICAL,
            library="etna",
            year=1960,
            paper="Holt (1957) & Winters (1960) - Exponential Smoothing",
            github_url="https://github.com/tinkoff-ai/etna",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("trend", "categorical", choices=['add', 'mul', None], default='add'),
            ParamSpace("seasonal", "categorical", choices=['add', 'mul', None], default='add'),
            ParamSpace("seasonal_periods", "int", low=2, high=52, default=7),
        ]
