"""
forecasters/wrappers/orbit_wrapper.py - Uber Orbit (Bayesian time series).

Wraps Orbit models:
- LGT (Local and Global Trend)
- DLT (Damped Local Trend)
- ETS (Exponential Smoothing)
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


class _BaseOrbitForecaster(Forecaster):
    """Base class for Orbit models."""

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BaseOrbitForecaster':
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        df = pd.DataFrame({
            'ds': safe_date_range(start='2000-01-01', periods=len(y), freq=self._freq),
            'y': y
        })

        self._model = self._create_model()
        self._model.fit(df=df)
        self._df = df

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        predicted = self._model.predict(df=self._df, decompose=False)
        return predicted['prediction'].values[-horizon:]

    def _compute_fitted_values(self):
        predicted = self._model.predict(df=self._df, decompose=False)
        return predicted['prediction'].values[:len(self._train_y)]

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        if hasattr(self._model, "_posterior_samples"):
            d["n_posterior_samples"] = len(next(iter(self._model._posterior_samples.values()), []))
        return d

    def _create_model(self):
        raise NotImplementedError


@register_model
class OrbitLGT(_BaseOrbitForecaster):
    """
    Orbit LGT (Local and Global Trend) model.

    Bayesian structural time series with global and local trends.

    Args:
        seasonality: Seasonal period
        estimator: 'stan-mcmc', 'stan-map', or 'pyro-svi'
    """

    def __init__(
        self,
        seasonality: Optional[int] = None,
        estimator: str = 'stan-map',
        **kwargs
    ):
        super().__init__(seasonality=seasonality, estimator=estimator, **kwargs)

    def _create_model(self):
        from orbit.models import LGT
        return LGT(
            response_col='y',
            date_col='ds',
            seasonality=self.params.get('seasonality'),
            estimator=self.params.get('estimator', 'stan-map'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="OrbitLGT",
            category=ModelCategory.PROBABILISTIC,
            library="orbit-ml",
            year=2021,
            paper="Ng et al. (2021)",
            probabilistic=True,
            github_url="https://github.com/uber/orbit",
        )


@register_model
class OrbitDLT(_BaseOrbitForecaster):
    """
    Orbit DLT (Damped Local Trend) model.

    More conservative than LGT with damping factor.
    """

    def __init__(
        self,
        seasonality: Optional[int] = None,
        estimator: str = 'stan-map',
        **kwargs
    ):
        super().__init__(seasonality=seasonality, estimator=estimator, **kwargs)

    def _create_model(self):
        from orbit.models import DLT
        return DLT(
            response_col='y',
            date_col='ds',
            seasonality=self.params.get('seasonality'),
            estimator=self.params.get('estimator', 'stan-map'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="OrbitDLT",
            category=ModelCategory.PROBABILISTIC,
            library="orbit-ml",
            year=2021,
            paper="Ng et al. (2021)",
            probabilistic=True,
        )


@register_model
class OrbitETS(_BaseOrbitForecaster):
    """
    Orbit ETS (Exponential Smoothing) model.

    Bayesian exponential smoothing with Orbit's inference.
    Simpler than LGT/DLT but fast and robust.

    Args:
        seasonality: Seasonal period (None for non-seasonal)
        estimator: 'stan-mcmc', 'stan-map', or 'pyro-svi'
    """

    def __init__(
        self,
        seasonality: Optional[int] = None,
        estimator: str = 'stan-map',
        **kwargs
    ):
        super().__init__(seasonality=seasonality, estimator=estimator, **kwargs)

    def _create_model(self):
        from orbit.models import ETS
        return ETS(
            response_col='y',
            date_col='ds',
            seasonality=self.params.get('seasonality'),
            estimator=self.params.get('estimator', 'stan-map'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="OrbitETS",
            category=ModelCategory.PROBABILISTIC,
            library="orbit-ml",
            year=2021,
            paper="Ng et al. (2021)",
            probabilistic=True,
            notes="Bayesian exponential smoothing",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seasonality", "int", low=2, high=52, default=7),
            ParamSpace("estimator", "categorical", choices=['stan-mcmc', 'stan-map', 'pyro-svi'], default='stan-map'),
        ]
