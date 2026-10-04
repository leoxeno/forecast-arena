"""
forecasters/wrappers/greykite_wrapper.py - LinkedIn Greykite.

Wraps LinkedIn's Greykite Silverkite forecaster.

Note: greykite has complex dependencies and may not be installed.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


def _register_if_available(module_path: str):
    """Only register model if its module exists."""
    def decorator(cls):
        try:
            import importlib
            importlib.import_module(module_path)
            return register_model(cls)
        except (ImportError, ModuleNotFoundError):
            return cls
    return decorator


@_register_if_available('greykite.framework.templates.forecaster')
class Silverkite(Forecaster):
    """
    LinkedIn Silverkite forecaster.

    Flexible decomposition-based forecaster from LinkedIn.

    Args:
        freq: Data frequency
        coverage: Prediction interval coverage
    """

    def __init__(
        self,
        freq: str = 'D',
        coverage: float = 0.95,
        **kwargs
    ):
        super().__init__(freq=freq, coverage=coverage, **kwargs)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'Silverkite':
        try:
            from greykite.framework.templates.autogen.forecast_config import ForecastConfig
            from greykite.framework.templates.forecaster import Forecaster as GKForecaster
            from greykite.framework.templates.model_templates import ModelTemplateEnum
        except ImportError:
            raise ImportError(
                "greykite not installed. Install with: pip install greykite"
            )

        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or self.params.get('freq', 'D')

        df = pd.DataFrame({
            'ts': safe_date_range(start='2000-01-01', periods=len(y), freq=self._freq),
            'y': y
        })

        self._forecaster = GKForecaster()
        result = self._forecaster.run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                forecast_horizon=12,
                coverage=self.params.get('coverage', 0.95),
            )
        )
        self._result = result

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        forecast = self._result.forecast
        return forecast['forecast'].values[-horizon:]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Silverkite",
            category=ModelCategory.CLASSICAL,
            library="greykite",
            year=2021,
            paper="Reza et al. (2021)",
            probabilistic=True,
            github_url="https://github.com/linkedin/greykite",
        )
