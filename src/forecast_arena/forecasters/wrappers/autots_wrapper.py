"""
forecasters/wrappers/autots_wrapper.py - AutoTS AutoML.

Wraps AutoTS for automatic model selection and ensembling.
Direct multi-step: AutoTS internally selects and ensembles models
that produce the full forecast horizon in one pass. No recursive prediction.
"""

import io
import contextlib
import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


@register_model
class AutoTS(Forecaster):
    """
    AutoTS - Automatic Time Series model selection.

    Automatically selects best model from ensemble of candidates.
    Produces direct multi-step forecasts (no recursive single-step).

    Args:
        forecast_length: Forecast horizon (overridden by horizon kwarg in fit)
        frequency: Data frequency ('infer' for auto)
        ensemble: Ensemble method ('auto', 'simple', etc.)
        max_generations: Genetic algorithm generations
        num_validations: CV folds
    """

    def __init__(
        self,
        forecast_length: int = 24,
        frequency: str = 'infer',
        ensemble: str = 'auto',
        max_generations: int = 5,
        num_validations: int = 2,
        **kwargs
    ):
        super().__init__(
            forecast_length=forecast_length,
            frequency=frequency,
            ensemble=ensemble,
            max_generations=max_generations,
            num_validations=num_validations,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'AutoTS':
        try:
            from autots import AutoTS as ATS
        except ImportError:
            raise ImportError("autots not installed. Install with: pip install autots")

        y = self._validate_y(y)
        self._train_y = y

        # Use horizon from kwargs (passed by runner/validation), fallback to param
        horizon = kwargs.get('horizon', self.params.get('forecast_length', 24))
        self._horizon = horizon

        df = pd.DataFrame({
            'date': safe_date_range(start='2000-01-01', periods=len(y), freq=freq or 'D'),
            'value': y
        })

        self._model = ATS(
            forecast_length=horizon,
            frequency=self.params.get('frequency', 'infer'),
            ensemble=self.params.get('ensemble', 'auto'),
            max_generations=self.params.get('max_generations', 5),
            num_validations=self.params.get('num_validations', 2),
            verbose=-1,
        )

        # Suppress AutoTS template eval error spam (bypasses verbose flag)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self._model = self._model.fit(df, date_col='date', value_col='value')

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        prediction = self._model.predict()
        return prediction.forecast.values.flatten()[:horizon]

    def _compute_fitted_values(self):
        if self._model is None or self._train_y is None:
            return None
        n = len(self._train_y)
        fitted = np.full(n, np.nan)
        # AutoTS back_forecast returns a PredictionObject with .forecast attr
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            back = self._model.back_forecast(n_splits=3, verbose=0)
        # back_forecast returns PredictionObject (has .forecast) or dict
        if hasattr(back, 'forecast'):
            forecasts_df = back.forecast
        elif isinstance(back, dict):
            forecasts_df = back.get('forecasts', back.get('forecast'))
        else:
            forecasts_df = back
        if forecasts_df is not None and hasattr(forecasts_df, 'values'):
            vals = forecasts_df.values.flatten()
            fill_len = min(len(vals), n)
            if fill_len > 0:
                fitted[n - fill_len:n] = vals[-fill_len:]
        return fitted if np.any(~np.isnan(fitted)) else None

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        try:
            # Best model selected by genetic algorithm — NOT derivable
            if hasattr(self._model, "best_model_name"):
                d["best_model_name"] = str(self._model.best_model_name)
            if hasattr(self._model, "best_model_params"):
                d["best_model_params"] = {
                    str(k): str(v) for k, v in self._model.best_model_params.items()
                }
            if hasattr(self._model, "best_model_transformation_params"):
                d["best_transformation_params"] = str(
                    self._model.best_model_transformation_params
                )
        except Exception:
            pass
        return d

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="AutoTS",
            category=ModelCategory.AUTOML,
            library="autots",
            year=2021,
            paper="N/A",
            probabilistic=True,
            github_url="https://github.com/winedarksea/AutoTS",
        )
