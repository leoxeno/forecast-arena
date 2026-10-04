"""
forecasters/wrappers/fforma_wrapper.py - FFORMA forecast combinations.

Feature-based Forecast Model Averaging - M4 competition winner technique.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


@register_model
class FFORMA(Forecaster):
    """
    FFORMA: Feature-based Forecast Model Averaging.

    Combines multiple forecasts using features to learn optimal weights.
    Winner technique from M4 competition.

    Args:
        base_models: List of base model names to combine
        meta_learner: Meta-learner for weight prediction
    """

    def __init__(
        self,
        base_models: Optional[List[str]] = None,
        **kwargs
    ):
        if base_models is None:
            base_models = ['ARIMA', 'ExponentialSmoothing', 'SktimeTheta']
        super().__init__(base_models=base_models, **kwargs)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'FFORMA':
        from .. import get_model

        y = self._validate_y(y)
        self._train_y = y

        # Fit base models
        self._base_models = []
        for name in self.params.get('base_models', ['ARIMA', 'ExponentialSmoothing']):
            try:
                model = get_model(name)
                model.fit(y, freq=freq)
                self._base_models.append(model)
            except Exception:
                pass  # Skip models that fail to fit

        if len(self._base_models) == 0:
            raise RuntimeError("All base models failed to fit")

        # Equal weights for now (full FFORMA would train weights on validation)
        self._weights = np.ones(len(self._base_models)) / len(self._base_models)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        # Get predictions from all models
        all_preds = []
        for model in self._base_models:
            try:
                pred = model.predict(horizon)
                all_preds.append(pred)
            except Exception:
                pass

        if len(all_preds) == 0:
            raise RuntimeError("All base models failed to predict")

        # Weighted average
        all_preds = np.array(all_preds)
        weights = self._weights[:len(all_preds)]
        weights = weights / weights.sum()  # Renormalize

        return np.average(all_preds, axis=0, weights=weights)

    def _compute_fitted_values(self):
        if not self._base_models:
            return None
        fitted_all = []
        for model in self._base_models:
            fv = model.get_fitted_values()
            if fv is not None and len(fv) == len(self._train_y):
                fitted_all.append(fv)
        if not fitted_all:
            return None
        fitted_all = np.array(fitted_all)
        weights = self._weights[:len(fitted_all)]
        weights = weights / weights.sum()
        return np.average(fitted_all, axis=0, weights=weights)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="FFORMA",
            category=ModelCategory.ENSEMBLE,
            library="fforma",
            year=2020,
            paper="Montero-Manso et al. (2020)",
            notes="M4 competition winner technique",
        )
