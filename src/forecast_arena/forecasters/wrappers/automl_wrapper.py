"""
forecasters/wrappers/automl_wrapper.py - AutoML time series forecasters.

Wraps automated ML frameworks:
- AutoGluon: Multi-model ensembling
- FLAML: Fast, lightweight AutoML
- MLForecast: Nixtla's ML-based forecasting
"""

import numpy as np
from typing import List, Optional, Tuple, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


@register_model
class AutoGluonForecaster(Forecaster):
    """
    AutoGluon Time Series: Automated multi-model forecasting.

    Trains and ensembles multiple models automatically.
    Handles preprocessing, hyperparameter tuning, and model selection.

    Args:
        prediction_length: Forecast horizon
        time_limit: Training time limit in seconds
        preset: Quality preset ('fast_training', 'medium_quality', 'best_quality')
    """

    def __init__(
        self,
        prediction_length: int = 24,
        time_limit: int = 60,
        preset: str = 'medium_quality',
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            time_limit=time_limit,
            preset=preset,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'AutoGluonForecaster':
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        # Use horizon from kwargs (passed by runner/validation), fallback to param
        horizon = kwargs.get('horizon', self.params.get('prediction_length', 24))
        self._prediction_length = horizon

        # Create TimeSeriesDataFrame
        df = pd.DataFrame({
            'item_id': ['series'] * len(y),
            'timestamp': safe_date_range(start='2000-01-01', periods=len(y), freq=self._freq),
            'target': y
        })
        train_data = TimeSeriesDataFrame.from_data_frame(df)

        self._predictor = TimeSeriesPredictor(
            prediction_length=self._prediction_length,
            eval_metric='MASE',
            verbosity=0,
        )

        self._predictor.fit(
            train_data,
            time_limit=self.params.get('time_limit', 60),
            presets=self.params.get('preset', 'medium_quality'),
        )

        self._train_data = train_data
        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        predictions = self._predictor.predict(self._train_data)
        return predictions['mean'].values[:horizon]

    def _compute_fitted_values(self):
        if self._predictor is None or self._train_y is None:
            return None
        from autogluon.timeseries import TimeSeriesDataFrame

        n = len(self._train_y)
        pred_len = self._prediction_length
        if n < pred_len + 10:
            return None
        fitted = np.full(n, np.nan)
        # Rolling-origin: stride by pred_len, reuse trained predictor
        for end in range(pred_len + 10, n + 1, pred_len):
            trunc_df = pd.DataFrame({
                'item_id': ['series'] * end,
                'timestamp': safe_date_range(start='2000-01-01', periods=end, freq=self._freq),
                'target': self._train_y[:end]
            })
            trunc_data = TimeSeriesDataFrame.from_data_frame(trunc_df)
            predictions = self._predictor.predict(trunc_data)
            fcast = predictions['mean'].values
            h = min(len(fcast), n - end)
            if h > 0:
                fitted[end:end + h] = fcast[:h]
        return fitted

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="AutoGluonForecaster",
            category=ModelCategory.AUTOML,
            library="autogluon.timeseries",
            year=2020,
            paper="Erickson et al. (2020) - AutoGluon-Tabular",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("prediction_length", "int", low=1, high=48, default=12),
            ParamSpace("time_limit", "int", low=30, high=600, default=60),
            ParamSpace("preset", "categorical", choices=['fast_training', 'medium_quality', 'best_quality'], default='medium_quality'),
        ]


@register_model
class FLAMLForecaster(Forecaster):
    """
    FLAML: Fast and Lightweight AutoML.

    Microsoft's efficient AutoML with time series support.
    Uses recursive single-step prediction (FLAML selects single-output regressors).

    Args:
        prediction_length: Forecast horizon
        time_budget: Time budget in seconds
        estimator_list: List of estimators to try
    """

    def __init__(
        self,
        prediction_length: int = 24,
        time_budget: int = 60,
        estimator_list: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            time_budget=time_budget,
            estimator_list=estimator_list or ['lgbm', 'xgboost', 'rf'],
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'FLAMLForecaster':
        from flaml import AutoML

        y = self._validate_y(y)
        self._train_y = y

        # Single-output lag features
        n_lags = min(24, len(y) // 4)
        self._n_lags = n_lags
        X_train, y_train = self._create_features(y, n_lags)

        # FLAML finds the best single-step regressor
        self._model = AutoML()
        self._X_train_feats = X_train
        self._model.fit(
            X_train, y_train,
            task='regression',
            time_budget=self.params.get('time_budget', 60),
            estimator_list=self.params.get('estimator_list', ['lgbm', 'xgboost', 'rf']),
            verbose=0,
        )

        self._is_fitted = True
        return self

    def _create_features(self, y: np.ndarray, n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create lag features with single-output targets."""
        n = len(y)
        n_samples = n - n_lags
        if n_samples < 1:
            raise ValueError(
                f"Series too short ({n}) for n_lags={n_lags}. "
                f"Need at least {n_lags + 1} points."
            )
        X = np.zeros((n_samples, n_lags))
        y_out = np.zeros(n_samples)
        for i in range(n_samples):
            X[i] = y[i:i + n_lags][::-1]
            y_out[i] = y[i + n_lags]
        return X, y_out

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        # Recursive: predict 1 step, shift lags, repeat
        last_values = list(self._train_y[-self._n_lags:])
        predictions = []

        for _ in range(horizon):
            features = np.array(last_values[-self._n_lags:][::-1]).reshape(1, -1)
            pred = self._model.predict(features)[0]
            predictions.append(pred)
            last_values.append(pred)

        return np.array(predictions)

    def _compute_fitted_values(self):
        if self._model is None or not hasattr(self, '_X_train_feats'):
            return None
        preds = self._model.predict(self._X_train_feats)
        fitted = np.full(len(self._train_y), np.nan)
        fitted[self._n_lags:self._n_lags + len(preds)] = preds
        return fitted

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="FLAMLForecaster",
            category=ModelCategory.AUTOML,
            library="flaml",
            year=2021,
            paper="Wang et al. (2021) - FLAML: A Fast and Lightweight AutoML Library",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("prediction_length", "int", low=1, high=48, default=12),
            ParamSpace("time_budget", "int", low=30, high=600, default=60),
        ]


@register_model
class MLForecast(Forecaster):
    """
    MLForecast: Nixtla's ML-based time series forecasting.

    Combines feature engineering with ML models for fast forecasting.
    Direct multi-step: MLForecast library handles multi-step prediction internally.

    Args:
        prediction_length: Forecast horizon
        lags: List of lag values to use as features
        model: Model type ('lgbm', 'xgboost', 'linear')
    """

    def __init__(
        self,
        prediction_length: int = 24,
        lags: Optional[List[int]] = None,
        model: str = 'lgbm',
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            lags=lags or [1, 2, 3, 4, 5, 6, 7, 12, 24],
            model=model,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'MLForecast':
        from mlforecast import MLForecast as MF
        from mlforecast.target_transforms import Differences

        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'
        self._prediction_length = kwargs.get('horizon', self.params.get('prediction_length', 24))

        # Create dataframe
        df = pd.DataFrame({
            'unique_id': ['series'] * len(y),
            'ds': safe_date_range(start='2000-01-01', periods=len(y), freq=self._freq),
            'y': y
        })

        # Select model
        model_type = self.params.get('model', 'lgbm')
        if model_type == 'lgbm':
            from lightgbm import LGBMRegressor
            models = [LGBMRegressor(verbosity=-1, n_estimators=100)]
        elif model_type == 'xgboost':
            from xgboost import XGBRegressor
            models = [XGBRegressor(verbosity=0, n_estimators=100)]
        else:
            from sklearn.linear_model import Ridge
            models = [Ridge()]

        self._mlf = MF(
            models=models,
            freq=self._freq,
            lags=self.params.get('lags', [1, 2, 3, 4, 5, 6, 7, 12, 24]),
            target_transforms=[Differences([1])],
        )

        self._mlf.fit(df)
        self._train_df = df
        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        predictions = self._mlf.predict(h=horizon)
        # Get the model column (first model)
        model_col = [c for c in predictions.columns if c not in ['unique_id', 'ds']][0]
        return predictions[model_col].values[:horizon]

    def _compute_fitted_values(self):
        if not hasattr(self, '_mlf') or self._mlf is None or self._train_y is None:
            return None
        from mlforecast import MLForecast as MF
        from mlforecast.target_transforms import Differences

        n = len(self._train_y)
        pred_len = self._prediction_length
        fitted = np.full(n, np.nan)
        lags = self.params.get('lags', [1, 2, 3, 4, 5, 6, 7, 12, 24])
        model_type = self.params.get('model', 'lgbm')

        # Rolling-origin with fresh MLForecast objects (LightGBM refit is <0.1s)
        for end in range(pred_len + max(lags) + 2, n + 1, pred_len):
            trunc_df = self._train_df.iloc[:end].copy()
            if model_type == 'lgbm':
                from lightgbm import LGBMRegressor
                models = [LGBMRegressor(verbosity=-1, n_estimators=100)]
            elif model_type == 'xgboost':
                from xgboost import XGBRegressor
                models = [XGBRegressor(verbosity=0, n_estimators=100)]
            else:
                from sklearn.linear_model import Ridge
                models = [Ridge()]
            mlf = MF(models=models, freq=self._freq, lags=lags,
                     target_transforms=[Differences([1])])
            mlf.fit(trunc_df)
            preds = mlf.predict(h=pred_len)
            model_col = [c for c in preds.columns if c not in ['unique_id', 'ds']][0]
            fcast = preds[model_col].values
            h = min(len(fcast), n - end)
            if h > 0:
                fitted[end:end + h] = fcast[:h]
        return fitted

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="MLForecast",
            category=ModelCategory.AUTOML,
            library="mlforecast",
            year=2022,
            paper="Nixtla - MLForecast: Scalable machine learning for time series",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("prediction_length", "int", low=1, high=48, default=12),
            ParamSpace("model", "categorical", choices=['lgbm', 'xgboost', 'linear'], default='lgbm'),
        ]
