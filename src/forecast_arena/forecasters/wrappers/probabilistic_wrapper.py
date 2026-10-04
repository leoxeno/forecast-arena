"""
forecasters/wrappers/probabilistic_wrapper.py - Probabilistic forecasting models.

Wraps models that provide uncertainty quantification:
- ConformalForecaster (MAPIE)
- NGBoost
- QuantileRegressor
- BayesianRidge
- GaussianProcess
"""

import numpy as np
from typing import List, Optional, Union, Tuple
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


def _create_single_lag_features(y: np.ndarray, n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create lagged features with single-step target for recursive forecasting.

    Returns X (n_samples, n_lags) and y_out (n_samples,).
    """
    n = len(y)
    n_samples = n - n_lags
    if n_samples < 1:
        raise ValueError(f"Series too short ({n}) for n_lags={n_lags}.")
    X = np.zeros((n_samples, n_lags))
    y_out = np.zeros(n_samples)
    for i in range(n_samples):
        X[i] = y[i:i + n_lags][::-1]
        y_out[i] = y[i + n_lags]
    return X, y_out


class _BaseProbabilisticForecaster(Forecaster):
    """Base class for probabilistic forecasters with uncertainty.

    Trains a single model on single-step targets and predicts recursively.
    Each recursive step returns both a point prediction and uncertainty estimate.
    """

    def __init__(self, n_lags: int = 10, **kwargs):
        super().__init__(n_lags=n_lags, **kwargs)
        self._n_lags = n_lags

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BaseProbabilisticForecaster':
        y = self._validate_y(y)
        self._train_y = y

        n_lags = self.params.get('n_lags', 10)
        self._n_lags = n_lags

        X_train, y_train = _create_single_lag_features(y, n_lags)
        self._X_train_feats = X_train

        self._model = self._create_model()
        self._fit_model(self._model, X_train, y_train)

        self._is_fitted = True
        return self

    def _compute_fitted_values(self):
        preds = self._model.predict(self._X_train_feats)
        fitted = np.full(len(self._train_y), np.nan)
        fitted[self._n_lags:self._n_lags + len(preds)] = preds
        return fitted

    def _fit_model(self, model: object, X: np.ndarray, y: np.ndarray):
        """Fit the model. Override for custom fit logic."""
        if model is not None:
            model.fit(X, y)

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        return_std: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        self._check_fitted()

        # Recursive: predict 1 step with uncertainty, shift lags, repeat
        last_values = list(self._train_y[-self._n_lags:])
        predictions = []
        stds = []

        for _ in range(horizon):
            features = np.array(last_values[-self._n_lags:][::-1]).reshape(1, -1)
            pred, std = self._predict_with_std(self._model, features)
            predictions.append(pred)
            stds.append(std)
            last_values.append(pred)

        if return_std:
            return np.array(predictions), np.array(stds)
        return np.array(predictions)

    def _predict_with_std(self, model: object, X: np.ndarray) -> Tuple[float, float]:
        """Return prediction and standard deviation. Override in subclass."""
        pred = model.predict(X)[0]
        return pred, 0.0  # Default: no uncertainty

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        if hasattr(self._model, "feature_importances_"):
            d["feature_importances"] = self._model.feature_importances_.tolist()
        if hasattr(self._model, "coef_"):
            coef = self._model.coef_
            d["coef"] = coef.tolist() if hasattr(coef, "tolist") else coef
        if hasattr(self._model, "intercept_"):
            d["intercept"] = float(self._model.intercept_)
        if hasattr(self._model, "sigma_"):
            sigma = self._model.sigma_
            if hasattr(sigma, "tolist"):
                d["sigma"] = sigma.tolist()
            else:
                d["sigma"] = float(sigma)
        if hasattr(self._model, "kernel_"):
            d["kernel"] = str(self._model.kernel_)
        return d

    def _create_model(self):
        """Override in subclass to return model instance."""
        raise NotImplementedError


@register_model
class MAPIEConformalForecaster(_BaseProbabilisticForecaster):
    """
    MAPIE Conformal Prediction for time series.

    Uses conformal inference for distribution-free prediction intervals.
    Wraps any point forecaster with calibrated uncertainty.

    Args:
        n_lags: Number of lag features
        alpha: Confidence level (1 - alpha = coverage)
        cv: Number of cross-validation folds for calibration
    """

    def __init__(
        self,
        n_lags: int = 10,
        alpha: float = 0.1,
        cv: int = 5,
        **kwargs
    ):
        super().__init__(n_lags=n_lags, alpha=alpha, cv=cv, **kwargs)

    def _create_model(self):
        from mapie.regression import CrossConformalRegressor
        from sklearn.ensemble import GradientBoostingRegressor

        base_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42,
        )
        return CrossConformalRegressor(
            estimator=base_model,
            cv=self.params.get('cv', 5),
            confidence_level=1.0 - self.params.get('alpha', 0.1),
        )

    def _fit_model(self, model: object, X: np.ndarray, y: np.ndarray):
        model.fit_conformalize(X, y)

    def _predict_with_std(self, model: object, X: np.ndarray) -> Tuple[float, float]:
        pred, intervals = model.predict_interval(X)
        # Estimate std from interval width
        interval_width = intervals[0, 1, 0] - intervals[0, 0, 0]
        std = interval_width / 4  # Approximate: 90% CI ≈ 4 std
        return pred[0], std

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="MAPIEConformalForecaster",
            category=ModelCategory.PROBABILISTIC,
            library="mapie",
            year=2021,
            paper="Taquet et al. (2022) - MAPIE",
            probabilistic=True,
            notes="Distribution-free conformal prediction intervals",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("alpha", "float", low=0.01, high=0.5, default=0.1),
            ParamSpace("cv", "int", low=3, high=10, default=5),
        ]


@register_model
class NGBoost(_BaseProbabilisticForecaster):
    """
    Natural Gradient Boosting for Probabilistic Prediction.

    Gradient boosting with natural gradient for probabilistic outputs.
    Directly models prediction distribution (mean + variance).

    Args:
        n_lags: Number of lag features
        n_estimators: Number of boosting stages
        learning_rate: Boosting learning rate
        minibatch_frac: Fraction of data per iteration
    """

    def __init__(
        self,
        n_lags: int = 10,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        minibatch_frac: float = 1.0,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            **kwargs
        )

    def _create_model(self):
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        return NGBRegressor(
            n_estimators=self.params.get('n_estimators', 500),
            learning_rate=self.params.get('learning_rate', 0.01),
            minibatch_frac=self.params.get('minibatch_frac', 1.0),
            Dist=Normal,
            random_state=42,
            verbose=False,
        )

    def _predict_with_std(self, model: object, X: np.ndarray) -> Tuple[float, float]:
        dist = model.pred_dist(X)
        return dist.mean()[0], dist.std()[0]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NGBoost",
            category=ModelCategory.PROBABILISTIC,
            library="ngboost",
            year=2020,
            paper="Duan et al. (2020) - NGBoost",
            probabilistic=True,
            notes="Natural gradient boosting for probabilistic prediction",
            github_url="https://github.com/stanfordmlgroup/ngboost",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("n_estimators", "int", low=100, high=1000, default=500),
            ParamSpace("learning_rate", "log_float", low=0.001, high=0.1, default=0.01),
        ]


@register_model
class QuantileRegressor(_BaseProbabilisticForecaster):
    """
    Quantile Regression for probabilistic forecasting.

    Fits multiple quantiles to estimate prediction distribution.

    Args:
        n_lags: Number of lag features
        quantiles: List of quantiles to fit (default: 0.1, 0.5, 0.9)
        alpha: Regularization strength
    """

    def __init__(
        self,
        n_lags: int = 10,
        quantiles: Optional[List[float]] = None,
        alpha: float = 1.0,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            quantiles=quantiles or [0.1, 0.5, 0.9],
            alpha=alpha,
            **kwargs
        )
        self._quantile_models = {}

    def _create_model(self):
        return {}  # Dict of quantile models, populated in _fit_model

    def _fit_model(self, model: object, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import QuantileRegressor as SkQuantile

        quantiles = self.params.get('quantiles', [0.1, 0.5, 0.9])
        alpha = self.params.get('alpha', 1.0)

        for q in quantiles:
            qr = SkQuantile(quantile=q, alpha=alpha, solver='highs')
            qr.fit(X, y)
            model[q] = qr  # model is a dict, mutated in place

    def _predict_with_std(self, model: object, X: np.ndarray) -> Tuple[float, float]:
        quantiles = self.params.get('quantiles', [0.1, 0.5, 0.9])

        # Median prediction
        median_q = 0.5 if 0.5 in model else quantiles[len(quantiles) // 2]
        pred = model[median_q].predict(X)[0]

        # Estimate std from quantile spread
        q_low = min(quantiles)
        q_high = max(quantiles)
        pred_low = model[q_low].predict(X)[0]
        pred_high = model[q_high].predict(X)[0]

        # Rough approximation: 80% interval spans ~2.56 std
        std = (pred_high - pred_low) / 2.56

        return pred, max(std, 0.0)

    def _compute_fitted_values(self):
        """Use median quantile model for fitted values."""
        if not self._model or not hasattr(self, '_X_train_feats'):
            return None
        median_q = 0.5 if 0.5 in self._model else list(self._model.keys())[len(self._model) // 2]
        preds = self._model[median_q].predict(self._X_train_feats)
        fitted = np.full(len(self._train_y), np.nan)
        fitted[self._n_lags:self._n_lags + len(preds)] = preds
        return fitted

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="QuantileRegressor",
            category=ModelCategory.PROBABILISTIC,
            library="sklearn",
            year=2021,
            paper="Koenker & Bassett (1978)",
            probabilistic=True,
            notes="Fits multiple quantiles for uncertainty estimation",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("alpha", "log_float", low=0.001, high=100.0, default=1.0),
        ]


@register_model
class BayesianRidge(_BaseProbabilisticForecaster):
    """
    Bayesian Ridge Regression for probabilistic forecasting.

    Ridge regression with Bayesian treatment for uncertainty.
    Provides posterior predictive distribution.

    Args:
        n_lags: Number of lag features
        alpha_1: Shape for Gamma prior on alpha
        alpha_2: Rate for Gamma prior on alpha
        lambda_1: Shape for Gamma prior on lambda
        lambda_2: Rate for Gamma prior on lambda
    """

    def __init__(
        self,
        n_lags: int = 10,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            **kwargs
        )

    def _create_model(self):
        from sklearn.linear_model import BayesianRidge as SkBayesianRidge

        return SkBayesianRidge(
            alpha_1=self.params.get('alpha_1', 1e-6),
            alpha_2=self.params.get('alpha_2', 1e-6),
            lambda_1=self.params.get('lambda_1', 1e-6),
            lambda_2=self.params.get('lambda_2', 1e-6),
        )

    def _predict_with_std(self, model: object, X: np.ndarray) -> Tuple[float, float]:
        pred, std = model.predict(X, return_std=True)
        return pred[0], std[0]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="BayesianRidge",
            category=ModelCategory.PROBABILISTIC,
            library="sklearn",
            year=1992,
            paper="MacKay (1992)",
            probabilistic=True,
            notes="Bayesian treatment of ridge regression",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("alpha_1", "log_float", low=1e-8, high=1e-2, default=1e-6),
            ParamSpace("lambda_1", "log_float", low=1e-8, high=1e-2, default=1e-6),
        ]


@register_model
class GaussianProcessRegressor(_BaseProbabilisticForecaster):
    """
    Gaussian Process Regression for probabilistic forecasting.

    Non-parametric Bayesian regression with kernel learning.
    Provides full posterior predictive distribution.

    Args:
        n_lags: Number of lag features
        kernel: Kernel type ('rbf', 'matern', 'rational_quadratic')
        alpha: Noise level regularization
        n_restarts_optimizer: Kernel hyperparameter optimization restarts
    """

    def __init__(
        self,
        n_lags: int = 10,
        kernel: str = 'rbf',
        alpha: float = 1e-2,
        n_restarts_optimizer: int = 5,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            **kwargs
        )

    def fit(self, y, X=None, freq=None, **kwargs):
        from sklearn.preprocessing import StandardScaler
        y = self._validate_y(y)
        self._train_y = y
        # GPR needs normalized features for kernel length scales to work
        self._scaler = StandardScaler()
        n_lags = self.params.get('n_lags', 10)
        self._n_lags = n_lags
        X_train, y_train = _create_single_lag_features(y, n_lags)
        self._X_train_feats = self._scaler.fit_transform(X_train)
        self._model = self._create_model()
        self._fit_model(self._model, self._X_train_feats, y_train)
        self._is_fitted = True
        return self

    def predict(self, horizon, X=None, return_std=False, **kwargs):
        self._check_fitted()
        last_values = list(self._train_y[-self._n_lags:])
        predictions = []
        stds = []
        for _ in range(horizon):
            features = np.array(last_values[-self._n_lags:][::-1]).reshape(1, -1)
            features = self._scaler.transform(features)
            pred, std = self._predict_with_std(self._model, features)
            predictions.append(pred)
            stds.append(std)
            last_values.append(pred)
        if return_std:
            return np.array(predictions), np.array(stds)
        return np.array(predictions)

    def _compute_fitted_values(self):
        preds = self._model.predict(self._X_train_feats)
        fitted = np.full(len(self._train_y), np.nan)
        fitted[self._n_lags:self._n_lags + len(preds)] = preds
        return fitted

    def _create_model(self):
        from sklearn.gaussian_process import GaussianProcessRegressor as SkGPR
        from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel

        kernel_name = self.params.get('kernel', 'rbf')
        n_lags = self.params.get('n_lags', 10)

        # Select kernel
        if kernel_name == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_lags))
        elif kernel_name == 'matern':
            kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_lags), nu=1.5)
        elif kernel_name == 'rational_quadratic':
            kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
        else:
            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_lags))

        return SkGPR(
            kernel=kernel,
            alpha=self.params.get('alpha', 1e-2),
            n_restarts_optimizer=self.params.get('n_restarts_optimizer', 5),
            normalize_y=True,
            random_state=42,
        )

    def _predict_with_std(self, model: object, X: np.ndarray) -> Tuple[float, float]:
        pred, std = model.predict(X, return_std=True)
        return pred[0], std[0]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GaussianProcessRegressor",
            category=ModelCategory.PROBABILISTIC,
            library="sklearn",
            year=2006,
            paper="Rasmussen & Williams (2006)",
            probabilistic=True,
            notes="Non-parametric Bayesian regression with uncertainty",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=30, default=10),
            ParamSpace("kernel", "categorical", choices=['rbf', 'matern', 'rational_quadratic'], default='rbf'),
            ParamSpace("alpha", "log_float", low=1e-6, high=1e-1, default=1e-2),
        ]
