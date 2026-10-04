"""
forecasters/wrappers/sklearn_wrapper.py - Scikit-learn ML forecasters.

Wraps ML regressors for time series via direct multi-output prediction:
- RandomForest
- GradientBoosting
- XGBoost (if installed)
- LightGBM (if installed)
- Ridge, Lasso

Strategy: Models that natively support multi-output (RandomForest, ExtraTrees,
DecisionTree, KNN, Ridge) train a single model on 2D targets.
Lasso/ElasticNet use their MultiTask variants for native multi-output.
Models that don't (SVR, GradientBoosting, HistGradientBoosting, AdaBoost,
XGBoost, LightGBM, CatBoost) use recursive single-step prediction.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


def _create_multioutput_lag_features(
    y: np.ndarray, n_lags: int, horizon: int,
    exog: Optional[np.ndarray] = None,
) -> tuple:
    """Create lagged features with multi-output targets for direct forecasting.

    Returns X (n_samples, n_features) and Y (n_samples, horizon).
    When exog is provided, X columns are [lags..., exog_cols...].
    """
    n = len(y)
    n_samples = n - n_lags - horizon + 1
    if n_samples < 1:
        raise ValueError(
            f"Series too short ({n}) for n_lags={n_lags} + horizon={horizon}. "
            f"Need at least {n_lags + horizon} points."
        )
    X_lags = np.zeros((n_samples, n_lags))
    Y = np.zeros((n_samples, horizon))
    for i in range(n_samples):
        X_lags[i] = y[i:i + n_lags][::-1]
        Y[i] = y[i + n_lags:i + n_lags + horizon]

    if exog is not None:
        # Align exog: for sample i, use exog at time i + n_lags (the prediction origin)
        X_exog = exog[n_lags:n_lags + n_samples]
        X = np.hstack([X_lags, X_exog])
    else:
        X = X_lags
    return X, Y


def _create_single_lag_features(
    y: np.ndarray, n_lags: int,
    exog: Optional[np.ndarray] = None,
) -> tuple:
    """Create lagged features with single-step target for recursive forecasting.

    Returns X (n_samples, n_features) and y_out (n_samples,).
    When exog is provided, X columns are [lags..., exog_cols...].
    """
    n = len(y)
    n_samples = n - n_lags
    if n_samples < 1:
        raise ValueError(f"Series too short ({n}) for n_lags={n_lags}.")
    X_lags = np.zeros((n_samples, n_lags))
    y_out = np.zeros(n_samples)
    for i in range(n_samples):
        X_lags[i] = y[i:i + n_lags][::-1]
        y_out[i] = y[i + n_lags]

    if exog is not None:
        # Align exog: for sample i, use exog at time i + n_lags (the target time)
        X_exog = exog[n_lags:n_lags + n_samples]
        X = np.hstack([X_lags, X_exog])
    else:
        X = X_lags
    return X, y_out


class _BaseSklearnForecaster(Forecaster):
    """Base class for sklearn-based forecasters.

    Models that natively support multi-output (RF, ET, DT, KNN, Ridge,
    MultiTaskLasso, MultiTaskElasticNet) train a single model on 2D targets.
    Models that don't (SVR, GradientBoosting, AdaBoost, XGBoost, LightGBM,
    CatBoost, etc.) train on single-step targets and forecast recursively.
    """

    # Override to True in subclasses whose _create_model() returns an estimator
    # that natively handles 2D targets (e.g. RandomForestRegressor, Ridge).
    _native_multioutput: bool = False

    def __init__(self, n_lags: int = 10, **kwargs):
        super().__init__(n_lags=n_lags, **kwargs)
        self._n_lags = n_lags
        self._horizon: int = 24

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BaseSklearnForecaster':
        y = self._validate_y(y)
        self._train_y = y

        # Validate and store exogenous features
        exog = self._validate_X(X, len(y)) if X is not None else None
        self._train_X = exog
        self._has_exog = exog is not None

        n_lags = self.params.get('n_lags', 10)
        self._n_lags = n_lags
        horizon = kwargs.get('horizon', 24)
        self._horizon = horizon

        self._model = self._create_model()
        if self._native_multioutput:
            X_train, Y_train = _create_multioutput_lag_features(y, n_lags, horizon, exog)
            self._model.fit(X_train, Y_train)
            self._X_train_feats = X_train
        else:
            X_train, y_train = _create_single_lag_features(y, n_lags, exog)
            self._model.fit(X_train, y_train)
            self._X_train_feats = X_train

        self._is_fitted = True
        return self

    def _compute_fitted_values(self):
        preds = self._model.predict(self._X_train_feats)
        n_lags = self._n_lags
        fitted = np.full(len(self._train_y), np.nan)
        if self._native_multioutput:
            # Multi-output: each row predicts H steps; use 1-step-ahead (col 0)
            fitted[n_lags:n_lags + len(preds)] = preds[:, 0]
        else:
            # Single-step: each prediction is 1-step-ahead
            fitted[n_lags:n_lags + len(preds)] = preds
        return fitted

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        # Validate future exogenous if model was fit with exog
        exog_future = None
        if self._has_exog:
            exog_future = self._validate_X(X, horizon) if X is not None else None

        if self._native_multioutput:
            if horizon > self._horizon:
                raise ValueError(
                    f"Requested horizon {horizon} > trained horizon {self._horizon}. "
                    f"Re-fit with horizon={horizon}."
                )
            lags = np.array(self._train_y[-self._n_lags:][::-1]).reshape(1, -1)
            if exog_future is not None:
                # Use first row of future exog for the one-shot multi-output prediction
                features = np.hstack([lags, exog_future[0:1]])
            else:
                features = lags
            return self._model.predict(features)[0][:horizon]
        else:
            # Recursive: predict one step, append, repeat
            last_values = list(self._train_y[-self._n_lags:])
            predictions = []
            for step in range(horizon):
                lags = np.array(last_values[-self._n_lags:][::-1]).reshape(1, -1)
                if exog_future is not None:
                    features = np.hstack([lags, exog_future[step:step + 1]])
                else:
                    features = lags
                pred = self._model.predict(features)[0]
                predictions.append(pred)
                last_values.append(pred)
            return np.array(predictions)

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
            intercept = self._model.intercept_
            d["intercept"] = float(intercept) if not hasattr(intercept, "__len__") else [float(x) for x in intercept]
        if hasattr(self._model, "n_iter_") and self._model.n_iter_ is not None:
            n = self._model.n_iter_
            d["n_iter"] = int(n) if not hasattr(n, "__len__") else [int(x) for x in n]
        return d

    def _create_model(self):
        """Override in subclass to return sklearn model instance."""
        raise NotImplementedError


@register_model
class RandomForest(_BaseSklearnForecaster):
    """
    Random Forest regressor for time series.

    Args:
        n_lags: Number of lag features
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf
    """

    _native_multioutput = True

    def __init__(
        self,
        n_lags: int = 8,
        n_estimators: int = 50,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 1,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            **kwargs
        )

    def _create_model(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=self.params.get('n_estimators', 50),
            max_depth=self.params.get('max_depth', 10),
            min_samples_leaf=self.params.get('min_samples_leaf', 1),
            random_state=42,
            n_jobs=-1,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="RandomForest",
            category=ModelCategory.ML,
            library="sklearn",
            year=2001,
            paper="Breiman (2001)",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("n_estimators", "int", low=50, high=500, default=100),
            ParamSpace("max_depth", "int", low=3, high=20, default=10),
        ]


@register_model
class GradientBoosting(_BaseSklearnForecaster):
    """
    Gradient Boosting regressor for time series.

    Args:
        n_lags: Number of lag features
        n_estimators: Number of boosting stages
        learning_rate: Shrinkage factor
        max_depth: Maximum tree depth
    """

    def __init__(
        self,
        n_lags: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            **kwargs
        )

    def _create_model(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=self.params.get('n_estimators', 100),
            learning_rate=self.params.get('learning_rate', 0.1),
            max_depth=self.params.get('max_depth', 3),
            random_state=42,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GradientBoosting",
            category=ModelCategory.ML,
            library="sklearn",
            year=2001,
            paper="Friedman (2001)",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("n_estimators", "int", low=50, high=500, default=100),
            ParamSpace("learning_rate", "log_float", low=0.01, high=0.3, default=0.1),
            ParamSpace("max_depth", "int", low=2, high=10, default=3),
        ]


@register_model
class XGBoost(_BaseSklearnForecaster):
    """
    XGBoost regressor for time series.

    Args:
        n_lags: Number of lag features
        n_estimators: Number of boosting rounds
        learning_rate: Step size shrinkage
        max_depth: Maximum tree depth
    """

    def __init__(
        self,
        n_lags: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            **kwargs
        )

    def _create_model(self):
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        return XGBRegressor(
            n_estimators=self.params.get('n_estimators', 100),
            learning_rate=self.params.get('learning_rate', 0.1),
            max_depth=self.params.get('max_depth', 6),
            random_state=42,
            verbosity=0,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="XGBoost",
            category=ModelCategory.ML,
            library="xgboost",
            year=2016,
            paper="Chen & Guestrin (2016)",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("n_estimators", "int", low=50, high=500, default=100),
            ParamSpace("learning_rate", "log_float", low=0.01, high=0.3, default=0.1),
            ParamSpace("max_depth", "int", low=2, high=10, default=6),
        ]


@register_model
class LightGBM(_BaseSklearnForecaster):
    """
    LightGBM regressor for time series.

    Args:
        n_lags: Number of lag features
        n_estimators: Number of boosting iterations
        learning_rate: Boosting learning rate
        num_leaves: Maximum number of leaves per tree
    """

    def __init__(
        self,
        n_lags: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            **kwargs
        )

    def _create_model(self):
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        return LGBMRegressor(
            n_estimators=self.params.get('n_estimators', 100),
            learning_rate=self.params.get('learning_rate', 0.1),
            num_leaves=self.params.get('num_leaves', 31),
            random_state=42,
            verbosity=-1,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="LightGBM",
            category=ModelCategory.ML,
            library="lightgbm",
            year=2017,
            paper="Ke et al. (2017)",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("n_estimators", "int", low=50, high=500, default=100),
            ParamSpace("learning_rate", "log_float", low=0.01, high=0.3, default=0.1),
            ParamSpace("num_leaves", "int", low=15, high=127, default=31),
        ]


@register_model
class Ridge(_BaseSklearnForecaster):
    """
    Ridge regression for time series.

    Args:
        n_lags: Number of lag features
        alpha: Regularization strength
    """

    _native_multioutput = True

    def __init__(self, n_lags: int = 10, alpha: float = 1.0, **kwargs):
        super().__init__(n_lags=n_lags, alpha=alpha, **kwargs)

    def _create_model(self):
        from sklearn.linear_model import Ridge as SkRidge
        return SkRidge(alpha=self.params.get('alpha', 1.0))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Ridge",
            category=ModelCategory.ML,
            library="sklearn",
            year=1970,
            paper="Hoerl & Kennard (1970)",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("alpha", "log_float", low=0.001, high=100.0, default=1.0),
        ]


@register_model
class Lasso(_BaseSklearnForecaster):
    """
    Lasso regression for time series (L1 regularization).

    Performs feature selection by shrinking some coefficients to zero.
    Uses MultiTaskLasso for native multi-output with shared sparsity.

    Args:
        n_lags: Number of lag features
        alpha: Regularization strength (L1 penalty)
    """

    _native_multioutput = True

    def __init__(self, n_lags: int = 10, alpha: float = 1.0, **kwargs):
        super().__init__(n_lags=n_lags, alpha=alpha, **kwargs)

    def _create_model(self):
        from sklearn.linear_model import MultiTaskLasso
        return MultiTaskLasso(
            alpha=self.params.get('alpha', 1.0),
            random_state=42,
            max_iter=5000,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Lasso",
            category=ModelCategory.ML,
            library="sklearn",
            year=1996,
            paper="Tibshirani (1996)",
            notes="L1 regularization for feature selection",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("alpha", "log_float", low=0.001, high=100.0, default=1.0),
        ]


@register_model
class ElasticNet(_BaseSklearnForecaster):
    """
    ElasticNet regression for time series (L1 + L2 regularization).

    Combines Ridge and Lasso penalties.
    Uses MultiTaskElasticNet for native multi-output with shared sparsity.

    Args:
        n_lags: Number of lag features
        alpha: Overall regularization strength
        l1_ratio: Mix between L1 and L2 (0=Ridge, 1=Lasso)
    """

    _native_multioutput = True

    def __init__(
        self,
        n_lags: int = 10,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__(n_lags=n_lags, alpha=alpha, l1_ratio=l1_ratio, **kwargs)

    def _create_model(self):
        from sklearn.linear_model import MultiTaskElasticNet
        return MultiTaskElasticNet(
            alpha=self.params.get('alpha', 1.0),
            l1_ratio=self.params.get('l1_ratio', 0.5),
            random_state=42,
            max_iter=5000,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ElasticNet",
            category=ModelCategory.ML,
            library="sklearn",
            year=2005,
            paper="Zou & Hastie (2005)",
            notes="Combines L1 and L2 regularization",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("alpha", "log_float", low=0.001, high=100.0, default=1.0),
            ParamSpace("l1_ratio", "float", low=0.0, high=1.0, default=0.5),
        ]


@register_model
class SVR(_BaseSklearnForecaster):
    """
    Support Vector Regression for time series.

    Uses kernel methods for non-linear regression.

    Args:
        n_lags: Number of lag features
        C: Regularization parameter
        epsilon: Epsilon-tube width
        kernel: Kernel type ('rbf', 'linear', 'poly')
    """

    def __init__(
        self,
        n_lags: int = 10,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = 'rbf',
        **kwargs
    ):
        super().__init__(n_lags=n_lags, C=C, epsilon=epsilon, kernel=kernel, **kwargs)

    def _create_model(self):
        from sklearn.svm import SVR as SkSVR
        return SkSVR(
            C=self.params.get('C', 1.0),
            epsilon=self.params.get('epsilon', 0.1),
            kernel=self.params.get('kernel', 'rbf'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SVR",
            category=ModelCategory.ML,
            library="sklearn",
            year=1996,
            paper="Drucker et al. (1996)",
            notes="Kernel-based regression",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("C", "log_float", low=0.1, high=100.0, default=1.0),
            ParamSpace("epsilon", "log_float", low=0.01, high=0.5, default=0.1),
            ParamSpace("kernel", "categorical", choices=['rbf', 'linear', 'poly'], default='rbf'),
        ]


@register_model
class KNN(_BaseSklearnForecaster):
    """
    K-Nearest Neighbors regression for time series.

    Non-parametric method that finds similar historical patterns.

    Args:
        n_lags: Number of lag features
        n_neighbors: Number of neighbors to use
        weights: Weight function ('uniform' or 'distance')
    """

    _native_multioutput = True

    def __init__(
        self,
        n_lags: int = 10,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        **kwargs
    ):
        super().__init__(n_lags=n_lags, n_neighbors=n_neighbors, weights=weights, **kwargs)

    def _create_model(self):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(
            n_neighbors=self.params.get('n_neighbors', 5),
            weights=self.params.get('weights', 'uniform'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="KNN",
            category=ModelCategory.ML,
            library="sklearn",
            year=1967,
            paper="Cover & Hart (1967)",
            notes="Non-parametric pattern matching",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("n_neighbors", "int", low=1, high=20, default=5),
            ParamSpace("weights", "categorical", choices=['uniform', 'distance'], default='uniform'),
        ]


@register_model
class DecisionTree(_BaseSklearnForecaster):
    """
    Decision Tree regressor for time series.

    Simple interpretable model, useful as baseline.

    Args:
        n_lags: Number of lag features
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf
    """

    _native_multioutput = True

    def __init__(
        self,
        n_lags: int = 10,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            **kwargs
        )

    def _create_model(self):
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(
            max_depth=self.params.get('max_depth'),
            min_samples_leaf=self.params.get('min_samples_leaf', 1),
            random_state=42,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DecisionTree",
            category=ModelCategory.ML,
            library="sklearn",
            year=1984,
            paper="Breiman et al. (1984) - CART",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("max_depth", "int", low=3, high=20, default=10),
            ParamSpace("min_samples_leaf", "int", low=1, high=10, default=1),
        ]


@register_model
class ExtraTrees(_BaseSklearnForecaster):
    """
    Extra Trees (Extremely Randomized Trees) regressor.

    More randomization than Random Forest, often faster.

    Args:
        n_lags: Number of lag features
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf
    """

    _native_multioutput = True

    def __init__(
        self,
        n_lags: int = 8,
        n_estimators: int = 50,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            **kwargs
        )

    def _create_model(self):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=self.params.get('n_estimators', 50),
            max_depth=self.params.get('max_depth'),
            min_samples_leaf=self.params.get('min_samples_leaf', 1),
            random_state=42,
            n_jobs=-1,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ExtraTrees",
            category=ModelCategory.ML,
            library="sklearn",
            year=2006,
            paper="Geurts et al. (2006) - Extremely Randomized Trees",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("n_estimators", "int", low=50, high=500, default=100),
            ParamSpace("max_depth", "int", low=3, high=20, default=10),
        ]


@register_model
class AdaBoost(_BaseSklearnForecaster):
    """
    AdaBoost regressor for time series.

    Sequential ensemble that focuses on hard examples.

    Args:
        n_lags: Number of lag features
        n_estimators: Number of boosting stages
        learning_rate: Learning rate shrinkage
    """

    def __init__(
        self,
        n_lags: int = 10,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            **kwargs
        )

    def _create_model(self):
        from sklearn.ensemble import AdaBoostRegressor
        return AdaBoostRegressor(
            n_estimators=self.params.get('n_estimators', 50),
            learning_rate=self.params.get('learning_rate', 1.0),
            random_state=42,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="AdaBoost",
            category=ModelCategory.ML,
            library="sklearn",
            year=1997,
            paper="Freund & Schapire (1997) - A Decision-Theoretic Generalization of On-Line Learning",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("n_estimators", "int", low=25, high=200, default=50),
            ParamSpace("learning_rate", "log_float", low=0.01, high=2.0, default=1.0),
        ]


@register_model
class CatBoost(_BaseSklearnForecaster):
    """
    CatBoost regressor for time series.

    Gradient boosting with ordered boosting and categorical support.

    Args:
        n_lags: Number of lag features
        iterations: Number of boosting iterations
        learning_rate: Boosting learning rate
        depth: Tree depth
    """

    def __init__(
        self,
        n_lags: int = 10,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 6,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            **kwargs
        )

    def _create_model(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")

        return CatBoostRegressor(
            iterations=self.params.get('iterations', 100),
            learning_rate=self.params.get('learning_rate', 0.1),
            depth=self.params.get('depth', 6),
            random_state=42,
            verbose=False,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="CatBoost",
            category=ModelCategory.ML,
            library="catboost",
            year=2018,
            paper="Prokhorenkova et al. (2018) - CatBoost: Unbiased Boosting with Categorical Features",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("iterations", "int", low=50, high=500, default=100),
            ParamSpace("learning_rate", "log_float", low=0.01, high=0.3, default=0.1),
            ParamSpace("depth", "int", low=3, high=12, default=6),
        ]


@register_model
class HistGradientBoosting(_BaseSklearnForecaster):
    """
    Histogram-based Gradient Boosting regressor.

    Fast native sklearn implementation inspired by LightGBM.

    Args:
        n_lags: Number of lag features
        max_iter: Maximum boosting iterations
        learning_rate: Learning rate
        max_depth: Maximum tree depth
    """

    def __init__(
        self,
        n_lags: int = 10,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            n_lags=n_lags,
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            **kwargs
        )

    def _create_model(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=self.params.get('max_iter', 100),
            learning_rate=self.params.get('learning_rate', 0.1),
            max_depth=self.params.get('max_depth'),
            random_state=42,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="HistGradientBoosting",
            category=ModelCategory.ML,
            library="sklearn",
            year=2019,
            paper="sklearn 0.21 - Inspired by LightGBM",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_lags", "int", low=5, high=50, default=10),
            ParamSpace("max_iter", "int", low=50, high=500, default=100),
            ParamSpace("learning_rate", "log_float", low=0.01, high=0.3, default=0.1),
            ParamSpace("max_depth", "int", low=3, high=15, default=10),
        ]
