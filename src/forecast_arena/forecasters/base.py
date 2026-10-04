"""
forecasters/base.py - Abstract base class for time series forecasters.

Provides unified fit/predict interface for 50+ models with:
- Consistent API across all libraries
- Parameter space definitions for tuning
- Metadata for model census
- Probabilistic forecasting support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import os
import numpy as np
import pandas as pd


def is_fast_mode() -> bool:
    """Check if fast mode is enabled for quick validation testing.

    Set FORECAST_ARENA_FAST_MODE=1 to enable minimal training epochs/steps.
    Useful for CI/CD and quick model validation.
    """
    return os.environ.get('FORECAST_ARENA_FAST_MODE', '0') == '1'


def get_fast_epochs(default: int) -> int:
    """Get epochs for training, reduced if fast mode is enabled."""
    return 3 if is_fast_mode() else default


def get_fast_steps(default: int) -> int:
    """Get training steps, reduced if fast mode is enabled."""
    return 50 if is_fast_mode() else default


class ModelCategory(str, Enum):
    """Model categories for census organization."""
    CLASSICAL = "classical"
    ML = "ml"
    DEEP_LEARNING = "deep_learning"
    LLM = "llm"
    AUTOML = "automl"
    PROBABILISTIC = "probabilistic"
    ENSEMBLE = "ensemble"
    VOLATILITY = "volatility"  # GARCH/ARCH models - forecast variance, not level
    INTERMITTENT = "intermittent"  # Croston/IMAPA/TSB - for sparse demand data with many zeros


@dataclass
class ParamSpace:
    """
    Parameter space definition for hyperparameter tuning.

    Supports: categorical, integer range, float range, log-uniform.
    """
    name: str
    param_type: str  # 'categorical', 'int', 'float', 'log_float'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Any = None

    def sample_random(self, rng: np.random.Generator) -> Any:
        """Sample a random value from this parameter space."""
        if self.param_type == 'categorical':
            return rng.choice(self.choices)
        elif self.param_type == 'int':
            return rng.integers(int(self.low), int(self.high) + 1)
        elif self.param_type == 'float':
            return rng.uniform(self.low, self.high)
        elif self.param_type == 'log_float':
            log_val = rng.uniform(np.log(self.low), np.log(self.high))
            return np.exp(log_val)
        raise ValueError(f"Unknown param_type: {self.param_type}")

    def to_optuna(self, trial, name: Optional[str] = None):
        """Convert to optuna trial suggestion."""
        param_name = name or self.name
        if self.param_type == 'categorical':
            return trial.suggest_categorical(param_name, self.choices)
        elif self.param_type == 'int':
            return trial.suggest_int(param_name, int(self.low), int(self.high))
        elif self.param_type == 'float':
            return trial.suggest_float(param_name, self.low, self.high)
        elif self.param_type == 'log_float':
            return trial.suggest_float(param_name, self.low, self.high, log=True)
        raise ValueError(f"Unknown param_type: {self.param_type}")


@dataclass
class ModelMetadata:
    """Metadata for model census documentation."""
    name: str
    category: ModelCategory
    library: str  # pip package name
    year: int
    paper: str  # citation or "N/A"

    # Capabilities
    probabilistic: bool = False
    multivariate: bool = False
    zero_shot: bool = False
    exogenous: bool = False
    online_learning: bool = False

    # GPU requirements (for remote execution routing)
    requires_gpu: bool = False  # If True, model needs GPU (e.g., Mamba-SSM)
    min_gpu_memory_gb: float = 0.0  # Minimum GPU VRAM in GB

    # Access
    huggingface_id: Optional[str] = None
    github_url: Optional[str] = None

    # Notes
    notes: str = ""

    @property
    def prefers_gpu(self) -> bool:
        """Returns True if model benefits from GPU (deep learning or LLM category)."""
        return self.requires_gpu or self.category in (
            ModelCategory.DEEP_LEARNING,
            ModelCategory.LLM,
        )


class Forecaster(ABC):
    """
    Abstract base class for time series forecasters.

    All model wrappers inherit from this and implement:
        - fit(y, X=None) -> self
        - predict(horizon, X=None) -> np.ndarray
        - get_metadata() -> ModelMetadata
        - get_param_space() -> List[ParamSpace]

    Usage:
        model = SomeForecaster(param1=value1)
        model.fit(train_y)
        forecast = model.predict(horizon=12)

        # Or use convenience method
        forecast = model.fit_predict(train_y, horizon=12)

        # Probabilistic models
        quantiles = model.predict_quantiles(horizon=12, quantiles=[0.1, 0.5, 0.9])
    """

    def __init__(self, **kwargs):
        """
        Initialize forecaster with hyperparameters.

        Args:
            **kwargs: Model-specific hyperparameters
        """
        self._model = None
        self._is_fitted = False
        self._train_y: Optional[np.ndarray] = None
        self._train_X: Optional[np.ndarray] = None
        self._freq: Optional[str] = None
        self._fitted_values: Optional[np.ndarray] = None
        self.params = kwargs

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @abstractmethod
    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'Forecaster':
        """
        Fit model to training data.

        Args:
            y: Target time series (1D array or Series)
            X: Exogenous features (optional, 2D array or DataFrame)
            freq: Frequency string (e.g., 'D', 'M', 'H')
            **kwargs: Additional fit parameters

        Returns:
            self for method chaining

        Note:
            Implementations should:
            1. Store training data in self._train_y, self._train_X
            2. Set self._is_fitted = True after fitting
            3. Return self
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate point forecasts.

        Args:
            horizon: Number of steps to forecast
            X: Future exogenous features (required if fit with X)
            **kwargs: Additional predict parameters

        Returns:
            1D numpy array of forecasts, shape (horizon,)

        Raises:
            RuntimeError: If model not fitted
        """
        pass

    @staticmethod
    @abstractmethod
    def get_metadata() -> ModelMetadata:
        """Return model metadata for census."""
        pass

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        """
        Return hyperparameter search space for tuning.

        Override this method to define tunable parameters.
        Default returns empty list (no tuning).

        Returns:
            List of ParamSpace definitions
        """
        return []

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate quantile forecasts for probabilistic models.

        Args:
            horizon: Number of steps to forecast
            quantiles: List of quantiles (e.g., [0.1, 0.5, 0.9])
            X: Future exogenous features
            **kwargs: Additional parameters

        Returns:
            2D numpy array, shape (horizon, len(quantiles))

        Raises:
            NotImplementedError: If model doesn't support probabilistic forecasts
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support quantile forecasts. "
            f"Check get_metadata().probabilistic before calling."
        )

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.9,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.

        Args:
            horizon: Number of steps to forecast
            coverage: Interval coverage (e.g., 0.9 for 90%)
            X: Future exogenous features

        Returns:
            Tuple of (lower, upper) arrays, each shape (horizon,)
        """
        alpha = (1 - coverage) / 2
        quantiles = [alpha, 1 - alpha]
        q_forecasts = self.predict_quantiles(horizon, quantiles, X, **kwargs)
        return q_forecasts[:, 0], q_forecasts[:, 1]

    def fit_predict(
        self,
        y: Union[np.ndarray, pd.Series],
        horizon: int,
        X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        X_future: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Convenience method: fit then predict.

        Args:
            y: Target time series
            horizon: Forecast horizon
            X_train: Training exogenous features
            X_future: Future exogenous features
            freq: Frequency string

        Returns:
            Point forecasts array
        """
        self.fit(y, X=X_train, freq=freq, **kwargs)
        return self.predict(horizon, X=X_future, **kwargs)

    def get_fitted_values(self) -> Optional[np.ndarray]:
        """Get in-sample fitted values from the trained model.

        Computed lazily on first call (does not affect fit/predict timing).
        Returns None if the model doesn't support fitted values (e.g., zero-shot).
        """
        if not self._is_fitted:
            return None
        if self._fitted_values is None:
            try:
                self._fitted_values = self._compute_fitted_values()
            except Exception:
                pass  # Silently return None if computation fails
        return self._fitted_values

    def _compute_fitted_values(self) -> Optional[np.ndarray]:
        """Override in subclass to compute in-sample fitted values.

        Should return a 1D array of length len(self._train_y), or None.
        NaN values are allowed for time points where no prediction is possible
        (e.g., initial lag window for sklearn models).
        """
        return None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Extract library-specific model internals after fitting.

        Override in subclass to return a dict of diagnostic data that is only
        available from the live fitted model object (AIC/BIC, feature
        importances, coefficients, convergence status, etc.).

        Called once after fit/predict by the runner. The returned dict must be
        JSON-serializable (use plain Python types, not numpy).

        Returns:
            Dict of diagnostic key-value pairs. Empty dict if nothing to report.
        """
        return {}

    def clone(self, **override_params) -> 'Forecaster':
        """
        Create a new unfitted instance with same (or overridden) parameters.

        Args:
            **override_params: Parameters to override

        Returns:
            New unfitted Forecaster instance
        """
        new_params = {**self.params, **override_params}
        return self.__class__(**new_params)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters as dict."""
        return self.params.copy()

    def set_params(self, **params) -> 'Forecaster':
        """Set parameters (returns self for chaining)."""
        self.params.update(params)
        return self

    def _check_fitted(self):
        """Raise error if model not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )

    def _validate_y(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Convert y to numpy array and validate."""
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y).flatten()
        if len(y) < 2:
            raise ValueError("y must have at least 2 observations")
        if np.any(np.isnan(y)):
            raise ValueError("y contains NaN values")
        return y

    def _validate_X(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]],
        expected_len: int
    ) -> Optional[np.ndarray]:
        """Convert X to numpy array and validate shape."""
        if X is None:
            return None
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) != expected_len:
            raise ValueError(
                f"X length ({len(X)}) must match expected ({expected_len})"
            )
        return X

    def __repr__(self):
        meta = self.get_metadata()
        status = "fitted" if self._is_fitted else "unfitted"
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        if params_str:
            return f"{meta.name}({params_str}) [{status}]"
        return f"{meta.name}() [{status}]"
