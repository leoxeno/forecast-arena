"""
forecasters/wrappers/tsai_wrapper.py - tsai library forecasters.

Wraps tsai models (fastai-style deep learning):
- InceptionTime
- ResNet
- TSTPlus (Time Series Transformer)

tsai uses a sliding window approach for forecasting, converting the time series
into (X, y) pairs where X is the lookback window and y is the forecast target.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


def _create_sliding_windows(y: np.ndarray, lookback: int, horizon: int):
    """
    Create sliding window dataset for tsai.

    Args:
        y: Time series array
        lookback: Input window size
        horizon: Forecast horizon (output size)

    Returns:
        X: Input windows, shape (n_samples, 1, lookback)
        Y: Target values, shape (n_samples, horizon)
    """
    n = len(y)
    n_samples = n - lookback - horizon + 1

    if n_samples < 1:
        raise ValueError(
            f"Series too short ({n}) for lookback={lookback}, horizon={horizon}. "
            f"Need at least {lookback + horizon} observations."
        )

    X = np.zeros((n_samples, 1, lookback))
    Y = np.zeros((n_samples, horizon))

    for i in range(n_samples):
        X[i, 0, :] = y[i:i + lookback]
        Y[i, :] = y[i + lookback:i + lookback + horizon]

    return X, Y


def _register_if_tsai_model_available(model_import_path: str, model_name: str):
    """
    Decorator to only register model if it can be imported from tsai.

    Args:
        model_import_path: e.g., 'tsai.models.InceptionTime'
        model_name: e.g., 'InceptionTime'
    """
    def decorator(cls):
        try:
            import importlib
            module = importlib.import_module(model_import_path)
            if hasattr(module, model_name):
                return register_model(cls)
            return cls
        except (ImportError, ModuleNotFoundError):
            return cls
    return decorator


class _BaseTsaiForecaster(Forecaster):
    """Base class for tsai-based forecasters.

    Uses direct fastai Learner instead of TSRegressor to avoid
    custom_head compatibility issues in tsai 0.4+.

    Applies instance normalization (z-score) before training and reverses
    it on outputs. This is essential for CNN/RNN architectures (FCN, ResNet,
    InceptionTime, etc.) that lack built-in RevIN. Without normalization,
    large-scale input values cause these models to produce near-zero forecasts.
    PatchTST has built-in RevIN, so it skips external normalization.
    """

    def _has_builtin_revin(self) -> bool:
        """Return True if the architecture has built-in RevIN normalization.

        Override in subclasses that handle their own normalization (e.g. PatchTST).
        """
        return False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BaseTsaiForecaster':
        from tsai.all import get_ts_dls
        from fastai.learner import Learner
        from fastai.losses import MSELossFlat
        import warnings

        y = self._validate_y(y)
        self._train_y = y

        lookback = self.params.get('lookback', 24)
        horizon = kwargs.get('horizon', self.params.get('horizon', 12))
        self.params['horizon'] = horizon
        self._lookback = lookback
        self._horizon = horizon

        # Instance normalization for models without built-in RevIN.
        # Store mean/std so we can reverse the transform on outputs.
        if not self._has_builtin_revin():
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y))
            if self._y_std < 1e-8:
                self._y_std = 1.0  # constant series guard
            y_train = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            y_train = y

        # Create sliding window dataset
        X_windows, Y_windows = _create_sliding_windows(y_train, lookback, horizon)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Get dataloaders
            dls = get_ts_dls(
                X_windows, Y_windows,
                bs=self.params.get('batch_size', 32),
            )

            # Create model directly with architecture config
            c_in = 1  # univariate
            c_out = horizon
            seq_len = lookback
            arch = self._get_architecture()
            arch_config = self._get_arch_config()

            # Some models don't accept seq_len (e.g., FCN, ResCNN)
            try:
                self._model = arch(c_in, c_out, seq_len=seq_len, **arch_config)
            except TypeError:
                self._model = arch(c_in, c_out, **arch_config)

            # Create learner with default_cbs=False to disable ProgressCallback
            # This fixes 'NBMasterBar' errors in Jupyter environments
            self._learner = Learner(
                dls, self._model, loss_func=MSELossFlat(),
                default_cbs=False,
            )

            # Ensure model and data are on the same device (CPU)
            # default_cbs=False disables device management callbacks
            import torch
            self._model = self._model.cpu()
            self._learner.model = self._model
            if hasattr(dls, 'device'):
                dls.device = torch.device('cpu')

            n_epochs = self.params.get('n_epochs', 25)
            lr = self.params.get('lr', 1e-3)

            # Train silently
            self._learner.fit_one_cycle(n_epochs, lr)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        import torch

        if horizon != self._horizon:
            raise ValueError(
                f"Prediction horizon ({horizon}) must match training horizon "
                f"({self._horizon}). Retrain with horizon={horizon}."
            )

        # Use last lookback values as input, normalized
        last_window = self._train_y[-self._lookback:]
        if not self._has_builtin_revin():
            last_window = (last_window - self._y_mean) / self._y_std
        X_pred = torch.tensor(last_window.reshape(1, 1, -1), dtype=torch.float32)

        # Move to same device as model
        device = next(self._model.parameters()).device
        X_pred = X_pred.to(device)

        # Get prediction
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_pred)

        forecast = pred.cpu().numpy().flatten()[:horizon]

        # Reverse normalization
        if not self._has_builtin_revin():
            forecast = forecast * self._y_std + self._y_mean

        return forecast

    def _compute_fitted_values(self):
        if self._model is None or self._train_y is None:
            return None
        import torch

        # Normalize input windows the same way as training
        y_for_windows = self._train_y
        if not self._has_builtin_revin():
            y_for_windows = (self._train_y - self._y_mean) / self._y_std

        X_windows, _ = _create_sliding_windows(y_for_windows, self._lookback, self._horizon)
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        device = next(self._model.parameters()).device
        X_tensor = X_tensor.to(device)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_tensor)
        preds = preds.cpu().numpy()
        # Some architectures (e.g. PatchTST) output 3D: (batch, 1, horizon)
        while preds.ndim > 2:
            preds = preds.squeeze(1)

        # Reverse normalization on predictions
        if not self._has_builtin_revin():
            preds = preds * self._y_std + self._y_mean

        fitted = np.full(len(self._train_y), np.nan)
        # Each prediction[i] gives horizon outputs; use 1-step-ahead (col 0)
        if preds.ndim == 2:
            fitted[self._lookback:self._lookback + len(preds)] = preds[:, 0]
        else:
            fitted[self._lookback:self._lookback + len(preds)] = preds
        return fitted

    def _get_architecture(self):
        """Override in subclass to return tsai architecture class."""
        raise NotImplementedError

    def get_diagnostics(self):
        if not self._is_fitted:
            return {}
        d = {"lookback": self._lookback, "horizon": self._horizon}
        if hasattr(self, "_learner") and hasattr(self._learner, "recorder"):
            try:
                losses = self._learner.recorder.losses
                d["final_train_loss"] = float(losses[-1]) if losses else None
                d["n_epochs"] = len(self._learner.recorder.lrs) if hasattr(self._learner.recorder, "lrs") else None
            except Exception:
                pass
        return d

    def _get_arch_config(self) -> dict:
        """Override to customize architecture config."""
        return {}


@register_model
class TsaiInceptionTime(_BaseTsaiForecaster):
    """
    InceptionTime: State-of-the-art CNN for time series.

    Ensemble of Inception modules with residual connections.
    Won most benchmarks on UCR archive.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            **kwargs
        )

    def _get_architecture(self):
        from tsai.models.InceptionTime import InceptionTime
        return InceptionTime

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiInceptionTime",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2020,
            paper="Fawaz et al. (2020)",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("n_epochs", "int", low=10, high=100, default=25),
            ParamSpace("lr", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@register_model
class TsaiResNet(_BaseTsaiForecaster):
    """
    ResNet for time series forecasting.

    Residual network adapted for 1D time series data.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            **kwargs
        )

    def _get_architecture(self):
        from tsai.models.ResNet import ResNet
        return ResNet

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiResNet",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2017,
            paper="Wang et al. (2017)",
            github_url="https://github.com/timeseriesAI/tsai",
        )


@register_model
class TsaiTST(_BaseTsaiForecaster):
    """
    TST: Time Series Transformer.

    Transformer architecture adapted for time series with
    positional encoding and attention mechanisms.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            **kwargs
        )

    def _get_architecture(self):
        from tsai.models.TSTPlus import TSTPlus
        return TSTPlus

    def _get_arch_config(self) -> dict:
        return {
            'd_model': self.params.get('d_model', 128),
            'n_heads': self.params.get('n_heads', 4),
            'n_layers': self.params.get('n_layers', 2),
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiTST",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2021,
            paper="Zerveas et al. (2021)",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("d_model", "categorical", choices=[64, 128, 256], default=128),
            ParamSpace("n_heads", "categorical", choices=[2, 4, 8], default=4),
            ParamSpace("n_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class TsaiROCKET(Forecaster):
    """
    ROCKET: Random Convolutional Kernel Transform.

    Ultra-fast feature extraction using random convolutions.
    State-of-the-art accuracy with linear classifier.

    Note: Uses standalone RocketRegressor (not TSRegressor architecture).

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        num_kernels: Number of random kernels (default 10000)
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        num_kernels: int = 10000,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            num_kernels=num_kernels,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'TsaiROCKET':
        from tsai.models.ROCKET import RocketRegressor
        import warnings

        y = self._validate_y(y)
        self._train_y = y

        lookback = self.params.get('lookback', 24)
        horizon = kwargs.get('horizon', self.params.get('horizon', 12))
        self._lookback = lookback
        self._horizon = horizon

        # Create sliding window dataset
        X_windows, Y_windows = _create_sliding_windows(y, lookback, horizon)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = RocketRegressor(
                num_kernels=self.params.get('num_kernels', 10000),
            )

            # Patch for sklearn 1.5+ compatibility (tsai's RocketRegressor
            # is a Pipeline subclass missing the new transform_input attribute)
            if not hasattr(self._model, 'transform_input'):
                self._model.transform_input = None

            # Fit - RocketRegressor expects (samples, channels, length)
            self._model.fit(X_windows, Y_windows)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        if horizon != self._horizon:
            raise ValueError(
                f"Prediction horizon ({horizon}) must match training horizon "
                f"({self._horizon}). Retrain with horizon={horizon}."
            )

        # Use last lookback values as input
        last_window = self._train_y[-self._lookback:]
        X_pred = last_window.reshape(1, 1, -1)

        # Get prediction
        preds = self._model.predict(X_pred)
        return np.asarray(preds).flatten()[:horizon]

    def _compute_fitted_values(self):
        if self._model is None or self._train_y is None:
            return None
        X_windows, _ = _create_sliding_windows(self._train_y, self._lookback, self._horizon)
        preds = self._model.predict(X_windows)
        preds = np.asarray(preds)
        fitted = np.full(len(self._train_y), np.nan)
        if preds.ndim == 2:
            fitted[self._lookback:self._lookback + len(preds)] = preds[:, 0]
        else:
            fitted[self._lookback:self._lookback + len(preds)] = preds
        return fitted

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        try:
            # RocketRegressor is a Pipeline: kernel transform → ridge regression
            ridge = self._model.steps[-1][1] if hasattr(self._model, "steps") else None
            if ridge is not None:
                if hasattr(ridge, "coef_"):
                    d["ridge_coef_shape"] = list(ridge.coef_.shape)
                if hasattr(ridge, "intercept_"):
                    intercept = ridge.intercept_
                    d["ridge_intercept"] = float(intercept) if not hasattr(intercept, "__len__") else [float(x) for x in intercept]
        except Exception:
            pass
        return d

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiROCKET",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2020,
            paper="Dempster et al. (2020) - ROCKET: Exceptionally fast and accurate time series classification",
            github_url="https://github.com/timeseriesAI/tsai",
            notes="Ultra-fast feature extraction",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("num_kernels", "categorical", choices=[1000, 5000, 10000, 20000], default=10000),
        ]


@register_model
class TsaiXceptionTime(_BaseTsaiForecaster):
    """
    XceptionTime: Xception-based architecture for time series.

    Depthwise separable convolutions adapted from Xception.
    Efficient and accurate for time series classification.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            **kwargs
        )

    def _get_architecture(self):
        from tsai.models.XceptionTime import XceptionTime
        return XceptionTime

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiXceptionTime",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2019,
            paper="Rahimian et al. (2019) - XceptionTime: Independent Time-Window XceptionTime Architecture",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("n_epochs", "int", low=10, high=100, default=25),
            ParamSpace("lr", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@register_model
class TsaiLSTMFCN(_BaseTsaiForecaster):
    """
    LSTM-FCN: LSTM with Fully Convolutional Network.

    Combines LSTM for temporal dependencies with FCN for
    feature extraction. Strong performer on many benchmarks.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            **kwargs
        )

    def _get_architecture(self):
        from tsai.models.RNN_FCN import LSTM_FCN
        return LSTM_FCN

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiLSTMFCN",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2018,
            paper="Karim et al. (2018) - LSTM Fully Convolutional Networks for Time Series Classification",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("n_epochs", "int", low=10, high=100, default=25),
            ParamSpace("lr", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@register_model
class TsaiPatchTST(_BaseTsaiForecaster):
    """
    PatchTST via tsai: Patching Time Series Transformer.

    Segments time series into patches like Vision Transformer.
    Channel-independent processing for multivariate series.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        patch_len: Length of each patch
        stride: Stride between patches
    """

    def __init__(
        self,
        lookback: int = 48,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        patch_len: int = 12,
        stride: int = 6,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            patch_len=patch_len,
            stride=stride,
            **kwargs
        )

    def _has_builtin_revin(self) -> bool:
        """PatchTST has built-in RevIN normalization."""
        return True

    def _get_architecture(self):
        from tsai.models.PatchTST import PatchTST
        return PatchTST

    def _get_arch_config(self) -> dict:
        return {
            'patch_len': self.params.get('patch_len', 12),
            'stride': self.params.get('stride', 6),
            'pred_dim': self.params.get('horizon', 12),  # Required for regression output
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiPatchTST",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2023,
            paper="Nie et al. (2023) - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=24, high=96, default=48),
            ParamSpace("patch_len", "int", low=8, high=24, default=12),
            ParamSpace("stride", "int", low=4, high=12, default=6),
            ParamSpace("n_epochs", "int", low=10, high=100, default=25),
        ]


@register_model
class TsaiMiniRocket(Forecaster):
    """
    MiniRocket: Minimally Random Convolutional Kernel Transform.

    Faster, more memory-efficient variant of ROCKET.
    Near-deterministic kernels for reproducibility.

    Note: Uses standalone MiniRocketRegressor (not TSRegressor architecture).

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        num_features: Number of features
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        num_features: int = 10000,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            num_features=num_features,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'TsaiMiniRocket':
        from tsai.models.MINIROCKET import MiniRocketRegressor
        import warnings

        y = self._validate_y(y)
        self._train_y = y

        lookback = self.params.get('lookback', 24)
        horizon = kwargs.get('horizon', self.params.get('horizon', 12))
        self._lookback = lookback
        self._horizon = horizon

        # Create sliding window dataset
        X_windows, Y_windows = _create_sliding_windows(y, lookback, horizon)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = MiniRocketRegressor(
                num_features=self.params.get('num_features', 10000),
            )

            # Patch for sklearn 1.5+ compatibility (tsai's MiniRocketRegressor
            # is a Pipeline subclass missing the new transform_input attribute)
            if not hasattr(self._model, 'transform_input'):
                self._model.transform_input = None

            self._model.fit(X_windows, Y_windows)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        if horizon != self._horizon:
            raise ValueError(
                f"Prediction horizon ({horizon}) must match training horizon "
                f"({self._horizon}). Retrain with horizon={horizon}."
            )

        # Use last lookback values as input
        last_window = self._train_y[-self._lookback:]
        X_pred = last_window.reshape(1, 1, -1)

        # Get prediction
        preds = self._model.predict(X_pred)
        return np.asarray(preds).flatten()[:horizon]

    def _compute_fitted_values(self):
        if self._model is None or self._train_y is None:
            return None
        X_windows, _ = _create_sliding_windows(self._train_y, self._lookback, self._horizon)
        preds = self._model.predict(X_windows)
        preds = np.asarray(preds)
        fitted = np.full(len(self._train_y), np.nan)
        if preds.ndim == 2:
            fitted[self._lookback:self._lookback + len(preds)] = preds[:, 0]
        else:
            fitted[self._lookback:self._lookback + len(preds)] = preds
        return fitted

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        try:
            ridge = self._model.steps[-1][1] if hasattr(self._model, "steps") else None
            if ridge is not None:
                if hasattr(ridge, "coef_"):
                    d["ridge_coef_shape"] = list(ridge.coef_.shape)
                if hasattr(ridge, "intercept_"):
                    intercept = ridge.intercept_
                    d["ridge_intercept"] = float(intercept) if not hasattr(intercept, "__len__") else [float(x) for x in intercept]
        except Exception:
            pass
        return d

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiMiniRocket",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2021,
            paper="Dempster et al. (2021) - MiniRocket: A Very Fast (Almost) Deterministic Transform",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("num_features", "categorical", choices=[1000, 5000, 10000], default=10000),
        ]


@register_model
class TsaiFCN(_BaseTsaiForecaster):
    """
    FCN: Fully Convolutional Network for time series.

    Simple yet effective CNN architecture for classification/regression.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            **kwargs
        )

    def _get_architecture(self):
        from tsai.models.FCN import FCN
        return FCN

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiFCN",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2017,
            paper="Wang et al. (2017) - Time Series Classification from Scratch",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("n_epochs", "int", low=10, high=100, default=25),
            ParamSpace("lr", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@register_model
class TsaiResCNN(_BaseTsaiForecaster):
    """
    ResCNN: Residual CNN for time series.

    CNN with residual connections for deeper networks.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            **kwargs
        )

    def _get_architecture(self):
        from tsai.models.ResCNN import ResCNN
        return ResCNN

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiResCNN",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2019,
            paper="Zou et al. (2019) - Integration of residual network and CNN for classification",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("n_epochs", "int", low=10, high=100, default=25),
            ParamSpace("lr", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@register_model
class TsaiOmniScaleCNN(_BaseTsaiForecaster):
    """
    OmniScale CNN: Multi-scale CNN for time series.

    Captures patterns at multiple temporal scales.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            **kwargs
        )

    def _get_architecture(self):
        from tsai.models.OmniScaleCNN import OmniScaleCNN
        return OmniScaleCNN

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaiOmniScaleCNN",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2022,
            paper="Tang et al. (2022) - Omni-Scale CNNs",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("n_epochs", "int", low=10, high=100, default=25),
            ParamSpace("lr", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@register_model
class TsaimWDN(_BaseTsaiForecaster):
    """
    mWDN: Multi-scale Wavelet Decomposition Network.

    Uses wavelet transforms for multi-resolution analysis.

    Args:
        lookback: Input window size
        horizon: Forecast horizon
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(
            lookback=lookback,
            horizon=horizon,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            **kwargs
        )

    def _get_architecture(self):
        from tsai.models.mWDN import mWDN
        return mWDN

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TsaimWDN",
            category=ModelCategory.DEEP_LEARNING,
            library="tsai",
            year=2020,
            paper="Wang et al. (2020) - Multilevel Wavelet Decomposition Network",
            github_url="https://github.com/timeseriesAI/tsai",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lookback", "int", low=12, high=48, default=24),
            ParamSpace("n_epochs", "int", low=10, high=100, default=25),
            ParamSpace("lr", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]
