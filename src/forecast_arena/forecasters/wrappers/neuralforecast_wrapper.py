# Modified for forecast-arena: extracted and adapted from AION benchmark utilities, October 2026.
"""
forecasters/wrappers/neuralforecast_wrapper.py - Nixtla NeuralForecast.

Wraps Nixtla's neural network models with automatic hyperparameter tuning:
- LSTM, GRU, RNN
- NBEATS, NHITS
- TFT, Autoformer, Informer
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd
import os
import sys
import types

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace, get_fast_steps
from ..date_utils import safe_date_range
from .. import register_model


def _should_stub_nf_gpu_models() -> bool:
    """Return True when optional CUDA-only NF models should be stubbed out.

    This prevents import-time failures when CUDA toolchain isn't available
    but only CPU models are being used locally.
    """
    if os.environ.get("FORECAST_ARENA_ALLOW_LOCAL_NF_GPU", "0") == "1":
        return False

    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    cuda_home = os.environ.get("CUDA_HOME")
    return (not cuda_available) or (not cuda_home)


def _install_stub_module(module_name: str, class_name: str, reason: str) -> None:
    """Install a stub module with a single class that raises on use."""
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    class _StubModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(reason)

    _StubModel.__name__ = class_name
    setattr(stub, class_name, _StubModel)
    sys.modules[module_name] = stub


def _prepare_neuralforecast_import() -> None:
    """Patch optional CUDA-only models to avoid import-time failures."""
    if not _should_stub_nf_gpu_models():
        return

    _install_stub_module(
        "neuralforecast.models.xlstm",
        "xLSTM",
        "xLSTM requires CUDA (mlstm_kernels). Use a GPU host or set "
        "FORECAST_ARENA_ALLOW_LOCAL_NF_GPU=1 with a working CUDA toolkit.",
    )
    _install_stub_module(
        "neuralforecast.models.timellm",
        "TimeLLM",
        "TimeLLM requires GPU + transformers. Use a GPU host or set "
        "FORECAST_ARENA_ALLOW_LOCAL_NF_GPU=1 with required deps installed.",
    )


def _import_neuralforecast():
    """Safely import NeuralForecast, stubbing CUDA-only models if needed."""
    _prepare_neuralforecast_import()
    try:
        from neuralforecast import NeuralForecast
        return NeuralForecast
    except Exception as exc:
        msg = str(exc)
        if "CUDA_HOME" in msg:
            raise RuntimeError(
                "NeuralForecast import failed because CUDA_HOME is not set. "
                "This is usually triggered by optional CUDA-only models. "
                "Either run on a GPU host or set "
                "FORECAST_ARENA_ALLOW_LOCAL_NF_GPU=1 with a working CUDA toolkit."
            ) from exc
        raise


class _BaseNeuralForecast(Forecaster):
    """Base class for NeuralForecast models."""

    # Override in subclass to enable exogenous support.
    # When True, X columns are passed as futr_exog_list to the model constructor
    # and futr_df to predict().
    _supports_exog: bool = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        horizon: Optional[int] = None,
        **kwargs
    ) -> '_BaseNeuralForecast':
        NeuralForecast = _import_neuralforecast()

        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        # Allow horizon override at fit time (important for validation)
        # Default to 24 if not specified anywhere
        if horizon is not None:
            self.params['horizon'] = horizon
        elif 'horizon' not in self.params:
            self.params['horizon'] = 24  # Default for validation notebooks
        self._horizon = self.params['horizon']

        df = pd.DataFrame({
            'unique_id': ['series'] * len(y),
            'ds': safe_date_range(start='2000-01-01', periods=len(y), freq=self._freq),
            'y': y
        })

        # Handle exogenous features
        self._exog_cols: list = []
        if X is not None and self._supports_exog:
            exog = self._validate_X(X, len(y))
            if exog is not None:
                for j in range(exog.shape[1]):
                    col_name = f'exog_{j}'
                    df[col_name] = exog[:, j]
                    self._exog_cols.append(col_name)

        self._nf = NeuralForecast(
            models=[self._create_model()],
            freq=self._freq,
        )
        self._nf.fit(df=df)
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

        # Warn if requested horizon exceeds model's trained horizon
        if horizon > self._horizon:
            import warnings
            warnings.warn(
                f"Requested horizon ({horizon}) exceeds model's trained horizon ({self._horizon}). "
                f"Returning {self._horizon} steps. Re-fit with horizon={horizon} for full forecast."
            )

        # Build futr_df for exogenous features at predict time
        futr_df = None
        if self._exog_cols:
            last_date = self._df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=self._horizon + 1, freq=self._freq)[1:]
            futr_df = pd.DataFrame({
                'unique_id': ['series'] * self._horizon,
                'ds': future_dates,
            })
            exog_future = self._validate_X(X, self._horizon) if X is not None else None
            for j, col in enumerate(self._exog_cols):
                if exog_future is not None:
                    futr_df[col] = exog_future[:, j]
                else:
                    futr_df[col] = 0.0

        forecast = self._nf.predict(futr_df=futr_df)
        # Get column name (model name)
        model_col = [c for c in forecast.columns if c not in ['unique_id', 'ds']][0]
        return forecast[model_col].values[:horizon]

    def get_diagnostics(self):
        if not self._is_fitted or not hasattr(self, "_nf"):
            return {}
        d = {}
        try:
            model = self._nf.models[0]
            # Actual epochs trained (may differ from max due to early stopping)
            if hasattr(model, "trainer") and model.trainer is not None:
                d["epochs_trained"] = int(model.trainer.current_epoch)
            # Final training loss — NOT derivable (includes regularization/dropout)
            if hasattr(model, "trainer") and model.trainer is not None:
                metrics = model.trainer.callback_metrics
                if "train_loss" in metrics:
                    d["final_train_loss"] = float(metrics["train_loss"])
            # Learning rate actually used
            if hasattr(model, "learning_rate"):
                d["learning_rate"] = float(model.learning_rate)
        except Exception:
            pass
        return d

    def _compute_fitted_values(self):
        if not hasattr(self, '_nf') or self._nf is None:
            return None
        insample_df = self._nf.predict_insample(step_size=self._horizon)
        if insample_df is None or len(insample_df) == 0:
            return None
        _meta_cols = {'unique_id', 'ds', 'cutoff', 'y'}
        pred_cols = [c for c in insample_df.columns if c not in _meta_cols]
        if not pred_cols:
            return None
        return insample_df[pred_cols[0]].values.astype(float)

    def _create_model(self):
        raise NotImplementedError


class _BaseNeuralForecastMultivariate(Forecaster):
    """
    Base class for MULTIVARIATE NeuralForecast models.

    These models (iTransformer, TSMixer, SOFTS, etc.) require n_series parameter
    and process multiple time series jointly for cross-series pattern learning.

    Input: 2D array of shape [n_samples, n_series]
    Output: 2D array of shape [horizon, n_series]
    """

    def fit(
        self,
        y: Union[np.ndarray, pd.DataFrame],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        horizon: Optional[int] = None,
        **kwargs
    ) -> '_BaseNeuralForecastMultivariate':
        NeuralForecast = _import_neuralforecast()

        # Handle multivariate input - don't flatten!
        if isinstance(y, pd.DataFrame):
            y = y.values
        y = np.asarray(y)

        if y.ndim == 1:
            raise ValueError(
                f"{self.__class__.__name__} requires multivariate input (2D array). "
                "Got 1D array. Use univariate NeuralForecast models instead."
            )

        self._train_y = y
        self._n_series = y.shape[1]
        self._freq = freq or 'D'

        # Store n_series in params for _create_model()
        self.params['n_series'] = self._n_series

        # Allow horizon override at fit time
        if horizon is not None:
            self.params['horizon'] = horizon
        elif 'horizon' not in self.params:
            self.params['horizon'] = 24
        self._horizon = self.params['horizon']

        # Convert to NeuralForecast long format: one row per (time, series)
        n_samples = y.shape[0]
        dates = safe_date_range(start='2000-01-01', periods=n_samples, freq=self._freq)

        dfs = []
        for i in range(self._n_series):
            df_i = pd.DataFrame({
                'unique_id': f'series_{i}',
                'ds': dates,
                'y': y[:, i]
            })
            dfs.append(df_i)
        df = pd.concat(dfs, ignore_index=True)

        self._nf = NeuralForecast(
            models=[self._create_model()],
            freq=self._freq,
        )
        self._nf.fit(df=df)
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

        if horizon > self._horizon:
            import warnings
            warnings.warn(
                f"Requested horizon ({horizon}) exceeds model's trained horizon ({self._horizon}). "
                f"Returning {self._horizon} steps."
            )

        forecast = self._nf.predict()
        # Get model column name
        model_col = [c for c in forecast.columns if c not in ['unique_id', 'ds']][0]

        # Reshape from long format to [horizon, n_series]
        result = np.zeros((min(horizon, self._horizon), self._n_series))
        for i in range(self._n_series):
            series_forecast = forecast[forecast['unique_id'] == f'series_{i}'][model_col].values
            result[:, i] = series_forecast[:horizon]

        return result

    def get_diagnostics(self):
        if not self._is_fitted or not hasattr(self, "_nf"):
            return {}
        d = {}
        try:
            model = self._nf.models[0]
            if hasattr(model, "trainer") and model.trainer is not None:
                d["epochs_trained"] = int(model.trainer.current_epoch)
                metrics = model.trainer.callback_metrics
                if "train_loss" in metrics:
                    d["final_train_loss"] = float(metrics["train_loss"])
            if hasattr(model, "learning_rate"):
                d["learning_rate"] = float(model.learning_rate)
        except Exception:
            pass
        return d

    def _create_model(self):
        raise NotImplementedError


@register_model
class NeuralForecastNBEATS(_BaseNeuralForecast):
    """
    NeuralForecast NBEATS implementation.

    Args:
        horizon: Forecast horizon
        input_size: Input sequence length (multiplied by horizon)
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import NBEATS

        h = self.params.get('horizon', 12)
        return NBEATS(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastNBEATS",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2022,
            paper="Garza et al. (2022)",
        )


@register_model
class NeuralForecastNHITS(_BaseNeuralForecast):
    """
    NeuralForecast NHITS implementation.

    Efficient hierarchical architecture.
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import NHITS

        h = self.params.get('horizon', 12)
        return NHITS(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastNHITS",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2022,
            paper="Challu et al. (2022)",
        )


@register_model
class NeuralForecastTFT(_BaseNeuralForecast):
    """
    NeuralForecast Temporal Fusion Transformer.

    Interpretable attention-based architecture.
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import TFT

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return TFT(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastTFT",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2022,
            paper="Lim et al. (2021)",
            probabilistic=True,
            exogenous=True,
        )


@register_model
class NeuralForecastAutoformer(_BaseNeuralForecast):
    """
    Autoformer: Auto-Correlation mechanism for long-term forecasting.

    Decomposes series using auto-correlation to capture temporal dependencies.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier (input_size = horizon * mult)
        hidden_size: Hidden layer dimension
        n_heads: Number of attention heads
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 64,
        n_heads: int = 4,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            n_heads=n_heads,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import Autoformer

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'hidden_size': self.params.get('hidden_size', 64),
            'n_head': self.params.get('n_heads', 4),  # API uses n_head not n_heads
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return Autoformer(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastAutoformer",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2021,
            paper="Wu et al. (2021) - Autoformer",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("n_heads", "int", low=2, high=8, default=4),
        ]


@register_model
class NeuralForecastInformer(_BaseNeuralForecast):
    """
    Informer: Efficient long-horizon forecasting with ProbSparse attention.

    Uses ProbSparse self-attention for O(L log L) complexity.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden layer dimension
        n_heads: Number of attention heads
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 64,
        n_heads: int = 4,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            n_heads=n_heads,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import Informer

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'hidden_size': self.params.get('hidden_size', 64),
            'n_head': self.params.get('n_heads', 4),  # API uses n_head not n_heads
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
            'scaler_type': 'robust',
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return Informer(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastInformer",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2021,
            paper="Zhou et al. (2021) - Informer: Beyond Efficient Transformer",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("n_heads", "int", low=2, high=8, default=4),
        ]


@register_model
class NeuralForecastFEDformer(_BaseNeuralForecast):
    """
    FEDformer: Frequency Enhanced Decomposed Transformer.

    Operates in frequency domain for global view of time series.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden layer dimension
        n_heads: Number of attention heads
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 64,
        n_heads: int = 4,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            n_heads=n_heads,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import FEDformer

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'hidden_size': self.params.get('hidden_size', 64),
            'n_head': 8,  # FEDformer requires n_head=8
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return FEDformer(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastFEDformer",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2022,
            paper="Zhou et al. (2022) - FEDformer",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
        ]


@register_model
class NeuralForecastPatchTST(_BaseNeuralForecast):
    """
    PatchTST: Patching and channel-independence for long-term forecasting.

    Segments time series into patches like Vision Transformer.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        patch_len: Length of each patch
        stride: Stride between patches
        hidden_size: Hidden layer dimension
        n_heads: Number of attention heads
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        hidden_size: int = 64,
        n_heads: int = 4,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            patch_len=patch_len,
            stride=stride,
            hidden_size=hidden_size,
            n_heads=n_heads,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import PatchTST

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 4),
            'patch_len': self.params.get('patch_len', 16),
            'stride': self.params.get('stride', 8),
            'hidden_size': self.params.get('hidden_size', 64),
            'n_heads': self.params.get('n_heads', 4),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return PatchTST(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastPatchTST",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Nie et al. (2023) - A Time Series is Worth 64 Words",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=2, high=8, default=4),
            ParamSpace("patch_len", "int", low=8, high=32, default=16),
            ParamSpace("stride", "int", low=4, high=16, default=8),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
        ]


@register_model
class NeuralForecastLSTM(_BaseNeuralForecast):
    """
    NeuralForecast LSTM implementation.

    Standard LSTM for time series forecasting.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        encoder_hidden_size: LSTM hidden dimension
        encoder_n_layers: Number of LSTM layers
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        encoder_hidden_size: int = 64,
        encoder_n_layers: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            encoder_hidden_size=encoder_hidden_size,
            encoder_n_layers=encoder_n_layers,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import LSTM

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'encoder_hidden_size': self.params.get('encoder_hidden_size', 64),
            'encoder_n_layers': self.params.get('encoder_n_layers', 2),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return LSTM(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastLSTM",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=1997,
            paper="Hochreiter & Schmidhuber (1997)",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("encoder_hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("encoder_n_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastiTransformer(_BaseNeuralForecastMultivariate):
    """
    iTransformer: Inverted Transformer for MULTIVARIATE time series forecasting.

    Applies attention on the variate dimension instead of time dimension,
    treating each time point as a token. REQUIRES multivariate input.

    Input: 2D array [n_samples, n_series]
    Output: 2D array [horizon, n_series]

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden layer dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            n_heads=n_heads,
            e_layers=e_layers,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import iTransformer

        h = self.params.get('horizon', 12)
        return iTransformer(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            n_series=self.params['n_series'],  # Set at fit() time
            hidden_size=self.params.get('hidden_size', 128),
            n_heads=self.params.get('n_heads', 4),
            e_layers=self.params.get('e_layers', 2),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastiTransformer",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2024,
            paper="Liu et al. (2024) - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
            notes="MULTIVARIATE-ONLY. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=64, high=512, default=128),
            ParamSpace("n_heads", "int", low=2, high=8, default=4),
            ParamSpace("e_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastDLinear(_BaseNeuralForecast):
    """
    DLinear: Simple decomposition linear model.

    Decomposes series into trend and seasonal, applies linear layers.
    Surprisingly competitive with complex transformers.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import DLinear

        h = self.params.get('horizon', 12)
        return DLinear(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastDLinear",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Zeng et al. (2023) - Are Transformers Effective for Time Series Forecasting?",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastNLinear(_BaseNeuralForecast):
    """
    NLinear: Normalization-based linear model.

    Subtracts last value before applying linear layer.
    Very simple but effective baseline.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import NLinear

        h = self.params.get('horizon', 12)
        return NLinear(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastNLinear",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Zeng et al. (2023) - Are Transformers Effective for Time Series Forecasting?",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastGRU(_BaseNeuralForecast):
    """
    NeuralForecast GRU implementation.

    Gated Recurrent Unit - simpler alternative to LSTM.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        encoder_hidden_size: GRU hidden dimension
        encoder_n_layers: Number of GRU layers
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        encoder_hidden_size: int = 64,
        encoder_n_layers: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            encoder_hidden_size=encoder_hidden_size,
            encoder_n_layers=encoder_n_layers,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import GRU

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'encoder_hidden_size': self.params.get('encoder_hidden_size', 64),
            'encoder_n_layers': self.params.get('encoder_n_layers', 2),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return GRU(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastGRU",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2014,
            paper="Cho et al. (2014) - Learning Phrase Representations using RNN Encoder-Decoder",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("encoder_hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("encoder_n_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastDeepAR(_BaseNeuralForecast):
    """
    DeepAR: Probabilistic forecasting with autoregressive RNNs.

    Outputs distribution parameters for probabilistic forecasts.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: LSTM hidden dimension
        n_layers: Number of LSTM layers
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 64,
        n_layers: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            n_layers=n_layers,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import DeepAR

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'lstm_hidden_size': self.params.get('hidden_size', 64),
            'lstm_n_layers': self.params.get('n_layers', 2),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
            'scaler_type': 'robust',
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return DeepAR(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastDeepAR",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2020,
            paper="Salinas et al. (2020) - DeepAR: Probabilistic Forecasting with Autoregressive RNNs",
            probabilistic=True,
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("n_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastRNN(_BaseNeuralForecast):
    """
    NeuralForecast vanilla RNN implementation.

    Simple recurrent neural network baseline.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        encoder_hidden_size: RNN hidden dimension
        encoder_n_layers: Number of RNN layers
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        encoder_hidden_size: int = 64,
        encoder_n_layers: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            encoder_hidden_size=encoder_hidden_size,
            encoder_n_layers=encoder_n_layers,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import RNN

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'encoder_hidden_size': self.params.get('encoder_hidden_size', 64),
            'encoder_n_layers': self.params.get('encoder_n_layers', 2),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return RNN(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastRNN",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=1986,
            paper="Rumelhart et al. (1986)",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("encoder_hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("encoder_n_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastTCN(_BaseNeuralForecast):
    """
    Temporal Convolutional Network.

    Dilated causal convolutions for sequence modeling.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        kernel_size: Convolution kernel size
        num_filters: Number of convolution filters
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        kernel_size: int = 3,
        num_filters: int = 64,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            kernel_size=kernel_size,
            num_filters=num_filters,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import TCN

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'kernel_size': self.params.get('kernel_size', 3),
            'encoder_hidden_size': self.params.get('num_filters', 64),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return TCN(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastTCN",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2018,
            paper="Bai et al. (2018) - An Empirical Evaluation of Generic Convolutional and Recurrent Networks",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("kernel_size", "int", low=2, high=7, default=3),
            ParamSpace("num_filters", "int", low=32, high=256, default=64),
        ]


@register_model
class NeuralForecastBiTCN(_BaseNeuralForecast):
    """
    Bidirectional Temporal Convolutional Network.

    Extends TCN with backward temporal convolutions.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        kernel_size: Convolution kernel size
        num_filters: Number of convolution filters
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        kernel_size: int = 3,
        num_filters: int = 64,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            kernel_size=kernel_size,
            num_filters=num_filters,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import BiTCN

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'hidden_size': self.params.get('num_filters', 64),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return BiTCN(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastBiTCN",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Olivares et al. (2023) - NeuralForecast",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("kernel_size", "int", low=2, high=7, default=3),
            ParamSpace("num_filters", "int", low=32, high=256, default=64),
        ]


@register_model
class NeuralForecastMLP(_BaseNeuralForecast):
    """
    NeuralForecast MLP (Multi-Layer Perceptron).

    Simple feedforward network for time series.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        num_layers: Number of hidden layers
        hidden_size: Hidden layer dimension
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        num_layers: int = 2,
        hidden_size: int = 512,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            num_layers=num_layers,
            hidden_size=hidden_size,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import MLP

        h = self.params.get('horizon', 12)
        return MLP(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            num_layers=self.params.get('num_layers', 2),
            hidden_size=self.params.get('hidden_size', 512),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastMLP",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=1986,
            paper="Rumelhart et al. (1986)",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("num_layers", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=128, high=1024, default=512),
        ]


@register_model
class NeuralForecastTiDE(_BaseNeuralForecast):
    """
    TiDE: Time-series Dense Encoder.

    Efficient MLP-based architecture that rivals transformers.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden dimension
        num_encoder_layers: Encoder layers
        num_decoder_layers: Decoder layers
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 256,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import TiDE

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'hidden_size': self.params.get('hidden_size', 256),
            'num_encoder_layers': self.params.get('num_encoder_layers', 2),
            'num_decoder_layers': self.params.get('num_decoder_layers', 2),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return TiDE(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastTiDE",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Das et al. (2023) - Long-term Forecasting with TiDE",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=64, high=512, default=256),
            ParamSpace("num_encoder_layers", "int", low=1, high=4, default=2),
            ParamSpace("num_decoder_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastTimeMixer(_BaseNeuralForecastMultivariate):
    """
    TimeMixer: Fully MLP-based architecture for MULTIVARIATE time series forecasting.

    Uses multi-scale decomposition and mixing for efficient forecasting.
    REQUIRES multivariate input.

    Input: 2D array [n_samples, n_series]
    Output: 2D array [horizon, n_series]

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        d_model: Model dimension (default 32)
        d_ff: Feedforward dimension (default 32)
        e_layers: Number of encoder layers (default 4)
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        d_model: int = 32,
        d_ff: int = 32,
        e_layers: int = 4,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import TimeMixer

        h = self.params.get('horizon', 12)
        return TimeMixer(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            n_series=self.params['n_series'],
            d_model=self.params.get('d_model', 32),
            d_ff=self.params.get('d_ff', 32),
            e_layers=self.params.get('e_layers', 4),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastTimeMixer",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2024,
            paper="Wang et al. (2024) - TimeMixer: Decomposable Multiscale Mixing",
            notes="MULTIVARIATE-ONLY. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("d_model", "int", low=16, high=128, default=32),
            ParamSpace("d_ff", "int", low=16, high=128, default=32),
            ParamSpace("e_layers", "int", low=2, high=6, default=4),
        ]


@register_model
class NeuralForecastTSMixer(_BaseNeuralForecastMultivariate):
    """
    TSMixer: Time Series Mixer with token/channel mixing for MULTIVARIATE time series.

    MLP-based architecture using time and feature mixing.
    REQUIRES multivariate input.

    Input: 2D array [n_samples, n_series]
    Output: 2D array [horizon, n_series]

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        n_block: Number of mixer blocks
        ff_dim: Feedforward dimension
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        n_block: int = 2,
        ff_dim: int = 64,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            n_block=n_block,
            ff_dim=ff_dim,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import TSMixer

        h = self.params.get('horizon', 12)
        return TSMixer(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            n_series=self.params['n_series'],
            n_block=self.params.get('n_block', 2),
            ff_dim=self.params.get('ff_dim', 64),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastTSMixer",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Chen et al. (2023) - TSMixer: An All-MLP Architecture",
            notes="MULTIVARIATE-ONLY. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("n_block", "int", low=1, high=6, default=2),
            ParamSpace("ff_dim", "int", low=32, high=256, default=64),
        ]


@register_model
class NeuralForecastTSMixerx(_BaseNeuralForecastMultivariate):
    """
    TSMixerx: Extended TSMixer with exogenous features for MULTIVARIATE time series.

    Handles multivariate inputs with external covariates.
    REQUIRES multivariate input.

    Input: 2D array [n_samples, n_series]
    Output: 2D array [horizon, n_series]

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        n_block: Number of mixer blocks
        ff_dim: Feedforward dimension
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        n_block: int = 2,
        ff_dim: int = 64,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            n_block=n_block,
            ff_dim=ff_dim,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import TSMixerx

        h = self.params.get('horizon', 12)
        return TSMixerx(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            n_series=self.params['n_series'],
            n_block=self.params.get('n_block', 2),
            ff_dim=self.params.get('ff_dim', 64),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastTSMixerx",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Chen et al. (2023) - TSMixer: An All-MLP Architecture",
            notes="MULTIVARIATE-ONLY. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("n_block", "int", low=1, high=6, default=2),
            ParamSpace("ff_dim", "int", low=32, high=256, default=64),
        ]


@register_model
class NeuralForecastVanillaTransformer(_BaseNeuralForecast):
    """
    Vanilla Transformer for time series forecasting.

    Standard transformer architecture adapted for time series.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            n_heads=n_heads,
            e_layers=e_layers,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import VanillaTransformer

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'hidden_size': self.params.get('hidden_size', 64),
            'n_head': self.params.get('n_heads', 4),  # API uses n_head not n_heads
            'encoder_layers': self.params.get('e_layers', 2),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
            'scaler_type': 'robust',
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return VanillaTransformer(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastVanillaTransformer",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2017,
            paper="Vaswani et al. (2017) - Attention Is All You Need",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("n_heads", "int", low=2, high=8, default=4),
            ParamSpace("e_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastKAN(_BaseNeuralForecast):
    """
    KAN: Kolmogorov-Arnold Network for time series.

    Uses learnable activation functions on edges instead of nodes.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden dimension
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 256,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import KAN

        h = self.params.get('horizon', 12)
        return KAN(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            hidden_size=self.params.get('hidden_size', 256),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastKAN",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2024,
            paper="Liu et al. (2024) - KAN: Kolmogorov-Arnold Networks",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=64, high=512, default=256),
        ]


@register_model
class NeuralForecastSOFTS(_BaseNeuralForecastMultivariate):
    """
    SOFTS: Efficient MLP-based model with star topology for MULTIVARIATE time series.

    Uses star-aggregate-dispatch architecture for efficiency.
    REQUIRES multivariate input.

    Input: 2D array [n_samples, n_series]
    Output: 2D array [horizon, n_series]

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden dimension
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 256,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import SOFTS

        h = self.params.get('horizon', 12)
        return SOFTS(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            n_series=self.params['n_series'],
            hidden_size=self.params.get('hidden_size', 256),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastSOFTS",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2024,
            paper="Lu et al. (2024) - SOFTS: Efficient Multivariate Time Series Forecasting",
            notes="MULTIVARIATE-ONLY. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=64, high=512, default=256),
        ]


@register_model
class NeuralForecastStemGNN(_BaseNeuralForecastMultivariate):
    """
    StemGNN: Spectral Temporal Graph Neural Network for MULTIVARIATE time series.

    Combines spectral and temporal analysis in GNN framework.
    REQUIRES multivariate input.

    Input: 2D array [n_samples, n_series]
    Output: 2D array [horizon, n_series]

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        n_stacks: Number of stacks
        multi_layer: Number of layers in GNN
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        n_stacks: int = 2,
        multi_layer: int = 5,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            n_stacks=n_stacks,
            multi_layer=multi_layer,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import StemGNN

        h = self.params.get('horizon', 12)
        return StemGNN(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            n_series=self.params['n_series'],
            n_stacks=self.params.get('n_stacks', 2),
            multi_layer=self.params.get('multi_layer', 5),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastStemGNN",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2020,
            paper="Cao et al. (2020) - Spectral Temporal Graph Neural Network",
            notes="MULTIVARIATE-ONLY. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("n_stacks", "int", low=1, high=4, default=2),
            ParamSpace("multi_layer", "int", low=3, high=8, default=5),
        ]


@register_model
class NeuralForecastMLPMultivariate(_BaseNeuralForecastMultivariate):
    """
    MLPMultivariate: MLP for MULTIVARIATE time series.

    Processes multiple series jointly for cross-series patterns.
    REQUIRES multivariate input.

    Input: 2D array [n_samples, n_series]
    Output: 2D array [horizon, n_series]

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        num_layers: Number of hidden layers
        hidden_size: Hidden layer dimension
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        num_layers: int = 2,
        hidden_size: int = 512,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            num_layers=num_layers,
            hidden_size=hidden_size,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import MLPMultivariate

        h = self.params.get('horizon', 12)
        return MLPMultivariate(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            n_series=self.params['n_series'],
            num_layers=self.params.get('num_layers', 2),
            hidden_size=self.params.get('hidden_size', 512),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastMLPMultivariate",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Olivares et al. (2023) - NeuralForecast",
            notes="MULTIVARIATE-ONLY. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("num_layers", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=128, high=1024, default=512),
        ]


# NOTE: WaveNet was removed from neuralforecast library in recent versions
# Entire class commented out - WaveNet import no longer exists
#
# class NeuralForecastWaveNet(_BaseNeuralForecast):
#     """
#     WaveNet: Dilated causal convolutions from DeepMind.
#
#     Originally designed for audio, effective for time series.
#
#     Args:
#         horizon: Forecast horizon
#         input_size_mult: Input size multiplier
#         kernel_size: Convolution kernel size
#         max_steps: Training steps
#     """
#
#     def __init__(
#         self,
#         horizon: int = 12,
#         input_size_mult: int = 2,
#         kernel_size: int = 2,
#         max_steps: int = 1000,
#         **kwargs
#     ):
#         super().__init__(
#             horizon=horizon,
#             input_size_mult=input_size_mult,
#             kernel_size=kernel_size,
#             max_steps=max_steps,
#             **kwargs
#         )
#
#     def _create_model(self):
#         from neuralforecast.models import WaveNet
#
#         h = self.params.get('horizon', 12)
#         return WaveNet(
#             h=h,
#             input_size=h * self.params.get('input_size_mult', 2),
#             kernel_size=self.params.get('kernel_size', 2),
#             max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
#         )
#
#     @staticmethod
#     def get_metadata() -> ModelMetadata:
#         return ModelMetadata(
#             name="NeuralForecastWaveNet",
#             category=ModelCategory.DEEP_LEARNING,
#             library="neuralforecast",
#             year=2016,
#             paper="van den Oord et al. (2016) - WaveNet: A Generative Model for Raw Audio",
#         )
#
#     @staticmethod
#     def get_param_space() -> List[ParamSpace]:
#         return [
#             ParamSpace("horizon", "int", low=6, high=48, default=12),
#             ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
#             ParamSpace("kernel_size", "int", low=2, high=5, default=2),
#         ]


# NeuralForecastTimeLLM removed — NeuralForecast 3.1.4 TimeLLM has an unfixable memory leak:
# 706MB model (GPT-2 backbone) allocates ALL available VRAM regardless of GPU size.
# Tested: A10G (21.7/22 GiB), A100-40GB (39.4/39.5 GiB), A100-80GB (78+/80 GiB) — always OOMs.
# Use PyPOTSFcstTimeLLM instead (works correctly, same TimeLLM architecture).


@register_model
class NeuralForecastTimesNet(_BaseNeuralForecast):
    """
    TimesNet: Temporal 2D-variation modeling via multi-periodicity.

    Transforms 1D time series to 2D tensors based on multiple periods,
    then applies 2D convolutions for pattern extraction.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden dimension
        num_kernels: Number of inception kernels
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 64,
        num_kernels: int = 6,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            num_kernels=num_kernels,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import TimesNet

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'hidden_size': self.params.get('hidden_size', 64),
            'num_kernels': self.params.get('num_kernels', 6),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return TimesNet(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastTimesNet",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Wu et al. (2023) - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("num_kernels", "int", low=3, high=10, default=6),
        ]


@register_model
class NeuralForecastTimeXer(_BaseNeuralForecastMultivariate):
    """
    TimeXer: Cross-variable attention transformer for MULTIVARIATE time series.

    Combines patching with cross-variate attention for multivariate forecasting.
    REQUIRES multivariate input.

    Input: 2D array [n_samples, n_series]
    Output: 2D array [horizon, n_series]

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden dimension
        n_heads: Number of attention heads
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 64,
        n_heads: int = 4,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            n_heads=n_heads,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import TimeXer

        h = self.params.get('horizon', 12)
        return TimeXer(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            n_series=self.params['n_series'],
            hidden_size=self.params.get('hidden_size', 64),
            n_heads=self.params.get('n_heads', 4),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastTimeXer",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2024,
            paper="Wang et al. (2024) - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables",
            notes="MULTIVARIATE-ONLY. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("n_heads", "int", low=2, high=8, default=4),
        ]


@register_model
class NeuralForecastNBEATSx(_BaseNeuralForecast):
    """
    NBEATSx: NBEATS with exogenous variables.

    Extension of NBEATS that incorporates external covariates.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        n_harmonics: Number of harmonics for trend/seasonality
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        n_harmonics: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            n_harmonics=n_harmonics,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import NBEATSx

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'n_harmonics': self.params.get('n_harmonics', 2),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return NBEATSx(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastNBEATSx",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2022,
            paper="Olivares et al. (2022) - NBEATSx: Neural basis expansion with exogenous variables",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("n_harmonics", "int", low=1, high=5, default=2),
        ]


@register_model
class NeuralForecastDilatedRNN(_BaseNeuralForecast):
    """
    Dilated RNN: RNN with dilated skip connections.

    Extends RNN with dilated recurrent connections for longer dependencies.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        encoder_hidden_size: Hidden dimension
        max_steps: Training steps
    """

    _supports_exog = True

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        encoder_hidden_size: int = 64,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            encoder_hidden_size=encoder_hidden_size,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import DilatedRNN

        h = self.params.get('horizon', 12)
        kwargs = {
            'h': h,
            'input_size': h * self.params.get('input_size_mult', 2),
            'encoder_hidden_size': self.params.get('encoder_hidden_size', 64),
            'max_steps': get_fast_steps(self.params.get('max_steps', 1000)),
        }
        if self._exog_cols:
            kwargs['futr_exog_list'] = self._exog_cols
        return DilatedRNN(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastDilatedRNN",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2017,
            paper="Chang et al. (2017) - Dilated Recurrent Neural Networks",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("encoder_hidden_size", "int", low=32, high=256, default=64),
        ]


@register_model
class NeuralForecastDeepNPTS(_BaseNeuralForecast):
    """
    DeepNPTS: Deep Neural Point Time Series.

    Deep learning version of NPTS for point forecasting.

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        hidden_size: Hidden dimension
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        hidden_size: int = 64,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            hidden_size=hidden_size,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import DeepNPTS

        h = self.params.get('horizon', 12)
        return DeepNPTS(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            hidden_size=self.params.get('hidden_size', 64),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastDeepNPTS",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2023,
            paper="Olivares et al. (2023) - NeuralForecast",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
        ]


@register_model
class NeuralForecastxLSTM(_BaseNeuralForecast):
    """
    xLSTM: Extended LSTM with exponential gating and memory mixing.

    Modern LSTM variant with improved long-range dependency handling.

    REQUIRES OPTIONAL DEPENDENCY: pip install xlstm mlstm_kernels ninja

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        encoder_hidden_size: Encoder hidden dimension
        encoder_n_blocks: Number of encoder blocks
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        encoder_hidden_size: int = 128,
        encoder_n_blocks: int = 2,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            encoder_hidden_size=encoder_hidden_size,
            encoder_n_blocks=encoder_n_blocks,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import xLSTM

        h = self.params.get('horizon', 12)
        return xLSTM(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            encoder_hidden_size=self.params.get('encoder_hidden_size', 128),
            encoder_n_blocks=self.params.get('encoder_n_blocks', 2),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastxLSTM",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2024,
            paper="Beck et al. (2024) - xLSTM: Extended Long Short-Term Memory",
            requires_gpu=True,  # xLSTM kernels require CUDA
            min_gpu_memory_gb=4.0,
            notes="Requires CUDA GPU. GPU-host dependencies: xlstm mlstm_kernels ninja",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("encoder_hidden_size", "int", low=64, high=256, default=128),
            ParamSpace("encoder_n_blocks", "int", low=1, high=4, default=2),
        ]


@register_model
class NeuralForecastRMoK(_BaseNeuralForecastMultivariate):
    """
    RMoK: Reversible Mixture of KAN for MULTIVARIATE time series.

    Combines KAN with reversible layers and mixture of experts.
    REQUIRES multivariate input.

    Input: 2D array [n_samples, n_series]
    Output: 2D array [horizon, n_series]

    Args:
        horizon: Forecast horizon
        input_size_mult: Input size multiplier
        taylor_order: Order for Taylor expansion in KAN
        jacobi_degree: Degree for Jacobi polynomial in KAN
        dropout: Dropout rate
        max_steps: Training steps
    """

    def __init__(
        self,
        horizon: int = 12,
        input_size_mult: int = 2,
        taylor_order: int = 3,
        jacobi_degree: int = 6,
        dropout: float = 0.1,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            horizon=horizon,
            input_size_mult=input_size_mult,
            taylor_order=taylor_order,
            jacobi_degree=jacobi_degree,
            dropout=dropout,
            max_steps=max_steps,
            **kwargs
        )

    def _create_model(self):
        from neuralforecast.models import RMoK

        h = self.params.get('horizon', 12)
        return RMoK(
            h=h,
            input_size=h * self.params.get('input_size_mult', 2),
            n_series=self.params['n_series'],
            taylor_order=self.params.get('taylor_order', 3),
            jacobi_degree=self.params.get('jacobi_degree', 6),
            dropout=self.params.get('dropout', 0.1),
            max_steps=get_fast_steps(self.params.get('max_steps', 1000)),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NeuralForecastRMoK",
            category=ModelCategory.DEEP_LEARNING,
            library="neuralforecast",
            year=2024,
            paper="Liu et al. (2024) - RMoK: Reversible Mixture of KAN",
            notes="MULTIVARIATE-ONLY. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("horizon", "int", low=6, high=48, default=12),
            ParamSpace("input_size_mult", "int", low=1, high=4, default=2),
            ParamSpace("taylor_order", "int", low=2, high=5, default=3),
            ParamSpace("jacobi_degree", "int", low=4, high=10, default=6),
        ]
