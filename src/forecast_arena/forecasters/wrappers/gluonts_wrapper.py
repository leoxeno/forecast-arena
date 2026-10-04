"""
forecasters/wrappers/gluonts_wrapper.py - GluonTS deep learning models.

Wraps Amazon's GluonTS models:
- DeepAR
- SimpleFeedForward
- WaveNet
- PatchTST
- TFT

Note: Some models were removed in GluonTS 0.14+ (Transformer, DeepState, etc.)
These are kept for reference but won't register if dependencies are missing.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace, get_fast_epochs
from .. import register_model


def _register_if_available(module_path: str):
    """
    Decorator factory that only registers model if its module exists.
    Use for models that may not exist in all GluonTS versions.
    """
    def decorator(cls):
        try:
            # Use importlib to properly verify module exists
            import importlib
            importlib.import_module(module_path)
            return register_model(cls)
        except (ImportError, ModuleNotFoundError):
            # Module doesn't exist in this GluonTS version, don't register
            return cls
    return decorator


def _to_gluonts_freq(freq: str) -> str:
    """Translate new pandas freq aliases to old ones for GluonTS compatibility.

    Pandas 2.2+ deprecated 'Y', 'M', 'Q' etc. in favor of 'YE', 'ME', 'QE'.
    GluonTS still uses the old convention internally.
    """
    _FREQ_MAP = {
        'YE': 'Y', 'YS': 'YS', 'ME': 'M', 'MS': 'MS',
        'QE': 'Q', 'QS': 'QS', 'BYE': 'BY', 'BME': 'BM',
        'BQE': 'BQ', 'h': 'H', 'min': 'T', 's': 'S',
    }
    for new, old in _FREQ_MAP.items():
        if freq == new or freq.startswith(new + '-'):
            return freq.replace(new, old, 1)
    return freq


class _BaseGluonTSForecaster(Forecaster):
    """Base class for GluonTS models.

    Accepts either `prediction_length` or `horizon` as the forecast horizon parameter.
    """

    # Override in subclass to enable exogenous support.
    # When True, X columns are added as feat_dynamic_real to the dataset.
    _supports_exog: bool = False

    # Set True in subclasses that need input normalization (e.g. MQF2).
    # Models like DeepAR handle scale internally.
    _needs_normalization: bool = False

    def __init__(self, **kwargs):
        # Accept 'horizon' as alias for 'prediction_length'
        if 'horizon' in kwargs and 'prediction_length' not in kwargs:
            kwargs['prediction_length'] = kwargs.pop('horizon')
        super().__init__(**kwargs)

    def _build_dataset(self, y, exog=None):
        """Build a GluonTS ListDataset with explicit freq (avoids pandas freq alias issues)."""
        from gluonts.dataset.common import ListDataset

        entry = {
            "start": pd.Period("2000-01", freq=self._freq),
            "target": y,
        }
        if exog is not None:
            entry["feat_dynamic_real"] = exog.T  # shape: (n_features, n_timesteps)

        return ListDataset([entry], freq=self._freq)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BaseGluonTSForecaster':
        y = self._validate_y(y)
        # Convert to float32 for PyTorch compatibility
        y = y.astype(np.float32)
        self._train_y = y
        self._freq = _to_gluonts_freq(freq or 'D')

        # Z-score normalize for models that need it (e.g. MQF2)
        self._y_mean = np.float32(0.0)
        self._y_std = np.float32(1.0)
        if self._needs_normalization:
            self._y_mean = np.float32(np.mean(y))
            self._y_std = np.float32(np.std(y)) or np.float32(1.0)
            y = (y - self._y_mean) / self._y_std

        # Use horizon from fit kwargs (passed by runner), override constructor default
        if 'horizon' in kwargs:
            self.params['prediction_length'] = kwargs['horizon']

        # Handle exogenous features
        self._train_exog = None
        self._has_exog = False
        if X is not None and self._supports_exog:
            exog = self._validate_X(X, len(y))
            if exog is not None:
                self._train_exog = exog.astype(np.float32)
                self._has_exog = True

        self._dataset = self._build_dataset(y, self._train_exog)

        self._model = self._create_model()
        self._predictor = self._model.train(self._dataset)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        # If model was trained with exogenous, extend dataset with future X
        dataset = self._dataset
        if self._has_exog and self._train_exog is not None:
            pred_len = self.params.get('prediction_length', 12)

            # Extend target with NaN for future period
            extended_target = np.concatenate([
                self._train_y, np.full(pred_len, np.nan)
            ]).astype(np.float32)

            exog_future = self._validate_X(X, pred_len) if X is not None else None
            if exog_future is not None:
                full_exog = np.vstack([self._train_exog, exog_future.astype(np.float32)])
            else:
                n_features = self._train_exog.shape[1]
                full_exog = np.vstack([
                    self._train_exog,
                    np.zeros((pred_len, n_features), dtype=np.float32),
                ])
            dataset = self._build_dataset(extended_target, full_exog)

        forecasts = list(self._predictor.predict(dataset))
        result = forecasts[0].mean
        if self._needs_normalization:
            result = result * self._y_std + self._y_mean
        return result

    def _compute_fitted_values(self):
        if self._predictor is None or self._train_y is None:
            return None
        pred_len = self.params.get('prediction_length', 12)
        n = len(self._train_y)
        if n < pred_len + 10:
            return None
        # Create truncated datasets, predict 1-step-ahead via rolling origin
        fitted = np.full(n, np.nan)
        # Stride by pred_len to keep it fast (not every single point)
        for end in range(pred_len + 10, n + 1, pred_len):
            trunc_y = self._train_y[:end].astype(np.float32)
            if self._needs_normalization:
                trunc_y = (trunc_y - self._y_mean) / self._y_std
            ds = self._build_dataset(trunc_y, None)
            forecasts = list(self._predictor.predict(ds))
            fcast = forecasts[0].mean
            if self._needs_normalization:
                fcast = fcast * self._y_std + self._y_mean
            h = min(len(fcast), n - end)
            if h > 0:
                fitted[end:end + h] = fcast[:h]
        return fitted

    def get_diagnostics(self):
        if not self._is_fitted:
            return {}
        d = {"prediction_length": self.params.get("prediction_length", 12)}
        return d

    def _create_model(self):
        raise NotImplementedError


@register_model
class DeepAR(_BaseGluonTSForecaster):
    """
    DeepAR: Probabilistic Forecasting with Autoregressive RNN.

    Amazon's production forecasting model.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        num_layers: RNN layers
        hidden_size: RNN hidden units
        epochs: Training epochs
    """

    _supports_exog = True

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            num_layers=num_layers,
            hidden_size=hidden_size,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.deepar import DeepAREstimator

        return DeepAREstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            num_layers=self.params.get('num_layers', 2),
            hidden_size=self.params.get('hidden_size', 40),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DeepAR",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2019,
            paper="Salinas et al. (2019)",
            probabilistic=True,
            github_url="https://github.com/awslabs/gluonts",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("num_layers", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=20, high=100, default=40),
        ]


@register_model
class GluonTSSimpleFeedForward(_BaseGluonTSForecaster):
    """
    SimpleFeedForward: Basic MLP-based forecaster.

    Simple but fast baseline for comparison.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        hidden_dimensions: List of hidden layer sizes
        epochs: Training epochs
    """

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        hidden_dimensions: Optional[List[int]] = None,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            hidden_dimensions=hidden_dimensions or [40, 40],
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator

        # Note: freq param removed in GluonTS 0.14+
        return SimpleFeedForwardEstimator(
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            hidden_dimensions=self.params.get('hidden_dimensions', [40, 40]),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSSimpleFeedForward",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2019,
            paper="Alexandrov et al. (2019)",
            probabilistic=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("prediction_length", "int", low=6, high=48, default=12),
        ]


@_register_if_available('gluonts.torch.model.transformer')
class GluonTSTransformer(_BaseGluonTSForecaster):
    """
    GluonTS Transformer-based forecaster.

    Attention-based architecture for time series.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        d_model: Model dimension
        nhead: Number of attention heads
        epochs: Training epochs
    """

    _supports_exog = True

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        d_model: int = 32,
        nhead: int = 4,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            d_model=d_model,
            nhead=nhead,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.transformer import TransformerEstimator

        return TransformerEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            d_model=self.params.get('d_model', 32),
            nhead=self.params.get('nhead', 4),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSTransformer",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2019,
            paper="Alexandrov et al. (2019)",
            probabilistic=True,
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("d_model", "int", low=16, high=128, default=32),
            ParamSpace("nhead", "int", low=2, high=8, default=4),
        ]


@_register_if_available('gluonts.torch.model.deepstate')
class GluonTSDeepState(_BaseGluonTSForecaster):
    """
    DeepState: Deep State-Space Models.

    Combines state-space models with deep learning for interpretability.
    Models trend, seasonality, and residuals with learned components.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        cardinality: Cardinalities for categorical features
        epochs: Training epochs
    """

    _supports_exog = True

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        use_feat_static_cat: bool = False,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            use_feat_static_cat=use_feat_static_cat,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.deepstate import DeepStateEstimator

        return DeepStateEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSDeepState",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2018,
            paper="Rangapuram et al. (2018) - Deep State Space Models",
            probabilistic=True,
            notes="Interpretable state-space decomposition",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("prediction_length", "int", low=6, high=48, default=12),
        ]


@register_model
class GluonTSWaveNet(_BaseGluonTSForecaster):
    """
    WaveNet: Dilated causal convolutions for time series.

    Adapted from DeepMind's audio synthesis architecture.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        n_residue: Number of residual channels
        n_skip: Number of skip channels
        dilation_depth: Depth of dilation stack
        epochs: Training epochs
    """

    _supports_exog = True
    _needs_normalization: bool = True  # WaveNet uses quantized softmax — unstable on raw large values

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        n_residue: int = 24,
        n_skip: int = 32,
        dilation_depth: int = 3,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            n_residue=n_residue,
            n_skip=n_skip,
            dilation_depth=dilation_depth,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.wavenet import WaveNetEstimator

        # Note: param names changed in GluonTS 0.14+ (n_residue -> num_residual_channels)
        return WaveNetEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            num_residual_channels=self.params.get('n_residue', 24),
            num_skip_channels=self.params.get('n_skip', 32),
            dilation_depth=self.params.get('dilation_depth', 3),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSWaveNet",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2019,
            paper="van den Oord et al. (2016) / GluonTS adaptation",
            probabilistic=True,
            notes="Dilated causal convolutions from WaveNet architecture",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("n_residue", "int", low=16, high=64, default=24),
            ParamSpace("n_skip", "int", low=16, high=64, default=32),
            ParamSpace("dilation_depth", "int", low=2, high=5, default=3),
        ]


# ============================================================================
# Phase 3: Additional GluonTS Models
# ============================================================================


@_register_if_available('gluonts.torch.model.deepvar')
class GluonTSDeepVAR(_BaseGluonTSForecaster):
    """
    DeepVAR: Deep Vector Autoregression.

    Multivariate probabilistic forecasting using RNNs.
    Models cross-series dependencies for panel data.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        num_layers: RNN layers
        hidden_size: RNN hidden units
        epochs: Training epochs
    """

    _supports_exog = True

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            num_layers=num_layers,
            hidden_size=hidden_size,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.deepvar import DeepVAREstimator

        return DeepVAREstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            num_layers=self.params.get('num_layers', 2),
            hidden_size=self.params.get('hidden_size', 40),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSDeepVAR",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2019,
            paper="Salinas et al. (2019) - DeepAR extension",
            probabilistic=True,
            notes="Multivariate deep autoregression",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("num_layers", "int", low=1, high=4, default=2),
            ParamSpace("hidden_size", "int", low=20, high=100, default=40),
        ]


@_register_if_available('gluonts.torch.model.mqcnn')
class GluonTSMQCNN(_BaseGluonTSForecaster):
    """
    MQ-CNN: Multi-horizon Quantile Convolutional Neural Network.

    CNN-based quantile forecasting for multiple horizons.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        channels_seq: Sequence of channel sizes
        epochs: Training epochs
    """

    _supports_exog = True

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        channels_seq: Optional[List[int]] = None,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            channels_seq=channels_seq or [30, 30, 30],
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.mqcnn import MQCNNEstimator

        return MQCNNEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSMQCNN",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2019,
            paper="Wen et al. (2017) - MQ-CNN",
            probabilistic=True,
            notes="Multi-horizon quantile CNN",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("prediction_length", "int", low=6, high=48, default=12),
        ]


@register_model
class GluonTSTFT(_BaseGluonTSForecaster):
    """
    GluonTS Temporal Fusion Transformer.

    Attention-based architecture with variable selection.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        hidden_size: Hidden layer size
        epochs: Training epochs
    """

    _supports_exog = True

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        hidden_size: int = 32,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            hidden_size=hidden_size,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.tft import TemporalFusionTransformerEstimator

        return TemporalFusionTransformerEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            hidden_dim=self.params.get('hidden_size', 32),  # API: hidden_dim not hidden_size
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSTFT",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2021,
            paper="Lim et al. (2021) - Temporal Fusion Transformers",
            probabilistic=True,
            notes="Interpretable attention-based forecasting",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("hidden_size", "int", low=16, high=128, default=32),
        ]


@_register_if_available('gluonts.torch.model.deep_factor')
class GluonTSDeepFactor(_BaseGluonTSForecaster):
    """
    DeepFactor: Deep Factor Models.

    Combines factor model structure with deep learning.
    Good for hierarchical/grouped time series.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        num_hidden_global: Global RNN hidden size
        epochs: Training epochs
    """

    _supports_exog = True

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        num_hidden_global: int = 50,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            num_hidden_global=num_hidden_global,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.deep_factor import DeepFactorEstimator

        return DeepFactorEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            num_hidden_global=self.params.get('num_hidden_global', 50),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSDeepFactor",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2019,
            paper="Wang et al. (2019) - Deep Factor Models",
            probabilistic=True,
            notes="Factor model with deep learning",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("num_hidden_global", "int", low=20, high=100, default=50),
        ]


@_register_if_available('gluonts.torch.model.lstnet')
class GluonTSLSTNet(_BaseGluonTSForecaster):
    """
    LSTNet: Long and Short-term Time-series Network.

    Combines CNN for short-term patterns with RNN for long-term.
    Uses skip connections across time.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        channels: Number of CNN channels
        rnn_hidden_size: RNN hidden size
        skip: Skip connection period
        epochs: Training epochs
    """

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        channels: int = 32,
        rnn_hidden_size: int = 50,
        skip: int = 24,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            channels=channels,
            rnn_hidden_size=rnn_hidden_size,
            skip=skip,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.lstnet import LSTNetEstimator

        return LSTNetEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            channels=self.params.get('channels', 32),
            rnn_hidden_size=self.params.get('rnn_hidden_size', 50),
            skip_size=self.params.get('skip', 24),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSLSTNet",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2018,
            paper="Lai et al. (2018) - LSTNet",
            probabilistic=True,
            notes="CNN + RNN with skip connections",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("channels", "int", low=16, high=64, default=32),
            ParamSpace("rnn_hidden_size", "int", low=20, high=100, default=50),
            ParamSpace("skip", "int", low=12, high=48, default=24),
        ]


# ============================================================================
# Phase 4: Additional GluonTS Models
# ============================================================================


@register_model
class GluonTSSeasonalNaive(Forecaster):
    """
    Seasonal Naive forecaster.

    Repeats the last season for forecasting. Simple but effective baseline.

    Args:
        prediction_length: Forecast horizon
        season_length: Length of seasonal period
    """

    def __init__(
        self,
        prediction_length: int = 12,
        season_length: int = 12,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            season_length=season_length,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'GluonTSSeasonalNaive':
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'
        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        season_length = self.params.get('season_length', 12)

        # Repeat last season
        forecasts = []
        for i in range(horizon):
            idx = len(self._train_y) - season_length + (i % season_length)
            if idx < 0:
                idx = i % len(self._train_y)
            forecasts.append(self._train_y[idx])

        return np.array(forecasts)

    def _compute_fitted_values(self):
        if self._train_y is None:
            return None
        season_length = self.params.get('season_length', 12)
        n = len(self._train_y)
        fitted = np.full(n, np.nan)
        for i in range(season_length, n):
            fitted[i] = self._train_y[i - season_length]
        return fitted

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSSeasonalNaive",
            category=ModelCategory.CLASSICAL,
            library="gluonts",
            year=2019,
            paper="Baseline - Seasonal Naive",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("prediction_length", "int", low=1, high=48, default=12),
            ParamSpace("season_length", "int", low=2, high=52, default=12),
        ]


@register_model
class GluonTSPatchTST(_BaseGluonTSForecaster):
    """
    PatchTST: Patching-based Transformer for Time Series.

    Segments time series into patches similar to Vision Transformer.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        patch_len: Length of each patch
        stride: Stride between patches
        d_model: Model dimension
        nhead: Number of attention heads
        epochs: Training epochs
    """

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 32,
        nhead: int = 4,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            nhead=nhead,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.patch_tst import PatchTSTEstimator

        # Note: freq param removed in GluonTS 0.14+
        return PatchTSTEstimator(
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            patch_len=self.params.get('patch_len', 16),
            stride=self.params.get('stride', 8),
            d_model=self.params.get('d_model', 32),
            nhead=self.params.get('nhead', 4),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSPatchTST",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2023,
            paper="Nie et al. (2023) - A Time Series is Worth 64 Words",
            probabilistic=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("patch_len", "int", low=8, high=32, default=16),
            ParamSpace("stride", "int", low=4, high=16, default=8),
            ParamSpace("d_model", "int", low=16, high=128, default=32),
            ParamSpace("nhead", "int", low=2, high=8, default=4),
        ]


@_register_if_available('gluonts.torch.model.timegrad')
class GluonTSTimeGrad(_BaseGluonTSForecaster):
    """
    TimeGrad: Autoregressive Denoising Diffusion Models.

    Uses diffusion models for probabilistic time series forecasting.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        hidden_size: Hidden layer size
        num_layers: Number of RNN layers
        epochs: Training epochs
    """

    _supports_exog = True

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        hidden_size: int = 40,
        num_layers: int = 2,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.timegrad import TimeGradEstimator

        return TimeGradEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            hidden_size=self.params.get('hidden_size', 40),
            num_layers=self.params.get('num_layers', 2),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSTimeGrad",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2021,
            paper="Rasul et al. (2021) - Autoregressive Denoising Diffusion Models",
            probabilistic=True,
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("hidden_size", "int", low=20, high=100, default=40),
            ParamSpace("num_layers", "int", low=1, high=4, default=2),
        ]


@_register_if_available('gluonts.torch.model.tactis')
class GluonTSTACTiS(_BaseGluonTSForecaster):
    """
    TACTiS: Transformer-Attentional Copulas for Time Series.

    Learns copula structure with attention for multivariate forecasting.

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        d_model: Model dimension
        num_heads: Number of attention heads
        epochs: Training epochs
    """

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        d_model: int = 64,
        num_heads: int = 4,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            d_model=d_model,
            num_heads=num_heads,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.tactis import TACTiSEstimator

        return TACTiSEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            model_dim=self.params.get('d_model', 64),
            num_heads=self.params.get('num_heads', 4),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSTACTiS",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2022,
            paper="Drouin et al. (2022) - TACTiS: Transformer-Attentional Copulas",
            probabilistic=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("d_model", "int", low=32, high=128, default=64),
            ParamSpace("num_heads", "int", low=2, high=8, default=4),
        ]


def _register_if_cpflows_available(cls):
    """Only register MQF2 if cpflows package is installed."""
    try:
        import cpflows
        return register_model(cls)
    except ImportError:
        return cls


@_register_if_cpflows_available
class GluonTSMQF2(_BaseGluonTSForecaster):
    """
    MQF2: Multi-horizon Quantile Forecaster v2.

    Improved quantile forecasting architecture.
    Requires: pip install cpflows

    Args:
        prediction_length: Forecast horizon
        context_length: History context length
        d_model: Model dimension
        epochs: Training epochs
    """

    _supports_exog = True
    _needs_normalization = True

    def __init__(
        self,
        prediction_length: int = 12,
        context_length: Optional[int] = None,
        d_model: int = 64,
        epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            d_model=d_model,
            epochs=epochs,
            **kwargs
        )

    def _create_model(self):
        from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator

        return MQF2MultiHorizonEstimator(
            freq=self._freq,
            prediction_length=self.params.get('prediction_length', 12),
            context_length=self.params.get('context_length'),
            hidden_size=self.params.get('hidden_size', 40),
            trainer_kwargs={'max_epochs': get_fast_epochs(self.params.get('epochs', 100))},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GluonTSMQF2",
            category=ModelCategory.DEEP_LEARNING,
            library="gluonts",
            year=2022,
            paper="Kan et al. (2022) - MQF2: Multi-horizon Quantile Forecaster v2",
            probabilistic=True,
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("hidden_size", "int", low=20, high=100, default=40),
        ]
