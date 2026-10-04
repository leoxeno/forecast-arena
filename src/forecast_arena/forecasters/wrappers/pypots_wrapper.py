"""
forecasters/wrappers/pypots_wrapper.py - PyPOTS imputation model wrappers.

Wraps imputation models from PyPOTS library for time series forecasting:
- SAITS: Self-Attention-based Imputation for Time Series
- BRITS: Bidirectional Recurrent Imputation for Time Series
- GPVAE: Gaussian Process VAE for incomplete time series
- ImputeFormer: Low-rank Transformer for imputation
- CSDI: Conditional Score-based Diffusion for imputation

NOTE: These models use imputation for forecasting, which is an unconventional
approach. They are only registered if pypots is installed.

Imputation → Forecasting Strategy:
These models learn temporal patterns by imputing missing values. For forecasting,
we extend the series with masked (NaN) future positions and use the model to
impute those values, effectively predicting the future.

Install: pip install pypots
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd
import warnings

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace, get_fast_epochs
from .. import register_model


def _register_if_pypots_available(model_name: str):
    """Only register model if the pypots model is available."""
    def decorator(cls):
        try:
            import importlib
            importlib.import_module(f'pypots.imputation')
            # Also check if specific model exists
            from pypots import imputation
            if hasattr(imputation, model_name):
                return register_model(cls)
            return cls
        except (ImportError, ModuleNotFoundError):
            return cls
    return decorator


class _BasePyPOTSForecaster(Forecaster):
    """Base class for PyPOTS imputation-based forecasters.

    PyPOTS models are trained on time series with missing values and learn to
    impute them. For forecasting, we:
    1. Normalize data to zero-mean/unit-variance (critical for convergence)
    2. Train the model on the normalized data (with optional artificial masking)
    3. Create a forecast input by appending masked future positions
    4. Use the model to impute the masked positions → forecasts
    5. Denormalize back to the original scale
    """

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        """Normalize to zero-mean, unit-variance. Stores mean/std for inverse."""
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y))
        if self._y_std < 1e-8:
            self._y_std = 1.0  # Avoid division by zero for constant series
        return (y - self._y_mean) / self._y_std

    def _denormalize(self, y: np.ndarray) -> np.ndarray:
        """Inverse of _normalize."""
        return y * self._y_std + self._y_mean

    def _get_device(self):
        """Auto-detect best available device."""
        try:
            import torch
            device = self.params.get('device')
            if device is None:
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            return device
        except ImportError:
            return 'cpu'

    def _prepare_pypots_data(self, y: np.ndarray, mask_ratio: float = 0.15):
        """Prepare data in PyPOTS format with artificial masking for training.

        Args:
            y: 1D time series array
            mask_ratio: Fraction of values to randomly mask for training

        Returns:
            dict with 'X' key containing (n_samples, n_steps, n_features) array
            where masked values are NaN
        """
        # PyPOTS expects 3D: (n_samples, n_steps, n_features)
        # For univariate, we create sliding windows
        seq_len = self.params.get('seq_len', 48)

        # Create sliding windows
        windows = []
        for i in range(max(1, len(y) - seq_len + 1)):
            window = y[i:i + seq_len].copy()
            windows.append(window)

        if not windows:
            # If series too short, pad and use as single sample
            window = np.pad(y, (seq_len - len(y), 0), mode='edge')
            windows.append(window)

        X = np.array(windows).reshape(-1, seq_len, 1)  # (N, T, 1)

        # Apply artificial masking for training
        if mask_ratio > 0:
            mask = np.random.random(X.shape) < mask_ratio
            X_masked = X.copy()
            X_masked[mask] = np.nan
        else:
            X_masked = X

        return {"X": X_masked.astype(np.float32)}

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        try:
            # PyPOTS stores training loss after fit
            if hasattr(self._model, "best_loss"):
                d["best_loss"] = float(self._model.best_loss)
            if hasattr(self._model, "best_epoch"):
                d["best_epoch"] = int(self._model.best_epoch)
        except Exception:
            pass
        return d

    def _compute_fitted_values(self):
        if self._model is None or self._train_y is None:
            return None
        seq_len = self.params.get('seq_len', 48)
        n = len(self._train_y)
        if n < seq_len:
            return None
        # Work in normalized space (model was trained on normalized data)
        y_norm = (self._train_y - self._y_mean) / self._y_std
        # Create sliding windows, mask last position → 1-step-ahead reconstruction
        n_windows = n - seq_len + 1
        X = np.zeros((n_windows, seq_len, 1), dtype=np.float32)
        for i in range(n_windows):
            X[i, :, 0] = y_norm[i:i + seq_len]
        X_masked = X.copy()
        X_masked[:, -1, :] = np.nan  # mask last position for model to predict
        imputed = self._model.impute({"X": X_masked})
        if isinstance(imputed, dict):
            imputed = imputed.get('imputation', imputed.get('X_imputed', imputed))
        imputed = np.asarray(imputed).squeeze()  # (n_windows, seq_len)
        preds = imputed[:, -1] if imputed.ndim == 2 else imputed
        fitted = np.full(n, np.nan)
        fitted[seq_len - 1:seq_len - 1 + len(preds)] = self._denormalize(preds)
        return fitted

    def _prepare_forecast_input(self, y: np.ndarray, horizon: int):
        """Prepare input for forecasting by masking the last positions.

        PyPOTS models require the same sequence length at inference as training.
        We use context from the history and mask the last `horizon` positions
        for the model to impute (forecast). Data is normalized to match training.

        Args:
            y: Historical time series (original scale)
            horizon: Number of future steps to forecast

        Returns:
            Input array with same seq_len as training, last horizon values masked,
            in normalized space
        """
        seq_len = self.params.get('seq_len', 48)

        if horizon > seq_len:
            raise ValueError(
                f"Forecast horizon ({horizon}) cannot exceed seq_len ({seq_len}). "
                f"Increase seq_len or reduce horizon."
            )

        # Normalize to match training scale
        y_norm = (y - self._y_mean) / self._y_std

        # Take last seq_len values as the sequence
        if len(y_norm) >= seq_len:
            sequence = y_norm[-seq_len:].copy()
        else:
            sequence = np.pad(y_norm, (seq_len - len(y_norm), 0), mode='edge')

        # Mask the last `horizon` positions for the model to impute
        sequence[-horizon:] = np.nan

        # Reshape to PyPOTS format: (1, seq_len, 1)
        return sequence.reshape(1, seq_len, 1).astype(np.float32)


@_register_if_pypots_available('SAITS')
class PyPOTSSAITS(_BasePyPOTSForecaster):
    """
    SAITS: Self-Attention-based Imputation for Time Series.

    Uses diagonal-masked self-attention for joint imputation and forecasting.
    The model learns bidirectional temporal dependencies through two parallel
    attention branches.

    Key features:
    - Self-attention for capturing long-range dependencies
    - Bidirectional encoding (forward + backward)
    - Diagonal masking to prevent information leakage
    - Lightweight compared to full Transformer

    Args:
        seq_len: Input sequence length (default: 48)
        n_layers: Number of encoder layers (default: 2)
        d_model: Model dimension (default: 256)
        n_heads: Number of attention heads (default: 4)
        d_ffn: Feedforward dimension (default: 128)
        dropout: Dropout rate (default: 0.1)
        epochs: Training epochs (default: 100)
        batch_size: Training batch size (default: 32)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)

    Install: pip install pypots
    """

    def __init__(
        self,
        seq_len: int = 48,
        n_layers: int = 2,
        d_model: int = 256,
        n_heads: int = 4,
        d_ffn: int = 128,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            **kwargs
        )
        self._model = None

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'PyPOTSSAITS':
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        try:
            from pypots.imputation import SAITS
        except ImportError:
            raise ImportError("PyPOTS not installed. Install with: pip install pypots")

        device = self._get_device()
        seq_len = self.params.get('seq_len', 48)

        # Normalize for stable training (denormalized on output)
        y_norm = self._normalize(y)

        # Prepare training data with artificial masking
        train_data = self._prepare_pypots_data(y_norm, mask_ratio=0.15)

        # Suppress PyPOTS verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = SAITS(
                n_steps=seq_len,
                n_features=1,
                n_layers=self.params.get('n_layers', 2),
                d_model=self.params.get('d_model', 256),
                d_ffn=self.params.get('d_ffn', 128),
                n_heads=self.params.get('n_heads', 4),
                d_k=self.params.get('d_model', 256) // self.params.get('n_heads', 4),
                d_v=self.params.get('d_model', 256) // self.params.get('n_heads', 4),
                dropout=self.params.get('dropout', 0.1),
                attn_dropout=self.params.get('dropout', 0.1),
                epochs=get_fast_epochs(self.params.get('epochs', 100)),
                batch_size=self.params.get('batch_size', 32),
                device=device,
            )
            self._model.fit(train_data)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        assert self._train_y is not None
        assert self._model is not None

        # Prepare forecast input with context + masked future (normalized)
        forecast_input = self._prepare_forecast_input(self._train_y, horizon)

        # Impute (including masked future values)
        imputed = self._model.impute({"X": forecast_input})

        # PyPOTS returns numpy array directly (not dict) in newer versions
        if isinstance(imputed, dict):
            imputed = imputed.get('imputation', imputed.get('X_imputed', imputed))

        # Extract forecasted values (the last 'horizon' positions) and denormalize
        forecast = np.asarray(imputed).squeeze()[-horizon:]

        return self._denormalize(np.array(forecast))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSSAITS",
            category=ModelCategory.DEEP_LEARNING,
            library="pypots",
            year=2023,
            paper="Du et al. (2023) - SAITS: Self-Attention-based Imputation for Time Series",
            probabilistic=False,
            zero_shot=False,
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="ESWA 2023, diagonal-masked self-attention",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("n_layers", "int", low=1, high=4, default=2),
            ParamSpace("d_model", "int", low=64, high=512, default=256),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
            ParamSpace("learning_rate", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@_register_if_pypots_available('BRITS')
class PyPOTSBRITS(_BasePyPOTSForecaster):
    """
    BRITS: Bidirectional Recurrent Imputation for Time Series.

    Uses bidirectional RNNs with decay mechanisms to handle missing data.
    Captures both forward and backward temporal dynamics.

    Key features:
    - Bidirectional RNN architecture
    - Decay mechanism for handling irregular gaps
    - Joint regression and classification support
    - Lightweight and fast training

    Args:
        seq_len: Input sequence length (default: 48)
        rnn_hidden_size: RNN hidden dimension (default: 64)
        epochs: Training epochs (default: 100)
        batch_size: Training batch size (default: 32)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)

    Install: pip install pypots
    """

    def __init__(
        self,
        seq_len: int = 48,
        rnn_hidden_size: int = 64,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            rnn_hidden_size=rnn_hidden_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            **kwargs
        )
        self._model = None

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'PyPOTSBRITS':
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        try:
            from pypots.imputation import BRITS
        except ImportError:
            raise ImportError("PyPOTS not installed. Install with: pip install pypots")

        device = self._get_device()
        seq_len = self.params.get('seq_len', 48)

        # Normalize for stable training (denormalized on output)
        y_norm = self._normalize(y)

        train_data = self._prepare_pypots_data(y_norm, mask_ratio=0.15)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = BRITS(
                n_steps=seq_len,
                n_features=1,
                rnn_hidden_size=self.params.get('rnn_hidden_size', 64),
                epochs=get_fast_epochs(self.params.get('epochs', 100)),
                batch_size=self.params.get('batch_size', 32),
                device=device,
            )
            self._model.fit(train_data)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        assert self._train_y is not None
        assert self._model is not None

        forecast_input = self._prepare_forecast_input(self._train_y, horizon)
        imputed = self._model.impute({"X": forecast_input})
        # PyPOTS returns numpy array directly (not dict) in newer versions
        if isinstance(imputed, dict):
            imputed = imputed.get('imputation', imputed.get('X_imputed', imputed))
        forecast = np.asarray(imputed).squeeze()[-horizon:]

        return self._denormalize(np.array(forecast))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSBRITS",
            category=ModelCategory.DEEP_LEARNING,
            library="pypots",
            year=2018,
            paper="Cao et al. (2018) - BRITS: Bidirectional Recurrent Imputation for Time Series",
            probabilistic=False,
            zero_shot=False,
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="NeurIPS 2018, bidirectional RNN with decay",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("rnn_hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
            ParamSpace("learning_rate", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@_register_if_pypots_available('GPVAE')
class PyPOTSGPVAE(_BasePyPOTSForecaster):
    """
    GP-VAE: Gaussian Process VAE for incomplete time series.

    Combines variational autoencoders with Gaussian process priors for
    smooth temporal interpolation with uncertainty quantification.

    Key features:
    - GP prior for smooth latent trajectories
    - VAE for flexible data distribution
    - Probabilistic imputation with uncertainty
    - Handles irregular sampling naturally

    Args:
        seq_len: Input sequence length (default: 48)
        latent_size: Latent dimension (default: 16)
        encoder_sizes: Encoder hidden sizes (default: [128, 64])
        decoder_sizes: Decoder hidden sizes (default: [64, 128])
        epochs: Training epochs (default: 100)
        batch_size: Training batch size (default: 32)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)

    Install: pip install pypots
    """

    def __init__(
        self,
        seq_len: int = 48,
        latent_size: int = 16,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            latent_size=latent_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            **kwargs
        )
        self._model = None

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'PyPOTSGPVAE':
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        try:
            from pypots.imputation import GPVAE
        except ImportError:
            raise ImportError("PyPOTS not installed. Install with: pip install pypots")

        device = self._get_device()
        seq_len = self.params.get('seq_len', 48)

        # Normalize for stable training (denormalized on output)
        y_norm = self._normalize(y)

        train_data = self._prepare_pypots_data(y_norm, mask_ratio=0.15)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = GPVAE(
                n_steps=seq_len,
                n_features=1,
                latent_size=self.params.get('latent_size', 16),
                encoder_sizes=[128, 64],
                decoder_sizes=[64, 128],
                epochs=get_fast_epochs(self.params.get('epochs', 100)),
                batch_size=self.params.get('batch_size', 32),
                device=device,
            )
            self._model.fit(train_data)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        assert self._train_y is not None
        assert self._model is not None

        forecast_input = self._prepare_forecast_input(self._train_y, horizon)
        imputed = self._model.impute({"X": forecast_input})
        # PyPOTS returns numpy array directly (not dict) in newer versions
        if isinstance(imputed, dict):
            imputed = imputed.get('imputation', imputed.get('X_imputed', imputed))
        forecast = np.asarray(imputed).squeeze()[-horizon:]

        return self._denormalize(np.array(forecast))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSGPVAE",
            category=ModelCategory.PROBABILISTIC,
            library="pypots",
            year=2020,
            paper="Fortuin et al. (2020) - GP-VAE: Deep Probabilistic Time Series Imputation",
            probabilistic=True,
            zero_shot=False,
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="AISTATS 2020, Gaussian Process prior + VAE",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("latent_size", "int", low=8, high=64, default=16),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
            ParamSpace("learning_rate", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@_register_if_pypots_available('ImputeFormer')
class PyPOTSImputeFormer(_BasePyPOTSForecaster):
    """
    ImputeFormer: Low-rank Transformer for time series imputation.

    Uses low-rank attention approximation for efficient imputation
    with linear complexity in sequence length.

    Key features:
    - Low-rank attention mechanism (O(n) complexity)
    - Handles long sequences efficiently
    - Learnable projection for reduced memory
    - State-of-the-art imputation performance

    Args:
        seq_len: Input sequence length (default: 48)
        n_layers: Number of Transformer layers (default: 2)
        d_model: Model dimension (default: 128)
        n_heads: Number of attention heads (default: 4)
        d_ffn: Feedforward dimension (default: 256)
        epochs: Training epochs (default: 100)
        batch_size: Training batch size (default: 32)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)

    Install: pip install pypots
    """

    def __init__(
        self,
        seq_len: int = 48,
        n_layers: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        d_ffn: int = 256,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            **kwargs
        )
        self._model = None

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'PyPOTSImputeFormer':
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        try:
            from pypots.imputation import ImputeFormer
        except ImportError:
            raise ImportError("PyPOTS not installed. Install with: pip install pypots")

        device = self._get_device()
        seq_len = self.params.get('seq_len', 48)

        # Normalize for stable training (denormalized on output)
        y_norm = self._normalize(y)

        train_data = self._prepare_pypots_data(y_norm, mask_ratio=0.15)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = ImputeFormer(
                n_steps=seq_len,
                n_features=1,
                n_layers=self.params.get('n_layers', 2),
                d_input_embed=self.params.get('d_input_embed', 64),
                d_learnable_embed=self.params.get('d_learnable_embed', 64),
                d_proj=self.params.get('d_proj', 64),
                d_ffn=self.params.get('d_ffn', 128),
                n_temporal_heads=self.params.get('n_temporal_heads', 4),
                dropout=0.1,
                epochs=get_fast_epochs(self.params.get('epochs', 100)),
                batch_size=self.params.get('batch_size', 32),
                device=device,
            )
            self._model.fit(train_data)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        assert self._train_y is not None
        assert self._model is not None

        forecast_input = self._prepare_forecast_input(self._train_y, horizon)
        imputed = self._model.impute({"X": forecast_input})
        # PyPOTS returns numpy array directly (not dict) in newer versions
        if isinstance(imputed, dict):
            imputed = imputed.get('imputation', imputed.get('X_imputed', imputed))
        forecast = np.asarray(imputed).squeeze()[-horizon:]

        return self._denormalize(np.array(forecast))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSImputeFormer",
            category=ModelCategory.DEEP_LEARNING,
            library="pypots",
            year=2024,
            paper="Nie et al. (2024) - ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation",
            probabilistic=False,
            zero_shot=False,
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="KDD 2024, low-rank attention for efficiency",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("n_layers", "int", low=1, high=4, default=2),
            ParamSpace("d_model", "int", low=64, high=256, default=128),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
            ParamSpace("learning_rate", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


@_register_if_pypots_available('CSDI')
class PyPOTSCSDI(_BasePyPOTSForecaster):
    """
    CSDI: Conditional Score-based Diffusion Models for Imputation.

    Uses diffusion models conditioned on observed data to generate
    plausible imputations with uncertainty quantification.

    Key features:
    - Score-based diffusion for high-quality generation
    - Conditional on observed values
    - Probabilistic with multiple samples
    - State-of-the-art imputation quality

    Args:
        seq_len: Input sequence length (default: 48)
        n_features: Number of features (default: 1 for univariate)
        n_layers: Number of residual layers (default: 4)
        n_channels: Number of channels (default: 64)
        d_time_embedding: Time embedding dimension (default: 128)
        n_heads: Number of attention heads (default: 8)
        n_diffusion_steps: Number of diffusion steps (default: 50)
        epochs: Training epochs (default: 100)
        batch_size: Training batch size (default: 16)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)

    Install: pip install pypots
    """

    def __init__(
        self,
        seq_len: int = 48,
        n_layers: int = 4,
        n_channels: int = 64,
        d_time_embedding: int = 128,
        n_heads: int = 8,
        n_diffusion_steps: int = 50,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            n_layers=n_layers,
            n_channels=n_channels,
            d_time_embedding=d_time_embedding,
            n_heads=n_heads,
            n_diffusion_steps=n_diffusion_steps,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            **kwargs
        )
        self._model = None

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'PyPOTSCSDI':
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        try:
            from pypots.imputation import CSDI
        except ImportError:
            raise ImportError("PyPOTS not installed. Install with: pip install pypots")

        device = self._get_device()
        seq_len = self.params.get('seq_len', 48)

        # Normalize for stable training (denormalized on output)
        y_norm = self._normalize(y)

        train_data = self._prepare_pypots_data(y_norm, mask_ratio=0.15)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = CSDI(
                n_steps=seq_len,
                n_features=1,
                n_layers=self.params.get('n_layers', 4),
                n_heads=self.params.get('n_heads', 8),
                n_channels=self.params.get('n_channels', 64),
                d_time_embedding=self.params.get('d_time_embedding', 128),
                d_feature_embedding=self.params.get('d_feature_embedding', 16),
                d_diffusion_embedding=self.params.get('d_diffusion_embedding', 128),
                n_diffusion_steps=self.params.get('n_diffusion_steps', 50),
                epochs=get_fast_epochs(self.params.get('epochs', 100)),
                batch_size=self.params.get('batch_size', 16),
                device=device,
            )
            self._model.fit(train_data)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        assert self._train_y is not None
        assert self._model is not None

        forecast_input = self._prepare_forecast_input(self._train_y, horizon)
        imputed = self._model.impute({"X": forecast_input})
        # PyPOTS returns numpy array directly (not dict) in newer versions
        if isinstance(imputed, dict):
            imputed = imputed.get('imputation', imputed.get('X_imputed', imputed))
        forecast = np.asarray(imputed).squeeze()[-horizon:]

        return self._denormalize(np.array(forecast))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSCSDI",
            category=ModelCategory.PROBABILISTIC,
            library="pypots",
            year=2021,
            paper="Tashiro et al. (2021) - CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation",
            probabilistic=True,
            zero_shot=False,
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="NeurIPS 2021, diffusion model for imputation",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("n_layers", "int", low=2, high=8, default=4),
            ParamSpace("n_channels", "int", low=32, high=128, default=64),
            ParamSpace("n_diffusion_steps", "int", low=20, high=100, default=50),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
            ParamSpace("learning_rate", "log_float", low=1e-4, high=1e-2, default=1e-3),
        ]


# =============================================================================
# PyPOTS NATIVE FORECASTING MODELS (pypots.forecasting)
#
# These models use pypots.forecasting which provides direct multi-step
# forecasting (NOT imputation-based). Each model predicts n_pred_steps
# future values in a single forward pass.
# =============================================================================


def _register_if_pypots_forecasting_available(model_name: str):
    """Only register model if the pypots.forecasting model is available."""
    def decorator(cls):
        try:
            from pypots import forecasting
            if hasattr(forecasting, model_name):
                return register_model(cls)
            return cls
        except (ImportError, ModuleNotFoundError):
            return cls
    return decorator


class _BasePyPOTSNativeForecaster(Forecaster):
    """Base class for PyPOTS native forecasting models (pypots.forecasting).

    Unlike imputation-based models, these directly predict future values
    using model.forecast(). They output n_pred_steps values in a single
    forward pass — NO recursive single-step prediction.

    All data is normalized to zero-mean/unit-variance before training to
    ensure stable convergence, and denormalized on output.
    """

    _pypots_cls_name: str = ""

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        """Normalize to zero-mean, unit-variance. Stores mean/std for inverse."""
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y))
        if self._y_std < 1e-8:
            self._y_std = 1.0  # Avoid division by zero for constant series
        return (y - self._y_mean) / self._y_std

    def _denormalize(self, y: np.ndarray) -> np.ndarray:
        """Inverse of _normalize."""
        return y * self._y_std + self._y_mean

    def _get_device(self):
        """Auto-detect best available device."""
        try:
            import torch
            device = self.params.get('device')
            if device is None:
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            return device
        except ImportError:
            return 'cpu'

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        try:
            if hasattr(self._model, "best_loss"):
                d["best_loss"] = float(self._model.best_loss)
            if hasattr(self._model, "best_epoch"):
                d["best_epoch"] = int(self._model.best_epoch)
        except Exception:
            pass
        return d

    def _compute_fitted_values(self):
        if self._model is None or self._train_y is None:
            return None
        n = len(self._train_y)
        seq_len = self._seq_len
        horizon = self._horizon
        n_windows = max(1, n - seq_len - horizon + 1)
        if n_windows < 1:
            return None
        # Work in normalized space (model was trained on normalized data)
        y_norm = (self._train_y - self._y_mean) / self._y_std
        X_windows = np.zeros((n_windows, seq_len, 1), dtype=np.float32)
        for i in range(n_windows):
            X_windows[i, :, 0] = y_norm[i:i + seq_len]
        X_pred_test = np.zeros((n_windows, horizon, 1), dtype=np.float32)
        forecast = self._model.forecast({"X": X_windows, "X_pred": X_pred_test})
        forecast = np.asarray(forecast).squeeze()
        preds = forecast[:, 0] if forecast.ndim == 2 else forecast
        fitted = np.full(n, np.nan)
        fitted[seq_len:seq_len + len(preds)] = self._denormalize(preds)
        return fitted

    def _get_model_kwargs(self) -> dict:
        """Return extra kwargs for PyPOTS model constructor. Override per model."""
        return {}

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BasePyPOTSNativeForecaster':
        y = self._validate_y(y)
        self._train_y = y

        import importlib
        mod = importlib.import_module('pypots.forecasting')
        cls = getattr(mod, self._pypots_cls_name)

        horizon = kwargs.get('horizon', 24)
        seq_len = self.params.get('seq_len', 48)

        # Auto-reduce seq_len when data is too short for seq_len + horizon
        if len(y) < seq_len + horizon:
            seq_len = max(10, len(y) - horizon)

        self._horizon = horizon
        self._seq_len = seq_len

        # Normalize for stable training (denormalized on output)
        y_norm = self._normalize(y)

        # Create sliding windows with ground truth future for training
        # PyPOTS forecasting requires {"X": context, "X_pred": future} for both fit and forecast
        n_windows = max(1, len(y_norm) - seq_len - horizon + 1)
        X_windows = []
        X_pred_windows = []
        for i in range(n_windows):
            X_windows.append(y_norm[i:i + seq_len])
            X_pred_windows.append(y_norm[i + seq_len:i + seq_len + horizon])
        X_train = np.array(X_windows).reshape(-1, seq_len, 1).astype(np.float32)
        X_pred_train = np.array(X_pred_windows).reshape(-1, horizon, 1).astype(np.float32)

        device = self._get_device()
        model_kwargs = self._get_model_kwargs()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = cls(
                n_steps=seq_len,
                n_features=1,
                n_pred_steps=horizon,
                n_pred_features=1,
                device=device,
                epochs=get_fast_epochs(self.params.get('epochs', 100)),
                batch_size=self.params.get('batch_size', 32),
                **model_kwargs,
            )
            self._model.fit({"X": X_train, "X_pred": X_pred_train})

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        assert self._train_y is not None
        assert self._model is not None

        # Normalize context to match training scale
        y_norm = (self._train_y - self._y_mean) / self._y_std
        context = y_norm[-self._seq_len:]
        X_test = context.reshape(1, self._seq_len, 1).astype(np.float32)
        X_pred_test = np.zeros((1, self._horizon, 1), dtype=np.float32)

        # Direct forecast — single forward pass, no recursion
        forecast = self._model.forecast({"X": X_test, "X_pred": X_pred_test})
        forecast = np.asarray(forecast).squeeze()

        return self._denormalize(forecast[:horizon])


# --- BTTF (special constructor — uses pred_step, not n_pred_steps) ---

@_register_if_pypots_forecasting_available('BTTF')
class PyPOTSFcstBTTF(Forecaster):
    """
    BTTF: Bayesian Temporal Tensor Factorization for forecasting.

    Uses Bayesian tensor factorization with Gibbs sampling for
    temporal pattern discovery and multi-step forecasting.

    Args:
        seq_len: Input sequence length (default: 48)
        rank: Tensor rank (default: 10)
        burn_iter: Burn-in iterations (default: 200)
        gibbs_iter: Gibbs sampling iterations (default: 200)
    """

    def __init__(self, seq_len: int = 48, rank: int = 10,
                 burn_iter: int = 200, gibbs_iter: int = 200,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, rank=rank, burn_iter=burn_iter,
                         gibbs_iter=gibbs_iter, device=device, **kwargs)

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        """Normalize to zero-mean, unit-variance. Stores mean/std for inverse."""
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y))
        if self._y_std < 1e-8:
            self._y_std = 1.0
        return (y - self._y_mean) / self._y_std

    def _denormalize(self, y: np.ndarray) -> np.ndarray:
        """Inverse of _normalize."""
        return y * self._y_std + self._y_mean

    def fit(self, y: Union[np.ndarray, pd.Series], X=None, freq=None, **kwargs) -> 'PyPOTSFcstBTTF':
        y = self._validate_y(y)
        self._train_y = y

        from pypots.forecasting import BTTF

        horizon = kwargs.get('horizon', 24)
        seq_len = self.params.get('seq_len', 48)
        self._horizon = horizon
        self._seq_len = seq_len

        # Normalize for stable training (denormalized on output)
        y_norm = self._normalize(y)

        n_windows = max(1, len(y_norm) - seq_len + 1)
        windows = [y_norm[i:i + seq_len] for i in range(n_windows)]
        X_train = np.array(windows).reshape(-1, seq_len, 1).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = BTTF(
                n_steps=seq_len,
                n_features=1,
                pred_step=horizon,
                rank=self.params.get('rank', 10),
                time_lags=[1, 2, 3, 5, 7, 12, 24],
                burn_iter=self.params.get('burn_iter', 200),
                gibbs_iter=self.params.get('gibbs_iter', 200),
                device=self.params.get('device') or 'cpu',
            )
            self._model.fit({"X": X_train})

        self._is_fitted = True
        return self

    def _compute_fitted_values(self):
        if self._model is None or self._train_y is None:
            return None
        n = len(self._train_y)
        seq_len = self._seq_len
        horizon = self._horizon
        n_windows = max(1, n - seq_len - horizon + 1)
        if n_windows < 1:
            return None
        # Work in normalized space
        y_norm = (self._train_y - self._y_mean) / self._y_std
        X_windows = np.zeros((n_windows, seq_len, 1), dtype=np.float32)
        for i in range(n_windows):
            X_windows[i, :, 0] = y_norm[i:i + seq_len]
        forecast = self._model.forecast({"X": X_windows})
        forecast = np.asarray(forecast).squeeze()
        preds = forecast[:, 0] if forecast.ndim == 2 else forecast
        fitted = np.full(n, np.nan)
        fitted[seq_len:seq_len + len(preds)] = self._denormalize(preds)
        return fitted

    def predict(self, horizon: int, X=None, **kwargs) -> np.ndarray:
        self._check_fitted()
        assert self._train_y is not None and self._model is not None
        y_norm = (self._train_y - self._y_mean) / self._y_std
        context = y_norm[-self._seq_len:]
        X_test = context.reshape(1, self._seq_len, 1).astype(np.float32)
        forecast = self._model.forecast({"X": X_test})
        return self._denormalize(np.asarray(forecast).squeeze()[:horizon])

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstBTTF", category=ModelCategory.ML,
            library="pypots", year=2021,
            paper="Chen et al. (2021) - Bayesian Temporal Factorization for Multidimensional Time Series Prediction",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="TKDE 2021, Bayesian tensor factorization with Gibbs sampling",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("rank", "int", low=5, high=30, default=10),
        ]


# --- Standard PyPOTS forecasting models (all share n_pred_steps API) ---

@_register_if_pypots_forecasting_available('CSDI')
class PyPOTSFcstCSDI(_BasePyPOTSNativeForecaster):
    """PyPOTS CSDI: Conditional Score-based Diffusion for forecasting."""
    _pypots_cls_name = "CSDI"

    def __init__(self, seq_len: int = 48, n_layers: int = 4, n_heads: int = 8,
                 n_channels: int = 64, n_diffusion_steps: int = 50,
                 epochs: int = 100, batch_size: int = 16,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, n_layers=n_layers, n_heads=n_heads,
                         n_channels=n_channels, n_diffusion_steps=n_diffusion_steps,
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {
            "n_layers": self.params.get("n_layers", 4),
            "n_heads": self.params.get("n_heads", 8),
            "n_channels": self.params.get("n_channels", 64),
            "d_time_embedding": 128, "d_feature_embedding": 16,
            "d_diffusion_embedding": 128,
            "n_diffusion_steps": self.params.get("n_diffusion_steps", 50),
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstCSDI", category=ModelCategory.PROBABILISTIC,
            library="pypots", year=2021,
            paper="Tashiro et al. (2021) - CSDI: Conditional Score-based Diffusion Models",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="NeurIPS 2021, diffusion model for forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("n_diffusion_steps", "int", low=20, high=100, default=50),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]


@_register_if_pypots_forecasting_available('DLinear')
class PyPOTSFcstDLinear(_BasePyPOTSNativeForecaster):
    """PyPOTS DLinear: Decomposition-Linear for forecasting."""
    _pypots_cls_name = "DLinear"

    def __init__(self, seq_len: int = 48, moving_avg_window_size: int = 25,
                 d_model: int = 128, epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, moving_avg_window_size=moving_avg_window_size,
                         d_model=d_model, epochs=epochs, batch_size=batch_size,
                         device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {
            "moving_avg_window_size": self.params.get("moving_avg_window_size", 25),
            "individual": False,
            "d_model": self.params.get("d_model", 128),
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstDLinear", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2023,
            paper="Zeng et al. (2023) - Are Transformers Effective for Time Series Forecasting?",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="AAAI 2023, simple linear decomposition beats complex models",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("moving_avg_window_size", "int", low=5, high=49, default=25),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]


@_register_if_pypots_forecasting_available('FITS')
class PyPOTSFcstFITS(_BasePyPOTSNativeForecaster):
    """PyPOTS FITS: Frequency Interpolation Time Series forecasting."""
    _pypots_cls_name = "FITS"

    def __init__(self, seq_len: int = 48, cut_freq: int = 2,
                 epochs: int = 200, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, cut_freq=cut_freq,
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {"cut_freq": self.params.get("cut_freq", 2), "individual": False}

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstFITS", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2024,
            paper="Xu et al. (2024) - FITS: Modeling Time Series with 10k Parameters",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="ICLR 2024, frequency-domain interpolation, ultra-lightweight",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("cut_freq", "int", low=1, high=10, default=2),
            ParamSpace("epochs", "int", low=100, high=400, default=200),
        ]


@_register_if_pypots_forecasting_available('FiLM')
class PyPOTSFcstFiLM(_BasePyPOTSNativeForecaster):
    """PyPOTS FiLM: Frequency improved Legendre Memory Model."""
    _pypots_cls_name = "FiLM"

    def __init__(self, seq_len: int = 48, d_model: int = 128,
                 epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, d_model=d_model,
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {"d_model": self.params.get("d_model", 128), "window_size": [2], "multiscale": [1, 2]}

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstFiLM", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2022,
            paper="Zhou et al. (2022) - FiLM: Frequency improved Legendre Memory Model for LTSF",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="NeurIPS 2022, Legendre projections + frequency features",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("d_model", "int", low=64, high=256, default=128),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]


# GPT4TS removed — PyPOTS TemporalEmbedding requires timestamp columns (month, day,
# See: https://github.com/WenjieDu/PyPOTS — pypots.forecasting.GPT4TS


@_register_if_pypots_forecasting_available('MICN')
class PyPOTSFcstMICN(_BasePyPOTSNativeForecaster):
    """PyPOTS MICN: Multi-scale Isometric Convolution Network."""
    _pypots_cls_name = "MICN"

    def __init__(self, seq_len: int = 48, n_layers: int = 2, d_model: int = 128,
                 conv_kernel: Optional[List[int]] = None,
                 epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, n_layers=n_layers, d_model=d_model,
                         conv_kernel=conv_kernel or [12, 16],
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {
            "n_layers": self.params.get("n_layers", 2),
            "d_model": self.params.get("d_model", 128),
            "conv_kernel": self.params.get("conv_kernel", [12, 16]),
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstMICN", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2023,
            paper="Wang et al. (2023) - MICN: Multi-scale Local and Global Context Modeling for LTSF",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="ICLR 2023, multi-scale isometric convolution",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("n_layers", "int", low=1, high=4, default=2),
            ParamSpace("d_model", "int", low=64, high=256, default=128),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]


@_register_if_pypots_forecasting_available('ModernTCN')
class PyPOTSFcstModernTCN(_BasePyPOTSNativeForecaster):
    """PyPOTS ModernTCN: A Modern Pure Convolution Structure for forecasting."""
    _pypots_cls_name = "ModernTCN"

    def __init__(self, seq_len: int = 48, patch_size: int = 8, patch_stride: int = 4,
                 epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, patch_size=patch_size, patch_stride=patch_stride,
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {
            "patch_size": self.params.get("patch_size", 8),
            "patch_stride": self.params.get("patch_stride", 4),
            "downsampling_ratio": 2,
            "ffn_ratio": 2,
            "num_blocks": [1],
            "large_size": [51],
            "small_size": [5],
            "dims": [64],
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstModernTCN", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2024,
            paper="Luo & Wang (2024) - ModernTCN: A Modern Pure Convolution Structure for General TS Analysis",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="ICLR 2024, pure convolution beats transformers",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("patch_size", "int", low=4, high=16, default=8),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]


# MOMENT removed — PyPOTS upstream bug: forward() pads input by n_pred_steps but
# _get_head() uses only n_steps for the linear projection layer dimension.
# This causes a size mismatch (e.g., head expects 3072 but gets 4608).
# See: https://github.com/WenjieDu/PyPOTS — pypots.forecasting.MOMENT


@_register_if_pypots_forecasting_available('SegRNN')
class PyPOTSFcstSegRNN(_BasePyPOTSNativeForecaster):
    """PyPOTS SegRNN: Segment Recurrent Neural Network."""
    _pypots_cls_name = "SegRNN"

    def __init__(self, seq_len: int = 48, seg_len: int = 12, d_model: int = 128,
                 dropout: float = 0.1, epochs: int = 200, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, seg_len=seg_len, d_model=d_model,
                         dropout=dropout, epochs=epochs, batch_size=batch_size,
                         device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        seg_len = self.params.get("seg_len", 12)
        # seg_len must divide BOTH n_steps (seq_len) AND n_pred_steps (horizon)
        horizon = getattr(self, '_horizon', 18)
        seq_len = getattr(self, '_seq_len', 48)
        from math import gcd
        divisor = gcd(seq_len, horizon)  # GCD guarantees divisibility by both
        if seg_len > divisor or seq_len % seg_len != 0 or horizon % seg_len != 0:
            seg_len = divisor  # Largest value that divides both
        return {
            "seg_len": seg_len,
            "d_model": self.params.get("d_model", 128),
            "dropout": self.params.get("dropout", 0.1),
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstSegRNN", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2024,
            paper="Lin et al. (2024) - SegRNN: Segment Recurrent Neural Network for LTSF",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="ICML 2024, segment-wise RNN for efficiency",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("seg_len", "int", low=4, high=24, default=12),
            ParamSpace("d_model", "int", low=64, high=256, default=128),
            ParamSpace("epochs", "int", low=100, high=400, default=200),
        ]


@_register_if_pypots_forecasting_available('TEFN')
class PyPOTSFcstTEFN(_BasePyPOTSNativeForecaster):
    """PyPOTS TEFN: Time-Enhanced Feature Network."""
    _pypots_cls_name = "TEFN"

    def __init__(self, seq_len: int = 48, n_fod: int = 2,
                 epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, n_fod=n_fod,
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {"n_fod": self.params.get("n_fod", 2)}

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstTEFN", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2024,
            paper="Wu et al. (2024) - TEFN: Time-Enhanced Feature Network",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="Fractional-order derivative features for forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("n_fod", "int", low=1, high=5, default=2),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]


@_register_if_pypots_forecasting_available('TimeLLM')
class PyPOTSFcstTimeLLM(_BasePyPOTSNativeForecaster):
    """PyPOTS TimeLLM: Time Series Forecasting by Reprogramming Large Language Models."""
    _pypots_cls_name = "TimeLLM"

    def __init__(self, seq_len: int = 48, patch_size: int = 16, patch_stride: int = 8,
                 n_layers: int = 6, d_model: int = 128,
                 epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, patch_size=patch_size, patch_stride=patch_stride,
                         n_layers=n_layers, d_model=d_model,
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {
            "term": "short",
            "llm_model_type": "GPT2",
            "n_layers": self.params.get("n_layers", 6),
            "patch_size": self.params.get("patch_size", 16),
            "patch_stride": self.params.get("patch_stride", 8),
            "d_llm": 768,  # GPT-2 hidden size
            "d_model": self.params.get("d_model", 128),
            "d_ffn": 256,
            "n_heads": 4,
            "dropout": 0.1,
            "domain_prompt_content": "Univariate time series forecasting.",
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstTimeLLM", category=ModelCategory.LLM,
            library="pypots", year=2024,
            paper="Jin et al. (2024) - Time-LLM: Time Series Forecasting by Reprogramming Large Language Models",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="ICLR 2024, reprogram frozen LLM for time series",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("patch_size", "int", low=8, high=32, default=16),
            ParamSpace("n_layers", "int", low=2, high=12, default=6),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]


@_register_if_pypots_forecasting_available('TimeMixer')
class PyPOTSFcstTimeMixer(_BasePyPOTSNativeForecaster):
    """PyPOTS TimeMixer: Decomposable Multiscale Mixing."""
    _pypots_cls_name = "TimeMixer"

    def __init__(self, seq_len: int = 48, n_layers: int = 2, d_model: int = 128,
                 d_ffn: int = 256, top_k: int = 5, dropout: float = 0.1,
                 epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, n_layers=n_layers, d_model=d_model,
                         d_ffn=d_ffn, top_k=top_k, dropout=dropout,
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {
            "term": "short",
            "n_layers": self.params.get("n_layers", 2),
            "d_model": self.params.get("d_model", 128),
            "d_ffn": self.params.get("d_ffn", 256),
            "top_k": self.params.get("top_k", 5),
            "dropout": self.params.get("dropout", 0.1),
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstTimeMixer", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2024,
            paper="Wang et al. (2024) - TimeMixer: Decomposable Multiscale Mixing for TS Forecasting",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="ICLR 2024, multiscale mixing with decomposition",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("n_layers", "int", low=1, high=4, default=2),
            ParamSpace("d_model", "int", low=64, high=256, default=128),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]


@_register_if_pypots_forecasting_available('TimesNet')
class PyPOTSFcstTimesNet(_BasePyPOTSNativeForecaster):
    """PyPOTS TimesNet: Temporal 2D-Variation Modeling."""
    _pypots_cls_name = "TimesNet"

    def __init__(self, seq_len: int = 48, n_layers: int = 2, top_k: int = 3,
                 d_model: int = 64, d_ffn: int = 128,
                 epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, n_layers=n_layers, top_k=top_k,
                         d_model=d_model, d_ffn=d_ffn,
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        return {
            "n_layers": self.params.get("n_layers", 2),
            "top_k": self.params.get("top_k", 3),
            "d_model": self.params.get("d_model", 64),
            "d_ffn": self.params.get("d_ffn", 128),
            "n_kernels": 6,
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstTimesNet", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2023,
            paper="Wu et al. (2023) - TimesNet: Temporal 2D-Variation Modeling for General TS Analysis",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="ICLR 2023, 2D variation blocks via FFT period detection",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("n_layers", "int", low=1, high=4, default=2),
            ParamSpace("d_model", "int", low=32, high=128, default=64),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]


@_register_if_pypots_forecasting_available('Transformer')
class PyPOTSFcstTransformer(_BasePyPOTSNativeForecaster):
    """PyPOTS Transformer: Standard Transformer for forecasting."""
    _pypots_cls_name = "Transformer"

    def __init__(self, seq_len: int = 48, n_encoder_layers: int = 2,
                 n_decoder_layers: int = 1, d_model: int = 128,
                 n_heads: int = 4, d_ffn: int = 256, dropout: float = 0.1,
                 epochs: int = 100, batch_size: int = 32,
                 device: Optional[str] = None, **kwargs):
        super().__init__(seq_len=seq_len, n_encoder_layers=n_encoder_layers,
                         n_decoder_layers=n_decoder_layers, d_model=d_model,
                         n_heads=n_heads, d_ffn=d_ffn, dropout=dropout,
                         epochs=epochs, batch_size=batch_size, device=device, **kwargs)

    def _get_model_kwargs(self) -> dict:
        d_model = self.params.get("d_model", 128)
        n_heads = self.params.get("n_heads", 4)
        return {
            "n_encoder_layers": self.params.get("n_encoder_layers", 2),
            "n_decoder_layers": self.params.get("n_decoder_layers", 1),
            "d_model": d_model,
            "n_heads": n_heads,
            "d_k": d_model // n_heads,
            "d_v": d_model // n_heads,
            "d_ffn": self.params.get("d_ffn", 256),
            "dropout": self.params.get("dropout", 0.1),
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyPOTSFcstTransformer", category=ModelCategory.DEEP_LEARNING,
            library="pypots", year=2017,
            paper="Vaswani et al. (2017) - Attention Is All You Need",
            github_url="https://github.com/WenjieDu/PyPOTS",
            notes="Standard encoder-decoder Transformer for forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=24, high=96, default=48),
            ParamSpace("n_encoder_layers", "int", low=1, high=4, default=2),
            ParamSpace("d_model", "int", low=64, high=256, default=128),
            ParamSpace("n_heads", "int", low=2, high=8, default=4),
            ParamSpace("epochs", "int", low=50, high=200, default=100),
        ]
