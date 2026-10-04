"""
forecasters/wrappers/tsfm_wrapper.py - IBM Granite TSFM model wrappers.

Wraps IBM's time series foundation models from granite-tsfm:
- TinyTimeMixer (TTM): Lightweight zero-shot/few-shot foundation model
- TTM variants: Different context/prediction length configurations

Install: pip install granite-tsfm

All models support zero-shot forecasting with optional fine-tuning.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


@register_model
class TinyTimeMixer(Forecaster):
    """
    TinyTimeMixer (TTM): Lightweight time series foundation model from IBM.

    First "tiny" pre-trained model for time series forecasting. Achieves
    state-of-the-art zero-shot performance with only ~1M parameters.

    Key features:
    - Extremely fast inference (works on CPU/laptops)
    - Zero-shot and few-shot forecasting
    - Multi-variate support via channel independence/mixing
    - Pre-trained on ~700M time series samples

    Args:
        model_variant: '512-96', '1024-96', '1536-96' (context-prediction lengths)
        context_length: Override context length (default from variant)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        model_variant: str = '512-96',
        context_length: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_variant=model_variant,
            context_length=context_length,
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
    ) -> 'TinyTimeMixer':
        """TTM is zero-shot - just stores context."""
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'
        self._is_fitted = True
        return self

    def _get_device(self):
        """Auto-detect best available device."""
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

    def _get_model_path(self) -> str:
        """Get HuggingFace model path based on variant."""
        variant = self.params.get('model_variant', '512-96')
        # TTM r2 models with different context lengths
        return f"ibm-granite/granite-timeseries-ttm-r2"

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        try:
            import torch
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        except ImportError:
            raise ImportError(
                "TTM (granite-tsfm) not installed. Install with:\n"
                "pip install granite-tsfm"
            )

        device = self._get_device()
        model_path = self._get_model_path()

        # Load model (cached after first call)
        if self._model is None:
            self._model = TinyTimeMixerForPrediction.from_pretrained(
                model_path,
                revision='main',
            )
            self._model = self._model.to(device)
            self._model.eval()

        # Get context length from model config
        context_length = self.params.get('context_length') or self._model.config.context_length
        prediction_length = self._model.config.prediction_length

        # Prepare context - TTM expects (batch, n_channels, seq_len)
        assert self._train_y is not None, "Model not fitted"
        y: np.ndarray = self._train_y.copy()
        if len(y) > context_length:
            y = y[-context_length:]

        # Pad if necessary (TTM can accept padded short sequences)
        if len(y) < context_length:
            y = np.pad(y, (context_length - len(y), 0), mode='edge')

        # Standard scale the data (TTM requires this)
        mean = float(np.mean(y))
        std = float(np.std(y)) + 1e-8
        y_scaled = (y - mean) / std

        # Shape: (batch=1, seq_len, n_channels=1) — granite-tsfm 0.3+ format
        context = torch.tensor(y_scaled, dtype=torch.float32).view(1, context_length, 1).to(device)

        # Forecast
        with torch.no_grad():
            output = self._model(past_values=context)
            forecast = output.prediction_outputs.squeeze().cpu().numpy()

        # Inverse scale
        forecast = forecast * std + mean

        # Handle horizon mismatch - TTM outputs fixed prediction_length
        if horizon <= len(forecast):
            return forecast[:horizon]
        else:
            # Need rolling prediction for longer horizons
            return self._rolling_predict(horizon, context_length, mean, std, device)

    def _rolling_predict(
        self,
        horizon: int,
        context_length: int,
        mean: float,
        std: float,
        device: str
    ) -> np.ndarray:
        """Rolling prediction for horizons longer than model's prediction_length."""
        import torch

        assert self._model is not None
        assert self._train_y is not None
        prediction_length = self._model.config.prediction_length
        all_forecasts: list = []
        y: np.ndarray = self._train_y.copy()

        while len(all_forecasts) < horizon:
            # Get latest context
            if len(y) > context_length:
                context_y = y[-context_length:]
            else:
                context_y = np.pad(y, (context_length - len(y), 0), mode='edge')

            # Scale
            y_scaled = (context_y - mean) / std
            context = torch.tensor(y_scaled, dtype=torch.float32).view(1, context_length, 1).to(device)

            # Predict
            with torch.no_grad():
                output = self._model(past_values=context)
                forecast = output.prediction_outputs.squeeze().cpu().numpy()

            # Inverse scale
            forecast = forecast * std + mean

            # Append forecasts and extend context
            all_forecasts.extend(forecast)
            y = np.concatenate([y, forecast])

        return np.array(all_forecasts[:horizon])

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TinyTimeMixer",
            category=ModelCategory.LLM,
            library="granite-tsfm",
            year=2024,
            paper="Ekambaram et al. (2024) - Tiny Time Mixers: Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting",
            probabilistic=False,
            zero_shot=True,
            huggingface_id="ibm-granite/granite-timeseries-ttm-r2",
            github_url="https://github.com/ibm-granite/granite-tsfm",
            notes="1M params, works on laptops, NeurIPS 2024",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("model_variant", "categorical", choices=['512-96', '1024-96', '1536-96'], default='512-96'),
            ParamSpace("context_length", "int", low=64, high=1536, default=512),
        ]


@register_model
class TTMFinetuned(Forecaster):
    """
    TinyTimeMixer with fine-tuning capability.

    Same as TinyTimeMixer but supports fine-tuning on the training data
    for improved performance on specific domains.

    Args:
        model_variant: '512-96', '1024-96', '1536-96'
        epochs: Number of fine-tuning epochs (default: 5)
        learning_rate: Learning rate for fine-tuning
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        model_variant: str = '512-96',
        epochs: int = 5,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_variant=model_variant,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            **kwargs
        )
        self._model = None

    def _get_device(self):
        """Auto-detect best available device."""
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

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'TTMFinetuned':
        """Fine-tune TTM on the provided data."""
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        try:
            import torch
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        except ImportError:
            raise ImportError(
                "TTM (granite-tsfm) not installed. Install with:\n"
                "pip install granite-tsfm"
            )

        device = self._get_device()

        # Load base model
        self._model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2",
            revision='main',
        )
        self._model = self._model.to(device)

        context_length = self._model.config.context_length
        prediction_length = self._model.config.prediction_length

        # Only fine-tune if we have enough data
        if len(y) > context_length + prediction_length:
            self._fine_tune(y, context_length, prediction_length, device)

        self._model.eval()
        self._is_fitted = True
        return self

    def _fine_tune(
        self,
        y: np.ndarray,
        context_length: int,
        prediction_length: int,
        device: str
    ):
        """Perform fine-tuning on the training data."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        epochs = self.params.get('epochs', 5)
        lr = self.params.get('learning_rate', 1e-4)

        # Standard scale
        mean = np.mean(y)
        std = np.std(y) + 1e-8
        y_scaled = (y - mean) / std

        # Create sliding window samples
        X_samples = []
        Y_samples = []
        for i in range(len(y_scaled) - context_length - prediction_length + 1):
            X_samples.append(y_scaled[i:i + context_length])
            Y_samples.append(y_scaled[i + context_length:i + context_length + prediction_length])

        X_tensor = torch.tensor(np.array(X_samples), dtype=torch.float32).unsqueeze(-1)  # (N, context, 1)
        Y_tensor = torch.tensor(np.array(Y_samples), dtype=torch.float32).unsqueeze(-1)  # (N, pred, 1)

        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

        # Fine-tune with frozen backbone, only train head
        assert self._model is not None
        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        for _ in range(epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                output = self._model(past_values=batch_x)
                loss = loss_fn(output.prediction_outputs, batch_y)
                loss.backward()
                optimizer.step()

        self._mean = mean
        self._std = std

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        import torch

        assert self._model is not None
        assert self._train_y is not None

        device = self._get_device()
        context_length = self._model.config.context_length
        prediction_length = self._model.config.prediction_length

        y: np.ndarray = self._train_y.copy()
        if len(y) > context_length:
            y = y[-context_length:]
        if len(y) < context_length:
            y = np.pad(y, (context_length - len(y), 0), mode='edge')

        # Use stored scaling if fine-tuned, else compute fresh
        mean = float(getattr(self, '_mean', np.mean(y)))
        std = float(getattr(self, '_std', np.std(y) + 1e-8))

        y_scaled = (y - mean) / std

        # Autoregressive rolling if horizon > prediction_length
        all_preds: list[np.ndarray] = []
        remaining = horizon
        context_buf = y_scaled.copy()

        with torch.no_grad():
            while remaining > 0:
                ctx = torch.tensor(
                    context_buf[-context_length:], dtype=torch.float32
                ).view(1, context_length, 1).to(device)
                output = self._model(past_values=ctx)
                chunk = output.prediction_outputs.squeeze().cpu().numpy()
                take = min(len(chunk), remaining)
                all_preds.append(chunk[:take])
                remaining -= take
                # Slide context forward
                context_buf = np.concatenate([context_buf, chunk[:take]])

        forecast = np.concatenate(all_preds) * std + mean
        return forecast[:horizon]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TTMFinetuned",
            category=ModelCategory.LLM,
            library="granite-tsfm",
            year=2024,
            paper="Ekambaram et al. (2024) - Tiny Time Mixers",
            probabilistic=False,
            zero_shot=False,
            huggingface_id="ibm-granite/granite-timeseries-ttm-r2",
            github_url="https://github.com/ibm-granite/granite-tsfm",
            notes="Fine-tunable version, 5% data for competitive results",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("model_variant", "categorical", choices=['512-96', '1024-96', '1536-96'], default='512-96'),
            ParamSpace("epochs", "int", low=1, high=20, default=5),
            ParamSpace("learning_rate", "log_float", low=1e-5, high=1e-3, default=1e-4),
        ]


@register_model
class FlowState(Forecaster):
    """
    FlowState: Timescale-invariant SSM foundation model from IBM.

    First timescale-adjustable time series foundation model using
    State Space Model (SSM) encoder with functional basis decoder.

    Key features:
    - 9.1M parameters (smallest model in GIFT-Eval top 10)
    - Zero-shot forecasting with dynamic horizon support
    - Sampling rate invariant - adapts to different timescales
    - SSM architecture (different from Transformer-based models)

    Args:
        context_length: Number of past observations (default: 512)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        context_length: int = 512,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            context_length=context_length,
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
    ) -> 'FlowState':
        """FlowState is zero-shot - just stores context."""
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'
        self._is_fitted = True
        return self

    def _get_device(self):
        """Auto-detect best available device."""
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

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        try:
            import torch
            from tsfm_public.models.flowstate import FlowStateForPrediction
        except ImportError:
            raise ImportError(
                "FlowState (granite-tsfm) not installed. Install with:\n"
                "pip install granite-tsfm"
            )

        device = self._get_device()
        context_length = self.params.get('context_length', 512)

        # Load model (cached after first call)
        if self._model is None:
            self._model = FlowStateForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-flowstate-r1",
                revision='main',
            )
            self._model = self._model.to(device)
            self._model.eval()

        # Prepare context
        assert self._train_y is not None
        y: np.ndarray = self._train_y.copy()
        if len(y) > context_length:
            y = y[-context_length:]

        # Pad if necessary
        if len(y) < context_length:
            y = np.pad(y, (context_length - len(y), 0), mode='edge')

        # Standard scale the data
        mean = float(np.mean(y))
        std = float(np.std(y)) + 1e-8
        y_scaled = (y - mean) / std

        # Shape: (batch=1, seq_len, n_channels=1) — granite-tsfm 0.3+ expects channels last
        context = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        # FlowState can handle arbitrary horizons natively
        with torch.no_grad():
            output = self._model(context, prediction_length=horizon)
            forecast = output.prediction_outputs.squeeze().cpu().numpy()

        # Inverse scale
        forecast = forecast * std + mean

        return forecast[:horizon]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="FlowState",
            category=ModelCategory.LLM,
            library="granite-tsfm",
            year=2025,
            paper="IBM Research (2025) - FlowState: Sampling-Rate Invariant Time Series Foundation Model",
            probabilistic=False,
            zero_shot=True,
            huggingface_id="ibm-granite/granite-timeseries-flowstate-r1",
            github_url="https://github.com/ibm-granite/granite-tsfm",
            notes="9.1M params, SSM architecture, timescale invariant, GIFT-Eval #2",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("context_length", "int", low=64, high=1024, default=512),
        ]
