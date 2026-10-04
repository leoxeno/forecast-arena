"""
forecasters/wrappers/mamba_wrapper.py - Mamba/SSM time series model wrappers.

Wraps State Space Model (SSM) based forecasters:
- S-Mamba: Bidirectional Mamba for multivariate time series
- TimeMachine: Quadruple-Mamba architecture for multi-scale forecasting

These models leverage Mamba's selective state space mechanism for efficient
long-range dependency modeling with linear complexity.

NOTE: These models require CUDA and mamba-ssm which may not be available.
Models are only registered if dependencies are present.

Install:
    # S-Mamba
    git clone https://github.com/wzhwzhwzh0921/S-D-Mamba
    export PYTHONPATH=$PYTHONPATH:/path/to/S-D-Mamba

    # TimeMachine
    git clone https://github.com/Atik-Ahamed/TimeMachine
    export PYTHONPATH=$PYTHONPATH:/path/to/TimeMachine

    # Core dependency
    pip install mamba-ssm  # Requires CUDA
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


def _register_if_mamba_available():
    """Only register model if mamba-ssm and the model are available."""
    def decorator(cls):
        try:
            # Check if mamba-ssm is available (requires CUDA)
            import torch
            if not torch.cuda.is_available():
                return cls  # Don't register without CUDA
            import mamba_ssm  # noqa: F401
            return register_model(cls)
        except (ImportError, ModuleNotFoundError):
            return cls
    return decorator


class _BaseMambaForecaster(Forecaster):
    """Base class for Mamba-based forecasters.

    Mamba models use State Space Models (SSMs) with selective mechanisms
    to achieve linear-time sequence modeling while capturing long-range
    dependencies.

    Note: mamba-ssm requires CUDA. These wrappers will raise ImportError
    on CPU-only systems.
    """

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

    def _create_sequences(self, y: np.ndarray, seq_len: int, pred_len: int):
        """Create sliding window sequences for training."""
        X, Y = [], []
        for i in range(len(y) - seq_len - pred_len + 1):
            X.append(y[i:i + seq_len].reshape(-1, 1))
            Y.append(y[i + seq_len:i + seq_len + pred_len].reshape(-1, 1))
        return np.array(X) if X else np.array([]), np.array(Y) if Y else np.array([])


@_register_if_mamba_available()
class SMamba(_BaseMambaForecaster):
    """
    S-Mamba: Bidirectional Mamba for Time Series Forecasting.

    Delegates inter-variate correlations and temporal dependencies to a
    bidirectional Mamba block, achieving strong performance with linear
    complexity.

    Key features:
    - Bidirectional Mamba for capturing both forward/backward patterns
    - Effective for multivariate time series
    - Linear complexity in sequence length
    - Lightweight compared to Transformer-based models

    Args:
        seq_len: Input sequence length (default: 96)
        pred_len: Prediction length (default: 96)
        d_model: Model dimension (default: 128)
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        e_layers: Number of encoder layers (default: 2)
        epochs: Training epochs (default: 10)
        learning_rate: Learning rate (default: 1e-4)
        batch_size: Batch size (default: 32)
        device: 'cpu', 'cuda' (auto-detected if None)

    Install:
        git clone https://github.com/wzhwzhwzh0921/S-D-Mamba
        export PYTHONPATH=$PYTHONPATH:/path/to/S-D-Mamba
        pip install mamba-ssm  # Requires CUDA
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        e_layers: int = 2,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            e_layers=e_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
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
    ) -> 'SMamba':
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch required: pip install torch")

        try:
            # Try to import S-Mamba model
            from models.S_Mamba import Model as SMambaModel
        except ImportError:
            raise ImportError(
                "S-Mamba not installed. Install with:\n"
                "git clone https://github.com/wzhwzhwzh0921/S-D-Mamba\n"
                "export PYTHONPATH=$PYTHONPATH:/path/to/S-D-Mamba\n"
                "pip install mamba-ssm  # Requires CUDA"
            )

        device = self._get_device()
        seq_len = self.params.get('seq_len', 96)
        pred_len = self.params.get('pred_len', 96)
        epochs = self.params.get('epochs', 10)
        lr = self.params.get('learning_rate', 1e-4)
        batch_size = self.params.get('batch_size', 32)

        # Create config for S-Mamba
        class Config:
            def __init__(self):
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.label_len = seq_len // 2
                self.enc_in = 1
                self.dec_in = 1
                self.c_out = 1
                self.d_model = 128
                self.d_state = 16
                self.d_conv = 4
                self.expand = 2
                self.e_layers = 2
                self.dropout = 0.1

        config = Config()
        self._model = SMambaModel(config).to(device)

        # Prepare training data
        X_train, y_train = self._create_sequences(y, seq_len, pred_len)

        if len(X_train) < 1:
            self._is_fitted = True
            return self

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                output = self._model(batch_x, None, None, None)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

        self._model.eval()
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

        assert self._train_y is not None

        device = self._get_device()
        seq_len = self.params.get('seq_len', 96)

        # Get last seq_len values
        y = self._train_y
        if len(y) >= seq_len:
            context = y[-seq_len:]
        else:
            context = np.pad(y, (seq_len - len(y), 0), mode='edge')

        # Prepare input
        x = torch.tensor(context.reshape(1, -1, 1), dtype=torch.float32).to(device)

        # Predict
        with torch.no_grad():
            if self._model is not None:
                output = self._model(x, None, None, None)
                forecast = output.squeeze().cpu().numpy()
            else:
                forecast = np.full(horizon, y[-1])

        return forecast[:horizon] if len(forecast) >= horizon else np.pad(forecast, (0, horizon - len(forecast)), mode='edge')

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SMamba",
            category=ModelCategory.DEEP_LEARNING,
            library="S-D-Mamba",
            year=2024,
            paper="Wang et al. (2024) - Is Mamba Effective for Time Series Forecasting?",
            probabilistic=False,
            zero_shot=False,
            github_url="https://github.com/wzhwzhwzh0921/S-D-Mamba",
            notes="Neurocomputing 2024, bidirectional SSM, 368★",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=48, high=192, default=96),
            ParamSpace("pred_len", "int", low=24, high=192, default=96),
            ParamSpace("d_model", "int", low=64, high=256, default=128),
            ParamSpace("epochs", "int", low=5, high=50, default=10),
            ParamSpace("learning_rate", "log_float", low=1e-5, high=1e-3, default=1e-4),
        ]


@_register_if_mamba_available()
class TimeMachine(_BaseMambaForecaster):
    """
    TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting.

    Uses a quadruple-Mamba architecture with multi-scale contextual cues
    for effective long-range forecasting.

    Key features:
    - Four Mamba modules at two resolution levels
    - Multi-scale context extraction (high + low resolution)
    - Handles channel-mixing and channel-independence
    - Linear scalability with small memory footprint

    Args:
        seq_len: Input sequence length (default: 96)
        pred_len: Prediction length (default: 96)
        d_model: Model dimension (default: 64)
        d_state: SSM state dimension (default: 16)
        e_layers: Number of encoder layers (default: 2)
        epochs: Training epochs (default: 10)
        learning_rate: Learning rate (default: 1e-4)
        batch_size: Batch size (default: 32)
        device: 'cpu', 'cuda' (auto-detected if None)

    Install:
        git clone https://github.com/Atik-Ahamed/TimeMachine
        export PYTHONPATH=$PYTHONPATH:/path/to/TimeMachine
        pip install mamba-ssm  # Requires CUDA
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        d_model: int = 64,
        d_state: int = 16,
        e_layers: int = 2,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            d_state=d_state,
            e_layers=e_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
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
    ) -> 'TimeMachine':
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch required: pip install torch")

        try:
            from models.TimeMachine import Model as TimeMachineModel
        except ImportError:
            raise ImportError(
                "TimeMachine not installed. Install with:\n"
                "git clone https://github.com/Atik-Ahamed/TimeMachine\n"
                "export PYTHONPATH=$PYTHONPATH:/path/to/TimeMachine\n"
                "pip install mamba-ssm  # Requires CUDA"
            )

        device = self._get_device()
        seq_len = self.params.get('seq_len', 96)
        pred_len = self.params.get('pred_len', 96)
        epochs = self.params.get('epochs', 10)
        lr = self.params.get('learning_rate', 1e-4)
        batch_size = self.params.get('batch_size', 32)

        # Create config for TimeMachine
        class Config:
            def __init__(self):
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.label_len = seq_len // 2
                self.enc_in = 1
                self.c_out = 1
                self.d_model = 64
                self.d_state = 16
                self.e_layers = 2
                self.dropout = 0.1
                self.patch_len = 16
                self.stride = 8

        config = Config()
        self._model = TimeMachineModel(config).to(device)

        # Prepare training data
        X_train, y_train = self._create_sequences(y, seq_len, pred_len)

        if len(X_train) < 1:
            self._is_fitted = True
            return self

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                output = self._model(batch_x, None, None, None)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

        self._model.eval()
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

        assert self._train_y is not None

        device = self._get_device()
        seq_len = self.params.get('seq_len', 96)

        y = self._train_y
        if len(y) >= seq_len:
            context = y[-seq_len:]
        else:
            context = np.pad(y, (seq_len - len(y), 0), mode='edge')

        x = torch.tensor(context.reshape(1, -1, 1), dtype=torch.float32).to(device)

        with torch.no_grad():
            if self._model is not None:
                output = self._model(x, None, None, None)
                forecast = output.squeeze().cpu().numpy()
            else:
                forecast = np.full(horizon, y[-1])

        return forecast[:horizon] if len(forecast) >= horizon else np.pad(forecast, (0, horizon - len(forecast)), mode='edge')

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TimeMachine",
            category=ModelCategory.DEEP_LEARNING,
            library="TimeMachine",
            year=2024,
            paper="Ahamed & Cheng (2024) - TimeMachine: A Time Series is Worth 4 Mambas",
            probabilistic=False,
            zero_shot=False,
            github_url="https://github.com/Atik-Ahamed/TimeMachine",
            notes="ECAI 2024, quadruple-Mamba, multi-scale, 209★",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=48, high=192, default=96),
            ParamSpace("pred_len", "int", low=24, high=192, default=96),
            ParamSpace("d_model", "int", low=32, high=128, default=64),
            ParamSpace("epochs", "int", low=5, high=50, default=10),
            ParamSpace("learning_rate", "log_float", low=1e-5, high=1e-3, default=1e-4),
        ]
