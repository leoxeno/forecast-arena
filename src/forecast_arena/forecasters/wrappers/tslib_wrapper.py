"""
forecasters/wrappers/tslib_wrapper.py - Time-Series-Library model wrappers.

Wraps models from thuml/Time-Series-Library (THUML):
- Koopa: Koopman predictors for non-stationary time series
- FreTS: Frequency-domain MLP learners
- SCINet: Sample Convolution and Interaction Network
- ModernTCN: Pure CNN architecture with extended receptive field

NOTE: These models require the Time-Series-Library repo to be cloned and
added to PYTHONPATH. Models are only registered if available.

Install: Clone https://github.com/thuml/Time-Series-Library
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


def _register_if_tslib_model_available(model_name: str):
    """Only register model if the TSLib model module is available."""
    def decorator(cls):
        try:
            import importlib
            importlib.import_module(f'models.{model_name}')
            return register_model(cls)
        except (ImportError, ModuleNotFoundError):
            return cls
    return decorator


class _BaseTSLibForecaster(Forecaster):
    """Base class for Time-Series-Library models.

    TSLib models require the GitHub repo to be cloned and added to PYTHONPATH.

    Install:
        git clone https://github.com/thuml/Time-Series-Library
        export PYTHONPATH=$PYTHONPATH:/path/to/Time-Series-Library
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


@_register_if_tslib_model_available('Koopa')
class TSLibKoopa(_BaseTSLibForecaster):
    """
    Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors.

    Uses Koopman theory to decompose non-stationary dynamics into
    time-invariant components that are easier to predict.

    Key features:
    - Handles non-stationary time series well
    - Koopman-based temporal and frequency decomposition
    - Lightweight architecture

    Args:
        seq_len: Input sequence length (default: 96)
        pred_len: Prediction length (default: 96)
        d_model: Model dimension (default: 512)
        d_ff: Feedforward dimension (default: 2048)
        e_layers: Number of encoder layers (default: 2)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)

    Install:
        git clone https://github.com/thuml/Time-Series-Library
        export PYTHONPATH=$PYTHONPATH:/path/to/Time-Series-Library
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        d_model: int = 512,
        d_ff: int = 2048,
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
            d_ff=d_ff,
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
    ) -> 'TSLibKoopa':
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
            from models.Koopa import Model as KoopaModel
        except ImportError:
            raise ImportError(
                "Time-Series-Library not installed. Install with:\n"
                "git clone https://github.com/thuml/Time-Series-Library\n"
                "export PYTHONPATH=$PYTHONPATH:/path/to/Time-Series-Library"
            )

        device = self._get_device()
        seq_len = self.params.get('seq_len', 96)
        pred_len = self.params.get('pred_len', 96)
        epochs = self.params.get('epochs', 10)
        lr = self.params.get('learning_rate', 1e-4)
        batch_size = self.params.get('batch_size', 32)

        # Create config object for Koopa
        class Config:
            def __init__(self):
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.label_len = seq_len // 2
                self.enc_in = 1
                self.dec_in = 1
                self.c_out = 1
                self.d_model = 512
                self.d_ff = 2048
                self.e_layers = 2
                self.seg_len = 48
                self.num_series = 1
                self.dynamic_dim = 128
                self.hidden_dim = 64
                self.hidden_layers = 2

        config = Config()

        # Initialize model
        self._model = KoopaModel(config).to(device)

        # Prepare training data - create sliding windows
        X_train, y_train = self._create_sequences(y, seq_len, pred_len)

        if len(X_train) < 1:
            # Not enough data for training, store raw data
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
                # Koopa expects specific input format
                output = self._model(batch_x, None, None, None)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

        self._model.eval()
        self._is_fitted = True
        return self

    def _create_sequences(self, y: np.ndarray, seq_len: int, pred_len: int):
        """Create sliding window sequences for training."""
        X, Y = [], []
        for i in range(len(y) - seq_len - pred_len + 1):
            X.append(y[i:i + seq_len].reshape(-1, 1))
            Y.append(y[i + seq_len:i + seq_len + pred_len].reshape(-1, 1))
        return np.array(X) if X else np.array([]), np.array(Y) if Y else np.array([])

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
                # Fallback to last value
                forecast = np.full(horizon, y[-1])

        return forecast[:horizon] if len(forecast) >= horizon else np.pad(forecast, (0, horizon - len(forecast)), mode='edge')

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TSLibKoopa",
            category=ModelCategory.DEEP_LEARNING,
            library="Time-Series-Library",
            year=2023,
            paper="Liu et al. (2023) - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors",
            probabilistic=False,
            zero_shot=False,
            github_url="https://github.com/thuml/Time-Series-Library",
            notes="NeurIPS 2023, Koopman theory for non-stationary TS",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=48, high=192, default=96),
            ParamSpace("pred_len", "int", low=24, high=192, default=96),
            ParamSpace("epochs", "int", low=5, high=50, default=10),
            ParamSpace("learning_rate", "log_float", low=1e-5, high=1e-3, default=1e-4),
        ]


@_register_if_tslib_model_available('FreTS')
class TSLibFreTS(_BaseTSLibForecaster):
    """
    FreTS: Frequency-domain MLPs for Time Series Forecasting.

    Uses frequency-domain learning via DFT for efficient pattern capture.

    Key features:
    - Learns directly in frequency domain
    - Channel independence for multivariate
    - Computationally efficient

    Args:
        seq_len: Input sequence length (default: 96)
        pred_len: Prediction length (default: 96)
        embed_size: Embedding dimension (default: 128)
        hidden_size: Hidden dimension (default: 256)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)

    Install:
        git clone https://github.com/thuml/Time-Series-Library
        export PYTHONPATH=$PYTHONPATH:/path/to/Time-Series-Library
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        embed_size: int = 128,
        hidden_size: int = 256,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            embed_size=embed_size,
            hidden_size=hidden_size,
            epochs=epochs,
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
    ) -> 'TSLibFreTS':
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
            from models.FreTS import Model as FreTSModel
        except ImportError:
            raise ImportError(
                "Time-Series-Library not installed. Install with:\n"
                "git clone https://github.com/thuml/Time-Series-Library\n"
                "export PYTHONPATH=$PYTHONPATH:/path/to/Time-Series-Library"
            )

        device = self._get_device()
        seq_len = self.params.get('seq_len', 96)
        pred_len = self.params.get('pred_len', 96)
        epochs = self.params.get('epochs', 10)
        lr = self.params.get('learning_rate', 1e-4)

        # Create config for FreTS
        class Config:
            def __init__(self):
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.label_len = seq_len // 2
                self.enc_in = 1
                self.dec_in = 1
                self.c_out = 1
                self.embed_size = 128
                self.hidden_size = 256
                self.channel_independence = True

        config = Config()
        self._model = FreTSModel(config).to(device)

        # Training (similar to Koopa)
        X_train, y_train = self._create_sequences(y, seq_len, pred_len)

        if len(X_train) < 1:
            self._is_fitted = True
            return self

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

    def _create_sequences(self, y: np.ndarray, seq_len: int, pred_len: int):
        """Create sliding window sequences for training."""
        X, Y = [], []
        for i in range(len(y) - seq_len - pred_len + 1):
            X.append(y[i:i + seq_len].reshape(-1, 1))
            Y.append(y[i + seq_len:i + seq_len + pred_len].reshape(-1, 1))
        return np.array(X) if X else np.array([]), np.array(Y) if Y else np.array([])

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
            name="TSLibFreTS",
            category=ModelCategory.DEEP_LEARNING,
            library="Time-Series-Library",
            year=2023,
            paper="Yi et al. (2023) - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting",
            probabilistic=False,
            zero_shot=False,
            github_url="https://github.com/thuml/Time-Series-Library",
            notes="NeurIPS 2023, frequency-domain learning",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=48, high=192, default=96),
            ParamSpace("pred_len", "int", low=24, high=192, default=96),
            ParamSpace("epochs", "int", low=5, high=50, default=10),
        ]


@_register_if_tslib_model_available('SCINet')
class TSLibSCINet(_BaseTSLibForecaster):
    """
    SCINet: Sample Convolution and Interaction Network.

    Recursive downsampling with interaction for multi-scale pattern extraction.

    Key features:
    - Sample convolution for downsampling
    - Interactive learning between sub-series
    - Multi-resolution decomposition

    Args:
        seq_len: Input sequence length (default: 96)
        pred_len: Prediction length (default: 96)
        hidden_size: Hidden dimension (default: 64)
        num_levels: Number of decomposition levels (default: 2)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)

    Install:
        git clone https://github.com/thuml/Time-Series-Library
        export PYTHONPATH=$PYTHONPATH:/path/to/Time-Series-Library
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        hidden_size: int = 64,
        num_levels: int = 2,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            num_levels=num_levels,
            epochs=epochs,
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
    ) -> 'TSLibSCINet':
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
            from models.SCINet import Model as SCINetModel
        except ImportError:
            raise ImportError(
                "Time-Series-Library not installed. Install with:\n"
                "git clone https://github.com/thuml/Time-Series-Library\n"
                "export PYTHONPATH=$PYTHONPATH:/path/to/Time-Series-Library"
            )

        device = self._get_device()
        seq_len = self.params.get('seq_len', 96)
        pred_len = self.params.get('pred_len', 96)
        epochs = self.params.get('epochs', 10)
        lr = self.params.get('learning_rate', 1e-4)

        class Config:
            def __init__(self):
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.label_len = seq_len // 2
                self.enc_in = 1
                self.dec_in = 1
                self.c_out = 1
                self.hidden_size = 64
                self.num_levels = 2
                self.kernel_size = 5
                self.dropout = 0.5

        config = Config()
        self._model = SCINetModel(config).to(device)

        # Training
        X_train, y_train = self._create_sequences(y, seq_len, pred_len)

        if len(X_train) < 1:
            self._is_fitted = True
            return self

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

    def _create_sequences(self, y: np.ndarray, seq_len: int, pred_len: int):
        X, Y = [], []
        for i in range(len(y) - seq_len - pred_len + 1):
            X.append(y[i:i + seq_len].reshape(-1, 1))
            Y.append(y[i + seq_len:i + seq_len + pred_len].reshape(-1, 1))
        return np.array(X) if X else np.array([]), np.array(Y) if Y else np.array([])

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
            name="TSLibSCINet",
            category=ModelCategory.DEEP_LEARNING,
            library="Time-Series-Library",
            year=2022,
            paper="Liu et al. (2022) - SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction",
            probabilistic=False,
            zero_shot=False,
            github_url="https://github.com/thuml/Time-Series-Library",
            notes="NeurIPS 2022, multi-scale sample convolution",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=48, high=192, default=96),
            ParamSpace("pred_len", "int", low=24, high=192, default=96),
            ParamSpace("epochs", "int", low=5, high=50, default=10),
        ]


@_register_if_tslib_model_available('ModernTCN')
class TSLibModernTCN(_BaseTSLibForecaster):
    """
    ModernTCN: Pure CNN architecture with modernized design.

    Modernizes traditional TCN with extended receptive fields and
    efficient convolutional operations.

    Key features:
    - Pure convolutional architecture (no attention)
    - Extended receptive field through dilated convolutions
    - Efficient channel mixing

    Args:
        seq_len: Input sequence length (default: 96)
        pred_len: Prediction length (default: 96)
        d_model: Model dimension (default: 64)
        kernel_size: Convolution kernel size (default: 3)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)

    Install:
        git clone https://github.com/thuml/Time-Series-Library
        export PYTHONPATH=$PYTHONPATH:/path/to/Time-Series-Library
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        d_model: int = 64,
        kernel_size: int = 3,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            kernel_size=kernel_size,
            epochs=epochs,
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
    ) -> 'TSLibModernTCN':
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
            from models.ModernTCN import Model as ModernTCNModel
        except ImportError:
            raise ImportError(
                "Time-Series-Library not installed. Install with:\n"
                "git clone https://github.com/thuml/Time-Series-Library\n"
                "export PYTHONPATH=$PYTHONPATH:/path/to/Time-Series-Library"
            )

        device = self._get_device()
        seq_len = self.params.get('seq_len', 96)
        pred_len = self.params.get('pred_len', 96)
        epochs = self.params.get('epochs', 10)
        lr = self.params.get('learning_rate', 1e-4)

        class Config:
            def __init__(self):
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.label_len = seq_len // 2
                self.enc_in = 1
                self.dec_in = 1
                self.c_out = 1
                self.d_model = 64
                self.ffn_ratio = 2
                self.patch_size = 8
                self.patch_stride = 4
                self.num_blocks = 2
                self.large_size = 51
                self.small_size = 5
                self.dw_ks = [5, 13, 25]
                self.nvars = 1
                self.dropout = 0.2

        config = Config()
        self._model = ModernTCNModel(config).to(device)

        # Training
        X_train, y_train = self._create_sequences(y, seq_len, pred_len)

        if len(X_train) < 1:
            self._is_fitted = True
            return self

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

    def _create_sequences(self, y: np.ndarray, seq_len: int, pred_len: int):
        X, Y = [], []
        for i in range(len(y) - seq_len - pred_len + 1):
            X.append(y[i:i + seq_len].reshape(-1, 1))
            Y.append(y[i + seq_len:i + seq_len + pred_len].reshape(-1, 1))
        return np.array(X) if X else np.array([]), np.array(Y) if Y else np.array([])

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
            name="TSLibModernTCN",
            category=ModelCategory.DEEP_LEARNING,
            library="Time-Series-Library",
            year=2024,
            paper="Luo et al. (2024) - ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis",
            probabilistic=False,
            zero_shot=False,
            github_url="https://github.com/thuml/Time-Series-Library",
            notes="ICLR 2024, pure CNN with extended receptive field",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seq_len", "int", low=48, high=192, default=96),
            ParamSpace("pred_len", "int", low=24, high=192, default=96),
            ParamSpace("epochs", "int", low=5, high=50, default=10),
        ]
