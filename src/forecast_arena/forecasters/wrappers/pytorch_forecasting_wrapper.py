"""
forecasters/wrappers/pytorch_forecasting_wrapper.py - PyTorch Forecasting models.

Wraps PyTorch Forecasting (Jan Beitner):
- TemporalFusionTransformer
- DeepAR
- NBeats
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace, get_fast_epochs
from .. import register_model


@register_model
class TFT(Forecaster):
    """
    Temporal Fusion Transformer from PyTorch Forecasting.

    Attention-based architecture with interpretable outputs.

    Args:
        max_prediction_length: Maximum forecast horizon
        max_encoder_length: Maximum history length
        hidden_size: Hidden layer size
        attention_head_size: Size of attention heads
        dropout: Dropout rate
        max_epochs: Training epochs
    """

    def __init__(
        self,
        max_prediction_length: int = 12,
        max_encoder_length: int = 24,
        hidden_size: int = 16,
        attention_head_size: int = 1,
        dropout: float = 0.1,
        max_epochs: int = 50,
        **kwargs
    ):
        super().__init__(
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            max_epochs=max_epochs,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'TFT':
        try:
            import lightning.pytorch as pl
            from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
            from pytorch_forecasting.data import GroupNormalizer
            from pytorch_forecasting.metrics import MAE
            import warnings
        except ImportError:
            raise ImportError(
                "pytorch_forecasting not installed. Install with: "
                "pip install pytorch-forecasting pytorch-lightning"
            )

        y = self._validate_y(y)
        self._train_y = y

        # Use horizon from fit kwargs (passed by runner), override constructor default
        if 'horizon' in kwargs:
            self.params['max_prediction_length'] = kwargs['horizon']

        max_encoder_length = self.params.get('max_encoder_length', 24)
        max_prediction_length = self.params.get('max_prediction_length', 12)

        # Prepare data
        df = pd.DataFrame({
            'time_idx': np.arange(len(y)),
            'series': 0,  # Integer group ID
            'target': y.astype(np.float32),
        })

        # Handle exogenous features
        self._exog_cols: list = []
        self._train_exog = None
        if X is not None:
            exog = self._validate_X(X, len(y))
            if exog is not None:
                self._train_exog = exog.astype(np.float32)
                for j in range(exog.shape[1]):
                    col_name = f'exog_{j}'
                    df[col_name] = exog[:, j].astype(np.float32)
                    self._exog_cols.append(col_name)

        self._train_df = df

        ds_kwargs = {
            'time_idx': 'time_idx',
            'target': 'target',
            'group_ids': ['series'],
            'max_encoder_length': max_encoder_length,
            'max_prediction_length': max_prediction_length,
            'time_varying_unknown_reals': ['target'],
            'target_normalizer': GroupNormalizer(groups=['series']),
            'allow_missing_timesteps': True,
        }
        if self._exog_cols:
            ds_kwargs['time_varying_known_reals'] = self._exog_cols

        training = TimeSeriesDataSet(df, **ds_kwargs)

        train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = TemporalFusionTransformer.from_dataset(
                training,
                hidden_size=self.params.get('hidden_size', 16),
                attention_head_size=self.params.get('attention_head_size', 1),
                dropout=self.params.get('dropout', 0.1),
                hidden_continuous_size=8,
                loss=MAE(),  # Use MAE loss metric
            )

            trainer = pl.Trainer(
                max_epochs=get_fast_epochs(self.params.get('max_epochs', 50)),
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(self._model, train_dataloaders=train_dataloader)

        self._training = training
        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        if self._exog_cols:
            from pytorch_forecasting import TimeSeriesDataSet

            max_prediction_length = self.params.get('max_prediction_length', 12)
            n_train = len(self._train_y)

            future_df = pd.DataFrame({
                'time_idx': np.arange(n_train, n_train + max_prediction_length),
                'series': 0,
                'target': np.zeros(max_prediction_length, dtype=np.float32),
            })
            exog_future = self._validate_X(X, max_prediction_length) if X is not None else None
            for j, col in enumerate(self._exog_cols):
                if exog_future is not None:
                    future_df[col] = exog_future[:, j].astype(np.float32)
                else:
                    future_df[col] = 0.0

            full_df = pd.concat([self._train_df, future_df], ignore_index=True)
            pred_dataset = TimeSeriesDataSet.from_dataset(
                self._training, full_df, predict=True, stop_randomization=True
            )
            pred_data = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        else:
            pred_data = self._training.to_dataloader(train=False, batch_size=1, num_workers=0)

        predictions = self._model.predict(pred_data)

        return predictions[0, :horizon].numpy()

    def _compute_fitted_values(self):
        if self._model is None or not hasattr(self, '_training'):
            return None
        pred_data = self._training.to_dataloader(
            train=False, batch_size=64, num_workers=0
        )
        predictions = self._model.predict(pred_data)
        preds = predictions[:, 0].numpy().flatten()  # 1-step-ahead
        fitted = np.full(len(self._train_y), np.nan)
        fitted[-len(preds):] = preds
        return fitted

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TFT",
            category=ModelCategory.DEEP_LEARNING,
            library="pytorch-forecasting",
            year=2021,
            paper="Lim et al. (2021)",
            probabilistic=True,
            multivariate=True,
            exogenous=True,
            github_url="https://github.com/jdb78/pytorch-forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("hidden_size", "int", low=8, high=64, default=16),
            ParamSpace("attention_head_size", "int", low=1, high=4, default=1),
            ParamSpace("dropout", "float", low=0.0, high=0.5, default=0.1),
        ]


class _BasePyTorchForecasting(Forecaster):
    """Base class for PyTorch Forecasting models."""

    def _prepare_data(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        simple_model: bool = False,
    ):
        """Prepare data in PyTorch Forecasting format.

        Args:
            y: Time series data
            X: Optional exogenous features (2D array)
            simple_model: If True, use minimal config for models like NBeats
                          that only accept target as input
        """
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        max_encoder_length = self.params.get('max_encoder_length', 24)
        max_prediction_length = self.params.get('max_prediction_length', 12)

        df = pd.DataFrame({
            'time_idx': np.arange(len(y)),
            'series': 0,  # Integer group ID (required)
            'target': y.astype(np.float32),
        })

        # Handle exogenous features
        self._exog_cols: list = []
        self._train_exog = None
        if X is not None and not simple_model:
            exog = self._validate_X(X, len(y))
            if exog is not None:
                self._train_exog = exog.astype(np.float32)
                for j in range(exog.shape[1]):
                    col_name = f'exog_{j}'
                    df[col_name] = exog[:, j].astype(np.float32)
                    self._exog_cols.append(col_name)

        self._train_df = df

        if simple_model:
            # For NBeats and similar models that only accept target
            training = TimeSeriesDataSet(
                df,
                time_idx='time_idx',
                target='target',
                group_ids=['series'],
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                time_varying_unknown_reals=['target'],
                allow_missing_timesteps=True,
            )
        else:
            # For TFT, DeepAR, RecurrentNetwork that can use additional features
            ds_kwargs = {
                'time_idx': 'time_idx',
                'target': 'target',
                'group_ids': ['series'],
                'max_encoder_length': max_encoder_length,
                'max_prediction_length': max_prediction_length,
                'time_varying_unknown_reals': ['target'],
                'target_normalizer': GroupNormalizer(groups=['series']),
                'allow_missing_timesteps': True,
            }
            if self._exog_cols:
                ds_kwargs['time_varying_known_reals'] = self._exog_cols
            training = TimeSeriesDataSet(df, **ds_kwargs)

        return training

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        if self._exog_cols:
            from pytorch_forecasting import TimeSeriesDataSet

            max_prediction_length = self.params.get('max_prediction_length', 12)
            n_train = len(self._train_y)

            # Build future rows with exog values
            future_df = pd.DataFrame({
                'time_idx': np.arange(n_train, n_train + max_prediction_length),
                'series': 0,
                'target': np.zeros(max_prediction_length, dtype=np.float32),
            })
            exog_future = self._validate_X(X, max_prediction_length) if X is not None else None
            for j, col in enumerate(self._exog_cols):
                if exog_future is not None:
                    future_df[col] = exog_future[:, j].astype(np.float32)
                else:
                    future_df[col] = 0.0

            full_df = pd.concat([self._train_df, future_df], ignore_index=True)
            pred_dataset = TimeSeriesDataSet.from_dataset(
                self._training, full_df, predict=True, stop_randomization=True
            )
            pred_data = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        else:
            pred_data = self._training.to_dataloader(train=False, batch_size=1, num_workers=0)

        predictions = self._model.predict(pred_data)

        return predictions[0, :horizon].numpy()

    def _compute_fitted_values(self):
        if self._model is None or not hasattr(self, '_training'):
            return None
        pred_data = self._training.to_dataloader(
            train=False, batch_size=64, num_workers=0
        )
        predictions = self._model.predict(pred_data)
        preds = predictions[:, 0].numpy().flatten()
        fitted = np.full(len(self._train_y), np.nan)
        fitted[-len(preds):] = preds
        return fitted

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {"max_prediction_length": self.params.get("max_prediction_length", 12)}
        if hasattr(self._model, "hparams"):
            try:
                d["hidden_size"] = int(self._model.hparams.get("hidden_size", 0))
            except Exception:
                pass
        return d


@register_model
class PyTorchForecastingDeepAR(_BasePyTorchForecasting):
    """
    DeepAR from PyTorch Forecasting.

    Autoregressive RNN model for probabilistic forecasting.
    Outputs distribution parameters for each time step.

    Args:
        max_prediction_length: Maximum forecast horizon
        max_encoder_length: Maximum history length
        hidden_size: RNN hidden size
        rnn_layers: Number of RNN layers
        dropout: Dropout rate
        max_epochs: Training epochs
    """

    def __init__(
        self,
        max_prediction_length: int = 12,
        max_encoder_length: int = 24,
        hidden_size: int = 32,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        max_epochs: int = 50,
        **kwargs
    ):
        super().__init__(
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            dropout=dropout,
            max_epochs=max_epochs,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'PyTorchForecastingDeepAR':
        try:
            import lightning.pytorch as pl
            from pytorch_forecasting import DeepAR
            import warnings
        except ImportError:
            raise ImportError(
                "pytorch_forecasting not installed. Install with: "
                "pip install pytorch-forecasting pytorch-lightning"
            )

        y = self._validate_y(y)
        self._train_y = y

        # Use horizon from fit kwargs (passed by runner), override constructor default
        if 'horizon' in kwargs:
            self.params['max_prediction_length'] = kwargs['horizon']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            exog = self._validate_X(X, len(y)) if X is not None else None
            self._training = self._prepare_data(y, X=exog, simple_model=False)
            train_dataloader = self._training.to_dataloader(train=True, batch_size=32, num_workers=0)

            self._model = DeepAR.from_dataset(
                self._training,
                hidden_size=self.params.get('hidden_size', 32),
                rnn_layers=self.params.get('rnn_layers', 2),
                dropout=self.params.get('dropout', 0.1),
            )

            trainer = pl.Trainer(
                max_epochs=get_fast_epochs(self.params.get('max_epochs', 50)),
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(self._model, train_dataloaders=train_dataloader)

        self._is_fitted = True
        return self

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyTorchForecastingDeepAR",
            category=ModelCategory.DEEP_LEARNING,
            library="pytorch-forecasting",
            year=2020,
            paper="Salinas et al. (2020) - DeepAR: Probabilistic Forecasting with Autoregressive RNNs",
            probabilistic=True,
            exogenous=True,
            github_url="https://github.com/jdb78/pytorch-forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("hidden_size", "int", low=16, high=128, default=32),
            ParamSpace("rnn_layers", "int", low=1, high=4, default=2),
            ParamSpace("dropout", "float", low=0.0, high=0.5, default=0.1),
        ]


@register_model
class PyTorchForecastingNBeats(_BasePyTorchForecasting):
    """
    N-BEATS from PyTorch Forecasting.

    Neural Basis Expansion Analysis for Time Series forecasting.
    Interpretable stacked architecture with trend and seasonality.

    Args:
        max_prediction_length: Maximum forecast horizon
        max_encoder_length: Maximum history length
        widths: Hidden layer widths
        stack_types: Types of stacks ('trend', 'seasonality', 'generic')
        max_epochs: Training epochs
    """

    def __init__(
        self,
        max_prediction_length: int = 12,
        max_encoder_length: int = 24,
        widths: List[int] = None,
        max_epochs: int = 50,
        **kwargs
    ):
        if widths is None:
            widths = [32, 32]
        super().__init__(
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            widths=widths,
            max_epochs=max_epochs,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'PyTorchForecastingNBeats':
        try:
            import lightning.pytorch as pl
            from pytorch_forecasting import NBeats
            import warnings
        except ImportError:
            raise ImportError(
                "pytorch_forecasting not installed. Install with: "
                "pip install pytorch-forecasting pytorch-lightning"
            )

        y = self._validate_y(y)
        self._train_y = y

        # Use horizon from fit kwargs (passed by runner), override constructor default
        if 'horizon' in kwargs:
            self.params['max_prediction_length'] = kwargs['horizon']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # NBeats requires simple_model=True (only target as input)
            self._training = self._prepare_data(y, simple_model=True)
            train_dataloader = self._training.to_dataloader(train=True, batch_size=32, num_workers=0)

            self._model = NBeats.from_dataset(
                self._training,
                widths=self.params.get('widths', [32, 32]),
            )

            trainer = pl.Trainer(
                max_epochs=get_fast_epochs(self.params.get('max_epochs', 50)),
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(self._model, train_dataloaders=train_dataloader)

        self._is_fitted = True
        return self

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyTorchForecastingNBeats",
            category=ModelCategory.DEEP_LEARNING,
            library="pytorch-forecasting",
            year=2019,
            paper="Oreshkin et al. (2019) - N-BEATS: Neural basis expansion analysis for time series",
            github_url="https://github.com/jdb78/pytorch-forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("widths", "categorical", choices=[[16, 16], [32, 32], [64, 64], [128, 128]], default=[32, 32]),
        ]


@register_model
class PyTorchForecastingRecurrentNetwork(_BasePyTorchForecasting):
    """
    RecurrentNetwork from PyTorch Forecasting.

    General purpose RNN (LSTM/GRU) for time series forecasting.
    Flexible architecture for various forecasting tasks.

    Args:
        max_prediction_length: Maximum forecast horizon
        max_encoder_length: Maximum history length
        hidden_size: RNN hidden size
        rnn_layers: Number of RNN layers
        cell_type: RNN cell type ('LSTM', 'GRU')
        dropout: Dropout rate
        max_epochs: Training epochs
    """

    def __init__(
        self,
        max_prediction_length: int = 12,
        max_encoder_length: int = 24,
        hidden_size: int = 32,
        rnn_layers: int = 2,
        cell_type: str = 'LSTM',
        dropout: float = 0.1,
        max_epochs: int = 50,
        **kwargs
    ):
        super().__init__(
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            cell_type=cell_type,
            dropout=dropout,
            max_epochs=max_epochs,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'PyTorchForecastingRecurrentNetwork':
        try:
            import lightning.pytorch as pl
            from pytorch_forecasting import RecurrentNetwork
            import warnings
        except ImportError:
            raise ImportError(
                "pytorch_forecasting not installed. Install with: "
                "pip install pytorch-forecasting pytorch-lightning"
            )

        y = self._validate_y(y)
        self._train_y = y

        # Use horizon from fit kwargs (passed by runner), override constructor default
        if 'horizon' in kwargs:
            self.params['max_prediction_length'] = kwargs['horizon']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            exog = self._validate_X(X, len(y)) if X is not None else None
            self._training = self._prepare_data(y, X=exog, simple_model=False)
            train_dataloader = self._training.to_dataloader(train=True, batch_size=32, num_workers=0)

            self._model = RecurrentNetwork.from_dataset(
                self._training,
                hidden_size=self.params.get('hidden_size', 32),
                rnn_layers=self.params.get('rnn_layers', 2),
                cell_type=self.params.get('cell_type', 'LSTM'),
                dropout=self.params.get('dropout', 0.1),
            )

            trainer = pl.Trainer(
                max_epochs=get_fast_epochs(self.params.get('max_epochs', 50)),
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(self._model, train_dataloaders=train_dataloader)

        self._is_fitted = True
        return self

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="PyTorchForecastingRecurrentNetwork",
            category=ModelCategory.DEEP_LEARNING,
            library="pytorch-forecasting",
            year=2020,
            paper="General LSTM/GRU architecture",
            exogenous=True,
            github_url="https://github.com/jdb78/pytorch-forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("hidden_size", "int", low=16, high=128, default=32),
            ParamSpace("rnn_layers", "int", low=1, high=4, default=2),
            ParamSpace("cell_type", "categorical", choices=['LSTM', 'GRU'], default='LSTM'),
            ParamSpace("dropout", "float", low=0.0, high=0.5, default=0.1),
        ]
