"""
forecasters/wrappers/darts_wrapper.py - Darts library forecasters.

Wraps Darts models:
- NBEATS
- NHiTS
- TCN
- TiDE
- Transformer
- TSMixer
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


def _to_darts_series(y: np.ndarray):
    """Convert numpy array to Darts TimeSeries."""
    from darts import TimeSeries
    return TimeSeries.from_values(y)


def _align_fitted(fv, n):
    """Align fitted values array to training data length n."""
    fv = np.asarray(fv, dtype=float).flatten()
    if len(fv) == n:
        return fv
    fitted = np.full(n, np.nan)
    m = min(len(fv), n)
    fitted[n - m:] = fv[-m:]
    return fitted


class _BaseDartsForecaster(Forecaster):
    """Base class for Darts-based forecasters."""

    # Override in subclass to enable exogenous support:
    # 'past' = past_covariates (NBEATS, NHiTS, TCN, Transformer, BlockRNN, regression)
    # 'future' = future_covariates (DartsRNN)
    # None = no covariate support (ETS, Theta, TBATS, FFT, Croston, Kalman)
    _covariate_type: Optional[str] = None

    # Set True in neural subclasses that lack built-in normalization
    # (e.g. Transformer, RNN, BlockRNN, TSMixer).
    # NBEATS/NHiTS/DLinear have internal normalization and don't need this.
    _needs_normalization: bool = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BaseDartsForecaster':
        from darts import TimeSeries

        y = self._validate_y(y)
        self._train_y = y

        # Z-score normalize for neural models without built-in normalization
        self._y_mean = 0.0
        self._y_std = 1.0
        if self._needs_normalization:
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y)) or 1.0
            y_norm = (y - self._y_mean) / self._y_std
        else:
            y_norm = y
        self._train_series = _to_darts_series(y_norm)

        # Handle exogenous covariates
        self._past_cov = None
        self._future_cov_train = None
        self._future_cov_fit = None
        self._has_exog = False

        if X is not None and self._covariate_type is not None:
            exog = self._validate_X(X, len(y))
            self._has_exog = True
            cov_series = TimeSeries.from_values(exog)
            if self._covariate_type == 'past':
                self._past_cov = cov_series
            elif self._covariate_type == 'future':
                self._future_cov_train = exog
                self._future_cov_fit = cov_series

        # Adapt chunk lengths to fit within series length (prevents Darts min-length errors)
        self._adapt_chunk_lengths(len(y))

        self._model = self._create_model()

        # Build fit kwargs for covariates
        fit_kwargs = {}
        if self._past_cov is not None:
            fit_kwargs['past_covariates'] = self._past_cov
        if self._future_cov_fit is not None:
            fit_kwargs['future_covariates'] = self._future_cov_fit

        self._model.fit(self._train_series, **fit_kwargs)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        from darts import TimeSeries

        self._check_fitted()

        pred_kwargs = {}
        if self._past_cov is not None:
            if X is not None:
                from darts import TimeSeries as _TS
                exog_future = self._validate_X(X, horizon)
                full_past = np.vstack([self._past_cov.values(), exog_future])
                pred_kwargs['past_covariates'] = _TS.from_values(full_past)
            else:
                pred_kwargs['past_covariates'] = self._past_cov
        if self._covariate_type == 'future' and self._future_cov_train is not None:
            exog_future = self._validate_X(X, horizon) if X is not None else None
            if exog_future is not None:
                full_exog = np.vstack([self._future_cov_train, exog_future])
            else:
                n_features = self._future_cov_train.shape[1]
                full_exog = np.vstack([self._future_cov_train, np.zeros((horizon, n_features))])
            pred_kwargs['future_covariates'] = TimeSeries.from_values(full_exog)

        pred = self._model.predict(n=horizon, **pred_kwargs)
        values = pred.values().flatten()
        if self._needs_normalization:
            values = values * self._y_std + self._y_mean
        return values

    def _compute_fitted_values(self):
        """Compute fitted values for Darts models.

        Global models: historical_forecasts(retrain=False) — fast rolling eval.
        Local models: extract from underlying model internals — no retraining.
        """
        if self._model is None or not hasattr(self, '_train_series'):
            return None

        # Try historical_forecasts(retrain=False) — works for GlobalForecastingModel
        try:
            hist = self._model.historical_forecasts(
                series=self._train_series,
                forecast_horizon=1,
                stride=1,
                retrain=False,
                verbose=False,
            )
            if isinstance(hist, list):
                if len(hist) == 0:
                    return None
                hist = hist[0]
            vals = hist.values().flatten()
            if self._needs_normalization:
                vals = vals * self._y_std + self._y_mean
            fitted = np.full(len(self._train_y), np.nan)
            fitted[-len(vals):] = vals
            return fitted
        except Exception:
            pass

        # LOCAL model fallback: extract from model internals (no retraining)
        return self._extract_local_fitted_values()

    def _extract_local_fitted_values(self):
        """Extract fitted values from LOCAL Darts model without retraining.

        Tries multiple strategies based on the underlying library.
        Override in subclass for model-specific extraction (e.g. FFT).
        """
        n = len(self._train_y)
        inner = getattr(self._model, 'model', None)

        # S1: statsmodels fittedvalues (ETS, Theta, ARIMA variants)
        if inner is not None:
            for attr in ('fittedvalues', 'fitted_values'):
                if hasattr(inner, attr):
                    try:
                        fv = np.asarray(getattr(inner, attr)).flatten()
                        if len(fv) > 0:
                            return _align_fitted(fv, n)
                    except Exception:
                        pass

        # S2: pmdarima predict_in_sample (AutoARIMA)
        if inner is not None and hasattr(inner, 'predict_in_sample'):
            try:
                fv = np.asarray(inner.predict_in_sample()).flatten()
                if len(fv) > 0:
                    return _align_fitted(fv, n)
            except Exception:
                pass

        # S3: pmdarima arima_res_ fittedvalues (AutoARIMA variant)
        if inner is not None and hasattr(inner, 'arima_res_'):
            try:
                fv = np.asarray(inner.arima_res_.fittedvalues).flatten()
                if len(fv) > 0:
                    return _align_fitted(fv, n)
            except Exception:
                pass

        # S4: TBATS/BATS y_hat
        if inner is not None and hasattr(inner, 'y_hat'):
            try:
                fv = np.asarray(inner.y_hat).flatten()
                if len(fv) > 0:
                    return _align_fitted(fv, n)
            except Exception:
                pass

        # S5: Prophet — predict on training history
        if inner is not None and hasattr(inner, 'history') and hasattr(inner, 'predict'):
            try:
                preds = inner.predict(inner.history)
                fv = preds['yhat'].values.flatten()
                if len(fv) > 0:
                    return _align_fitted(fv, n)
            except Exception:
                pass

        # S6: StatsForecast model dict (newer Darts wraps statsforecast)
        if inner is not None and hasattr(inner, 'model_'):
            try:
                sf = inner.model_
                if isinstance(sf, dict) and 'fitted' in sf:
                    fv = np.asarray(sf['fitted']).flatten()
                    if len(fv) > 0:
                        return _align_fitted(fv, n)
            except Exception:
                pass

        # S7: Constant forecast fallback (Croston, etc.) — use predict(1)
        try:
            pred = self._model.predict(n=1)
            rate = pred.values().flatten()[0]
            if np.isfinite(rate):
                return np.full(n, rate)
        except Exception:
            pass

        return None

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        if hasattr(self._model, "epochs_trained"):
            d["epochs_trained"] = int(self._model.epochs_trained)
        return d

    def _adapt_chunk_lengths(self, n: int):
        """Shrink input/output chunk lengths so icl + ocl <= n.

        Preserves lookback (input) as much as possible; shrinks output first.
        Minimum icl=2, ocl=1.  If the series is too short even for that,
        leave params unchanged and let Darts raise naturally.
        """
        icl = self.params.get('input_chunk_length')
        ocl = self.params.get('output_chunk_length')
        if icl is None or ocl is None:
            return
        if icl + ocl <= n:
            return  # fits already
        # Shrink output_chunk_length first, then input_chunk_length
        new_ocl = max(1, n - icl)
        if new_ocl < 1 or icl > n - 1:
            new_ocl = max(1, min(ocl, n // 2))
            new_icl = max(2, n - new_ocl)
        else:
            new_icl = icl
        if new_icl + new_ocl > n:
            new_icl = max(2, n - new_ocl)
        if new_icl < 2 or new_ocl < 1 or new_icl + new_ocl > n:
            return  # unfixable — let Darts raise
        self.params['input_chunk_length'] = new_icl
        self.params['output_chunk_length'] = new_ocl

    def _create_model(self):
        """Override in subclass."""
        raise NotImplementedError


@register_model
class NBEATS(_BaseDartsForecaster):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series.

    Deep neural network architecture with interpretable basis functions.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        num_stacks: Number of stacks
        num_blocks: Blocks per stack
        num_layers: Layers per block
        layer_widths: Width of each layer
        n_epochs: Training epochs
    """

    _covariate_type = 'past'

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        num_stacks: int = 2,
        num_blocks: int = 1,
        num_layers: int = 4,
        layer_widths: int = 256,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import NBEATSModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return NBEATSModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            num_stacks=self.params.get('num_stacks', 2),
            num_blocks=self.params.get('num_blocks', 1),
            num_layers=self.params.get('num_layers', 4),
            layer_widths=self.params.get('layer_widths', 256),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NBEATS",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2019,
            paper="Oreshkin et al. (2019)",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("input_chunk_length", "int", low=12, high=48, default=24),
            ParamSpace("num_stacks", "int", low=1, high=4, default=2),
            ParamSpace("num_blocks", "int", low=1, high=4, default=1),
            ParamSpace("layer_widths", "int", low=64, high=512, default=256),
        ]


@register_model
class NHiTS(_BaseDartsForecaster):
    """
    N-HiTS: Neural Hierarchical Interpolation for Time Series.

    Efficient architecture with hierarchical interpolation for long horizons.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        num_stacks: Number of stacks
        num_blocks: Blocks per stack
        num_layers: Layers per block
        layer_widths: Width of each layer
        n_epochs: Training epochs
    """

    _covariate_type = 'past'

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        num_stacks: int = 2,
        num_blocks: int = 1,
        num_layers: int = 2,
        layer_widths: int = 256,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import NHiTSModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return NHiTSModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            num_stacks=self.params.get('num_stacks', 2),
            num_blocks=self.params.get('num_blocks', 1),
            num_layers=self.params.get('num_layers', 2),
            layer_widths=self.params.get('layer_widths', 256),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NHiTS",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2022,
            paper="Challu et al. (2022)",
            exogenous=True,
        )


@register_model
class TCN(_BaseDartsForecaster):
    """
    Temporal Convolutional Network.

    Causal dilated convolutions for sequence modeling.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        kernel_size: Convolution kernel size
        num_filters: Number of filters
        n_epochs: Training epochs
    """

    _covariate_type = 'past'

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        kernel_size: int = 3,
        num_filters: int = 3,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            kernel_size=kernel_size,
            num_filters=num_filters,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import TCNModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return TCNModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            kernel_size=self.params.get('kernel_size', 3),
            num_filters=self.params.get('num_filters', 3),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TCN",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2018,
            paper="Bai et al. (2018)",
            exogenous=True,
        )


@register_model
class TiDE(_BaseDartsForecaster):
    """
    TiDE: Time-series Dense Encoder.

    Simple MLP-based architecture competitive with transformers.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        hidden_size: Hidden layer size
        n_epochs: Training epochs
    """

    _covariate_type = 'past'

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        hidden_size: int = 256,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import TiDEModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return TiDEModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            hidden_size=self.params.get('hidden_size', 256),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TiDE",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2023,
            paper="Das et al. (2023)",
            exogenous=True,
        )


@register_model
class DartsTransformer(_BaseDartsForecaster):
    """
    Transformer model for time series.

    Attention-based architecture for sequence-to-sequence forecasting.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        d_model: Model dimension
        nhead: Number of attention heads
        n_epochs: Training epochs
    """

    _covariate_type = 'past'
    _needs_normalization = True

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            d_model=d_model,
            nhead=nhead,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import TransformerModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return TransformerModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            d_model=self.params.get('d_model', 64),
            nhead=self.params.get('nhead', 4),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsTransformer",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2017,
            paper="Vaswani et al. (2017)",
            exogenous=True,
        )


@register_model
class DLinear(_BaseDartsForecaster):
    """
    DLinear: Are Transformers Effective for Time Series Forecasting?

    Simple linear model that decomposes series into trend and seasonal.
    Often outperforms complex transformer models on benchmark datasets.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        kernel_size: Moving average kernel size for decomposition
        n_epochs: Training epochs
    """

    _covariate_type = 'past'

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        kernel_size: int = 25,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            kernel_size=kernel_size,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import DLinearModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return DLinearModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            kernel_size=self.params.get('kernel_size', 25),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DLinear",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2023,
            paper="Zeng et al. (2023) - Are Transformers Effective for Time Series Forecasting?",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("input_chunk_length", "int", low=12, high=96, default=24),
            ParamSpace("output_chunk_length", "int", low=6, high=48, default=12),
            ParamSpace("kernel_size", "int", low=11, high=51, default=25),
        ]


@register_model
class NLinear(_BaseDartsForecaster):
    """
    NLinear: Normalized Linear model.

    Subtracts the last value before linear prediction, adds it back after.
    Handles distribution shift in time series.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        n_epochs: Training epochs
    """

    _covariate_type = 'past'

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import NLinearModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return NLinearModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="NLinear",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2023,
            paper="Zeng et al. (2023) - Are Transformers Effective for Time Series Forecasting?",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("input_chunk_length", "int", low=12, high=96, default=24),
            ParamSpace("output_chunk_length", "int", low=6, high=48, default=12),
        ]


@register_model
class DartsRNN(_BaseDartsForecaster):
    """
    RNN-based forecaster (vanilla RNN, LSTM, or GRU).

    Recurrent neural network for sequence-to-sequence forecasting.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        model: RNN type ('RNN', 'LSTM', 'GRU')
        hidden_dim: Hidden layer dimension
        n_rnn_layers: Number of RNN layers
        n_epochs: Training epochs
    """

    _covariate_type = 'future'
    _needs_normalization = True

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        model: str = 'LSTM',
        hidden_dim: int = 64,
        n_rnn_layers: int = 2,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model=model,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import RNNModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return RNNModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            model=self.params.get('model', 'LSTM'),
            hidden_dim=self.params.get('hidden_dim', 64),
            n_rnn_layers=self.params.get('n_rnn_layers', 2),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsRNN",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2014,
            paper="Cho et al. (2014) - GRU / Hochreiter & Schmidhuber (1997) - LSTM",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("input_chunk_length", "int", low=12, high=96, default=24),
            ParamSpace("model", "categorical", choices=['RNN', 'LSTM', 'GRU'], default='LSTM'),
            ParamSpace("hidden_dim", "int", low=32, high=256, default=64),
            ParamSpace("n_rnn_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class BlockRNN(_BaseDartsForecaster):
    """
    Block RNN: RNN with block input processing.

    Processes input in blocks for efficiency with long sequences.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        model: RNN type ('RNN', 'LSTM', 'GRU')
        hidden_dim: Hidden layer dimension
        n_rnn_layers: Number of RNN layers
        n_epochs: Training epochs
    """

    _covariate_type = 'past'
    _needs_normalization = True

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        model: str = 'LSTM',
        hidden_dim: int = 64,
        n_rnn_layers: int = 2,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model=model,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import BlockRNNModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return BlockRNNModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            model=self.params.get('model', 'LSTM'),
            hidden_dim=self.params.get('hidden_dim', 64),
            n_rnn_layers=self.params.get('n_rnn_layers', 2),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="BlockRNN",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2014,
            paper="Hochreiter & Schmidhuber (1997) - LSTM",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("input_chunk_length", "int", low=12, high=96, default=24),
            ParamSpace("model", "categorical", choices=['RNN', 'LSTM', 'GRU'], default='LSTM'),
            ParamSpace("hidden_dim", "int", low=32, high=256, default=64),
            ParamSpace("n_rnn_layers", "int", low=1, high=4, default=2),
        ]


@register_model
class DartsProphet(_BaseDartsForecaster):
    """
    Prophet model via Darts wrapper.

    Facebook's Prophet model for forecasting with trend changes and seasonality.
    Uses Darts' unified interface.

    Args:
        changepoint_prior_scale: Flexibility of trend changes
        seasonality_prior_scale: Flexibility of seasonality
        seasonality_mode: 'additive' or 'multiplicative'
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = 'additive',
        **kwargs
    ):
        super().__init__(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'DartsProphet':
        """Prophet requires datetime index - create synthetic one if needed."""
        from darts import TimeSeries
        import pandas as pd

        y = self._validate_y(y)
        self._train_y = y

        # Initialize covariate attributes (base class sets these, but we bypass it)
        self._past_cov = None
        self._future_cov_train = None
        self._future_cov_fit = None
        self._has_exog = False

        # Prophet needs datetime index
        if freq is None:
            freq = 'D'  # Default daily
        dates = safe_date_range(start='2020-01-01', periods=len(y), freq=freq)
        self._train_series = TimeSeries.from_times_and_values(dates, y)
        self._freq = freq

        self._model = self._create_model()
        self._model.fit(self._train_series)

        self._is_fitted = True
        return self

    def _create_model(self):
        from darts.models import Prophet

        return Prophet(
            changepoint_prior_scale=self.params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=self.params.get('seasonality_prior_scale', 10.0),
            seasonality_mode=self.params.get('seasonality_mode', 'additive'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsProphet",
            category=ModelCategory.CLASSICAL,
            library="darts",
            year=2017,
            paper="Taylor & Letham (2017) - Forecasting at Scale",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("changepoint_prior_scale", "log_float", low=0.001, high=0.5, default=0.05),
            ParamSpace("seasonality_prior_scale", "log_float", low=0.01, high=10.0, default=10.0),
            ParamSpace("seasonality_mode", "categorical", choices=['additive', 'multiplicative'], default='additive'),
        ]


@register_model
class DartsTSMixer(_BaseDartsForecaster):
    """
    TSMixer: MLP-Mixer for Time Series Forecasting.

    Applies MLP-Mixer architecture (token mixing + channel mixing) to time series.
    Simple and efficient alternative to transformers.

    Args:
        input_chunk_length: Lookback window
        output_chunk_length: Forecast horizon
        hidden_size: Hidden layer size
        ff_size: Feed-forward layer size
        n_epochs: Training epochs
    """

    _covariate_type = 'past'
    _needs_normalization = True

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        hidden_size: int = 64,
        ff_size: int = 128,
        n_epochs: int = 100,
        **kwargs
    ):
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size,
            ff_size=ff_size,
            n_epochs=n_epochs,
            **kwargs
        )

    def _create_model(self):
        from darts.models import TSMixerModel
        import pytorch_lightning as pl
        pl.seed_everything(42)

        return TSMixerModel(
            input_chunk_length=self.params.get('input_chunk_length', 24),
            output_chunk_length=self.params.get('output_chunk_length', 12),
            hidden_size=self.params.get('hidden_size', 64),
            ff_size=self.params.get('ff_size', 128),
            n_epochs=self.params.get('n_epochs', 100),
            random_state=42,
            pl_trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsTSMixer",
            category=ModelCategory.DEEP_LEARNING,
            library="darts",
            year=2023,
            paper="Chen et al. (2023) - TSMixer: An All-MLP Architecture for Time Series Forecasting",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("input_chunk_length", "int", low=12, high=96, default=24),
            ParamSpace("output_chunk_length", "int", low=6, high=48, default=12),
            ParamSpace("hidden_size", "int", low=32, high=256, default=64),
            ParamSpace("ff_size", "int", low=64, high=512, default=128),
        ]


@register_model
class DartsCroston(_BaseDartsForecaster):
    """
    Croston's method for intermittent demand forecasting.

    Separately forecasts demand size and inter-arrival times.
    Designed for sporadic/intermittent time series.

    WARNING: Designed for intermittent demand (many zeros). Will produce
    flat constant forecasts. Not suitable for continuous or seasonal data.

    Args:
        version: Croston variant ('classic', 'optimized', 'sba')
    """

    def __init__(
        self,
        version: str = 'classic',
        **kwargs
    ):
        super().__init__(
            version=version,
            **kwargs
        )

    def _create_model(self):
        from darts.models import Croston

        return Croston(
            version=self.params.get('version', 'classic'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsCroston",
            category=ModelCategory.INTERMITTENT,
            library="darts",
            year=1972,
            paper="Croston (1972) - Forecasting and Stock Control for Intermittent Demands",
            notes="For sparse demand (many zeros). Outputs flat forecasts - not for continuous data.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("version", "categorical", choices=['classic', 'optimized', 'sba'], default='classic'),
        ]


@register_model
class DartsFFT(_BaseDartsForecaster):
    """
    FFT-based forecasting.

    Uses Fast Fourier Transform to extract frequency components
    and extrapolate for forecasting.

    Args:
        nr_freqs_to_keep: Number of frequency components to keep
        trend: Trend type ('poly', 'exp', None)
        trend_poly_degree: Polynomial degree if trend='poly'
    """

    def __init__(
        self,
        nr_freqs_to_keep: Optional[int] = None,
        trend: Optional[str] = 'poly',
        trend_poly_degree: int = 1,
        **kwargs
    ):
        super().__init__(
            nr_freqs_to_keep=nr_freqs_to_keep,
            trend=trend,
            trend_poly_degree=trend_poly_degree,
            **kwargs
        )

    def _create_model(self):
        from darts.models import FFT

        return FFT(
            nr_freqs_to_keep=self.params.get('nr_freqs_to_keep'),
            trend=self.params.get('trend', 'poly'),
            trend_poly_degree=self.params.get('trend_poly_degree', 1),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsFFT",
            category=ModelCategory.CLASSICAL,
            library="darts",
            year=1965,
            paper="Cooley & Tukey (1965) - An Algorithm for the Machine Calculation of Complex Fourier Series",
        )

    def _extract_local_fitted_values(self):
        """FFT fitted = detrend + IFFT of top-N frequencies + re-trend."""
        try:
            y = self._train_y
            n = len(y)
            nr_freqs = self.params.get('nr_freqs_to_keep')
            if nr_freqs is None or nr_freqs >= n // 2:
                return y.copy()  # All freqs kept → perfect reconstruction

            # Compute and remove trend (matches Darts FFT logic)
            x = np.arange(n, dtype=float)
            trend_type = self.params.get('trend', 'poly')
            if trend_type == 'poly':
                degree = self.params.get('trend_poly_degree', 1)
                coeffs = np.polyfit(x, y, degree)
                trend = np.polyval(coeffs, x)
            elif trend_type == 'exp':
                log_y = np.log(np.maximum(y, 1e-10))
                coeffs = np.polyfit(x, log_y, 1)
                trend = np.exp(np.polyval(coeffs, x))
            else:
                trend = np.zeros(n)

            detrended = y - trend

            # FFT, keep top-N frequencies, IFFT
            fft_vals = np.fft.fft(detrended)
            magnitudes = np.abs(fft_vals)
            indices = np.argsort(magnitudes[1:])[::-1] + 1  # Skip DC
            keep = {0}  # Always keep DC
            for idx in indices[:nr_freqs]:
                keep.add(idx)
                if n - idx != idx:
                    keep.add(n - idx)  # Symmetric pair

            filtered = np.zeros(n, dtype=complex)
            for idx in keep:
                if idx < n:
                    filtered[idx] = fft_vals[idx]

            return trend + np.real(np.fft.ifft(filtered))
        except Exception:
            return super()._extract_local_fitted_values()

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("nr_freqs_to_keep", "int", low=1, high=20, default=10),
            ParamSpace("trend", "categorical", choices=['poly', 'exp', None], default='poly'),
            ParamSpace("trend_poly_degree", "int", low=0, high=3, default=1),
        ]


@register_model
class DartsETS(_BaseDartsForecaster):
    """
    ExponentialSmoothing (ETS) via Darts.

    Error-Trend-Seasonality state space models.
    Automatically selects best ETS configuration.

    Args:
        seasonal_periods: Number of periods in a season
        seasonal: Seasonal type ('add', 'mul', None)
        trend: Trend type ('add', 'mul', None)
    """

    def __init__(
        self,
        seasonal_periods: Optional[int] = None,
        seasonal: Optional[str] = 'add',
        trend: Optional[str] = 'add',
        **kwargs
    ):
        super().__init__(
            seasonal_periods=seasonal_periods,
            seasonal=seasonal,
            trend=trend,
            **kwargs
        )

    def _create_model(self):
        from darts.models import ExponentialSmoothing
        from darts.utils.utils import ModelMode, SeasonalityMode

        # Map string params to enums (darts 0.28+ uses enums)
        trend_map = {'add': ModelMode.ADDITIVE, 'mul': ModelMode.MULTIPLICATIVE, None: ModelMode.NONE}
        seasonal_map = {'add': SeasonalityMode.ADDITIVE, 'mul': SeasonalityMode.MULTIPLICATIVE, None: SeasonalityMode.NONE}

        trend_val = self.params.get('trend', 'add')
        seasonal_val = self.params.get('seasonal', 'add')

        return ExponentialSmoothing(
            seasonal_periods=self.params.get('seasonal_periods'),
            seasonal=seasonal_map.get(seasonal_val, seasonal_val),
            trend=trend_map.get(trend_val, trend_val),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsETS",
            category=ModelCategory.CLASSICAL,
            library="darts",
            year=2002,
            paper="Hyndman et al. (2002) - A State Space Framework for Automatic Forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("seasonal_periods", "int", low=2, high=52, default=12),
            ParamSpace("seasonal", "categorical", choices=['add', 'mul', None], default='add'),
            ParamSpace("trend", "categorical", choices=['add', 'mul', None], default='add'),
        ]


@register_model
class DartsTheta(_BaseDartsForecaster):
    """
    Theta method for time series forecasting.

    Decomposes series and applies different smoothing to components.
    Simple but robust method that won M3 competition.

    Args:
        theta: Theta parameter (0=linear regression, 2=SES, etc.)
        seasonality_period: Period of seasonality (None for auto-detection)
    """

    def __init__(
        self,
        theta: float = 2.0,
        seasonality_period: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            theta=theta,
            seasonality_period=seasonality_period,
            **kwargs
        )

    def _create_model(self):
        from darts.models import Theta
        from darts.utils.utils import SeasonalityMode

        return Theta(
            theta=self.params.get('theta', 2.0),
            seasonality_period=self.params.get('seasonality_period'),
            season_mode=SeasonalityMode.ADDITIVE,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsTheta",
            category=ModelCategory.CLASSICAL,
            library="darts",
            year=2000,
            paper="Assimakopoulos & Nikolopoulos (2000) - The Theta Model",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("theta", "float", low=0.0, high=4.0, default=2.0),
            ParamSpace("seasonality_period", "int", low=2, high=52, default=12),
        ]


@register_model
class DartsTBATS(_BaseDartsForecaster):
    """
    TBATS: Exponential smoothing with Box-Cox, ARMA errors, Trend, Seasonality.

    Handles complex seasonal patterns including multiple seasonalities.

    Args:
        use_box_cox: Whether to use Box-Cox transformation
        use_trend: Whether to include trend
        use_damped_trend: Whether to dampen the trend
        use_arma_errors: Whether to model residuals as ARMA
        seasonal_periods: List of seasonal periods (None for auto-detection)
    """

    def __init__(
        self,
        use_box_cox: Optional[bool] = None,
        use_trend: Optional[bool] = None,
        use_damped_trend: Optional[bool] = None,
        use_arma_errors: Optional[bool] = None,
        seasonal_periods: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(
            use_box_cox=use_box_cox,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            use_arma_errors=use_arma_errors,
            seasonal_periods=seasonal_periods,
            **kwargs
        )

    def _create_model(self):
        from darts.models import TBATS

        # Note: darts 0.40+ requires season_length as required positional arg
        # Use `or` to handle both None and missing keys
        season_length = self.params.get('seasonal_periods') or [12]
        if isinstance(season_length, list):
            season_length = season_length[0] if season_length else 12

        return TBATS(season_length=season_length)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsTBATS",
            category=ModelCategory.CLASSICAL,
            library="darts",
            year=2011,
            paper="De Livera et al. (2011) - Forecasting Time Series with Complex Seasonal Patterns",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("use_box_cox", "categorical", choices=[True, False, None], default=None),
            ParamSpace("use_trend", "categorical", choices=[True, False, None], default=None),
            ParamSpace("use_damped_trend", "categorical", choices=[True, False, None], default=None),
            ParamSpace("use_arma_errors", "categorical", choices=[True, False, None], default=None),
        ]


@register_model
class DartsAutoARIMA(_BaseDartsForecaster):
    """
    AutoARIMA: Automatic ARIMA model selection.

    Automatically selects best ARIMA(p,d,q)(P,D,Q)m model via grid search.

    Args:
        start_p: Starting value of p for grid search
        max_p: Maximum value of p
        start_q: Starting value of q
        max_q: Maximum value of q
        seasonal: Whether to fit seasonal ARIMA
        m: Period for seasonal differencing
    """

    def __init__(
        self,
        start_p: int = 1,
        max_p: int = 5,
        start_q: int = 1,
        max_q: int = 5,
        seasonal: bool = True,
        m: int = 12,
        **kwargs
    ):
        super().__init__(
            start_p=start_p,
            max_p=max_p,
            start_q=start_q,
            max_q=max_q,
            seasonal=seasonal,
            m=m,
            **kwargs
        )

    def _create_model(self):
        from darts.models import AutoARIMA

        # Note: darts 0.28+ doesn't accept 'm' directly - it's inferred from data
        return AutoARIMA(
            start_p=self.params.get('start_p', 1),
            max_p=self.params.get('max_p', 5),
            start_q=self.params.get('start_q', 1),
            max_q=self.params.get('max_q', 5),
            seasonal=self.params.get('seasonal', True),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsAutoARIMA",
            category=ModelCategory.CLASSICAL,
            library="darts",
            year=2008,
            paper="Hyndman & Khandakar (2008) - Automatic Time Series Forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("max_p", "int", low=2, high=10, default=5),
            ParamSpace("max_q", "int", low=2, high=10, default=5),
            ParamSpace("seasonal", "categorical", choices=[True, False], default=True),
            ParamSpace("m", "int", low=2, high=52, default=12),
        ]


@register_model
class DartsVARIMA(_BaseDartsForecaster):
    """
    VARIMA: Vector ARIMA for multivariate time series.

    Extends ARIMA to multiple time series with cross-correlations.
    REQUIRES multivariate input (2D array with shape [n_samples, n_series]).

    Args:
        p: Autoregressive order
        d: Differencing order
        q: Moving average order
        trend: Trend type ('n', 'c', 't', 'ct')
    """

    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 0,
        trend: str = 'c',
        **kwargs
    ):
        super().__init__(
            p=p,
            d=d,
            q=q,
            trend=trend,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'DartsVARIMA':
        """Fit VARIMA model. Requires multivariate input."""
        from darts import TimeSeries

        # Convert to numpy without flattening (skip base _validate_y which flattens)
        if isinstance(y, pd.DataFrame):
            y = y.values
        elif isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y)

        # VARIMA requires multivariate data
        if y.ndim == 1:
            raise ValueError(
                "DartsVARIMA requires multivariate input (2D array). "
                "Got 1D array. Use ARIMA for univariate series."
            )

        self._train_y = y

        # Create multivariate TimeSeries (shape: [n_samples, n_series])
        self._train_series = TimeSeries.from_values(y)

        self._model = self._create_model()
        self._model.fit(self._train_series)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        """Predict with VARIMA. Returns multivariate forecast."""
        self._check_fitted()
        pred = self._model.predict(n=horizon)
        # Return full multivariate array [horizon, n_series]
        return pred.values()

    def _create_model(self):
        from darts.models import VARIMA

        return VARIMA(
            p=self.params.get('p', 1),
            d=self.params.get('d', 0),
            q=self.params.get('q', 0),
            trend=self.params.get('trend', 'c'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsVARIMA",
            category=ModelCategory.CLASSICAL,
            library="darts",
            year=1951,
            paper="Tiao & Box (1981) - Modeling Multiple Time Series",
            notes="Multivariate-only. Requires 2D input [n_samples, n_series].",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("p", "int", low=0, high=5, default=1),
            ParamSpace("d", "int", low=0, high=2, default=0),
            ParamSpace("q", "int", low=0, high=5, default=0),
            ParamSpace("trend", "categorical", choices=['n', 'c', 't', 'ct'], default='c'),
        ]


@register_model
class DartsKalmanForecaster(_BaseDartsForecaster):
    """
    Kalman Filter-based forecasting.

    State space model with Kalman filtering for forecasting.

    Args:
        dim_x: Dimension of state vector
    """

    def __init__(
        self,
        dim_x: int = 1,
        **kwargs
    ):
        super().__init__(
            dim_x=dim_x,
            **kwargs
        )

    def _create_model(self):
        from darts.models import KalmanForecaster

        return KalmanForecaster(
            dim_x=self.params.get('dim_x', 1),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsKalmanForecaster",
            category=ModelCategory.CLASSICAL,
            library="darts",
            year=1960,
            paper="Kalman (1960) - A New Approach to Linear Filtering",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("dim_x", "int", low=1, high=10, default=1),
        ]


@register_model
class DartsLinearRegression(_BaseDartsForecaster):
    """
    Linear Regression for time series.

    Uses lagged values as features for linear regression.

    Args:
        lags: Number of lagged values to use
        output_chunk_length: Forecast horizon
    """

    _covariate_type = 'past'

    def __init__(
        self,
        lags: int = 12,
        output_chunk_length: int = 1,
        **kwargs
    ):
        super().__init__(
            lags=lags,
            output_chunk_length=output_chunk_length,
            **kwargs
        )

    def _create_model(self):
        from darts.models import LinearRegressionModel

        lags = self.params.get('lags', 12)
        kwargs = {
            'lags': lags,
            'output_chunk_length': self.params.get('output_chunk_length', 1),
        }
        if self._has_exog:
            kwargs['lags_past_covariates'] = lags
        return LinearRegressionModel(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsLinearRegression",
            category=ModelCategory.CLASSICAL,
            library="darts",
            year=1805,
            paper="Legendre (1805) - Method of Least Squares",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lags", "int", low=1, high=48, default=12),
            ParamSpace("output_chunk_length", "int", low=1, high=24, default=1),
        ]


@register_model
class DartsRandomForest(_BaseDartsForecaster):
    """
    Random Forest for time series regression.

    Ensemble of decision trees using lagged values as features.

    Args:
        lags: Number of lagged values to use
        output_chunk_length: Forecast horizon
        n_estimators: Number of trees
        max_depth: Maximum tree depth
    """

    _covariate_type = 'past'

    def __init__(
        self,
        lags: int = 12,
        output_chunk_length: int = 1,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            lags=lags,
            output_chunk_length=output_chunk_length,
            n_estimators=n_estimators,
            max_depth=max_depth,
            **kwargs
        )

    def _create_model(self):
        from darts.models import RandomForest

        lags = self.params.get('lags', 12)
        kwargs = {
            'lags': lags,
            'output_chunk_length': self.params.get('output_chunk_length', 1),
            'n_estimators': self.params.get('n_estimators', 100),
            'max_depth': self.params.get('max_depth'),
        }
        if self._has_exog:
            kwargs['lags_past_covariates'] = lags
        return RandomForest(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsRandomForest",
            category=ModelCategory.ML,
            library="darts",
            year=2001,
            paper="Breiman (2001) - Random Forests",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lags", "int", low=1, high=48, default=12),
            ParamSpace("n_estimators", "int", low=50, high=500, default=100),
            ParamSpace("max_depth", "int", low=3, high=20, default=10),
        ]


@register_model
class DartsLightGBM(_BaseDartsForecaster):
    """
    LightGBM for time series regression.

    Gradient boosting using lagged values as features.

    Args:
        lags: Number of lagged values to use
        output_chunk_length: Forecast horizon
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate
        max_depth: Maximum tree depth
    """

    def __init__(
        self,
        lags: int = 12,
        output_chunk_length: int = 1,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        **kwargs
    ):
        super().__init__(
            lags=lags,
            output_chunk_length=output_chunk_length,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            **kwargs
        )

    _covariate_type = 'past'

    def _create_model(self):
        from darts.models import LightGBMModel

        lags = self.params.get('lags', 12)
        kwargs = {
            'lags': lags,
            'output_chunk_length': self.params.get('output_chunk_length', 1),
            'n_estimators': self.params.get('n_estimators', 100),
            'learning_rate': self.params.get('learning_rate', 0.1),
            'max_depth': self.params.get('max_depth', -1),
            'verbose': -1,
        }
        if self._has_exog:
            kwargs['lags_past_covariates'] = lags
        return LightGBMModel(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsLightGBM",
            category=ModelCategory.ML,
            library="darts",
            year=2017,
            paper="Ke et al. (2017) - LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lags", "int", low=1, high=48, default=12),
            ParamSpace("n_estimators", "int", low=50, high=500, default=100),
            ParamSpace("learning_rate", "log_float", low=0.01, high=0.3, default=0.1),
            ParamSpace("max_depth", "int", low=3, high=15, default=-1),
        ]


@register_model
class DartsXGBoost(_BaseDartsForecaster):
    """
    XGBoost for time series regression.

    Extreme gradient boosting using lagged values as features.

    Args:
        lags: Number of lagged values to use
        output_chunk_length: Forecast horizon
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate
        max_depth: Maximum tree depth
    """

    def __init__(
        self,
        lags: int = 12,
        output_chunk_length: int = 1,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        **kwargs
    ):
        super().__init__(
            lags=lags,
            output_chunk_length=output_chunk_length,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            **kwargs
        )

    _covariate_type = 'past'

    def _create_model(self):
        from darts.models import XGBModel

        lags = self.params.get('lags', 12)
        kwargs = {
            'lags': lags,
            'output_chunk_length': self.params.get('output_chunk_length', 1),
            'n_estimators': self.params.get('n_estimators', 100),
            'learning_rate': self.params.get('learning_rate', 0.1),
            'max_depth': self.params.get('max_depth', 6),
            'verbosity': 0,
        }
        if self._has_exog:
            kwargs['lags_past_covariates'] = lags
        return XGBModel(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsXGBoost",
            category=ModelCategory.ML,
            library="darts",
            year=2016,
            paper="Chen & Guestrin (2016) - XGBoost: A Scalable Tree Boosting System",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lags", "int", low=1, high=48, default=12),
            ParamSpace("n_estimators", "int", low=50, high=500, default=100),
            ParamSpace("learning_rate", "log_float", low=0.01, high=0.3, default=0.1),
            ParamSpace("max_depth", "int", low=3, high=15, default=6),
        ]


@register_model
class DartsCatBoost(_BaseDartsForecaster):
    """
    CatBoost for time series regression.

    Gradient boosting with ordered boosting and categorical feature support.

    Args:
        lags: Number of lagged values to use
        output_chunk_length: Forecast horizon
        iterations: Number of boosting iterations
        learning_rate: Learning rate
        depth: Tree depth
    """

    def __init__(
        self,
        lags: int = 12,
        output_chunk_length: int = 1,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 6,
        **kwargs
    ):
        super().__init__(
            lags=lags,
            output_chunk_length=output_chunk_length,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            **kwargs
        )

    _covariate_type = 'past'

    def _create_model(self):
        from darts.models import CatBoostModel

        lags = self.params.get('lags', 12)
        kwargs = {
            'lags': lags,
            'output_chunk_length': self.params.get('output_chunk_length', 1),
            'iterations': self.params.get('iterations', 100),
            'learning_rate': self.params.get('learning_rate', 0.1),
            'depth': self.params.get('depth', 6),
            'verbose': False,
        }
        if self._has_exog:
            kwargs['lags_past_covariates'] = lags
        return CatBoostModel(**kwargs)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="DartsCatBoost",
            category=ModelCategory.ML,
            library="darts",
            year=2018,
            paper="Prokhorenkova et al. (2018) - CatBoost: Unbiased Boosting with Categorical Features",
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("lags", "int", low=1, high=48, default=12),
            ParamSpace("iterations", "int", low=50, high=500, default=100),
            ParamSpace("learning_rate", "log_float", low=0.01, high=0.3, default=0.1),
            ParamSpace("depth", "int", low=3, high=12, default=6),
        ]
