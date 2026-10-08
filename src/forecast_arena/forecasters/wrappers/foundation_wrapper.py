# Modified for forecast-arena: extracted and adapted from AION benchmark utilities, October 2026.
"""
forecasters/wrappers/foundation_wrapper.py - Foundation model wrappers.

Wraps zero-shot time series foundation models:
- Lag-Llama (ServiceNow): Probabilistic LLM-based forecasting
- Moirai (Salesforce): Universal time series foundation model
- TimesFM (Google): Time series foundation model
- MOMENT (CMU): Masked time series pre-training

All models are zero-shot: no training required, just context conditioning.

Performance notes:
- All modules cached at class level (loaded once, reused for all series)
- Batch inference via predict_batch() for multi-series benchmarks
- GluonTS ListDataset with batch_size=32 for Moirai/LagLlama families
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


class _FoundationModuleCache:
    """Class-level module cache for foundation models.

    Caches the heavy from_pretrained() call. Keyed by model_id.
    Shared across all instances. Saves 2-5s per series.
    """
    _cache: Dict[str, object] = {}

    @classmethod
    def get(cls, model_id: str, loader_fn) -> object:
        """Get or create a cached module.

        Args:
            model_id: HuggingFace model ID (e.g., 'Salesforce/moirai-1.1-R-base')
            loader_fn: Callable that loads the module (e.g., MoiraiModule.from_pretrained)
        """
        if model_id not in cls._cache:
            cls._cache[model_id] = loader_fn(model_id)
        return cls._cache[model_id]

    @classmethod
    def clear(cls):
        """Clear cache (useful for freeing GPU memory)."""
        cls._cache.clear()


@register_model
class LagLlama(Forecaster):
    """
    Lag-Llama: Probabilistic time series foundation model.

    Zero-shot forecasting using autoregressive transformer with lagged features.
    Outputs probabilistic forecasts via sampling.

    Args:
        context_length: Number of past observations to use (default: 32)
        prediction_length: Forecast horizon (used internally)
        num_samples: Number of samples for probabilistic forecasts
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        context_length: int = 32,
        num_samples: int = 100,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            context_length=context_length,
            num_samples=num_samples,
            device=device,
            **kwargs
        )
        self._ckpt = None

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'LagLlama':
        """Lag-Llama is zero-shot - just stores context."""
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

    _ckpt_cache: Dict[str, tuple] = {}

    @classmethod
    def _get_ckpt_and_args(cls, torch_module=None):
        """Cache checkpoint download + architecture param extraction."""
        cache_key = "lag-llama"
        if cache_key not in cls._ckpt_cache:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id="time-series-foundation-models/Lag-Llama",
                filename="lag-llama.ckpt"
            )
            if torch_module is None:
                import torch as torch_module
            ckpt = torch_module.load(ckpt_path, map_location="cpu", weights_only=False)
            estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
            cls._ckpt_cache[cache_key] = (ckpt_path, estimator_args)
        return cls._ckpt_cache[cache_key]

    @classmethod
    def predict_batch(
        cls,
        contexts: List[np.ndarray],
        horizon: int,
        context_length: int = 32,
        num_samples: int = 100,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Batch inference for LagLlama via GluonTS PandasDataset.

        Returns list of 1D forecast arrays (median of samples).
        """
        if not contexts:
            return []

        import torch
        from lag_llama.gluon.estimator import LagLlamaEstimator
        from gluonts.dataset.pandas import PandasDataset

        ckpt_path, estimator_args = cls._get_ckpt_and_args(torch)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=horizon,
            context_length=context_length,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args.get("scaling", "mean"),
            time_feat=estimator_args.get("time_feat", True),
            lags_seq=["Q", "M", "W", "D", "H", "T", "S"],
            num_parallel_samples=num_samples,
            device=torch.device(device),
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        # Build multi-series dataset
        all_dfs = []
        for i, ctx in enumerate(contexts):
            c = ctx[-context_length:].astype(np.float32) if len(ctx) > context_length else ctx.astype(np.float32)
            df = pd.DataFrame({
                'item_id': [f's{i}'] * len(c),
                'target': c,
                'timestamp': safe_date_range(start='2000-01-01', periods=len(c), freq='D'),
            })
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        dataset = PandasDataset.from_long_dataframe(
            combined, item_id='item_id', target='target', timestamp='timestamp',
        )

        forecasts = list(predictor.predict(dataset, num_samples=num_samples))
        return [np.median(f.samples, axis=0) for f in forecasts]

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        try:
            import torch
            from huggingface_hub import hf_hub_download
            from gluonts.dataset.pandas import PandasDataset
            from gluonts.torch.model.predictor import PyTorchPredictor
        except ImportError:
            raise ImportError(
                "Lag-Llama dependencies not installed. Install with:\n"
                "pip install lag-llama gluonts torch huggingface_hub"
            )

        # Load model checkpoint
        try:
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError:
            raise ImportError(
                "lag-llama not installed. Install with:\n"
                "pip install git+https://github.com/time-series-foundation-models/lag-llama.git"
            )

        device = self._get_device()
        context_length = self.params.get('context_length', 32)
        num_samples = self.params.get('num_samples', 100)

        # Cached: download checkpoint + extract architecture params once
        ckpt_path, estimator_args = self._get_ckpt_and_args(torch)

        # GluonTS 0.14 uses old pandas freq strings
        _FREQ_MAP = {"ME": "M", "QE": "Q", "YE": "Y", "h": "H", "min": "T", "s": "S"}
        gluon_freq = _FREQ_MAP.get(self._freq, self._freq)

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=horizon,
            context_length=context_length,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args.get("scaling", "mean"),
            time_feat=estimator_args.get("time_feat", True),
            lags_seq=["Q", "M", "W", "D", "H", "T", "S"],
            num_parallel_samples=num_samples,
            device=torch.device(device),
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        # Create dataset in GluonTS format — use float32 to match model weights
        y_f32 = self._train_y.astype(np.float32)
        df = pd.DataFrame({
            'item_id': ['series'] * len(y_f32),
            'target': y_f32,
            'timestamp': safe_date_range(start='2000-01-01', periods=len(y_f32), freq=gluon_freq),
        })

        dataset = PandasDataset.from_long_dataframe(
            df, item_id='item_id', target='target', timestamp='timestamp',
        )

        # Get forecast samples
        forecasts = list(predictor.predict(dataset, num_samples=num_samples))

        # Return median as point forecast
        samples = forecasts[0].samples
        return np.median(samples, axis=0)

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate quantile forecasts using Lag-Llama samples."""
        self._check_fitted()

        try:
            import torch
            from huggingface_hub import hf_hub_download
            from gluonts.dataset.pandas import PandasDataset
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError:
            raise ImportError(
                "Lag-Llama dependencies not installed. Install with:\n"
                "pip install lag-llama gluonts torch huggingface_hub"
            )

        device = self._get_device()
        context_length = self.params.get('context_length', 32)
        num_samples = self.params.get('num_samples', 100)

        ckpt_path = hf_hub_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt"
        )

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=horizon,
            context_length=context_length,
            input_size=1,
            n_layer=8,
            n_embd_per_head=32,
            n_head=4,
            device=torch.device(device),
        )

        predictor = estimator.create_lightning_module().to(device)

        df = pd.DataFrame({
            'target': self._train_y,
        }, index=safe_date_range(
            start='2000-01-01',
            periods=len(self._train_y),
            freq=self._freq
        ))

        dataset = PandasDataset.from_long_dataframe(
            df.reset_index().rename(columns={'index': 'timestamp'}),
            target='target',
            timestamp='timestamp',
        )

        forecasts = list(estimator.create_predictor(
            estimator.create_transformation(),
            predictor
        ).predict(dataset, num_samples=num_samples))

        samples = forecasts[0].samples

        result = np.zeros((horizon, len(quantiles)))
        for i, q in enumerate(quantiles):
            result[:, i] = np.quantile(samples, q, axis=0)

        return result

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="LagLlama",
            category=ModelCategory.LLM,
            library="lag-llama",
            year=2023,
            paper="Rasul et al. (2023) - Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting",
            probabilistic=True,
            zero_shot=True,
            requires_gpu=True,
            min_gpu_memory_gb=4.0,
            huggingface_id="time-series-foundation-models/Lag-Llama",
            github_url="https://github.com/time-series-foundation-models/lag-llama",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("context_length", "int", low=32, high=512, default=32),
            ParamSpace("num_samples", "int", low=20, high=200, default=100),
        ]


@register_model
class Moirai(Forecaster):
    """
    Moirai: Universal time series foundation model from Salesforce.

    Masked encoder-based model pretrained on large-scale time series data.
    Zero-shot forecasting with multiple model sizes.

    Args:
        model_size: 'small', 'base', 'large' (default: 'base')
        context_length: Number of past observations (default: 512)
        num_samples: Number of samples for probabilistic forecasts
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        model_size: str = 'base',
        context_length: int = 512,
        num_samples: int = 100,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_size=model_size,
            context_length=context_length,
            num_samples=num_samples,
            device=device,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'Moirai':
        """Moirai is zero-shot - just stores context."""
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

    _MOIRAI_MODEL_MAP = {
        'small': 'Salesforce/moirai-1.1-R-small',
        'base': 'Salesforce/moirai-1.1-R-base',
        'large': 'Salesforce/moirai-1.1-R-large',
    }

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        from gluonts.dataset.common import ListDataset

        model_size = self.params.get('model_size', 'base')
        context_length = self.params.get('context_length', 512)
        num_samples = self.params.get('num_samples', 100)
        hf_model = self._MOIRAI_MODEL_MAP.get(model_size, self._MOIRAI_MODEL_MAP['base'])

        y = self._train_y
        if len(y) > context_length:
            y = y[-context_length:]

        # Module cached — loaded once, reused for all series
        module = _FoundationModuleCache.get(hf_model, MoiraiModule.from_pretrained)
        model = MoiraiForecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            patch_size="auto",
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1)

        ds = ListDataset(
            [{"start": pd.Timestamp("2000-01-01"), "target": y}],
            freq="D",
        )
        forecasts = list(predictor.predict(ds))
        samples = forecasts[0].samples  # (num_samples, horizon)
        return np.median(samples, axis=0)

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate quantile forecasts using Moirai samples."""
        self._check_fitted()

        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        from gluonts.dataset.common import ListDataset

        model_size = self.params.get('model_size', 'base')
        context_length = self.params.get('context_length', 512)
        num_samples = self.params.get('num_samples', 100)
        hf_model = self._MOIRAI_MODEL_MAP.get(model_size, self._MOIRAI_MODEL_MAP['base'])

        y = self._train_y
        if len(y) > context_length:
            y = y[-context_length:]

        module = _FoundationModuleCache.get(hf_model, MoiraiModule.from_pretrained)
        model = MoiraiForecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            patch_size="auto",
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1)

        ds = ListDataset(
            [{"start": pd.Timestamp("2000-01-01"), "target": y}],
            freq="D",
        )
        forecasts = list(predictor.predict(ds))
        samples = forecasts[0].samples

        result = np.zeros((horizon, len(quantiles)))
        for i, q in enumerate(quantiles):
            result[:, i] = np.quantile(samples, q, axis=0)

        return result

    @classmethod
    def predict_batch(
        cls,
        contexts: List[np.ndarray],
        horizon: int,
        model_size: str = 'base',
        context_length: int = 512,
        num_samples: int = 100,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Batch inference: multiple series in one forward pass via GluonTS ListDataset.

        Args:
            contexts: List of 1D training arrays
            horizon: Forecast horizon (same for all series in batch)
            model_size: 'small', 'base', 'large'
            context_length: Max context window
            num_samples: Probabilistic samples
            batch_size: GPU batch size

        Returns:
            List of 1D forecast arrays (median), one per context
        """
        if not contexts:
            return []

        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        from gluonts.dataset.common import ListDataset

        hf_model = cls._MOIRAI_MODEL_MAP.get(model_size, cls._MOIRAI_MODEL_MAP['base'])
        module = _FoundationModuleCache.get(hf_model, MoiraiModule.from_pretrained)

        model = MoiraiForecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            patch_size="auto",
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=batch_size)

        # Truncate contexts to context_length and build dataset
        truncated = [c[-context_length:] if len(c) > context_length else c for c in contexts]
        ds = ListDataset(
            [{"start": pd.Timestamp("2000-01-01"), "target": ctx} for ctx in truncated],
            freq="D",
        )

        forecasts = list(predictor.predict(ds))
        return [np.median(f.samples, axis=0) for f in forecasts]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Moirai",
            category=ModelCategory.LLM,
            library="uni2ts",
            year=2024,
            paper="Woo et al. (2024) - Unified Training of Universal Time Series Forecasting Transformers",
            probabilistic=True,
            zero_shot=True,
            requires_gpu=True,
            min_gpu_memory_gb=4.0,
            huggingface_id="Salesforce/moirai-1.0-R-base",
            github_url="https://github.com/SalesforceAIResearch/uni2ts",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("model_size", "categorical", choices=['small', 'base', 'large'], default='base'),
            ParamSpace("context_length", "int", low=64, high=1024, default=512),
            ParamSpace("num_samples", "int", low=20, high=200, default=100),
        ]


# ============================================================================
# ============================================================================


@register_model
class Moirai2(Forecaster):
    """
    Moirai 2.0: Next-generation universal time series foundation model from Salesforce.

    Key improvements over Moirai 1.0:
    - Decoder-only architecture (vs encoder-only)
    - Quantile forecasting with multi-token prediction
    - New diverse pretraining dataset (36M time series)
    - Better GIFT-Eval benchmark performance

    Zero-shot forecasting with multiple model sizes.

    Args:
        model_size: 'small' (only size available on HuggingFace)
        context_length: Number of past observations (default: 512)
        num_samples: Number of samples for probabilistic forecasts
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        model_size: str = 'small',
        context_length: int = 512,
        num_samples: int = 100,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_size=model_size,
            context_length=context_length,
            num_samples=num_samples,
            device=device,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'Moirai2':
        """Moirai 2.0 is zero-shot - just stores context."""
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
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            from gluonts.dataset.common import ListDataset
        except ImportError:
            raise ImportError(
                "Moirai (uni2ts) not installed. Install with:\n"
                "pip install git+https://github.com/SalesforceAIResearch/uni2ts.git"
            )

        model_size = self.params.get('model_size', 'small')
        context_length = self.params.get('context_length', 512)

        # Only 'small' is available on HuggingFace as of March 2026
        hf_model = 'Salesforce/moirai-2.0-R-small'

        y = self._train_y
        if len(y) > context_length:
            y = y[-context_length:]

        # Module cached — loaded once, reused for all series
        module = _FoundationModuleCache.get(hf_model, Moirai2Module.from_pretrained)
        model = Moirai2Forecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1)

        ds = ListDataset(
            [{"start": pd.Timestamp("2000-01-01"), "target": y}],
            freq="D",
        )
        forecasts = list(predictor.predict(ds))
        # Moirai2 returns QuantileForecast, not SampleForecast
        return forecasts[0].quantile(0.5)

    @classmethod
    def predict_batch(
        cls,
        contexts: List[np.ndarray],
        horizon: int,
        context_length: int = 512,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Batch inference for Moirai2 via GluonTS ListDataset.

        Returns list of 1D forecast arrays (median quantile).
        """
        if not contexts:
            return []

        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        from gluonts.dataset.common import ListDataset

        hf_model = 'Salesforce/moirai-2.0-R-small'
        module = _FoundationModuleCache.get(hf_model, Moirai2Module.from_pretrained)

        model = Moirai2Forecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=batch_size)

        truncated = [c[-context_length:] if len(c) > context_length else c for c in contexts]
        ds = ListDataset(
            [{"start": pd.Timestamp("2000-01-01"), "target": ctx} for ctx in truncated],
            freq="D",
        )

        forecasts = list(predictor.predict(ds))
        # Moirai2 returns QuantileForecast — extract median
        return [f.quantile(0.5) for f in forecasts]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Moirai2",
            category=ModelCategory.LLM,
            library="uni2ts",
            year=2025,
            paper="Woo et al. (2025) - Moirai 2.0: When Less Is More",
            probabilistic=True,
            zero_shot=True,
            requires_gpu=True,
            min_gpu_memory_gb=4.0,
            huggingface_id="Salesforce/moirai-2.0-R-small",
            github_url="https://github.com/SalesforceAIResearch/uni2ts",
            notes="Decoder-only, quantile forecasting, multi-token prediction. Only 'small' size released.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("model_size", "categorical", choices=['small'], default='small'),
            ParamSpace("context_length", "int", low=64, high=1024, default=512),
        ]


@register_model
class MoiraiMoE(Forecaster):
    """
    Moirai-MoE: Mixture-of-Experts time series foundation model from Salesforce.

    First MoE architecture for time series forecasting:
    - Token-level expert specialization
    - 17% improvement over Moirai 1.0
    - 65x fewer activated parameters than competitors

    Zero-shot forecasting with sparse expert activation.

    Args:
        model_size: 'small', 'base', 'large' (default: 'base')
        context_length: Number of past observations (default: 512)
        num_samples: Number of samples for probabilistic forecasts
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        model_size: str = 'base',
        context_length: int = 512,
        num_samples: int = 100,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_size=model_size,
            context_length=context_length,
            num_samples=num_samples,
            device=device,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'MoiraiMoE':
        """Moirai-MoE is zero-shot - just stores context."""
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

    _MOIRAI_MOE_MODEL_MAP = {
        'small': 'Salesforce/moirai-moe-1.0-R-small',
        'base': 'Salesforce/moirai-moe-1.0-R-base',
        'large': 'Salesforce/moirai-moe-1.0-R-large',
    }

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        try:
            import torch
            from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
            from gluonts.dataset.common import ListDataset
        except ImportError:
            raise ImportError(
                "Moirai (uni2ts) not installed. Install with:\n"
                "pip install git+https://github.com/SalesforceAIResearch/uni2ts.git"
            )

        model_size = self.params.get('model_size', 'base')
        context_length = self.params.get('context_length', 512)
        num_samples = self.params.get('num_samples', 100)
        hf_model = self._MOIRAI_MOE_MODEL_MAP.get(model_size, self._MOIRAI_MOE_MODEL_MAP['base'])

        y = self._train_y
        if len(y) > context_length:
            y = y[-context_length:]

        # Module cached — loaded once, reused for all series
        module = _FoundationModuleCache.get(hf_model, MoiraiMoEModule.from_pretrained)
        model = MoiraiMoEForecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            patch_size=16,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1)

        ds = ListDataset(
            [{"start": pd.Timestamp("2000-01-01"), "target": y}],
            freq="D",
        )
        forecasts = list(predictor.predict(ds))
        samples = forecasts[0].samples  # (num_samples, horizon)
        return np.median(samples, axis=0)

    @classmethod
    def predict_batch(
        cls,
        contexts: List[np.ndarray],
        horizon: int,
        model_size: str = 'base',
        context_length: int = 512,
        num_samples: int = 100,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Batch inference for MoiraiMoE via GluonTS ListDataset.

        Returns list of 1D forecast arrays (median of samples).
        """
        if not contexts:
            return []

        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
        from gluonts.dataset.common import ListDataset

        hf_model = cls._MOIRAI_MOE_MODEL_MAP.get(model_size, cls._MOIRAI_MOE_MODEL_MAP['base'])
        module = _FoundationModuleCache.get(hf_model, MoiraiMoEModule.from_pretrained)

        model = MoiraiMoEForecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            patch_size=16,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=batch_size)

        truncated = [c[-context_length:] if len(c) > context_length else c for c in contexts]
        ds = ListDataset(
            [{"start": pd.Timestamp("2000-01-01"), "target": ctx} for ctx in truncated],
            freq="D",
        )

        forecasts = list(predictor.predict(ds))
        return [np.median(f.samples, axis=0) for f in forecasts]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="MoiraiMoE",
            category=ModelCategory.LLM,
            library="uni2ts",
            year=2025,
            paper="Woo et al. (2025) - Moirai-MoE: Mixture of Experts for Time Series",
            probabilistic=True,
            zero_shot=True,
            requires_gpu=True,
            min_gpu_memory_gb=6.0,  # MoE models need more VRAM
            huggingface_id="Salesforce/moirai-moe-1.0-R-base",
            github_url="https://github.com/SalesforceAIResearch/uni2ts",
            notes="First MoE for TS, 17% improvement, 65x fewer activated params",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("model_size", "categorical", choices=['small', 'base', 'large'], default='base'),
            ParamSpace("context_length", "int", low=64, high=1024, default=512),
            ParamSpace("num_samples", "int", low=20, high=200, default=100),
        ]


@register_model
class TimesFM(Forecaster):
    """
    TimesFM: Time Series Foundation Model from Google.

    Decoder-only transformer pretrained on Google Trends and synthetic data.
    Zero-shot forecasting with 200M parameters.

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

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'TimesFM':
        """TimesFM is zero-shot - just stores context."""
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
            import timesfm
        except ImportError:
            raise ImportError(
                "TimesFM not installed. Install with:\n"
                "pip install timesfm"
            )

        device = self._get_device()
        context_length = min(self.params.get('context_length', 512), 512)

        # timesfm 1.3.0+ uses TimesFmHparams + TimesFmCheckpoint
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend='gpu' if device == 'cuda' else 'cpu',
                per_core_batch_size=32,
                horizon_len=horizon,
                num_layers=20,
                model_dims=1280,
                input_patch_len=32,
                output_patch_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
            ),
        )

        y = self._train_y
        if len(y) > context_length:
            y = y[-context_length:]

        freq_map = {
            'D': 0, 'W': 1, 'M': 2, 'H': 3, 'T': 4, 'S': 5,
        }
        freq_code = freq_map.get(self._freq, 0)

        forecast, _ = tfm.forecast([y], [freq_code])
        return forecast[0][:horizon]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TimesFM",
            category=ModelCategory.LLM,
            library="timesfm",
            year=2024,
            paper="Das et al. (2024) - A decoder-only foundation model for time-series forecasting",
            probabilistic=False,
            zero_shot=True,
            requires_gpu=True,
            min_gpu_memory_gb=4.0,
            huggingface_id="google/timesfm-1.0-200m",
            github_url="https://github.com/google-research/timesfm",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("context_length", "int", low=64, high=512, default=512),
        ]


# ============================================================================
# ============================================================================


@register_model
class TimesFM25(Forecaster):
    """
    TimesFM 2.5: Updated Google Time Series Foundation Model.

    Key improvements over TimesFM 1.0:
    - Reduced from 500M to 200M parameters (more efficient)
    - Extended context length to 16,384 data points
    - Support for daily and weekly resolutions
    - Better GIFT-Eval benchmark performance

    Zero-shot forecasting with extended context.

    Args:
        context_length: Number of past observations (default: 512, max: 16384)
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

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'TimesFM25':
        """TimesFM 2.5 is zero-shot - just stores context."""
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
            from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
        except ImportError:
            raise ImportError(
                "TimesFM not installed. Install with:\n"
                "pip install git+https://github.com/google-research/timesfm.git"
            )

        context_length = min(self.params.get('context_length', 512), 16384)

        # TimesFM 2.5 uses from_pretrained + compile(ForecastConfig)
        model = TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
        )
        model.compile(ForecastConfig(
            max_context=context_length,
            max_horizon=horizon,
            normalize_inputs=True,
            per_core_batch_size=32,
        ))

        y = self._train_y.astype(np.float32)
        if len(y) > context_length:
            y = y[-context_length:]

        # forecast returns (point_forecast, quantile_forecast)
        point_forecast, _ = model.forecast(horizon=horizon, inputs=[y])
        return point_forecast[0][:horizon]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TimesFM25",
            category=ModelCategory.LLM,
            library="timesfm",
            year=2025,
            paper="Das et al. (2025) - TimesFM 2.5",
            probabilistic=False,
            zero_shot=True,
            requires_gpu=True,
            min_gpu_memory_gb=4.0,
            huggingface_id="google/timesfm-2.5-200m-pytorch",
            github_url="https://github.com/google-research/timesfm",
            notes="200M params, 16K context, daily/weekly support",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("context_length", "int", low=64, high=16384, default=512),
        ]


@register_model
class MOMENT(Forecaster):
    """
    MOMENT: Masked time series pre-training from CMU/AutonLab.

    Encoder-only model using masked reconstruction objective.
    Zero-shot forecasting via embedding-based approach.

    Args:
        model_size: 'small', 'base', 'large' (default: 'large')
        context_length: Number of past observations (default: 512)
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        model_size: str = 'large',
        context_length: int = 512,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_size=model_size,
            context_length=context_length,
            device=device,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'MOMENT':
        """MOMENT is zero-shot - just stores context."""
        y = self._validate_y(y)
        self._train_y = y
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
            from momentfm import MOMENTPipeline
        except ImportError:
            raise ImportError(
                "MOMENT not installed. Install with:\n"
                "pip install momentfm"
            )

        device = self._get_device()
        model_size = self.params.get('model_size', 'large')
        context_length = self.params.get('context_length', 512)

        model_map = {
            'small': 'AutonLab/MOMENT-1-small',
            'base': 'AutonLab/MOMENT-1-base',
            'large': 'AutonLab/MOMENT-1-large',
        }
        model_name = model_map.get(model_size, model_map['large'])

        # Load model
        model = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': horizon,
            },
        )
        model.init()
        model = model.to(device)

        # Prepare context - MOMENT expects (batch, n_channels, seq_len)
        y = self._train_y
        if len(y) > context_length:
            y = y[-context_length:]

        # Pad if necessary
        if len(y) < context_length:
            y = np.pad(y, (context_length - len(y), 0), mode='edge')

        context = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        input_mask = torch.ones(1, context_length, device=device)

        # Forecast - MOMENT requires keyword args x_enc + input_mask
        with torch.no_grad():
            output = model(x_enc=context, input_mask=input_mask)
            forecast = output.forecast.squeeze().cpu().numpy()

        return forecast[:horizon]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="MOMENT",
            category=ModelCategory.LLM,
            library="momentfm",
            year=2024,
            paper="Goswami et al. (2024) - MOMENT: A Family of Open Time-series Foundation Models",
            probabilistic=False,
            zero_shot=True,
            requires_gpu=True,
            min_gpu_memory_gb=4.0,
            huggingface_id="AutonLab/MOMENT-1-large",
            github_url="https://github.com/moment-timeseries-foundation-model/moment",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("model_size", "categorical", choices=['small', 'base', 'large'], default='large'),
            ParamSpace("context_length", "int", low=64, high=512, default=512),
        ]
