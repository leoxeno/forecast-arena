# Modified for forecast-arena: extracted and adapted from AION benchmark utilities, October 2026.
"""
forecasters/wrappers/chronos_wrapper.py - Amazon Chronos foundation models.

Wraps Chronos - pretrained time series foundation models from Amazon.

Chronos v1: T5-based encoder-decoder (2024)
Chronos-Bolt: Encoder-only, multivariate support (2025)

Zero-shot forecasting without training.

Performance notes:
- Pipeline is cached at class level (loaded once, reused for all series)
- Batch inference via predict_batch() for multi-series benchmarks (3-6x GPU speedup)
- BF16 by default on CUDA (same exponent range as FP32, halved memory bandwidth)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


def _detect_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _get_dtype(device: str, dtype_str: Optional[str] = None):
    """Get torch dtype. BF16 on CUDA by default (optimal for foundation models)."""
    import torch
    if dtype_str == 'float16':
        return torch.float16
    if dtype_str == 'float32':
        return torch.float32
    # Default: BF16 on GPU (same exponent range as FP32), FP32 on CPU
    return torch.bfloat16 if device != 'cpu' else torch.float32


class _ChronosPipelineCache:
    """Class-level pipeline cache. Load model once, reuse for all series.

    This eliminates the 2-5s per-series overhead of from_pretrained().
    The cache is keyed by (model_name, device) — shared across all instances.
    """
    _cache: Dict[Tuple[str, str], object] = {}

    @classmethod
    def get(cls, model_id: str, device: str, torch_dtype) -> object:
        """Get or create a cached pipeline."""
        from chronos import BaseChronosPipeline

        key = (model_id, device)
        if key not in cls._cache:
            cls._cache[key] = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device, torch_dtype=torch_dtype,
            )
        return cls._cache[key]

    @classmethod
    def clear(cls):
        """Clear cache (useful for freeing GPU memory)."""
        cls._cache.clear()


@register_model
class Chronos(Forecaster):
    """
    Chronos: Amazon's pretrained time series foundation model (v1).

    Zero-shot forecasting - no training required.
    Supports multiple model sizes for accuracy/speed tradeoff.

    Pipeline is cached at class level — first call loads model (~3s),
    subsequent calls reuse it (~0ms overhead).

    For batch benchmarks, use predict_batch() to process multiple series
    in a single forward pass (3-6x GPU throughput improvement).

    Args:
        model_size: 'tiny', 'mini', 'small', 'base', 'large'
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
        num_samples: Number of samples for probabilistic forecasts
        torch_dtype: 'bfloat16' (default on GPU), 'float16', 'float32'
    """

    def __init__(
        self,
        model_size: str = 'small',
        device: Optional[str] = None,
        num_samples: int = 20,
        torch_dtype: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_size=model_size,
            device=device,
            num_samples=num_samples,
            torch_dtype=torch_dtype,
            **kwargs
        )

    def _get_pipeline(self):
        """Get cached pipeline (loads model on first call only)."""
        import torch
        model_size = self.params.get('model_size', 'small')
        device = self.params.get('device') or _detect_device()
        dtype = _get_dtype(device, self.params.get('torch_dtype'))
        model_id = f"amazon/chronos-t5-{model_size}"
        return _ChronosPipelineCache.get(model_id, device, dtype)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'Chronos':
        """Chronos is zero-shot - just stores context."""
        y = self._validate_y(y)
        self._train_y = y
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
        pipeline = self._get_pipeline()

        context = torch.tensor(self._train_y, dtype=torch.float32)
        forecast = pipeline.predict(
            context,
            prediction_length=horizon,
            num_samples=self.params.get('num_samples', 20),
        )

        # Return median as point forecast — shape: (batch, num_samples, pred_len) → (pred_len,)
        return np.median(forecast.numpy(), axis=1).squeeze()[:horizon]

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate quantile forecasts using Chronos samples."""
        self._check_fitted()

        import torch
        pipeline = self._get_pipeline()

        context = torch.tensor(self._train_y, dtype=torch.float32)
        samples = pipeline.predict(
            context,
            prediction_length=horizon,
            num_samples=self.params.get('num_samples', 100),
        ).numpy()

        # Compute quantiles — shape: (1, num_samples, pred_len) → squeeze batch
        samples = samples.squeeze(0)  # (num_samples, pred_len)
        result = np.zeros((horizon, len(quantiles)))
        for i, q in enumerate(quantiles):
            result[:, i] = np.quantile(samples, q, axis=0)[:horizon]

        return result

    @classmethod
    def predict_batch(
        cls,
        contexts: List[np.ndarray],
        horizon: int,
        model_size: str = 'small',
        device: Optional[str] = None,
        num_samples: int = 20,
        torch_dtype: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Batch inference: multiple series in one forward pass.

        Chronos natively accepts a list of variable-length tensors.
        This amortizes GPU dispatch overhead and fills tensor cores,
        giving 3-6x throughput vs. single-series calls.

        Args:
            contexts: List of 1D training arrays (variable lengths OK)
            horizon: Forecast horizon (same for all series in batch)
            model_size: Chronos model size
            device: Device (auto-detected if None)
            num_samples: Number of probabilistic samples
            torch_dtype: Precision override

        Returns:
            List of 1D forecast arrays, one per input context
        """
        import torch

        if not contexts:
            return []

        device = device or _detect_device()
        dtype = _get_dtype(device, torch_dtype)
        model_id = f"amazon/chronos-t5-{model_size}"
        pipeline = _ChronosPipelineCache.get(model_id, device, dtype)

        # Chronos accepts a list of 1D tensors for batch prediction
        ctx_tensors = [torch.tensor(c, dtype=torch.float32) for c in contexts]

        with torch.no_grad():
            forecast = pipeline.predict(
                ctx_tensors,
                prediction_length=horizon,
                num_samples=num_samples,
            )

        # Shape: (batch, num_samples, pred_len) → median → (batch, pred_len)
        medians = np.median(forecast.numpy(), axis=1)

        return [medians[i, :horizon] for i in range(len(contexts))]

    @classmethod
    def predict_batch_quantiles(
        cls,
        contexts: List[np.ndarray],
        horizon: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        model_size: str = 'small',
        device: Optional[str] = None,
        num_samples: int = 100,
        torch_dtype: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Batch quantile inference for multiple series.

        Returns:
            List of (horizon, n_quantiles) arrays, one per input context
        """
        import torch

        if not contexts:
            return []

        device = device or _detect_device()
        dtype = _get_dtype(device, torch_dtype)
        model_id = f"amazon/chronos-t5-{model_size}"
        pipeline = _ChronosPipelineCache.get(model_id, device, dtype)

        ctx_tensors = [torch.tensor(c, dtype=torch.float32) for c in contexts]

        with torch.no_grad():
            samples = pipeline.predict(
                ctx_tensors,
                prediction_length=horizon,
                num_samples=num_samples,
            ).numpy()

        # Shape: (batch, num_samples, pred_len)
        results = []
        for b in range(len(contexts)):
            result = np.zeros((horizon, len(quantiles)))
            for i, q in enumerate(quantiles):
                result[:, i] = np.quantile(samples[b], q, axis=0)[:horizon]
            results.append(result)

        return results

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Chronos",
            category=ModelCategory.LLM,
            library="chronos-forecasting",
            year=2024,
            paper="Ansari et al. (2024)",
            probabilistic=True,
            zero_shot=True,
            requires_gpu=True,
            min_gpu_memory_gb=4.0,
            huggingface_id="amazon/chronos-t5-small",
            github_url="https://github.com/amazon-science/chronos-forecasting",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace(
                "model_size", "categorical",
                choices=['tiny', 'mini', 'small', 'base', 'large'],
                default='small'
            ),
            ParamSpace("num_samples", "int", low=10, high=100, default=20),
        ]


# ============================================================================
# ============================================================================


@register_model
class Chronos2(Forecaster):
    """
    Chronos-Bolt: Amazon's next-generation time series foundation model.

    Released October 2025. Key improvements over Chronos v1:
    - Encoder-only architecture (120M params)
    - Native multivariate support
    - Covariate-informed forecasting
    - 250x faster than original, 20x more memory efficient

    Zero-shot forecasting - no training required.
    Pipeline cached at class level (same as Chronos v1).

    Args:
        model_size: 'tiny', 'mini', 'small', 'base', 'large' (default: 'small')
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
        num_samples: Number of samples for probabilistic forecasts
        torch_dtype: 'bfloat16' (default on GPU), 'float16', 'float32'
    """

    def __init__(
        self,
        model_size: str = 'small',
        device: Optional[str] = None,
        num_samples: int = 20,
        torch_dtype: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_size=model_size,
            device=device,
            num_samples=num_samples,
            torch_dtype=torch_dtype,
            **kwargs
        )

    def _get_pipeline(self):
        """Get cached pipeline (loads model on first call only)."""
        import torch
        model_size = self.params.get('model_size', 'small')
        device = self.params.get('device') or _detect_device()
        dtype = _get_dtype(device, self.params.get('torch_dtype'))
        model_id = f"amazon/chronos-bolt-{model_size}"
        return _ChronosPipelineCache.get(model_id, device, dtype)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'Chronos2':
        """Chronos-Bolt is zero-shot - just stores context."""
        y = self._validate_y(y)
        self._train_y = y
        self._covariates = X
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
        pipeline = self._get_pipeline()

        context = torch.tensor(self._train_y, dtype=torch.float32)

        # Chronos-Bolt is deterministic — predict() does NOT accept num_samples
        # Output shape varies by version/device:
        #   Bolt: (batch, pred_len, num_quantiles=9) or (batch, num_quantiles, pred_len)
        #   T5:   (batch, num_samples, pred_len)
        forecast = pipeline.predict(context, prediction_length=horizon)
        out = forecast.numpy().squeeze(0)  # remove batch dim → 2D
        if out.ndim == 2:
            # Determine axis order: one axis should be horizon, other is samples/quantiles
            if out.shape[0] == horizon:
                # (pred_len, num_quantiles) — take median quantile (index 4 if 9 quantiles)
                mid = out.shape[1] // 2
                return out[:, mid]
            elif out.shape[1] == horizon:
                # (num_samples/quantiles, pred_len) — take median across axis 0
                return np.median(out, axis=0)[:horizon]
            else:
                # Neither axis matches horizon — take median of larger axis
                return np.median(out, axis=0)[:horizon] if out.shape[0] < out.shape[1] else np.median(out, axis=1)[:horizon]
        # 1D fallback
        return out.flatten()[:horizon]

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate quantile forecasts using Chronos-Bolt's native quantile output."""
        self._check_fitted()

        import torch
        pipeline = self._get_pipeline()

        context = torch.tensor(self._train_y, dtype=torch.float32)
        forecast = pipeline.predict(context, prediction_length=horizon)
        out = forecast.numpy()

        # Bolt quantile levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        bolt_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        result = np.zeros((horizon, len(quantiles)))
        if out.ndim == 3:
            # Bolt: (batch, pred_len, 9) — map requested quantiles to nearest index
            for i, q in enumerate(quantiles):
                idx = min(range(len(bolt_quantiles)), key=lambda j: abs(bolt_quantiles[j] - q))
                result[:, i] = out[0, :, idx]
        else:
            # T5 fallback: compute from samples
            for i, q in enumerate(quantiles):
                result[:, i] = np.quantile(out, q, axis=1).squeeze()

        return result

    @classmethod
    def predict_batch(
        cls,
        contexts: List[np.ndarray],
        horizon: int,
        model_size: str = 'small',
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Batch inference for Chronos-Bolt (deterministic, no num_samples).

        Returns:
            List of 1D forecast arrays (median quantile), one per context
        """
        import torch

        if not contexts:
            return []

        device = device or _detect_device()
        dtype = _get_dtype(device, torch_dtype)
        model_id = f"amazon/chronos-bolt-{model_size}"
        pipeline = _ChronosPipelineCache.get(model_id, device, dtype)

        ctx_tensors = [torch.tensor(c, dtype=torch.float32) for c in contexts]

        with torch.no_grad():
            forecast = pipeline.predict(ctx_tensors, prediction_length=horizon)

        out = forecast.numpy()  # (batch, pred_len, 9) or (batch, 9, pred_len)

        results = []
        for b in range(len(contexts)):
            arr = out[b]  # 2D
            if arr.ndim == 2:
                if arr.shape[0] == horizon:
                    mid = arr.shape[1] // 2
                    results.append(arr[:, mid])
                elif arr.shape[1] == horizon:
                    results.append(np.median(arr, axis=0)[:horizon])
                else:
                    results.append(
                        np.median(arr, axis=0)[:horizon]
                        if arr.shape[0] < arr.shape[1]
                        else np.median(arr, axis=1)[:horizon]
                    )
            else:
                results.append(arr.flatten()[:horizon])
        return results

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Chronos2",
            category=ModelCategory.LLM,
            library="chronos-forecasting",
            year=2025,
            paper="Ansari et al. (2025) - Chronos-Bolt",
            probabilistic=True,
            zero_shot=True,
            requires_gpu=True,
            min_gpu_memory_gb=4.0,
            huggingface_id="amazon/chronos-bolt-small",
            github_url="https://github.com/amazon-science/chronos-forecasting",
            notes="250x faster than v1, multivariate support, covariate-informed",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace(
                "model_size", "categorical",
                choices=['tiny', 'mini', 'small', 'base', 'large'],
                default='small'
            ),
            ParamSpace("num_samples", "int", low=10, high=100, default=20),
        ]
