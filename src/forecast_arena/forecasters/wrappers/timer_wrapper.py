"""
forecasters/wrappers/timer_wrapper.py - Timer foundation model.

Wraps Timer (Time-series foundation model from THUML) using HuggingFace weights.
Reference: https://github.com/thuml/Timer
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


@register_model
class Timer(Forecaster):
    """
    Timer - Time-series foundation model.

    Pretrained transformer for zero-shot time series forecasting.
    Uses HuggingFace weights from thuml/timer-base-84m.

    Args:
        model_id: HuggingFace model ID (default: thuml/timer-base-84m)
        context_length: Number of context tokens (default: 96)
        device: 'cpu' or 'cuda' (default: 'cpu')
        use_gpu: Whether to use GPU if available (default: True)
    """

    # Model variants available on HuggingFace
    MODEL_VARIANTS = {
        'base': 'thuml/timer-base-84m',
    }

    def __init__(
        self,
        model_id: str = 'thuml/timer-base-84m',
        context_length: int = 96,
        device: Optional[str] = None,
        use_gpu: bool = True,
        **kwargs
    ):
        super().__init__(
            model_id=model_id,
            context_length=context_length,
            device=device,
            use_gpu=use_gpu,
            **kwargs
        )
        self._pipeline = None

    @staticmethod
    def _patch_dynamic_cache():
        """Monkey-patch DynamicCache for transformers>=4.45 compatibility.

        Timer's remote code (thuml/timer-base-84m) calls
        DynamicCache.get_max_length which was removed in transformers 4.45.
        We add it back as an alias for get_max_cache_shape.
        """
        try:
            from transformers.cache_utils import DynamicCache
            if not hasattr(DynamicCache, 'get_max_length'):
                DynamicCache.get_max_length = (
                    DynamicCache.get_max_cache_shape
                    if hasattr(DynamicCache, 'get_max_cache_shape')
                    else lambda self: 0
                )
        except ImportError:
            pass

    def _load_model(self):
        """Lazily load the Timer model from HuggingFace."""
        if self._pipeline is not None:
            return

        try:
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError as e:
            raise ImportError(
                "Timer requires transformers and torch. "
                "Install with: pip install transformers torch"
            ) from e

        self._patch_dynamic_cache()

        # Determine device
        device = self.params.get('device')
        if device is None:
            use_gpu = self.params.get('use_gpu', True)
            if use_gpu and torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        model_id = self.params.get('model_id', 'thuml/timer-base-84m')

        # Timer is a causal LM that uses .generate() for forecasting
        self._pipeline = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(device)
        self._device = device

        # Monkey-patch for transformers>=4.45 where _extract_past_from_model_output was removed
        if not hasattr(self._pipeline, '_extract_past_from_model_output'):
            import types

            def _extract_past(self_model, outputs, standardize_cache_format=False):
                past = getattr(outputs, 'past_key_values', None)
                return past

            self._pipeline._extract_past_from_model_output = types.MethodType(
                _extract_past, self._pipeline
            )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'Timer':
        """
        Timer is zero-shot - fit just stores the context for prediction.

        Args:
            y: Historical time series (used as context for prediction)
            X: Not used (Timer doesn't support exogenous features)
            freq: Not used
        """
        y = self._validate_y(y)

        # Store context - use last context_length points
        context_length = self.params.get('context_length', 96)
        if len(y) > context_length:
            self._train_y = y[-context_length:]
        else:
            self._train_y = y

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate forecasts using the Timer foundation model.

        Args:
            horizon: Number of steps to forecast
            X: Not used (Timer doesn't support exogenous features)

        Returns:
            Point forecasts as 1D numpy array
        """
        self._check_fitted()
        self._load_model()

        import torch

        # Timer uses AutoModelForCausalLM.generate() for forecasting
        seqs = torch.tensor(self._train_y, dtype=torch.float32).unsqueeze(0).to(self._device)

        # Normalize to mitigate scale effects
        mean = seqs.mean(dim=-1, keepdim=True)
        std = seqs.std(dim=-1, keepdim=True) + 1e-8
        normed_seqs = (seqs - mean) / std

        try:
            with torch.no_grad():
                normed_output = self._pipeline.generate(normed_seqs, max_new_tokens=horizon)
        except RuntimeError as e:
            if 'CUDA' in str(e) and self._device != 'cpu':
                # Retry on CPU as fallback for CUDA kernel errors
                self._pipeline = self._pipeline.cpu()
                self._device = 'cpu'
                seqs_cpu = seqs.cpu()
                mean_cpu = mean.cpu()
                std_cpu = std.cpu()
                normed_cpu = (seqs_cpu - mean_cpu) / std_cpu
                with torch.no_grad():
                    normed_output = self._pipeline.generate(normed_cpu, max_new_tokens=horizon)
                mean, std = mean_cpu, std_cpu
            else:
                raise

        # Rescale back to original scale
        output = (std * normed_output + mean).squeeze().cpu().numpy()

        return output[:horizon]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Timer",
            category=ModelCategory.LLM,
            library="transformers",
            year=2024,
            paper="Liu et al. (2024) - Timer: Transformers for Time Series at Scale",
            zero_shot=True,
            huggingface_id="thuml/timer-base-84m",
            github_url="https://github.com/thuml/Timer",
            notes="Requires transformers>=4.35.0, torch. Zero-shot foundation model.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace(
                "context_length",
                "int",
                low=32,
                high=512,
                default=96,
            ),
            ParamSpace(
                "model_id",
                "categorical",
                choices=['thuml/timer-base-84m'],
                default='thuml/timer-base-84m',
            ),
        ]
