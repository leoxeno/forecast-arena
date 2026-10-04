"""
forecasters/wrappers/timemoe_wrapper.py - Time-MoE foundation model.

Wraps Time-MoE, a decoder-only mixture-of-experts foundation model for
time series forecasting (ICLR 2025 Spotlight).

Uses AutoModelForCausalLM with .generate() for auto-regressive forecasting.
Requires transformers==4.40.1 and runs on a GPU via dedicated image.

Reference: https://github.com/Time-MoE/Time-MoE
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


@register_model
class TimeMoE(Forecaster):
    """
    Time-MoE - Mixture-of-Experts foundation model for time series.

    Decoder-only auto-regressive model with mixture-of-experts layers.
    50M to 2.4B params. Default: 50M variant for feasibility.

    Args:
        model_id: HuggingFace model ID (default: Maple728/TimeMoE-50M)
        context_length: Number of context points (default: 512)
        device: 'cpu' or 'cuda'
        use_gpu: Whether to use GPU if available
    """

    def __init__(
        self,
        model_id: str = 'Maple728/TimeMoE-50M',
        context_length: int = 512,
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

        TimeMoE's remote code calls DynamicCache.get_max_length which was
        removed in transformers 4.45. Add it back as alias.
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
        """Lazily load the Time-MoE model from HuggingFace."""
        if self._pipeline is not None:
            return

        try:
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError as e:
            raise ImportError(
                "Time-MoE requires transformers and torch. "
                "Install with: pip install transformers==4.40.1 torch"
            ) from e

        self._patch_dynamic_cache()

        device = self.params.get('device')
        if device is None:
            use_gpu = self.params.get('use_gpu', True)
            if use_gpu and torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        model_id = self.params.get('model_id', 'Maple728/TimeMoE-50M')

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
    ) -> 'TimeMoE':
        """Zero-shot: just store context."""
        y = self._validate_y(y)
        context_length = self.params.get('context_length', 512)
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
        """Generate forecasts using Time-MoE."""
        self._check_fitted()
        self._load_model()

        import torch

        seqs = torch.tensor(
            self._train_y.astype(np.float32), dtype=torch.float32
        ).unsqueeze(0).to(self._device)

        # Normalize
        mean = seqs.mean(dim=-1, keepdim=True)
        std = seqs.std(dim=-1, keepdim=True) + 1e-8
        normed_seqs = (seqs - mean) / std

        with torch.no_grad():
            normed_output = self._pipeline.generate(normed_seqs, max_new_tokens=horizon)

        # Denormalize and extract forecast
        output = (std * normed_output + mean).squeeze().cpu().numpy()

        return output[-horizon:]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TimeMoE",
            category=ModelCategory.LLM,
            library="transformers",
            year=2025,
            paper="Shi et al. (2025) - Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts",
            zero_shot=True,
            huggingface_id="Maple728/TimeMoE-50M",
            github_url="https://github.com/Time-MoE/Time-MoE",
            notes="ICLR 2025 Spotlight. Requires transformers==4.40.1. GPU recommended.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("context_length", "int", low=256, high=2048, default=512),
            ParamSpace(
                "model_id", "categorical",
                choices=['Maple728/TimeMoE-50M'],
                default='Maple728/TimeMoE-50M',
            ),
        ]
