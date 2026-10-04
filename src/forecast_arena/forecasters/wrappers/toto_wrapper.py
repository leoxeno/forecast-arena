"""
forecasters/wrappers/toto_wrapper.py - Toto foundation model (Datadog).

Wraps Toto, a 151M parameter foundation model trained on 1 trillion
time series points from Datadog's telemetry data.

NOTE: toto-ts requires gluonts>=0.15.1 which conflicts with LagLlama's
gluonts==0.14.4. This model runs ONLY on a GPU via a dedicated image.
The wrapper is a thin stub that enables registration; actual inference
happens in the GPU container.

Reference: https://huggingface.co/Datadog/Toto-Open-Base-1.0
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


@register_model
class Toto(Forecaster):
    """
    Toto - Datadog's time series foundation model.

    151M parameter model trained on 1 trillion time points.
    Zero-shot forecasting via TotoForecaster.

    This model requires GPU execution due to dependency conflicts
    (gluonts>=0.15.1 conflicts with LagLlama's gluonts==0.14.4).

    Args:
        model_id: HuggingFace model ID
        context_length: Number of context points (default: 512)
        num_samples: Number of forecast samples for uncertainty (default: 100, must be divisible by 10)
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model_id: str = 'Datadog/Toto-Open-Base-1.0',
        context_length: int = 512,
        num_samples: int = 100,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_id=model_id,
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
    ) -> 'Toto':
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
        """Generate forecasts via Toto foundation model."""
        self._check_fitted()

        try:
            from toto.model.toto import Toto as TotoModel
            from toto.inference.forecaster import TotoForecaster
            from toto.data.util.dataset import MaskedTimeseries
            import torch
        except ImportError:
            raise ImportError(
                "toto-ts not installed. This model runs on a GPU. "
                "Install with: pip install toto-ts (WARNING: conflicts with gluonts==0.14.4)"
            )

        model_id = self.params.get('model_id', 'Datadog/Toto-Open-Base-1.0')
        num_samples = self.params.get('num_samples', 100)

        device = self.params.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        toto = TotoModel.from_pretrained(model_id).to(device)
        forecaster = TotoForecaster(toto.model)

        # Prepare input
        series = torch.tensor(
            self._train_y.astype(np.float32), dtype=torch.float32
        ).unsqueeze(0).to(device)  # (1, context_len)

        context_len = series.shape[1]
        inputs = MaskedTimeseries(
            series=series,
            padding_mask=torch.ones(1, context_len, dtype=torch.bool, device=device),
            id_mask=torch.zeros(1, context_len, dtype=torch.long, device=device),
            timestamp_seconds=torch.zeros(1, context_len, dtype=torch.float32, device=device),
            time_interval_seconds=torch.full((1,), 60 * 15, dtype=torch.float32, device=device),
        )

        with torch.no_grad():
            forecast = forecaster.forecast(inputs, prediction_length=horizon, num_samples=num_samples)

        return forecast.median.squeeze().cpu().numpy()[:horizon]

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Toto",
            category=ModelCategory.LLM,
            library="toto-ts",
            year=2025,
            paper="Datadog (2025) - Toto: Time Series Optimized Transformer",
            zero_shot=True,
            huggingface_id="Datadog/Toto-Open-Base-1.0",
            notes="151M params, trained on 1T time points. Requires a GPU (gluonts conflict).",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("context_length", "int", low=256, high=2048, default=512),
            ParamSpace("num_samples", "int", low=100, high=500, default=100),
        ]
