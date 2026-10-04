"""Fit one registered model on one training window and time it.

The runner is deliberately small: look the model up in the registry, fit,
predict, collect what the model knows about itself (parameters, metadata,
quantiles, diagnostics) and never raise. A failure becomes a record with an
``error`` string, so a batch of hundreds of models runs to completion.
"""

from __future__ import annotations

import importlib.util
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .forecasters import MODEL_REGISTRY, get_model, get_models_by_library


@dataclass
class ModelResult:
    model_name: str
    forecast: Optional[np.ndarray] = None
    fitted_values: Optional[np.ndarray] = None
    fit_time_seconds: float = 0.0
    predict_time_seconds: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    prediction_quantiles: Optional[np.ndarray] = None
    diagnostics: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.forecast is not None


def run_single_model(
    model_name: str,
    y: np.ndarray,
    horizon: int = 1,
    freq: Optional[str] = None,
    X: Optional[np.ndarray] = None,
    X_future: Optional[np.ndarray] = None,
    **model_kwargs,
) -> ModelResult:
    """Fit ``model_name`` on ``y`` and forecast ``horizon`` steps."""
    result = ModelResult(model_name=model_name)
    try:
        model = get_model(model_name, **model_kwargs)
        result.params = model.get_params()

        t0 = time.perf_counter()
        model.fit(y, X=X, freq=freq, horizon=horizon)
        result.fit_time_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        forecast = model.predict(horizon=horizon, X=X_future)
        result.predict_time_seconds = time.perf_counter() - t0
        result.forecast = np.asarray(forecast, dtype=float).flatten()

        result.fitted_values = model.get_fitted_values()

        try:
            meta = model.get_metadata()
            result.metadata = {
                "category": meta.category.value,
                "library": meta.library,
                "probabilistic": meta.probabilistic,
                "requires_gpu": meta.requires_gpu,
                "zero_shot": meta.zero_shot,
                "exogenous": meta.exogenous,
                "year": meta.year,
            }
        except Exception:
            pass

        try:
            result.prediction_quantiles = model.predict_quantiles(
                horizon=horizon, quantiles=[0.1, 0.5, 0.9]
            )
        except Exception:
            pass

        try:
            diag = model.get_diagnostics()
            if diag:
                result.diagnostics = diag
        except Exception:
            pass

    except ImportError as e:
        result.error = f"Missing dependency: {e}"
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    return result


# pip distribution name -> importable module, where the two differ
_MODULE_FOR_LIBRARY = {
    "chronos-forecasting": "chronos",
    "granite-tsfm": "tsfm_public",
    "lag-llama": "lag_llama",
    "orbit-ml": "orbit",
    "toto-ts": "toto",
    "pytorch-forecasting": "pytorch_forecasting",
    "scikit-learn": "sklearn",
}


def library_available(model_name: str) -> bool:
    """True when the library behind ``model_name`` is importable in this environment."""
    library = MODEL_REGISTRY[model_name].get_metadata().library
    if library in ("numpy", "forecast_arena"):
        return True
    module = _MODULE_FOR_LIBRARY.get(library, library.replace("-", "_"))
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def available_models() -> List[str]:
    """Registered models whose library is installed here."""
    return sorted(m for m in MODEL_REGISTRY if library_available(m))


def select_models(
    models: Optional[List[str]] = None,
    library: Optional[str] = None,
) -> List[str]:
    """Resolve an explicit list or a library filter into registered model names."""
    if models is not None:
        unknown = sorted(set(models) - set(MODEL_REGISTRY))
        if unknown:
            raise ValueError(f"Unknown models: {unknown}. See `forecast-arena models`.")
        return list(models)
    if library is not None:
        by_library = get_models_by_library()
        if library not in by_library:
            raise ValueError(f"Unknown library '{library}'. Available: {sorted(by_library)}")
        return sorted(by_library[library])
    return sorted(MODEL_REGISTRY)


def run_models(
    y: Union[np.ndarray, pd.Series],
    horizon: int = 1,
    models: Optional[List[str]] = None,
    library: Optional[str] = None,
    freq: Optional[str] = None,
    X: Optional[np.ndarray] = None,
    X_future: Optional[np.ndarray] = None,
    verbose: bool = True,
    **model_kwargs,
) -> Dict[str, ModelResult]:
    """Run several models on the same training window."""
    y = np.asarray(y.values if isinstance(y, pd.Series) else y, dtype=float).flatten()
    names = select_models(models=models, library=library)
    results: Dict[str, ModelResult] = {}
    for i, name in enumerate(names, 1):
        if verbose:
            print(f"  [{i}/{len(names)}] {name} ...", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_single_model(name, y, horizon=horizon, freq=freq, X=X, X_future=X_future,
                                 **model_kwargs)
        results[name] = r
        if verbose:
            print("OK" if r.ok else f"ERROR ({(r.error or '')[:60]})")
    return results


def results_to_dataframe(results: Dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "ok": r.ok,
            "fit_time_s": round(r.fit_time_seconds, 4),
            "predict_time_s": round(r.predict_time_seconds, 4),
            "error": r.error,
            "library": (r.metadata or {}).get("library"),
            "category": (r.metadata or {}).get("category"),
        })
    return pd.DataFrame(rows)


__all__ = ["ModelResult", "run_single_model", "run_models", "select_models", "results_to_dataframe",
           "library_available", "available_models"]
