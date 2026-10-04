"""
forecasters/wrappers - Model wrappers for various time series libraries.

Each wrapper file contains Forecaster implementations for a specific library.
Wrappers are automatically registered via @register_model decorator on import.

Supported libraries:
- statsmodels: ARIMA, ETS, VAR, Holt-Winters
- sklearn: ML regressors (RF, XGBoost, CatBoost, etc.)
- darts: NBEATS, NHiTS, TCN, TiDE, Transformer, boosting models
- gluonts: DeepAR, TFT, TimeGrad, PatchTST, TACTiS
- neuralforecast: Neural models (LSTM, Transformers, TimeLLM, KAN, etc.)
- statsforecast: Fast statistical models
- pytorch_forecasting: TFT, DeepAR
- chronos: Amazon's foundation model
- prophet: Facebook Prophet
- orbit: Bayesian time series
- greykite: LinkedIn's forecaster
- etna: Tinkoff's framework
- autots: AutoML for time series
- automl: AutoGluon, FLAML, MLForecast
- sktime: Unified TS interface
- timer: TIME Embeddings
- tsfresh: Feature extraction
- tslearn: TS machine learning
- fforma: Feature-based forecast combinations
- tsai: fastai-style deep learning (InceptionTime, ROCKET, TST, etc.)
- foundation: Lag-Llama, Moirai, TimesFM, MOMENT
- probabilistic: MAPIE, NGBoost
"""

# Import all wrappers to trigger @register_model
# Imports are wrapped in try/except to handle missing optional deps

import os


def _safe_import(module_name: str):
    """Import a module, silently skipping if dependencies are missing."""
    try:
        __import__(f"forecast_arena.forecasters.wrappers.{module_name}", fromlist=[module_name])
    except (ImportError, AttributeError):
        pass  # Optional dependency not installed or partially initialized


# Core libraries (most commonly available)
_safe_import("naive_wrapper")
_safe_import("statsmodels_wrapper")
_safe_import("sklearn_wrapper")

# Deep learning
_safe_import("darts_wrapper")
_safe_import("gluonts_wrapper")
_safe_import("neuralforecast_wrapper")
_safe_import("pytorch_forecasting_wrapper")

# Statistical
_safe_import("statsforecast_wrapper")
_safe_import("prophet_wrapper")

# Bayesian / Probabilistic
_safe_import("orbit_wrapper")
_safe_import("probabilistic_wrapper")

# AutoML
_safe_import("autots_wrapper")
_safe_import("etna_wrapper")
_safe_import("greykite_wrapper")
_safe_import("automl_wrapper")

# Foundation models / LLMs
_safe_import("chronos_wrapper")
_safe_import("timer_wrapper")
_safe_import("foundation_wrapper")

# Other
_safe_import("sktime_wrapper")
_safe_import("tsfresh_wrapper")
_safe_import("tslearn_wrapper")
_safe_import("fforma_wrapper")
_safe_import("tsai_wrapper")

_safe_import("arch_wrapper")

_safe_import("tsfm_wrapper")

_safe_import("tslib_wrapper")

# PyPOTS prints a banner on import and every PyPOTS model is excluded by the speed audit
# (benchmarks/exclusions.py); load it on request only.
if os.environ.get("FORECAST_ARENA_LOAD_PYPOTS") == "1":
    _safe_import("pypots_wrapper")

_safe_import("mamba_wrapper")

# Toto Foundation Model (Datadog) - requires a GPU (gluonts conflict)
_safe_import("toto_wrapper")

# Time-MoE Foundation Model - requires a GPU (transformers==4.40.1)
_safe_import("timemoe_wrapper")

_safe_import("sugihara_wrapper")
_safe_import("gibson_legendre_wrapper")
