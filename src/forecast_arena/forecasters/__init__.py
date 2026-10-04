"""
forecasters - Unified time series forecasting framework.

Provides:
- 50+ model wrappers with consistent fit/predict interface
- Evaluation metrics (MAE, RMSE, MAPE, MASE, RMSSE, OWA, DM test)
- Cross-validation strategies (expanding, sliding, holdout, purged)
- Hyperparameter tuning (random, grid, bayesian)
- Rolling forecast evaluation

Usage:
    from forecasters import get_model, list_models, evaluate

    # List available models
    print(list_models())
    print(list_models(category='deep_learning'))

    # Get and use a model
    model = get_model('ARIMA', order=(1, 1, 1))
    model.fit(train_y)
    forecast = model.predict(horizon=12)

    # Evaluate with cross-validation
    from forecasters import expanding_cv, evaluate_cv
    cv = expanding_cv(n_splits=5, horizon=12)
    results = evaluate_cv(model, y, cv)
"""

from typing import Dict, List, Optional, Type

from .base import Forecaster, ModelMetadata, ModelCategory, ParamSpace

# Global model registry
MODEL_REGISTRY: Dict[str, Type[Forecaster]] = {}


def register_model(cls: Type[Forecaster]) -> Type[Forecaster]:
    """
    Decorator to register a model class in the global registry.

    Usage:
        @register_model
        class MyForecaster(Forecaster):
            ...
    """
    meta = cls.get_metadata()
    MODEL_REGISTRY[meta.name] = cls
    return cls


def get_model(name: str, **kwargs) -> Forecaster:
    """
    Factory: instantiate a registered model by name.

    Args:
        name: Model name (case-sensitive)
        **kwargs: Model hyperparameters

    Returns:
        Instantiated Forecaster

    Raises:
        ValueError: If model not found

    Example:
        model = get_model('ARIMA', order=(1, 1, 1))
        model = get_model('Chronos', model_size='small')
    """
    if name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{name}'. "
            f"Available models ({len(available)}): {available}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def list_models(
    category: Optional[ModelCategory] = None,
    library: Optional[str] = None,
    probabilistic: Optional[bool] = None,
    multivariate: Optional[bool] = None,
    zero_shot: Optional[bool] = None,
) -> List[str]:
    """
    List available models with optional filtering.

    Args:
        category: Filter by ModelCategory
        library: Filter by pip package name
        probabilistic: Filter by probabilistic support
        multivariate: Filter by multivariate support
        zero_shot: Filter by zero-shot capability

    Returns:
        Sorted list of model names
    """
    results = []
    for name, cls in MODEL_REGISTRY.items():
        meta = cls.get_metadata()

        if category is not None and meta.category != category:
            continue
        if library is not None and meta.library != library:
            continue
        if probabilistic is not None and meta.probabilistic != probabilistic:
            continue
        if multivariate is not None and meta.multivariate != multivariate:
            continue
        if zero_shot is not None and meta.zero_shot != zero_shot:
            continue

        results.append(name)

    return sorted(results)


def get_all_metadata() -> List[ModelMetadata]:
    """Get metadata for all registered models."""
    return [cls.get_metadata() for cls in MODEL_REGISTRY.values()]


def get_models_by_library() -> Dict[str, List[str]]:
    """Group model names by their source library."""
    by_library: Dict[str, List[str]] = {}
    for name, cls in MODEL_REGISTRY.items():
        lib = cls.get_metadata().library
        if lib not in by_library:
            by_library[lib] = []
        by_library[lib].append(name)
    return {k: sorted(v) for k, v in sorted(by_library.items())}


def get_models_by_category() -> Dict[ModelCategory, List[str]]:
    """Group model names by category."""
    by_category: Dict[ModelCategory, List[str]] = {}
    for name, cls in MODEL_REGISTRY.items():
        cat = cls.get_metadata().category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(name)
    return {k: sorted(v) for k, v in by_category.items()}


# Public API
__all__ = [
    # Base classes
    'Forecaster',
    'ModelMetadata',
    'ModelCategory',
    'ParamSpace',
    # Registry
    'MODEL_REGISTRY',
    'register_model',
    'get_model',
    'list_models',
    'get_all_metadata',
    'get_models_by_library',
    'get_models_by_category',
]

# Import submodules to trigger registration
# These are imported at the end to avoid circular imports
def _register_all_models():
    """Lazily import all wrappers to register models."""
    from . import wrappers  # noqa: F401


# Auto-register all models on import
_register_all_models()
