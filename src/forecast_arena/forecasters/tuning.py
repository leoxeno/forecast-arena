"""
forecasters/tuning.py - Hyperparameter tuning strategies.

Implements:
- RandomSearch (default, fast)
- GridSearch (exhaustive)
- BayesianSearch (optuna, sample-efficient)

All strategies use cross-validation to evaluate parameter combinations.

Design note: We implement these ourselves (lean) rather than using
sklearn GridSearchCV or similar, since time series CV requires
special handling of temporal splits.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

from .base import Forecaster, ParamSpace
from .validation import CVStrategy, expanding_cv, cross_validate


@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""
    best_params: Dict[str, Any]
    best_score: float
    best_model: Forecaster
    cv_results: List[Dict[str, Any]]  # All evaluated combinations
    n_iterations: int


class TuningStrategy(ABC):
    """Abstract base class for tuning strategies."""

    @abstractmethod
    def tune(
        self,
        forecaster: Forecaster,
        y: np.ndarray,
        cv: CVStrategy,
        param_space: List[ParamSpace],
        X: Optional[np.ndarray] = None,
        metric_fn: Optional[Callable] = None,
        maximize: bool = False,
        verbose: bool = False
    ) -> TuningResult:
        """
        Tune hyperparameters.

        Args:
            forecaster: Base forecaster (will be cloned with different params)
            y: Target time series
            cv: Cross-validation strategy
            param_space: Parameter search space
            X: Exogenous features (optional)
            metric_fn: Evaluation metric (default: MAE)
            maximize: If True, maximize metric; else minimize
            verbose: Print progress

        Returns:
            TuningResult with best parameters and model
        """
        pass


class RandomSearch(TuningStrategy):
    """
    Random search over parameter space.

    Samples random parameter combinations and evaluates via CV.
    Usually finds good params faster than grid search for high-dimensional
    spaces (Bergstra & Bengio, 2012).

    Args:
        n_iter: Number of random combinations to try
        seed: Random seed for reproducibility
    """

    def __init__(self, n_iter: int = 50, seed: Optional[int] = None):
        self.n_iter = n_iter
        self.seed = seed

    def tune(
        self,
        forecaster: Forecaster,
        y: np.ndarray,
        cv: CVStrategy,
        param_space: List[ParamSpace],
        X: Optional[np.ndarray] = None,
        metric_fn: Optional[Callable] = None,
        maximize: bool = False,
        verbose: bool = False
    ) -> TuningResult:
        """Run random search tuning."""
        from .metrics import mae as default_metric
        metric_fn = metric_fn or default_metric

        rng = np.random.default_rng(self.seed)

        cv_results = []
        best_score = float('inf') if not maximize else float('-inf')
        best_params = None
        best_model = None

        for i in range(self.n_iter):
            # Sample random parameters
            params = {p.name: p.sample_random(rng) for p in param_space}

            if verbose:
                print(f"[{i + 1}/{self.n_iter}] Trying: {params}")

            # Clone model with new params
            model = forecaster.clone(**params)

            # Evaluate with CV
            try:
                result = cross_validate(model, y, cv, X=X, metric_fn=metric_fn)
                score = result['mean']
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                continue

            cv_results.append({
                'params': params,
                'mean_score': score,
                'std_score': result['std'],
            })

            if verbose:
                print(f"  Score: {score:.4f} (+/- {result['std']:.4f})")

            # Update best
            is_better = (score > best_score) if maximize else (score < best_score)
            if is_better:
                best_score = score
                best_params = params
                best_model = model

        if best_model is None:
            raise RuntimeError("All parameter combinations failed")

        # Refit on full data
        best_model = forecaster.clone(**best_params)
        best_model.fit(y, X=X)

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            cv_results=cv_results,
            n_iterations=len(cv_results)
        )


class GridSearch(TuningStrategy):
    """
    Exhaustive grid search over parameter space.

    Evaluates all combinations of categorical params and sampled
    points for continuous params.

    Args:
        n_points_continuous: Number of points to sample for continuous params
    """

    def __init__(self, n_points_continuous: int = 5):
        self.n_points_continuous = n_points_continuous

    def _discretize_param_space(
        self,
        param_space: List[ParamSpace]
    ) -> Dict[str, List[Any]]:
        """Convert continuous params to discrete grid."""
        grid = {}

        for p in param_space:
            if p.param_type == 'categorical':
                grid[p.name] = p.choices
            elif p.param_type == 'int':
                # Linear spacing for integers
                values = np.linspace(p.low, p.high, self.n_points_continuous)
                grid[p.name] = sorted(set(int(v) for v in values))
            elif p.param_type == 'float':
                grid[p.name] = list(np.linspace(p.low, p.high, self.n_points_continuous))
            elif p.param_type == 'log_float':
                # Log spacing
                grid[p.name] = list(np.geomspace(p.low, p.high, self.n_points_continuous))

        return grid

    def tune(
        self,
        forecaster: Forecaster,
        y: np.ndarray,
        cv: CVStrategy,
        param_space: List[ParamSpace],
        X: Optional[np.ndarray] = None,
        metric_fn: Optional[Callable] = None,
        maximize: bool = False,
        verbose: bool = False
    ) -> TuningResult:
        """Run grid search tuning."""
        from itertools import product
        from .metrics import mae as default_metric
        metric_fn = metric_fn or default_metric

        # Create grid
        grid = self._discretize_param_space(param_space)
        param_names = list(grid.keys())
        param_values = list(grid.values())

        n_combinations = 1
        for vals in param_values:
            n_combinations *= len(vals)

        if verbose:
            print(f"Grid search over {n_combinations} combinations")

        cv_results = []
        best_score = float('inf') if not maximize else float('-inf')
        best_params = None
        best_model = None

        for i, values in enumerate(product(*param_values)):
            params = dict(zip(param_names, values))

            if verbose:
                print(f"[{i + 1}/{n_combinations}] Trying: {params}")

            model = forecaster.clone(**params)

            try:
                result = cross_validate(model, y, cv, X=X, metric_fn=metric_fn)
                score = result['mean']
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                continue

            cv_results.append({
                'params': params,
                'mean_score': score,
                'std_score': result['std'],
            })

            if verbose:
                print(f"  Score: {score:.4f}")

            is_better = (score > best_score) if maximize else (score < best_score)
            if is_better:
                best_score = score
                best_params = params
                best_model = model

        if best_model is None:
            raise RuntimeError("All parameter combinations failed")

        # Refit on full data
        best_model = forecaster.clone(**best_params)
        best_model.fit(y, X=X)

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            cv_results=cv_results,
            n_iterations=len(cv_results)
        )


class BayesianSearch(TuningStrategy):
    """
    Bayesian optimization using Optuna.

    Uses Tree-structured Parzen Estimator (TPE) to intelligently
    explore the parameter space, focusing on promising regions.

    More sample-efficient than random/grid search for expensive
    evaluations.

    Args:
        n_trials: Number of optimization trials
        seed: Random seed
        show_progress_bar: Show optuna progress bar
    """

    def __init__(
        self,
        n_trials: int = 50,
        seed: Optional[int] = None,
        show_progress_bar: bool = False
    ):
        self.n_trials = n_trials
        self.seed = seed
        self.show_progress_bar = show_progress_bar

    def tune(
        self,
        forecaster: Forecaster,
        y: np.ndarray,
        cv: CVStrategy,
        param_space: List[ParamSpace],
        X: Optional[np.ndarray] = None,
        metric_fn: Optional[Callable] = None,
        maximize: bool = False,
        verbose: bool = False
    ) -> TuningResult:
        """Run Bayesian optimization with Optuna."""
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "BayesianSearch requires optuna. Install with: pip install optuna"
            )

        from .metrics import mae as default_metric
        metric_fn = metric_fn or default_metric

        # Suppress optuna logging unless verbose
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        cv_results = []

        def objective(trial):
            # Sample parameters using ParamSpace.to_optuna
            params = {}
            for p in param_space:
                params[p.name] = p.to_optuna(trial)

            # Clone and evaluate
            model = forecaster.clone(**params)

            try:
                result = cross_validate(model, y, cv, X=X, metric_fn=metric_fn)
                score = result['mean']
            except Exception as e:
                if verbose:
                    print(f"Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned()

            cv_results.append({
                'params': params,
                'mean_score': score,
                'std_score': result['std'],
            })

            return score

        # Create study
        direction = 'maximize' if maximize else 'minimize'
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction=direction, sampler=sampler)

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.show_progress_bar,
        )

        # Extract best
        best_params = study.best_params
        best_score = study.best_value

        # Refit on full data
        best_model = forecaster.clone(**best_params)
        best_model.fit(y, X=X)

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            cv_results=cv_results,
            n_iterations=len(cv_results)
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def random_search(
    forecaster: Forecaster,
    y: np.ndarray,
    param_space: Optional[List[ParamSpace]] = None,
    cv: Optional[CVStrategy] = None,
    n_iter: int = 50,
    X: Optional[np.ndarray] = None,
    metric_fn: Optional[Callable] = None,
    maximize: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False
) -> TuningResult:
    """
    Convenience function for random search tuning.

    If param_space not provided, uses forecaster.get_param_space().

    Args:
        forecaster: Model to tune
        y: Target series
        param_space: Parameter search space (default: from forecaster)
        cv: CV strategy (default: expanding_cv with 5 splits)
        n_iter: Number of iterations
        X: Exogenous features
        metric_fn: Evaluation metric
        maximize: Whether to maximize metric
        seed: Random seed
        verbose: Print progress

    Returns:
        TuningResult with best params and model
    """
    param_space = param_space or forecaster.get_param_space()
    if not param_space:
        raise ValueError(
            f"{forecaster.__class__.__name__} has no param_space defined. "
            f"Provide param_space argument or implement get_param_space()."
        )

    cv = cv or expanding_cv(n_splits=5, horizon=1)

    tuner = RandomSearch(n_iter=n_iter, seed=seed)
    return tuner.tune(
        forecaster, y, cv, param_space,
        X=X, metric_fn=metric_fn, maximize=maximize, verbose=verbose
    )


def grid_search(
    forecaster: Forecaster,
    y: np.ndarray,
    param_space: Optional[List[ParamSpace]] = None,
    cv: Optional[CVStrategy] = None,
    n_points: int = 5,
    X: Optional[np.ndarray] = None,
    metric_fn: Optional[Callable] = None,
    maximize: bool = False,
    verbose: bool = False
) -> TuningResult:
    """Convenience function for grid search tuning."""
    param_space = param_space or forecaster.get_param_space()
    if not param_space:
        raise ValueError(f"{forecaster.__class__.__name__} has no param_space defined")

    cv = cv or expanding_cv(n_splits=5, horizon=1)

    tuner = GridSearch(n_points_continuous=n_points)
    return tuner.tune(
        forecaster, y, cv, param_space,
        X=X, metric_fn=metric_fn, maximize=maximize, verbose=verbose
    )


def bayesian_search(
    forecaster: Forecaster,
    y: np.ndarray,
    param_space: Optional[List[ParamSpace]] = None,
    cv: Optional[CVStrategy] = None,
    n_trials: int = 50,
    X: Optional[np.ndarray] = None,
    metric_fn: Optional[Callable] = None,
    maximize: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False
) -> TuningResult:
    """Convenience function for Bayesian optimization."""
    param_space = param_space or forecaster.get_param_space()
    if not param_space:
        raise ValueError(f"{forecaster.__class__.__name__} has no param_space defined")

    cv = cv or expanding_cv(n_splits=5, horizon=1)

    tuner = BayesianSearch(n_trials=n_trials, seed=seed)
    return tuner.tune(
        forecaster, y, cv, param_space,
        X=X, metric_fn=metric_fn, maximize=maximize, verbose=verbose
    )


# Public API
__all__ = [
    # Classes
    'TuningResult',
    'TuningStrategy',
    'RandomSearch',
    'GridSearch',
    'BayesianSearch',
    # Convenience functions
    'random_search',
    'grid_search',
    'bayesian_search',
]
