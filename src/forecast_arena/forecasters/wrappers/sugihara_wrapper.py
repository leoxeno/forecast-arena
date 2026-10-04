"""
Sugihara Simplex projection and S-map: empirical dynamic modelling baselines via pyEDM.

The canonical Empirical Dynamic Modeling (EDM) forecasters, nearest-neighbour
prediction in a delay-embedded state space, called directly through pyEDM's
C++ implementation.

References:
  Sugihara, G., May, R. (1990). "Nonlinear forecasting as a way of
     distinguishing chaos from measurement error in time series."
     Nature, 344(6268), 734-741.
  Sugihara, G. (1994). "Nonlinear forecasting for the classification
     of natural time series." Phil. Trans. R. Soc. A, 348(1688), 477-495.

Parameters selected via expanding-window cross-validation on training data:
  - E: embedding dimension
  - Simplex: E from a grid; forecast via weighted sum of E+1 nearest neighbors.
  - SMap: E from a grid; theta from a grid; locally-weighted linear regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory
from .. import register_model


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------

def _extend_df_for_forecast(y_train: np.ndarray, horizon: int) -> pd.DataFrame:
    """Extend y_train with NaN-padded horizon slots so pyEDM can predict."""
    y_ext = np.concatenate([y_train, np.full(horizon, np.nan)])
    return pd.DataFrame({'Time': np.arange(len(y_ext)), 'y': y_ext})


def _simplex_one_horizon(y_train: np.ndarray, E: int, tau: int, Tp: int) -> float:
    """Return Simplex prediction Tp steps ahead from the last training point.

    pyEDM semantics: `pred` range specifies TARGET times; the query for
    target t is at t-Tp. We want target = n-1+Tp (i.e., the forecast starting
    from the last training observation at t=n-1). Library is 1..n-1 which
    contains all valid (q, q+Tp) pairs with q <= n-1-Tp; the query at n-1
    produces an out-of-sample target at n-1+Tp not contained in the library.
    """
    try:
        import pyEDM
    except ImportError as e:
        raise ImportError("pyEDM is required for SugiharaSimplex and SugiharaSMap: pip install pyEDM") from e

    n = len(y_train)
    target = n - 1 + Tp
    if n - 1 < 2 * E + abs(tau) * (E - 1):
        return float('nan')
    df = _extend_df_for_forecast(y_train, horizon=Tp + 2)
    try:
        result = pyEDM.Simplex(
            dataFrame=df, lib=f'1 {n-1}', pred=f'{n-1} {n}',
            E=E, Tp=Tp, tau=-abs(tau),
            columns='y', target='y', showPlot=False,
        )
        row = result[result['Time'] == target]
        if len(row):
            p = row.iloc[0]['Predictions']
            return float(p) if np.isfinite(p) else float('nan')
        return float('nan')
    except Exception:
        return float('nan')


def _smap_one_horizon(y_train: np.ndarray, E: int, tau: int, theta: float,
                       Tp: int) -> float:
    """Return S-map prediction Tp steps ahead from the last training point."""
    try:
        import pyEDM
    except ImportError as e:
        raise ImportError("pyEDM is required for SugiharaSimplex and SugiharaSMap: pip install pyEDM") from e

    n = len(y_train)
    target = n - 1 + Tp
    if n - 1 < 2 * E + abs(tau) * (E - 1):
        return float('nan')
    df = _extend_df_for_forecast(y_train, horizon=Tp + 2)
    try:
        result = pyEDM.SMap(
            dataFrame=df, lib=f'1 {n-1}', pred=f'{n-1} {n}',
            E=E, Tp=Tp, tau=-abs(tau), theta=float(theta),
            columns='y', target='y', showPlot=False,
        )
        preds_df = result.get('predictions') if isinstance(result, dict) else None
        if preds_df is None:
            return float('nan')
        row = preds_df[preds_df['Time'] == target]
        if len(row):
            p = row.iloc[0]['Predictions']
            return float(p) if np.isfinite(p) else float('nan')
        return float('nan')
    except Exception:
        return float('nan')


def _select_E_simplex(y: np.ndarray, E_range, tau: int, horizon: int, n_cv: int) -> int:
    """Cross-validate E for Simplex on the final n_cv training folds."""
    n = len(y)
    fold_h = max(1, min(horizon, n // (n_cv + 2)))
    best_E, best_score = E_range[0], float('inf')
    for E in E_range:
        if (E - 1) * tau + 2 * fold_h > n // 2:
            continue
        errors = []
        for fold in range(n_cv):
            val_end = n - fold * fold_h
            val_start = val_end - fold_h
            if val_start < (E - 1) * tau + 2 * fold_h:
                break
            y_tr = y[:val_start]
            # Run Simplex at Tp=1..fold_h and score MAE on fold_h observations
            fold_errs = []
            for t in range(1, fold_h + 1):
                p = _simplex_one_horizon(y_tr, E=E, tau=tau, Tp=t)
                if np.isfinite(p) and val_start - 1 + t < n:
                    fold_errs.append(abs(p - y[val_start - 1 + t]))
            if fold_errs:
                errors.append(np.mean(fold_errs))
        if errors:
            s = np.mean(errors)
            if s < best_score:
                best_score = s
                best_E = E
    return best_E


def _select_E_theta_smap(y: np.ndarray, E_range, theta_range, tau: int,
                          horizon: int, n_cv: int):
    """Cross-validate (E, theta) for S-map."""
    n = len(y)
    fold_h = max(1, min(horizon, n // (n_cv + 2)))
    best = (E_range[0], theta_range[0])
    best_score = float('inf')
    for E in E_range:
        if (E - 1) * tau + 2 * fold_h > n // 2:
            continue
        for theta in theta_range:
            errors = []
            for fold in range(n_cv):
                val_end = n - fold * fold_h
                val_start = val_end - fold_h
                if val_start < (E - 1) * tau + 2 * fold_h:
                    break
                y_tr = y[:val_start]
                fold_errs = []
                for t in range(1, fold_h + 1):
                    p = _smap_one_horizon(y_tr, E=E, tau=tau, theta=theta, Tp=t)
                    if np.isfinite(p) and val_start - 1 + t < n:
                        fold_errs.append(abs(p - y[val_start - 1 + t]))
                if fold_errs:
                    errors.append(np.mean(fold_errs))
            if errors:
                s = np.mean(errors)
                if s < best_score:
                    best_score = s
                    best = (E, theta)
    return best


# -----------------------------------------------------------------------
# Simplex projection
# -----------------------------------------------------------------------

@register_model
class SugiharaSimplex(Forecaster):
    """Sugihara (1990) Simplex projection — pure EDM baseline.

    Delay-coordinate embedding at dimension E and lag tau. Forecast at
    horizon Tp is the weighted sum of the E+1 nearest neighbors' targets,
    with weights exponentially decaying in neighbor distance.

    Args:
        tau: Time delay. Default 1.
        E_range: Embedding dimensions to search. Default [2,3,4,5,6,7,8].
        n_cv: CV folds for E selection. Default 3.
    """

    def __init__(self, tau: int = 1, E_range=None, n_cv: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._tau = tau
        self._E_range = E_range or [2, 3, 4, 5, 6, 7, 8]
        self._n_cv = n_cv
        self._best_E = None
        self._train_y = None
        self._horizon = None

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SugiharaSimplex",
            category=ModelCategory.CLASSICAL,
            library="pyEDM",
            year=1990,
            paper="Sugihara, May (1990). Nonlinear forecasting, Nature 344.",
            notes="Pure Simplex projection. E selected by CV; Tp iterated for multi-step.",
        )

    def fit(self, y, X=None, freq=None, **kwargs):
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq
        horizon = kwargs.get('horizon', 1)
        self._horizon = horizon
        self._best_E = _select_E_simplex(
            y, self._E_range, self._tau, horizon, self._n_cv
        )
        self._is_fitted = True
        return self

    def predict(self, horizon, X=None, **kwargs):
        self._check_fitted()
        if self._train_y is None or self._best_E is None:
            return np.full(horizon, np.nan)
        preds = np.array([
            _simplex_one_horizon(self._train_y, E=self._best_E, tau=self._tau, Tp=h)
            for h in range(1, horizon + 1)
        ])
        return preds

    def _compute_fitted_values(self):
        return None

    def get_diagnostics(self):
        return {'method': 'Sugihara Simplex (1990)',
                'tau': self._tau, 'best_E': self._best_E}

    def get_params(self):
        return {'method': 'sugihara_simplex',
                'tau': self._tau, 'E_range': self._E_range, 'n_cv': self._n_cv}


# -----------------------------------------------------------------------
# S-map
# -----------------------------------------------------------------------

@register_model
class SugiharaSMap(Forecaster):
    """Sugihara (1994) S-map — locally-weighted linear regression on embeddings.

    For each query state in embedding space, solve a weighted linear
    regression on all library points; weights decay exponentially with
    distance from the query, rate controlled by theta. theta=0 is global
    linear regression; theta large is purely local.

    Args:
        tau: Time delay. Default 1.
        E_range: Embedding dimensions to search. Default [2,3,4,5,6,7,8].
        theta_range: Non-linearity parameter grid. Default [0,0.5,1,2,4,8].
        n_cv: CV folds. Default 3.
    """

    def __init__(self, tau: int = 1, E_range=None, theta_range=None,
                  n_cv: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._tau = tau
        self._E_range = E_range or [2, 3, 4, 5, 6, 7, 8]
        self._theta_range = theta_range or [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        self._n_cv = n_cv
        self._best_E = None
        self._best_theta = None
        self._train_y = None
        self._horizon = None

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SugiharaSMap",
            category=ModelCategory.CLASSICAL,
            library="pyEDM",
            year=1994,
            paper="Sugihara (1994). Nonlinear forecasting for classification, Phil. Trans. R. Soc. A 348.",
            notes="Pure S-map. E and theta selected by CV; Tp iterated for multi-step.",
        )

    def fit(self, y, X=None, freq=None, **kwargs):
        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq
        horizon = kwargs.get('horizon', 1)
        self._horizon = horizon
        self._best_E, self._best_theta = _select_E_theta_smap(
            y, self._E_range, self._theta_range, self._tau, horizon, self._n_cv
        )
        self._is_fitted = True
        return self

    def predict(self, horizon, X=None, **kwargs):
        self._check_fitted()
        if self._train_y is None or self._best_E is None:
            return np.full(horizon, np.nan)
        preds = np.array([
            _smap_one_horizon(self._train_y, E=self._best_E, tau=self._tau,
                              theta=self._best_theta, Tp=h)
            for h in range(1, horizon + 1)
        ])
        return preds

    def _compute_fitted_values(self):
        return None

    def get_diagnostics(self):
        return {'method': 'Sugihara S-map (1994)',
                'tau': self._tau, 'best_E': self._best_E,
                'best_theta': self._best_theta}

    def get_params(self):
        return {'method': 'sugihara_smap',
                'tau': self._tau, 'E_range': self._E_range,
                'theta_range': self._theta_range, 'n_cv': self._n_cv}
