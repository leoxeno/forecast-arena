"""
Legendre-polynomial state-space reconstruction as a forecaster.

Gibson, Farmer, Casdagli and Eubank (1992) proposed an alternative to
time-delay embedding: project a sliding window of the observed signal onto
the first d orthonormal Legendre polynomials (rescaled to [-1, +1] over the
window). The d-dimensional coordinate vector is a low-pass-filtered
reconstruction that is less sensitive to noise than raw delay coordinates.

This wrapper:
  * Embedding stage: sliding-window projection onto the first d Legendre polynomials.
  * Prediction stage: inverse-distance-weighted k-NN in the embedded space,
    the local prediction rule of Farmer and Sidorowich (1987), so the two
    reconstructions can be compared under one predictor.
  * Hyperparameters (d, window length L, k) selected by expanding-window
    cross-validation on the training data only.

Reference:
  Gibson, J.F., Farmer, J.D., Casdagli, M., Eubank, S. (1992).
  "An analytic approach to practical state space reconstruction."
  Physica D 57, 1-30.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legvander

from ..base import Forecaster, ModelMetadata, ModelCategory
from .. import register_model


def _legendre_basis(L: int, d: int) -> np.ndarray:
    """Return L x d matrix of orthonormal Legendre polynomials evaluated at
    L equally-spaced points on [-1, +1]. Columns are P_0, P_1, ..., P_{d-1}."""
    s = np.linspace(-1.0, 1.0, L)
    V = legvander(s, d - 1)  # L x d
    # Orthonormalise columns so projection coefficients are independent
    # of window length.
    Q, _ = np.linalg.qr(V)
    return Q  # L x d


def _gibson_embed(y: np.ndarray, L: int, d: int) -> np.ndarray:
    """Sliding-window Legendre embedding. Returns (N-L+1) x d coordinate matrix.

    Each row is the d-vector of projection coefficients of the window
    y[t-L+1 : t+1] onto the first d orthonormal Legendre polynomials.
    """
    y = np.asarray(y, dtype=np.float64)
    N = len(y)
    if N < L:
        return np.empty((0, d))
    Q = _legendre_basis(L, d)  # L x d
    # Use stride tricks or just loop — N is typically <10K, d <8, L <64
    # so the straightforward loop is fast and readable.
    n_windows = N - L + 1
    coords = np.empty((n_windows, d))
    for t in range(n_windows):
        window = y[t:t + L]
        coords[t] = window @ Q
    return coords


def _predict_one_horizon(y_train: np.ndarray, L: int, d: int, k: int, Tp: int) -> float:
    """Inverse-distance-weighted k-NN forecast Tp steps ahead."""
    N = len(y_train)
    # Embed first N-Tp points so that target y[t+Tp] exists in training.
    coords = _gibson_embed(y_train, L, d)      # shape: (N-L+1, d)
    if len(coords) <= k + Tp + 1:
        return float("nan")
    # Query = embedding at the last available window (ending at t=N-1).
    query = coords[-1]                         # shape: (d,)
    # Library = all earlier embeddings whose target y[t_end + Tp] is in y_train
    # t_end = L-1 + i; target index = L-1 + i + Tp; must be < N; so
    # i < N - L - Tp + 1; library = coords[:N - L - Tp + 1]
    lib_end = N - L - Tp + 1
    if lib_end <= k:
        return float("nan")
    lib = coords[:lib_end]                     # (lib_end, d)
    targets = np.array([y_train[L - 1 + i + Tp] for i in range(lib_end)])
    # k nearest by Euclidean distance in Legendre-coordinate space
    d2 = np.sum((lib - query) ** 2, axis=1)
    order = np.argsort(d2)[:k]
    dists = np.sqrt(d2[order])
    neigh = targets[order]
    # Inverse-distance weights with small epsilon
    eps = 1e-12
    w = 1.0 / (dists + eps)
    w = w / w.sum()
    return float(np.dot(w, neigh))


def _cv_select(y: np.ndarray, L_range, d_range, k_range, horizon: int, n_cv: int) -> tuple:
    """Expanding-window CV to pick (L, d, k) minimizing MAE."""
    N = len(y)
    best = (L_range[0], d_range[0], k_range[0])
    best_score = float("inf")
    fold_h = max(1, min(horizon, N // (n_cv + 2)))
    for L in L_range:
        for d in d_range:
            if d > L:
                continue
            for k in k_range:
                errs = []
                for fold in range(n_cv):
                    val_end = N - fold * fold_h
                    val_start = val_end - fold_h
                    if val_start < L + 2 * fold_h + k:
                        break
                    y_tr = y[:val_start]
                    fold_errs = []
                    for t in range(1, fold_h + 1):
                        p = _predict_one_horizon(y_tr, L, d, k, t)
                        if val_start - 1 + t < N and np.isfinite(p):
                            fold_errs.append(abs(p - y[val_start - 1 + t]))
                    if fold_errs:
                        errs.append(np.mean(fold_errs))
                if errs:
                    score = np.mean(errs)
                    if score < best_score:
                        best_score = score
                        best = (L, d, k)
    return best


@register_model
class GibsonLegendre(Forecaster):
    """Gibson-Farmer (1992) Legendre-polynomial embedding + k-NN forecast.

    An alternative to time-delay embedding in which the delay-coordinate
    vector is replaced by the projection of a sliding window onto the
    first d orthonormal Legendre polynomials. Hyperparameters (window
    length L, embedding dimension d, number of neighbours k) are chosen
    by expanding-window CV on the training data.

    Args:
        L_range: Window lengths to search. Default [16, 24, 32, 48].
        d_range: Embedding dimensions. Default [3, 4, 5, 6, 7].
        k_range: Neighbour counts. Default [3, 5, 8, 12].
        n_cv:    CV folds. Default 3.
    """

    def __init__(self, L_range=None, d_range=None, k_range=None, n_cv=3, **kwargs):
        super().__init__(**kwargs)
        self._L_range = L_range or [16, 24, 32, 48]
        self._d_range = d_range or [3, 4, 5, 6, 7]
        self._k_range = k_range or [3, 5, 8, 12]
        self._n_cv = n_cv
        self._best_L = None
        self._best_d = None
        self._best_k = None
        self._train_y = None

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GibsonLegendre",
            category=ModelCategory.CLASSICAL,
            library="forecast_arena",
            year=1992,
            paper="Gibson, Farmer, Casdagli, Eubank (1992). Physica D 57, 1.",
            notes="Legendre-polynomial embedding + IDW k-NN. Alternative to Takens delay coordinates.",
        )

    def fit(self, y, X=None, freq=None, **kwargs):
        y = self._validate_y(y)
        self._train_y = np.asarray(y, dtype=np.float64)
        self._freq = freq
        horizon = int(kwargs.get("horizon", 1))
        # Cap search on very long series for tractability
        y_gs = self._train_y[-1500:] if len(self._train_y) > 1500 else self._train_y
        self._best_L, self._best_d, self._best_k = _cv_select(
            y_gs, self._L_range, self._d_range, self._k_range, horizon, self._n_cv
        )
        self._is_fitted = True
        return self

    def predict(self, horizon, X=None, **kwargs):
        self._check_fitted()
        if self._train_y is None or self._best_L is None:
            return np.full(horizon, np.nan)
        preds = np.array([
            _predict_one_horizon(self._train_y, self._best_L, self._best_d,
                                   self._best_k, t)
            for t in range(1, horizon + 1)
        ])
        return preds

    def _compute_fitted_values(self):
        return None

    def get_diagnostics(self):
        return {"method": "Gibson-Legendre (1992)",
                "best_L": self._best_L,
                "best_d": self._best_d,
                "best_k": self._best_k}

    def get_params(self):
        return {"method": "gibson_legendre",
                "L_range": self._L_range, "d_range": self._d_range,
                "k_range": self._k_range, "n_cv": self._n_cv}
