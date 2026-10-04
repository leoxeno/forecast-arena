"""Rolling-origin split generation.

The arena evaluates every model on the same expanding-window splits of a
series. Origins are spaced evenly between a minimum and a maximum training
fraction, and the test window keeps one length across origins so that errors
are comparable from split to split, as in the M-competitions.

References:
    Tashman (2000), "Out-of-sample tests of forecasting accuracy".
    Hyndman and Athanasopoulos (2021), "Forecasting: Principles and Practice".
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def generate_rolling_origins(
    n: int,
    min_train_frac: float = 0.5,
    max_train_frac: float = 0.85,
    n_origins: int = 15,
    min_test_len: int = 12,
) -> List[Tuple[int, int]]:
    """Return expanding-window ``(n_train, n_test)`` pairs for a series of length ``n``.

    Each split trains on ``y[:n_train]`` and tests on ``y[n_train:n_train + n_test]``.
    The test length is the remainder after the largest training set, so every
    origin forecasts the same number of steps.

    Args:
        n: Series length.
        min_train_frac: Training fraction at the first origin.
        max_train_frac: Training fraction at the last origin.
        n_origins: Number of origins. With one origin the split is 70/30.
        min_test_len: Smallest test window allowed.
    """
    if n < 2 * min_test_len:
        raise ValueError(f"Series too short for rolling origins: n={n}, min_test_len={min_test_len}")

    min_train = max(int(n * min_train_frac), 30)
    max_train = min(int(n * max_train_frac), n - min_test_len)

    if max_train <= min_train:
        max_train = n - min_test_len
        min_train = min(min_train, max_train - 1)

    test_len = n - max_train
    if test_len < min_test_len:
        test_len = min_test_len
        max_train = n - test_len

    if n_origins == 1:
        origin = int(n * 0.7)
        return [(origin, n - origin)]

    origins = sorted(set(np.linspace(min_train, max_train, n_origins, dtype=int).tolist()))
    return [(int(origin), int(test_len)) for origin in origins]


def describe_splits(n: int, splits: List[Tuple[int, int]]) -> pd.DataFrame:
    """Tabulate the splits: origin index, train and test sizes, train fraction."""
    rows = [
        {
            "split": i,
            "n_train": n_train,
            "n_test": n_test,
            "train_frac": round(n_train / n, 3),
            "test_end": n_train + n_test,
        }
        for i, (n_train, n_test) in enumerate(splits)
    ]
    return pd.DataFrame(rows)


__all__ = ["generate_rolling_origins", "describe_splits"]
