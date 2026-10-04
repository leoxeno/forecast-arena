"""The arena loop: every model, every task, every rolling origin, one protocol.

Each (task, model, origin) evaluation is written to ``<output_dir>/<task>.jsonl``
the moment it finishes, with the data hash, the protocol, the package version
and the wall time. A re-run with ``resume=True`` skips what is already on disk,
so a long sweep survives interruptions and can be spread over machines by
merging the files.
"""

from __future__ import annotations

import json
import random
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ._version import __version__
from .leaderboard import load_records
from .runner import run_single_model
from .splits import generate_rolling_origins
from .tasks import get_task, series_sha256


@dataclass(frozen=True)
class Protocol:
    """The evaluation design shared by every model in a run."""

    n_origins: int = 15
    min_train_frac: float = 0.5
    max_train_frac: float = 0.85
    min_test_len: int = 12
    seed: int = 0

    def splits(self, n: int) -> List[Tuple[int, int]]:
        return generate_rolling_origins(
            n, self.min_train_frac, self.max_train_frac, self.n_origins, self.min_test_len
        )


def set_global_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and, when present, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def compute_metrics(y_train: np.ndarray, y_test: np.ndarray, forecast: np.ndarray,
                    seasonality: int = 1) -> Dict[str, Optional[float]]:
    """MAE, RMSE, sMAPE and MASE (scaled by the in-sample seasonal naive error)."""
    f = np.asarray(forecast, dtype=float).flatten()[: len(y_test)]
    if len(f) < len(y_test):
        f = np.concatenate([f, np.full(len(y_test) - len(f), np.nan)])
    valid = ~(np.isnan(y_test) | np.isnan(f))
    empty = {"mae": None, "rmse": None, "smape": None, "mase": None, "valid_points": int(valid.sum())}
    if valid.sum() == 0:
        return empty

    yt, fp = y_test[valid], f[valid]
    err = yt - fp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom = np.abs(yt) + np.abs(fp)
    nz = denom > 1e-10
    smape = float(np.mean(2.0 * np.abs(err[nz]) / denom[nz]) * 100) if nz.any() else None

    m = max(int(seasonality), 1)
    naive_err = np.abs(y_train[m:] - y_train[:-m]) if len(y_train) > m else np.array([])
    naive_mae = float(np.mean(naive_err)) if naive_err.size else np.nan
    mase = float(mae / naive_mae) if naive_mae > 0 else None

    return {
        "mae": round(mae, 6), "rmse": round(rmse, 6),
        "smape": round(smape, 4) if smape is not None else None,
        "mase": round(mase, 6) if mase is not None else None,
        "valid_points": int(valid.sum()),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def completed_pairs(path: Path) -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    if not path.exists():
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["model"], int(rec["split"])))
            except Exception:
                continue
    return done


def run_arena(
    tasks: Sequence[str],
    models: Sequence[str],
    *,
    protocol: Protocol = Protocol(),
    output_dir: str | Path = "results",
    resume: bool = True,
    verbose: bool = True,
    store_forecasts: bool = True,
    model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Evaluate ``models`` on ``tasks`` under ``protocol`` and return all records."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs = model_kwargs or {}

    for task_id in tasks:
        spec = get_task(task_id)
        y = spec.load()
        splits = protocol.splits(len(y))
        digest = series_sha256(y)
        path = output_dir / f"{task_id}.jsonl"
        done = completed_pairs(path) if resume else set()
        todo = [(m, i) for m in models for i in range(len(splits)) if (m, i) not in done]

        if verbose:
            print(f"\n== {task_id}: n={len(y)}, origins={len(splits)}, "
                  f"test_len={splits[0][1]}, evaluations={len(todo)} "
                  f"(skipping {len(models) * len(splits) - len(todo)} done)")

        t_task = time.perf_counter()
        with open(path, "a") as fh:
            for k, (model, si) in enumerate(todo, 1):
                n_train, n_test = splits[si]
                y_train, y_test = y[:n_train], y[n_train:n_train + n_test]
                set_global_seed(protocol.seed)
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mr = run_single_model(model, y_train, horizon=n_test, freq=spec.freq,
                                          **model_kwargs.get(model, {}))
                wall = time.perf_counter() - t0
                if mr.ok and not np.isfinite(mr.forecast[: n_test]).all():
                    bad = int((~np.isfinite(mr.forecast[: n_test])).sum())
                    mr.error = f"NonFiniteForecast: {bad} of {n_test} values are NaN or inf"
                if mr.ok and len(mr.forecast) < n_test:
                    mr.error = f"ShortForecast: {len(mr.forecast)} values returned for horizon {n_test}"
                metrics = (compute_metrics(y_train, y_test, mr.forecast, spec.seasonality)
                           if mr.ok else {"mae": None, "rmse": None, "smape": None,
                                          "mase": None, "valid_points": 0})
                rec = {
                    "task": task_id, "model": model, "split": si,
                    "n_train": int(n_train), "n_test": int(n_test),
                    **metrics,
                    "error": mr.error,
                    "fit_time_s": round(mr.fit_time_seconds, 4),
                    "predict_time_s": round(mr.predict_time_seconds, 4),
                    "elapsed_s": round(wall, 4),
                    "library": (mr.metadata or {}).get("library"),
                    "category": (mr.metadata or {}).get("category"),
                    "params": mr.params,
                    "diagnostics": mr.diagnostics,
                    "forecast": mr.forecast if (store_forecasts and mr.ok) else None,
                    "data_sha256": digest,
                    "protocol": asdict(protocol),
                    "arena_version": __version__,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                fh.write(json.dumps(_jsonable(rec)) + "\n")
                fh.flush()

                if verbose:
                    status = (f"MAE={metrics['mae']:.4g}" if metrics["mae"] is not None
                              else f"ERR {str(mr.error)[:50]}")
                    print(f"  [{k:>4}/{len(todo)}] {model:<32} origin {si:>2}  {status}  ({wall:.2f}s)")

        if verbose:
            print(f"   done in {time.perf_counter() - t_task:.1f}s -> {path}")

    return load_records(output_dir, tasks=list(tasks))


__all__ = ["Protocol", "run_arena", "compute_metrics", "set_global_seed", "completed_pairs"]
