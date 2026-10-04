"""Benchmark result aggregation: leaderboards, OWA, per-frequency tables.

CRITICAL — OWA aggregation order (GluonTS got this wrong):
  1. sMAPE per series → average across ALL series → one number
  2. MASE per series → average across ALL series → one number
  3. Divide each by Naive2's corresponding overall average
  4. OWA = 0.5 × (ratio_smape + ratio_mase)

Do NOT compute per-series OWA then average — that gives incorrect results.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_jsonl_results(path: str | Path) -> pd.DataFrame:
    """Load benchmark results from a JSONL file into a DataFrame.

    Each line is a JSON object with keys: series_id, model_name, group,
    smape, mase, fit_time_seconds, predict_time_seconds, error, etc.
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def aggregate_by_group(
    df: pd.DataFrame,
    group_col: str = "group",
    metrics: tuple[str, ...] = ("smape", "mase"),
) -> pd.DataFrame:
    """Per-frequency group averages for each model.

    Returns a DataFrame with columns: model_name, group, smape, mase, n_series.
    """
    rows = []
    for (model, grp), sub in df.groupby(["model_name", group_col]):
        successful = sub[sub["error"].isna() | (sub["error"] == "")]
        row = {"model_name": model, "group": grp, "n_series": len(successful)}
        for m in metrics:
            if m in successful.columns:
                row[m] = successful[m].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compute_owa(
    model_smape: float,
    model_mase: float,
    naive2_smape: float,
    naive2_mase: float,
) -> float:
    """Compute Overall Weighted Average (OWA) relative to Naive2.

    OWA = 0.5 × (model_smape / naive2_smape) + 0.5 × (model_mase / naive2_mase)

    IMPORTANT: model_smape and naive2_smape must already be averaged across
    ALL series (not per-group averages). Same for MASE.
    """
    if naive2_smape == 0 or naive2_mase == 0:
        return float("inf")
    return 0.5 * (model_smape / naive2_smape) + 0.5 * (model_mase / naive2_mase)


def build_leaderboard(
    df: pd.DataFrame,
    naive2_smape: float,
    naive2_mase: float,
    sort_by: str = "smape",
) -> pd.DataFrame:
    """Build a ranked leaderboard from benchmark results.

    Parameters
    ----------
    df : pd.DataFrame
        Raw results with columns: model_name, smape, mase, error,
        fit_time_seconds, predict_time_seconds.
    naive2_smape : float
        Naive2 overall average sMAPE (for OWA computation).
    naive2_mase : float
        Naive2 overall average MASE (for OWA computation).
    sort_by : str
        Column to sort by. Use "smape" for M3, "owa" for M4.

    Returns
    -------
    pd.DataFrame
        Ranked leaderboard with columns: rank, model_name, smape, mase, owa,
        n_series, success_rate, mean_fit_time, mean_predict_time.
    """
    rows = []
    for model, sub in df.groupby("model_name"):
        total = len(sub)
        successful = sub[sub["error"].isna() | (sub["error"] == "")]
        n_success = len(successful)

        avg_smape = successful["smape"].mean() if n_success > 0 else float("nan")
        avg_mase = successful["mase"].mean() if n_success > 0 else float("nan")
        owa = compute_owa(avg_smape, avg_mase, naive2_smape, naive2_mase)

        rows.append({
            "model_name": model,
            "smape": round(avg_smape, 2) if not np.isnan(avg_smape) else None,
            "mase": round(avg_mase, 4) if not np.isnan(avg_mase) else None,
            "owa": round(owa, 4) if not np.isinf(owa) and not np.isnan(owa) else None,
            "n_series": n_success,
            "success_rate": round(n_success / total * 100, 1) if total > 0 else 0,
            "mean_fit_time": round(successful["fit_time_seconds"].mean(), 2) if n_success > 0 else None,
            "mean_predict_time": round(successful["predict_time_seconds"].mean(), 4) if n_success > 0 else None,
        })

    lb = pd.DataFrame(rows)
    if sort_by in lb.columns:
        lb = lb.sort_values(sort_by, ascending=True, na_position="last")
    lb.insert(0, "rank", range(1, len(lb) + 1))
    return lb.reset_index(drop=True)


def build_per_frequency_table(
    df: pd.DataFrame,
    metric: str = "smape",
    group_order: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build a per-frequency metric table (publication Table 1 format).

    Rows = models, columns = frequency groups + "Overall".

    Parameters
    ----------
    df : pd.DataFrame
        Raw results with columns: model_name, group, smape/mase, error.
    metric : str
        Which metric to tabulate ("smape" or "mase").
    group_order : list[str], optional
        Column order for groups. Default: sorted.

    Returns
    -------
    pd.DataFrame
        Pivot table with models as rows, groups + "Overall" as columns.
    """
    successful = df[df["error"].isna() | (df["error"] == "")].copy()

    # Per-group averages
    pivot = successful.pivot_table(
        values=metric, index="model_name", columns="group", aggfunc="mean"
    )

    # Overall = average across ALL series (not average of group averages)
    overall = successful.groupby("model_name")[metric].mean()
    pivot["Overall"] = overall

    if group_order:
        cols = [c for c in group_order if c in pivot.columns] + ["Overall"]
        pivot = pivot[cols]

    return pivot.round(2).sort_values("Overall")
