"""Turn per-origin records into leaderboards.

Within a task, a model's score is its mean error across origins and its rank is
the position of that mean among all models that completed every origin. Across
tasks, models are summarised by mean rank, wins and podium finishes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

METRICS = ["mae", "rmse", "smape", "mase"]


def load_records(output_dir: str | Path, tasks: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read every ``<task>.jsonl`` under ``output_dir`` into one DataFrame."""
    output_dir = Path(output_dir)
    wanted = set(tasks) if tasks is not None else None
    rows = []
    for path in sorted(output_dir.glob("*.jsonl")):
        if wanted is not None and path.stem not in wanted:
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["task", "model", "split", *METRICS, "error"])
    df = pd.DataFrame(rows)
    # Keep the latest record if a (task, model, split) was written twice.
    df = df.drop_duplicates(subset=["task", "model", "split"], keep="last")
    return df.reset_index(drop=True)


def per_task_table(df: pd.DataFrame, metric: str = "mae", require_complete: bool = True) -> pd.DataFrame:
    """Mean metrics per (task, model) with a rank within each task."""
    if df.empty:
        return pd.DataFrame(columns=["task", "model", *METRICS, "n_splits", "n_errors", "rank"])
    work = df.copy()
    for m in METRICS:
        if m not in work:
            work[m] = np.nan
        work[m] = pd.to_numeric(work[m], errors="coerce")
    work["failed"] = work["error"].notna() | work[metric].isna()

    table = (
        work.groupby(["task", "model"], sort=True)
        .agg(
            mae=("mae", "mean"), rmse=("rmse", "mean"), smape=("smape", "mean"),
            mase=("mase", "mean"), n_splits=("split", "nunique"), n_errors=("failed", "sum"),
            fit_time_s=("fit_time_s", "mean"),
        )
        .reset_index()
    )
    n_expected = work.groupby("task")["split"].nunique().rename("n_expected")
    table = table.merge(n_expected, on="task", how="left")
    if require_complete:
        incomplete = (table["n_errors"] > 0) | (table["n_splits"] < table["n_expected"])
        table.loc[incomplete, METRICS] = np.nan
    table["rank"] = table.groupby("task")[metric].rank(method="min")
    table = table.sort_values(["task", "rank", "model"]).reset_index(drop=True)
    return table.drop(columns=["n_expected"])


def summarise(table: pd.DataFrame, rule: str = "mean_rank") -> pd.DataFrame:
    """One row per model across tasks: mean and median rank, wins, podiums, worst rank."""
    ranked = table.dropna(subset=["rank"])
    if ranked.empty:
        return pd.DataFrame(columns=["model", "tasks", "mean_rank", "median_rank", "wins", "podiums", "worst_rank"])
    g = ranked.groupby("model")["rank"]
    out = pd.DataFrame({
        "tasks": g.size(),
        "mean_rank": g.mean().round(2),
        "median_rank": g.median(),
        "wins": g.apply(lambda s: int((s == 1).sum())),
        "podiums": g.apply(lambda s: int((s <= 3).sum())),
        "worst_rank": g.max(),
    }).reset_index()

    if rule == "mean_rank":
        out = out.sort_values(["mean_rank", "wins"], ascending=[True, False])
    elif rule == "podium":
        out = out.sort_values(["wins", "podiums", "mean_rank"], ascending=[False, False, True])
    elif rule == "custom":
        out["score"] = composite_score(table)
        out = out.sort_values("score", ascending=False)
    else:
        raise ValueError(f"Unknown rule '{rule}'. Use 'mean_rank', 'podium' or 'custom'.")
    return out.reset_index(drop=True)


def composite_score(table: pd.DataFrame) -> pd.Series:
    """Cross-task score per model, indexed by model name (higher is better).

    TODO(user): choose the aggregation rule. ``table`` is the per-task table with
    columns task, model, mae, rmse, smape, mase, n_splits, n_errors, rank. Options
    with different incentives:

    * mean rank penalises a single collapse heavily (a model that wins ten tasks
      and fails one can lose to a model that is fourth everywhere);
    * podium counting (wins and top-three finishes) rewards peaks and ignores
      the tail;
    * Borda-style points per task (n_models minus rank) sit in between;
    * a scaled-error rule (mean log MASE) ignores ranks and compares magnitudes.

    Return a Series aligned to the models in ``table``.
    """
    raise NotImplementedError("composite_score is left for the maintainer to define; see the docstring.")


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4g", max_rows: Optional[int] = None) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table without extra dependencies."""
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        if v is None:
            return ""
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def task_leaderboard(df: pd.DataFrame, task: str, metric: str = "mae", top: Optional[int] = None) -> pd.DataFrame:
    """Leaderboard for one task, best first."""
    table = per_task_table(df, metric=metric)
    sub = table[table["task"] == task].sort_values(["rank", "model"]).reset_index(drop=True)
    cols = ["rank", "model", "mae", "rmse", "smape", "mase", "n_splits", "n_errors", "fit_time_s"]
    sub = sub[cols]
    return sub.head(top) if top else sub


__all__ = ["load_records", "per_task_table", "summarise", "composite_score", "to_markdown", "task_leaderboard"]
