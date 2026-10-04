"""Benchmark analysis: one-call pipeline from JSONL to publication-ready tables.

Usage:
    from forecast_arena.benchmarks.analysis import analyze_benchmark, export_analysis

    results = analyze_benchmark("output/results/m3/Other.jsonl", "Other")
    export_analysis(results, "output/results/m3/analysis")

Architecture:
    analyze_benchmark() → BenchmarkAnalysis dataclass (all tables pre-computed)
    Notebook cells just display attributes and create charts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forecast_arena.benchmarks.aggregation import (
    build_leaderboard,
    compute_owa,
    load_jsonl_results,
)
from forecast_arena.benchmarks.constants import M3_GROUPS, M3_NAIVE2_SMAPE, M4_GROUPS
from forecast_arena.benchmarks.exclusions import get_active_models
from forecast_arena.forecasters.metrics import (
    BootstrapResult,
    DMTestResult,
    diebold_mariano_test,
)


# ── Cross-library algorithm families ─────────────────────────────────────────

ALGORITHM_FAMILIES: dict[str, list[str]] = {
    "ARIMA": [
        "ARIMA", "DartsAutoARIMA", "SARIMAX", "SktimeARIMA",
        "SktimeAutoARIMA", "StatsForecastAutoARIMA",
    ],
    "ETS": [
        "ExponentialSmoothing", "OrbitETS", "SktimeAutoETS",
        "SktimeExponentialSmoothing", "StatsForecastAutoETS",
    ],
    "Theta": [
        "DartsTheta", "SktimeTheta", "StatsForecastAutoTheta",
        "StatsForecastDynamicOptimizedTheta", "StatsForecastOptimizedTheta",
        "StatsForecastTheta",
    ],
    "N-BEATS": ["NBEATS", "NeuralForecastNBEATS", "NeuralForecastNBEATSx"],
    "N-HiTS": ["NHiTS", "NeuralForecastNHITS"],
    "Prophet": ["DartsProphet", "Prophet", "SktimeProphet"],
    "DeepAR": ["DeepAR", "NeuralForecastDeepAR"],
    "PatchTST": ["NeuralForecastPatchTST", "GluonTSPatchTST"],
    "TFT": ["NeuralForecastTFT", "GluonTSTFT"],
    "Moirai": ["Moirai", "Moirai2", "MoiraiMoE"],
    "TimesFM": ["TimesFM", "TimesFM25"],
    "Chronos": ["Chronos", "Chronos2"],
    "TBATS": ["StatsForecastTBATS", "DartsTBATS"],
    "Naive": [
        "SktimeNaiveForecaster", "StatsForecastNaive",
        "StatsForecastSeasonalNaive",
    ],
}


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class BenchmarkAnalysis:
    """All pre-computed analysis results from a single benchmark run."""
    group: str
    competition: str
    df_raw: pd.DataFrame
    df_active: pd.DataFrame
    n_total_records: int
    n_active_records: int
    data_quality: pd.DataFrame
    naive2_smape: float
    naive2_mase: float
    naive2_published_smape: float | None
    leaderboard: pd.DataFrame
    category_summary: pd.DataFrame
    cross_library: pd.DataFrame
    dm_tests: pd.DataFrame | None = None
    bootstrap_cis: pd.DataFrame | None = None
    rank_distribution: pd.DataFrame | None = None
    foundation_summary: pd.DataFrame | None = None
    fitted_coverage: pd.DataFrame | None = None


# ── Metadata enrichment ──────────────────────────────────────────────────────

def _load_registry() -> dict[str, dict]:
    """Load model_registry.yaml, searching upward from this file."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        candidate = parent / "model_registry.yaml"
        if candidate.exists():
            with open(candidate) as f:
                data = yaml.safe_load(f) or {}
            # Registry has a top-level 'models' key
            return data.get("models", data)
    return {}


def enrich_with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add category and library columns from model_registry.yaml."""
    registry = _load_registry()
    df = df.copy()
    df["category"] = df["model_name"].map(
        lambda m: registry.get(m, {}).get("category", "unknown")
    )
    df["library"] = df["model_name"].map(
        lambda m: registry.get(m, {}).get("library", "unknown")
    )
    # Special-case Naive2 (not in registry)
    df.loc[df["model_name"] == "Naive2", "category"] = "baseline"
    df.loc[df["model_name"] == "Naive2", "library"] = "forecast_arena"
    return df


# ── Data quality ─────────────────────────────────────────────────────────────

def compute_data_quality(df: pd.DataFrame, n_expected: int) -> pd.DataFrame:
    """Per-model completeness and error rates."""
    rows = []
    for model, sub in df.groupby("model_name"):
        n_total = len(sub)
        is_error = sub["error"].notna() & (sub["error"] != "")
        n_error = is_error.sum()
        n_success = n_total - n_error
        top_error = (
            sub.loc[is_error, "error"].value_counts().index[0][:80]
            if n_error > 0 else ""
        )
        rows.append({
            "model_name": model,
            "n_records": n_total,
            "n_success": n_success,
            "n_error": n_error,
            "success_rate": round(n_success / n_total * 100, 1) if n_total > 0 else 0,
            "completeness": round(n_total / n_expected * 100, 1) if n_expected > 0 else 0,
            "top_error": top_error,
        })
    return pd.DataFrame(rows).sort_values("success_rate", ascending=False).reset_index(drop=True)


# ── Naive2 validation ────────────────────────────────────────────────────────

def validate_naive2(
    df: pd.DataFrame, group: str, competition: str = "M3",
) -> tuple[float, float, float | None]:
    """Extract Naive2 metrics and compare against published values.

    Returns (naive2_smape, naive2_mase, published_smape_or_None).
    """
    naive2 = df[df["model_name"] == "Naive2"]
    successful = naive2[naive2["error"].isna() | (naive2["error"] == "")]

    if len(successful) == 0:
        raise ValueError(f"No successful Naive2 records for group={group}")

    naive2_smape = successful["smape"].mean()
    naive2_mase = successful["mase"].mean()

    published = None
    if competition == "M3":
        published = M3_NAIVE2_SMAPE.get(group)

    return naive2_smape, naive2_mase, published


# ── Category summary ─────────────────────────────────────────────────────────

def compute_category_summary(
    leaderboard: pd.DataFrame, df_active: pd.DataFrame,
) -> pd.DataFrame:
    """Per-category aggregates: n_models, mean metrics, best model."""
    if "category" not in leaderboard.columns:
        return pd.DataFrame()

    rows = []
    for cat, sub in leaderboard.groupby("category"):
        valid = sub.dropna(subset=["smape"])
        rows.append({
            "category": cat,
            "n_models": len(sub),
            "mean_smape": round(valid["smape"].mean(), 2) if len(valid) > 0 else None,
            "mean_mase": round(valid["mase"].mean(), 4) if len(valid) > 0 else None,
            "mean_owa": round(valid["owa"].mean(), 4) if len(valid) > 0 else None,
            "best_model": valid.iloc[0]["model_name"] if len(valid) > 0 else None,
            "best_smape": valid.iloc[0]["smape"] if len(valid) > 0 else None,
        })
    return pd.DataFrame(rows).sort_values("mean_smape", na_position="last").reset_index(drop=True)


# ── Cross-library comparison ─────────────────────────────────────────────────

def compute_cross_library(
    leaderboard: pd.DataFrame, df_active: pd.DataFrame,
) -> pd.DataFrame:
    """Compare implementations of the same algorithm across libraries."""
    rows = []
    for family, members in ALGORITHM_FAMILIES.items():
        family_models = leaderboard[leaderboard["model_name"].isin(members)]
        if len(family_models) < 2:
            continue
        for _, row in family_models.iterrows():
            time_cols = ["mean_fit_time", "mean_predict_time"]
            total_time = sum(row.get(c, 0) or 0 for c in time_cols)
            rows.append({
                "family": family,
                "model_name": row["model_name"],
                "rank": row["rank"],
                "smape": row["smape"],
                "mase": row["mase"],
                "owa": row["owa"],
                "total_time": round(total_time, 2),
                "n_series": row["n_series"],
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values(["family", "smape"], na_position="last")

    # Add speed ratio within each family (slowest / fastest)
    speed_ratios = []
    for _, grp in result.groupby("family"):
        times = grp["total_time"]
        min_t = times.min()
        for _, r in grp.iterrows():
            ratio = r["total_time"] / min_t if min_t > 0 else 1.0
            speed_ratios.append(round(ratio, 1))
    result["speed_ratio"] = speed_ratios

    return result.reset_index(drop=True)


# ── DM tests ─────────────────────────────────────────────────────────────────

def _extract_raw_forecasts(
    df: pd.DataFrame, model_a: str, model_b: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract concatenated raw (y_test, forecast_a, forecast_b) for shared series.

    Returns None if insufficient overlap.
    """
    a = df[
        (df["model_name"] == model_a) &
        (df["error"].isna() | (df["error"] == "")) &
        df["forecast"].notna()
    ].set_index("series_id")

    b = df[
        (df["model_name"] == model_b) &
        (df["error"].isna() | (df["error"] == "")) &
        df["forecast"].notna()
    ].set_index("series_id")

    shared = sorted(set(a.index) & set(b.index))
    if len(shared) < 10:
        return None

    y_true_all, pred_a_all, pred_b_all = [], [], []
    for sid in shared:
        y_test = np.array(a.loc[sid, "y_test"], dtype=float)
        fa = np.array(a.loc[sid, "forecast"], dtype=float)
        fb = np.array(b.loc[sid, "forecast"], dtype=float)
        # Ensure all same length (trim to min)
        min_len = min(len(y_test), len(fa), len(fb))
        if min_len == 0:
            continue
        y_true_all.append(y_test[:min_len])
        pred_a_all.append(fa[:min_len])
        pred_b_all.append(fb[:min_len])

    if not y_true_all:
        return None

    return (
        np.concatenate(y_true_all),
        np.concatenate(pred_a_all),
        np.concatenate(pred_b_all),
    )


def compute_dm_tests(
    df: pd.DataFrame,
    leaderboard: pd.DataFrame,
    baseline: str = "Naive2",
    n_vs_baseline: int = 10,
    n_pairwise: int = 5,
    horizon: int = 8,
) -> pd.DataFrame:
    """Run Diebold-Mariano tests: top N vs baseline + pairwise top M."""
    results = []
    valid_lb = leaderboard.dropna(subset=["smape"])
    top_models = valid_lb["model_name"].head(max(n_vs_baseline, n_pairwise)).tolist()

    # Top N vs baseline
    for model in top_models[:n_vs_baseline]:
        raw = _extract_raw_forecasts(df, model, baseline)
        if raw is None:
            results.append({
                "test_type": "vs_baseline",
                "model_a": model,
                "model_b": baseline,
                "dm_stat": None,
                "p_value": None,
                "significant": None,
                "better": None,
                "n_obs": 0,
            })
            continue
        y_true, pred_a, pred_b = raw
        dm = diebold_mariano_test(y_true, pred_a, pred_b, loss="mae", h=horizon)
        results.append({
            "test_type": "vs_baseline",
            "model_a": model,
            "model_b": baseline,
            "dm_stat": round(dm.statistic, 4) if not np.isnan(dm.statistic) else None,
            "p_value": round(dm.pvalue, 6) if not np.isnan(dm.pvalue) else None,
            "significant": dm.significant,
            "better": model if dm.better_model == "1" else (
                baseline if dm.better_model == "2" else "neither"
            ),
            "n_obs": len(y_true),
        })

    # Pairwise among top M
    pairwise_models = top_models[:n_pairwise]
    for i, m_a in enumerate(pairwise_models):
        for m_b in pairwise_models[i + 1:]:
            raw = _extract_raw_forecasts(df, m_a, m_b)
            if raw is None:
                results.append({
                    "test_type": "pairwise",
                    "model_a": m_a,
                    "model_b": m_b,
                    "dm_stat": None,
                    "p_value": None,
                    "significant": None,
                    "better": None,
                    "n_obs": 0,
                })
                continue
            y_true, pred_a, pred_b = raw
            dm = diebold_mariano_test(y_true, pred_a, pred_b, loss="mae", h=horizon)
            results.append({
                "test_type": "pairwise",
                "model_a": m_a,
                "model_b": m_b,
                "dm_stat": round(dm.statistic, 4) if not np.isnan(dm.statistic) else None,
                "p_value": round(dm.pvalue, 6) if not np.isnan(dm.pvalue) else None,
                "significant": dm.significant,
                "better": m_a if dm.better_model == "1" else (
                    m_b if dm.better_model == "2" else "neither"
                ),
                "n_obs": len(y_true),
            })

    return pd.DataFrame(results)


# ── Bootstrap CIs (metric vector, NOT block bootstrap) ───────────────────────

def metric_vector_bootstrap(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap CI for the mean of independent metric values (e.g., per-series sMAPE).

    Series are independent → i.i.d. bootstrap (block_size=1).
    This is NOT block_bootstrap_ci (which resamples temporal observations).
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    point = float(np.mean(values))
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    return BootstrapResult(
        point_estimate=point,
        ci_lower=float(np.percentile(means, 100 * alpha / 2)),
        ci_upper=float(np.percentile(means, 100 * (1 - alpha / 2))),
        std_error=float(np.std(means)),
        n_bootstrap=n_bootstrap,
    )


def compute_bootstrap_cis(
    df: pd.DataFrame,
    models: list[str],
    metrics: tuple[str, ...] | list[str] = ("smape", "mase"),
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute bootstrap CIs for per-series metric vectors of each model."""
    successful = df[(df["error"].isna()) | (df["error"] == "")]
    rows = []

    for model in models:
        model_data = successful[successful["model_name"] == model]
        if len(model_data) == 0:
            continue
        for metric in metrics:
            if metric not in model_data.columns:
                continue
            values = model_data[metric].dropna().values
            if len(values) < 5:
                continue
            bs = metric_vector_bootstrap(values, n_bootstrap=n_bootstrap, seed=seed)
            rows.append({
                "model_name": model,
                "metric": metric,
                "point_estimate": round(bs.point_estimate, 4),
                "ci_lower": round(bs.ci_lower, 4),
                "ci_upper": round(bs.ci_upper, 4),
                "std_error": round(bs.std_error, 4),
                "n_series": len(values),
            })

    return pd.DataFrame(rows)


# ── Rank distribution ────────────────────────────────────────────────────────

def compute_rank_distribution(
    df: pd.DataFrame, metric: str = "smape",
) -> pd.DataFrame:
    """Per-series ranking statistics for each model."""
    successful = df[(df["error"].isna()) | (df["error"] == "")].copy()
    if metric not in successful.columns:
        return pd.DataFrame()

    # Rank each model per series
    successful["rank"] = successful.groupby("series_id")[metric].rank(method="min")
    n_series_total = successful["series_id"].nunique()

    rows = []
    for model, sub in successful.groupby("model_name"):
        ranks = sub["rank"]
        rows.append({
            "model_name": model,
            "mean_rank": round(ranks.mean(), 1),
            "median_rank": round(ranks.median(), 1),
            "n_wins": int((ranks == 1).sum()),
            "n_top3": int((ranks <= 3).sum()),
            "n_top10": int((ranks <= 10).sum()),
            "n_series": len(sub),
        })

    return pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)


# ── Foundation model deep-dive ───────────────────────────────────────────────

def compute_foundation_summary(
    df: pd.DataFrame, leaderboard: pd.DataFrame,
) -> pd.DataFrame:
    """Foundation model metrics with train-length correlation."""
    if "category" not in leaderboard.columns:
        return pd.DataFrame()

    foundation = leaderboard[leaderboard["category"] == "llm"].copy()
    if len(foundation) == 0:
        return pd.DataFrame()

    successful = df[(df["error"].isna()) | (df["error"] == "")]

    rows = []
    for _, row in foundation.iterrows():
        model = row["model_name"]
        model_data = successful[successful["model_name"] == model]

        # Train-length vs sMAPE correlation
        corr = None
        if len(model_data) > 5 and "y_train_len" in model_data.columns:
            valid = model_data.dropna(subset=["smape", "y_train_len"])
            if len(valid) > 5:
                from scipy.stats import spearmanr
                corr_val, p_val = spearmanr(valid["y_train_len"], valid["smape"])
                corr = round(corr_val, 3)

        rows.append({
            "model_name": model,
            "rank": row["rank"],
            "smape": row["smape"],
            "mase": row["mase"],
            "owa": row["owa"],
            "n_series": row["n_series"],
            "train_len_corr": corr,
        })

    return pd.DataFrame(rows).sort_values("smape", na_position="last").reset_index(drop=True)


# ── Fitted values coverage ───────────────────────────────────────────────────

def compute_fitted_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Which models provide fitted values, quantiles, and diagnostics."""
    successful = df[(df["error"].isna()) | (df["error"] == "")]

    rows = []
    for model, sub in successful.groupby("model_name"):
        has_fitted = sub["fitted_values"].notna().any()
        has_quantiles = sub["prediction_quantiles"].notna().any()
        has_diagnostics = sub["diagnostics"].notna().any()
        rows.append({
            "model_name": model,
            "has_fitted": has_fitted,
            "has_quantiles": has_quantiles,
            "has_diagnostics": has_diagnostics,
            "n_series": len(sub),
        })

    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


# ── Main entry point ─────────────────────────────────────────────────────────

def analyze_benchmark(
    jsonl_path: str | Path,
    group: str,
    competition: str = "M3",
    run_statistical_tests: bool = True,
    n_dm_top: int = 10,
    n_dm_pairwise: int = 5,
    n_bootstrap_models: int = 20,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BenchmarkAnalysis:
    """Run the full analysis pipeline on a benchmark JSONL file.

    Parameters
    ----------
    jsonl_path : path to JSONL results file
    group : frequency group ("Yearly", "Quarterly", "Monthly", "Other")
    competition : "M3" or "M4"
    run_statistical_tests : whether to compute DM tests and bootstrap CIs
    n_dm_top : number of top models to test vs Naive2
    n_dm_pairwise : number of top models for pairwise DM tests
    n_bootstrap_models : number of top models for bootstrap CIs
    n_bootstrap : number of bootstrap resamples
    seed : random seed for reproducibility
    """
    groups = M3_GROUPS if competition == "M3" else M4_GROUPS
    freq_group = groups[group]

    # Load and enrich
    df_raw = load_jsonl_results(jsonl_path)
    df_raw = enrich_with_metadata(df_raw)

    # Filter to active models + Naive2
    active_set = set(get_active_models())
    active_set.add("Naive2")  # Always include baseline
    df_active = df_raw[df_raw["model_name"].isin(active_set)].copy()

    # Data quality
    data_quality = compute_data_quality(df_active, n_expected=freq_group.n_series)

    # Naive2 validation
    naive2_smape, naive2_mase, published = validate_naive2(
        df_active, group, competition
    )

    # Leaderboard (all active models including Naive2)
    leaderboard = build_leaderboard(df_active, naive2_smape, naive2_mase)
    leaderboard = enrich_with_metadata(leaderboard)

    # Category summary
    category_summary = compute_category_summary(leaderboard, df_active)

    # Cross-library
    cross_library = compute_cross_library(leaderboard, df_active)

    # Statistical tests
    dm_tests = None
    bootstrap_cis = None
    rank_dist = None
    foundation = None
    fitted_cov = None

    if run_statistical_tests:
        dm_tests = compute_dm_tests(
            df_active, leaderboard, baseline="Naive2",
            n_vs_baseline=n_dm_top, n_pairwise=n_dm_pairwise,
            horizon=freq_group.horizon,
        )

        top_models = (
            leaderboard.dropna(subset=["smape"])
            ["model_name"].head(n_bootstrap_models).tolist()
        )
        bootstrap_cis = compute_bootstrap_cis(
            df_active, top_models,
            n_bootstrap=n_bootstrap, seed=seed,
        )

    # Rank distribution (always compute — cheap)
    rank_dist = compute_rank_distribution(df_active)

    # Foundation deep-dive
    foundation = compute_foundation_summary(df_active, leaderboard)

    # Fitted coverage
    fitted_cov = compute_fitted_coverage(df_active)

    return BenchmarkAnalysis(
        group=group,
        competition=competition,
        df_raw=df_raw,
        df_active=df_active,
        n_total_records=len(df_raw),
        n_active_records=len(df_active),
        data_quality=data_quality,
        naive2_smape=naive2_smape,
        naive2_mase=naive2_mase,
        naive2_published_smape=published,
        leaderboard=leaderboard,
        category_summary=category_summary,
        cross_library=cross_library,
        dm_tests=dm_tests,
        bootstrap_cis=bootstrap_cis,
        rank_distribution=rank_dist,
        foundation_summary=foundation,
        fitted_coverage=fitted_cov,
    )


# ── Export ────────────────────────────────────────────────────────────────────

def export_analysis(analysis: BenchmarkAnalysis, output_dir: str | Path) -> list[Path]:
    """Write all analysis tables as CSVs. Returns list of written paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = analysis.group

    written = []
    tables = {
        "leaderboard": analysis.leaderboard,
        "data_quality": analysis.data_quality,
        "category_summary": analysis.category_summary,
        "cross_library": analysis.cross_library,
        "dm_tests": analysis.dm_tests,
        "bootstrap_cis": analysis.bootstrap_cis,
        "rank_distribution": analysis.rank_distribution,
        "foundation_summary": analysis.foundation_summary,
        "fitted_coverage": analysis.fitted_coverage,
    }

    for name, df in tables.items():
        if df is not None and len(df) > 0:
            path = out / f"{prefix}_{name}.csv"
            df.to_csv(path, index=False)
            written.append(path)

    return written
