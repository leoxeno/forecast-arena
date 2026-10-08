# Modified for forecast-arena: extracted and adapted from AION benchmark utilities, October 2026.
"""Multi-split robustness evaluation framework — PNAS/Nature/Science grade.

Implements the statistical validation protocol for forecasting benchmarks:
1. Rolling-origin (expanding window) evaluation across multiple train/test splits
2. Diebold-Mariano test for pairwise statistical significance
3. Bootstrap confidence intervals on MAE
4. Friedman test + Nemenyi post-hoc for multi-model comparison
5. Summary statistics: mean +/- std, median, CI across splits

Protocol follows M-competition standards (Makridakis et al.), NeurIPS
reproducibility requirements, and PNAS statistical reporting guidelines.

Usage:
    from forecast_arena.benchmarks.robustness import (
        generate_rolling_origins,
        evaluate_model_multi_split,
        diebold_mariano_test,
        bootstrap_ci,
        friedman_nemenyi_test,
        RobustnessReport,
    )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Rolling-origin split generation ───────────────────────────────────────

# The split generator lives in forecast_arena.splits; re-exported here for callers of this module.
from ..splits import generate_rolling_origins  # noqa: E402,F401


# ── Core evaluation ──────────────────────────────────────────────────────

def evaluate_model_multi_split(
    model_class,
    y_full: np.ndarray,
    splits: List[Tuple[int, int]],
    freq: Optional[str] = None,
    model_kwargs: Optional[dict] = None,
) -> Dict:
    """Evaluate a model across multiple rolling-origin splits.

    Args:
        model_class: Forecaster class (or callable returning a Forecaster).
        y_full: Full time series array.
        splits: List of (n_train, n_test) from generate_rolling_origins().
        freq: Frequency string for the model.
        model_kwargs: Additional kwargs for model constructor.

    Returns:
        Dict with keys: 'maes', 'rmses', 'forecasts', 'split_details'.
    """
    model_kwargs = model_kwargs or {}
    maes = []
    rmses = []
    forecasts = []
    split_details = []

    for n_train, n_test in splits:
        y_train = y_full[:n_train]
        y_test = y_full[n_train:n_train + n_test]
        horizon = len(y_test)

        if horizon < 1:
            continue

        try:
            model = model_class(**model_kwargs)
            model.fit(y_train, freq=freq, horizon=horizon)
            fc = model.predict(horizon)
            fc = np.asarray(fc).flatten()[:horizon]

            valid = ~(np.isnan(y_test) | np.isnan(fc))
            if valid.sum() == 0:
                maes.append(np.nan)
                rmses.append(np.nan)
            else:
                mae = float(np.mean(np.abs(y_test[valid] - fc[valid])))
                rmse = float(np.sqrt(np.mean((y_test[valid] - fc[valid]) ** 2)))
                maes.append(mae)
                rmses.append(rmse)

            forecasts.append(fc)
            split_details.append({
                'n_train': n_train,
                'n_test': horizon,
                'mae': maes[-1],
                'rmse': rmses[-1],
            })
        except Exception as e:
            maes.append(np.nan)
            rmses.append(np.nan)
            forecasts.append(None)
            split_details.append({
                'n_train': n_train,
                'n_test': n_test,
                'mae': np.nan,
                'rmse': np.nan,
                'error': str(e),
            })

    return {
        'maes': np.array(maes),
        'rmses': np.array(rmses),
        'forecasts': forecasts,
        'split_details': split_details,
    }


# ── Diebold-Mariano test ────────────────────────────────────────────────

def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    horizon: int = 1,
    alternative: str = 'two-sided',
) -> Dict:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t) and L = absolute loss.
    Uses Harvey, Leybourne, and Newbold (1997) finite-sample correction.

    Args:
        errors_1: Forecast errors from model 1 (array of length T).
        errors_2: Forecast errors from model 2 (array of length T).
        horizon: Forecast horizon (for HAC correction).
        alternative: 'two-sided', 'less' (model 1 better), 'greater'.

    Returns:
        Dict with 'dm_stat', 'p_value', 'mean_loss_diff', 'conclusion'.
    """
    from scipy import stats

    e1 = np.asarray(errors_1).flatten()
    e2 = np.asarray(errors_2).flatten()

    # Loss differential: positive means model 2 is better
    d = np.abs(e1) - np.abs(e2)
    T = len(d)

    if T < 3:
        return {'dm_stat': np.nan, 'p_value': 1.0,
                'mean_loss_diff': float(np.mean(d)),
                'conclusion': 'insufficient_data'}

    d_bar = np.mean(d)

    # HAC variance estimator (Newey-West with h-1 lags)
    h = max(1, horizon)
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2.0 * gamma_k
    var_d = (gamma_0 + gamma_sum) / T

    if var_d <= 0:
        return {'dm_stat': np.nan, 'p_value': 1.0,
                'mean_loss_diff': float(d_bar),
                'conclusion': 'zero_variance'}

    # Harvey et al. (1997) finite-sample correction
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat = (d_bar / np.sqrt(var_d)) * correction

    # Use t-distribution with T-1 df (Harvey et al. recommendation)
    if alternative == 'two-sided':
        p_value = 2.0 * stats.t.sf(np.abs(dm_stat), df=T - 1)
    elif alternative == 'less':
        p_value = stats.t.cdf(dm_stat, df=T - 1)
    else:
        p_value = stats.t.sf(dm_stat, df=T - 1)

    if p_value < 0.01:
        conclusion = 'significant_1pct'
    elif p_value < 0.05:
        conclusion = 'significant_5pct'
    elif p_value < 0.10:
        conclusion = 'significant_10pct'
    else:
        conclusion = 'not_significant'

    return {
        'dm_stat': float(dm_stat),
        'p_value': float(p_value),
        'mean_loss_diff': float(d_bar),
        'conclusion': conclusion,
    }


def diebold_mariano_multi_split(
    maes_1: np.ndarray,
    maes_2: np.ndarray,
) -> Dict:
    """DM-like test using MAEs across rolling origins as paired observations.

    For multi-split robustness, we treat the MAE from each split as a paired
    observation and use a paired t-test (equivalent to DM without HAC when
    splits are non-overlapping). Also reports Wilcoxon signed-rank test.

    Args:
        maes_1: MAE values from model 1 across splits.
        maes_2: MAE values from model 2 across splits.

    Returns:
        Dict with paired t-test and Wilcoxon results.
    """
    from scipy import stats

    m1 = np.asarray(maes_1)
    m2 = np.asarray(maes_2)
    valid = np.isfinite(m1) & np.isfinite(m2)
    m1 = m1[valid]
    m2 = m2[valid]

    n = len(m1)
    if n < 3:
        return {
            'paired_t_stat': np.nan, 'paired_t_pvalue': 1.0,
            'wilcoxon_stat': np.nan, 'wilcoxon_pvalue': 1.0,
            'n_splits': n, 'mean_diff': np.nan,
            'conclusion': 'insufficient_data',
        }

    diffs = m1 - m2  # positive = model 2 better

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(m1, m2)

    # Wilcoxon signed-rank (non-parametric alternative)
    try:
        w_stat, w_pval = stats.wilcoxon(diffs, alternative='two-sided')
    except ValueError:
        w_stat, w_pval = np.nan, 1.0

    mean_diff = float(np.mean(diffs))

    # Effect size: Cohen's d for paired samples
    d_std = np.std(diffs, ddof=1)
    cohens_d = mean_diff / d_std if d_std > 0 else 0.0

    sig = min(t_pval, w_pval)
    if sig < 0.01:
        conclusion = 'significant_1pct'
    elif sig < 0.05:
        conclusion = 'significant_5pct'
    elif sig < 0.10:
        conclusion = 'significant_10pct'
    else:
        conclusion = 'not_significant'

    return {
        'paired_t_stat': float(t_stat),
        'paired_t_pvalue': float(t_pval),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_pvalue': float(w_pval),
        'n_splits': n,
        'mean_diff': mean_diff,
        'cohens_d': float(cohens_d),
        'conclusion': conclusion,
    }


# ── Bootstrap confidence intervals ──────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Bootstrap confidence interval for the mean.

    Uses BCa (bias-corrected and accelerated) bootstrap.

    Args:
        values: Array of metric values (e.g., MAEs across splits).
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'mean', 'std', 'median', 'ci_lower', 'ci_upper', 'ci_level'.
    """
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    n = len(valid)

    if n < 2:
        m = float(valid[0]) if n == 1 else np.nan
        return {'mean': m, 'std': 0.0, 'median': m,
                'ci_lower': m, 'ci_upper': m, 'ci_level': confidence}

    observed_mean = np.mean(valid)

    # Bootstrap resamples
    boot_means = np.array([
        np.mean(rng.choice(valid, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        'mean': float(observed_mean),
        'std': float(np.std(valid, ddof=1)),
        'median': float(np.median(valid)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_level': confidence,
        'n_splits': n,
    }


# ── Friedman test + Nemenyi post-hoc ────────────────────────────────────

def friedman_nemenyi_test(
    model_maes: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> Dict:
    """Friedman test with Nemenyi post-hoc for multi-model comparison.

    Standard in M-competitions and NeurIPS forecasting papers.
    Ranks models within each split, then tests whether rank differences
    are statistically significant.

    Args:
        model_maes: Dict mapping model_name -> array of MAEs across splits.
        alpha: Significance level for post-hoc tests.

    Returns:
        Dict with Friedman statistic, p-value, average ranks, and
        critical difference for Nemenyi post-hoc.
    """
    from scipy import stats

    names = list(model_maes.keys())
    k = len(names)
    if k < 3:
        return {'error': 'Need at least 3 models for Friedman test'}

    # Build matrix: rows=splits, cols=models
    n_splits = min(len(v) for v in model_maes.values())
    matrix = np.column_stack([model_maes[name][:n_splits] for name in names])

    # Remove rows with NaN
    valid_rows = ~np.any(np.isnan(matrix), axis=1)
    matrix = matrix[valid_rows]
    N = len(matrix)

    if N < 3:
        return {'error': f'Only {N} valid splits (need >=3)'}

    # Rank within each split (row). Lower MAE = better = rank 1
    ranks = np.zeros_like(matrix)
    for i in range(N):
        ranks[i] = stats.rankdata(matrix[i])

    avg_ranks = np.mean(ranks, axis=0)

    # Friedman test
    chi2_stat, friedman_p = stats.friedmanchisquare(*[matrix[:, j] for j in range(k)])

    # Nemenyi critical difference
    # CD = q_alpha * sqrt(k*(k+1)/(6*N))
    # q_alpha values from Demsar (2006), Table 5
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.314,
        14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458,
        18: 3.489, 19: 3.517, 20: 3.544,
    }
    # For larger k, approximate
    if k <= 20:
        q_alpha_val = q_alpha_table.get(k, 2.343)
    else:
        q_alpha_val = 2.326 + 0.05 * k  # rough approximation

    cd = q_alpha_val * np.sqrt(k * (k + 1) / (6 * N))

    # Build pairwise significance matrix
    rank_dict = {name: float(avg_ranks[i]) for i, name in enumerate(names)}
    sorted_models = sorted(rank_dict.items(), key=lambda x: x[1])

    # Nemenyi: two models are significantly different if
    # |avg_rank_i - avg_rank_j| > CD
    pairwise_sig = {}
    for i, (name_i, rank_i) in enumerate(sorted_models):
        for j, (name_j, rank_j) in enumerate(sorted_models):
            if i >= j:
                continue
            diff = abs(rank_i - rank_j)
            pairwise_sig[f"{name_i} vs {name_j}"] = {
                'rank_diff': round(diff, 3),
                'significant': diff > cd,
            }

    return {
        'friedman_chi2': float(chi2_stat),
        'friedman_p': float(friedman_p),
        'n_splits': N,
        'n_models': k,
        'avg_ranks': rank_dict,
        'ranked_models': [(name, round(rank, 3)) for name, rank in sorted_models],
        'critical_difference': round(cd, 3),
        'alpha': alpha,
        'pairwise_significant': pairwise_sig,
    }


# ── Robustness Report ───────────────────────────────────────────────────

@dataclass
class RobustnessReport:
    """Complete robustness analysis for a single dataset."""
    task_name: str
    n_splits: int
    split_type: str = 'rolling_origin'

    model_stats: Dict[str, Dict] = field(default_factory=dict)
    # model_name -> {mean, std, median, ci_lower, ci_upper, maes}

    pairwise_dm: Dict[str, Dict] = field(default_factory=dict)
    # "model_a vs model_b" -> DM test result

    friedman_result: Optional[Dict] = None

    def add_model(self, name: str, maes: np.ndarray):
        ci = bootstrap_ci(maes)
        self.model_stats[name] = {**ci, 'maes': maes.tolist()}

    def compute_pairwise_dm(self, model_a: str, model_b: str):
        if model_a not in self.model_stats or model_b not in self.model_stats:
            return
        maes_a = np.array(self.model_stats[model_a]['maes'])
        maes_b = np.array(self.model_stats[model_b]['maes'])
        result = diebold_mariano_multi_split(maes_a, maes_b)
        self.pairwise_dm[f"{model_a} vs {model_b}"] = result

    def compute_friedman(self, model_names: Optional[List[str]] = None):
        names = model_names or list(self.model_stats.keys())
        if len(names) < 3:
            return
        model_maes = {n: np.array(self.model_stats[n]['maes']) for n in names
                      if n in self.model_stats}
        self.friedman_result = friedman_nemenyi_test(model_maes)

    def ranking_table(self) -> List[Dict]:
        """Return sorted ranking table with CI."""
        rows = []
        for name, stats in self.model_stats.items():
            rows.append({
                'model': name,
                'mean_mae': stats['mean'],
                'std_mae': stats['std'],
                'median_mae': stats['median'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'n_splits': stats['n_splits'],
            })
        rows.sort(key=lambda x: x['mean_mae'])
        for i, row in enumerate(rows):
            row['rank'] = i + 1
        return rows

    def summary_dict(self) -> Dict:
        """Full report as serializable dict."""
        return {
            'task': self.task_name,
            'n_splits': self.n_splits,
            'split_type': self.split_type,
            'ranking': self.ranking_table(),
            'pairwise_dm': self.pairwise_dm,
            'friedman': self.friedman_result,
        }
