"""M-competition benchmark infrastructure.

Loaders, baselines, and aggregation for standardized model evaluation.
"""

from forecast_arena.benchmarks.aggregation import (
    build_leaderboard,
    build_per_frequency_table,
    compute_owa,
    load_jsonl_results,
)
from forecast_arena.benchmarks.constants import (
    M3_GROUPS,
    M3_NAIVE2_SMAPE,
    M3_ZENODO,
    M4_GROUPS,
    M4_NAIVE2_SMAPE,
    FrequencyGroup,
)
from forecast_arena.benchmarks.loaders import BenchmarkSeries, load_m3, load_m4, load_m4_info
from forecast_arena.benchmarks.naive2 import naive2_forecast
from forecast_arena.benchmarks.analysis import (
    ALGORITHM_FAMILIES,
    BenchmarkAnalysis,
    analyze_benchmark,
    export_analysis,
)

__all__ = [
    # Constants
    "FrequencyGroup",
    "M3_GROUPS",
    "M3_ZENODO",
    "M3_NAIVE2_SMAPE",
    "M4_GROUPS",
    "M4_NAIVE2_SMAPE",
    # Loaders
    "BenchmarkSeries",
    "load_m3",
    "load_m4",
    "load_m4_info",
    # Baseline
    "naive2_forecast",
    # Aggregation
    "compute_owa",
    "load_jsonl_results",
    "build_leaderboard",
    "build_per_frequency_table",
    # Analysis
    "ALGORITHM_FAMILIES",
    "BenchmarkAnalysis",
    "analyze_benchmark",
    "export_analysis",
]
