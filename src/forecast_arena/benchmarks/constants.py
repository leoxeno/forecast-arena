# Modified for forecast-arena: extracted and adapted from AION benchmark utilities, October 2026.
"""M-competition constants: horizons, seasonalities, frequencies, series counts.

Reference for all official competition parameters.

"""

from typing import NamedTuple


class FrequencyGroup(NamedTuple):
    horizon: int
    seasonality: int
    freq: str       # Pandas offset alias
    n_series: int


# ── M3 (Makridakis 2000) ────────────────────────────────────────────────────

M3_GROUPS = {
    "Yearly":    FrequencyGroup(horizon=6,  seasonality=1,  freq="YE",  n_series=645),
    "Quarterly": FrequencyGroup(horizon=8,  seasonality=4,  freq="QE",  n_series=756),
    "Monthly":   FrequencyGroup(horizon=18, seasonality=12, freq="ME",  n_series=1428),
    "Other":     FrequencyGroup(horizon=8,  seasonality=1,  freq="YE",  n_series=174),
}

# Zenodo record IDs (Monash Time Series Forecasting Archive, Hyndman's team)
M3_ZENODO = {
    "Yearly":    4656222,
    "Quarterly": 4656262,
    "Monthly":   4656298,
    "Other":     4656335,
}

M3_TOTAL_SERIES = 3003  # 645 + 756 + 1428 + 174


# ── M4 (Makridakis 2018) ────────────────────────────────────────────────────

M4_GROUPS = {
    "Yearly":    FrequencyGroup(horizon=6,  seasonality=1,  freq="YE",  n_series=23000),
    "Quarterly": FrequencyGroup(horizon=8,  seasonality=4,  freq="QE",  n_series=24000),
    "Monthly":   FrequencyGroup(horizon=18, seasonality=12, freq="ME",  n_series=48000),
    "Weekly":    FrequencyGroup(horizon=13, seasonality=1,  freq="W",   n_series=359),
    "Daily":     FrequencyGroup(horizon=14, seasonality=1,  freq="D",   n_series=4227),
    "Hourly":    FrequencyGroup(horizon=48, seasonality=24, freq="h",   n_series=414),
}

M4_GITHUB_BASE = (
    "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
)

M4_TOTAL_SERIES = 100000


# ── Published Naive2 sMAPE values (for sanity checking) ─────────────────────
# Source: Makridakis & Hibon (2000), Table 3
# Note: Exact reproduction is difficult — Hyndman documented sMAPE formula
# ambiguity in original M3 paper. Target: within ~2%.

M3_NAIVE2_SMAPE = {
    "Yearly":    17.88,
    "Quarterly": 9.95,
    "Monthly":   16.91,
    "Other":     5.33,
}

# Source: Makridakis et al. (2020), M4 competition results — SNavie2 sMAPE
M4_NAIVE2_SMAPE = {
    "Yearly":    16.342,
    "Quarterly": 11.012,
    "Monthly":   14.427,
    "Weekly":    9.161,
    "Daily":     3.045,
    "Hourly":    18.383,
}
