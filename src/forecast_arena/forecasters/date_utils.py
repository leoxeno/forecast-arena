"""Safe date range utility for forecaster wrappers.

Handles pandas Timestamp overflow for long time series (e.g., M4 Yearly
with 800+ data points). Pandas nanosecond-precision timestamps are
bounded to ~1677–2262 AD, so yearly series longer than ~262 points
overflow when starting from 2000-01-01.

Strategy:
1. Try the requested date range directly.
2. On overflow, shift start date backward to keep end within bounds.
3. If the range exceeds ~585 years (Timestamp.max - Timestamp.min),
   fall back to monthly frequency to compress the date span.
"""

import pandas as pd

# Pandas Timestamp bounds (nanosecond precision)
_TS_MIN_YEAR = 1678  # safe margin above Timestamp.min (1677-09-21)
_TS_MAX_YEAR = 2262  # Timestamp.max is 2262-04-11
# Pandas Timedelta max is ~292 years — Prophet and others compute
# (max_date - min_date) internally, which overflows for large spans.
_TIMEDELTA_MAX_YEARS = 290  # safe margin below actual 292


def safe_date_range(start="2000-01-01", periods=1, freq=None):
    """Create a DatetimeIndex that stays within pandas Timestamp bounds.

    Drop-in replacement for pd.date_range() that handles overflow
    for long time series with low-frequency data (yearly, quarterly).

    Handles two overflow scenarios:
    1. Timestamp overflow: dates past 2262 AD (nanosecond precision limit)
    2. Timedelta overflow: date spans >292 years (Prophet's t_scale, etc.)
    """
    try:
        return pd.date_range(start=start, periods=periods, freq=freq)
    except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError):
        pass

    freq_upper = (freq or "D").upper().replace("-", "")

    if freq_upper in {"YE", "Y", "A", "AS", "YS", "YEDEC", "BA", "BAS"}:
        if periods > _TIMEDELTA_MAX_YEARS:
            # Series spans >290 years — Timedelta overflow even with shifted
            # start. Fall back to monthly spacing (N months ≈ N/12 years).
            return pd.date_range(
                start="2000-01-01", periods=periods, freq="MS"
            )
        # Each period ≈ 1 year. Shift start back so end stays ≤ 2262.
        # Leave 50-year margin for prediction horizon (max M4 horizon is 48).
        safe_start = _TS_MAX_YEAR - periods - 50
        if safe_start >= _TS_MIN_YEAR:
            return pd.date_range(
                start=f"{safe_start}-01-01", periods=periods, freq=freq
            )
        # Shouldn't reach here (290 < 585), but fall back safely.
        return pd.date_range(
            start="2000-01-01", periods=periods, freq="MS"
        )

    if freq_upper in {"QE", "Q", "QS", "QEA", "BQ", "BQS"}:
        # Each period ≈ 3 months. Shift start with margin for horizon.
        years_needed = periods // 4 + 15
        if years_needed > _TIMEDELTA_MAX_YEARS:
            return pd.date_range(
                start="2000-01-01", periods=periods, freq="D"
            )
        safe_start = _TS_MAX_YEAR - years_needed
        if safe_start >= _TS_MIN_YEAR:
            return pd.date_range(
                start=f"{safe_start}-01-01", periods=periods, freq=freq
            )
        return pd.date_range(
            start="2000-01-01", periods=periods, freq="D"
        )

    # Generic fallback: use daily frequency to keep within bounds
    return pd.date_range(start="2000-01-01", periods=periods, freq="D")
