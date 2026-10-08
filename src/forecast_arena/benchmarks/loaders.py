# Modified for forecast-arena: extracted and adapted from AION benchmark utilities, October 2026.
"""Generator-based loaders for M-competition datasets.

Yields one BenchmarkSeries at a time — never loads all series into memory.
Critical for M4's 100K series.

M3: .tsf files from Zenodo (train+test concatenated, split by last H values)
M4: separate train/test CSV files from GitHub (wide format)
"""

from pathlib import Path
from typing import Generator, NamedTuple

import numpy as np

from forecast_arena.benchmarks.constants import M3_GROUPS, M4_GROUPS, FrequencyGroup
from forecast_arena.benchmarks.tsf_parser import parse_tsf


class BenchmarkSeries(NamedTuple):
    """A single time series with official train/test split."""

    series_id: str        # e.g. "N0001", "T1"
    y_train: np.ndarray   # training portion (1D float64)
    y_test: np.ndarray    # holdout portion (1D float64)
    horizon: int          # = len(y_test)
    seasonality: int      # MASE denominator
    freq: str             # pandas offset alias
    group: str            # "Yearly", "Monthly", etc.


def load_m3(
    group: str = "Yearly",
    data_dir: str | Path = "data/m_competitions/M3",
) -> Generator[BenchmarkSeries, None, None]:
    """Load M3 competition series from .tsf files.

    The .tsf files contain train+test concatenated. Official split:
    last H values = test (where H is the group's horizon).

    Parameters
    ----------
    group : str
        One of "Yearly", "Quarterly", "Monthly", "Other".
    data_dir : str or Path
        Directory containing m3_*.tsf files.

    Yields
    ------
    BenchmarkSeries
        One per series, with official train/test split.
    """
    if group not in M3_GROUPS:
        raise ValueError(f"Unknown M3 group '{group}'. Choose from: {list(M3_GROUPS)}")

    fg: FrequencyGroup = M3_GROUPS[group]
    tsf_file = Path(data_dir) / f"m3_{group.lower()}_dataset.tsf"

    if not tsf_file.exists():
        raise FileNotFoundError(
            f"{tsf_file} not found. Run: python3 scripts/download_m_competitions.py --competition m3"
        )

    records = parse_tsf(tsf_file)

    for rec in records:
        values = rec["values"]
        series_id = rec.get("series_name", "unknown")
        h = fg.horizon

        if len(values) <= h:
            # Series too short — shouldn't happen with official M3 data
            continue

        y_train = values[:-h]
        y_test = values[-h:]

        yield BenchmarkSeries(
            series_id=series_id,
            y_train=y_train,
            y_test=y_test,
            horizon=h,
            seasonality=fg.seasonality,
            freq=fg.freq,
            group=group,
        )


def load_m4_info(
    data_dir: str | Path = "data/m_competitions/M4",
) -> dict[str, dict]:
    """Load M4-info.csv metadata: domain category, starting date, etc.

    Returns dict mapping series_id -> {category, frequency, horizon, sp, starting_date}.

    NOTE: In M4-info.csv, column "Frequency" = seasonal period (int),
    column "SP" = frequency string. This is a known quirk in the official data.
    """
    import csv

    info_file = Path(data_dir) / "M4-info.csv"
    if not info_file.exists():
        raise FileNotFoundError(
            f"{info_file} not found. Run: python3 scripts/download_m_competitions.py --competition m4"
        )

    result = {}
    with open(info_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["M4id"]] = {
                "category": row["category"],
                "frequency": row["SP"],          # column "SP" = freq string
                "sp": int(row["Frequency"]),     # column "Frequency" = seasonal period
                "horizon": int(row["Horizon"]),
                "starting_date": row["StartingDate"],
            }
    return result


def load_m4(
    group: str = "Yearly",
    data_dir: str | Path = "data/m_competitions/M4",
) -> Generator[BenchmarkSeries, None, None]:
    """Load M4 competition series from separate train/test CSV files.

    M4 uses wide-format CSVs with one row per series. First column is the
    series ID, remaining columns are values (NaN-padded for unequal lengths).

    Parameters
    ----------
    group : str
        One of "Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly".
    data_dir : str or Path
        Directory containing M4 CSV files.

    Yields
    ------
    BenchmarkSeries
        One per series, with official train/test split.
    """
    if group not in M4_GROUPS:
        raise ValueError(f"Unknown M4 group '{group}'. Choose from: {list(M4_GROUPS)}")

    fg: FrequencyGroup = M4_GROUPS[group]
    data_dir = Path(data_dir)
    train_file = data_dir / f"{group}-train.csv"
    test_file = data_dir / f"{group}-test.csv"

    for f in (train_file, test_file):
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found. Run: python3 scripts/download_m_competitions.py --competition m4"
            )

    # Read both CSVs line by line (memory-efficient for 48K Monthly series)
    import csv

    train_rows = {}
    with open(train_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            sid = row[0]
            vals = np.array(
                [float(v) for v in row[1:] if v.strip() != "" and v.strip().lower() != "nan"],
                dtype=np.float64,
            )
            train_rows[sid] = vals

    with open(test_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            sid = row[0]
            vals = np.array(
                [float(v) for v in row[1:] if v.strip() != "" and v.strip().lower() != "nan"],
                dtype=np.float64,
            )

            y_train = train_rows.get(sid)
            if y_train is None:
                continue

            yield BenchmarkSeries(
                series_id=sid,
                y_train=y_train,
                y_test=vals,
                horizon=fg.horizon,
                seasonality=fg.seasonality,
                freq=fg.freq,
                group=group,
            )
