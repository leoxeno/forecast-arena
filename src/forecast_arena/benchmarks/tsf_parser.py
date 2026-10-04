"""Minimal parser for Monash Time Series Forecasting Archive .tsf format.

The .tsf format uses @-prefixed headers for metadata and colon-separated
attributes with comma-separated values in the data section.

No external dependencies — stdlib + numpy only.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_tsf(path: str | Path) -> List[Dict]:
    """Parse a .tsf file into a list of series dicts.

    Each dict has keys:
        - 'series_name': str
        - 'values': np.ndarray (float64, NaNs for missing)
        - plus any other @attribute fields from the header

    Parameters
    ----------
    path : str or Path
        Path to the .tsf file.

    Returns
    -------
    list[dict]
        One dict per series.
    """
    path = Path(path)
    attributes: list[str] = []
    series_list: list[dict] = []
    in_data = False

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("@"):
                token = line.split()[0].lower()
                if token == "@data":
                    in_data = True
                    continue
                if token == "@attribute":
                    # @attribute series_name {string}
                    parts = line.split()
                    if len(parts) >= 2:
                        attributes.append(parts[1])
                continue

            if not in_data:
                continue

            # Data line: attr1:attr2:...:val1,val2,val3,...
            # Split on colon — last segment contains the numeric values
            segments = line.split(":")
            values_str = segments[-1].strip()
            attr_values = [s.strip() for s in segments[:-1]]

            # Parse numeric values (handle "?" as NaN)
            raw_vals = values_str.split(",")
            values = np.array(
                [float(v) if v.strip() != "?" else np.nan for v in raw_vals],
                dtype=np.float64,
            )

            record: dict = {}
            for i, attr_name in enumerate(attributes):
                if i < len(attr_values):
                    record[attr_name] = attr_values[i]
            record["values"] = values

            series_list.append(record)

    return series_list
