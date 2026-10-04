"""Fetchers for public series, cached on disk, with the source URL kept on the task.

Nothing here is redistributed with the package. Each function downloads from
the provider on first use, stores the raw file under the cache directory
(``FORECAST_ARENA_CACHE`` or ``~/.cache/forecast-arena``) and parses it into a
one-dimensional float array in time order.
"""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

USER_AGENT = "forecast-arena/0.1 (+https://github.com/leoxeno/forecast-arena)"


def cache_dir() -> Path:
    root = os.environ.get("FORECAST_ARENA_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "forecast-arena"
    )
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_text(url: str, filename: str, refresh: bool = False) -> str:
    """Download ``url`` once into the cache and return its text."""
    target = cache_dir() / filename
    if target.exists() and not refresh:
        return target.read_text()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    target.write_text(text)
    return text


# ── FRED (Federal Reserve Bank of St. Louis) ─────────────────────────────────

def fred_series(series_id: str, refresh: bool = False) -> np.ndarray:
    """Any FRED series by id, for example ``FEDFUNDS`` or ``DCOILWTICO``."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = fetch_text(url, f"fred_{series_id}.csv", refresh)
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    value_col = [c for c in df.columns if c.lower() not in ("date", "observation_date")][0]
    return df[value_col].dropna().to_numpy(dtype=float)


# ── NOAA Global Monitoring Laboratory: Mauna Loa CO2 ─────────────────────────

def mauna_loa_co2_monthly(refresh: bool = False) -> np.ndarray:
    """Monthly mean CO2 at Mauna Loa, 1958 to present (NOAA GML, public domain)."""
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    text = fetch_text(url, "noaa_co2_mm_mlo.csv", refresh)
    df = pd.read_csv(io.StringIO(text), comment="#")
    return df["average"].to_numpy(dtype=float)


# ── SILSO: monthly sunspot number ────────────────────────────────────────────

def silso_sunspots_monthly(refresh: bool = False) -> np.ndarray:
    """Monthly mean total sunspot number, 1749 to present (WDC-SILSO, CC BY-NC 4.0)."""
    url = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"
    text = fetch_text(url, "silso_SN_m_tot_V2.0.csv", refresh)
    df = pd.read_csv(io.StringIO(text), sep=";", header=None)
    values = df.iloc[:, 3].to_numpy(dtype=float)
    return values[values >= 0]


# ── NASA GISTEMP: global temperature anomaly ─────────────────────────────────

def gistemp_annual(refresh: bool = False) -> np.ndarray:
    """Annual global mean surface temperature anomaly (NASA GISTEMP v4, public domain)."""
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    text = fetch_text(url, "nasa_gistemp_GLB.csv", refresh)
    df = pd.read_csv(io.StringIO(text), skiprows=1, na_values=["***", "****"])
    return df["J-D"].dropna().to_numpy(dtype=float)


# ── NOAA CPC teleconnection indices (year + 12 monthly columns) ──────────────

_CPC_TABLES = {
    "nao": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table",
    "pna": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table",
    "ao": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table",
    "aao": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii.table",
}


def _parse_year_by_month_table(text: str, missing: float = -99.9) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if year < 1000:
            continue
        vals = []
        for p in parts[1:13]:
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(np.nan)
        rows.extend(vals)
    arr = np.array(rows, dtype=float)
    arr[arr <= missing] = np.nan
    arr = arr[~np.isnan(arr)]
    return arr


def cpc_index_monthly(name: str, refresh: bool = False) -> np.ndarray:
    """Monthly NAO, PNA, AO or AAO index from NOAA CPC (public domain)."""
    url = _CPC_TABLES[name]
    text = fetch_text(url, f"cpc_{name}_monthly.txt", refresh)
    return _parse_year_by_month_table(text)


# ── NOAA PSL: Nino 3.4 anomaly ───────────────────────────────────────────────

def nino34_monthly(refresh: bool = False) -> np.ndarray:
    """Monthly Nino 3.4 SST anomaly from NOAA PSL (public domain)."""
    url = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
    text = fetch_text(url, "psl_nina34_anom.txt", refresh)
    lines = text.splitlines()
    body = "\n".join(lines[1:])
    return _parse_year_by_month_table(body, missing=-99.0)


# ── Canadian lynx (Rdatasets mirror of R's ``datasets::lynx``) ───────────────

def lynx_annual(refresh: bool = False) -> np.ndarray:
    """Annual Canadian lynx trappings 1821 to 1934 (Brockwell and Davis, 1991)."""
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lynx.csv"
    text = fetch_text(url, "rdatasets_lynx.csv", refresh)
    df = pd.read_csv(io.StringIO(text))
    return df["value"].to_numpy(dtype=float)


__all__ = [
    "cache_dir", "fetch_text", "fred_series", "mauna_loa_co2_monthly",
    "silso_sunspots_monthly", "gistemp_annual", "cpc_index_monthly",
    "nino34_monthly", "lynx_annual",
]
