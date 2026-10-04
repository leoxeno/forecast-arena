"""The task atlas: named series with provenance, frequency and seasonality.

A task is a one-dimensional series plus the metadata the protocol needs. Tasks
are registered once and loaded by id, so a run can be described by a list of
task ids and a list of model names.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .data import bundled, fetch, synthetic


@dataclass(frozen=True)
class TaskSpec:
    id: str
    description: str
    domain: str
    freq: str
    loader: Callable[[], np.ndarray] = field(repr=False, compare=False)
    source: str
    license: str
    seasonality: int = 1
    citation: str = ""
    offline: bool = True

    def load(self) -> np.ndarray:
        y = np.asarray(self.loader(), dtype=float).flatten()
        y = y[~np.isnan(y)]
        if len(y) < 50:
            raise ValueError(f"Task {self.id} loaded only {len(y)} observations")
        return y


TASK_REGISTRY: Dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec) -> TaskSpec:
    if spec.id in TASK_REGISTRY:
        raise ValueError(f"Task id already registered: {spec.id}")
    TASK_REGISTRY[spec.id] = spec
    return spec


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASK_REGISTRY[task_id]
    except KeyError:
        raise KeyError(f"Unknown task '{task_id}'. Known: {sorted(TASK_REGISTRY)}") from None


def list_tasks(domain: Optional[str] = None, offline_only: bool = False) -> List[str]:
    ids = []
    for tid, spec in TASK_REGISTRY.items():
        if domain and spec.domain != domain:
            continue
        if offline_only and not spec.offline:
            continue
        ids.append(tid)
    return sorted(ids)


def load_series(task_id: str) -> np.ndarray:
    return get_task(task_id).load()


def series_sha256(y: np.ndarray) -> str:
    """Content hash of a series, recorded with every result for provenance."""
    return hashlib.sha256(np.ascontiguousarray(y, dtype=np.float64).tobytes()).hexdigest()


def tasks_table() -> pd.DataFrame:
    rows = [
        {
            "task": s.id, "domain": s.domain, "freq": s.freq, "seasonality": s.seasonality,
            "offline": s.offline, "source": s.source, "license": s.license,
            "description": s.description,
        }
        for s in TASK_REGISTRY.values()
    ]
    return pd.DataFrame(rows).sort_values(["domain", "task"]).reset_index(drop=True)


def register_csv_task(task_id: str, path: str, column: str, *, description: str,
                      domain: str, freq: str, source: str, license: str,
                      seasonality: int = 1, **read_csv_kwargs) -> TaskSpec:
    """Register a task backed by a CSV column on disk (your own data, never committed)."""
    def loader() -> np.ndarray:
        df = pd.read_csv(path, **read_csv_kwargs)
        return df[column].to_numpy(dtype=float)
    return register_task(TaskSpec(
        id=task_id, description=description, domain=domain, freq=freq, loader=loader,
        source=source, license=license, seasonality=seasonality, offline=True,
    ))


# ── Built-in atlas ───────────────────────────────────────────────────────────

def _col(gen: Callable[[], pd.DataFrame], col: str) -> Callable[[], np.ndarray]:
    return lambda: gen()[col].to_numpy(dtype=float)


_SYNTH = "Generated at load time by forecast_arena.data.synthetic from a fixed initial condition"

for _spec in [
    TaskSpec("lorenz-x", "Lorenz (1963) attractor, X component, dt = 0.01", "chaos", "D",
             _col(synthetic.lorenz63, "X"), _SYNTH, "Apache-2.0 (this repository)"),
    TaskSpec("rossler-x", "Rossler (1976) attractor, X component, dt = 0.05", "chaos", "D",
             _col(synthetic.rossler, "X"), _SYNTH, "Apache-2.0 (this repository)"),
    TaskSpec("thomas-x", "Thomas cyclically symmetric attractor, X component, b = 0.208186", "chaos", "D",
             _col(synthetic.thomas, "X"), _SYNTH, "Apache-2.0 (this repository)"),
    TaskSpec("mackey-glass", "Mackey-Glass delay equation, tau = 17, sampled every time unit", "chaos", "D",
             _col(synthetic.mackey_glass, "X"), _SYNTH, "Apache-2.0 (this repository)"),
    TaskSpec("henon-x", "Henon map, X component, a = 1.4, b = 0.3", "chaos", "D",
             _col(synthetic.henon, "X"), _SYNTH, "Apache-2.0 (this repository)"),
    TaskSpec("logistic", "Logistic map, r = 3.9", "chaos", "D",
             _col(synthetic.logistic, "X"), _SYNTH, "Apache-2.0 (this repository)"),
    TaskSpec("sunspots-yearly", "Yearly sunspot numbers 1700 to 2008", "astrophysics", "YE",
             bundled.sunspots_yearly, "statsmodels.datasets.sunspots (bundled)", "Public domain",
             citation="Source: SILSO / Royal Observatory of Belgium, as shipped in statsmodels"),
    TaskSpec("nile", "Annual Nile river flow at Aswan 1871 to 1970", "hydrology", "YE",
             bundled.nile_annual, "statsmodels.datasets.nile (bundled)", "Public domain",
             citation="Cobb (1978), Biometrika 65(2)"),
    TaskSpec("co2-monthly-1958-2001", "Mauna Loa CO2, monthly means from the weekly statsmodels series",
             "climate", "ME", bundled.co2_monthly_1958_2001, "statsmodels.datasets.co2 (bundled)",
             "Public domain", seasonality=12, citation="Keeling et al., Scripps Institution of Oceanography"),
    TaskSpec("sunspots-monthly", "Monthly mean total sunspot number 1749 to present", "astrophysics", "ME",
             fetch.silso_sunspots_monthly, "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv",
             "CC BY-NC 4.0 (WDC-SILSO)", offline=False),
    TaskSpec("co2-mauna-loa", "Monthly mean CO2 at Mauna Loa 1958 to present", "climate", "ME",
             fetch.mauna_loa_co2_monthly, "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
             "Public domain (NOAA GML)", seasonality=12, offline=False),
    TaskSpec("enso-nino34", "Nino 3.4 sea-surface temperature anomaly, monthly", "climate", "ME",
             fetch.nino34_monthly, "https://psl.noaa.gov/data/correlation/nina34.anom.data",
             "Public domain (NOAA PSL)", offline=False),
    TaskSpec("nao", "North Atlantic Oscillation index, monthly, 1950 to present", "climate", "ME",
             lambda: fetch.cpc_index_monthly("nao"), fetch._CPC_TABLES["nao"],
             "Public domain (NOAA CPC)", offline=False),
    TaskSpec("pna", "Pacific-North American index, monthly, 1950 to present", "climate", "ME",
             lambda: fetch.cpc_index_monthly("pna"), fetch._CPC_TABLES["pna"],
             "Public domain (NOAA CPC)", offline=False),
    TaskSpec("ao", "Arctic Oscillation index, monthly, 1950 to present", "climate", "ME",
             lambda: fetch.cpc_index_monthly("ao"), fetch._CPC_TABLES["ao"],
             "Public domain (NOAA CPC)", offline=False),
    TaskSpec("aao", "Antarctic Oscillation index, monthly, 1979 to present", "climate", "ME",
             lambda: fetch.cpc_index_monthly("aao"), fetch._CPC_TABLES["aao"],
             "Public domain (NOAA CPC)", offline=False),
    TaskSpec("global-temp", "NASA GISTEMP annual global temperature anomaly 1880 to present",
             "climate", "YE", fetch.gistemp_annual,
             "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
             "Public domain (NASA)", offline=False),
    TaskSpec("fed-funds", "Effective federal funds rate, monthly, 1954 to present", "macro", "ME",
             lambda: fetch.fred_series("FEDFUNDS"), "https://fred.stlouisfed.org/series/FEDFUNDS",
             "Public domain (Federal Reserve Board via FRED)", offline=False),
    TaskSpec("wti", "WTI crude oil spot price, daily, 1986 to present", "finance", "D",
             lambda: fetch.fred_series("DCOILWTICO"), "https://fred.stlouisfed.org/series/DCOILWTICO",
             "Public domain (US EIA via FRED)", offline=False),
    TaskSpec("lynx", "Annual Canadian lynx trappings 1821 to 1934", "ecology", "YE",
             fetch.lynx_annual, "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lynx.csv",
             "Public data (Brockwell and Davis, 1991; R datasets)", offline=False),
]:
    register_task(_spec)


__all__ = [
    "TaskSpec", "TASK_REGISTRY", "register_task", "register_csv_task", "get_task",
    "list_tasks", "load_series", "series_sha256", "tasks_table",
]
