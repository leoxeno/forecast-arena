"""Real series that ship inside ``statsmodels`` and therefore load offline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sunspots_yearly() -> np.ndarray:
    """Yearly sunspot numbers 1700 to 2008 (statsmodels ``sunspots`` dataset)."""
    from statsmodels.datasets import sunspots
    df = sunspots.load_pandas().data
    return df["SUNACTIVITY"].to_numpy(dtype=float)


def nile_annual() -> np.ndarray:
    """Annual Nile flow at Aswan 1871 to 1970 (statsmodels ``nile`` dataset)."""
    from statsmodels.datasets import nile
    df = nile.load_pandas().data
    return df["volume"].to_numpy(dtype=float)


def co2_monthly_1958_2001() -> np.ndarray:
    """Mauna Loa CO2, weekly 1958 to 2001 (statsmodels ``co2``), averaged to months."""
    from statsmodels.datasets import co2
    s = co2.load_pandas().data["co2"]
    s.index = pd.to_datetime(s.index)
    monthly = s.resample("MS").mean().interpolate(limit_direction="both")
    return monthly.to_numpy(dtype=float)


__all__ = ["sunspots_yearly", "nile_annual", "co2_monthly_1958_2001"]
