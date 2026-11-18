"""Synthetic time-series CSVs with the layout of the real ones (daily weather, VIs on sparse clear-sky days)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veg_forecast.constants import METEO_COLUMNS, TARGET_COLUMNS  # noqa: E402


def synthetic_csv(path, start, days, rng):
    time = pd.date_range(start, periods=days, freq="D")
    doy = time.dayofyear.to_numpy()
    df = pd.DataFrame({"time": time.strftime("%Y-%m-%d")})
    df["wind"] = rng.gamma(2.0, 2.0, days)
    df["humidity"] = rng.uniform(40, 95, days)
    df["solar_irradiance"] = 150 + 100 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 10, days)
    df["rainfall"] = rng.exponential(2.0, days) * (rng.random(days) < 0.3)
    df["pressure"] = rng.normal(1013, 5, days)
    df["avg_temperature"] = 12 + 12 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 2, days)
    df["min_temperature"] = df["avg_temperature"] - 5
    df["max_temperature"] = df["avg_temperature"] + 5
    for c in ("thick_clouds", "thin_clouds", "clouds_shadows"):
        df[c] = rng.uniform(0, 1, days)
    assert list(df.columns[1:]) == METEO_COLUMNS
    # vegetation indices only on irregular "clear-sky" days (every 5 days, ~40% lost to clouds)
    ndvi = 0.45 + 0.3 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.normal(0, 0.02, days)
    obs = (np.arange(days) % 5 == 0) & (rng.random(days) > 0.4)
    for col, v in zip(TARGET_COLUMNS, (ndvi, 0.8 * ndvi, 0.9 * ndvi - 0.05)):
        df[col] = np.where(obs, v, np.nan)
    df["s2_avail"] = obs.astype(int)  # dropped by the builder
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture(scope="session")
def csv_root(tmp_path_factory):
    """Training-split layout: one folder per tile, cubes spanning the 2019-06-01 temporal cutoff."""
    root = tmp_path_factory.mktemp("csv") / "train"
    rng = np.random.default_rng(0)
    for tile in ("31TCJ", "32UNA"):
        for i in range(3):
            synthetic_csv(root / tile / f"{tile}_cube{i}.csv", f"2019-0{1 + i}-01", 300, rng)
    return root


@pytest.fixture(scope="session")
def cache_dir(csv_root, tmp_path_factory):
    from veg_forecast.data.cache_builder import CacheBuilder

    out = tmp_path_factory.mktemp("cache") / "train"
    n = CacheBuilder(csv_root, out, group_mode="directory", show_progress=False).build()
    assert n > 0
    return out
