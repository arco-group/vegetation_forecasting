"""Build the forecasting cache (one compressed .npz + .json per Sentinel-2 tile) from per-minicube time-series CSVs.

Each CSV holds one GreenEarthNet minicube at daily resolution: a `time` column, the daily weather / cloud-fraction
columns (`constants.METEO_COLUMNS`) and the vegetation-index statistics over cropland pixels (NaN on days without a
valid acquisition). Windows are defined in number of valid observations, not in days:

- history: from a valid observation up to and including the `history_window`-th valid observation, every day in
  between included (weather is daily, VIs are present on observation days only);
- future: the following days, up to and including the `forecast_window`-th next valid observation. The targets are
  the VIs at those observations and `future_target_positions` their indices in the future window.

A day counts as a valid observation when ALL target VIs are finite. Window starts are every `step`-th valid
observation. Future weather receives synthetic forecast noise (Gaussian, std = noise_percentage x monthly
climatological std of the tile (global fallback) x a factor growing linearly from 1 to `g5_target` at the last
future day). The builder of the training split also writes `scaler.json` with the feature and target statistics.

Python/NumPy/pandas only (no PyTorch).
"""
import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import METEO_COLUMNS, TARGET_COLUMNS, VEGETATION_COLUMNS
from .scaler import Scaler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


class RunningStats:
    """Streaming mean/var/min/max per column, ignoring NaNs."""

    def __init__(self, num_features):
        self.count = np.zeros(num_features, dtype=np.float64)
        self.sum = np.zeros(num_features, dtype=np.float64)
        self.sum_sq = np.zeros(num_features, dtype=np.float64)
        self.min_value = np.full(num_features, np.inf, dtype=np.float64)
        self.max_value = np.full(num_features, -np.inf, dtype=np.float64)

    def update(self, array):
        flat = array.reshape(-1, len(self.count))
        valid = ~np.isnan(flat)
        if not valid.any():
            return
        values = np.where(valid, flat, 0.0)
        self.count += valid.sum(axis=0)
        self.sum += values.sum(axis=0)
        self.sum_sq += (values ** 2).sum(axis=0)
        has = valid.any(axis=0)
        self.min_value[has] = np.minimum(self.min_value[has], np.where(valid, flat, np.inf).min(axis=0)[has])
        self.max_value[has] = np.maximum(self.max_value[has], np.where(valid, flat, -np.inf).max(axis=0)[has])

    def mean_var(self):
        valid = self.count > 0
        mean = np.zeros(len(self.count), dtype=np.float32)
        var = np.ones(len(self.count), dtype=np.float32)
        mean[valid] = (self.sum[valid] / self.count[valid]).astype(np.float32)
        v = self.sum_sq[valid] / self.count[valid] - mean[valid] ** 2
        var[valid] = np.where(v < 0, 0, v).astype(np.float32)
        return mean, var


def add_calendar_features(df, harmonics=3):
    """Day of year and its Fourier harmonics (period 366 days)."""
    doy = df.index.dayofyear
    df["day_of_year"] = doy
    base = (np.asarray(doy - 1) % 366) / 366
    for h in range(1, harmonics + 1):
        df[f"day_of_year_fourier_sin_{h}"] = np.sin(2 * np.pi * h * base)
        df[f"day_of_year_fourier_cos_{h}"] = np.cos(2 * np.pi * h * base)
    return df


def load_tile_metadata(path):
    """tile -> {'max_crop_pixels_pct': float, 'climate': str} from metadata/tile_metadata.csv."""
    out = {}
    if path is None or not Path(path).exists():
        return out
    with open(path, encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            crop = row.get("max_crop_pixels_pct", "")
            out[row["tile"]] = {"max_crop_pixels_pct": float(crop) if crop else None,
                                "climate": row.get("climate") or "unknown"}
    return out


def _object_array(items):
    """1-D object array (one entry per window), whatever the shapes of the entries."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


class CacheBuilder:
    def __init__(self, data_root, cache_root, target_columns=TARGET_COLUMNS, tile_metadata=None,
                 history_window=3, forecast_window=3, step=4, noise_percentage=0.1, min_samples_per_month=100,
                 g5_target=2.0, scaler_mode="standardization_arcsinh", fourier_harmonics=3, seed=42,
                 group_mode="directory", pattern_index=2, show_progress=True):
        self.data_root = Path(data_root)
        self.cache_root = Path(cache_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"data root {self.data_root} does not exist")
        if isinstance(target_columns, str):
            target_columns = [t.strip() for t in target_columns.split(",") if t.strip()]
        self.target_columns = list(target_columns)
        self.history_window = history_window
        self.forecast_window = forecast_window
        self.step = step
        self.noise_percentage = noise_percentage
        self.min_samples_per_month = int(min_samples_per_month)
        self.g5_target = float(g5_target)
        self.scaler_mode = scaler_mode
        self.fourier_harmonics = fourier_harmonics
        self.group_mode = group_mode
        self.pattern_index = pattern_index
        self.show_progress = show_progress
        self.rng = np.random.default_rng(seed)
        self.tiles = load_tile_metadata(tile_metadata)

        self.feature_names = None
        self.base_columns = None
        self.feature_stats = None
        self.target_stats = None
        self.month_counts_area = defaultdict(lambda: defaultdict(int))
        self.monthly_std_area = defaultdict(dict)
        self.monthly_std_global = {}

    def _progress(self, iterable, desc):
        return tqdm(iterable, desc=desc) if self.show_progress else iterable

    # ---- input ------------------------------------------------------------------------------------------------
    def collect_groups(self):
        """Tile -> sorted CSV files. 'directory': one sub-folder per tile (training split); 'pattern': tile code
        read from the file name token `pattern_index` (test splits: minicube_<i>_<tile>_<lat>_<lon>.csv)."""
        groups = defaultdict(list)
        if self.group_mode == "directory":
            for d in sorted(self.data_root.iterdir()):
                if d.is_dir():
                    groups[d.name].extend(sorted(d.glob("*.csv")))
        elif self.group_mode == "pattern":
            for p in self.data_root.rglob("*.csv"):
                parts = p.stem.split("_")
                if p.is_file() and len(parts) > self.pattern_index:
                    groups[parts[self.pattern_index]].append(p)
        else:
            raise ValueError("group_mode must be 'directory' or 'pattern'")
        return {k: sorted(v) for k, v in groups.items() if v}

    def load_csv(self, path):
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # unreadable file: skipped
            logger.warning("cannot read %s: %s", path, exc)
            return None
        # pandas renames repeated headers to name.1, name.2: keep the first occurrence only
        seen, dup = set(), []
        for col in df.columns:
            root, _, suffix = col.rpartition(".")
            base = root if root and suffix.isdigit() else col
            if base in seen:
                dup.append(col)
            else:
                seen.add(base)
        df = df.drop(columns=dup)
        if "time" not in df.columns:
            logger.warning("no time column in %s", path)
            return None
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates("time").set_index("time")
        if "wind" in df.columns:
            df["wind"] = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0)
        other = [c for c in df.columns if c != "wind"]
        df[other] = df[other].apply(pd.to_numeric, errors="coerce")
        return add_calendar_features(df, self.fourier_harmonics)

    def split_features(self, df):
        missing = [c for c in self.target_columns if c not in df.columns]
        if missing:
            logger.warning("missing target columns %s", missing)
            return None, None
        drop = [c for c in VEGETATION_COLUMNS if c in df.columns and c not in self.target_columns]
        features = df.drop(columns=drop)
        features = features[[c for c in features.columns if c not in self.target_columns]]
        return features, df[self.target_columns]

    # ---- noise ------------------------------------------------------------------------------------------------
    def meteo_columns(self, columns):
        cols = [c for c in METEO_COLUMNS if c in columns]
        return cols or list(columns)

    def compute_monthly_stats(self, groups):
        sums, sumsq, counts = (defaultdict(lambda: defaultdict(lambda: defaultdict(float))) for _ in range(3))
        g_sums, g_sumsq, g_counts = (defaultdict(lambda: defaultdict(float)) for _ in range(3))
        for area in self._progress(sorted(groups), "Monthly stats"):
            for path in groups[area]:
                df = self.load_csv(path)
                if df is None or df.empty:
                    continue
                features, _ = self.split_features(df)
                if features is None or features.empty:
                    continue
                cols = self.meteo_columns(features.columns)
                months = features.index.month
                for month in range(1, 13):
                    sel = months == month
                    if not sel.any():
                        continue
                    block = features.loc[sel, cols]
                    self.month_counts_area[area][month] += len(block)
                    for col in cols:
                        v = block[col].to_numpy(dtype=np.float64)
                        v = v[~np.isnan(v)]
                        if v.size == 0:
                            continue
                        for s, ss, c in ((sums[area][month], sumsq[area][month], counts[area][month]),
                                         (g_sums[month], g_sumsq[month], g_counts[month])):
                            s[col] += float(v.sum())
                            ss[col] += float((v ** 2).sum())
                            c[col] += int(v.size)

        def std(s, ss, c):
            out = {}
            for col, n in c.items():
                if n <= 1:
                    out[col] = 1.0
                else:
                    m = s[col] / n
                    out[col] = float(math.sqrt(max(ss[col] / n - m ** 2, 0.0)))
            return out

        for area, months in counts.items():
            for month, c in months.items():
                self.monthly_std_area[area][month] = std(sums[area][month], sumsq[area][month], c)
        for month, c in g_counts.items():
            self.monthly_std_global[month] = std(g_sums[month], g_sumsq[month], c)

    def _monthly_std(self, area, month, col):
        value = None
        if self.month_counts_area.get(area, {}).get(month, 0) >= self.min_samples_per_month:
            value = self.monthly_std_area.get(area, {}).get(month, {}).get(col)
        if value is None:
            value = self.monthly_std_global.get(month, {}).get(col)
        if value is None or not np.isfinite(value) or value <= 0:
            value = 1.0
        return value

    def add_future_noise(self, future, future_mask, future_ts, history_end, area):
        noisy = future.copy()
        noise_matrix = np.zeros_like(future, dtype=np.float32)
        if not self.noise_percentage:
            return noisy, noise_matrix
        cols_idx = [self.feature_names.index(c) for c in METEO_COLUMNS if c in self.feature_names]
        if not cols_idx:
            cols_idx = list(range(future.shape[1]))
        valid = ~future_mask[:, cols_idx]
        if not valid.any():
            return noisy, noise_matrix
        values = future[:, cols_idx]
        names = [self.feature_names[i] for i in cols_idx]

        days = (np.asarray(future_ts, dtype="datetime64[ns]") - np.datetime64(history_end))
        days = np.maximum(days.astype("timedelta64[D]").astype(np.float32), 0.0)
        last = float(days.max()) if days.size else 0.0
        beta = (self.g5_target - 1.0) / last if last > 0 else 0.0
        growth = np.maximum(1.0 + beta * days, 1.0)

        month_std = {m: np.array([self._monthly_std(area, m, n) for n in names], dtype=np.float32)
                     for m in range(1, 13)}
        months = [pd.Timestamp(t).month for t in future_ts]
        scale = np.vstack([month_std[m] for m in months]).astype(np.float32) * self.noise_percentage * growth[:, None]
        noise = self.rng.normal(loc=0.0, scale=scale, size=values.shape).astype(np.float32)
        noise = np.where(valid, noise, 0.0)
        noise_matrix[:, cols_idx] = noise
        noisy[:, cols_idx] = np.where(valid, values + noise, values)
        return noisy, noise_matrix

    # ---- windows ----------------------------------------------------------------------------------------------
    def process_area(self, area, csv_files):
        keys = ("history", "history_mask", "history_timestamps", "future", "future_mask", "future_noise",
                "future_timestamps", "target", "target_mask", "target_timestamps", "future_target_positions",
                "climate")
        samples = {k: [] for k in keys}
        records = []
        climate = self.tiles.get(area, {}).get("climate", "unknown")
        k_targets = len(self.target_columns)

        for path in self._progress(csv_files, area):
            df = self.load_csv(path)
            if df is None or df.empty:
                continue
            features, targets = self.split_features(df)
            if features is None:
                continue
            columns = features.columns.tolist()
            if self.feature_names is None:
                self.base_columns = columns
                self.feature_names = columns + [f"target_history_{t}" for t in self.target_columns]
                self.feature_stats = RunningStats(len(self.feature_names))
                self.target_stats = RunningStats(k_targets)
            elif columns != self.base_columns:
                raise ValueError(f"column order of {path} differs from the first CSV")

            x = features.to_numpy(dtype=np.float32)
            x_mask = np.isnan(x)
            y = targets.to_numpy(dtype=np.float32)
            ts = features.index.to_numpy().astype("datetime64[ns]")
            n_rows = len(features)
            valid_rows = ~np.isnan(y).any(axis=1)
            valid_pos = np.flatnonzero(valid_rows).tolist()
            if not valid_pos:
                continue
            starts = valid_pos if self.step <= 0 else valid_pos[:: self.step]

            for start in starts:
                hist_idx, n_obs, ptr = [], 0, start
                while ptr < n_rows and n_obs < self.history_window:
                    hist_idx.append(ptr)
                    n_obs += int(valid_rows[ptr])
                    ptr += 1
                if n_obs < self.history_window:
                    continue
                hist_vi = y[hist_idx]  # NaN where not observed
                history = np.concatenate([x[hist_idx], hist_vi], axis=1)
                history_mask = np.isnan(history)
                history_ts = ts[hist_idx]

                fut_idx, tgt, tgt_ts, tgt_pos = [], [], [], []
                while ptr < n_rows and len(tgt) < self.forecast_window:
                    fut_idx.append(ptr)
                    if valid_rows[ptr]:
                        tgt.append(y[ptr])
                        tgt_ts.append(str(ts[ptr]))
                        tgt_pos.append(len(fut_idx) - 1)
                    ptr += 1
                if len(tgt) < self.forecast_window:
                    continue
                empty_vi = np.full((len(fut_idx), k_targets), np.nan, dtype=np.float32)
                future = np.concatenate([x[fut_idx], empty_vi], axis=1)
                future_mask = np.concatenate([x_mask[fut_idx], np.isnan(empty_vi)], axis=1)
                future_ts = ts[fut_idx]
                target = np.stack(tgt).astype(np.float32)
                noisy, noise = self.add_future_noise(future, future_mask, future_ts, history_ts[-1], area)

                self.feature_stats.update(history)
                self.feature_stats.update(future)  # statistics of the noise-free covariates
                self.target_stats.update(target)

                hist_ts_str = [str(t) for t in history_ts]
                for k, v in (("history", history), ("history_mask", history_mask),
                             ("history_timestamps", hist_ts_str), ("future", noisy), ("future_mask", future_mask),
                             ("future_noise", noise), ("future_timestamps", [str(t) for t in future_ts]),
                             ("target", target), ("target_mask", np.zeros_like(target, dtype=bool)),
                             ("target_timestamps", tgt_ts), ("future_target_positions", tgt_pos),
                             ("climate", climate)):
                    samples[k].append(v)
                records.append({"area": area, "csv": path.name, "history_start": hist_ts_str[0],
                                "history_end": hist_ts_str[-1], "future_start": tgt_ts[0], "future_end": tgt_ts[-1],
                                "climate": climate})

        if not records:
            logger.info("no valid window for %s", area)
            return 0
        np.savez_compressed(
            self.cache_root / f"{area}.npz",
            history=_object_array(samples["history"]),
            history_mask=_object_array(samples["history_mask"]),
            history_timestamps=_object_array(samples["history_timestamps"]),
            future=_object_array(samples["future"]),
            future_mask=_object_array(samples["future_mask"]),
            future_noise=_object_array(samples["future_noise"]),
            future_timestamps=_object_array(samples["future_timestamps"]),
            target=np.stack(samples["target"]).astype(np.float32),
            target_mask=np.stack(samples["target_mask"]).astype(bool),
            future_target_positions=np.array(samples["future_target_positions"], dtype=np.int64),
            target_timestamps=np.array(samples["target_timestamps"], dtype=object),
            climate=np.array(samples["climate"]),
        )
        meta = {"area": area, "feature_names": self.feature_names, "target_names": self.target_columns,
                "history_window": self.history_window, "forecast_window": self.forecast_window, "step": self.step,
                "noise_percentage": self.noise_percentage, "scaler_mode": self.scaler_mode,
                "min_samples_per_month": self.min_samples_per_month, "g5_target": self.g5_target,
                "metadata_records": records}
        with open(self.cache_root / f"{area}.json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)
        return len(records)

    def build(self):
        groups = self.collect_groups()
        if not groups:
            raise RuntimeError(f"no CSV found under {self.data_root}")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.compute_monthly_stats(groups)
        total = sum(self.process_area(a, groups[a]) for a in self._progress(sorted(groups), "Tiles"))
        if self.feature_names is not None:
            scaler = Scaler(mode=self.scaler_mode, eps=1e-5)
            t_mean, t_var = self.target_stats.mean_var()
            if self.scaler_mode == "min-max":
                valid = self.feature_stats.count > 0
                scaler.min_value = np.where(valid, self.feature_stats.min_value, 0).astype(np.float32)
                scaler.max_value = np.where(valid, self.feature_stats.max_value, 1).astype(np.float32)
                scaler.target_min_value = self.target_stats.min_value.astype(np.float32)
                scaler.target_max_value = self.target_stats.max_value.astype(np.float32)
            else:
                scaler.mean, scaler.var = self.feature_stats.mean_var()
                scaler.target_mean, scaler.target_var = t_mean, t_var
            scaler.save(self.cache_root / "scaler.json")
        logger.info("%d windows written to %s", total, self.cache_root)
        return total


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the forecasting cache from per-minicube time-series CSVs.")
    ap.add_argument("--data-root", required=True, help="split folder with the CSVs (e.g. .../global/train)")
    ap.add_argument("--cache-root", required=True, help="output folder")
    ap.add_argument("--group-mode", choices=("directory", "pattern"), default="directory",
                    help="'directory' for the training split (one folder per tile), 'pattern' for the test splits")
    ap.add_argument("--pattern-index", type=int, default=2)
    ap.add_argument("--tile-metadata", default=str(Path(__file__).resolve().parents[2] / "metadata/tile_metadata.csv"))
    ap.add_argument("--target-columns", default=",".join(TARGET_COLUMNS))
    ap.add_argument("--history-window", type=int, default=3, help="valid observations in the history window")
    ap.add_argument("--forecast-window", type=int, default=3, help="valid observations to forecast")
    ap.add_argument("--step", type=int, default=4, help="a window starts every `step` valid observations")
    ap.add_argument("--noise-percentage", type=float, default=0.1)
    ap.add_argument("--min-samples-per-month", type=int, default=100)
    ap.add_argument("--g5-target", type=float, default=2.0)
    ap.add_argument("--scaler-mode", default="standardization_arcsinh",
                    choices=("standardization", "standardization_arcsinh", "min-max"))
    ap.add_argument("--fourier-harmonics", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    CacheBuilder(
        data_root=args.data_root, cache_root=args.cache_root, target_columns=args.target_columns,
        tile_metadata=args.tile_metadata, history_window=args.history_window, forecast_window=args.forecast_window,
        step=args.step, noise_percentage=args.noise_percentage, min_samples_per_month=args.min_samples_per_month,
        g5_target=args.g5_target, scaler_mode=args.scaler_mode, fourier_harmonics=args.fourier_harmonics,
        seed=args.seed, group_mode=args.group_mode, pattern_index=args.pattern_index,
        show_progress=not args.no_progress,
    ).build()


if __name__ == "__main__":
    main()
