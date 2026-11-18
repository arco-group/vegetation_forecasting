"""PyTorch dataset over a cache folder built by `veg_forecast.data.cache_builder`.

Per window it returns the daily history and future sequences with their missing-value masks, the positions of the
target days in the future window and the targets. On loading it

1. adds the engineered weather features (`constants.ENGINEERED_FEATURES`: rain sum and cold/hot day counts over
   the last 7 / 14 days and since the previous valid observation), computed on raw values before scaling;
2. scales features and targets with the training `scaler.json` (engineered-feature statistics from the training
   set: computed here for the training cache, read from `feature_engineering_stats.json` otherwise);
3. masks the features in `masked_future_features` in the future branch (default: future cloud fractions and the
   *_between_targets features, see `constants.FUTURE_MASKED_FEATURES`).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import ENGINEERED_FEATURES, FUTURE_MASKED_FEATURES
from .cache_builder import load_tile_metadata
from .scaler import Scaler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _between_targets(seq, rain_idx, temp_idx, bounds):
    """Rain sum and days < 10 / > 30 degC since the previous bound, stored at each bound (NaN elsewhere)."""
    n = seq.shape[0]
    rain, cold, hot = (np.full(n, np.nan, dtype=np.float32) for _ in range(3))
    for prev, b in zip([-1] + bounds[:-1], bounds):
        if b >= n:
            continue
        window = seq[prev + 1: b + 1]
        temp = window[:, temp_idx] if temp_idx is not None else np.full(len(window), np.nan)
        rain[b] = float(window[:, rain_idx].sum()) if rain_idx is not None else np.nan
        cold[b] = float((temp < 10.0).sum()) if temp_idx is not None else np.nan
        hot[b] = float((temp > 30.0).sum()) if temp_idx is not None else np.nan
    return rain, cold, hot


def _rolling(seq, ts, rain_idx, temp_idx, days):
    """Rain sum and days < 10 / > 30 degC over the last `days` days (inclusive)."""
    ts = np.asarray(ts, dtype="datetime64[ns]")
    rain, cold, hot = (np.zeros(len(ts), dtype=np.float32) for _ in range(3))
    for i, t in enumerate(ts):
        window = seq[(ts >= t - np.timedelta64(days, "D")) & (ts <= t)]
        if window.size == 0:
            continue
        if rain_idx is not None:
            rain[i] = float(window[:, rain_idx].sum())
        if temp_idx is not None:
            cold[i] = float((window[:, temp_idx] < 10.0).sum())
            hot[i] = float((window[:, temp_idx] > 30.0).sum())
    return rain, cold, hot


def engineered_features(seq, ts, bounds, rain_idx, temp_idx):
    """(T, 9) features in the order of constants.ENGINEERED_FEATURES."""
    r_bt, c_bt, h_bt = _between_targets(seq, rain_idx, temp_idx, bounds)
    r7, c7, h7 = _rolling(seq, ts, rain_idx, temp_idx, 7)
    r14, c14, h14 = _rolling(seq, ts, rain_idx, temp_idx, 14)
    return np.stack([r_bt, c_bt, h_bt, r7, r14, c7, h7, c14, h14], axis=1).astype(np.float32)


class ForecastDataset(Dataset):
    """
    Args:
        cache_dir: folder with <tile>.npz / <tile>.json.
        scaler_path: training scaler.json (default: <cache_dir>/scaler.json, i.e. the training cache itself).
        engineered_stats_path: training feature_engineering_stats.json; None = compute the statistics on this
            dataset (training set only).
        targets: subset/order of the cached targets to predict (default: all).
        masked_future_features: feature names masked in the future branch.
        tile_metadata, min_crop_pixels: keep only tiles whose cropland share (metadata/tile_metadata.csv) is
            >= min_crop_pixels percent (training filter of the paper: 60). 0 disables the filter.
        apply_scaling: False returns raw values (for inspection).
    """

    def __init__(self, cache_dir, scaler_path=None, engineered_stats_path=None, targets=None,
                 masked_future_features=tuple(FUTURE_MASKED_FEATURES), tile_metadata=None, min_crop_pixels=0.0,
                 apply_scaling=True, show_progress=True):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"cache folder {self.cache_dir} does not exist")
        self.show_progress = show_progress
        self._load(load_tile_metadata(tile_metadata) if min_crop_pixels > 0 else None, float(min_crop_pixels))

        requested = list(targets) if targets else list(self.available_target_names)
        missing = [t for t in requested if t not in self.available_target_names]
        if missing:
            raise ValueError(f"targets {missing} not in the cache ({self.available_target_names})")
        self.target_names = requested
        self.target_indices = [self.available_target_names.index(t) for t in requested]
        self.targets = self.targets[..., self.target_indices]
        self.target_masks = self.target_masks[..., self.target_indices]

        self.raw_targets = np.where(self.target_masks, np.nan, self.targets).astype(np.float32)
        self.last_observed = self._last_observed()
        self._add_engineered_features()

        self.scaler = None
        if apply_scaling:
            self._scale(scaler_path or self.cache_dir / "scaler.json", engineered_stats_path)

        self.masked_future_idx = [self.feature_names.index(n) for n in masked_future_features]

    # ---- loading ----------------------------------------------------------------------------------------------
    def _load(self, tiles, min_crop):
        files = sorted(self.cache_dir.glob("*.npz"))
        if not files:
            raise RuntimeError(f"no .npz file in {self.cache_dir}")
        self.history, self.history_masks, self.history_timestamps = [], [], []
        self.future, self.future_masks, self.future_timestamps = [], [], []
        self.future_target_positions, self.target_timestamps, self.records, self.keys = [], [], [], []
        targets, target_masks = [], []
        self.feature_names = self.available_target_names = None
        for path in files:
            with open(path.with_suffix(".json"), encoding="utf-8") as fp:
                meta = json.load(fp)
            area = meta["metadata_records"][0]["area"]
            if tiles is not None:
                crop = tiles.get(area, {}).get("max_crop_pixels_pct")
                if crop is None or crop < min_crop:
                    continue
            if self.feature_names is None:
                self.feature_names = list(meta["feature_names"])
                self.available_target_names = list(meta.get("target_names") or meta["target_column"])
            elif meta["feature_names"] != self.feature_names:
                raise ValueError(f"{path}: feature names differ from the other tiles")
            with np.load(path, allow_pickle=True) as data:
                d = {k: data[k] for k in ("history", "history_mask", "history_timestamps", "future", "future_mask",
                                          "future_timestamps", "future_target_positions", "target", "target_mask",
                                          "target_timestamps")}
            self.history += [np.asarray(s, dtype=np.float32) for s in d["history"]]
            self.history_masks += [np.asarray(s, dtype=bool) for s in d["history_mask"]]
            self.history_timestamps += [list(s) for s in d["history_timestamps"]]
            self.future += [np.asarray(s, dtype=np.float32) for s in d["future"]]
            self.future_masks += [np.asarray(s, dtype=bool) for s in d["future_mask"]]
            self.future_timestamps += [list(s) for s in d["future_timestamps"]]
            self.future_target_positions += [np.asarray(p, dtype=np.int64) for p in d["future_target_positions"]]
            self.target_timestamps += [[str(t) for t in row] for row in d["target_timestamps"]]
            t = d["target"].astype(np.float32)
            targets.append(t if t.ndim == 3 else t[..., None])
            m = d["target_mask"].astype(bool)
            target_masks.append(m if m.ndim == 3 else m[..., None])
            self.records += meta["metadata_records"]
            self.keys += [(path.stem, i) for i in range(len(t))]
        if self.feature_names is None:
            raise RuntimeError(f"no tile of {self.cache_dir} passes the cropland filter")
        self.targets = np.concatenate(targets)
        self.target_masks = np.concatenate(target_masks)

    def _target_history_indices(self):
        return [i for i, n in enumerate(self.feature_names) if n.startswith("target_history_")]

    def _last_observed(self):
        """Last valid historical value of each target in raw units (persistence forecast, MASE scale)."""
        cols = [self.feature_names.index(f"target_history_{t}") for t in self.target_names]
        out = np.full((len(self.history), len(cols)), np.nan, dtype=np.float32)
        for i, (seq, mask) in enumerate(zip(self.history, self.history_masks)):
            for k, c in enumerate(cols):
                valid = ~mask[:, c] & np.isfinite(seq[:, c])
                if valid.any():
                    out[i, k] = seq[valid, c][-1]
        return out

    # ---- feature engineering ----------------------------------------------------------------------------------
    def _add_engineered_features(self):
        names = self.feature_names
        rain_idx = names.index("rainfall") if "rainfall" in names else None
        temp_idx = names.index("avg_temperature") if "avg_temperature" in names else None
        th = self._target_history_indices()
        start, end = (th[0], th[-1] + 1) if th else (len(names), len(names))
        n_eng = len(ENGINEERED_FEATURES)
        s_sum, s_sq = np.zeros(n_eng), np.zeros(n_eng)
        s_min, s_max = np.full(n_eng, np.inf), np.full(n_eng, -np.inf)
        s_cnt = np.zeros(n_eng, dtype=np.int64)

        def insert(seq, eng):
            return np.concatenate([seq[:, :start], eng, seq[:, start:end]], axis=1)

        iterator = zip(self.history, self.history_masks, self.history_timestamps, self.future, self.future_masks,
                       self.future_timestamps, self.future_target_positions)
        if self.show_progress:
            iterator = tqdm(iterator, total=len(self.history), desc="Feature engineering", leave=False)
        new_h, new_hm, new_f, new_fm = [], [], [], []
        for seq_h, mask_h, ts_h, seq_f, mask_f, ts_f, pos_f in iterator:
            # bounds = days on which every target VI is observed
            bounds_h = np.flatnonzero(~mask_h[:, th].any(axis=1)).tolist() if th else list(range(len(seq_h)))
            eng_h = engineered_features(seq_h, ts_h, bounds_h, rain_idx, temp_idx)
            # future features look back into the history: compute them on the concatenated sequence
            seq_c = np.concatenate([seq_h, seq_f], axis=0)
            ts_c = np.concatenate([np.asarray(ts_h, dtype="datetime64[ns]"), np.asarray(ts_f, dtype="datetime64[ns]")])
            bounds_c = bounds_h + [len(seq_h) + int(b) for b in pos_f]
            eng_f = engineered_features(seq_c, ts_c, bounds_c, rain_idx, temp_idx)[len(seq_h):]

            new_h.append(insert(seq_h, eng_h))
            new_hm.append(insert(mask_h, np.isnan(eng_h)))
            new_f.append(insert(seq_f, eng_f))
            new_fm.append(insert(mask_f, np.isnan(eng_f)))
            for j in range(n_eng):
                for block in (eng_h, eng_f):
                    col = block[:, j]
                    v = col[~np.isnan(col)].astype(np.float64)
                    if v.size:
                        s_sum[j] += v.sum()
                        s_sq[j] += (v ** 2).sum()
                        s_min[j] = min(s_min[j], float(v.min()))
                        s_max[j] = max(s_max[j], float(v.max()))
                        s_cnt[j] += len(v)
        self.history, self.history_masks, self.future, self.future_masks = new_h, new_hm, new_f, new_fm
        self.feature_names = names[:start] + list(ENGINEERED_FEATURES) + names[start:end]
        self.engineered_insert_position = start

        valid = s_cnt > 0
        stats = {k: np.full(n_eng, np.nan, dtype=np.float32) for k in ("mean", "var", "min", "max")}
        stats["mean"][valid] = (s_sum[valid] / s_cnt[valid]).astype(np.float32)
        stats["var"][valid] = (s_sq[valid] / s_cnt[valid] - stats["mean"][valid] ** 2).astype(np.float32)
        stats["min"][valid] = s_min[valid].astype(np.float32)
        stats["max"][valid] = s_max[valid].astype(np.float32)
        self.engineered_stats = stats if valid.any() else None

    def save_engineered_stats(self, path):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump({k: np.asarray(v).tolist() for k, v in self.engineered_stats.items()}, fp)

    # ---- scaling ----------------------------------------------------------------------------------------------
    def _scale(self, scaler_path, engineered_stats_path):
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"scaler not found: {scaler_path}")
        scaler = Scaler.load(scaler_path)
        if scaler.has_target_stats() and len(self.target_indices) != scaler.num_targets():
            for attr in ("target_mean", "target_var", "target_min_value", "target_max_value"):
                if getattr(scaler, attr) is not None:
                    setattr(scaler, attr, np.asarray(getattr(scaler, attr))[self.target_indices])
        if engineered_stats_path is not None:
            with open(engineered_stats_path, encoding="utf-8") as fp:
                self.engineered_stats = {k: np.asarray(v, dtype=np.float32) for k, v in json.load(fp).items()}
        if self.engineered_stats is None:
            raise ValueError("engineered-feature statistics unavailable")
        scaler.insert_feature_stats(self.engineered_insert_position, self.engineered_stats)
        if scaler.num_features() != len(self.feature_names):
            raise ValueError(f"scaler has {scaler.num_features()} features, data {len(self.feature_names)}")
        self.history = [scaler.transform(s).astype(np.float32) for s in self.history]
        self.future = [scaler.transform(s).astype(np.float32) for s in self.future]
        self.targets = scaler.transform_target(self.targets, self.target_masks)
        self.scaler = scaler

    # ---- access -----------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        future = torch.tensor(self.future[index], dtype=torch.float32)
        future_mask = torch.tensor(self.future_masks[index], dtype=torch.bool)
        if self.masked_future_idx:
            future[:, self.masked_future_idx] = 0.0
            future_mask[:, self.masked_future_idx] = True
        return {
            "index": index,
            "history": torch.tensor(self.history[index], dtype=torch.float32),
            "history_mask": torch.tensor(self.history_masks[index], dtype=torch.bool),
            "future": future,
            "future_mask": future_mask,
            "future_target_positions": torch.as_tensor(self.future_target_positions[index], dtype=torch.long),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "target_mask": torch.tensor(self.target_masks[index], dtype=torch.bool),
            "history_timestamps": self.history_timestamps[index],
            "target_timestamps": self.target_timestamps[index],
        }
