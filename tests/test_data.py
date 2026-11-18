import json

import numpy as np
import torch

from veg_forecast.constants import ENGINEERED_FEATURES, FUTURE_MASKED_FEATURES, TARGET_COLUMNS
from veg_forecast.data.collate import collate_batch
from veg_forecast.data.dataset import ForecastDataset
from veg_forecast.data.splits import temporal_split
from conftest import ROOT


def test_cache_windows(cache_dir):
    files = sorted(cache_dir.glob("*.npz"))
    assert len(files) == 2 and (cache_dir / "scaler.json").exists()
    meta = json.loads(files[0].with_suffix(".json").read_text())
    names = meta["feature_names"]
    # same base features as the released model (engineered features are added by the dataset)
    cfg = json.loads((ROOT / "weights/config.json").read_text())
    assert [n for n in cfg["feature_names"] if n not in ENGINEERED_FEATURES] == names
    k = len(TARGET_COLUMNS)
    with np.load(files[0], allow_pickle=True) as d:
        for hist, fut, pos, tgt, noise in zip(d["history"], d["future"], d["future_target_positions"], d["target"],
                                              d["future_noise"]):
            hist_vi = hist[:, -k:]
            valid = ~np.isnan(hist_vi).any(axis=1)
            assert valid.sum() == 3 and valid[0] and valid[-1]  # history starts and ends on an observation
            assert len(pos) == 3 and pos[-1] == len(fut) - 1  # the future window ends at the 3rd target
            assert np.isnan(fut[:, -k:]).all()  # no VI in the future inputs
            assert np.isfinite(tgt).all()
            assert np.abs(noise[:, len(names) - k:]).sum() == 0  # noise only on weather columns


def test_dataset_and_collate(cache_dir):
    ds = ForecastDataset(cache_dir, show_progress=False)
    assert len(ds.feature_names) == 30 and ds.feature_names[-3:] == [f"target_history_{t}" for t in TARGET_COLUMNS]
    item = ds[0]
    masked = [ds.feature_names.index(n) for n in FUTURE_MASKED_FEATURES]
    assert item["future_mask"][:, masked].all() and (item["future"][:, masked] == 0).all()
    assert not item["history_mask"][:, ds.feature_names.index("thick_clouds")].all()  # still in the history
    assert torch.isfinite(item["target"]).all()

    batch = collate_batch([ds[i] for i in range(4)])
    assert batch["history"].shape[0] == 4 and batch["history"].shape[-1] == 30
    assert batch["target"].shape == (4, 3, 3)
    assert (batch["target_delta_days"] > 0).all()
    assert np.isfinite(ds.last_observed).all()


def test_temporal_split(cache_dir):
    ds = ForecastDataset(cache_dir, show_progress=False)
    train, val, purged = temporal_split(ds, "2019-06-01")
    assert len(train) + len(val) + purged == len(ds) and len(train) and len(val)
    cut = np.datetime64("2019-06-01")
    assert all(np.datetime64(ds.target_timestamps[i][-1][:10]) < cut for i in train.indices)
    assert all(np.datetime64(ds.history_timestamps[i][0][:10]) >= cut for i in val.indices)
