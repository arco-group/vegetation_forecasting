"""End-to-end smoke tests: released weights on a synthetic cache, and train -> predict with the scripts."""
import importlib.util
import json

import numpy as np
import torch

from veg_forecast.inference import load_dataset, load_pretrained, predict
from conftest import ROOT


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_released_weights(cache_dir):
    model, config = load_pretrained()  # strict state-dict loading
    assert sum(p.numel() for p in model.parameters()) > 1e6
    ds = load_dataset(cache_dir, config=config, show_progress=False)
    pred = predict(model, ds, batch_size=16)
    assert pred.shape == (len(ds), 3, 3, 3) and np.isfinite(pred).all()
    assert np.abs(pred).max() < 2.0  # raw VI units


def test_train_then_predict(cache_dir, tmp_path):
    out = tmp_path / "run"
    _script("train").main(["--cache-root", str(cache_dir), "--out-dir", str(out), "--min-crop-pixels", "0",
                           "--epochs", "2", "--batch-size", "8", "--d-model", "32", "--num-layers", "1",
                           "--num-heads", "4", "--dim-feedforward", "64", "--no-progress"])
    for f in ("config.json", "model.pt", "scaler.json", "feature_engineering_stats.json", "history.csv"):
        assert (out / f).exists(), f
    pred_file = tmp_path / "pred.npz"
    _script("predict").main(["--cache-dir", str(cache_dir), "--weights", str(out), "--out", str(pred_file),
                             "--no-progress"])
    with np.load(pred_file) as d:
        assert d["pred"].shape[1:] == (3, 3, 3)
    metrics = json.loads(pred_file.with_suffix(".metrics.json").read_text())["metrics"]
    assert set(metrics) == {"NDVI", "GNDVI", "EVI"}
    assert all(np.isfinite(m["mae"]) and 0 <= m["coverage"] <= 1 for m in metrics.values())
    torch.load(out / "checkpoint_last.pt", map_location="cpu", weights_only=False)
