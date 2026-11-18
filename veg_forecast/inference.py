"""Load a trained model folder and forecast a cache."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.collate import collate_batch, to_device
from .data.dataset import ForecastDataset
from .model import VegForecaster

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


def load_pretrained(weights_dir=DEFAULT_WEIGHTS, device="cpu"):
    """Model in eval mode + its config (dict). The folder holds config.json, model.pt (state dict), scaler.json and
    feature_engineering_stats.json (the files written by scripts/train.py)."""
    weights_dir = Path(weights_dir)
    with open(weights_dir / "config.json", encoding="utf-8") as fp:
        config = json.load(fp)
    model = VegForecaster.from_config(config)
    state = torch.load(weights_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), config


def load_dataset(cache_dir, weights_dir=DEFAULT_WEIGHTS, config=None, **kwargs):
    """Dataset scaled with the training statistics stored in the weights folder."""
    weights_dir = Path(weights_dir)
    if config is None:
        with open(weights_dir / "config.json", encoding="utf-8") as fp:
            config = json.load(fp)
    ds = ForecastDataset(cache_dir, scaler_path=weights_dir / "scaler.json",
                         engineered_stats_path=weights_dir / "feature_engineering_stats.json",
                         targets=config["target_names"], masked_future_features=config["masked_future_features"],
                         **kwargs)
    if ds.feature_names != config["feature_names"]:
        raise ValueError("feature names of the cache do not match the model config")
    return ds


@torch.no_grad()
def predict(model, dataset, batch_size=256, device="cpu"):
    """Quantile forecasts in raw VI units, shape (N, S, K, Q), in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    model.eval()
    out = [model(to_device(b, device)).cpu().numpy() for b in loader]
    return dataset.scaler.inverse_transform_quantiles(np.concatenate(out))
