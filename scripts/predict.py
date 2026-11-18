"""Forecast a cache with a trained model and report the metrics.

    python scripts/predict.py --cache-dir data/cache/val --out predictions/val.npz

Writes <out> (.npz: tile, row, pred (N, 3 targets, K VIs, Q quantiles) in raw VI units, target, quantiles,
target_timestamps) and <out>.metrics.json (MAE, RMSE, WMAPE, MASE, pinball, CRPS, q0.1-q0.9 coverage and width per VI).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veg_forecast.inference import DEFAULT_WEIGHTS, load_dataset, load_pretrained, predict  # noqa: E402
from veg_forecast.metrics import evaluate  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", required=True, help="cache folder of one evaluation split")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="model folder (default: released weights)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_pretrained(args.weights, device)
    ds = load_dataset(args.cache_dir, args.weights, config, show_progress=not args.no_progress)
    pred = predict(model, ds, args.batch_size, device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, tile=np.asarray([k[0] for k in ds.keys]), row=np.asarray([k[1] for k in ds.keys]),
                        pred=pred, target=ds.raw_targets, quantiles=np.asarray(config["quantiles"]),
                        target_names=np.asarray(config["target_names"]),
                        target_timestamps=np.asarray(ds.target_timestamps, dtype=str))
    vis = [t.replace("avg_", "").replace("_clear_sky", "") for t in config["target_names"]]
    metrics = {"cache_dir": str(args.cache_dir), "n_windows": len(ds),
               "metrics": evaluate(pred, ds.raw_targets, ds.last_observed, config["quantiles"], vis)}
    with open(out.with_suffix(".metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=1)
    print(f"{len(ds)} windows -> {out}")
    for vi, m in metrics["metrics"].items():
        print(f"  {vi:6s} MAE {m['mae']:.4f}  RMSE {m['rmse']:.4f}  MASE {m['mase']:.3f}  CRPS {m['crps']:.4f}  "
              f"coverage {m['coverage']:.3f}  width {m['interval_width']:.3f}")


if __name__ == "__main__":
    main()
