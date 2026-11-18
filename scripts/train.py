"""Train the forecaster on a training cache (defaults = configuration of the released model).

    python scripts/train.py --cache-root data/cache/train --out-dir runs/seed42

Internal validation: temporal split with purge at --temporal-cutoff (training windows end before it, validation
windows start after it); the checkpoint with the lowest validation pinball loss is kept. The output folder has the
same layout as weights/ and can be passed to scripts/predict.py.
"""
import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veg_forecast.constants import FUTURE_MASKED_FEATURES, TARGET_COLUMNS  # noqa: E402
from veg_forecast.data.collate import collate_batch, to_device  # noqa: E402
from veg_forecast.data.dataset import ForecastDataset  # noqa: E402
from veg_forecast.data.splits import temporal_split  # noqa: E402
from veg_forecast.model import VegForecaster, multi_target_quantile_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-root", required=True, help="training cache (with scaler.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile-metadata", default=str(ROOT / "metadata/tile_metadata.csv"))
    ap.add_argument("--min-crop-pixels", type=float, default=60.0, help="minimum tile cropland share (%%)")
    ap.add_argument("--targets", default=",".join(TARGET_COLUMNS))
    ap.add_argument("--temporal-cutoff", default="2019-06-01")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-factor", type=float, default=0.2, help="ReduceLROnPlateau factor")
    ap.add_argument("--lr-patience", type=int, default=20)
    ap.add_argument("--lr-min", type=float, default=5e-5)
    ap.add_argument("--time-weight-alpha", type=float, default=0.1,
                    help="loss weight 1 / (1 + alpha x days since the last observation); <0 disables")
    ap.add_argument("--quantiles", default="0.1,0.5,0.9")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dim-feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0, help="limit batches per epoch (smoke tests); 0 = all")
    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, quantiles, alpha, device, optimizer=None, max_batches=0):
    """Item-weighted mean pinball loss per VI over the epoch (and the parameter updates if optimizer is given)."""
    train = optimizer is not None
    model.train(train)
    k = model.num_targets
    loss_sum, weight_sum = np.zeros(k), np.zeros(k)
    with torch.set_grad_enabled(train):
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = to_device(batch, device)
            weights = 1.0 / (1.0 + alpha * batch["target_delta_days"]) if alpha >= 0 else None
            preds = model(batch)
            losses = multi_target_quantile_loss(preds, batch["target"], batch["target_mask"], quantiles, weights)
            if train:
                optimizer.zero_grad()
                torch.stack(losses).sum().backward()
                optimizer.step()
            keep = (~batch["target_mask"]).float()
            w = keep * weights.unsqueeze(-1) if weights is not None else keep
            w = w.sum(dim=(0, 1)).cpu().numpy()
            loss_sum += np.array([l.item() for l in losses]) * w
            weight_sum += w
    return loss_sum / np.maximum(weight_sum, 1.0)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = [float(q) for q in args.quantiles.split(",")]
    targets = [t for t in args.targets.split(",") if t]

    ds = ForecastDataset(args.cache_root, targets=targets, masked_future_features=FUTURE_MASKED_FEATURES,
                         tile_metadata=args.tile_metadata, min_crop_pixels=args.min_crop_pixels,
                         show_progress=not args.no_progress)
    train_set, val_set, purged = temporal_split(ds, args.temporal_cutoff)
    print(f"windows: train {len(train_set)} | internal val {len(val_set)} | purged {purged}", flush=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch,
                            num_workers=args.num_workers)

    config = {
        "input_dim": len(ds.feature_names), "quantiles": quantiles, "num_targets": len(ds.target_names),
        "d_model": args.d_model, "num_layers": args.num_layers, "num_heads": args.num_heads,
        "dim_feedforward": args.dim_feedforward, "dropout": args.dropout,
        "target_names": ds.target_names, "feature_names": ds.feature_names,
        "masked_future_features": list(FUTURE_MASKED_FEATURES),
        "training": {k: v for k, v in vars(args).items() if k not in ("cache_root", "out_dir", "tile_metadata")},
    }
    with open(out / "config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
    shutil.copy(Path(args.cache_root) / "scaler.json", out / "scaler.json")
    ds.save_engineered_stats(out / "feature_engineering_stats.json")

    model = VegForecaster.from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor,
                                                           patience=args.lr_patience, min_lr=args.lr_min)
    best, history = float("inf"), []
    names = [t.replace("avg_", "").replace("_clear_sky", "") for t in ds.target_names]
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, quantiles, args.time_weight_alpha, device, optimizer, args.max_batches)
        va = run_epoch(model, val_loader, quantiles, args.time_weight_alpha, device, None, args.max_batches)
        val_loss = float(va.sum())
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_pinball": float(tr.sum()),
               "val_pinball": val_loss, **{f"val_pinball_{n}": float(v) for n, v in zip(names, va)}}
        history.append(row)
        print(" | ".join(f"{k} {v:.6f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), out / "model.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_val_pinball": best}, out / "checkpoint_last.pt")
        scheduler.step(val_loss)

    with open(out / "history.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"best internal-val pinball {best:.6f}; model saved to {out / 'model.pt'}")


if __name__ == "__main__":
    main()
