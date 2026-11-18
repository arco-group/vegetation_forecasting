"""Evaluation metrics, pooled over all valid (window, target) items of one vegetation index.

Point metrics use the median quantile; MASE is scaled by the MAE of the persistence forecast (last observed value
of the same window); CRPS is approximated by trapezoidal integration of twice the pinball loss over the quantile
levels; coverage and width refer to the outermost quantiles (q0.1-q0.9 for the released model).
"""
import math

import numpy as np


def vi_metrics(pred, target, persistence, quantiles, abs_threshold=1.0):
    """pred (N, S, Q) raw units, target (N, S) with NaN = missing, persistence (N,), quantiles increasing with 0.5."""
    quantiles = [float(q) for q in quantiles]
    valid = np.isfinite(target)
    if abs_threshold is not None:
        valid &= np.abs(np.nan_to_num(target)) <= abs_threshold
    y = target[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)  # (n, Q)
    med = p[:, quantiles.index(0.5)]
    err = med - y
    out = {
        "n_items": int(valid.sum()),
        "mae": float(np.abs(err).mean()),
        "rmse": math.sqrt(float((err ** 2).mean())),
        "wmape": float(np.abs(err).sum() / np.abs(y).sum()),
    }
    naive = np.broadcast_to(persistence[:, None], target.shape)[valid].astype(np.float64)
    ok = np.isfinite(naive)
    naive_mae = float(np.abs(y[ok] - naive[ok]).mean()) if ok.any() else float("nan")
    out["mase"] = out["mae"] / naive_mae if naive_mae > 0 else float("nan")
    if len(quantiles) > 1:
        d = y[:, None] - p
        q = np.asarray(quantiles)[None, :]
        rho = np.abs(d) * (q * (d >= 0) + (1 - q) * (d < 0))
        out["pinball"] = float(rho.mean())
        out["crps"] = float((2 * np.trapezoid(rho, quantiles, axis=1)).mean())
        lo, hi = p[:, 0], p[:, -1]
        out["coverage"] = float(((y >= lo) & (y <= hi)).mean())
        out["interval_width"] = float((hi - lo).mean())
        out["crossing_rate"] = float(np.any(p[:, :-1] > p[:, 1:], axis=1).mean())
    return out


def evaluate(pred, target, persistence, quantiles, vi_names):
    """pred (N, S, K, Q), target (N, S, K), persistence (N, K) -> {vi: metrics}."""
    return {vi: vi_metrics(pred[:, :, k], target[:, :, k], persistence[:, k], quantiles)
            for k, vi in enumerate(vi_names)}
