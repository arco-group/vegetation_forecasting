import json

import numpy as np

_MODES = ("standardization", "standardization_arcsinh", "min-max")


class Scaler:
    """Feature and target scaler stored as `scaler.json` next to the training cache.

    Feature statistics are vectors over the input features; target statistics are vectors over the K forecast
    targets. The released model uses mode `standardization_arcsinh`: z = arcsinh((x - mean) / sqrt(var + eps)).
    NaNs (missing values) are propagated unchanged.
    """

    def __init__(self, mode="standardization_arcsinh", eps=1e-5):
        if mode not in _MODES:
            raise ValueError(f"unsupported scaler mode {mode}")
        self.mode = mode
        self.eps = eps
        self.mean = self.var = self.min_value = self.max_value = None
        self.target_mean = self.target_var = self.target_min_value = self.target_max_value = None

    def has_target_stats(self):
        return any(v is not None for v in (self.target_mean, self.target_var, self.target_min_value,
                                           self.target_max_value))

    def num_features(self):
        return len(self.mean if self.mode != "min-max" else self.min_value)

    # ---- features -------------------------------------------------------------------------------------------
    def transform(self, array):
        x = np.array(array, copy=True)
        if self.mode == "min-max":
            rng = self.max_value - self.min_value
            valid = rng != 0
            x = x - self.min_value
            x[:, valid] = x[:, valid] / rng[valid]
            return x
        x = (x - self.mean) / np.sqrt(self.var + self.eps)
        return np.arcsinh(x) if self.mode == "standardization_arcsinh" else x

    def insert_feature_stats(self, position, stats):
        """Insert statistics of extra features (dict with mean/var/min/max arrays) before column `position`."""
        if self.mode == "min-max":
            self.min_value = np.concatenate([self.min_value[:position], stats["min"], self.min_value[position:]])
            self.max_value = np.concatenate([self.max_value[:position], stats["max"], self.max_value[position:]])
        else:
            self.mean = np.concatenate([self.mean[:position], stats["mean"], self.mean[position:]])
            self.var = np.concatenate([self.var[:position], stats["var"], self.var[position:]])

    # ---- targets (arrays (N, S, K) or (N, K) or (N,) for K = 1) -------------------------------------------------
    def _target_view(self, array):
        x = np.array(array, dtype=np.float32, copy=True)
        k = self.num_targets()
        squeeze = x.ndim == 1 or (x.ndim == 2 and k == 1)
        if squeeze:
            x = x[..., None]
        if x.shape[-1] != k:
            raise ValueError(f"scaler has {k} targets, array has shape {x.shape}")
        return x, squeeze

    def num_targets(self):
        for v in (self.target_mean, self.target_var, self.target_min_value, self.target_max_value):
            if v is not None:
                return int(np.asarray(v).size)
        raise ValueError("scaler has no target statistics")

    def transform_target(self, array, mask=None):
        if not self.has_target_stats():
            return array
        x, squeeze = self._target_view(array)
        valid = np.isfinite(x)
        if mask is not None:
            valid &= ~self._target_view(mask)[0].astype(bool)
        if self.mode == "min-max":
            rng = np.maximum(self.target_max_value - self.target_min_value, self.eps)
            y = (x - self.target_min_value) / rng
        else:
            y = (x - self.target_mean) / np.sqrt(self.target_var + self.eps)
            if self.mode == "standardization_arcsinh":
                y = np.arcsinh(y)
        x[valid] = y[valid]
        return x[..., 0] if squeeze else x

    def inverse_transform_target(self, array):
        if not self.has_target_stats():
            return array
        x, squeeze = self._target_view(array)
        valid = np.isfinite(x)
        if self.mode == "min-max":
            rng = np.maximum(self.target_max_value - self.target_min_value, self.eps)
            y = x * rng + self.target_min_value
        else:
            y = np.sinh(x) if self.mode == "standardization_arcsinh" else x
            y = y * np.sqrt(self.target_var + self.eps) + self.target_mean
        x[valid] = y[valid]
        return x[..., 0] if squeeze else x

    def inverse_transform_quantiles(self, preds):
        """preds (N, S, K, Q) in the scaled space -> raw VI units."""
        preds = np.asarray(preds, dtype=np.float32)
        return np.stack([self.inverse_transform_target(preds[..., q]) for q in range(preds.shape[-1])], axis=-1)

    # ---- io ---------------------------------------------------------------------------------------------------
    _ARRAYS = ("mean", "var", "min_value", "max_value", "target_mean", "target_var", "target_min_value",
               "target_max_value")

    def to_dict(self):
        out = {"mode": self.mode, "eps": self.eps}
        for k in self._ARRAYS:
            v = getattr(self, k)
            out[k] = None if v is None else np.asarray(v).tolist()
        return out

    @classmethod
    def from_dict(cls, data):
        scaler = cls(mode=data["mode"], eps=data["eps"])
        for k in cls._ARRAYS:
            if data.get(k) is not None:
                setattr(scaler, k, np.array(data[k], dtype=np.float32))
        return scaler

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fp:
            return cls.from_dict(json.load(fp))
