import json
import numpy as np


class Scaler:
    """
    A scaler compatible with both single-output and multi-output targets.
    Target statistics are vectors of length K, where K is the number of predicted indices.
    """

    def __init__(self, mode="min-max", eps=1e-5):
        self.mode = mode
        self.eps = eps
        self.mean = None
        self.var = None
        self.min_value = None
        self.max_value = None
        self.target_mean = None
        self.target_var = None
        self.target_min_value = None
        self.target_max_value = None

    def has_target_stats(self):
        return (
            self.target_mean is not None
            or self.target_var is not None
            or self.target_min_value is not None
            or self.target_max_value is not None
        )

    def fit(self, array):
        if self.mode in {"standardization", "standardization_arcsinh"}:
            self.mean = np.nanmean(array, axis=0)
            self.var = np.nanvar(array, axis=0)
        elif self.mode == "min-max":
            self.min_value = np.nanmin(array, axis=0)
            self.max_value = np.nanmax(array, axis=0)
        else:
            raise ValueError("Unsupported mode. Use ‘standardization’ or 'min-max'.")

    def transform(self, array):
        transformed = array.copy()
        if self.mode == "standardization":
            transformed = (transformed - self.mean) / np.sqrt(self.var + self.eps)
        elif self.mode == "standardization_arcsinh":
            transformed = (transformed - self.mean) / np.sqrt(self.var + self.eps)
            transformed = np.arcsinh(transformed)
        elif self.mode == "min-max":
            rng = self.max_value - self.min_value
            valid = rng != 0
            transformed = transformed - self.min_value
            transformed[:, valid] = transformed[:, valid] / rng[valid]
        else:
            raise ValueError("Unsupported mode. Use ‘standardization’ or 'min-max'.")
        return transformed

    def _target_stats_and_shape(self, array):
        arr = np.asarray(array)
        squeeze_last = False
        if arr.ndim == 1:
            arr = arr[:, None]
            squeeze_last = True
        elif arr.ndim == 2:
            if self._num_target_stats() == 1:
                arr = arr[..., None]
                squeeze_last = True
        elif arr.ndim != 3:
            raise ValueError(f"Shape target Unsupported: {arr.shape}")

        num_targets = arr.shape[-1]
        expected = self._num_target_stats()
        if expected != num_targets:
            raise ValueError(
                f"Incompatible target statistics: scaler={expected}, array={num_targets}."
            )
        return arr.copy(), squeeze_last

    def _num_target_stats(self):
        for values in (self.target_mean, self.target_var, self.target_min_value, self.target_max_value):
            if values is not None:
                return int(np.asarray(values).reshape(-1).shape[0])
        raise ValueError(" Missed Target Statistics.")

    def transform_target(self, array, mask=None):
        if not self.has_target_stats():
            return array
        transformed, squeeze_last = self._target_stats_and_shape(array)
        if mask is None:
            mask = np.zeros_like(transformed, dtype=bool)
        else:
            mask, _ = self._target_stats_and_shape(mask)
            mask = mask.astype(bool)
        valid = (~mask) & np.isfinite(transformed)
        if not valid.any():
            return transformed[..., 0] if squeeze_last else transformed

        if self.mode in {"standardization", "standardization_arcsinh"}:
            mean = np.asarray(self.target_mean, dtype=np.float32).reshape(1, 1, -1)
            var = np.asarray(self.target_var, dtype=np.float32).reshape(1, 1, -1)
            denom = np.sqrt(var + self.eps)
            transformed[valid] = ((transformed - mean) / denom)[valid]
            if self.mode == "standardization_arcsinh":
                transformed[valid] = np.arcsinh(transformed[valid])
        elif self.mode == "min-max":
            min_val = np.asarray(self.target_min_value, dtype=np.float32).reshape(1, 1, -1)
            max_val = np.asarray(self.target_max_value, dtype=np.float32).reshape(1, 1, -1)
            rng = np.maximum(max_val - min_val, self.eps)
            transformed[valid] = ((transformed - min_val) / rng)[valid]
        else:
            raise ValueError("Unsupported mode. Use ‘standardization’ or 'min-max'.")

        return transformed[..., 0] if squeeze_last else transformed

    def inverse_transform_target(self, array, mask=None):
        if not self.has_target_stats():
            return array
        restored, squeeze_last = self._target_stats_and_shape(array)
        if mask is None:
            mask = np.zeros_like(restored, dtype=bool)
        else:
            mask, _ = self._target_stats_and_shape(mask)
            mask = mask.astype(bool)
        valid = (~mask) & np.isfinite(restored)
        if not valid.any():
            return restored[..., 0] if squeeze_last else restored

        if self.mode in {"standardization", "standardization_arcsinh"}:
            mean = np.asarray(self.target_mean, dtype=np.float32).reshape(1, 1, -1)
            var = np.asarray(self.target_var, dtype=np.float32).reshape(1, 1, -1)
            denom = np.sqrt(var + self.eps)
            if self.mode == "standardization_arcsinh":
                restored[valid] = np.sinh(restored[valid])
            restored[valid] = (restored * denom + mean)[valid]
        elif self.mode == "min-max":
            min_val = np.asarray(self.target_min_value, dtype=np.float32).reshape(1, 1, -1)
            max_val = np.asarray(self.target_max_value, dtype=np.float32).reshape(1, 1, -1)
            rng = np.maximum(max_val - min_val, self.eps)
            restored[valid] = (restored * rng + min_val)[valid]
        else:
            raise ValueError("Unsupported mode. Use ‘standardization’ or 'min-max'.")

        return restored[..., 0] if squeeze_last else restored

    def to_dict(self):
        return {
            "mode": self.mode,
            "eps": self.eps,
            "mean": None if self.mean is None else self.mean.tolist(),
            "var": None if self.var is None else self.var.tolist(),
            "min_value": None if self.min_value is None else self.min_value.tolist(),
            "max_value": None if self.max_value is None else self.max_value.tolist(),
            "target_mean": None if self.target_mean is None else self.target_mean.tolist(),
            "target_var": None if self.target_var is None else self.target_var.tolist(),
            "target_min_value": None if self.target_min_value is None else self.target_min_value.tolist(),
            "target_max_value": None if self.target_max_value is None else self.target_max_value.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        scaler = cls(mode=data["mode"], eps=data["eps"])
        if data["mean"] is not None:
            scaler.mean = np.array(data["mean"], dtype=np.float32)
        if data["var"] is not None:
            scaler.var = np.array(data["var"], dtype=np.float32)
        if data["min_value"] is not None:
            scaler.min_value = np.array(data["min_value"], dtype=np.float32)
        if data["max_value"] is not None:
            scaler.max_value = np.array(data["max_value"], dtype=np.float32)
        if data.get("target_mean") is not None:
            scaler.target_mean = np.array(data["target_mean"], dtype=np.float32)
        if data.get("target_var") is not None:
            scaler.target_var = np.array(data["target_var"], dtype=np.float32)
        if data.get("target_min_value") is not None:
            scaler.target_min_value = np.array(data["target_min_value"], dtype=np.float32)
        if data.get("target_max_value") is not None:
            scaler.target_max_value = np.array(data["target_max_value"], dtype=np.float32)
        return scaler

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return cls.from_dict(data)
