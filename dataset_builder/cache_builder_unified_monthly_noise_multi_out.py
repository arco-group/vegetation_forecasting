import argparse
import json
import logging
import math

import numpy as np

from .cache_builder_unified import DatasetCacheBuilder, FeatureStats
from .cache_builder_unified_monthly_noise import MonthlyNoiseCacheBuilder
from .constants import VEGETATION_COLUMNS
from .scaler_multi_out import Scaler


class MonthlyNoiseCacheBuilderMultiOut(MonthlyNoiseCacheBuilder):
    """
    Multi-output version of the monthly cache builder.
    Saves future targets with shape (N, T, K) and adds a dedicated history for each target.
    """

    def __init__(self, *args, target_columns=None, **kwargs):
        if target_columns is None:
            target_columns = kwargs.pop("target_column", "avg_NDVI_clear_sky")
        if isinstance(target_columns, str):
            target_columns = [item.strip() for item in target_columns.split(",") if item.strip()]
        if not target_columns:
            raise ValueError("Specificare almeno un target in target_columns.")
        kwargs["target_column"] = list(target_columns)
        super().__init__(*args, **kwargs)
        self.target_columns = list(target_columns)
        self.target_column = list(target_columns)

    def build(self):
        self.compute_monthly_stats()
        groups = self.collect_groups()
        if not groups:
            self.logger.warning("Nessun CSV trovato in %s", self.data_root)
            return

        for area_name in self.progress(sorted(groups.keys()), "Aree"):
            self.process_area(area_name, groups[area_name])

        if self.stats_accumulator and self.feature_names:
            scaler = self._feature_stats_to_scaler(mode=self.scaler_mode)
            if self.target_stats:
                scaler = self._attach_target_stats(scaler, self.target_stats)
            scaler_path = self.cache_root / "scaler.json"
            scaler.save(scaler_path)
            self.logger.info("Salvato lo scaler globale in %s", scaler_path)

    def _feature_stats_to_scaler(self, mode="min-max"):
        scaler = Scaler(mode=mode, eps=1e-5)
        valid = self.stats_accumulator.count > 0
        num_features = self.stats_accumulator.num_features
        if mode in {"standardization", "standardization_arcsinh"}:
            mean = np.zeros(num_features, dtype=np.float32)
            var = np.ones(num_features, dtype=np.float32)
            mean[valid] = (self.stats_accumulator.sum[valid] / self.stats_accumulator.count[valid]).astype(np.float32)
            var_values = (self.stats_accumulator.sum_sq[valid] / self.stats_accumulator.count[valid]) - (mean[valid] ** 2)
            var_values = np.where(var_values < 0, 0, var_values)
            var[valid] = var_values.astype(np.float32)
            scaler.mean = mean
            scaler.var = var
        elif mode == "min-max":
            min_val = np.zeros(num_features, dtype=np.float32)
            max_val = np.ones(num_features, dtype=np.float32)
            min_val[valid] = self.stats_accumulator.min_value[valid].astype(np.float32)
            max_val[valid] = self.stats_accumulator.max_value[valid].astype(np.float32)
            scaler.min_value = min_val
            scaler.max_value = max_val
        else:
            raise ValueError("Modalita` non supportata per lo scaler.")
        return scaler

    def initialise_stats(self, num_features):
        if self.stats_accumulator is None:
            self.stats_accumulator = FeatureStats(num_features)
        if self.target_stats is None:
            self.target_stats = FeatureStats(len(self.target_columns))

    def _attach_target_stats(self, scaler, target_stats: FeatureStats):
        if scaler.mode in {"standardization", "standardization_arcsinh"}:
            scaler.target_mean = (target_stats.sum / np.maximum(target_stats.count, 1)).astype(np.float32)
            var_values = (target_stats.sum_sq / np.maximum(target_stats.count, 1)) - (scaler.target_mean ** 2)
            var_values = np.where(var_values < 0, 0, var_values)
            scaler.target_var = var_values.astype(np.float32)
        elif scaler.mode == "min-max":
            scaler.target_min_value = target_stats.min_value.astype(np.float32)
            scaler.target_max_value = target_stats.max_value.astype(np.float32)
        return scaler

    def prepare_features(self, dataframe):
        missing = [col for col in self.target_columns if col not in dataframe.columns]
        if missing:
            self.logger.warning("Colonne target assenti: %s", missing)
            return None, None

        feature_columns = [col for col in dataframe.columns if col not in self.target_columns]
        drop_columns = [col for col in VEGETATION_COLUMNS if col not in self.target_columns and col in feature_columns]
        feature_frame = dataframe.drop(columns=drop_columns)
        feature_columns = [col for col in feature_frame.columns if col not in self.target_columns]
        feature_frame = feature_frame[feature_columns]
        target_frame = dataframe[self.target_columns].copy()
        return feature_frame, target_frame

    def process_area(self, area_name, csv_files):
        samples_history = []
        samples_history_mask = []
        samples_history_timestamps = []
        samples_future = []
        samples_future_mask = []
        samples_future_noise = []
        samples_future_timestamps = []
        samples_target = []
        samples_target_mask = []
        samples_target_timestamps = []
        samples_future_target_positions = []
        samples_climate = []
        metadata_records = []

        climate_value = self.climate_map[area_name]

        for csv_path in self.progress(csv_files, f"{area_name}"):
            dataframe = self.load_csv(csv_path)
            if dataframe is None or dataframe.empty:
                continue

            feature_frame, target_frame = self.prepare_features(dataframe)
            if feature_frame is None:
                continue

            columns = feature_frame.columns.tolist()
            history_target_names = [f"target_history_{name}" for name in self.target_columns]
            if self.feature_names is None:
                self.base_feature_columns = columns
                self.feature_names = columns + history_target_names
                self.initialise_stats(len(self.feature_names))
            elif columns != self.base_feature_columns:
                self.logger.error("L'ordine delle colonne non coincide in %s", csv_path)
                raise ValueError("Colonne incoerenti fra i CSV. Uniformare prima i dati.")

            feature_values = feature_frame.to_numpy(dtype=np.float32)
            feature_masks = np.isnan(feature_values)
            target_values = target_frame.to_numpy(dtype=np.float32)

            timestamps = feature_frame.index.to_numpy()
            total_rows = len(feature_frame)

            valid_rows = ~np.isnan(target_values).any(axis=1)
            valid_positions = np.flatnonzero(valid_rows).tolist()
            if not valid_positions:
                continue
            if self.step <= 0:
                start_positions = valid_positions
            else:
                start_positions = [valid_positions[i] for i in range(0, len(valid_positions), self.step)]

            for start in start_positions:
                history_indices = []
                history_target_count = 0
                ptr = start

                while ptr < total_rows and history_target_count < self.history_window:
                    history_indices.append(ptr)
                    if valid_rows[ptr]:
                        history_target_count += 1
                    ptr += 1

                if history_target_count < self.history_window:
                    continue

                history_slice = feature_values[history_indices]
                history_ts = timestamps[history_indices]
                target_history_block = np.full((len(history_indices), len(self.target_columns)), np.nan, dtype=np.float32)
                for pos, original_idx in enumerate(history_indices):
                    row = target_values[original_idx]
                    valid = ~np.isnan(row)
                    if valid.any():
                        target_history_block[pos, valid] = row[valid]

                history_slice = np.concatenate([history_slice, target_history_block], axis=1)
                history_mask = np.isnan(history_slice)

                future_indices = []
                future_target_values = []
                future_target_masks = []
                future_target_timestamps = []
                future_target_positions = []
                while ptr < total_rows and len(future_target_values) < self.forecast_window:
                    future_indices.append(ptr)
                    row = target_values[ptr]
                    row_mask = np.isnan(row)
                    if not row_mask.any():
                        future_target_values.append(row.astype(np.float32))
                        future_target_masks.append(row_mask.astype(bool))
                        future_target_timestamps.append(str(timestamps[ptr]))
                        future_target_positions.append(len(future_indices) - 1)
                    ptr += 1

                if len(future_target_values) < self.forecast_window:
                    continue

                future_slice = feature_values[future_indices]
                future_mask = feature_masks[future_indices]
                future_ts = timestamps[future_indices]
                future_target_history_block = np.full((len(future_indices), len(self.target_columns)), np.nan, dtype=np.float32)
                future_slice = np.concatenate([future_slice, future_target_history_block], axis=1)
                future_mask = np.concatenate([future_mask, np.isnan(future_target_history_block)], axis=1)

                target_slice = np.stack(future_target_values).astype(np.float32)
                target_mask = np.stack(future_target_masks).astype(bool)

                future_noisy, noise = self.add_noise_to_future(
                    future_slice,
                    future_mask,
                    future_ts,
                    history_ts[-1],
                    area_name,
                )

                if self.stats_accumulator:
                    self.stats_accumulator.update(history_slice)
                    self.stats_accumulator.update(future_slice)
                    if self.target_stats is not None:
                        self.target_stats.update(target_slice.reshape(-1, len(self.target_columns)))

                samples_history.append(history_slice)
                samples_history_mask.append(history_mask)
                samples_history_timestamps.append([str(ts) for ts in history_ts])
                samples_future.append(future_noisy)
                samples_future_mask.append(future_mask)
                samples_future_noise.append(noise)
                samples_future_timestamps.append([str(ts) for ts in future_ts])
                samples_target.append(target_slice)
                samples_target_mask.append(target_mask)
                samples_target_timestamps.append(future_target_timestamps)
                samples_future_target_positions.append(future_target_positions)
                samples_climate.append(climate_value)

                record = {
                    "area": area_name,
                    "csv": csv_path.name,
                    "history_start": samples_history_timestamps[-1][0],
                    "history_end": samples_history_timestamps[-1][-1],
                    "future_start": future_target_timestamps[0],
                    "future_end": future_target_timestamps[-1],
                    "climate": climate_value,
                }
                metadata_records.append(record)

        if not samples_history:
            self.logger.info("Nessun campione valido per %s", area_name)
            return

        history_array = np.array(samples_history, dtype=object)
        history_mask_array = np.array(samples_history_mask, dtype=object)
        history_ts_array = np.array(samples_history_timestamps, dtype=object)
        future_array = np.array(samples_future, dtype=object)
        future_mask_array = np.array(samples_future_mask, dtype=object)
        future_noise_array = np.array(samples_future_noise, dtype=object)
        future_ts_array = np.array(samples_future_timestamps, dtype=object)
        target_array = np.stack(samples_target).astype(np.float32)
        target_mask_array = np.stack(samples_target_mask).astype(bool)
        target_ts_array = np.array(samples_target_timestamps, dtype=object)
        future_target_pos_array = np.array(samples_future_target_positions, dtype=np.int64)

        cache_path = self.cache_root / f"{area_name}.npz"
        np.savez_compressed(
            cache_path,
            history=history_array,
            history_mask=history_mask_array,
            history_timestamps=history_ts_array,
            future=future_array,
            future_mask=future_mask_array,
            future_noise=future_noise_array,
            future_timestamps=future_ts_array,
            target=target_array,
            target_mask=target_mask_array,
            future_target_positions=future_target_pos_array,
            target_timestamps=target_ts_array,
            climate=np.array(samples_climate),
        )

        metadata = {
            "area": area_name,
            "feature_names": self.feature_names,
            "target_names": self.target_columns,
            "target_column": self.target_columns,
            "history_window": self.history_window,
            "forecast_window": self.forecast_window,
            "step": self.step,
            "noise_percentage": self.noise_percentage,
            "scaler_mode": self.scaler_mode,
            "min_samples_per_month": self.min_samples_per_month,
            "g5_target": self.g5_target,
            "metadata_records": metadata_records,
        }

        metadata_path = self.cache_root / f"{area_name}.json"
        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        self.logger.info("Creati %d campioni per %s", len(samples_history), area_name)


def parse_target_columns(value):
    targets = [item.strip() for item in value.split(",") if item.strip()]
    if not targets:
        raise argparse.ArgumentTypeError("Specificare almeno una colonna target.")
    return targets


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-output cache generator with monthly noise per area.")
    parser.add_argument("--data-root", type=str, default="timeSeries/global/train", help="Root directory containing the source CSV files.")
    parser.add_argument("--cache-root", type=str, default="timeSeries/cache/train_multi_out", help="Destination directory for saving the caches.")
    parser.add_argument("--metadata-path", type=str, default="timeSeries/dataSummary.csv", help="Path to the metadata file (dataSummary.csv).")
    parser.add_argument("--legend-path", type=str, default="timeSeries/csv_legend.xlsx", help="Path to the column legend file.")
    parser.add_argument("--target-columns", type=parse_target_columns, default=["avg_NDVI_clear_sky"], help="Comma-separated list of target columns.")
    parser.add_argument("--history-window", type=int, default=5, help="Number of past Sentinel observations to include.")
    parser.add_argument("--forecast-window", type=int, default=5, help="Number of future Sentinel observations to predict.")
    parser.add_argument("--step", type=int, default=6, help="Interval (in timesteps) used to sample windows.")
    parser.add_argument("--noise-percentage", type=float, default=0.1, help="Relative Gaussian noise amplitude to apply to future features.")
    parser.add_argument("--min-samples-per-month", type=int, default=100, help="Minimum number of monthly samples required to use the local area std.")
    parser.add_argument("--g5-target", type=float, default=2.0, help="Desired growth factor at the last step (g5).")
    parser.add_argument("--scaler-mode", type=str, default="min-max", choices=("min-max", "standardization", "standardization_arcsinh"), help="Type of scaler to save with the cache.")
    parser.add_argument("--add-temporal", action=argparse.BooleanOptionalAction, default=True, help="Add derived temporal features (disable with --no-add-temporal).")
    parser.add_argument("--fourier-harmonics", type=int, default=3, help="Number of Fourier harmonics to compute for temporal features.")
    parser.add_argument("--include-hour", action="store_true", help="Include the hour of day among temporal features.")
    parser.add_argument("--include-day", action="store_true", help="Include the day of week among temporal features.")
    parser.add_argument("--include-month", action="store_true", help="Include the month among temporal features.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random noise generation.")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True, help="Show progress bars (disable with --no-show-progress).")
    parser.add_argument("--group-mode", type=str, default="directory", choices=("directory", "pattern"), help="CSV grouping mode (by directory or by filename pattern).")
    parser.add_argument("--pattern-index", type=int, default=2, help="Token index in the filename to use as the area identifier (when group-mode=pattern).")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    cli_args = parse_args()
    builder = MonthlyNoiseCacheBuilderMultiOut(**vars(cli_args))
    builder.build()
