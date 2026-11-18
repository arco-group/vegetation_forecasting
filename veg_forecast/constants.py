"""Column names shared by the data pipeline and the model."""

# Vegetation indices forecast by the released model, in output order.
VIS = ["NDVI", "GNDVI", "EVI"]
TARGET_COLUMNS = [f"avg_{vi}_clear_sky" for vi in VIS]

# Daily meteorological and cloud-fraction columns of the time-series CSVs. Synthetic future-covariate noise is
# applied to these columns (in this order) when the cache is built.
METEO_COLUMNS = [
    "wind",
    "humidity",
    "solar_irradiance",
    "rainfall",
    "pressure",
    "avg_temperature",
    "min_temperature",
    "max_temperature",
    "thick_clouds",
    "thin_clouds",
    "clouds_shadows",
]

# Satellite-derived columns of the CSVs. They are dropped from the inputs, except the requested targets, which
# enter the model only through their history (target_history_* features).
_STATS = ["avg", "std", "Q1", "Q2", "Q3"]
_INDICES = ["RVI", "NDVI", "GNDVI", "EVI", "SAVI"]
_MASKS = ["all", "no_thick", "no_clouds", "clear_sky"]
VEGETATION_COLUMNS = [f"{s}_{i}_{m}" for m in _MASKS for i in _INDICES for s in _STATS] + [
    "s2_avail",
    "avg_NDVI_clear_sky_interpolated",
    "avg_NDVI_all_filtered",
]

# Weather features computed on the fly by the dataset (inserted before the target-history block).
ENGINEERED_FEATURES = [
    "rain_cum_between_targets",
    "temp_lt_10_between_targets",
    "temp_gt_30_between_targets",
    "rain_cum_7d",
    "rain_cum_14d",
    "temp_lt_10_7d",
    "temp_gt_30_7d",
    "temp_lt_10_14d",
    "temp_gt_30_14d",
]

# Features removed from the FUTURE branch (they stay available in the history branch):
# - future satellite cloud fractions are not known at forecast time;
# - the *_between_targets features are defined only at target days, so in the future branch they would reveal
#   when the next clear-sky acquisitions happen.
FUTURE_MASKED_FEATURES = [
    "thick_clouds",
    "thin_clouds",
    "clouds_shadows",
    "rain_cum_between_targets",
    "temp_lt_10_between_targets",
    "temp_gt_30_between_targets",
]
