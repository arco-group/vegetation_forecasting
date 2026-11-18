"""Build the forecasting cache of one split from the time-series CSVs (see veg_forecast/data/cache_builder.py).

Paper setting (history 3 and forecast 3 valid observations, a window every 4 observations, 10% future-weather
noise, seed 42):

    python scripts/build_cache.py --data-root <csv>/train         --cache-root data/cache/train --group-mode directory
    python scripts/build_cache.py --data-root <csv>/val_chopped   --cache-root data/cache/val   --group-mode pattern
    (same for ood-t_chopped, ood-s_chopped, ood-st_chopped)

Only the training cache's scaler.json is used (by training and, through weights/scaler.json, at inference).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veg_forecast.data.cache_builder import main  # noqa: E402

if __name__ == "__main__":
    main()
