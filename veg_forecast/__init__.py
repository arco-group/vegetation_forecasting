"""Probabilistic multi-output vegetation-index forecasting from sparse clear-sky Sentinel-2 time series.

Submodules that need PyTorch (`veg_forecast.model`, `veg_forecast.data.dataset`) are not imported here, so that the
cache builder runs on machines without PyTorch.
"""

__version__ = "2.0.0"
