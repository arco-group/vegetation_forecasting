# Vegetation Health Forecasting Across Europe Using Satellite Time Series and Weather Data

[Irene Iele](https://scholar.google.com/citations?user=srLH7lkAAAAJ&hl=it&oi=ao)<sup>1</sup>,
[Paolo Soda](https://scholar.google.com/citations?user=E7rcYCQAAAAJ&hl=it&oi=ao)<sup>1,2</sup>,
[Matteo Tortora](https://matteotortora.github.io)<sup>3</sup>

<sup>1</sup>  University Campus Bio-Medico of Rome,
<sup>2</sup>  Umeå University,
<sup>3</sup>  University of Genoa
</div>

# Overview
Public release of the code for multi-output vegetation health forecasting using satellite time series and weather data.

<img width="2140" height="1042" alt="method_aia_3_page-0001" src="https://github.com/user-attachments/assets/952e798c-08e5-48dd-ab6c-9a3a210ee39f" />


## Main Scripts

- `dataset_builder/cache_builder_unified_monthly_noise_multi_out.py`: builds the multi-output cache from the raw CSV files and saves the scaler.
- `dataset_builder/torch_dataset_multi_out_fe.py`: loads the cached data as a PyTorch dataset, with optional feature engineering and scaling.
- `dataset_builder/scaler_multi_out.py`: handles feature and target scaling for single- and multi-output setups.
- `model/model_quantile_multi_out.py`: defines the quantile model and the pinball loss.
- `model/train_quantile_ablation_multi_out_fe.py`: trains the quantile model and supports ablation experiments.
- `model/test_quantile_multi_out_fe.py`: evaluates the trained model and exports metrics and predictions.
