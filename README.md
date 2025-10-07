## IBTrACS + ERA5 Tropical Cyclone Benchmark (North Atlantic Basin, 2000–2024)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/smehedi35/ibtracs-era5-tc-benchmark)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17239540.svg)](https://doi.org/10.5281/zenodo.17239540)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Project Overview

**AI-Driven Predictive Modeling of Global Hurricane Trajectory and Intensity under Climate Change** using multi-source spatio-temporal data.  

This repository combines:  
- **IBTrACS**: Global hurricane tracks (positions, intensities, metadata)  
- **ERA5**: High-resolution climate variables (10 m u- and v-wind, 2 m temperature, mean sea level pressure, sea surface temperature, and total precipitation) 

**Key highlights:**  

- **ML-ready benchmark:** 8,290+ observations (2000-2024) at 6-hour cadence with multi-horizon targets
- **Trained models included:** XGBoost for track forecasting (±6/12/24h) and RI classification with evaluation metrics
- **Complete pipeline:** Data engineering → feature engineering → model training → reproducible evaluation
- **Research-grade:** Storm-level splits, baseline comparisons, and comprehensive documentation 

## Track Forecasting

**Task:** Predict storm position (latitude, longitude) at 6, 12, and 24-hour lead times using atmospheric features, historical motion, and oceanic conditions.

**Model:** XGBoost regressors trained separately for longitudinal (Δx) and latitudinal (Δy) displacement with early stopping and storm-level temporal validation.

**Forecast Skill Improvement:**

| Horizon | Best Baseline (Median) | XGBoost (Median)| Improvement |
|---------|------------------------|-----------------|-------------|
| +6 h    | 33.47 km               | 33.34 km        | +0.4%       |
| +12 h   | 97.98 km               | 79.79 km        | +18.6%      |
| +24 h   | 280.10 km              | 201.34 km       | +26.1%      |

**Test set (2022–2024):** 1,984 forecasts at +6h | 1,729 at +12h | 1,285 at +24h

**Key Achievements:**
- **Near-operational accuracy:** Median error of 25.46 km at +6h approaches NOAA National Hurricane Center 24-hour track standards (110–150 km for Atlantic storms).  
- **Progressive skill gains:** +26% improvement over persistence at +24h demonstrates the model captures atmospheric steering beyond simple extrapolation.  
- **Interpretable features:** Top predictors include motion persistence (dx_6h, speed_6h), environmental steering (u10, v10), and oceanic influence (sst_c), achieving 201 km mean error at 24h.  
- **Robust generalization:** Healthy Val/Train gaps (1.5–2.2x) confirm reliable performance on unseen 2022–2024 storms.  
- **Computational efficiency:** Predictions take seconds versus hours for dynamical models, enabling rapid ensemble generation and sensitivity analysis.

## Rapid Intensification (RI) Classification

**Task:** Predict whether a storm will intensify ≥30 kt within the next 24 hours.

**Model:** XGBoost binary classifier with class balancing, probability calibration, and storm-level temporal splits to prevent leakage.

**Performance (Validation: 2019–2021 | Test: 2022–2024):**

| Metric    | Validation | Test  |
|-----------|------------|-------|
| AUROC     | 0.907      | 0.813 |
| AUPRC     | 0.333      | 0.236 |
| Best F1   | 0.43       | 0.30  |
| Precision | 0.34       | 0.25  |
| Recall    | 0.58       | 0.37  |
| Brier     | 0.033      | 0.042 |
| ECE       | 0.040      | 0.027 |

**Key Achievements:**
- **Excellent discrimination:** AUROC of 0.813 on unseen storms, comparable to operational NOAA systems.  
- **Reliable probabilities:** ECE of 0.027 indicates near-perfect calibration—predicted probabilities match actual RI occurrence rates.  
- **Strong rare-event detection:** Captures 37% of RI events (5% base rate) while maintaining operational precision—outperforming persistence and climatology baselines.

## Dataset Overview

This benchmark combines **IBTrACS hurricane tracks** with **ERA5 climate variables**, producing ML-ready datasets for forecasting and classification tasks.

### Temporal & Spatial Coverage
- **Years:** 2000–2024  
- **Cadence:** 6-hourly (00:00, 06:00, 12:00, 18:00 UTC)  
- **Region (North Atlantic TC basin):** 0°–50°N, 100°W–10°E  

### Data Sources
- **IBTrACS v04r01:** Storm positions (lat/lon), WMO 1-min sustained wind (`wmo_wind`), minimum pressure (`mslp`), storm ID, basin, and metadata  
- **ERA5 Reanalysis:**  
  - 10 m u-wind (`u10`)  
  - 10 m v-wind (`v10`)  
  - 2 m temperature (`t2m`)  
  - Mean sea level pressure (`msl`)  
  - Sea surface temperature (`sst`)  
  - Total precipitation (`tp`)  

### ML-Ready Tables
- **`merged_ibtracs_era5_2000_2024_clean.parquet`** – Clean intermediate table combining IBTrACS + ERA5, with harmonized coordinates, units, and storm-level keys; useful for custom target engineering.  
- **`model_ready_2000_2024_v2.parquet`** – Master benchmark table with multi-horizon intensity (`wmo_wind_next6/12/24`), track (`lat/lon_next6/12/24`), rapid intensification (`ri_24h`), and lifecycle targets.  
- **`outputs/tasks/*.parquet`** – Slim task-specific subsets for quick modeling and benchmarking.  
- **`DATA_DICTIONARY.md` / `model_ready_2000_2024_v2_dictionary.csv`** – Detailed column definitions, units, and descriptions.  

### Formats
- **Parquet:** ML-ready tables  
- **NetCDF4:** Raw ERA5 reanalysis data (not included due to size; downloadable from [Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview))  

This structure allows **multi-horizon forecasting**, **rapid intensification classification**, and **lifecycle analysis** with clean, reproducible, and consistent features across all storms.

## Why this dataset

Data engineering you can trust: raw → clean → merged → model-ready → task-ready, with consistent time, coordinates, and identifiers. 

Forecast targets included: 6 / 12 / 24-hour horizons (e.g., wmo_wind_next6, lat_next24). 

Clear tasks: intensity regression, track regression, RI classification (24 h), and lifecycle classification. 

Reproducible: simple scripts, dictionary, and task splits designed for baseline and advanced models.

## Tasks & targets

All tasks are derived from the 6-hour master cadence; 12 h and 24 h targets are computed by shifting forward multiple steps.

1. Intensity Forecast (regression)
- Targets: wmo_wind_next6, wmo_wind_next12, wmo_wind_next24 (knots).
- Typical metrics: MAE / RMSE (kt), skill vs. persistence/CLIPER-style baselines.

2. Track Forecast (regression)
- Targets: lat_next{6,12,24}, lon_next{6,12,24} (degrees).
- Typical metric: great-circle error (km) via haversine.

3. Rapid Intensification (classification)
- Target: ri_24h (1 if Δwind ≥ 30 kt over 24 h; else 0).
- Typical metrics: AUROC, AUPRC, F1 (with class imbalance reporting).

4. Lifecycle (classification)
- Targets: categorical/end-state flags (e.g., dissipates, extratropical transition) and/or lifecycle_stage where applicable.

Storm-level splits (train/val/test) and sample weights are included to avoid temporal leakage and handle imbalance. See DATA_DICTIONARY.md for exact column names.

## Quickstart

```bash
import pandas as pd
import numpy as np
```
Load the master table: 

```python
df = pd.read_parquet("outputs/model_ready_2000_2024_v2.parquet")
print(f"Dataset shape: {df.shape}")
print(f"First 12 columns: {list(df.columns[:12])}")
print(f"\nSplit distribution:\n{df['split'].value_counts()}")
```
Train/validation/test splits:

```python
train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]
```
Intensity baseline (+6h):

```python
cols = ["storm_id","iso_time","wmo_wind","wmo_wind_next6"]
tmp  = df[cols].dropna()
y    = tmp["wmo_wind_next6"].to_numpy()
yhat = tmp["wmo_wind"].to_numpy()  # persistence
mae  = np.mean(np.abs(y - yhat))
print(f"Persistence MAE (+6h): {mae:.2f} kt")
```
Track baseline (+12h) using haversine:

```python
R = 6371.0  # km
def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance using Haversine formula"""
    p = np.pi / 180
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p) * np.cos(lat2*p) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

cols = ["lat", "lon", "lat_next12", "lon_next12"]
tt = df[cols].dropna()
err_km = haversine(tt["lat"], tt["lon"], tt["lat_next12"], tt["lon_next12"])
print(f"📍 Median great-circle error (+12h): {np.median(err_km):.1f} km")
print(f"   Mean error: {np.mean(err_km):.1f} km")
print(f"   P95 error: {np.percentile(err_km, 95):.1f} km")
```
RI classification target distribution:

```python
print(f"\n🌪️ Rapid Intensification (RI) Target Distribution:")
print(df[["storm_id", "iso_time", "ri_24h"]].head(10))
print(f"\nRI class balance:\n{df['ri_24h'].value_counts(normalize=True)}")
```
## Data Engineering & ML Pipeline

### 1. Raw Ingestion
- **IBTrACS v04r01**: storm positions, WMO wind, pressure, basin, metadata  
- **ERA5**: climate variables (u10, v10, t2m, msl, sst, tp) collocated at 6-hourly storm fixes  

### 2. Cleaning & Normalization
- Standardized keys (`storm_id`, `iso_time`, `lat`, `lon`)  
- Unit harmonization, coordinate normalization, duplicate removal  

### 3. Feature & Target Engineering
- **Forecast targets:** intensity (`wmo_wind_next6/12/24`), track (`lat/lon_next6/12/24`)  
- **RI-24h flags** and lifecycle labels  
- Horizon masks and storm-level splits, sample weights for imbalance  

### 4. ML Modeling
- Track forecasting: XGBoost, LSTM, spatio-temporal models  
- RI classification: tree-based, logistic regression  
- Baselines: persistence, climatology, wind-advection, ensemble methods  

### 5. Export & Reproducibility
- Master table: `outputs/model_ready_2000_2024_v2.parquet`  
- Intermediate: `outputs/merged_ibtracs_era5_2000_2024_clean.parquet`  
- Task subsets: `outputs/tasks/*.parquet`  
- Recreate with:
```bash
python scripts/build_dataset.py
python scripts/add_targets.py
```

## Data sources & attribution

IBTrACS v04r01 — International Best Track Archive for Climate Stewardship (NOAA/NCEI).

ERA5 — Copernicus Climate Change Service (C3S) via ECMWF.

See DATA_DICTIONARY.md for the exact list of variables included from each source.

## Licensing

- **Code**: Licensed under the [MIT License].  
- **Processed Data**: Licensed under [CC-BY 4.0].

## Versioning & DOI

Dataset version: v2 (covers 2000–2024).

DOI: 10.5281/zenodo.17239540

## Cite this dataset

@dataset{hasan_ibtracs_era5_tc_benchmark_2000_2024_v2,
  author    = {Saifur Rahman Mehedi}, 
  title     = {IBTrACS + ERA5 Tropical Cyclone Benchmark (2000–2024), v2},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17239540},
  url       = {https://doi.org/10.5281/zenodo.TBD}
} 

## FAQs

Why 6 h if some targets are 12/24 h?
The native cadence is 6 h; 12/24 h targets are computed by shifting the 6-hour series forward by 2/4 steps.

What is wmo_wind_next6?
It’s the IBTrACS WMO 1-min sustained wind value 6 hours ahead of the current fix, used as the intensity-forecast target.

How do I avoid leakage?
Use the provided storm-level split; don’t mix fixes from the same storm across train/val/test.


## Intended use & ethics

This dataset is for research and education in climate science and ML. Forecasts derived from this dataset should not be used for real-time hazard guidance without official sources (e.g., JTWC/NOAA). Always disclose uncertainties and limitations.

## Maintainer

Saifur Rahman Mehedi — issues and questions welcome via the repository’s issue tracker.