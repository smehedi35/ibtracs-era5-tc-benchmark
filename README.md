## IBTrACS + ERA5 Tropical Cyclone Benchmark (USA, 2000–2024)


[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/smehedi35/ibtracs-era5-tc-benchmark)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17239540.svg)](https://doi.org/10.5281/zenodo.17239540)


A clean, citable, ML-ready dataset merging NOAA IBTrACS storm tracks with ERA5 reanalysis at 6-hour cadence, with slim task subsets for intensity, track, rapid-intensification, and lifecycle prediction.

## Why this dataset

Data engineering you can trust: raw → clean → merged → model-ready → task-ready, with consistent time, coordinates, and identifiers. 

Forecast targets included: 6 / 12 / 24-hour horizons (e.g., wmo_wind_next6, lat_next24). 

Clear tasks: intensity regression, track regression, RI classification (24 h), and lifecycle classification. 

Reproducible: simple scripts, dictionary, and task splits designed for baseline and advanced models.

## Key artifacts

 outputs/model_ready_2000_2024_v2.parquet: master benchmark table at 6-hour cadence with harmonized IBTrACS + ERA5 features and multi-horizon targets.

 outputs/merged_ibtracs_era5_2000_2024_clean.parquet: merged & cleaned intermediate table (pre-target engineering), useful for custom target design.

 outputs/DATA_DICTIONARY.md and model_ready_2000_2024_v2_dictionary.csv: human- and machine-readable schemas (variables, units, descriptions).

 outputs/tasks/*.parquet: slim, task-specific subsets for quick modeling and benchmarking. 

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

Load the master table: 

```python
import pandas as pd

df = pd.read_parquet("outputs/model_ready_2000_2024_v2.parquet")
print(df.shape, df.columns[:12])
```
Train/test split usage:

```python
train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]
```
Example: intensity baseline (persistence @ +6 h)

```python
import numpy as np

cols = ["storm_id","iso_time","wmo_wind","wmo_wind_next6"]
tmp  = df[cols].dropna()
y    = tmp["wmo_wind_next6"].to_numpy()
yhat = tmp["wmo_wind"].to_numpy()        # persistence baseline
mae  = np.mean(np.abs(y - yhat))
print(f"Persistence MAE (+6h): {mae:.2f} kt")
```
Example: track error (haversine, +12 h)

```python
import numpy as np

R = 6371.0  # km
def haversine(lat1, lon1, lat2, lon2):
    p = np.pi/180
    dlat = (lat2-lat1)*p
    dlon = (lon2-lon1)*p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

cols = ["lat","lon","lat_next12","lon_next12"]
tt = df[cols].dropna()
err_km = haversine(tt["lat"], tt["lon"], tt["lat_next12"], tt["lon_next12"])
print(f"Median great-circle error (+12h): {np.median(err_km):.1f} km")
```

## How the data were built

1. Raw ingestion

- IBTrACS v04r01 global tracks (storm positions, WMO 1-min sustained wind, min pressure, basin, metadata).
- ERA5 reanalysis variables collocated/spatiotemporally joined to storm fixes (6-hourly cadence).

2. Cleaning & normalization

- Standardized keys: storm_id, iso_time (UTC, 6-hour steps), lat, lon.
- Coordinate normalization, unit harmonization, duplicate handling.

3. Feature & target engineering

- Multi-horizon targets: {6,12,24}-hour shifts for intensity (wmo_wind) and position (lat, lon).
- Masks for horizon availability near storm end (h{6,12,24}_mask), RI-24h flag, lifecycle labels.
- Storm-level split assignment and sample_weight for imbalanced tasks.

4. Export

- Master: model_ready_2000_2024_v2.parquet.
- Intermediates: merged_ibtracs_era5_2000_2024_clean.parquet.
- Slim task files under outputs/tasks/.

Recreate with scripts/build_dataset.py (ETL) followed by scripts/add_targets.py (targets & masks). Paths and variable lists are defined in the script headers.

## Data sources & attribution

IBTrACS v04r01 — International Best Track Archive for Climate Stewardship (NOAA/NCEI).
DOI/Access: public archive (cite NOAA/NCEI IBTrACS).

ERA5 — Copernicus Climate Change Service (C3S) via ECMWF.
Please follow the C3S/ECMWF attribution guidelines.

See DATA_DICTIONARY.md for the exact list of variables included from each source.

## Licensing

Data: CC BY 4.0 (plus IBTrACS/ERA5 attribution)

Code: MIT License. 

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

## Reproducibility notes

Environment: Python ≥ 3.9, packages: pandas, numpy, xarray, netCDF4, pyarrow (or polars).

Regeneration (high-level):
- Configure input paths in scripts/build_dataset.py.
- Run the script to produce merged_ibtracs_era5_2000_2024_clean.parquet.
- Run scripts/add_targets.py to compute targets, masks, splits, and task files.

Example:

python scripts/build_dataset.py
python scripts/add_targets.py

## Intended use & ethics

This dataset is for research and education in climate science and ML. Forecasts derived from this dataset should not be used for real-time hazard guidance without official sources (e.g., JTWC/NOAA). Always disclose uncertainties and limitations.

## Changelog

v2 (current): unified 2000–2024 master; improved dictionary; task subsets refreshed.

v1: initial merge & targets.

## Maintainer

Saifur Rahman Mehedi — issues and questions welcome via the repository’s issue tracker.