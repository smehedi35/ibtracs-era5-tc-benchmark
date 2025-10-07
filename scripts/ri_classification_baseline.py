#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rapid Intensification (RI) Classification Baseline
==================================================

This script trains and evaluates a binary classifier that predicts whether a
tropical cyclone will undergo Rapid Intensification (RI) within the next
24 hours. It is designed for *reproducibility* and *operational clarity*,
with careful attention to feature engineering, temporal splits, calibration,
and metrics logging.

Highlights
----------
- Robust feature engineering pipeline (track kinematics, lags/rollups,
  environmental lags, cyclic encodings, target encoding).
- Flexible model family via a compact "stacked" ensemble wrapper supporting
  LightGBM and XGBoost sub-models, plus optional logistic top layer.
- Probability calibration (auto-select between isotonic and sigmoid) with
  Brier/ECE reporting and operational thresholding derived from validation.
- Strict time-aware train/val/test split to avoid leakage.
- Self-documenting artifacts written to `outdir` for auditability:
  metrics.json, ops_on_test.json, feature importances, PR/ROC/reliability
  curves, predictions parquet, and full config snapshot.
- No business logic changes should be made inside `main()` per request; all
  comments added are non-invasive and focus on clarity and maintainability.

Usage (example)
---------------
python scripts/ri_classification_baseline.py \
  --model lgbm --calibrate auto --ensemble_size 5 \
  --stack_include_deltas --env_lag_vars 16 --stack_top_k 14

Author: (Your name / lab)
"""

from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- scikit-learn primitives for pipelines and evaluation ---
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import mutual_info_classif

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Optional gradient boosting libraries (used if available) ---
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

# ------------------------
# Global constants / paths
# ------------------------
RANDOM_STATE = 137

# Default output directory for this flavor of experiment
OUTDIR = Path("outputs/ri_classification_plus")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Fallback set of candidate dataset paths (first existing will be used)
DEFAULT_CANDIDATE_PATHS = [
    "outputs/model_ready_2000_2024_v2.parquet",
    "outputs/model_ready_2000_2024_v1.parquet",
    "outputs/tc_ri_classification_v1.parquet",
]

# Dictionary files mapping columns→source (used to auto-discover ERA5 vs IBTrACS)
DICT_CSV_CANDIDATES = [
    "outputs/model_ready_2000_2024_v2_dictionary.csv",
    "outputs/model_ready_2000_2024_v1_dictionary.csv",
]


def find_first_existing(paths: List[str]) -> Optional[Path]:
    """
    Return the first path that exists from a candidate list.

    This allows running the script without hardcoding a single dataset path.
    """
    for p in paths:
        pth = Path(p)
        if pth.exists():
            return pth
    return None


def to_datetime_utc(s: pd.Series) -> pd.Series:
    """
    Parse timestamps into pandas datetime with UTC timezone, coercing errors.

    If the incoming series is already tz-aware, ensure UTC; otherwise make it UTC.
    """
    t = pd.to_datetime(s, errors="coerce", utc=True)
    return t.dt.tz_convert("UTC") if t.dt.tz is not None else t


def safe_ratio(a: int, b: int, default: float = 1.0) -> float:
    """
    Compute a/b with protection against division by zero.
    """
    return float(a / b) if b else default


def norm_lon(s: pd.Series) -> pd.Series:
    """
    Normalize longitudes into the canonical [-180, 180) range.
    """
    return ((s + 180.0) % 360.0) - 180.0


def has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    """
    True if all columns in `cols` exist in `df`.
    """
    return all(c in df.columns for c in cols)


def detect_sources(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Infer which columns are ERA5 numeric features vs. IBTrACS fields vs. categorical.

    Priority:
    1) Use an available dictionary CSV (column, source) if present.
    2) Fallback heuristics: prefix-based detection (e.g., "era5_").
    """
    era5_cols, ib_cols, cat_cols = [], [], []
    dict_path = find_first_existing(DICT_CSV_CANDIDATES)
    if dict_path is not None:
        try:
            dd = pd.read_csv(dict_path)
            if {"column", "source"}.issubset(dd.columns):
                for _, r in dd.iterrows():
                    c = r["column"]
                    if c not in df.columns:
                        continue
                    src = str(r["source"]).lower()
                    if "era5" in src:
                        era5_cols.append(c)
                    elif "ibtracs" in src:
                        ib_cols.append(c)
        except Exception:
            # Non-fatal; we fall back to heuristics
            pass

    # Heuristic fallback: anything numeric starting with era5_
    if not era5_cols:
        era5_cols = [c for c in df.columns if str(c).lower().startswith("era5_") and pd.api.types.is_numeric_dtype(df[c])]

    # Minimal IBTrACS schema used throughout the pipeline
    if not ib_cols:
        ib_cols = [c for c in ["storm_id","iso_time","lat","lon","wmo_wind","wmo_pres"] if c in df.columns]

    # Keep human-readable categorical metadata (safe to one-hot if needed)
    for c in ["basin","storm_status","storm_name"]:
        if c in df.columns:
            cat_cols.append(c)
    return era5_cols, ib_cols, cat_cols


def ensure_label(df: pd.DataFrame, ri_threshold_kt: int = 30) -> pd.DataFrame:
    """
    Ensure a binary RI label exists.

    - If a column named RI/ri_label already exists, use it (as int).
    - Otherwise, compute: (wmo_wind_next24 - wmo_wind) ≥ `ri_threshold_kt`.

    Assumes 6-hourly records; a 24h delta corresponds to shift(-4).
    """
    df = df.copy()
    existing = [c for c in df.columns if c.lower() in {"ri","ri_label"}]
    if existing:
        df["RI"] = df[existing[0]].astype(int)
        return df

    if "wmo_wind_next24" not in df.columns:
        df["wmo_wind_next24"] = df.groupby("storm_id")["wmo_wind"].shift(-4)

    m = df["wmo_wind"].notna() & df["wmo_wind_next24"].notna()
    df = df.loc[m].copy()
    df["RI"] = ((df["wmo_wind_next24"] - df["wmo_wind"]) >= ri_threshold_kt).astype(int)
    return df


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add core time/space and simple derived environmental features.

    - Year/month/hour and sine/cosine of day-of-year (seasonality)
    - Absolute latitude and normalized longitude
    - 10m wind magnitude/angle (if u10/v10 present)
    - 200–850 hPa shear magnitude/angle (if components present)
    - Simple pressure/wind interactions
    - Lightweight interactions for SST and moisture vs shear
    """
    df = df.copy()
    tcol = "iso_time" if "iso_time" in df.columns else "time"
    t = to_datetime_utc(df[tcol])
    df["year"] = t.dt.year.astype("Int64")
    df["month"] = t.dt.month.astype("Int64")
    df["hour"] = t.dt.hour.astype("Int64")
    doy = t.dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    df["doy_cos"] = np.cos(2*np.pi*doy/365.25)

    df["abs_lat"] = df["lat"].abs()
    df["lon_norm"] = norm_lon(df["lon"])

    # 10m wind vector features
    u10 = next((c for c in df.columns if c.lower() in {"era5_u10","u10","u10m"}), None)
    v10 = next((c for c in df.columns if c.lower() in {"era5_v10","v10","v10m"}), None)
    if u10 and v10:
        df["wind10_mag"] = np.sqrt(df[u10]**2 + df[v10]**2)
        ang = np.arctan2(df[v10], df[u10])
        df["wind10_dir_sin"] = np.sin(ang)
        df["wind10_dir_cos"] = np.cos(ang)

    # Deep-layer shear 200–850
    if has_cols(df, ["era5_u200","era5_v200","era5_u850","era5_v850"]):
        du = df["era5_u200"] - df["era5_u850"]
        dv = df["era5_v200"] - df["era5_v850"]
        shear = np.sqrt(du**2 + dv**2)
        df["shear_200_850"] = shear.clip(0.0, 80.0)
        angs = np.arctan2(dv, du)
        df["shear_dir_sin"] = np.sin(angs)
        df["shear_dir_cos"] = np.cos(angs)

    # Pressure-wind interactions (simple intensity proxies)
    if has_cols(df, ["wmo_wind","wmo_pres"]):
        pres = df["wmo_pres"].replace(0, np.nan)
        df["wind_pres_ratio"] = df["wmo_wind"] / pres
        df["neg_pres"] = -df["wmo_pres"]
        df["wind_x_negpres"] = df["wmo_wind"] * df["neg_pres"]

    # Mildly physics-informed interactions
    if "era5_sst" in df.columns and "abs_lat" in df.columns:
        df["sst_lat_x"] = df["era5_sst"] * (1 + 0.02 * (30 - df["abs_lat"]).clip(lower=-30, upper=30))
    if "era5_tcwv" in df.columns and "shear_200_850" in df.columns:
        df["moist_shear_interact"] = df["era5_tcwv"] / (1.0 + df["shear_200_850"])
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in km via Haversine. Inputs can be arrays/series.
    """
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(norm_lon(lon2) - norm_lon(lon1))
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.0)**2
    return 2*r*np.arcsin(np.sqrt(a))


def add_track_kinematics(df: pd.DataFrame, by: str = "storm_id") -> pd.DataFrame:
    """
    Add low-order kinematics from the IBTrACS track:

    - Translational speed over {6,12,18,24}h windows (km/h).
    - Forward direction (sin/cos) using 6h forward difference.
    - Simple curvature proxy using 12h turning angle.
    """
    df = df.copy()

    # Speed over historical windows
    speeds = {}
    for k, hours in [(1,6),(2,12),(3,18),(4,24)]:
        lat_lag = df.groupby(by)["lat"].shift(k)
        lon_lag = df.groupby(by)["lon"].shift(k)
        dist_km = haversine_km(lat_lag, lon_lag, df["lat"], df["lon"])
        speeds[f"track_speed_{hours}h"] = dist_km / (hours/1.0)

    # Forward direction from current → next 6h
    lat_f = df.groupby(by)["lat"].shift(-1)
    lon_f = df.groupby(by)["lon"].shift(-1)
    dlat = lat_f - df["lat"]
    dlon = norm_lon(lon_f) - norm_lon(df["lon"])
    ang = np.arctan2(dlat, dlon)
    track_dir_sin = np.sin(ang)
    track_dir_cos = np.cos(ang)

    # Curvature proxy using heading change (12h)
    lat_b = df.groupby(by)["lat"].shift(2)
    lon_b = df.groupby(by)["lon"].shift(2)
    dlat_prev = df["lat"] - lat_b
    dlon_prev = norm_lon(df["lon"]) - norm_lon(lon_b)
    ang_prev = np.arctan2(dlat_prev, dlon_prev)
    track_curv_12h = np.sin(ang - ang_prev).fillna(0.0)

    block = {
        **speeds,
        "track_dir_sin": track_dir_sin,
        "track_dir_cos": track_dir_cos,
        "track_curv_12h": track_curv_12h
    }
    return pd.concat([df, pd.DataFrame(block, index=df.index)], axis=1)


def add_lags_and_rolling(df: pd.DataFrame, by: str = "storm_id") -> pd.DataFrame:
    """
    Add lagged values, deltas, and rolling statistics for intensity proxies
    (`wmo_wind`, `wmo_pres`) over multiple horizons.

    This provides short-term temporal context for the model while avoiding
    look-ahead leakage (purely lagged, not forward-looking).
    """
    df = df.copy()
    newc = {}
    for col in [c for c in ["wmo_wind","wmo_pres"] if c in df.columns]:
        grp = df.groupby(by)[col]

        # Multi-horizon lags and finite differences
        for k, hours in [(1,6),(2,12),(3,18),(4,24),(6,36),(8,48)]:
            newc[f"{col}_lag{hours}h"] = grp.shift(k)
            newc[f"{col}_d{hours}h"] = df[col] - newc[f"{col}_lag{hours}h"]

        # 12/24h rolling mean/std
        for win, label in [(2,"12h"),(4,"24h")]:
            roll = grp.rolling(win, min_periods=1)
            newc[f"{col}_rollmean_{label}"] = roll.mean().reset_index(level=0, drop=True)
            newc[f"{col}_rollstd_{label}"] = roll.std().reset_index(level=0, drop=True)
    return pd.concat([df, pd.DataFrame(newc, index=df.index)], axis=1)


def add_env_lags(df: pd.DataFrame, era5_cols: List[str], by: str = "storm_id", max_vars: int = 24) -> pd.DataFrame:
    """
    Add compact environmental memory:

    - Pick the highest-variance ERA5 columns (proxy for signal strength).
    - For each, add 6h and 12h lags and deltas (current - lag).

    `max_vars` bounds the footprint to remain computationally practical.
    """
    df = df.copy()
    numeric_era5 = [c for c in era5_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_era5:
        return df

    v = df[numeric_era5].var(numeric_only=True).sort_values(ascending=False)
    topk = [c for c in v.index[:max_vars]]

    newc = {}
    for col in topk:
        grp = df.groupby(by)[col]
        for k, hours in [(1,6),(2,12)]:
            newc[f"{col}_lag{hours}h"] = grp.shift(k)
            newc[f"{col}_d{hours}h"] = df[col] - newc[f"{col}_lag{hours}h"]
    return pd.concat([df, pd.DataFrame(newc, index=df.index)], axis=1)


def pick_top_base_predictors_for_windowing(
    df: pd.DataFrame, era5_cols: List[str], train_mask: np.ndarray, top_k: int = 18
) -> List[str]:
    """
    Select a base set of strong predictors to *window-stack* across multiple
    past hours. The intent is to limit dimensionality explosion.

    Strategy:
    - Start from a curated pool of physics-informed features (intensity,
      shear, SST, motion).
    - Augment with top-variance ERA5 numerics.
    - Rank via Mutual Information on TRAIN ONLY (time-safe).
    """
    core_pool = [c for c in [
        "wmo_wind","wmo_pres","shear_200_850","wind10_mag","era5_sst","moist_shear_interact",
        "track_speed_12h","track_speed_24h","track_curv_12h"
    ] if c in df.columns]
    era5_num = [c for c in era5_cols if pd.api.types.is_numeric_dtype(df[c])]

    era5_top = []
    if era5_num:
        v = df.loc[train_mask, era5_num].var(numeric_only=True).sort_values(ascending=False)
        era5_top = list(v.index[: max(0, top_k*2 - len(core_pool))])

    pool = list(dict.fromkeys(core_pool + era5_top))

    # MI ranking (fallback to variance if MI fails due to NaNs etc.)
    X = df.loc[train_mask, pool].replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df.loc[train_mask, "RI"].astype(int).values
    try:
        mi = mutual_info_classif(X.values, y, random_state=RANDOM_STATE)
        order = np.argsort(mi)[::-1]
        top = [pool[i] for i in order[:top_k]]
    except Exception:
        var = X.var(numeric_only=True).sort_values(ascending=False)
        top = list(var.index[:top_k])
    return top


def add_window_stack(
    df: pd.DataFrame, cols: List[str], hours: List[int],
    group_col: str = "storm_id", include_deltas: bool = False
) -> pd.DataFrame:
    """
    For each selected `cols`, create features at specific past times:

    Example: for `hours=[6,12,24]`, create `col_t-6h`, `col_t-12h`, `col_t-24h`.
    Optionally also create deltas relative to the current value.

    Notes:
    - Assumes 6-hour time step; hours are rounded to the nearest 6h step.
    - Grouped by storm to avoid cross-storm leakage.
    """
    df = df.copy()
    step = 6
    new_cols = {}
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        grp = df.groupby(group_col)[col]
        for h in hours:
            k = int(round(h / step))
            lag_col = f"{col}_t-{h}h"
            new_cols[lag_col] = grp.shift(k)
            if include_deltas:
                new_cols[f"{col}_d{h}h_stack"] = df[col] - new_cols[lag_col]
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def target_encode_mean_smooth(
    train_s: pd.Series, train_y: np.ndarray, test_s: pd.Series, m: float = 50.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Target encode a categorical series with additive smoothing (mean prior).

    te = (count * mean_cat + m * global_mean) / (count + m)

    Returns (mapped_train, mapped_test).
    """
    global_mean = float(np.mean(train_y))
    stats = pd.DataFrame({"cat": train_s, "y": train_y}).groupby("cat")["y"].agg(["mean","count"])
    stats["te"] = (stats["count"]*stats["mean"] + m*global_mean)/(stats["count"]+m)
    train_map = train_s.map(stats["te"])
    test_map = test_s.map(stats["te"]).fillna(global_mean)
    return train_map, test_map


def choose_feature_lists(
    df: pd.DataFrame, era5_cols: List[str], cat_cols: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Assemble three lists used by the ColumnTransformer:
      - `num_cols`: numeric predictors (core + env + stacked).
      - `cat_keep`: categorical columns to one-hot.
      - `era5_numeric`: raw numeric ERA5 columns (optionally PCA-reduced).
    """
    base = [c for c in [
        "wmo_wind","wmo_pres","lat","lon","abs_lat","lon_norm","doy_sin","doy_cos","month","hour",
        "wind10_mag","wind10_dir_sin","wind10_dir_cos","shear_200_850","shear_dir_sin","shear_dir_cos",
        "track_speed_6h","track_speed_12h","track_speed_18h","track_speed_24h","track_dir_sin","track_dir_cos",
        "track_curv_12h","sst_lat_x","moist_shear_interact",
        "wind_pres_ratio","neg_pres","wind_x_negpres",
        "wmo_wind_lag6h","wmo_wind_lag12h","wmo_wind_lag18h","wmo_wind_lag24h","wmo_wind_lag36h","wmo_wind_lag48h",
        "wmo_wind_d6h","wmo_wind_d12h","wmo_wind_d18h","wmo_wind_d24h","wmo_wind_d36h","wmo_wind_d48h",
        "wmo_wind_rollmean_12h","wmo_wind_rollstd_12h","wmo_wind_rollmean_24h","wmo_wind_rollstd_24h",
        "wmo_pres_lag6h","wmo_pres_lag12h","wmo_pres_lag18h","wmo_pres_lag24h","wmo_pres_lag36h","wmo_pres_lag48h",
        "wmo_pres_d6h","wmo_pres_d12h","wmo_pres_d18h","wmo_pres_d24h","wmo_pres_d36h","wmo_pres_d48h"
    ] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    # Environmental full + their lag/delta expansions
    env_plus = [c for c in df.columns
                if (c in era5_cols or any(c.startswith(p) for p in [e+"_lag" for e in era5_cols] + [e+"_d" for e in era5_cols]))
                and pd.api.types.is_numeric_dtype(df[c])]

    # Any stacked temporal windows we added
    stacked = [c for c in df.columns
               if (("_t-" in c and c.endswith("h")) or c.endswith("_stack"))
               and pd.api.types.is_numeric_dtype(df[c])]

    num_cols = list(dict.fromkeys(base + env_plus + stacked))
    cat_keep = [c for c in cat_cols if c in df.columns]
    era5_numeric = [c for c in era5_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_keep, era5_numeric


def temporal_split_mask(
    df: pd.DataFrame, train_end: int, val_start: int, val_end: int, test_start: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct boolean masks for train/val/test based on YEAR boundaries.

    This avoids temporal leakage and aligns with a realistic forecasting regime.
    """
    y = df["year"].astype(int)
    tr = (y <= train_end).values
    va = ((y >= val_start) & (y <= val_end)).values
    te = (y >= test_start).values
    return tr, va, te


class StackedEnsemble:
    """
    Lightweight stacked ensemble wrapper.

    - Base layer: several LightGBM and/or XGBoost models with optional
      column subsampling (works on *preprocessed* feature matrices from
      a ColumnTransformer).
    - Meta layer: logistic regression combining averaged LightGBM/XGBoost
      probabilities.
    - Calibration: auto-select isotonic vs. sigmoid based on validation Brier.

    Notes:
    - This class expects *already-split* data. Do not pass test data to fit().
    - feature_names_out_ mirrors ColumnTransformer.get_feature_names_out().
    """

    def __init__(self,
                 preproc: ColumnTransformer,
                 pos_weight: float,
                 base_seed: int,
                 lgbm_ens: int = 4,
                 xgb_ens: int = 3,
                 col_subsample: float = 0.9,
                 calibrate: str = "auto"):
        self.preproc = preproc
        self.pos_weight = pos_weight
        self.base_seed = base_seed
        self.lgbm_ens = int(max(0, lgbm_ens))
        self.xgb_ens = int(max(0, xgb_ens))
        self.col_subsample = float(np.clip(col_subsample, 0.6, 1.0))
        self.calibrate = calibrate

        # Will be populated during fit()
        self.feature_names_out_ = None
        self.lgbm_models: List = []
        self.xgb_models: List = []
        self.col_indices_lgbm: List[Optional[np.ndarray]] = []
        self.col_indices_xgb: List[Optional[np.ndarray]] = []
        self.meta_stack_model: Optional[LogisticRegression] = None
        self.calibrator: Optional[object] = None
        self._cal_method: Optional[str] = None

    def _new_lgbm(self, seed: int):
        """Create a LightGBM classifier with mild randomness for robustness."""
        if lgb is None:
            return None
        return lgb.LGBMClassifier(
            n_estimators=1800, learning_rate=np.random.uniform(0.026, 0.045),
            num_leaves=np.random.choice([63,95,127]), max_depth=-1,
            min_child_samples=25, subsample=0.90, colsample_bytree=0.85,
            reg_lambda=np.random.uniform(0.8, 1.8), reg_alpha=np.random.uniform(0.2, 0.7),
            objective="binary", scale_pos_weight=self.pos_weight,
            random_state=seed, n_jobs=-1, verbose=-1
        )

    def _new_xgb(self, seed: int):
        """Create an XGBoost classifier with mild randomness for robustness."""
        if xgb is None:
            return None
        return xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=np.random.uniform(0.026, 0.045),
            max_depth=np.random.choice([3,4]),
            min_child_weight=np.random.choice([6,8,10]),
            subsample=0.90, colsample_bytree=0.85,
            reg_lambda=np.random.uniform(1.5, 3.5), reg_alpha=np.random.uniform(0.3, 1.0),
            objective="binary:logistic", eval_metric="aucpr",
            scale_pos_weight=self.pos_weight, tree_method="hist",
            random_state=seed, n_jobs=-1, verbosity=0
        )

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None,
            early_stopping_rounds: int = 200):
        """
        Fit the stack:
        1) Fit ColumnTransformer on train; transform train/val.
        2) Train several LGB/XGB models with optional column subsampling.
        3) Train a logistic meta-model on val stacking features.
        4) Fit a probability calibrator (auto iso/sigmoid) on val.

        Early stopping is forwarded where the backend supports it.
        """
        # 1) Preprocess
        Xt = self.preproc.fit_transform(X_train)
        self.feature_names_out_ = list(self.preproc.get_feature_names_out())
        Xv = self.preproc.transform(X_val) if (X_val is not None and y_val is not None) else None
        nfeat = Xt.shape[1]
        keep_dim = int(max(6, round(self.col_subsample * nfeat)))

        # 2) Train base models (LGBM, XGB)
        self.lgbm_models, self.xgb_models = [], []
        self.col_indices_lgbm, self.col_indices_xgb = [], []

        # LightGBM ensemble
        for i in range(self.lgbm_ens):
            sd = self.base_seed + 101*i
            mdl = self._new_lgbm(sd)
            if mdl is None: break
            idx = None
            if self.col_subsample < 0.999:
                rng = np.random.default_rng(sd + 7)
                idx = np.sort(rng.choice(nfeat, size=keep_dim, replace=False))
            Xt_i = Xt if idx is None else Xt[:, idx]
            Xv_i = None if Xv is None else (Xv if idx is None else Xv[:, idx])
            if Xv_i is not None:
                try:
                    mdl.set_params(early_stopping_rounds=int(early_stopping_rounds))
                except Exception:
                    pass
                mdl.fit(Xt_i, y_train, eval_set=[(Xv_i, y_val)])
            else:
                mdl.fit(Xt_i, y_train)
            self.lgbm_models.append(mdl)
            self.col_indices_lgbm.append(idx)

        # XGBoost ensemble
        for i in range(self.xgb_ens):
            sd = self.base_seed + 203*i
            mdl = self._new_xgb(sd)
            if mdl is None: break
            idx = None
            if self.col_subsample < 0.999:
                rng = np.random.default_rng(sd + 9)
                idx = np.sort(rng.choice(nfeat, size=keep_dim, replace=False))
            Xt_i = Xt if idx is None else Xt[:, idx]
            Xv_i = None if Xv is None else (Xv if idx is None else Xv[:, idx])
            if Xv_i is not None:
                try:
                    mdl.set_params(early_stopping_rounds=int(early_stopping_rounds))
                except Exception:
                    pass
                mdl.fit(Xt_i, y_train, eval_set=[(Xv_i, y_val)])
            else:
                mdl.fit(Xt_i, y_train)
            self.xgb_models.append(mdl)
            self.col_indices_xgb.append(idx)

        # 3) Meta logistic regression on val stacking features
        if Xv is not None:
            p_l = self._avg_model_pred(self.lgbm_models, self.col_indices_lgbm, Xv)
            p_x = self._avg_model_pred(self.xgb_models, self.col_indices_xgb, Xv)
            self.meta_stack_model = LogisticRegression(max_iter=200, class_weight="balanced")
            Z = np.vstack([p_l, p_x]).T
            self.meta_stack_model.fit(Z, y_val)

            # 4) Probability calibration (choose best by Brier)
            raw = self.predict_raw_stack(Xv)
            if self.calibrate == "auto":
                iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(raw, y_val)
                cal_iso = np.clip(iso.predict(raw), 0, 1)
                bri_iso = brier_score_loss(y_val, cal_iso)

                lr = LogisticRegression(max_iter=300); lr.fit(raw.reshape(-1,1), y_val)
                cal_sig = lr.predict_proba(raw.reshape(-1,1))[:,1]
                bri_sig = brier_score_loss(y_val, cal_sig)

                if bri_iso <= bri_sig:
                    self.calibrator, self._cal_method = iso, "isotonic"
                else:
                    self.calibrator, self._cal_method = lr, "sigmoid"
            elif self.calibrate in {"sigmoid","isotonic"}:
                if self.calibrate == "isotonic":
                    self.calibrator = IsotonicRegression(out_of_bounds="clip")
                    self.calibrator.fit(raw, y_val)
                    self._cal_method = "isotonic"
                else:
                    self.calibrator = LogisticRegression(max_iter=300)
                    self.calibrator.fit(raw.reshape(-1,1), y_val)
                    self._cal_method = "sigmoid"
        return self

    def _avg_model_pred(self, models: List, idx_list: List[Optional[np.ndarray]], Z):
        """
        Average the predicted probabilities across a list of models, respecting
        column subsampling selections.
        """
        if not models:
            return np.zeros(Z.shape[0])
        ps = []
        for m, idx in zip(models, idx_list):
            Z_i = Z if idx is None else Z[:, idx]
            ps.append(m.predict_proba(Z_i)[:,1])
        return np.mean(np.vstack(ps), axis=0)

    def predict_raw_stack(self, Z):
        """
        Combine base-model probabilities and feed the meta-model (if any).
        """
        p_l = self._avg_model_pred(self.lgbm_models, self.col_indices_lgbm, Z)
        p_x = self._avg_model_pred(self.xgb_models, self.col_indices_xgb, Z)
        if self.meta_stack_model is None:
            return 0.5*(p_l + p_x)
        meta_in = np.vstack([p_l, p_x]).T
        return self.meta_stack_model.predict_proba(meta_in)[:,1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        End-to-end probability prediction, including preprocessing and (optional)
        probability calibration. Returns an (n,2) array: [P(class=0), P(class=1)].
        """
        Z = self.preproc.transform(X)
        p = self.predict_raw_stack(Z)
        if self.calibrator is not None:
            if self._cal_method == "isotonic":
                p = np.clip(self.calibrator.predict(p), 0.0, 1.0)
            else:
                p = self.calibrator.predict_proba(p.reshape(-1,1))[:,1]
        return np.vstack([1.0-p, p]).T


def make_preprocessor(num_cols: List[str],
                      cat_cols: List[str],
                      era5_numeric: List[str],
                      scale_numeric: bool,
                      pca_era5_k: int = 12) -> ColumnTransformer:
    """
    Construct a ColumnTransformer with:
      - Numeric pipeline: median impute (+ optional standardize).
      - Categorical pipeline: one-hot with unknown handling.
      - ERA5 numeric pipeline: median impute + scale + optional PCA (to compress
        correlated fields while preserving variance).
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    transformers = [("num", num_pipe, num_cols)]

    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

    if era5_numeric:
        era5_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        if pca_era5_k > 0:
            era5_steps.append(("pca", PCA(n_components=min(pca_era5_k, max(1, len(era5_numeric)-1)), random_state=RANDOM_STATE)))
        era5_pipe = Pipeline(steps=era5_steps)
        transformers.append(("era5_pca", era5_pipe, era5_numeric))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def pr_thresholds(y_true: np.ndarray, proba: np.ndarray, targets=(0.20,0.30,0.40)) -> Dict[str,float]:
    """
    Compute operating thresholds on validation:
      - For each precision target, choose the threshold with max recall.
      - Also compute the F1-optimal threshold (argmax F1 along PR curve).

    Returns a dict with thresholds and associated recalls.
    """
    p, r, thr = precision_recall_curve(y_true, proba)
    out = {}
    for pt in targets:
        idx = np.where(p[:-1] >= pt)[0]
        if idx.size:
            best = idx[np.argmax(r[idx])]
            out[f"thr_at_P>={pt:.2f}"] = float(thr[max(best-1,0)])
            out[f"R_at_P>={pt:.2f}"] = float(r[best])
        else:
            out[f"thr_at_P>={pt:.2f}"] = None
            out[f"R_at_P>={pt:.2f}"] = 0.0
    f1 = (2*p*r)/(p+r+1e-12)
    best_idx = int(np.nanargmax(f1))
    out["thr_F1_opt"] = float(thr[max(best_idx-1,0)])
    out["P_F1_opt"] = float(p[best_idx])
    out["R_F1_opt"] = float(r[best_idx])
    return out


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 20) -> float:
    """
    Expected Calibration Error (ECE) using quantile binning.
    """
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return float(np.nanmean(np.abs(prob_true - prob_pred)))


def eval_split(y_true: np.ndarray, proba: np.ndarray, meta: pd.DataFrame, split: str, outdir: Path) -> Dict:
    """
    Evaluate a split and persist diagnostic artifacts:
      - AUPRC / AUROC / Brier / ECE
      - PR/ROC curve CSVs
      - Reliability curve CSV
      - Predictions parquet (with y_true, y_proba, and F1-opt labels)
    """
    auprc = float(average_precision_score(y_true, proba))
    auroc = float(roc_auc_score(y_true, proba))
    p, r, thr = precision_recall_curve(y_true, proba)
    fpr, tpr, roc_thr = roc_curve(y_true, proba)
    brier = float(brier_score_loss(y_true, proba))
    ece = expected_calibration_error(y_true, proba, n_bins=20)

    # F1-optimal operating point for reporting convenience
    f1 = (2*p*r)/(p+r+1e-12)
    best = int(np.nanargmax(f1))
    thr_f1 = float(thr[max(best-1,0)]) if len(thr)>0 else 0.5
    yhat = (proba >= thr_f1).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()

    # Persist diagnostics
    pd.DataFrame({"precision":p, "recall":r}).to_csv(outdir/f"pr_curve_{split}.csv", index=False)
    pd.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":roc_thr}).to_csv(outdir/f"roc_curve_{split}.csv", index=False)
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=20, strategy="quantile")
    pd.DataFrame({"prob_true":prob_true, "prob_pred":prob_pred}).to_csv(outdir/f"reliability_{split}.csv", index=False)

    pred = meta.copy()
    pred["y_true"] = y_true
    pred["y_proba"] = proba
    pred["y_pred_f1opt"] = yhat
    pred.to_parquet(outdir/f"predictions_{split}.parquet", index=False)

    return {
        "split": split,
        "AUPRC": auprc, "AUROC": auroc, "Brier": brier, "ECE": ece,
        "best_F1": float(f1[best]) if len(f1)>0 else None, "best_threshold": thr_f1,
        "best_precision": float(p[best]) if len(p)>0 else None, "best_recall": float(r[best]) if len(r)>0 else None,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "positives": int(y_true.sum()), "negatives": int((y_true==0).sum()),
    }


def save_importances(ensemble: "StackedEnsemble", outpath: Path):
    """
    Aggregate feature importances over all tree models (LGB/XGB) into a single
    CSV. If column subsampling is used, we expand back to full feature length.
    """
    try:
        feats = ensemble.feature_names_out_ or []
        imp_rows = []
        for m, idx in [(mm, ii) for mm, ii in zip(ensemble.lgbm_models, ensemble.col_indices_lgbm)] + \
                      [(mm, ii) for mm, ii in zip(ensemble.xgb_models, ensemble.col_indices_xgb)]:
            if hasattr(m, "feature_importances_"):
                imp = getattr(m, "feature_importances_")
                if idx is not None:
                    full = np.zeros(len(feats))
                    full[idx] = imp[:len(idx)]
                    imp_rows.append(full)
                else:
                    imp_rows.append(imp[:len(feats)])
        if imp_rows:
            arr = np.vstack(imp_rows).mean(axis=0)
            pd.DataFrame({"feature":feats[:len(arr)], "importance":arr}).sort_values("importance", ascending=False).to_csv(outpath, index=False)
    except Exception as e:
        with open(outpath.with_suffix(".error.txt"), "w") as f:
            f.write(str(e))


def per_basin_metrics(df: pd.DataFrame, mask: np.ndarray, proba: np.ndarray, outdir: Path, split: str) -> Dict[str, Dict]:
    """
    Slice AUPRC/AUROC by ocean basin, if available. Useful for regional QA.
    """
    if "basin" not in df.columns:
        return {}
    out = {}
    y = df.loc[mask, "RI"].astype(int).values
    bas = df.loc[mask, "basin"].astype(str).values
    for b in sorted(pd.unique(bas)):
        idx = (bas == b)
        if idx.sum() < 5:
            continue
        yb, pb = y[idx], proba[idx]
        out[str(b)] = {
            "AUPRC": float(average_precision_score(yb, pb)),
            "AUROC": float(roc_auc_score(yb, pb)) if len(np.unique(yb))>1 else None,
            "count": int(idx.sum())
        }
    with open(outdir/f"metrics_{split}_by_basin.json","w") as f:
        json.dump(out, f, indent=2)
    return out


def regime_metrics(df: pd.DataFrame, mask: np.ndarray, proba: np.ndarray, outdir: Path, split: str) -> Dict[str, Dict]:
    """
    Simple *environmental regime* slicing for interpretability:
      - Bins of SST and vertical shear to see where the model performs better.
    """
    out = {}
    y = df.loc[mask, "RI"].astype(int).values

    if "era5_sst" in df.columns:
        sst = df.loc[mask, "era5_sst"].values
        bins = [ -np.inf, 27.0, 29.0, np.inf ]
        names = ["sst_low_<27", "sst_mid_27_29", "sst_high_>29"]
        for (lo,hi), nm in zip(zip(bins[:-1], bins[1:]), names):
            idx = (sst > lo) & (sst <= hi)
            if np.sum(idx) >= 20:
                out[nm] = {
                    "AUPRC": float(average_precision_score(y[idx], proba[idx])),
                    "count": int(np.sum(idx))
                }
    if "shear_200_850" in df.columns:
        sh = df.loc[mask, "shear_200_850"].values
        bins = [ -np.inf, 10.0, 20.0, np.inf ]
        names = ["shear_low_<10kt", "shear_mid_10_20", "shear_high_>20"]
        for (lo,hi), nm in zip(zip(bins[:-1], bins[1:]), names):
            idx = (sh > lo) & (sh <= hi)
            if np.sum(idx) >= 20:
                out[nm] = {
                    "AUPRC": float(average_precision_score(y[idx], proba[idx])),
                    "count": int(np.sum(idx))
                }
    with open(outdir/f"metrics_{split}_by_regime.json","w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    """
    Orchestrates the end-to-end experiment:
      1) Load data and validate schema.
      2) Feature engineering (core, kinematics, lags, ERA5 memory, stacking).
      3) Temporal split into train/val/test.
      4) Build ColumnTransformer + StackedEnsemble (per --model).
      5) Fit, calibrate, and evaluate; persist artifacts to `outdir`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--model", type=str, default="stacked", choices=["stacked","xgb","lgbm","lr"])
    ap.add_argument("--calibrate", type=str, default="auto", choices=["auto","sigmoid","isotonic"])
    ap.add_argument("--train_end", type=int, default=2018)
    ap.add_argument("--val_start", type=int, default=2019)
    ap.add_argument("--val_end", type=int, default=2021)
    ap.add_argument("--test_start", type=int, default=2022)
    ap.add_argument("--ri_threshold", type=int, default=30)
    ap.add_argument("--outdir", type=str, default=str(OUTDIR))
    ap.add_argument("--early_stop", type=int, default=200)
    ap.add_argument("--env_lag_vars", type=int, default=24)
    ap.add_argument("--stack_top_k", type=int, default=18)
    ap.add_argument("--stack_hours", type=str, default="6,12,18,24,30,36,42,48")
    ap.add_argument("--stack_include_deltas", action="store_true")
    ap.add_argument("--select_top_n", type=int, default=0)
    ap.add_argument("--ensemble_size", type=int, default=6)
    ap.add_argument("--col_subsample", type=float, default=0.9)
    ap.add_argument("--pca_era5_k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    # --------------------
    # 1) Load & basic QA
    # --------------------
    data_path = Path(args.data) if args.data else find_first_existing(DEFAULT_CANDIDATE_PATHS)
    if data_path is None:
        raise FileNotFoundError("Dataset not found. Provide --data path.")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading: {data_path}")
    df = pd.read_parquet(data_path)

    need = ["storm_id","lat","lon","wmo_wind","iso_time"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # ---------------------------------
    # 2) Feature engineering components
    # ---------------------------------
    df = ensure_label(df, ri_threshold_kt=args.ri_threshold)
    df = add_core_features(df)
    df = add_track_kinematics(df)
    era5_cols, ib_cols, cat_cols = detect_sources(df)
    df = add_lags_and_rolling(df)
    df = add_env_lags(df, era5_cols, max_vars=args.env_lag_vars)

    # ----------------------------
    # 3) Temporal split + stacking
    # ----------------------------
    tr, va, te = temporal_split_mask(df, args.train_end, args.val_start, args.val_end, args.test_start)
    stack_hours = [int(x) for x in args.stack_hours.split(",") if x.strip()]
    top_stack_cols = pick_top_base_predictors_for_windowing(df, era5_cols, tr, top_k=args.stack_top_k)
    df = add_window_stack(df, top_stack_cols, stack_hours, include_deltas=args.stack_include_deltas)

    # Optional, leakage-safe target encoding for limited categoricals
    if "basin" in df.columns:
        te_tr, te_va = target_encode_mean_smooth(
            df.loc[tr, "basin"], df.loc[tr, "RI"].astype(int).values, df.loc[va, "basin"], m=50
        )
        te_te = df.loc[te, "basin"].map(pd.Series(te_tr).groupby(df.loc[tr, "basin"]).mean()).fillna(
            float(df.loc[tr, "RI"].mean())
        )
        df.loc[tr, "basin_te"] = te_tr.values
        df.loc[va, "basin_te"] = te_va.values
        df.loc[te, "basin_te"] = te_te.values
    if "storm_status" in df.columns:
        t2_tr, t2_va = target_encode_mean_smooth(
            df.loc[tr, "storm_status"], df.loc[tr, "RI"].astype(int).values, df.loc[va, "storm_status"], m=30
        )
        t2_te = df.loc[te, "storm_status"].map(pd.Series(t2_tr).groupby(df.loc[tr, "storm_status"]).mean()).fillna(
            float(df.loc[tr, "RI"].mean())
        )
        df.loc[tr, "status_te"] = t2_tr.values
        df.loc[va, "status_te"] = t2_va.values
        df.loc[te, "status_te"] = t2_te.values

    # Feature lists for the ColumnTransformer
    num_cols, cat_keep, era5_numeric = choose_feature_lists(df, era5_cols, cat_cols)
    for tec in ["basin_te","status_te"]:
        if tec in df.columns:
            num_cols.append(tec)

    # --------------------------
    # 4) Build model & preproc
    # --------------------------
    df = df.dropna(subset=["RI"]).copy()
    y = df["RI"].astype(int).values
    meta_cols = [c for c in ["storm_id","iso_time","year","lat","lon","basin"] if c in df.columns]
    meta_val = df.loc[va, meta_cols].copy()
    meta_tst = df.loc[te, meta_cols].copy()

    if args.select_top_n and args.select_top_n > 0:
        Xtr = df.loc[tr, num_cols].replace([np.inf, -np.inf], np.nan)
        Xtr = Xtr.fillna(Xtr.median(numeric_only=True))
        ytr = df.loc[tr, "RI"].astype(int).values
        mi = mutual_info_classif(Xtr.values, ytr, random_state=args.seed)
        keep = np.argsort(mi)[::-1][: min(args.select_top_n, len(num_cols))]
        num_cols = [num_cols[i] for i in keep]
        print(f"[INFO] MI selection kept {len(num_cols)} numeric features.")

    pos_weight = safe_ratio(int((y[tr]==0).sum()), int((y[tr]==1).sum()), 1.0)
    scale_numeric = (args.model == "lr")  # only LR needs scaling; trees do not
    preproc = make_preprocessor(num_cols, cat_keep, era5_numeric, scale_numeric, pca_era5_k=args.pca_era5_k)

    print(f"[INFO] Train={tr.sum()}  Val={va.sum()}  Test={te.sum()}  PosWeight≈{pos_weight:.2f}")

    # Choose ensemble composition
    if args.model == "stacked":
        lgbm_ens = max(2, args.ensemble_size // 2)
        xgb_ens = max(1, args.ensemble_size - lgbm_ens)
        clf = StackedEnsemble(
            preproc=preproc,
            pos_weight=pos_weight,
            base_seed=args.seed,
            lgbm_ens=lgbm_ens,
            xgb_ens=xgb_ens,
            col_subsample=args.col_subsample,
            calibrate=args.calibrate
        )
    elif args.model == "xgb":
        clf = StackedEnsemble(preproc=preproc, pos_weight=pos_weight, base_seed=args.seed,
                              lgbm_ens=0, xgb_ens=max(1, args.ensemble_size),
                              col_subsample=args.col_subsample, calibrate=args.calibrate)
    elif args.model == "lgbm":
        clf = StackedEnsemble(preproc=preproc, pos_weight=pos_weight, base_seed=args.seed,
                              lgbm_ens=max(1, args.ensemble_size), xgb_ens=0,
                              col_subsample=args.col_subsample, calibrate=args.calibrate)
    else:
        # Fallback: LR on preprocessed features with calibration
        clf = StackedEnsemble(preproc=preproc, pos_weight=1.0, base_seed=args.seed,
                              lgbm_ens=0, xgb_ens=0, col_subsample=1.0, calibrate=args.calibrate)
        clf.meta_stack_model = LogisticRegression(max_iter=2000, class_weight="balanced")

    # ----------------
    # 5) Fit & evaluate
    # ----------------
    clf.fit(
        X_train=df.loc[tr, num_cols + cat_keep],
        y_train=y[tr],
        X_val=df.loc[va, num_cols + cat_keep],
        y_val=y[va],
        early_stopping_rounds=args.early_stop
    )

    proba_val = clf.predict_proba(df.loc[va, num_cols + cat_keep])[:,1]
    proba_tst = clf.predict_proba(df.loc[te, num_cols + cat_keep])[:,1]

    thr_info = pr_thresholds(y_true=y[va], proba=proba_val, targets=(0.20,0.30,0.40))
    with open(outdir/"thresholds_from_val.json","w") as f:
        json.dump(thr_info, f, indent=2)

    m_val = eval_split(y[va], proba_val, meta_val, "val", outdir)
    m_tst = eval_split(y[te], proba_tst, meta_tst, "test", outdir)

    # Sliced diagnostics
    per_basin_metrics(df, va, proba_val, outdir, "val")
    per_basin_metrics(df, te, proba_tst, outdir, "test")
    regime_metrics(df, va, proba_val, outdir, "val")
    regime_metrics(df, te, proba_tst, outdir, "test")

    # Operational-style summaries at validation-derived thresholds
    ops = {}
    preds_test = meta_tst.copy()
    preds_test["y_true"] = y[te]
    preds_test["y_proba"] = proba_tst

    thr_f1 = thr_info["thr_F1_opt"]
    if thr_f1 is not None:
        yhat = (proba_tst >= thr_f1).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], yhat).ravel()
        ops["TEST_at_VAL_F1opt"] = {"threshold": float(thr_f1), "TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn)}
        preds_test["y_pred_valF1"] = yhat

    for ptag in ["0.20","0.30","0.40"]:
        thr = thr_info.get(f"thr_at_P>={ptag}")
        if thr is None:
            continue
        yhat = (proba_tst >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], yhat).ravel()
        ops[f"TEST_at_VAL_P>={ptag}"] = {"threshold": float(thr), "TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn)}

    with open(outdir/"ops_on_test.json","w") as f:
        json.dump(ops, f, indent=2)
    preds_test.to_parquet(outdir/"predictions_test_with_ops.parquet", index=False)

    save_importances(clf, outdir/"feature_importances.csv")

    # Full experiment snapshot
    cfg = {
        "data_path": str(data_path),
        "model": args.model, "calibrate": args.calibrate,
        "train_end": args.train_end, "val_start": args.val_start, "val_end": args.val_end, "test_start": args.test_start,
        "ri_threshold": args.ri_threshold,
        "numeric_features": num_cols, "categorical_features": cat_keep,
        "early_stop": args.early_stop, "env_lag_vars": args.env_lag_vars,
        "stack_top_k": args.stack_top_k,
        "stack_hours": [int(h) for h in args.stack_hours.split(",") if h.strip()],
        "stack_include_deltas": bool(args.stack_include_deltas),
        "select_top_n": args.select_top_n,
        "random_state": args.seed,
        "ensemble_size": args.ensemble_size,
        "col_subsample": args.col_subsample,
        "pca_era5_k": args.pca_era5_k
    }
    with open(outdir/"config.json","w") as f:
        json.dump(cfg, f, indent=2)

    metrics = {"val": m_val, "test": m_tst}
    with open(outdir/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    # Persist the fitted model (best-effort)
    try:
        import joblib
        joblib.dump(clf, outdir/f"model_{args.model}_stacked.joblib")
    except Exception as e:
        with open(outdir/"model_save_error.txt","w") as f:
            f.write(str(e))

    print("[DONE] Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()