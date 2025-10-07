# Full updated YAML-based feature builder
from __future__ import annotations
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import yaml

from utils.geo import latlon_to_local_xy_km

# ---------- helpers ----------
def _require_columns(df: pd.DataFrame, need: List[str]):
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in source data: {missing}")

def _load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _load_split_json(path: Path) -> Dict[str, List[str]]:
    with open(path, "r") as f:
        return json.load(f)

def _assign_split_by_ids(df: pd.DataFrame, id_col: str, split_map: Dict[str, List[str]]) -> pd.Series:
    train_ids = set(split_map.get("train", []))
    val_ids = set(split_map.get("val", []))
    test_ids = set(split_map.get("test", []))
    split = np.full(len(df), "", dtype=object)
    id_vals = df[id_col].astype(str).values
    for i, sid in enumerate(id_vals):
        if sid in train_ids:
            split[i] = "train"
        elif sid in val_ids:
            split[i] = "val"
        elif sid in test_ids:
            split[i] = "test"
        else:
            split[i] = ""
    return pd.Series(split, index=df.index)

def _fallback_year_split(df: pd.DataFrame, time_col: str) -> pd.Series:
    years = pd.to_datetime(df[time_col]).dt.year
    cond_train = years.between(2000, 2017, inclusive="both")
    cond_val = years.between(2018, 2020, inclusive="both")
    cond_test = years.between(2021, 2024, inclusive="both")
    split = np.where(cond_train, "train",
                     np.where(cond_val, "val",
                              np.where(cond_test, "test", "")))
    return pd.Series(split, index=df.index)

def _compute_motion_and_targets(df: pd.DataFrame, id_col: str, time_col: str,
                                lat_col: str, lon_col: str, horizon_hours: int) -> pd.DataFrame:
    """
    Compute motion features and targets for 6h horizon.
    """
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    
    # Previous displacement (t-6h → t)
    lat_prev = df.groupby(id_col)[lat_col].shift(1)
    lon_prev = df.groupby(id_col)[lon_col].shift(1)
    prev_dx_dy = [latlon_to_local_xy_km(lp, lop, lt, lo) if pd.notnull(lp) else (np.nan, np.nan)
                  for lp, lop, lt, lo in zip(lat_prev, lon_prev, df[lat_col], df[lon_col])]
    dx_prev_km = [p[0] for p in prev_dx_dy]
    dy_prev_km = [p[1] for p in prev_dx_dy]
    
    # Speed & heading from previous displacement
    prev_dist = np.sqrt(np.square(dx_prev_km) + np.square(dy_prev_km))
    speed_prev_kmh = (np.array(prev_dist) / 6.0)
    heading_prev_rad = np.arctan2(dx_prev_km, dy_prev_km)
    heading_prev_sin = np.sin(heading_prev_rad)
    heading_prev_cos = np.cos(heading_prev_rad)
    
    # Next (target) displacement (t → t+6h)
    lat_next = df.groupby(id_col)[lat_col].shift(-1)
    lon_next = df.groupby(id_col)[lon_col].shift(-1)
    next_dx_dy = [latlon_to_local_xy_km(lt, lo, ln, lonx) if pd.notnull(ln) else (np.nan, np.nan)
                  for lt, lo, ln, lonx in zip(df[lat_col], df[lon_col], lat_next, lon_next)]
    dx_km = [p[0] for p in next_dx_dy]
    dy_km = [p[1] for p in next_dx_dy]
    
    df_out = df.copy()
    df_out["dx_prev_km"] = dx_prev_km
    df_out["dy_prev_km"] = dy_prev_km
    df_out["speed_prev_kmh"] = speed_prev_kmh
    df_out["heading_prev_sin"] = heading_prev_sin
    df_out["heading_prev_cos"] = heading_prev_cos
    df_out["dx_km"] = dx_km
    df_out["dy_km"] = dy_km
    
    # Drop last record of each storm where target is NaN
    df_out = df_out[~df_out["dx_km"].isna()].reset_index(drop=True)
    return df_out

def build_features(config_path: str):
    cfg = _load_yaml(Path(config_path))
    src_path = Path(cfg["source_path"])
    split_path = Path(cfg["split_path"])
    out_path = Path(cfg["features_out"])
    
    cols = cfg["columns"]
    id_col = cols["id"]
    time_col = cols["time"]
    lat_col = cols["lat"]
    lon_col = cols["lon"]
    
    # Strict required set
    required = list(cols.values())
    print(f"[INFO] Reading {src_path} …")
    df = pd.read_parquet(src_path)
    _require_columns(df, required)
    
    # Year window
    if "years" in cfg and cfg["years"]:
        y0, y1 = int(cfg["years"]["start"]), int(cfg["years"]["end"])
        df = df[(pd.to_datetime(df[time_col]).dt.year >= y0) &
                (pd.to_datetime(df[time_col]).dt.year <= y1)]
    
    # Compute engineered motion + targets
    horizon = int(cfg["target_horizon_hours"])
    if horizon != 6:
        raise ValueError("This pipeline is fixed to 6-hour cadence for now.")
    df_feat = _compute_motion_and_targets(df, id_col, time_col, lat_col, lon_col, horizon)
    
    # Split assignment
    split_json = _load_split_json(split_path)
    split_series = _assign_split_by_ids(df_feat, id_col, split_json)
    if (split_series == "").all():
        print("[WARN] Split JSON empty → falling back to year-based split.")
        split_series = _fallback_year_split(df_feat, time_col)
    df_feat["split"] = split_series
    
    # Keep only rows with valid split
    df_feat = df_feat[df_feat["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    
    # ========== NEW FEATURES ==========
    # Cyclic time features
    t = pd.to_datetime(df_feat[time_col], utc=True)
    month = t.dt.month.to_numpy()
    hour = t.dt.hour.to_numpy()
    doy = t.dt.dayofyear.to_numpy()
    
    df_feat["month_sin"] = np.sin(2*np.pi*(month/12.0))
    df_feat["month_cos"] = np.cos(2*np.pi*(month/12.0))
    df_feat["hour_sin"] = np.sin(2*np.pi*(hour/24.0))
    df_feat["hour_cos"] = np.cos(2*np.pi*(hour/24.0))
    df_feat["doy_sin"] = np.sin(2*np.pi*(doy/365.0))
    df_feat["doy_cos"] = np.cos(2*np.pi*(doy/365.0))
    
    # Basin encoding
    basin = df_feat[id_col].str[:2].values
    df_feat["basin_AL"] = (basin == 'AL').astype(int)
    df_feat["basin_EP"] = (basin == 'EP').astype(int)
    df_feat["basin_WP"] = (basin == 'WP').astype(int)
    df_feat["basin_SH"] = ((basin == 'SH') | (basin == 'SI') | (basin == 'SP')).astype(int)
    
    # Interaction: intensity × speed
    df_feat["intensity_x_speed"] = df_feat[cols["vmax"]].values * df_feat["speed_prev_kmh"].values
    
    # Wind shear magnitude (if not already in data)
    if "shear_mag" not in df_feat.columns:
        df_feat["shear_mag"] = np.sqrt(df_feat[cols["shear_200_850"]]**2)  # if shear is scalar
    
    # Relative humidity deficit (drier = less favorable)
    if cols["rh700"] in df_feat.columns:
        df_feat["rh_deficit"] = 100.0 - df_feat[cols["rh700"]]
    
    # Final column order
    ordered = [
        id_col, time_col, lat_col, lon_col,
        "dx_prev_km", "dy_prev_km", "speed_prev_kmh", "heading_prev_sin", "heading_prev_cos",
        cols["vmax"], cols["mslp"], cols["sst"], cols["slp"], cols["rh700"],
        cols["u200"], cols["v200"], cols["u500"], cols["v500"], cols["u850"], cols["v850"], cols["shear_200_850"],
        "month_sin", "month_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "basin_AL", "basin_EP", "basin_WP", "basin_SH",
        "intensity_x_speed", "rh_deficit",
        "dx_km", "dy_km", "split"
    ]
    
    # Ensure columns exist (skip shear_mag if not added)
    available = [c for c in ordered if c in df_feat.columns]
    df_out = df_feat[available].copy()
    
    # Ensure output directory
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing features → {out_path}")
    df_out.to_parquet(out_path, index=False)
    print(f"[DONE] Feature build complete: {len(df_out)} samples, {len(df_out.columns)} features")

if __name__ == "__main__":
    build_features("configs/track_forecast_xgb.yaml")
