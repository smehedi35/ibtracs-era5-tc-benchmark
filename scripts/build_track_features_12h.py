# scripts/build_track_features_12h.py - Advanced 12h track forecasting features
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from typing import Tuple

from utils.geo import latlon_to_local_xy_km

REQUIRED = [
    "storm_id", "season", "iso_time",
    "lat", "lon",
    "wmo_wind", "wmo_pres",
    "sst_c", "t2m_c", "msl_hpa",
    "u10", "v10", "wind10", "tp",
    "lat_next12", "lon_next12",
    "has_next12", "split", "sample_weight",
]

def compute_motion_features(df: pd.DataFrame, lookback: int = 2) -> Tuple[np.ndarray, ...]:
    """
    Motion from t-6h*lookback to t.
    For 12h: lookback=2 means t-12h to t (2 steps of 6h).
    """
    lat_prev = df.groupby("storm_id")["lat"].shift(lookback)
    lon_prev = df.groupby("storm_id")["lon"].shift(lookback)
    
    dx_prev, dy_prev = [], []
    for lp, lop, lt, lo in zip(lat_prev, lon_prev, df["lat"], df["lon"]):
        if pd.isna(lp) or pd.isna(lop):
            dx_prev.append(np.nan)
            dy_prev.append(np.nan)
        else:
            x, y = latlon_to_local_xy_km(lp, lop, lt, lo)
            dx_prev.append(x)
            dy_prev.append(y)
    
    dx_prev = np.asarray(dx_prev)
    dy_prev = np.asarray(dy_prev)
    dist_km = np.hypot(dx_prev, dy_prev)
    speed_kmh = dist_km / (6.0 * lookback)  # km traveled / hours
    
    heading_rad = np.arctan2(dx_prev, dy_prev)
    return dx_prev, dy_prev, speed_kmh, np.sin(heading_rad), np.cos(heading_rad)

def compute_storm_age(df: pd.DataFrame) -> np.ndarray:
    """Hours since first observation per storm."""
    t = pd.to_datetime(df["iso_time"], utc=True)
    age = t.groupby(df["storm_id"]).transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)
    return age.values

def compute_intensity_change(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """6h, 12h, 24h intensity change (kt)."""
    wind_6h = df.groupby("storm_id")["wmo_wind"].diff(1)
    wind_12h = df.groupby("storm_id")["wmo_wind"].diff(2)
    wind_24h = df.groupby("storm_id")["wmo_wind"].diff(4)
    return wind_6h.fillna(0).values, wind_12h.fillna(0).values, wind_24h.fillna(0).values

def compute_motion_acceleration(df: pd.DataFrame) -> np.ndarray:
    """
    Change in translation speed (acceleration).
    12h forecast: important to capture steering changes.
    """
    speed_6h_ago = df.groupby("storm_id")["speed_12h"].shift(2)  # speed from t-12 to t-6
    accel = (df["speed_12h"] - speed_6h_ago) / 6.0  # (km/h per hour)
    return accel.fillna(0).values

def compute_sst_gradient(df: pd.DataFrame) -> np.ndarray:
    """Spatial SST change (°C/100km)."""
    sst_grad = df.groupby("storm_id")["sst_c"].diff()
    dx = df.groupby("storm_id")["dx_12h"].transform(lambda x: x.fillna(0))
    dy = df.groupby("storm_id")["dy_12h"].transform(lambda x: x.fillna(0))
    dist = np.hypot(dx, dy)
    return np.where(dist > 1, (sst_grad / dist) * 100, 0)

def compute_distance_to_land(df: pd.DataFrame) -> np.ndarray:
    """Rough distance to nearest coast (simplified)."""
    lat = df["lat"].values
    lon = df["lon"].values
    
    coast_lat = np.where(
        (lon >= -100) & (lon <= -60), 30,  # Atlantic
        np.where((lon >= 120) | (lon <= -120), 20, -30)  # Pacific/SH
    )
    dist_to_coast = np.abs(lat - coast_lat) * 111
    return np.clip(dist_to_coast, 0, 2000)

def compute_latitude_zone(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Latitude zone encoding (tropical vs subtropical vs extratropical).
    Critical for 12h: steering dynamics differ by latitude.
    """
    lat = df["lat"].values
    tropical = ((lat >= -30) & (lat <= 30)).astype(int)
    subtropical = (((lat > 30) & (lat <= 40)) | ((lat < -30) & (lat >= -40))).astype(int)
    extratropical = ((lat > 40) | (lat < -40)).astype(int)
    return tropical, subtropical, extratropical

def build(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="outputs/model_ready_2000_2024_v2.parquet")
    p.add_argument("--out", default="outputs/track_forecast/features/track_features_2000_2024_12h.parquet")
    cfg = p.parse_args(args)
    
    src = Path(cfg.source)
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Reading {src} …")
    df = pd.read_parquet(src)
    
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Missing: {missing}")
    
    df = df[df["has_next12"]].copy()
    df = df.sort_values(["storm_id", "iso_time"]).reset_index(drop=True)
    
    # === Motion features (12h, 24h lookback) ===
    dx_12h, dy_12h, speed_12h, hsin_12h, hcos_12h = compute_motion_features(df, lookback=2)
    dx_24h, dy_24h, speed_24h, hsin_24h, hcos_24h = compute_motion_features(df, lookback=4)
    
    # Also compute 6h motion for short-term info
    dx_6h, dy_6h, speed_6h, hsin_6h, hcos_6h = compute_motion_features(df, lookback=1)
    
    # Temporarily assign for downstream calcs
    df["dx_12h"] = dx_12h
    df["dy_12h"] = dy_12h
    df["speed_12h"] = speed_12h
    
    # === Targets (t -> t+12h) ===
    dx_next, dy_next = [], []
    for lt, lo, ltn, lon in zip(df["lat"], df["lon"], df["lat_next12"], df["lon_next12"]):
        x, y = latlon_to_local_xy_km(lt, lo, ltn, lon)
        dx_next.append(x)
        dy_next.append(y)
    
    # === Time features ===
    t = pd.to_datetime(df["iso_time"], utc=True)
    month = t.dt.month.to_numpy()
    hour = t.dt.hour.to_numpy()
    doy = t.dt.dayofyear.to_numpy()
    
    month_sin = np.sin(2*np.pi*(month/12.0))
    month_cos = np.cos(2*np.pi*(month/12.0))
    hour_sin = np.sin(2*np.pi*(hour/24.0))
    hour_cos = np.cos(2*np.pi*(hour/24.0))
    doy_sin = np.sin(2*np.pi*(doy/365.0))
    doy_cos = np.cos(2*np.pi*(doy/365.0))
    
    # === Basin encoding ===
    basin = df["storm_id"].str[:2].values
    basin_AL = (basin == 'AL').astype(int)
    basin_EP = (basin == 'EP').astype(int)
    basin_WP = (basin == 'WP').astype(int)
    basin_SH = ((basin == 'SH') | (basin == 'SI') | (basin == 'SP')).astype(int)
    basin_NI = (basin == 'IO').astype(int)  # North Indian
    
    # === Latitude zones ===
    lat_tropical, lat_subtropical, lat_extratropical = compute_latitude_zone(df)
    
    # === Advanced features ===
    storm_age_hours = compute_storm_age(df)
    wind_chg_6h, wind_chg_12h, wind_chg_24h = compute_intensity_change(df)
    motion_accel = compute_motion_acceleration(df)
    sst_gradient = compute_sst_gradient(df)
    dist_to_land = compute_distance_to_land(df)
    
    # === Interactions ===
    intensity_x_speed = df["wmo_wind"].values * speed_12h
    sst_x_wind = df["sst_c"].values * df["wmo_wind"].values
    mslp_deficit = 1013 - df["msl_hpa"].values
    
    # Coriolis proxy (latitude effect on steering)
    coriolis_param = 2 * 7.2921e-5 * np.sin(np.deg2rad(df["lat"].values))
    
    # Wind shear components
    shear_u = df["u10"].values
    shear_v = df["v10"].values
    shear_mag = np.hypot(shear_u, shear_v)
    
    # === Assemble ===
    base_cols = [
        "storm_id", "season", "iso_time", "split", "sample_weight",
        "lat", "lon", "wmo_wind", "wmo_pres",
        "sst_c", "t2m_c", "msl_hpa", "u10", "v10", "wind10", "tp",
    ]
    feat = df[base_cols].copy()
    
    # Motion 6h
    feat["dx_6h"] = dx_6h
    feat["dy_6h"] = dy_6h
    feat["speed_6h"] = speed_6h
    feat["heading_sin_6h"] = hsin_6h
    feat["heading_cos_6h"] = hcos_6h
    
    # Motion 12h
    feat["dx_12h"] = dx_12h
    feat["dy_12h"] = dy_12h
    feat["speed_12h"] = speed_12h
    feat["heading_sin_12h"] = hsin_12h
    feat["heading_cos_12h"] = hcos_12h
    
    # Motion 24h
    feat["dx_24h"] = dx_24h
    feat["dy_24h"] = dy_24h
    feat["speed_24h"] = speed_24h
    feat["heading_sin_24h"] = hsin_24h
    feat["heading_cos_24h"] = hcos_24h
    
    # Time
    feat["month_sin"] = month_sin
    feat["month_cos"] = month_cos
    feat["hour_sin"] = hour_sin
    feat["hour_cos"] = hour_cos
    feat["doy_sin"] = doy_sin
    feat["doy_cos"] = doy_cos
    
    # Basin
    feat["basin_AL"] = basin_AL
    feat["basin_EP"] = basin_EP
    feat["basin_WP"] = basin_WP
    feat["basin_SH"] = basin_SH
    feat["basin_NI"] = basin_NI
    
    # Latitude zones
    feat["lat_tropical"] = lat_tropical
    feat["lat_subtropical"] = lat_subtropical
    feat["lat_extratropical"] = lat_extratropical
    
    # Advanced
    feat["storm_age_h"] = storm_age_hours
    feat["wind_chg_6h"] = wind_chg_6h
    feat["wind_chg_12h"] = wind_chg_12h
    feat["wind_chg_24h"] = wind_chg_24h
    feat["motion_accel"] = motion_accel
    feat["sst_gradient"] = sst_gradient
    feat["dist_to_land_km"] = dist_to_land
    feat["intensity_x_speed"] = intensity_x_speed
    feat["sst_x_wind"] = sst_x_wind
    feat["mslp_deficit"] = mslp_deficit
    feat["coriolis_param"] = coriolis_param
    feat["shear_u"] = shear_u
    feat["shear_v"] = shear_v
    feat["shear_mag"] = shear_mag
    
    # Targets
    feat["dx_km"] = np.asarray(dx_next, dtype=float)
    feat["dy_km"] = np.asarray(dy_next, dtype=float)
    feat["lat_next12"] = df["lat_next12"].values
    feat["lon_next12"] = df["lon_next12"].values
    
    # Drop first 4 points per storm (need 24h lookback)
    feat = feat.dropna(subset=["dx_6h", "dy_6h", "dx_12h", "dy_12h", "dx_24h", "dy_24h"]).reset_index(drop=True)
    
    print(f"[INFO] Writing → {out}")
    feat.to_parquet(out, index=False)
    print(f"[DONE] {len(feat)} samples, {len(feat.columns)} features")

if __name__ == "__main__":
    build()
