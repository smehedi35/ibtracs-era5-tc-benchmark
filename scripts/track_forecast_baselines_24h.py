# scripts/track_forecast_baselines_24h.py - Enhanced 24h baselines
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd

from utils.geo import local_xy_km_to_latlon
from metrics.track import great_circle_error_km, summarize_errors_km

REQUIRED = [
    "storm_id", "iso_time", "split",
    "lat", "lon", "lat_next24", "lon_next24",
    "dx_24h", "dy_24h",  # 24h lookback motion
    "u10", "v10"
]

KM_PER_MS_OVER_24H = 86.4  # 1 m/s over 24h = 86400m = 86.4 km

def _predict_latlon_from_dxdy(df: pd.DataFrame, dx: np.ndarray, dy: np.ndarray):
    latp, lonp = [], []
    for lt, lo, ddx, ddy in zip(df["lat"].values, df["lon"].values, dx, dy):
        la, lo2 = local_xy_km_to_latlon(float(lt), float(lo), float(ddx), float(ddy))
        latp.append(la)
        lonp.append(lo2)
    return np.asarray(latp), np.asarray(lonp)

def run(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="outputs/track_forecast/features/track_features_2000_2024_24h.parquet")
    p.add_argument("--out_dir", default="outputs/track_forecast")
    cfg = p.parse_args(args)
    
    feat_path = Path(cfg.features)
    out_root = Path(cfg.out_dir)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading: {feat_path}")
    df = pd.read_parquet(feat_path)
    
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Missing: {missing}")
    
    df = df.dropna(subset=["lat_next24", "lon_next24"]).copy()
    
    out_frames = []
    
    # ========== Baseline 1: Persistence (repeat 24h motion) ==========
    dx_pers = df["dx_24h"].to_numpy(dtype=float)
    dy_pers = df["dy_24h"].to_numpy(dtype=float)
    latp, lonp = _predict_latlon_from_dxdy(df, dx_pers, dy_pers)
    err = great_circle_error_km(df["lat_next24"], df["lon_next24"], latp, lonp)
    
    base1 = df[["storm_id", "iso_time", "split", "lat", "lon", "lat_next24", "lon_next24"]].copy()
    base1["lat_pred_next24"] = latp
    base1["lon_pred_next24"] = lonp
    base1["error_km"] = err
    base1["method"] = "persistence"
    out_frames.append(base1)
    
    # ========== Baseline 2: Wind10 advection ==========
    dx_w10 = df["u10"].to_numpy(dtype=float) * KM_PER_MS_OVER_24H
    dy_w10 = df["v10"].to_numpy(dtype=float) * KM_PER_MS_OVER_24H
    latp2, lonp2 = _predict_latlon_from_dxdy(df, dx_w10, dy_w10)
    err2 = great_circle_error_km(df["lat_next24"], df["lon_next24"], latp2, lonp2)
    
    base2 = df[["storm_id", "iso_time", "split", "lat", "lon", "lat_next24", "lon_next24"]].copy()
    base2["lat_pred_next24"] = latp2
    base2["lon_pred_next24"] = lonp2
    base2["error_km"] = err2
    base2["method"] = "wind10_advection"
    out_frames.append(base2)
    
    # ========== Baseline 3: Climatology (basin-specific) ==========
    train = df[df["split"] == "train"].copy()
    basin = df["storm_id"].str[:2].values
    train["basin"] = train["storm_id"].str[:2]
    
    clim_motion = train.groupby("basin")[["dx_24h", "dy_24h"]].mean().to_dict()
    dx_clim = np.array([clim_motion["dx_24h"].get(b, 0) for b in basin])
    dy_clim = np.array([clim_motion["dy_24h"].get(b, 0) for b in basin])
    
    latp3, lonp3 = _predict_latlon_from_dxdy(df, dx_clim, dy_clim)
    err3 = great_circle_error_km(df["lat_next24"], df["lon_next24"], latp3, lonp3)
    
    base3 = df[["storm_id", "iso_time", "split", "lat", "lon", "lat_next24", "lon_next24"]].copy()
    base3["lat_pred_next24"] = latp3
    base3["lon_pred_next24"] = lonp3
    base3["error_km"] = err3
    base3["method"] = "climatology"
    out_frames.append(base3)
    
    # ========== Baseline 4: Ensemble (pers + wind) ==========
    dx_ens = 0.7 * dx_pers + 0.3 * dx_w10
    dy_ens = 0.7 * dy_pers + 0.3 * dy_w10
    latp4, lonp4 = _predict_latlon_from_dxdy(df, dx_ens, dy_ens)
    err4 = great_circle_error_km(df["lat_next24"], df["lon_next24"], latp4, lonp4)
    
    base4 = df[["storm_id", "iso_time", "split", "lat", "lon", "lat_next24", "lon_next24"]].copy()
    base4["lat_pred_next24"] = latp4
    base4["lon_pred_next24"] = lonp4
    base4["error_km"] = err4
    base4["method"] = "ensemble_pers_wind"
    out_frames.append(base4)
    
    # ========== Baseline 5: Damped persistence (decay factor 0.9 for 24h) ==========
    # At 24h, storms often slow down/recurve → persistence overshoots
    dx_damp = dx_pers * 0.9
    dy_damp = dy_pers * 0.9
    latp5, lonp5 = _predict_latlon_from_dxdy(df, dx_damp, dy_damp)
    err5 = great_circle_error_km(df["lat_next24"], df["lon_next24"], latp5, lonp5)
    
    base5 = df[["storm_id", "iso_time", "split", "lat", "lon", "lat_next24", "lon_next24"]].copy()
    base5["lat_pred_next24"] = latp5
    base5["lon_pred_next24"] = lonp5
    base5["error_km"] = err5
    base5["method"] = "persistence_damped"
    out_frames.append(base5)
    
    # ========== Baseline 6: Latitude-adjusted persistence ==========
    lat = df["lat"].values
    recurvature_factor = np.where(np.abs(lat) > 30, 1.2, 1.0)  # 20% longer at high lat
    dx_lat_adj = dx_pers * recurvature_factor
    dy_lat_adj = dy_pers * recurvature_factor
    
    latp6, lonp6 = _predict_latlon_from_dxdy(df, dx_lat_adj, dy_lat_adj)
    err6 = great_circle_error_km(df["lat_next24"], df["lon_next24"], latp6, lonp6)
    
    base6 = df[["storm_id", "iso_time", "split", "lat", "lon", "lat_next24", "lon_next24"]].copy()
    base6["lat_pred_next24"] = latp6
    base6["lon_pred_next24"] = lonp6
    base6["error_km"] = err6
    base6["method"] = "persistence_lat_adjusted"
    out_frames.append(base6)
    
    # ========== Save ==========
    preds = pd.concat(out_frames, ignore_index=True)
    pred_path = out_root / "predictions" / "baselines_24h.parquet"
    preds.to_parquet(pred_path, index=False)
    print(f"[DONE] Baselines → {pred_path}")
    
    # Metrics
    rows = []
    for method in preds["method"].unique():
        sub_m = preds[preds["method"] == method]
        for sp in ["train", "val", "test"]:
            d = sub_m[sub_m["split"] == sp]["error_km"].dropna().values
            if d.size == 0:
                continue
            s = summarize_errors_km(d)
            s["method"] = method
            s["split"] = sp
            rows.append(s)
        s_all = summarize_errors_km(sub_m["error_km"].dropna().values)
        s_all["method"] = method
        s_all["split"] = "all"
        rows.append(s_all)
    
    metrics = pd.DataFrame(rows, columns=["method", "split", "count", "mean_km", "median_km", "p90_km", "p95_km", "max_km"])
    met_path = out_root / "metrics" / "baselines_24h.csv"
    metrics.to_csv(met_path, index=False)
    print(f"[DONE] Metrics → {met_path}")
    
    print("\n" + "="*70)
    print("BASELINE PERFORMANCE (test split)".center(70))
    print("="*70)
    test_metrics = metrics[metrics["split"] == "test"].sort_values("mean_km")
    for _, row in test_metrics.iterrows():
        print(f"{row['method']:30s}  {row['mean_km']:6.2f} km  "
              f"(median: {row['median_km']:5.2f} km, p95: {row['p95_km']:6.2f} km)")
    print("="*70 + "\n")

if __name__ == "__main__":
    run()
