# scripts/track_forecast_xgb_6h.py - Advanced training with tuned hyperparameters
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb

from utils.geo import local_xy_km_to_latlon
from metrics.track import great_circle_error_km

# Updated feature list matching build_track_features_6h.py
DEFAULT_FEATURES = [
    # Position & intensity
    "lat", "lon", "wmo_wind", "wmo_pres",
    # Environment
    "sst_c", "t2m_c", "msl_hpa", "u10", "v10", "wind10", "tp",
    # Motion 6h
    "dx_6h", "dy_6h", "speed_6h", "heading_sin_6h", "heading_cos_6h",
    # Motion 12h
    "dx_12h", "dy_12h", "speed_12h", "heading_sin_12h", "heading_cos_12h",
    # Time
    "season", "month_sin", "month_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    # Basin
    "basin_AL", "basin_EP", "basin_WP", "basin_SH",
    # Advanced
    "storm_age_h", "wind_chg_6h", "wind_chg_12h", "sst_gradient", "dist_to_land_km",
    "intensity_x_speed", "sst_x_wind", "mslp_deficit",
    "shear_u", "shear_v", "shear_mag",
]

def train(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features",
                   default="outputs/track_forecast/features/track_features_2000_2024.parquet")
    p.add_argument("--out_dir",
                   default="outputs/track_forecast")
    cfg = p.parse_args(args)
    
    feat_path = Path(cfg.features)
    out_root = Path(cfg.out_dir)
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading: {feat_path}")
    df = pd.read_parquet(feat_path)
    
    # Check all features exist
    must = DEFAULT_FEATURES + ["dx_km", "dy_km", "split", "lat", "lon", 
                               "lat_next6", "lon_next6", "sample_weight"]
    missing = [c for c in must if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    te = df[df["split"] == "test"].copy()
    
    print(f"[INFO] Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")
    
    X_tr, ydx_tr, ydy_tr = tr[DEFAULT_FEATURES], tr["dx_km"].values, tr["dy_km"].values
    X_va, ydx_va, ydy_va = va[DEFAULT_FEATURES], va["dx_km"].values, va["dy_km"].values
    
    # Tuned hyperparams for 55 features
    common = dict(
        n_estimators=4000,
        learning_rate=0.015,        # slower for more features
        max_depth=10,                # deeper (was 8)
        num_leaves=63,               # more (was 31)
        subsample=0.8,
        colsample_bytree=0.75,       # lower to prevent overfitting with 55 features
        reg_alpha=0.05,              # less L1 (was 0.1)
        reg_lambda=0.3,              # more L2 (was 0.2)
        min_child_samples=25,        # slightly higher (was 20)
        min_child_weight=0.01,       # regularization on hessian
        random_state=137,
        n_jobs=-1,
        verbosity=-1,
    )
    
    m_dx = LGBMRegressor(**common)
    m_dy = LGBMRegressor(**common)
    
    print("[INFO] Training Δx with early stopping…")
    m_dx.fit(
        X_tr, ydx_tr,
        sample_weight=tr["sample_weight"].values,
        eval_set=[(X_va, ydx_va)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),  # more patience
            lgb.log_evaluation(period=100)
        ]
    )
    
    print("[INFO] Training Δy with early stopping…")
    m_dy.fit(
        X_tr, ydy_tr,
        sample_weight=tr["sample_weight"].values,
        eval_set=[(X_va, ydy_va)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    joblib.dump(m_dx, out_root / "models" / "lgbm_dx_6h.joblib")
    joblib.dump(m_dy, out_root / "models" / "lgbm_dy_6h.joblib")
    print(f"[DONE] Saved models (dx: {m_dx.best_iteration_} iters, dy: {m_dy.best_iteration_} iters)")
    
    # Feature importance
    feat_imp = pd.DataFrame({
        'feature': DEFAULT_FEATURES,
        'importance_dx': m_dx.feature_importances_,
        'importance_dy': m_dy.feature_importances_
    }).sort_values('importance_dx', ascending=False)
    feat_imp.to_csv(out_root / "models" / "feature_importance_6h.csv", index=False)
    print(f"[INFO] Top 10 features (dx): {feat_imp.head(10)['feature'].tolist()}")
    
    # Predictions
    def _predict_split(dfin: pd.DataFrame, split_name: str) -> pd.DataFrame:
        if dfin.empty:
            return dfin.assign(lat_pred_next6=np.nan, lon_pred_next6=np.nan, error_km=np.nan)
        X = dfin[DEFAULT_FEATURES]
        dx_hat = m_dx.predict(X)
        dy_hat = m_dy.predict(X)
        latp, lonp = [], []
        for lt, lo, dx, dy in zip(dfin["lat"].values, dfin["lon"].values, dx_hat, dy_hat):
            la, lo2 = local_xy_km_to_latlon(float(lt), float(lo), float(dx), float(dy))
            latp.append(la)
            lonp.append(lo2)
        err = great_circle_error_km(dfin["lat_next6"], dfin["lon_next6"], latp, lonp)
        out = dfin[["storm_id", "iso_time", "lat", "lon", "lat_next6", "lon_next6", "split"]].copy()
        out["lat_pred_next6"] = np.asarray(latp)
        out["lon_pred_next6"] = np.asarray(lonp)
        out["error_km"] = err
        out["dx_pred_km"] = dx_hat
        out["dy_pred_km"] = dy_hat
        out["split"] = split_name
        return out
    
    preds = pd.concat([
        _predict_split(tr, "train"),
        _predict_split(va, "val"),
        _predict_split(te, "test"),
    ], ignore_index=True)
    
    pred_path = out_root / "predictions" / "predictions_6h.parquet"
    preds.to_parquet(pred_path, index=False)
    print(f"[DONE] Predictions → {pred_path}")
    
    # Quick metrics
    for sp in ["train", "val", "test"]:
        sub = preds[preds["split"] == sp]["error_km"].dropna()
        if len(sub) > 0:
            print(f"[{sp.upper()}] mean: {sub.mean():.2f} km, median: {sub.median():.2f} km, p95: {sub.quantile(0.95):.2f} km")

if __name__ == "__main__":
    train()
