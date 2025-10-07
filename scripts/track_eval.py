# scripts/track_eval.py - Enhanced with per-basin and intensity analysis
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def summarize_errors_km(errors: np.ndarray) -> dict:
    """Compute summary statistics for track errors."""
    return {
        "count": len(errors),
        "mean_km": np.mean(errors),
        "median_km": np.median(errors),
        "p90_km": np.percentile(errors, 90),
        "p95_km": np.percentile(errors, 95),
        "max_km": np.max(errors),
    }

def _detect_horizon_cols(df: pd.DataFrame):
    for H in (6, 12, 24):
        if f"lat_next{H}" in df.columns and f"lon_next{H}" in df.columns:
            return H, f"lat_next{H}", f"lon_next{H}", f"lat_pred_next{H}", f"lon_pred_next{H}"
    raise KeyError("Could not detect horizon: expected lat_next6/12/24 and lon_next6/12/24 in predictions.")

def evaluate(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preds",
                   default="outputs/track_forecast/predictions/predictions_6h.parquet",
                   help="Path to predictions parquet (6h/12h/24h).")
    p.add_argument("--out_dir",
                   default="outputs/track_forecast",
                   help="Root output dir (metrics/, figs/)")
    cfg = p.parse_args(args)
    
    pred_path = Path(cfg.preds)
    out_root = Path(cfg.out_dir)
    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    (out_root / "figs").mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Reading predictions: {pred_path}")
    df = pd.read_parquet(pred_path)
    
    H, lat_t, lon_t, lat_p, lon_p = _detect_horizon_cols(df)
    
    # Basic checks
    for c in ["split", "error_km", lat_t, lon_t, lat_p, lon_p]:
        if c not in df.columns:
            raise KeyError(f"Missing column in predictions: {c}")
    
    # ========== Overall metrics by split ==========
    rows = []
    for sp in ["train", "val", "test"]:
        d = df[df["split"] == sp]
        if d.empty:
            continue
        s = summarize_errors_km(d["error_km"].values)
        s["split"] = sp
        rows.append(s)
    summary = pd.DataFrame(rows, columns=["split", "count", "mean_km", "median_km", "p90_km", "p95_km", "max_km"])
    
    overall = summarize_errors_km(df["error_km"].values)
    overall["split"] = "all"
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    
    metrics_path = out_root / "metrics" / f"summary_by_split_{H}h.csv"
    summary.to_csv(metrics_path, index=False)
    print(f"[DONE] Metrics → {metrics_path}")
    
    # ========== TERMINAL OUTPUT ==========
    print(f"\n{'='*70}")
    print(f"+{H}H TRACK FORECAST RESULTS".center(70))
    print('='*70)
    print(f"{'Split':6s}  {'Count':>6s}  {'Mean':>8s}  {'Median':>8s}  {'P90':>8s}  {'P95':>8s}  {'Max':>8s}")
    print('-'*70)
    for _, row in summary.iterrows():
        print(f"{row['split']:6s}  {int(row['count']):6d}  "
              f"{row['mean_km']:8.2f}  {row['median_km']:8.2f}  "
              f"{row['p90_km']:8.2f}  {row['p95_km']:8.2f}  {row['max_km']:8.2f}")
    print('='*70)
    
    # Overfitting check
    train_mean = summary[summary["split"] == "train"]["mean_km"].values[0]
    val_mean = summary[summary["split"] == "val"]["mean_km"].values[0]
    test_mean = summary[summary["split"] == "test"]["mean_km"].values[0]
    val_gap = val_mean / train_mean
    test_gap = test_mean / train_mean
    
    print(f"\nOverfitting Analysis:")
    print(f"  Val/Train gap:  {val_gap:.2f}x {'✓ healthy' if val_gap < 2.5 else '⚠ overfitting'}")
    print(f"  Test/Train gap: {test_gap:.2f}x {'✓ healthy' if test_gap < 2.5 else '⚠ overfitting'}")
    
    # ========== Baseline comparison ==========
    base_path = out_root / "metrics" / f"baselines_{H}h.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        print(f"\nBaseline Comparison (test split):")
        print('-'*70)
        baseline_errors = []
        for method in base["method"].unique():
            sub = base[(base["method"] == method) & (base["split"] == "test")]
            if not sub.empty:
                base_mean = sub["mean_km"].values[0]
                improvement = ((base_mean - test_mean) / base_mean) * 100
                status = "✓" if improvement > 0 else "✗"
                print(f"  {method:25s}  {base_mean:8.2f} km  ({improvement:+.1f}% {status})")
                baseline_errors.append(base_mean)
        print(f"  {'XGBoost (this model)':25s}  {test_mean:8.2f} km")
        
        # Skill score vs best baseline
        if baseline_errors:
            best_baseline = min(baseline_errors)
            skill = ((best_baseline - test_mean) / best_baseline) * 100
            print(f"\n  Skill vs best baseline: {skill:+.2f}%")
    print('='*70 + '\n')
    
    # ========== Per-basin analysis (test only) ==========
    if "storm_id" in df.columns:
        test_df = df[df["split"] == "test"].copy()
        test_df["basin"] = test_df["storm_id"].str[:2]
        
        print("Per-Basin Error Analysis (test split):")
        print('-'*70)
        basin_rows = []
        for basin in sorted(test_df["basin"].unique()):
            sub = test_df[test_df["basin"] == basin]["error_km"].dropna().values
            if len(sub) < 10:  # skip basins with <10 samples
                continue
            s = summarize_errors_km(sub)
            s["basin"] = basin
            basin_rows.append(s)
            print(f"  {basin:4s}  n={len(sub):4d}  mean={s['mean_km']:6.2f} km  "
                  f"median={s['median_km']:5.2f} km  p95={s['p95_km']:5.2f} km")
        
        if basin_rows:
            basin_metrics = pd.DataFrame(basin_rows, columns=["basin", "count", "mean_km", "median_km", "p90_km", "p95_km", "max_km"])
            basin_path = out_root / "metrics" / f"by_basin_{H}h.csv"
            basin_metrics.to_csv(basin_path, index=False)
            print(f"\n[DONE] Basin metrics → {basin_path}")
        print('='*70 + '\n')
    
    # ========== Intensity stratification (test only) ==========
    if "wmo_wind" in df.columns:
        test_df = df[df["split"] == "test"].copy()
        
        # Saffir-Simpson categories
        test_df["category"] = pd.cut(
            test_df["wmo_wind"],
            bins=[0, 34, 64, 83, 96, 113, 1000],
            labels=["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4-5"]
        )
        
        print("Error by Intensity (test split):")
        print('-'*70)
        for cat in ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4-5"]:
            sub = test_df[test_df["category"] == cat]["error_km"].dropna().values
            if len(sub) < 5:
                continue
            s = summarize_errors_km(sub)
            print(f"  {cat:6s}  n={len(sub):4d}  mean={s['mean_km']:6.2f} km  "
                  f"p95={s['p95_km']:5.2f} km")
        print('='*70 + '\n')
    
    # ========== CDF plot ==========
    plt.figure(figsize=(8, 5))
    for sp in ["train", "val", "test"]:
        d = df[df["split"] == sp]["error_km"].dropna().values
        if d.size == 0:
            continue
        d_sorted = np.sort(d)
        y = np.linspace(0, 1, d_sorted.size, endpoint=True)
        plt.plot(d_sorted, y, label=f"{sp} (n={len(d):,})", linewidth=2)
    plt.xlabel(f"{H}-hour great-circle error (km)", fontsize=11)
    plt.ylabel("Cumulative Probability", fontsize=11)
    plt.title(f"Track Error CDF by Split (+{H}h)", fontsize=13, weight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim(0, np.percentile(df["error_km"].dropna(), 98))  # crop extreme outliers
    fig_path = out_root / "figs" / f"error_cdf_by_split_{H}h.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"[DONE] Figure → {fig_path}\n")

if __name__ == "__main__":
    evaluate()
