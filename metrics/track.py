# metrics/track.py
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.geo import latlon_to_local_xy_km

def great_circle_error_km(lat_true, lon_true, lat_pred, lon_pred):
    """
    Vectorized great-circle error (km) using NumPy.
    Accepts pandas Series / numpy arrays.
    """
    lt = np.asarray(lat_true, dtype=float)
    ln = np.asarray(lon_true, dtype=float)
    lp = np.asarray(lat_pred, dtype=float)
    lq = np.asarray(lon_pred, dtype=float)

    p = np.pi / 180.0
    dlat = (lp - lt) * p
    dlon = (lq - ln) * p
    a = np.sin(dlat/2.0)**2 + np.cos(lt*p) * np.cos(lp*p) * np.sin(dlon/2.0)**2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(a.clip(0.0, 1.0)))

def summarize_errors_km(err_km: np.ndarray | pd.Series) -> dict:
    e = np.asarray(err_km)
    e = e[np.isfinite(e)]
    return {
        "count": int(e.size),
        "mean_km": float(np.mean(e)) if e.size else np.nan,
        "median_km": float(np.median(e)) if e.size else np.nan,
        "p90_km": float(np.percentile(e, 90)) if e.size else np.nan,
        "p95_km": float(np.percentile(e, 95)) if e.size else np.nan,
        "max_km": float(np.max(e)) if e.size else np.nan,
    }

def along_cross_track_errors(df: pd.DataFrame,
                             lat_col: str, lon_col: str,
                             lat_true_next: str, lon_true_next: str,
                             lat_pred_next: str, lon_pred_next: str) -> pd.DataFrame:
    """
    Decompose forecast error into along- and cross-track components
    using a local tangent-plane approximation around the current fix.
    """
    dx_true, dy_true = [], []
    dx_pred, dy_pred = [], []
    for lt, lo, ltn, lon, ltp, lop in zip(df[lat_col], df[lon_col],
                                          df[lat_true_next], df[lon_true_next],
                                          df[lat_pred_next], df[lon_pred_next]):
        tx, ty = latlon_to_local_xy_km(lt, lo, ltn, lon)
        px, py = latlon_to_local_xy_km(lt, lo, ltp, lop)
        dx_true.append(tx); dy_true.append(ty)
        dx_pred.append(px); dy_pred.append(py)

    dx_true = np.asarray(dx_true); dy_true = np.asarray(dy_true)
    dx_pred = np.asarray(dx_pred); dy_pred = np.asarray(dy_pred)

    # unit vector along truth
    norm = np.hypot(dx_true, dy_true)
    norm = np.where(norm == 0, 1e-9, norm)
    ux = dx_true / norm
    uy = dy_true / norm

    # error vector
    ex = dx_pred - dx_true
    ey = dy_pred - dy_true

    along = ex * ux + ey * uy                  # projection onto along-track
    cross = ex * (-uy) + ey * ux               # projection onto left-normal

    out = pd.DataFrame({"along_km": along, "cross_km": cross})
    return out
