# scripts/ri_lstm.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rapid-Intensification (RI) Sequence Model — Minimal LSTM Baseline
=================================================================
This script trains a small LSTM on fixed-length (WINDOW) 6-hourly sequences
to predict whether a TC will undergo RI in the next 24h.

Design notes
------------
- We intentionally keep the architecture tiny (1× LSTM) to serve as a
  *reference sequence model* parallel to the tabular baselines.
- Sequence construction is strictly causal: targets at time t are predicted
  using windows up to t (no look-ahead).
- Year-based splits avoid temporal leakage and mimic an operational regime.
- Feature scaling is computed on TRAIN ONLY and applied to val/test.

Outputs
-------
- Prints VAL/TEST AUPRC to stdout.
- Writes test probabilities to:
  outputs/ri_classification/lstm_proba_test.csv

Caveats
-------
- This is not heavily tuned. For research use, you’ll likely want
  deeper stacks, dropout, class-balanced sampling, and calibration.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler  # retained for parity, not used directly

# --------------------------
# Configuration (keep small)
# --------------------------
WINDOW = 4  # 4 × 6h steps = 24h context
FEATS = [
    "wmo_wind", "wmo_pres", "abs_lat", "doy_sin", "doy_cos",
    "wind10_mag", "shear_200_850", "era5_mslp", "era5_sst"
]  # adjust to your available columns


class LSTM(nn.Module):
    """
    Tiny, single-layer LSTM classifier.
    - Consumes [B, T, F] and emits P(RI) via a sigmoid head.
    - “Head on last hidden” is sufficient for a baseline.
    """
    def __init__(self, f: int, h: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=f,
            hidden_size=h,
            batch_first=True,
            num_layers=1,
            bidirectional=False
        )
        self.head = nn.Sequential(nn.Linear(h, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out: [B, T, H]; we use only the last step
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn a storm’s time series into overlapping fixed-length windows.

    Returns
    -------
    X : float32 array [N, WINDOW, F]
    y : float32 array [N] (binary labels)
    """
    df = df.sort_values(["storm_id", "iso_time"])
    Xs, ys = [], []

    # Fit a scaler “later” on train only; keep raw values here for now
    for _, g in df.groupby("storm_id"):
        g = g.reset_index(drop=True)
        F = g[FEATS].values.astype(np.float32)
        y = g["RI"].values.astype(np.float32)

        # Sliding window ending at t (inclusive)
        for t in range(WINDOW - 1, len(g)):
            Xs.append(F[t - WINDOW + 1:t + 1])
            ys.append(y[t])

    X = np.stack(Xs)  # [N, T, F]
    y = np.array(ys)
    return X, y


def split_by_year(
    df: pd.DataFrame,
    train_end: int = 2018,
    val_start: int = 2019,
    val_end: int = 2021,
    test_start: int = 2022,
):
    """
    Year-based temporal split; returns boolean masks (train, val, test).
    """
    y = pd.to_datetime(df["iso_time"]).dt.year
    return (y <= train_end).values, ((y >= val_start) & (y <= val_end)).values, (y >= test_start).values


def run(path: str):
    """
    End-to-end training loop:
      1) Load parquet; ensure minimal features/labels exist.
      2) Build train/val/test windows.
      3) Scale using train stats; train tiny LSTM with early stopping on val AUPRC.
      4) Report VAL/TEST AUPRC and dump test probabilities to CSV.
    """
    df = pd.read_parquet(path)

    # ---- Minimal on-the-fly feature engineering to match baseline ----
    df["doy"] = pd.to_datetime(df["iso_time"]).dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

    if "wind10_mag" not in df and {"era5_u10", "era5_v10"}.issubset(df.columns):
        df["wind10_mag"] = np.sqrt(df["era5_u10"] ** 2 + df["era5_v10"] ** 2)

    if "shear_200_850" not in df and {"era5_u200", "era5_v200", "era5_u850", "era5_v850"}.issubset(df.columns):
        du = df["era5_u200"] - df["era5_u850"]
        dv = df["era5_v200"] - df["era5_v850"]
        df["shear_200_850"] = np.sqrt(du ** 2 + dv ** 2)

    # Label construction (24h delta via 4 × 6h shift)
    if "RI" not in df:
        df["wmo_wind_next24"] = df.groupby("storm_id")["wmo_wind"].shift(-4)
        df["RI"] = ((df["wmo_wind_next24"] - df["wmo_wind"]) >= 30).astype(int)

    # Keep rows with all required features and identifiers
    df = df.dropna(subset=FEATS + ["RI", "storm_id", "iso_time"]).copy()

    # -----------------------
    # Temporal split + build
    # -----------------------
    tr, va, te = split_by_year(df)
    X_tr, y_tr = build_sequences(df.loc[tr])
    X_va, y_va = build_sequences(df.loc[va])
    X_te, y_te = build_sequences(df.loc[te])

    # --------------------------------------
    # Scale each feature using TRAIN only
    # --------------------------------------
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(0)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(0) + 1e-6

    def norm(X: np.ndarray) -> np.ndarray:
        return (X - mu) / sd

    X_tr, X_va, X_te = norm(X_tr), norm(X_va), norm(X_te)

    # --------------------------
    # Model / optimizer set-up
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(f=X_tr.shape[-1]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # Class imbalance weighting: negatives / positives
    posw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    bce = nn.BCELoss(weight=torch.tensor(posw).to(device))

    def to_loader(X, y, b: int = 256, shuf: bool = True):
        ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        return torch.utils.data.DataLoader(ds, batch_size=b, shuffle=shuf, drop_last=False)

    Ltr, Lva = to_loader(X_tr, y_tr), to_loader(X_va, y_va, shuf=False)

    # -----------------------
    # Train with early stop
    # -----------------------
    best = None
    patience, bad = 5, 0
    for epoch in range(40):
        model.train()
        for xb, yb in Ltr:
            xb, yb = xb.to(device), yb.to(device)
            p = model(xb)
            loss = bce(p, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Early stopping: monitor validation AUPRC
        model.eval()
        with torch.no_grad():
            pv = model(torch.tensor(X_va).to(device)).cpu().numpy()
        ap = average_precision_score(y_va, pv)

        if best is None or ap > best[0]:
            best = (ap, model.state_dict().copy())
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    # -----------------------
    # Final eval + save preds
    # -----------------------
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(X_va).to(device)).cpu().numpy()
        pt = model(torch.tensor(X_te).to(device)).cpu().numpy()

    print("VAL AUPRC:", average_precision_score(y_va, pv))
    print("TEST AUPRC:", average_precision_score(y_te, pt))
    pd.Series(pt).to_csv("outputs/ri_classification/lstm_proba_test.csv", index=False)


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "outputs/model_ready_2000_2024_v2.parquet")
