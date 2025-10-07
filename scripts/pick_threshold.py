#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precision-Target Threshold Finder (Diagnostics Only)
====================================================
Reads a predictions parquet (with y_true, y_proba), sweeps the PR curve,
and returns the *highest-recall* operating threshold that achieves a given
precision target on TEST. Intended for quick post-hoc analysis; it does NOT
write any model weights or alter training in any way.

Why this exists
---------------
Operational users often want “give me a threshold that gets me ≥ X precision”.
This script gives a reproducible answer and basic confusion-matrix counts at
that operating point.

Inputs
------
--preds : path to parquet with columns {y_true:int, y_proba:float}
--target_precision : desired minimum precision (default 0.30)

Outputs
-------
- If --out is set: JSON file with threshold, achieved P/R, and confusion counts.
- Else: prints the same info to stdout.

Notes
-----
- If no point on the PR curve reaches the target precision, we report the
  maximum precision observed and do not invent a threshold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds",
        type=str,
        default="outputs/ri_classification/predictions_test.parquet",
        help="Path to predictions parquet with columns: y_true, y_proba",
    )
    parser.add_argument(
        "--target_precision",
        type=float,
        default=0.30,
        help="Target precision to satisfy on TEST (diagnostics only).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional path to write JSON result. If empty, prints to stdout.",
    )
    args = parser.parse_args()

    # --------------------
    # Load + basic schema
    # --------------------
    pth = Path(args.preds)
    if not pth.exists():
        raise FileNotFoundError(f"Could not find predictions file: {pth}")

    df = pd.read_parquet(pth)
    for need in ("y_true", "y_proba"):
        if need not in df.columns:
            raise ValueError(f"Column '{need}' not found in {pth}")

    y_true = df["y_true"].to_numpy().astype(int)
    scores = df["y_proba"].to_numpy()

    # --------------------
    # PR curve + selection
    # --------------------
    precision, recall, thr = precision_recall_curve(y_true, scores)

    # Ignore the last precision element (scikit-learn appends 1.0 with undefined threshold)
    idx = np.where(precision[:-1] >= args.target_precision)[0]

    if idx.size:
        # Among all thresholds that meet the precision target, choose the one
        # with the highest recall (most permissive subject to the constraint).
        i = idx[np.argmax(recall[idx])]
        t = float(thr[i])

        yhat = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()

        out = {
            "threshold": t,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "target_precision": args.target_precision,
            "preds_path": str(pth),
        }
    else:
        # No feasible point hits the requested precision.
        out = {
            "note": f"No point on test reaches P>={args.target_precision:.2f}",
            "max_precision_on_curve": float(precision[:-1].max()) if precision.size else 0.0,
            "target_precision": args.target_precision,
            "preds_path": str(pth),
        }

    # -------------
    # Persist/print
    # -------------
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(out)


if __name__ == "__main__":
    main()
