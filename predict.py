"""
Beehive Health Classifier — Inference script

Usage
-----
    python3 predict.py --input path/to/sensor.csv
    python3 predict.py --input path/to/sensor.csv --hive 42

Output
------
Prints a summary and writes predictions to models/predictions.csv.

Expected input CSV columns
--------------------------
  Date        — ISO-8601 timestamp (e.g. 2021-07-01 00:00:00+00:00)
  temperature — internal hive temperature in °C
  humidity    — internal hive humidity in %
  Tag number  — hive ID  (optional — use --hive if absent)
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from model import HiveNet  # needed so joblib can unpickle HiveNet

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

WINDOW_24H = 24
WINDOW_6H  =  6
WINDOW_48H = 48


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, hdf in df.groupby("Tag number"):
        hdf = hdf.sort_values("Date").reset_index(drop=True)
        t, h = hdf["temperature"], hdf["humidity"]

        rt = t.rolling(WINDOW_24H, min_periods=WINDOW_24H // 2)
        rh = h.rolling(WINDOW_24H, min_periods=WINDOW_24H // 2)

        hdf["temp_mean_24h"]  = rt.mean()
        hdf["temp_std_24h"]   = rt.std()
        hdf["temp_min_24h"]   = rt.min()
        hdf["temp_max_24h"]   = rt.max()
        hdf["temp_range_24h"] = hdf["temp_max_24h"] - hdf["temp_min_24h"]
        hdf["hum_mean_24h"]   = rh.mean()
        hdf["hum_std_24h"]    = rh.std()
        hdf["hum_min_24h"]    = rh.min()
        hdf["hum_max_24h"]    = rh.max()
        hdf["hum_range_24h"]  = hdf["hum_max_24h"] - hdf["hum_min_24h"]
        hdf["temp_trend_6h"]  = t.diff(WINDOW_6H)
        hdf["hum_trend_6h"]   = h.diff(WINDOW_6H)
        hdf["temp_trend_48h"] = t.diff(WINDOW_48H)
        hdf["hum_trend_48h"]  = h.diff(WINDOW_48H)

        frames.append(hdf)
    return pd.concat(frames, ignore_index=True)


def apply_zscore(df: pd.DataFrame, feature_cols: list, hive_stats: dict) -> pd.DataFrame:
    df = df.copy()
    for hive_id, hdf in df.groupby("Tag number"):
        stats = hive_stats.get(str(hive_id)) or hive_stats.get(hive_id)
        if stats is None:
            cutoff   = hdf["Date"].min() + pd.Timedelta(days=21)
            baseline = hdf[hdf["Date"] <= cutoff]
            stats = {
                col: {"mean": float(baseline[col].mean()),
                      "std":  max(float(baseline[col].std()), 1e-6)}
                for col in feature_cols if col in hdf.columns
            }
            print(f"  Hive {hive_id}: no saved baseline — computed from first 21 days.")
        for col, s in stats.items():
            if col in df.columns and s["std"] > 0:
                df.loc[hdf.index, col] = (df.loc[hdf.index, col] - s["mean"]) / s["std"]
    return df


def score(df: pd.DataFrame, model: HiveNet, feature_cols: list, scaler) -> pd.DataFrame:
    df    = df.copy()
    valid = df[feature_cols].notna().all(axis=1)
    df["stress_prob"] = np.nan
    df["prediction"]  = "insufficient_data"
    if valid.any():
        X = df.loc[valid, feature_cols].values
        if scaler is not None:
            X = scaler.transform(X)
        model.eval()
        with torch.no_grad():
            probs = model(torch.FloatTensor(X)).squeeze(-1).numpy()
        df.loc[valid, "stress_prob"] = probs
        df.loc[valid, "prediction"]  = np.where(probs >= 0.5, "stressed", "healthy")
    return df


def main():
    parser = argparse.ArgumentParser(description="Beehive health classifier — inference")
    parser.add_argument("--input", metavar="CSV", required=True,
                        help="Path to sensor CSV to score")
    parser.add_argument("--hive", type=int, default=0,
                        help="Hive ID to assign if CSV has no 'Tag number' column")
    args = parser.parse_args()

    for fname in ("hive_model.pkl", "feature_cols.pkl", "scaler.pkl"):
        if not os.path.exists(os.path.join(MODEL_DIR, fname)):
            sys.exit(f"{fname} not found. Run train_classifier.py first.")

    model        = joblib.load(os.path.join(MODEL_DIR, "hive_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    hive_stats   = {}
    stats_path   = os.path.join(MODEL_DIR, "hive_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            hive_stats = json.load(f)

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    if "Tag number" not in df.columns:
        df["Tag number"] = args.hive

    df = df[df["temperature"].between(10, 60) & df["humidity"].between(1, 100)].copy()
    df = build_features(df)
    df = apply_zscore(df, feature_cols, hive_stats)
    df = score(df, model, feature_cols, scaler)

    out_path = os.path.join(MODEL_DIR, "predictions.csv")
    df[["Date", "Tag number", "temperature", "humidity",
        "stress_prob", "prediction"]].to_csv(out_path, index=False)
    print(f"Saved {len(df):,} predictions → {out_path}")

    scored = df[df["prediction"] != "insufficient_data"]
    if scored.empty:
        print("No rows had enough history to score (need ≥12 prior readings).")
        return

    print(f"\nSummary — {len(scored):,} readings scored:")
    print(f"  healthy  : {(scored['prediction'] == 'healthy').sum():,}")
    print(f"  stressed : {(scored['prediction'] == 'stressed').sum():,}")
    print(f"  mean stress probability : {scored['stress_prob'].mean():.3f}")

    print("\nSample predictions (first 10 scored rows):")
    print(scored[["Date", "Tag number", "temperature", "humidity",
                  "stress_prob", "prediction"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
