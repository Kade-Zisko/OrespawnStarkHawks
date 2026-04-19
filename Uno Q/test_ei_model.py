#!/usr/bin/env python3
"""
test_ei_model.py  —  evaluate a built Edge-Impulse ONNX model against the dataset.

Normalisation pipeline
----------------------
Training applied two steps in sequence:
  1. Per-hive z-score  (baseline = first 21 days, sensor cols only)
  2. StandardScaler    (fit on training data)

The EI model bakes in only StandardScaler for the unknown-hive path, so this
script applies per-hive z-score first (step 1) before feeding raw features to
the model (which handles step 2 internally).  Pass --no-zscore when using a
hive-specific EI model that already has both steps baked in.

Usage:
  python test_ei_model.py --model hive_model_ei_unknown.onnx --split test
  python test_ei_model.py --model hive_model_ei_urban_01.onnx --hive urban_01 --no-zscore
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as rt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

# Locate the ML directory relative to this file
ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ML")
sys.path.insert(0, ML_DIR)
from load_datasets import load_all  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))

WINDOW_6H  =  6
WINDOW_12H = 12
WINDOW_24H = 24
WINDOW_48H = 48
WINDOW_72H = 72
BASELINE_DAYS = 21

SENSOR_COLS = [
    "temperature", "humidity",
    "temp_mean_24h", "temp_std_24h", "temp_min_24h", "temp_max_24h", "temp_range_24h",
    "hum_mean_24h",  "hum_std_24h",  "hum_min_24h",  "hum_max_24h",  "hum_range_24h",
    "temp_mean_12h", "temp_std_12h", "temp_min_12h", "temp_max_12h", "temp_range_12h",
    "hum_mean_12h",  "hum_std_12h",  "hum_min_12h",  "hum_max_12h",  "hum_range_12h",
    "temp_mean_72h", "temp_std_72h", "temp_min_72h", "temp_max_72h", "temp_range_72h",
    "hum_mean_72h",  "hum_std_72h",  "hum_min_72h",  "hum_max_72h",  "hum_range_72h",
    "temp_trend_6h",  "hum_trend_6h",
    "temp_trend_48h", "hum_trend_48h",
    "temp_hum_ratio",
]
TEMPORAL_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURE_COLS  = SENSOR_COLS + TEMPORAL_COLS


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for hive_id, hdf in df_raw.groupby("hive_id"):
        hdf = hdf.sort_values("date").reset_index(drop=True)
        t, h = hdf["temperature"], hdf["humidity"]

        for win, tag in [(WINDOW_24H, "24h"), (WINDOW_12H, "12h"), (WINDOW_72H, "72h")]:
            mp = win // 2
            rt_win = t.rolling(win, min_periods=mp)
            rh_win = h.rolling(win, min_periods=mp)
            hdf[f"temp_mean_{tag}"]  = rt_win.mean()
            hdf[f"temp_std_{tag}"]   = rt_win.std()
            hdf[f"temp_min_{tag}"]   = rt_win.min()
            hdf[f"temp_max_{tag}"]   = rt_win.max()
            hdf[f"temp_range_{tag}"] = hdf[f"temp_max_{tag}"] - hdf[f"temp_min_{tag}"]
            hdf[f"hum_mean_{tag}"]   = rh_win.mean()
            hdf[f"hum_std_{tag}"]    = rh_win.std()
            hdf[f"hum_min_{tag}"]    = rh_win.min()
            hdf[f"hum_max_{tag}"]    = rh_win.max()
            hdf[f"hum_range_{tag}"]  = hdf[f"hum_max_{tag}"] - hdf[f"hum_min_{tag}"]

        hdf["temp_trend_6h"]  = t.diff(WINDOW_6H)
        hdf["hum_trend_6h"]   = h.diff(WINDOW_6H)
        hdf["temp_trend_48h"] = t.diff(WINDOW_48H)
        hdf["hum_trend_48h"]  = h.diff(WINDOW_48H)
        hdf["temp_hum_ratio"] = t / h.replace(0, np.nan)

        hour = hdf["date"].dt.hour
        doy  = hdf["date"].dt.dayofyear
        hdf["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        hdf["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        hdf["doy_sin"]  = np.sin(2 * np.pi * doy / 365)
        hdf["doy_cos"]  = np.cos(2 * np.pi * doy / 365)

        frames.append(hdf)

    return pd.concat(frames, ignore_index=True)


def apply_hive_zscore(df: pd.DataFrame, baseline_days: int = BASELINE_DAYS) -> pd.DataFrame:
    """Apply per-hive z-score normalization to SENSOR_COLS using the first
    baseline_days of each hive's data as the baseline — mirrors step 3 of
    the training pipeline so the unknown-hive EI model receives correctly
    scaled inputs before its baked StandardScaler."""
    df = df.copy()
    stats_log = {}
    for hive_id, idx in df.groupby("hive_id").groups.items():
        hdf = df.loc[idx].sort_values("date")
        cutoff   = hdf["date"].min() + pd.Timedelta(days=baseline_days)
        baseline = hdf[hdf["date"] <= cutoff]
        n_baseline = len(baseline)
        for col in SENSOR_COLS:
            mu  = float(baseline[col].mean())
            std = float(baseline[col].std())
            std = max(std, 1e-6)
            df.loc[idx, col] = (df.loc[idx, col] - mu) / std
        stats_log[hive_id] = n_baseline
    print(f"  Per-hive z-score applied to {len(stats_log)} hives  "
          f"(baseline rows: min={min(stats_log.values())} max={max(stats_log.values())})")
    return df


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thresholds[np.argmax(f1[:-1])])


def consecutive_filter(preds: np.ndarray, n: int) -> np.ndarray:
    """Require n consecutive stressed readings before triggering an alert.
    Applied per-hive in date order; resets the counter on any healthy reading."""
    out, count = np.zeros_like(preds), 0
    for i, p in enumerate(preds):
        count = count + 1 if p == 1 else 0
        if count >= n:
            out[i] = 1
    return out


def print_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, tag: str = "") -> None:
    prefix = f"[{tag}] " if tag else ""
    print(f"\n{prefix}Samples : {len(y_true):,}  "
          f"(healthy={int((y_true==0).sum()):,}  stressed={int((y_true==1).sum()):,})")
    print(classification_report(y_true, y_pred, target_names=["healthy", "stressed"], digits=4))
    print(f"  ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1-stress: {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an EI ONNX model against the dataset.")
    parser.add_argument("--model", required=True,
                        help="Path to EI ONNX file (e.g. hive_model_ei_unknown.onnx).")
    parser.add_argument("--hive", default=None,
                        help="Restrict evaluation to a single hive_id.")
    parser.add_argument("--split", choices=["all", "train", "test"], default="all",
                        help="Which data split to evaluate (default: all).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the baked decision threshold (e.g. 0.15).")
    parser.add_argument("--no-zscore", action="store_true",
                        help="Skip per-hive z-score (use for hive-specific EI models "
                             "that already have z-score baked in).")
    parser.add_argument("--consecutive", type=int, default=3,
                        help="Consecutive stressed readings required before triggering (default: 3).")
    args = parser.parse_args()

    model_path = args.model if os.path.isabs(args.model) else os.path.join(BASE, args.model)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Model : {model_path}")
    sess = rt.InferenceSession(model_path)

    # ── Load & engineer features ──────────────────────────────────────────────
    print("\nLoading dataset...")
    df_raw = load_all()
    print(f"Raw rows: {len(df_raw):,}  hives: {df_raw['hive_id'].nunique()}")

    print("Engineering features...")
    df = engineer_features(df_raw)
    df = df[FEATURE_COLS + ["label", "hive_id", "source", "date"]].dropna()
    print(f"After feature engineering & dropna: {len(df):,} rows")

    if not args.no_zscore:
        print("Applying per-hive z-score normalization (first 21 days baseline)...")
        df = apply_hive_zscore(df)
    else:
        print("Skipping per-hive z-score (--no-zscore set).")

    # ── Optionally reproduce the same train/test split ────────────────────────
    if args.split != "all":
        rng = np.random.default_rng(42)
        test_hives: set = set()
        for src, grp in df.groupby("source"):
            hives  = grp["hive_id"].unique()
            n_test = max(1, int(len(hives) * 0.20))
            chosen = rng.choice(hives, size=n_test, replace=False)
            test_hives.update(chosen)
        if args.split == "test":
            df = df[df["hive_id"].isin(test_hives)]
        else:
            df = df[~df["hive_id"].isin(test_hives)]
        print(f"Split '{args.split}': {len(df):,} rows  hives: {df['hive_id'].nunique()}")

    # ── Optionally restrict to one hive ──────────────────────────────────────
    if args.hive:
        df = df[df["hive_id"] == args.hive]
        if df.empty:
            print(f"No data found for hive '{args.hive}'.")
            sys.exit(1)
        print(f"Filtered to hive '{args.hive}': {len(df):,} rows")

    # ── Run inference in batches ──────────────────────────────────────────────
    df = df.reset_index(drop=True)
    print("\nRunning inference...")
    X = df[FEATURE_COLS].values.astype(np.float32)
    y_true = df["label"].values.astype(int)

    output_names = [o.name for o in sess.get_outputs()]
    has_label    = "label" in output_names
    fetch        = ["probability", "label"] if has_label else ["probability"]
    if not has_label:
        print("  Model has no baked 'label' output — threshold will be derived from probabilities.")

    BATCH = 1024
    all_probs, all_preds = [], []
    for i in range(0, len(X), BATCH):
        batch = X[i : i + BATCH]
        probs_b = []
        preds_b = []
        for row in batch:
            outputs = sess.run(fetch, {"features": row[np.newaxis, :]})
            probs_b.append(float(outputs[0][0, 0]))
            preds_b.append(int(outputs[1][0, 0]) if has_label else -1)
        all_probs.extend(probs_b)
        all_preds.extend(preds_b)
        print(f"  {min(i + BATCH, len(X)):>7,} / {len(X):,}", end="\r")

    y_prob = np.array(all_probs)

    # ── Threshold analysis ────────────────────────────────────────────────────
    opt_thr    = args.threshold if args.threshold is not None else optimal_threshold(y_true, y_prob)
    y_pred_opt = (y_prob >= opt_thr).astype(int)

    if has_label:
        y_pred    = np.array(all_preds)
        baked_thr = float((y_prob[y_pred == 1].min()) if (y_pred == 1).any() else float("nan"))
    else:
        y_pred    = y_pred_opt
        baked_thr = float("nan")

    print(f"\n── Threshold analysis ───────────────────────────────────────────────────")
    print(f"  Prob range   : [{y_prob.min():.4f}, {y_prob.max():.4f}]  mean={y_prob.mean():.4f}")
    print(f"  Baked thr    : {baked_thr if not np.isnan(baked_thr) else '(model never predicted stressed)'}")
    print(f"  Optimal thr  : {opt_thr:.4f}  (maximises F1-stressed on this split)")
    if has_label and (y_pred == 1).sum() == 0:
        print("  WARNING: baked threshold is too high — model never predicts STRESSED.")
        print(f"           Re-bake the model with --threshold {opt_thr:.4f} or use the optimal below.")

    # ── Per-hive consecutive smoothing (applied in date order within each hive) ─
    y_pred_smooth = np.zeros(len(df), dtype=int)
    for hid, grp in df.groupby("hive_id"):
        pos = grp.sort_values("date").index.to_numpy()
        y_pred_smooth[pos] = consecutive_filter(y_pred_opt[pos], args.consecutive)

    # ── Overall metrics ───────────────────────────────────────────────────────
    print_metrics(y_true, y_prob, y_pred,        tag="OVERALL (baked threshold)")
    print_metrics(y_true, y_prob, y_pred_opt,    tag=f"OVERALL (optimal thr={opt_thr:.4f})")
    print_metrics(y_true, y_prob, y_pred_smooth,
                  tag=f"OVERALL (smoothed ≥{args.consecutive} consecutive, thr={opt_thr:.4f})")

    # ── Per-source breakdown ──────────────────────────────────────────────────
    for src in sorted(df["source"].unique()):
        mask = df["source"].values == src
        if mask.sum() == 0:
            continue
        print_metrics(y_true[mask], y_prob[mask], y_pred_smooth[mask],
                      tag=f"{src} (smoothed ≥{args.consecutive})")

    # ── Per-hive breakdown ────────────────────────────────────────────────────
    print(f"\nPer-hive breakdown (optimal thr={opt_thr:.4f}, smoothed ≥{args.consecutive} consecutive):")
    print(f"  {'hive_id':<20s}  {'n':>6}  {'AUC':>6}  {'F1-s':>6}  {'F1-sm':>6}  {'acc-sm':>6}")
    for hid in sorted(df["hive_id"].unique()):
        mask = df["hive_id"].values == hid
        yt, ypr = y_true[mask], y_prob[mask]
        yp_opt = y_pred_opt[mask]
        yp_sm  = y_pred_smooth[mask]
        try:
            auc = f"{roc_auc_score(yt, ypr):.4f}"
        except ValueError:
            auc = "  N/A "
        f1s  = f1_score(yt, yp_opt, pos_label=1, zero_division=0)
        f1sm = f1_score(yt, yp_sm,  pos_label=1, zero_division=0)
        acc  = accuracy_score(yt, yp_sm)
        print(f"  {hid:<20s}  {mask.sum():>6,}  {auc:>6}  {f1s:>6.4f}  {f1sm:>6.4f}  {acc:>6.4f}")


if __name__ == "__main__":
    main()
