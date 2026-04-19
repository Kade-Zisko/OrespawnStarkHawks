#!/usr/bin/env python3
"""
gen_validation_csv.py — Build a flat CSV for C++ model validation.

Runs the full training pipeline on the real dataset:
  raw data → feature engineering → per-hive z-score → StandardScaler → model

Output: models_v2/validation_data.csv
  hive_id, source, split, label,
  f0…f15  (StandardScaler-normalised features, ready for MLP forward pass),
  expected_prob  (PyTorch model output — C++ must match to ±1e-4)

Usage: cd ML && python gen_validation_csv.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models_v2")
sys.path.insert(0, BASE)

import joblib
from model import HiveNet  # noqa: F401
from load_datasets import load_all

model  = joblib.load(os.path.join(MODEL_DIR, "hive_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
model.eval()

WINDOW_24H    = 24
WINDOW_6H     =  6
WINDOW_48H    = 48
BASELINE_DAYS = 21

FEATURE_COLS = [
    "temperature", "humidity",
    "temp_mean_24h", "temp_std_24h", "temp_min_24h", "temp_max_24h", "temp_range_24h",
    "hum_mean_24h",  "hum_std_24h",  "hum_min_24h",  "hum_max_24h",  "hum_range_24h",
    "temp_trend_6h", "hum_trend_6h",
    "temp_trend_48h", "hum_trend_48h",
]

# ── 1. Load & engineer features ───────────────────────────────────────────────
print("Loading dataset...")
df_raw = load_all()

print("Engineering features...")
frames = []
for hive_id, hdf in df_raw.groupby("hive_id"):
    hdf = hdf.sort_values("date").reset_index(drop=True)
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

df = pd.concat(frames, ignore_index=True)
df = df[FEATURE_COLS + ["label", "hive_id", "source", "date"]].dropna()
print(f"After feature engineering: {len(df):,} rows")

# ── 2. Per-hive z-score (mirrors training pipeline exactly) ───────────────────
print("Applying per-hive z-score...")
df = df.copy()
for hive_id, idx in df.groupby("hive_id").groups.items():
    hdf = df.loc[idx].sort_values("date")
    cutoff   = hdf["date"].min() + pd.Timedelta(days=BASELINE_DAYS)
    baseline = hdf[hdf["date"] <= cutoff]
    for col in FEATURE_COLS:
        mu  = float(baseline[col].mean())
        std = float(baseline[col].std())
        std = max(std, 1e-6)
        df.loc[idx, col] = (df.loc[idx, col] - mu) / std

# ── 3. Reproduce train/test split ─────────────────────────────────────────────
rng = np.random.default_rng(42)
test_hives = set()
for src, grp in df.groupby("source"):
    hives  = grp["hive_id"].unique()
    n_test = max(1, int(len(hives) * 0.20))
    chosen = rng.choice(hives, size=n_test, replace=False)
    test_hives.update(chosen)
df["split"] = df["hive_id"].apply(lambda h: "test" if h in test_hives else "train")

# ── 4. StandardScaler ─────────────────────────────────────────────────────────
X_raw = df[FEATURE_COLS].values.astype(np.float32)
X_scaled = scaler.transform(X_raw).astype(np.float32)

# ── 5. Run PyTorch model ──────────────────────────────────────────────────────
print("Running inference...")
BATCH = 4096
probs = []
with torch.no_grad():
    for i in range(0, len(X_scaled), BATCH):
        batch = torch.FloatTensor(X_scaled[i:i + BATCH])
        probs.extend(model(batch).squeeze(-1).numpy().tolist())
        print(f"  {min(i + BATCH, len(X_scaled)):>7,} / {len(X_scaled):,}", end="\r")
print()
probs = np.array(probs, dtype=np.float32)

# ── 6. Write CSV ──────────────────────────────────────────────────────────────
out = df[["hive_id", "source", "split", "label"]].reset_index(drop=True)
for i, col in enumerate(FEATURE_COLS):
    out[f"f{i}"] = X_scaled[:, i]
out["expected_prob"] = probs

out = out.dropna()
out_path = os.path.join(MODEL_DIR, "validation_data.csv")
out.to_csv(out_path, index=False)
print(f"\nWrote {out_path}  ({len(out):,} rows, {os.path.getsize(out_path) / 1e6:.1f} MB)")

# ── 7. Quick sanity check ─────────────────────────────────────────────────────
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

y_true = out["label"].values
y_prob = out["expected_prob"].values
y_pred = (y_prob >= 0.5).astype(int)

test_mask  = out["split"].values == "test"
train_mask = ~test_mask

for tag, mask in [("train", train_mask), ("test", test_mask)]:
    yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
    print(f"\n[{tag}] n={mask.sum():,}  "
          f"acc={accuracy_score(yt, yp):.4f}  "
          f"f1={f1_score(yt, yp, pos_label=1, zero_division=0):.4f}  "
          f"auc={roc_auc_score(yt, ypr):.4f}  "
          f"stressed={yt.sum():,}/{len(yt):,}")
