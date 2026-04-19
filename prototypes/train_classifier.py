"""
Beehive Health Classifier — unified training script.
Merges UrBAN (7 hives), MSPB (85 hives), and German BOB (78 hives) datasets.

Pipeline
--------
1.  Load + merge all three sources via load_datasets.py
2.  Rolling feature engineering (24 h window, 6 h trend — hourly resolution)
3.  Per-hive z-score normalisation (first-21-day healthy baseline)
4.  Hive-based 80 / 20 train / test split (no hive appears in both sets)
5.  StandardScaler on all features (saved as scaler.pkl)
6.  5-fold CV on training hives, final evaluation on held-out test hives
7.  Save model artefacts to models/
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from load_datasets import load_all
from model import HiveNet

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_24H    = 24
WINDOW_6H     =  6
BASELINE_DAYS = 21

FEATURE_COLS = [
    "temperature", "humidity",
    "temp_mean_24h", "temp_std_24h", "temp_min_24h", "temp_max_24h", "temp_range_24h",
    "hum_mean_24h",  "hum_std_24h",  "hum_min_24h",  "hum_max_24h",  "hum_range_24h",
    "temp_trend_6h", "hum_trend_6h",
]

N_FEATURES = len(FEATURE_COLS)
EPOCHS     = 50
LR         = 1e-3
BATCH_SIZE = 512


# ── training helpers ──────────────────────────────────────────────────────────

def make_loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    weights = np.where(y == 1, n_neg / max(n_pos, 1), 1.0)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(y), replacement=True)
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


def train_model(X: np.ndarray, y: np.ndarray) -> HiveNet:
    model     = HiveNet(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    loader    = make_loader(X, y)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def eval_model(model: HiveNet, X: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(X)).squeeze(-1).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy":    float(accuracy_score(y, preds)),
        "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "roc_auc":     float(roc_auc_score(y, probs)),
    }


# ── 1. Load merged dataset ────────────────────────────────────────────────────
df_raw = load_all()


# ── 2. Rolling feature engineering ───────────────────────────────────────────
print("\nEngineering rolling features (hourly windows)...")
feature_frames = []
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

    feature_frames.append(hdf)

df_feats = pd.concat(feature_frames, ignore_index=True)
print(f"Feature matrix shape: {df_feats.shape}")


# ── 3. Per-hive z-score normalisation ────────────────────────────────────────
print(f"Applying per-hive z-score normalisation (baseline = first {BASELINE_DAYS} days)...")
hive_stats: dict = {}
df_feats = df_feats.copy()
for hive_id, hdf in df_feats.groupby("hive_id"):
    cutoff   = hdf["date"].min() + pd.Timedelta(days=BASELINE_DAYS)
    baseline = hdf[hdf["date"] <= cutoff]
    stats: dict = {}
    for col in FEATURE_COLS:
        mu  = float(baseline[col].mean())
        std = float(baseline[col].std())
        stats[col] = {"mean": mu, "std": max(std, 1e-6)}
        df_feats.loc[hdf.index, col] = (hdf[col] - mu) / stats[col]["std"]
    hive_stats[hive_id] = stats

with open(os.path.join(MODEL_DIR, "hive_stats.json"), "w") as f:
    json.dump(hive_stats, f, indent=2)
print(f"Normalised {len(hive_stats)} hives.")


# ── 4. Hive-based train / test split ─────────────────────────────────────────
df_model = df_feats[FEATURE_COLS + ["label", "hive_id", "source"]].dropna()

rng = np.random.default_rng(42)
test_hives = set()
for src, grp in df_model.groupby("source"):
    hives  = grp["hive_id"].unique()
    n_test = max(1, int(len(hives) * 0.20))
    chosen = rng.choice(hives, size=n_test, replace=False)
    test_hives.update(chosen)

train_mask = ~df_model["hive_id"].isin(test_hives)
test_mask  =  df_model["hive_id"].isin(test_hives)

X_train_raw = df_model.loc[train_mask, FEATURE_COLS].values
y_train     = df_model.loc[train_mask, "label"].values.astype(int)
X_test_raw  = df_model.loc[test_mask,  FEATURE_COLS].values
y_test      = df_model.loc[test_mask,  "label"].values.astype(int)

print(f"\nTrain: {len(X_train_raw):,} samples  "
      f"({(y_train==0).sum():,} healthy / {(y_train==1).sum():,} stressed)  "
      f"hives={df_model.loc[train_mask,'hive_id'].nunique()}")
print(f"Test : {len(X_test_raw):,} samples  "
      f"({(y_test==0).sum():,} healthy / {(y_test==1).sum():,} stressed)  "
      f"hives={df_model.loc[test_mask,'hive_id'].nunique()}")

print("\nTest hive counts per source:")
for src, grp in df_model[test_mask].groupby("source"):
    h  = grp["hive_id"].nunique()
    ht = (grp["label"] == 0).sum()
    hs = (grp["label"] == 1).sum()
    print(f"  {src:<8s}  hives={h}  healthy={ht:,}  stressed={hs:,}")


# ── 5. StandardScaler ─────────────────────────────────────────────────────────
print("\nFitting StandardScaler on training data...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("Saved scaler.pkl")


# ── 6. 5-fold CV on training hives ───────────────────────────────────────────
print("\n── 5-fold CV on training hives ──────────────────────────────────────────")
cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = {"accuracy": [], "f1_weighted": [], "roc_auc": []}

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    fold_model = train_model(X_train[tr_idx], y_train[tr_idx])
    m = eval_model(fold_model, X_train[val_idx], y_train[val_idx])
    fold_metrics["accuracy"].append(m["accuracy"])
    fold_metrics["f1_weighted"].append(m["f1_weighted"])
    fold_metrics["roc_auc"].append(m["roc_auc"])
    print(f"  Fold {fold}  Acc={m['accuracy']:.4f}  "
          f"F1={m['f1_weighted']:.4f}  AUC={m['roc_auc']:.4f}")

cv_results = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
print(f"\n  Mean  Acc={cv_results['accuracy']:.4f}  "
      f"F1={cv_results['f1_weighted']:.4f}  AUC={cv_results['roc_auc']:.4f}")


# ── 7. Train final model on all training data ─────────────────────────────────
print("\n── Training final model ─────────────────────────────────────────────────")
final_model = train_model(X_train, y_train)
final_model.eval()

with torch.no_grad():
    y_prob = final_model(torch.FloatTensor(X_test)).squeeze(-1).numpy()
y_pred = (y_prob >= 0.5).astype(int)

print(classification_report(y_test, y_pred, target_names=["healthy", "stressed"]))
test_roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC (test): {test_roc_auc:.4f}")


# ── 8. Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(final_model,  os.path.join(MODEL_DIR, "hive_model.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

dummy_input = torch.randn(1, N_FEATURES)
torch.onnx.export(
    final_model,
    dummy_input,
    os.path.join(MODEL_DIR, "hive_model.onnx"),
    input_names=["features"],
    output_names=["stress_prob"],
    opset_version=13,
    dynamic_axes={
        "features":    {0: "batch_size"},
        "stress_prob": {0: "batch_size"},
    },
)

report = {
    "model": "HiveNet MLP  14 → 32 → 16 → 1",
    "architecture": {"layers": [14, 32, 16, 1], "activation": "ReLU", "output": "Sigmoid"},
    "training": {"epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                 "optimizer": "Adam", "class_balance": "WeightedRandomSampler"},
    "dataset": {
        "sources": ["urban", "mspb", "german"],
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "n_features": int(N_FEATURES),
        "n_hives_train": int(df_model.loc[train_mask, "hive_id"].nunique()),
        "n_hives_test":  int(df_model.loc[test_mask,  "hive_id"].nunique()),
        "label_criteria": (
            "UrBAN: QNS or fob<12 | "
            "MSPB: hub-level varroa>2 or winter death | "
            "German: colony died vs survived"
        ),
        "normalisation": "per-hive z-score (first-21-day baseline) + StandardScaler",
    },
    "feature_cols": FEATURE_COLS,
    "cv_results_on_train": cv_results,
    "test_results": {
        "roc_auc": float(test_roc_auc),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["healthy", "stressed"],
            output_dict=True,
        ),
    },
}
with open(os.path.join(MODEL_DIR, "report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved: models/hive_model.pkl  hive_model.onnx  scaler.pkl  "
      f"feature_cols.pkl  hive_stats.json  report.json")
print("Done.")
