"""
Beehive Health Classifier — Heavy v2 (global StandardScaler only).

Identical to train_classifier_heavy.py except:
- Per-hive z-score normalization is removed; only StandardScaler is applied.
- ONNX model outputs sigmoid probability directly (no separate threshold node).
- Saves to models_heavy_v2/.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from load_datasets import load_all

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models_heavy_v2")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── feature config ────────────────────────────────────────────────────────────

WINDOW_6H  =  6
WINDOW_12H = 12
WINDOW_24H = 24
WINDOW_48H = 48
WINDOW_72H = 72

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
N_FEATURES    = len(FEATURE_COLS)

# ── hyper-parameters ──────────────────────────────────────────────────────────

DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 512
MAX_EPOCHS   = 300
PATIENCE     = 15
POS_WEIGHT   = torch.tensor([1859.0 / 380.0])  # n_healthy / n_stressed ≈ 4.89


# ── model ─────────────────────────────────────────────────────────────────────

class HiveNetHeavy(nn.Module):
    """N → 128 → 64 → 32 → 16 → 1  (BatchNorm + Dropout)."""

    def __init__(self, n_features: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # no Sigmoid — BCEWithLogitsLoss expects raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _HiveNetWithSigmoid(nn.Module):
    """Wraps HiveNetHeavy with a final Sigmoid for ONNX export."""

    def __init__(self, base: HiveNetHeavy):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.base(x))


# ── helpers ───────────────────────────────────────────────────────────────────

def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises F1 on the stressed (positive) class."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thresholds[np.argmax(f1[:-1])])


def make_loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    k = min(3, int((y == 1).sum()) - 1)
    smote = SMOTE(k_neighbors=max(1, k), random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    ds = TensorDataset(torch.FloatTensor(X_res), torch.FloatTensor(y_res))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


def train_model(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[HiveNetHeavy, int]:
    model     = HiveNetHeavy(X_tr.shape[1], dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    loader    = make_loader(X_tr, y_tr)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    stopped_epoch = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t).squeeze(-1), y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                stopped_epoch = epoch - PATIENCE
                break

    model.load_state_dict(best_state)
    return model, stopped_epoch


def eval_model(
    model: HiveNetHeavy,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(X))).squeeze(-1).numpy()
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":    float(accuracy_score(y, preds)),
        "f1_stressed": float(f1_score(y, preds, pos_label=1, zero_division=0)),
        "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "roc_auc":     float(roc_auc_score(y, probs)),
        "probs":       probs,
    }


# ── 1. Load data ──────────────────────────────────────────────────────────────

df_raw = load_all()


# ── 2. Feature engineering ────────────────────────────────────────────────────

print("\nEngineering features...")
frames = []
for hive_id, hdf in df_raw.groupby("hive_id"):
    hdf = hdf.sort_values("date").reset_index(drop=True)
    t, h = hdf["temperature"], hdf["humidity"]

    for win, tag in [(WINDOW_24H, "24h"), (WINDOW_12H, "12h"), (WINDOW_72H, "72h")]:
        mp = win // 2
        rt = t.rolling(win, min_periods=mp)
        rh = h.rolling(win, min_periods=mp)
        hdf[f"temp_mean_{tag}"]  = rt.mean()
        hdf[f"temp_std_{tag}"]   = rt.std()
        hdf[f"temp_min_{tag}"]   = rt.min()
        hdf[f"temp_max_{tag}"]   = rt.max()
        hdf[f"temp_range_{tag}"] = hdf[f"temp_max_{tag}"] - hdf[f"temp_min_{tag}"]
        hdf[f"hum_mean_{tag}"]   = rh.mean()
        hdf[f"hum_std_{tag}"]    = rh.std()
        hdf[f"hum_min_{tag}"]    = rh.min()
        hdf[f"hum_max_{tag}"]    = rh.max()
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

df_feats = pd.concat(frames, ignore_index=True)
print(f"Feature matrix shape: {df_feats.shape}  ({N_FEATURES} model features)")


# ── 3. Hive-based train / test split ──────────────────────────────────────────

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
      f"hives={df_model.loc[train_mask, 'hive_id'].nunique()}")
print(f"Test : {len(X_test_raw):,} samples  "
      f"({(y_test==0).sum():,} healthy / {(y_test==1).sum():,} stressed)  "
      f"hives={df_model.loc[test_mask,  'hive_id'].nunique()}")

print("\nTest hive counts per source:")
for src, grp in df_model[test_mask].groupby("source"):
    h  = grp["hive_id"].nunique()
    ht = (grp["label"] == 0).sum()
    hs = (grp["label"] == 1).sum()
    print(f"  {src:<8s}  hives={h}  healthy={ht:,}  stressed={hs:,}")


# ── 4. StandardScaler ─────────────────────────────────────────────────────────

print("\nFitting StandardScaler...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("Saved scaler.pkl")


# ── 5. 5-fold CV ──────────────────────────────────────────────────────────────

print("\n── 5-fold CV ────────────────────────────────────────────────────────────")
cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = {"accuracy": [], "f1_stressed": [], "f1_weighted": [], "roc_auc": []}
cv_thresholds = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    fold_model, stopped_epoch = train_model(
        X_train[tr_idx], y_train[tr_idx],
        X_train[val_idx], y_train[val_idx],
    )
    fold_model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(fold_model(torch.FloatTensor(X_train[val_idx]))).squeeze(-1).numpy()

    thr = optimal_threshold(y_train[val_idx], val_probs)
    cv_thresholds.append(thr)

    m = eval_model(fold_model, X_train[val_idx], y_train[val_idx], threshold=thr)
    fold_metrics["accuracy"].append(m["accuracy"])
    fold_metrics["f1_stressed"].append(m["f1_stressed"])
    fold_metrics["f1_weighted"].append(m["f1_weighted"])
    fold_metrics["roc_auc"].append(m["roc_auc"])
    print(f"  Fold {fold}  stopped@{stopped_epoch:>3}  thr={thr:.3f}  "
          f"Acc={m['accuracy']:.4f}  F1-stressed={m['f1_stressed']:.4f}  "
          f"F1-w={m['f1_weighted']:.4f}  AUC={m['roc_auc']:.4f}")

best_threshold = float(np.mean(cv_thresholds))
cv_results     = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
print(f"\n  Mean threshold : {best_threshold:.3f}")
print(f"  Mean  Acc={cv_results['accuracy']:.4f}  "
      f"F1-stressed={cv_results['f1_stressed']:.4f}  "
      f"F1-w={cv_results['f1_weighted']:.4f}  AUC={cv_results['roc_auc']:.4f}")


# ── 6. Final model ────────────────────────────────────────────────────────────

print("\n── Training final model ─────────────────────────────────────────────────")
val_size  = int(len(X_train) * 0.10)
X_tr_fin  = X_train[val_size:]
y_tr_fin  = y_train[val_size:]
X_val_fin = X_train[:val_size]
y_val_fin = y_train[:val_size]

final_model, stopped_epoch = train_model(X_tr_fin, y_tr_fin, X_val_fin, y_val_fin)
print(f"Early stopping triggered at epoch {stopped_epoch}")
final_model.eval()

with torch.no_grad():
    y_prob = torch.sigmoid(final_model(torch.FloatTensor(X_test))).squeeze(-1).numpy()
y_pred = (y_prob >= best_threshold).astype(int)

print(f"\nDecision threshold : {best_threshold:.3f}")
print(classification_report(y_test, y_pred, target_names=["healthy", "stressed"]))
test_roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC (test): {test_roc_auc:.4f}")


# ── 7. Save artefacts ─────────────────────────────────────────────────────────

joblib.dump(final_model,    os.path.join(MODEL_DIR, "hive_model.pkl"))
joblib.dump(FEATURE_COLS,   os.path.join(MODEL_DIR, "feature_cols.pkl"))
joblib.dump(best_threshold, os.path.join(MODEL_DIR, "threshold.pkl"))

export_model = _HiveNetWithSigmoid(final_model)
export_model.eval()
dummy_input = torch.randn(1, N_FEATURES)
torch.onnx.export(
    export_model,
    dummy_input,
    os.path.join(MODEL_DIR, "hive_model.onnx"),
    input_names=["features"],
    output_names=["probability"],
    opset_version=13,
    dynamic_axes={
        "features":    {0: "batch_size"},
        "probability": {0: "batch_size"},
    },
)

report = {
    "model": (
        f"HiveNetHeavy MLP  {N_FEATURES} → 128 → 64 → 32 → 16 → 1  "
        f"BatchNorm + Dropout={DROPOUT}  BCEWithLogitsLoss  ReduceLROnPlateau  early_stopping"
    ),
    "architecture": {
        "layers": [N_FEATURES, 128, 64, 32, 16, 1],
        "activation": "ReLU",
        "output": "Sigmoid",
        "dropout": DROPOUT,
        "batchnorm": True,
    },
    "training": {
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)",
        "loss": f"BCEWithLogitsLoss(pos_weight={POS_WEIGHT.item():.4f})",
        "class_balance": "SMOTE(k_neighbors=3)",
        "final_model_stopped_epoch": stopped_epoch,
        "decision_threshold": best_threshold,
    },
    "dataset": {
        "sources": ["urban", "mspb"],
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "n_features": int(N_FEATURES),
        "n_hives_train": int(df_model.loc[train_mask, "hive_id"].nunique()),
        "n_hives_test":  int(df_model.loc[test_mask,  "hive_id"].nunique()),
        "label_criteria": (
            "UrBAN: QNS or fob<12 | MSPB: hub-level varroa>2 or winter death"
        ),
        "normalisation": "StandardScaler only (no per-hive z-score)",
    },
    "feature_cols": FEATURE_COLS,
    "cv_results_on_train": cv_results,
    "test_results": {
        "roc_auc": float(test_roc_auc),
        "decision_threshold": best_threshold,
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["healthy", "stressed"],
            output_dict=True,
        ),
    },
}
with open(os.path.join(MODEL_DIR, "report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(
    f"\nSaved to models_heavy_v2/:  hive_model.pkl  hive_model.onnx  scaler.pkl  "
    f"feature_cols.pkl  threshold.pkl  report.json"
)
print("Done.")
