import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from load_datasets_masked import load_all

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models_masked")
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_24H    = 24
WINDOW_6H     =  6
BASELINE_DAYS = 21

# Original 14 features + 6 new Gas features
FEATURE_COLS = [
    "temperature", "humidity",
    "temp_mean_24h", "temp_std_24h", "temp_min_24h", "temp_max_24h", "temp_range_24h",
    "hum_mean_24h",  "hum_std_24h",  "hum_min_24h",  "hum_max_24h",  "hum_range_24h",
    "temp_trend_6h", "hum_trend_6h",
    # Gas features (MQ)
    "co2", 
    "co2_mean_24h", "co2_std_24h", "co2_min_24h", "co2_max_24h", "co2_trend_6h"
]

N_FEATURES = 20
DROPOUT    = 0.3
LR         = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 100
PATIENCE   = 7

class MaskedHiveNet(nn.Module):
    """
    Input(20) -> Masking -> Linear(32) -> ReLU -> Dropout -> Linear(16) -> ReLU -> Dropout -> Linear(1) -> Sigmoid
    """
    def __init__(self, n_features: int = 20, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Apply binary mask. Masked inputs will be 0.0, zeroing out their gradient contribution.
        x_masked = x * mask
        h1 = self.relu1(self.fc1(x_masked))
        h1 = self.drop1(h1)
        h2 = self.relu2(self.fc2(h1))
        h2 = self.drop2(h2)
        y = self.sigmoid(self.out(h2))
        return y


def make_loader(X: np.ndarray, M: np.ndarray, y: np.ndarray) -> DataLoader:
    # Resample features, masks, and labels using SMOTE
    # Since SMOTE expects 2D array, we can concatenate X and M, then split them after resampling
    smote = SMOTE(random_state=42)
    XM = np.hstack([X, M])
    XM_res, y_res = smote.fit_resample(XM, y)
    
    X_res = XM_res[:, :N_FEATURES]
    M_res = XM_res[:, N_FEATURES:]
    
    # SMOTE interpolation might create non-binary masks (e.g. 0.5), we threshold them back to binary
    M_res = (M_res > 0.5).astype(np.float32)

    dataset = TensorDataset(torch.FloatTensor(X_res), torch.FloatTensor(M_res), torch.FloatTensor(y_res))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train_model(X_tr: np.ndarray, M_tr: np.ndarray, y_tr: np.ndarray,
                X_val: np.ndarray, M_val: np.ndarray, y_val: np.ndarray) -> tuple[MaskedHiveNet, int]:
    model     = MaskedHiveNet(n_features=N_FEATURES, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    loader    = make_loader(X_tr, M_tr, y_tr)

    X_val_t = torch.FloatTensor(X_val)
    M_val_t = torch.FloatTensor(M_val)
    y_val_t = torch.FloatTensor(y_val)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, mb, yb in loader:
            pred = model(xb, mb).squeeze(-1)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t, M_val_t).squeeze(-1), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, epoch - PATIENCE


def eval_model(model: MaskedHiveNet, X: np.ndarray, M: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(X), torch.FloatTensor(M)).squeeze(-1).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy":    float(accuracy_score(y, preds)),
        "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "roc_auc":     float(roc_auc_score(y, probs)),
    }


def main():
    # ── 1. Load merged dataset ────────────────────────────────────────────────────
    df_raw = load_all()
    
    # [!] Placeholder for merging Kaggle CO2 data when available
    # For now, we will add dummy CO2 columns to df_raw so the pipeline works
    if "co2" not in df_raw.columns:
        print("Note: 'co2' feature missing. Initializing with NaN masks.")
        df_raw["co2"] = np.nan

    # ── 2. Rolling feature engineering ───────────────────────────────────────────
    print("\nEngineering rolling features (hourly windows)...")
    feature_frames = []
    for hive_id, hdf in df_raw.groupby("hive_id"):
        hdf = hdf.sort_values("date").reset_index(drop=True)
        t, h, c = hdf["temperature"], hdf["humidity"], hdf["co2"]

        rt = t.rolling(WINDOW_24H, min_periods=WINDOW_24H // 2)
        rh = h.rolling(WINDOW_24H, min_periods=WINDOW_24H // 2)
        rc = c.rolling(WINDOW_24H, min_periods=WINDOW_24H // 2)

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
        
        # CO2 features
        hdf["co2_mean_24h"]   = rc.mean()
        hdf["co2_std_24h"]    = rc.std()
        hdf["co2_min_24h"]    = rc.min()
        hdf["co2_max_24h"]    = rc.max()
        hdf["co2_trend_6h"]   = c.diff(WINDOW_6H)

        feature_frames.append(hdf)

    df_feats = pd.concat(feature_frames, ignore_index=True)
    print(f"Feature matrix shape: {df_feats.shape}")

    # ── 3. Mask Generation & Z-score normalisation ───────────────────────────
    print(f"Applying per-hive z-score normalisation (baseline = first {BASELINE_DAYS} days)...")
    hive_stats: dict = {}
    df_feats = df_feats.copy()
    
    # Generate Mask
    mask_cols = [f"{c}_mask" for c in FEATURE_COLS]
    for c in FEATURE_COLS:
        # If the feature is present and not NaN, mask is 1, else 0
        df_feats[f"{c}_mask"] = df_feats[c].notna().astype(float)
        # Fill missing values with 0.0
        df_feats[c] = df_feats[c].fillna(0.0)

    for hive_id, hdf in df_feats.groupby("hive_id"):
        cutoff   = hdf["date"].min() + pd.Timedelta(days=BASELINE_DAYS)
        baseline = hdf[hdf["date"] <= cutoff]
        stats: dict = {}
        for col in FEATURE_COLS:
            # Only calculate stats if there are non-zero mask values in the baseline
            has_data = hdf[f"{col}_mask"].sum() > 0
            if has_data:
                mu  = float(baseline[col].mean()) if not baseline[col].isna().all() else 0.0
                std = float(baseline[col].std()) if not baseline[col].isna().all() else 1.0
                if np.isnan(std):
                    std = 1.0
                stats[col] = {"mean": mu, "std": max(std, 1e-6)}
                # Apply normalization where mask == 1
                mask_idx = hdf[f"{col}_mask"] == 1
                if mask_idx.any():
                    df_feats.loc[hdf.index[mask_idx], col] = (hdf.loc[mask_idx, col] - mu) / stats[col]["std"]
            else:
                stats[col] = {"mean": 0.0, "std": 1.0}
                
        hive_stats[hive_id] = stats

    with open(os.path.join(MODEL_DIR, "hive_stats.json"), "w") as f:
        json.dump(hive_stats, f, indent=2)
    print(f"Normalised {len(hive_stats)} hives.")

    # Drop missing labels
    df_model = df_feats.dropna(subset=["label", "hive_id", "source"])

    # ── 4. Hive-based train / test split ─────────────────────────────────────────
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
    M_train     = df_model.loc[train_mask, mask_cols].values
    y_train     = df_model.loc[train_mask, "label"].values.astype(int)
    
    X_test_raw  = df_model.loc[test_mask,  FEATURE_COLS].values
    M_test      = df_model.loc[test_mask,  mask_cols].values
    y_test      = df_model.loc[test_mask,  "label"].values.astype(int)

    print(f"\nTrain: {len(X_train_raw):,} samples hives={df_model.loc[train_mask,'hive_id'].nunique()}")
    print(f"Test : {len(X_test_raw):,} samples hives={df_model.loc[test_mask,'hive_id'].nunique()}")

    # ── 5. StandardScaler ─────────────────────────────────────────────────────────
    print("\nFitting StandardScaler on training data...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
    
    # Force masked features back to 0.0
    X_train = X_train * M_train
    X_test = X_test * M_test
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print("Saved scaler.pkl")

    # ── 6. 5-fold CV ──────────────────────────────────────────────────────────────
    print("\n── 5-fold CV on training hives ──────────────────────────────────────────")
    cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "f1_weighted": [], "roc_auc": []}

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        fold_model, stopped_epoch = train_model(
            X_train[tr_idx], M_train[tr_idx], y_train[tr_idx],
            X_train[val_idx], M_train[val_idx], y_train[val_idx],
        )
        m = eval_model(fold_model, X_train[val_idx], M_train[val_idx], y_train[val_idx])
        fold_metrics["accuracy"].append(m["accuracy"])
        fold_metrics["f1_weighted"].append(m["f1_weighted"])
        fold_metrics["roc_auc"].append(m["roc_auc"])
        print(f"  Fold {fold}  stopped@{stopped_epoch:>3}  Acc={m['accuracy']:.4f}  "
              f"F1={m['f1_weighted']:.4f}  AUC={m['roc_auc']:.4f}")

    cv_results = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
    print(f"\n  Mean  Acc={cv_results['accuracy']:.4f}  AUC={cv_results['roc_auc']:.4f}")

    # ── 7. Train final model ───────────────────────────────────────────────────────
    print("\n── Training final model ─────────────────────────────────────────────────")
    val_size  = int(len(X_train) * 0.10)
    X_tr_fin  = X_train[val_size:]
    M_tr_fin  = M_train[val_size:]
    y_tr_fin  = y_train[val_size:]
    
    X_val_fin = X_train[:val_size]
    M_val_fin = M_train[:val_size]
    y_val_fin = y_train[:val_size]

    final_model, stopped_epoch = train_model(X_tr_fin, M_tr_fin, y_tr_fin, X_val_fin, M_val_fin, y_val_fin)
    print(f"Early stopping triggered at epoch {stopped_epoch}")
    final_model.eval()

    with torch.no_grad():
        y_prob = final_model(torch.FloatTensor(X_test), torch.FloatTensor(M_test)).squeeze(-1).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    print(classification_report(y_test, y_pred, target_names=["healthy", "stressed"]))
    test_roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC (test): {test_roc_auc:.4f}")

    # ── 8. Save artefacts ─────────────────────────────────────────────────────────
    joblib.dump(final_model,  os.path.join(MODEL_DIR, "hive_model.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    dummy_x = torch.randn(1, N_FEATURES)
    dummy_m = torch.ones(1, N_FEATURES)
    
    torch.onnx.export(
        final_model,
        (dummy_x, dummy_m),
        os.path.join(MODEL_DIR, "hive_model.onnx"),
        input_names=["features", "mask"],
        output_names=["stress_prob"],
        opset_version=13,
        dynamic_axes={
            "features":    {0: "batch_size"},
            "mask":        {0: "batch_size"},
            "stress_prob": {0: "batch_size"},
        },
    )

    report = {
        "model": "MaskedHiveNet MLP 20 -> 32 -> 16 -> 1",
        "feature_cols": FEATURE_COLS,
        "dataset": {
            "sources": ["urban", "mspb", "kaggle"],
            "n_train": int(len(X_train)),
            "n_test":  int(len(X_test)),
            "n_features": int(N_FEATURES),
        }
    }
    with open(os.path.join(MODEL_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
