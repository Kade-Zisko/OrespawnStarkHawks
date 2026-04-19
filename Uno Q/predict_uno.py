"""
Uno Q — Inference helper library (v2).

Uses the Edge Impulse-compatible ONNX (hive_model_ei.onnx) from
models_heavy_v2. The model accepts raw un-normalised features and
handles StandardScaler internally — no preprocessing required.

Consecutive smoothing: STRESSED is only reported after
CONSECUTIVE_NEEDED back-to-back raw stressed readings, reducing
false alarms from transient spikes.
"""

import csv
import os
from collections import deque
from datetime import datetime

import numpy as np
import onnxruntime as ort

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = BASE
HISTORY_CSV = os.path.join(BASE, "history.csv")

# ── feature definitions (must match training pipeline exactly) ────────────────
WINDOWS = {"6h": 6, "12h": 12, "24h": 24, "48h": 48, "72h": 72}

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
FEATURE_COLS  = SENSOR_COLS + TEMPORAL_COLS   # 41 total

HISTORY_LEN        = WINDOWS["72h"] + 10   # readings to keep in memory
CONSECUTIVE_NEEDED = 3                      # consecutive stressed reads to trigger alert
DECISION_THRESHOLD = 0.9363


# ── model loading ─────────────────────────────────────────────────────────────

def load_artifacts() -> tuple[ort.InferenceSession, str]:
    """Load the EI ONNX model.

    Returns (sess, output_name). output_name is read from the model so
    this works with any Edge Impulse-exported ONNX regardless of how
    the output tensor is named.
    """
    sess        = ort.InferenceSession(os.path.join(MODEL_DIR, "hive_model_ei.onnx"))
    output_name = sess.get_outputs()[0].name
    return sess, output_name


# ── feature engineering ───────────────────────────────────────────────────────

def build_features(history: list, now: datetime) -> dict | None:
    """Build the 41-feature dict from a list of (temp, hum) readings.
    Returns None if fewer than 7 readings (can't compute 6h trend)."""
    if len(history) < WINDOWS["6h"] + 1:
        return None

    temps = np.array([r[0] for r in history], dtype=np.float64)
    hums  = np.array([r[1] for r in history], dtype=np.float64)
    t, h  = temps[-1], hums[-1]

    feat = {"temperature": float(t), "humidity": float(h)}

    for tag, w in [("24h", WINDOWS["24h"]), ("12h", WINDOWS["12h"]), ("72h", WINDOWS["72h"])]:
        ts = temps[-w:] if len(temps) >= w else temps
        hs = hums[-w:]  if len(hums)  >= w else hums
        feat[f"temp_mean_{tag}"]  = float(ts.mean())
        feat[f"temp_std_{tag}"]   = float(ts.std(ddof=0))
        feat[f"temp_min_{tag}"]   = float(ts.min())
        feat[f"temp_max_{tag}"]   = float(ts.max())
        feat[f"temp_range_{tag}"] = float(ts.max() - ts.min())
        feat[f"hum_mean_{tag}"]   = float(hs.mean())
        feat[f"hum_std_{tag}"]    = float(hs.std(ddof=0))
        feat[f"hum_min_{tag}"]    = float(hs.min())
        feat[f"hum_max_{tag}"]    = float(hs.max())
        feat[f"hum_range_{tag}"]  = float(hs.max() - hs.min())

    def trend(arr, w):
        return float(arr[-1] - arr[-w]) if len(arr) >= w else 0.0

    feat["temp_trend_6h"]  = trend(temps, WINDOWS["6h"])
    feat["hum_trend_6h"]   = trend(hums,  WINDOWS["6h"])
    feat["temp_trend_48h"] = trend(temps, WINDOWS["48h"])
    feat["hum_trend_48h"]  = trend(hums,  WINDOWS["48h"])
    feat["temp_hum_ratio"] = float(t / h) if h != 0 else 0.0

    feat["hour_sin"] = float(np.sin(2 * np.pi * now.hour / 24))
    feat["hour_cos"] = float(np.cos(2 * np.pi * now.hour / 24))
    doy = now.timetuple().tm_yday
    feat["doy_sin"]  = float(np.sin(2 * np.pi * doy / 365))
    feat["doy_cos"]  = float(np.cos(2 * np.pi * doy / 365))

    return feat


# ── inference ─────────────────────────────────────────────────────────────────

def infer(
    sess: ort.InferenceSession,
    output_name: str,
    feat: dict,
    consecutive_buf: deque,
) -> tuple[float, str]:
    """Run inference on raw features using the EI ONNX model.

    The model handles StandardScaler internally; no preprocessing needed.
    consecutive_buf (deque maxlen=CONSECUTIVE_NEEDED) tracks recent raw
    binary predictions — STRESSED is only returned once the buffer is
    full and all entries are stressed.
    """
    x    = np.array([[feat[c] for c in FEATURE_COLS]], dtype=np.float32)
    prob = float(sess.run([output_name], {"features": x})[0][0, 0])

    consecutive_buf.append(1 if prob >= DECISION_THRESHOLD else 0)
    label = (
        "STRESSED"
        if len(consecutive_buf) == CONSECUTIVE_NEEDED and all(consecutive_buf)
        else "HEALTHY"
    )
    return prob, label


# ── history persistence ───────────────────────────────────────────────────────

def load_history() -> deque:
    buf = deque(maxlen=HISTORY_LEN)
    if os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, newline="") as f:
            for row in csv.DictReader(f):
                buf.append((float(row["temperature"]), float(row["humidity"])))
    return buf


def save_history(history: deque):
    with open(HISTORY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["temperature", "humidity"])
        w.writeheader()
        for temp, hum in history:
            w.writerow({"temperature": temp, "humidity": hum})
