#!/usr/bin/env python3
"""
build_edge_impulse_onnx.py  —  run once per hive at development time.

Edge Impulse only accepts models with a single input tensor.  This script
bakes the per-hive normalisation directly into the ONNX graph as constants,
eliminating the hive_idx input that build_combined_onnx.py requires.

Usage:
  python build_edge_impulse_onnx.py --hive urban_01
  python build_edge_impulse_onnx.py          # uses unknown-hive (StandardScaler only)
  python build_edge_impulse_onnx.py --list   # list all known hive keys

Output: hive_model_ei_<hive>.onnx
-------
  Input:
    features  float32 [1, 41]  — raw (un-normalised) 41-feature vector
  Outputs:
    probability  float32 [1, 1]  — sigmoid stress probability
    label        int64   [1, 1]  — 1 = STRESSED, 0 = HEALTHY
"""

import argparse
import json
import os

import joblib
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BASE         = os.path.dirname(os.path.abspath(__file__))
MODELS_HEAVY = os.path.join(os.path.dirname(BASE), "ML", "models_heavy")

SENSOR_COLS = [
    "temperature", "humidity",
    "temp_mean_24h", "temp_std_24h", "temp_min_24h", "temp_max_24h", "temp_range_24h",
    "hum_mean_24h",  "hum_std_24h",  "hum_min_24h",  "hum_max_24h",  "hum_range_24h",
    "temp_mean_12h", "temp_std_12h", "temp_min_12h", "temp_max_12h", "temp_range_12h",
    "hum_mean_12h",  "hum_std_12h",  "hum_min_12h",  "hum_max_12h",  "hum_range_12h",
    "temp_mean_72h", "temp_std_72h", "temp_min_72h", "temp_max_72h", "temp_range_72h",
    "hum_mean_72h",  "hum_std_72h",  "hum_min_72h",  "hum_max_72h",  "hum_range_72h",
    "temp_trend_6h", "hum_trend_6h", "temp_trend_48h", "hum_trend_48h",
    "temp_hum_ratio",
]
TEMPORAL_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURE_COLS  = SENSOR_COLS + TEMPORAL_COLS
col_to_idx    = {c: i for i, c in enumerate(FEATURE_COLS)}


def compute_norm_constants(
    hive_key: str | None,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    hive_stats: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cm, cs) shape [1, 41]: the fused affine normalisation constants.

    For z-scored sensor columns: cm = mu + scaler_mean*sigma, cs = sigma*scaler_scale
    For all other columns:       cm = scaler_mean,             cs = scaler_scale
    """
    cm = scaler_mean.copy()
    cs = scaler_scale.copy()
    stats = hive_stats.get(hive_key, {}) if hive_key else {}
    for col, s in stats.items():
        j = col_to_idx.get(col)
        if j is None:
            continue
        mu, sigma = float(s["mean"]), float(s["std"])
        if sigma <= 0:
            continue
        cm[j] = mu + scaler_mean[j] * sigma
        cs[j] = sigma * scaler_scale[j]
    return cm[np.newaxis, :].astype(np.float32), cs[np.newaxis, :].astype(np.float32)


def build(hive_key: str | None, out_path: str) -> None:
    print("Loading source artifacts...")
    scaler    = joblib.load(os.path.join(MODELS_HEAVY, "scaler.pkl"))
    threshold = float(joblib.load(os.path.join(MODELS_HEAVY, "threshold.pkl")))
    with open(os.path.join(MODELS_HEAVY, "hive_stats.json")) as f:
        hive_stats = json.load(f)
    base_model = onnx.load(os.path.join(MODELS_HEAVY, "hive_model.onnx"))
    onnx.checker.check_model(base_model)

    scaler_mean  = scaler.mean_.astype(np.float32)
    scaler_scale = scaler.scale_.astype(np.float32)
    opset        = base_model.opset_import[0].version
    orig_input   = base_model.graph.input[0].name
    orig_output  = base_model.graph.output[0].name
    print(f"  NN opset={opset}  input={orig_input!r}  output={orig_output!r}")

    cm_2d, cs_2d = compute_norm_constants(hive_key, scaler_mean, scaler_scale, hive_stats)
    thr_arr = np.array([[threshold]], dtype=np.float32)
    print(f"  Hive: {hive_key or '_unknown_'!r}  threshold={threshold:.4f}")

    # ── Assemble graph ────────────────────────────────────────────────────────
    inits = list(base_model.graph.initializer) + [
        numpy_helper.from_array(cm_2d,  name="__norm_mean__"),
        numpy_helper.from_array(cs_2d,  name="__norm_scale__"),
        numpy_helper.from_array(thr_arr, name="__threshold__"),
    ]

    NORM_OUT = "__norm_features__"
    nodes = [
        helper.make_node("Sub", ["features", "__norm_mean__"],  ["__centered__"]),
        helper.make_node("Div", ["__centered__", "__norm_scale__"], [NORM_OUT]),
    ]

    for n in base_model.graph.node:
        new_inputs = [NORM_OUT if inp == orig_input else inp for inp in n.input]
        if new_inputs != list(n.input):
            patched = helper.make_node(n.op_type, new_inputs, list(n.output), name=n.name)
            patched.attribute.extend(n.attribute)
            nodes.append(patched)
        else:
            nodes.append(n)

    nodes.append(helper.make_node("Identity", [orig_output],    ["probability"]))
    nodes.append(helper.make_node("Greater",  ["probability", "__threshold__"], ["__gt__"]))
    nodes.append(helper.make_node("Cast",     ["__gt__"],       ["__label_i64__"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Cast",     ["__label_i64__"], ["label"],        to=TensorProto.INT8))

    graph = helper.make_graph(
        nodes,
        "HiveNetHeavyEI",
        inputs=[
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 41]),
        ],
        outputs=[
            helper.make_tensor_value_info("probability", TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info("label",       TensorProto.INT8,  [1, 1]),
        ],
        initializer=inits,
        value_info=list(base_model.graph.value_info),
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8  # max IR version broadly supported by onnxruntime deployments
    model.doc_string = (
        f"HiveNetHeavy 41→128→64→32→16→1  hive={hive_key or '_unknown_'}  "
        "single-input Edge-Impulse-compatible ONNX."
    )

    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"\nSaved  {out_path}")
    print(f"Size   {os.path.getsize(out_path):,} bytes")
    print(f"Input  features float32 [1, 41]")
    print(f"Output probability float32 [1, 1],  label int8 [1, 1]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a single-input Edge-Impulse-compatible ONNX model."
    )
    parser.add_argument(
        "--hive", default=None,
        help="Hive key from hive_stats.json (e.g. 'urban_01'). "
             "Omit to use unknown-hive normalization (StandardScaler only).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output file path. Defaults to hive_model_ei_<hive>.onnx in the same directory.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all known hive keys and exit.",
    )
    args = parser.parse_args()

    if args.list:
        with open(os.path.join(MODELS_HEAVY, "hive_stats.json")) as f:
            hive_stats = json.load(f)
        print("Known hive keys:")
        for k in sorted(hive_stats.keys()):
            print(f"  {k}")
        return

    suffix   = args.hive.replace("/", "_").replace(" ", "_") if args.hive else "unknown"
    out_path = args.out or os.path.join(MODELS_HEAVY, f"hive_model_ei_{suffix}.onnx")
    build(args.hive, out_path)


if __name__ == "__main__":
    main()
