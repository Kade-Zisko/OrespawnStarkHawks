#!/usr/bin/env python3
"""
build_edge_impulse_onnx_v2.py  —  run once at development time.

Builds a single-input Edge-Impulse-compatible ONNX model from the
models_heavy_v2 artifacts (StandardScaler only — no per-hive z-score).

Bakes scaler mean and scale into the graph as constants so that the
model accepts raw un-normalised features directly.

Usage:
  python build_edge_impulse_onnx_v2.py
  python build_edge_impulse_onnx_v2.py --out /path/to/output.onnx

Output: models_heavy_v2/hive_model_ei.onnx
-------
  Input:
    features  float32 [1, 41]  — raw (un-normalised) 41-feature vector
  Output:
    probability  float32 [1, 1]  — sigmoid stress probability
"""

import argparse
import os

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

BASE         = os.path.dirname(os.path.abspath(__file__))
MODELS_HEAVY_V2 = os.path.join(os.path.dirname(BASE), "ML", "models_heavy_v2")


def build(out_path: str) -> None:
    print("Loading source artifacts from models_heavy_v2...")
    scaler_path = os.path.join(MODELS_HEAVY_V2, "scaler.pkl")
    model_onnx_path = os.path.join(MODELS_HEAVY_V2, "hive_model.onnx")
    print(f"  scaler : {scaler_path}")
    print(f"  onnx   : {model_onnx_path}")

    scaler     = joblib.load(scaler_path)
    base_model = onnx.load(model_onnx_path)
    onnx.checker.check_model(base_model)

    scaler_mean  = scaler.mean_.astype(np.float32)
    scaler_scale = scaler.scale_.astype(np.float32)
    opset        = base_model.opset_import[0].version
    orig_input   = base_model.graph.input[0].name
    orig_output  = base_model.graph.output[0].name
    print(f"  NN opset={opset}  input={orig_input!r}  output={orig_output!r}")
    print(f"  scaler mean  range : [{scaler_mean.min():.4f}, {scaler_mean.max():.4f}]")
    print(f"  scaler scale range : [{scaler_scale.min():.4f}, {scaler_scale.max():.4f}]")

    cm_2d = scaler_mean[np.newaxis, :].astype(np.float32)    # [1, 41]
    cs_2d = scaler_scale[np.newaxis, :].astype(np.float32)   # [1, 41]

    # ── Assemble graph ────────────────────────────────────────────────────────
    inits = list(base_model.graph.initializer) + [
        numpy_helper.from_array(cm_2d, name="__norm_mean__"),
        numpy_helper.from_array(cs_2d, name="__norm_scale__"),
    ]

    NORM_OUT = "__norm_features__"
    nodes = [
        helper.make_node("Sub", ["features", "__norm_mean__"],      ["__centered__"]),
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

    graph = helper.make_graph(
        nodes,
        "HiveNetHeavyV2EI",
        inputs=[
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 41]),
        ],
        outputs=[
            helper.make_tensor_value_info(orig_output, TensorProto.FLOAT, [1, 1]),
        ],
        initializer=inits,
        value_info=list(base_model.graph.value_info),
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    model.doc_string = (
        "HiveNetHeavy v2 41→128→64→32→16→1  StandardScaler only  "
        "single-input Edge-Impulse-compatible ONNX."
    )

    onnx.checker.check_model(model)

    # ── Validate: compare Python pipeline vs assembled ONNX on 10 rows ───────
    print("\n── Validation (Python vs ONNX, 10 synthetic rows) ───────────────────────")
    rng     = np.random.default_rng(0)
    X_raw   = rng.normal(
        loc=scaler_mean, scale=np.abs(scaler_scale), size=(10, len(scaler_mean))
    ).astype(np.float32)

    # Python path: scaler → base ONNX (expects scaled input, outputs probability)
    X_scaled  = scaler.transform(X_raw).astype(np.float32)
    base_sess = ort.InferenceSession(model_onnx_path)
    py_probs  = np.array([
        float(base_sess.run([orig_output], {orig_input: X_scaled[i:i+1]})[0][0, 0])
        for i in range(10)
    ])

    # ONNX path: assembled model (in-memory, raw features as input)
    sess = ort.InferenceSession(model.SerializeToString())
    print(f"  {'row':>3}  {'python':>10}  {'onnx':>10}  {'|diff|':>10}")
    max_diff = 0.0
    for i in range(10):
        onnx_prob = float(sess.run([orig_output], {"features": X_raw[i:i+1]})[0][0, 0])
        diff      = abs(float(py_probs[i]) - onnx_prob)
        max_diff  = max(max_diff, diff)
        print(f"  {i:>3}  {py_probs[i]:>10.6f}  {onnx_prob:>10.6f}  {diff:>10.2e}")

    if max_diff > 1e-4:
        print(f"\n  WARNING: max diff = {max_diff:.2e} — normalization mismatch detected!")
    else:
        print(f"\n  OK  max diff = {max_diff:.2e}")

    onnx.save(model, out_path)
    print(f"\nSaved  {out_path}")
    print(f"Size   {os.path.getsize(out_path):,} bytes")
    print(f"Input  features float32 [1, 41]")
    print(f"Output {orig_output} float32 [1, 1]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a single-input Edge-Impulse-compatible ONNX model from models_heavy_v2."
    )
    parser.add_argument(
        "--out", default=None,
        help="Output file path. Defaults to models_heavy_v2/hive_model_ei.onnx.",
    )
    args = parser.parse_args()

    out_path = args.out or os.path.join(MODELS_HEAVY_V2, "hive_model_ei.onnx")
    build(out_path)


if __name__ == "__main__":
    main()
