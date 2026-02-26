"""Generate a tiny ONNX model for testing the OnnxTracker pipeline.

Generated binary SHA256: 162ec2516ca9517202fde1feb33eb8ec0535d28d086cc12d050ecfe2629586cb
Regenerate if this script changes: python tests/fixtures/onnx/generate_simcc_3pt.py

Produces a constant-output SimCC model with 4 keypoints (nose, left_eye, right_eye, low_conf).
The model ignores its input and always outputs the same logit peaks, giving
deterministic, assertable coordinates when processed through OnnxTracker.

Coordinate design (for a 640x480 frame with 48x64 model input):
    letterbox scale = 0.075, pad_x = 0, pad_y = 14

    Keypoint     simcc_x_idx  simcc_y_idx  peak_logit  decoded_xy     frame_xy
    nose         48           64           5.0         (24.0, 32.0)   (320.0, 240.0)
    left_eye     30           52           5.0         (15.0, 26.0)   (200.0, 160.0)
    right_eye    66           52           5.0         (33.0, 26.0)   (440.0, 160.0)
    low_conf     10           20           0.1         (5.0, 10.0)    (66.7, 120.0)

low_conf has a peak logit of 0.1, producing confidence ~0.1, below the 0.3 threshold.
This keypoint tests the confidence filtering logic.

Usage:
    pip install onnx  # one-time, not needed for running tests
    python tests/fixtures/onnx/generate_simcc_3pt.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

NUM_KEYPOINTS = 4
INPUT_W, INPUT_H = 48, 64
SIMCC_X_LEN = INPUT_W * 2  # 96 (simcc_split_ratio = 2.0)
SIMCC_Y_LEN = INPUT_H * 2  # 128

# Peak positions in SimCC index space
PEAKS_X = [48, 30, 66, 10]  # nose, left_eye, right_eye, low_conf
PEAKS_Y = [64, 52, 52, 20]
PEAK_LOGITS = [5.0, 5.0, 5.0, 0.1]  # low_conf has sub-threshold logit


def build_model() -> onnx.ModelProto:
    # --- Graph I/O ---
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, INPUT_H, INPUT_W])
    output_x_info = helper.make_tensor_value_info("simcc_x", TensorProto.FLOAT, [1, NUM_KEYPOINTS, SIMCC_X_LEN])
    output_y_info = helper.make_tensor_value_info("simcc_y", TensorProto.FLOAT, [1, NUM_KEYPOINTS, SIMCC_Y_LEN])

    # --- Constant output data ---
    simcc_x = np.zeros((1, NUM_KEYPOINTS, SIMCC_X_LEN), dtype=np.float32)
    simcc_y = np.zeros((1, NUM_KEYPOINTS, SIMCC_Y_LEN), dtype=np.float32)
    for k, (px, py, logit) in enumerate(zip(PEAKS_X, PEAKS_Y, PEAK_LOGITS)):
        simcc_x[0, k, px] = logit
        simcc_y[0, k, py] = logit

    # --- Nodes ---
    # Identity consumes the input (keeps graph valid)
    identity = helper.make_node("Identity", ["input"], ["_input_consumed"])

    const_x = helper.make_node(
        "Constant",
        [],
        ["simcc_x"],
        value=helper.make_tensor("cx", TensorProto.FLOAT, simcc_x.shape, simcc_x.flatten().tolist()),
    )
    const_y = helper.make_node(
        "Constant",
        [],
        ["simcc_y"],
        value=helper.make_tensor("cy", TensorProto.FLOAT, simcc_y.shape, simcc_y.flatten().tolist()),
    )

    graph = helper.make_graph(
        [identity, const_x, const_y],
        "test_simcc_3pt",
        [input_info],
        [output_x_info, output_y_info],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out_path = Path(__file__).parent / "simcc_3pt.onnx"
    model = build_model()
    onnx.save(model, str(out_path))
    print(f"Saved {out_path} ({out_path.stat().st_size} bytes)")
