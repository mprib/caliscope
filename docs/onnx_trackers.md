# Custom ONNX Trackers

Caliscope can load custom ONNX pose estimation models for 2D landmark tracking. You can use models trained on your specific subjects (particular species, body regions, behavioral features) without modifying Caliscope's source code. A TOML "model card" file describes the model's input requirements and output format.

After installation, ONNX models appear alongside the built-in trackers in the reconstruction tab's dropdown menu.

## Requirements

ONNX model inference is included in the standard Caliscope installation — no extra packages needed. On first launch, Caliscope seeds your models directory with model cards for the RTMPose Halpe26 family (tiny through xlarge). The ONNX weight files are not shipped with the package; they can be downloaded in-app or placed manually.

## Built-in Models

Caliscope ships model card templates for the RTMPose Halpe26 family. On first launch, these are copied to your platform's models directory (see Models Directory below). The model cards describe the model but do not include the weights — the `.onnx` files must be downloaded separately.

### In-App Download

For built-in models that include a `[source]` section in their model card, Caliscope can download the weights directly:

1. Open the Reconstruction tab
2. Select an ONNX tracker from the dropdown
3. If the weights are not yet present, a download button appears with the model's license information
4. Click to download — the weights are fetched from the upstream source and placed in the models directory

You can also download weights manually and place the `.onnx` file in the models directory. The filename must match the `model_path` field in the model card.

## Setup Steps

1. **Obtain or export an ONNX pose estimation model**
   - Many pose estimation frameworks support ONNX export (MMPose, SLEAP, DeepLabCut with model zoo)
   - Ensure the model outputs either SimCC vectors or heatmaps (see format details below)

2. **Locate your models directory**
   - Caliscope uses platform-specific data directories (see Models Directory below)
   - The directory is created automatically on first run
   - Place your `.onnx` model file anywhere accessible (does not need to be in the models directory)

3. **Create a model card TOML file**
   - Must be placed in the models directory
   - Name it descriptively (e.g., `rtmpose-halpe26.toml`)
   - See Model Card Reference below for format details

4. **Restart Caliscope**
   - At startup, Caliscope scans the models directory for `.toml` files
   - Valid model cards are registered and appear in the tracker dropdown

5. **Verify the model appears**
   - Open Caliscope and navigate to the Reconstruction tab
   - Your custom model should appear in the tracker selection dropdown

## Models Directory

Caliscope uses platform-specific data directories following standard conventions:

| Platform | Models Directory |
|----------|-----------------|
| **Linux** | `~/.local/share/caliscope/models/` |
| **macOS** | `~/Library/Application Support/caliscope/models/` |
| **Windows** | `C:\Users\<user>\AppData\Local\caliscope\caliscope\models\` |

Place your `.toml` model card files in this directory. The `.onnx` model file can live anywhere; the model card points to it via the `model_path` field (use an absolute path).

**To find your models directory:**

1. Launch Caliscope
2. Check the log file location shown in the terminal output
3. The models directory is in the same parent directory

Or from Python:

```python
from caliscope import MODELS_DIR
print(MODELS_DIR)
```

## Model Card Reference

A model card is a TOML file that describes your ONNX model's configuration. Here's a complete annotated example:

```toml
[model]
# Display name in the GUI (optional, defaults to the .onnx filename if omitted)
name = "RTMPose-t Halpe26"

# Absolute path to your ONNX model file (required)
model_path = "/home/user/models/rtmpose-t-halpe26.onnx"

# Output format: "simcc" or "heatmap" (required)
format = "simcc"

# Model input dimensions as [width, height] (required)
input_size = [192, 256]

# Minimum confidence to report a point (optional, default 0.3)
# Lower values include more detections but may increase false positives
confidence_threshold = 0.3

# Keypoint name-to-index mapping (required)
# Maps human-readable names to the model's output indices
[points]
nose = 0
left_eye = 1
right_eye = 2
left_ear = 3
right_ear = 4
left_shoulder = 5
right_shoulder = 6
left_elbow = 7
right_elbow = 8
left_wrist = 9
right_wrist = 10
left_hip = 11
right_hip = 12
left_knee = 13
right_knee = 14
left_ankle = 15
right_ankle = 16
head = 17
neck = 18
hip = 19
left_big_toe = 20
right_big_toe = 21
left_small_toe = 22
right_small_toe = 23
left_heel = 24
right_heel = 25

# Wireframe segments for 3D visualization (optional)
# Each segment connects two points with a colored line
[segments.shoulders]
color = "y"  # Matplotlib color string (single char or full name)
points = ["left_shoulder", "right_shoulder"]

[segments.left_arm]
color = "g"
points = ["left_shoulder", "left_elbow"]

[segments.left_forearm]
color = "g"
points = ["left_elbow", "left_wrist"]

[segments.right_arm]
color = "r"
points = ["right_shoulder", "right_elbow"]

[segments.right_forearm]
color = "r"
points = ["right_elbow", "right_wrist"]

[segments.pelvis]
color = "y"
points = ["left_hip", "right_hip"]

[segments.left_flank]
color = "y"
points = ["left_hip", "left_shoulder"]

[segments.right_flank]
color = "y"
points = ["right_hip", "right_shoulder"]

[segments.left_thigh]
color = "g"
points = ["left_hip", "left_knee"]

[segments.left_shank]
color = "g"
points = ["left_knee", "left_ankle"]

[segments.right_thigh]
color = "r"
points = ["right_hip", "right_knee"]

[segments.right_shank]
color = "r"
points = ["right_knee", "right_ankle"]

[segments.neck_segment]
color = "y"
points = ["neck", "head"]
```

### Required Fields

| Field | Description |
|-------|-------------|
| `model.model_path` | Absolute path to the `.onnx` file |
| `model.format` | Either `"simcc"` or `"heatmap"` (see format details below) |
| `model.input_size` | `[width, height]` as the model expects (not your video resolution) |
| `[points]` | Maps point name (string) to output index (integer) |

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `model.name` | ONNX filename stem | Display name in the GUI |
| `model.confidence_threshold` | `0.3` | Minimum confidence to report a point (0.0 to 1.0) |
| `[segments.*]` | None | Wireframe segment definitions for 3D visualization |
| `[source]` | None | Download metadata for in-app weight fetching (see below) |

### Wireframe Segments

Each segment definition requires:
- `color`: Matplotlib color string (e.g., `"r"`, `"blue"`, `"#FF5733"`)
- `points`: 2-element list of point names (must exist in `[points]` section)

Segments are used by the 3D visualizer to draw connections between keypoints, making it easier to interpret motion trajectories.

### Source Section (For In-App Download)

Model cards can include a `[source]` section that enables in-app downloading of weights. This is optional — custom models without a `[source]` section work normally but require manual placement of the `.onnx` file.

```toml
[source]
url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.zip"
extraction = "zip_end2end"
license = "Apache-2.0"
license_url = "https://github.com/open-mmlab/mmpose/blob/main/LICENSE"
file_size_mb = 13
sha256 = "de5fa6ef754e1b19a0f8199d53affef122813e30c580c48be87fcf86c4ec47a7"
```

| Field | Required | Description |
|-------|----------|-------------|
| `url` | Yes | Direct download URL for the model weights |
| `extraction` | Yes (when `url` present) | How to handle the download: `"zip_end2end"` (extract `.onnx` from zip) or `"direct"` (URL points directly to `.onnx`) |
| `license` | No | SPDX license identifier shown to user before download |
| `license_url` | No | Link to the full license text |
| `file_size_mb` | No | Approximate download size shown in the UI |
| `sha256` | No | SHA-256 hash for integrity verification after download |

## SimCC vs Heatmap Formats

ONNX pose estimation models output predictions in different formats. You must specify which format your model uses in the `model.format` field.

### SimCC (Simulated Coordinate Classification)

**Used by:** RTMPose family (MMPose/OpenMMLab)

**How it works:** The model outputs two 1D probability distributions per keypoint, one for the X coordinate and one for the Y coordinate. The coordinate is the argmax of each distribution. This provides sub-pixel accuracy (0.5px resolution) built into the architecture.

**When to use:** If your model was trained with RTMPose or uses the SimCC head architecture, use `format = "simcc"`.

**Output structure:** Two tensors of shape `(batch, num_keypoints, input_width * simcc_split_ratio)` and `(batch, num_keypoints, input_height * simcc_split_ratio)`. The default `simcc_split_ratio` is 2.0, so for a model with `input_size = [192, 256]`, the X vector length is 384 and the Y vector length is 512.

### Heatmap

**Used by:** SLEAP ONNX exports, many pose estimation frameworks

**How it works:** The model outputs a 2D "heat image" per keypoint. The coordinate is the location of the brightest pixel, refined with quadratic interpolation for sub-pixel accuracy.

**When to use:** If your model was trained with SLEAP, or uses a heatmap-based architecture (common in many pose estimation systems), use `format = "heatmap"`.

**Output structure:** One tensor of shape `(batch, num_keypoints, heatmap_height, heatmap_width)`.

If you're unsure which format your model uses, check the training framework's documentation or inspect the model's output tensors. SimCC models will have two outputs, heatmap models will have one.

## Built-in Model Cards

The following RTMPose Halpe26 models ship as built-in model cards. Weights are downloaded in-app on first use.

| Model | Format | Input Size | Keypoints | Download Size | Notes |
|-------|--------|------------|-----------|---------------|-------|
| RTMPose-t Halpe26 | SimCC | 192×256 | 26 | 13 MB | Fastest, good for real-time |
| RTMPose-s Halpe26 | SimCC | 192×256 | 26 | 21 MB | Balanced speed/accuracy |
| RTMPose-m Halpe26 | SimCC | 192×256 | 26 | 50 MB | Higher accuracy |
| RTMPose-l Halpe26 | SimCC | 192×256 | 26 | 100 MB | High accuracy |
| RTMPose-x Halpe26 | SimCC | 288×384 | 26 | 178 MB | Highest accuracy, largest input |

All models use the Halpe26 keypoint set (26 body landmarks including feet) and are licensed under Apache-2.0 by OpenMMLab.

Other RTMPose variants and SLEAP-exported models should work with appropriate model card configuration. If you successfully use a model not listed here, consider contributing the model card.

## Troubleshooting

### Model not appearing in dropdown

**Symptoms:** After creating a model card and restarting Caliscope, your custom tracker doesn't appear in the reconstruction tab's dropdown menu.

**Diagnosis:**
1. Verify the `.toml` file is in the correct models directory (see platform paths above)
2. Launch Caliscope from a terminal to see error messages
3. Check the log file for parsing errors

**Common causes:**
- TOML syntax error (missing quotes, incorrect nesting)
- Missing required fields (`model_path`, `format`, `input_size`, or `[points]` section)
- Invalid `format` value (must be exactly `"simcc"` or `"heatmap"`)

### Poor detection quality

**Symptoms:** Few keypoints detected, erratic tracking, or complete detection failure.

**Solutions:**
- **Lower the confidence threshold:** Try `confidence_threshold = 0.1` to include more marginal detections
- **Verify input size:** Ensure `input_size` exactly matches what the model was trained on (check model documentation)
- **Check preprocessing compatibility:** SimCC format expects ImageNet normalization; if your model uses different preprocessing, it may not work correctly

**Note:** Wrong input size will produce garbage output because the model receives distorted or incorrectly scaled input.

### Wrong keypoint positions

**Symptoms:** Keypoints appear in incorrect anatomical locations (e.g., left elbow predicted where right wrist should be).

**Diagnosis:** The `[points]` mapping in your model card doesn't match the model's actual output ordering.

**Solution:**
1. Consult your model's training framework documentation for the keypoint ordering
2. Verify the indices in your `[points]` section match that ordering exactly
3. Different model families use different conventions even for the same body landmarks (e.g., COCO vs Halpe vs OpenPose keypoint orderings)

### Model file path issues

**Symptoms:** Error about model file not found, even though the file exists.

**Solution:**
- Use absolute paths in `model_path`, not relative paths
- Verify the path is correct (no typos, correct extension)
- Ensure the file has read permissions

## Performance Characteristics

ONNX trackers in Caliscope use CPU inference through onnxruntime. Processing speed depends on:

- **Model size:** Smaller models (RTMPose-t) process faster than larger ones (RTMPose-m)
- **Input resolution:** Models with smaller input sizes process faster
- **CPU capabilities:** More cores and newer processors improve throughput

A three-tier crop-and-track strategy helps maintain robust detection across frames:

1. **Tier 1:** Crop to previous detection (fast, common case)
2. **Tier 2:** Full-frame letterbox (cold start or lost tracking)
3. **Tier 3:** Sliding window scan (thorough search when full-frame fails)

This ensures reliable detection even when subjects move rapidly or temporarily leave the crop region.

## Limitations

The ONNX tracker currently supports single-person detection with CPU inference only. GPU acceleration and multi-person tracking are not yet implemented. If you have a use case that requires these features, please [open an issue](https://github.com/mprib/caliscope/issues).
