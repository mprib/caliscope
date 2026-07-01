# Multi-Marker ArUco Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ArucoTarget` with `ArucoMarkerSet` (N markers of varying sizes) for large-volume extrinsic calibration.

**Architecture:** New frozen dataclasses `ArucoMarker` and `ArucoMarkerSet` replace the single-marker `ArucoTarget`. TOML format changes from `[corners.N]` arrays to `[[markers]]` table array. PnP bootstrap keys are widened to include `object_id`. Legacy stereoCalibrate path is deleted.

**Tech Stack:** Python 3.10+, OpenCV ArUco, rtoml, PySide6, frozen dataclasses

**Spec:** `specs/multi-marker-aruco.md`

## Global Constraints

- Branch: `feature/multi-marker-aruco` off `epic/aruco-calibration`
- No backward compatibility with old `aruco_target.toml` format (beta rules)
- `@dataclass(frozen=True)` without `slots=True` (cached_property needs `__dict__`)
- Dictionary capacity validated at construction time (fail-fast)
- All marker IDs must fit within the ArUco dictionary's capacity
- basedpyright clean on all changed files
- Full test suite green, no relaxed tolerances

---

### Task 1: ArucoMarker and ArucoMarkerSet dataclasses + TOML persistence

**Files:**
- Create: `src/caliscope/core/aruco_marker.py`
- Test: `tests/test_aruco_marker.py`

**Interfaces:**
- Produces:
  - `ArucoMarker(marker_id: int, size_m: float)` with `corners: NDArray[np.float64]` cached property
  - `ArucoMarkerSet(dictionary: int, markers: dict[int, ArucoMarker])` with `from_toml(path)`, `to_toml(path)`, `generate_marker_image(marker_id, pixel_size) -> NDArray`

- [ ] **Step 1: Write failing tests for ArucoMarker**

```python
# tests/test_aruco_marker.py
import numpy as np
import pytest
from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet


def test_aruco_marker_corners_from_size():
    marker = ArucoMarker(marker_id=0, size_m=0.10)
    corners = marker.corners
    assert corners.shape == (4, 3)
    s = 0.05  # half-size
    expected = np.array([[-s, +s, 0], [+s, +s, 0], [+s, -s, 0], [-s, -s, 0]])
    np.testing.assert_allclose(corners, expected)


def test_aruco_marker_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="positive"):
        ArucoMarker(marker_id=0, size_m=0.0)
    with pytest.raises(ValueError, match="positive"):
        ArucoMarker(marker_id=0, size_m=-0.05)


def test_aruco_marker_frozen():
    marker = ArucoMarker(marker_id=0, size_m=0.05)
    with pytest.raises(AttributeError):
        marker.size_m = 0.1  # type: ignore[misc]
```

- [ ] **Step 2: Write failing tests for ArucoMarkerSet**

```python
# append to tests/test_aruco_marker.py
import cv2


def test_marker_set_construction():
    markers = {0: ArucoMarker(0, 0.165), 3: ArucoMarker(3, 0.10)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_100, markers=markers)
    assert len(ms.markers) == 2
    assert ms.markers[0].size_m == 0.165
    assert ms.markers[3].size_m == 0.10


def test_marker_set_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers={})


def test_marker_set_rejects_overcapacity_id():
    with pytest.raises(ValueError, match="capacity"):
        ArucoMarkerSet(
            dictionary=cv2.aruco.DICT_4X4_50,
            markers={99: ArucoMarker(99, 0.05)},
        )


def test_marker_set_toml_round_trip(tmp_path):
    markers = {0: ArucoMarker(0, 0.165), 3: ArucoMarker(3, 0.10), 7: ArucoMarker(7, 0.165)}
    original = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_100, markers=markers)
    path = tmp_path / "aruco_marker_set.toml"
    original.to_toml(path)
    loaded = ArucoMarkerSet.from_toml(path)
    assert loaded.dictionary == original.dictionary
    assert len(loaded.markers) == 3
    for mid in [0, 3, 7]:
        np.testing.assert_allclose(loaded.markers[mid].corners, original.markers[mid].corners)


def test_marker_set_from_toml_missing_file(tmp_path):
    from caliscope.persistence import PersistenceError
    with pytest.raises(PersistenceError):
        ArucoMarkerSet.from_toml(tmp_path / "nope.toml")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_aruco_marker.py -v`
Expected: ImportError — `aruco_marker` module does not exist

- [ ] **Step 4: Implement ArucoMarker**

```python
# src/caliscope/core/aruco_marker.py
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import rtoml
from numpy.typing import NDArray


@dataclass(frozen=True)
class ArucoMarker:
    marker_id: int
    size_m: float

    def __post_init__(self) -> None:
        if self.size_m <= 0:
            raise ValueError(f"size_m must be positive, got {self.size_m}")

    @cached_property
    def corners(self) -> NDArray[np.float64]:
        s = self.size_m / 2
        return np.array(
            [[-s, +s, 0.0], [+s, +s, 0.0], [+s, -s, 0.0], [-s, -s, 0.0]],
            dtype=np.float64,
        )
```

- [ ] **Step 5: Implement ArucoMarkerSet with validation and TOML persistence**

```python
# append to src/caliscope/core/aruco_marker.py

@dataclass(frozen=True)
class ArucoMarkerSet:
    dictionary: int
    markers: dict[int, ArucoMarker]

    def __post_init__(self) -> None:
        if not self.markers:
            raise ValueError("ArucoMarkerSet requires at least one marker")
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        capacity = len(aruco_dict.bytesList)
        for mid in self.markers:
            if mid < 0 or mid >= capacity:
                raise ValueError(
                    f"Marker ID {mid} exceeds dictionary capacity ({capacity})"
                )

    @classmethod
    def from_toml(cls, path: Path) -> ArucoMarkerSet:
        from caliscope.persistence import PersistenceError

        if not path.exists():
            raise PersistenceError(f"ArucoMarkerSet file not found: {path}")
        try:
            data = rtoml.load(path)
            dictionary = data["dictionary"]
            markers = {}
            for entry in data.get("markers", []):
                mid = entry["id"]
                size_m = entry["size_m"]
                markers[mid] = ArucoMarker(marker_id=mid, size_m=size_m)
            return cls(dictionary=dictionary, markers=markers)
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(
                f"Failed to load ArucoMarkerSet from {path}: {e}"
            ) from e

    def to_toml(self, path: Path) -> None:
        from caliscope.persistence import PersistenceError, _safe_write_toml

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "dictionary": self.dictionary,
                "markers": [
                    {"id": m.marker_id, "size_m": m.size_m}
                    for m in sorted(self.markers.values(), key=lambda m: m.marker_id)
                ],
            }
            _safe_write_toml(data, path)
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(
                f"Failed to save ArucoMarkerSet to {path}: {e}"
            ) from e

    def generate_marker_image(self, marker_id: int, pixel_size: int) -> NDArray:
        if marker_id not in self.markers:
            raise KeyError(
                f"Marker {marker_id} not in set (available: {sorted(self.markers.keys())})"
            )
        marker = self.markers[marker_id]
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, pixel_size)

        border = pixel_size // 2
        bordered = cv2.copyMakeBorder(
            marker_img, border, border, border, border,
            cv2.BORDER_CONSTANT, value=(255.0,),
        )
        annotated = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = pixel_size / 400
        thickness = max(1, int(pixel_size / 100))
        label_thick = max(1, thickness - 1)

        size_cm = marker.size_m * 100
        info_y = border + pixel_size + border - int(font_scale * 5)
        cv2.putText(
            annotated,
            f"ID: {marker_id}  Size: {size_cm:.1f} cm",
            (border, info_y),
            font, font_scale * 0.5, (0, 0, 0), label_thick,
        )
        return annotated
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_aruco_marker.py -v`
Expected: All pass

- [ ] **Step 7: Type check**

Run: `uv run basedpyright src/caliscope/core/aruco_marker.py tests/test_aruco_marker.py`
Expected: 0 errors

- [ ] **Step 8: Commit**

```bash
git add src/caliscope/core/aruco_marker.py tests/test_aruco_marker.py
git commit -m "feat: add ArucoMarker and ArucoMarkerSet with TOML persistence"
```

---

### Task 2: ArucoTracker migration + multi-marker filtering test

**Files:**
- Modify: `src/caliscope/trackers/aruco_tracker.py`
- Modify: `tests/test_aruco_target.py` → rename to `tests/test_aruco_marker.py` (merge)
- Test: `tests/test_aruco_marker.py` (append multi-marker filtering test)

**Interfaces:**
- Consumes: `ArucoMarkerSet` from Task 1
- Produces: `ArucoTracker(dictionary, inverted, mirror_flag_search, marker_set: ArucoMarkerSet | None)`

- [ ] **Step 1: Write multi-marker filtering test**

```python
# append to tests/test_aruco_marker.py
def test_tracker_multi_marker_uses_per_marker_size():
    """Two markers of different sizes produce different obj_loc corners."""
    small = ArucoMarker(0, 0.05)
    large = ArucoMarker(3, 0.10)
    marker_set = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_100,
        markers={0: small, 3: large},
    )

    # Generate a synthetic image with both markers
    img = np.ones((600, 800), dtype=np.uint8) * 255
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    marker_0_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 100)
    marker_3_img = cv2.aruco.generateImageMarker(aruco_dict, 3, 100)
    img[100:200, 100:200] = marker_0_img
    img[100:200, 400:500] = marker_3_img

    from caliscope.trackers.aruco_tracker import ArucoTracker
    tracker = ArucoTracker(marker_set=marker_set)
    packet = tracker.get_points(img)

    if packet.obj_loc is not None and len(packet.obj_loc) > 0:
        # Check that marker 0 and marker 3 have different corner magnitudes
        mask_0 = packet.object_id == 0
        mask_3 = packet.object_id == 3
        if mask_0.any() and mask_3.any():
            corners_0 = packet.obj_loc[mask_0]
            corners_3 = packet.obj_loc[mask_3]
            # Small marker corners are ±0.025, large are ±0.05
            max_0 = np.max(np.abs(corners_0[:, :2]))
            max_3 = np.max(np.abs(corners_3[:, :2]))
            assert max_0 == pytest.approx(0.025, abs=1e-6)
            assert max_3 == pytest.approx(0.05, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aruco_marker.py::test_tracker_multi_marker_uses_per_marker_size -v`
Expected: FAIL (ArucoTracker still expects `aruco_target` parameter)

- [ ] **Step 3: Migrate ArucoTracker to use ArucoMarkerSet**

In `src/caliscope/trackers/aruco_tracker.py`:
- Change import from `ArucoTarget` to `ArucoMarkerSet`
- Rename constructor parameter `aruco_target` to `marker_set`
- Update `_apply_target_filter`:
  - `set(self.aruco_target.marker_ids)` → `set(self.marker_set.markers.keys())`
  - `self.aruco_target.corners[int(oid)][int(kid)]` → `self.marker_set.markers[int(oid)].corners[int(kid)]`
- Update all `self.aruco_target` references to `self.marker_set`

- [ ] **Step 4: Migrate existing tests from test_aruco_target.py**

Move the tracker-related tests to `tests/test_aruco_marker.py`, updating them to use `ArucoMarkerSet` construction instead of `ArucoTarget.single_marker()`. For example:

```python
# Old:
target = ArucoTarget.single_marker(marker_id=0, marker_size_m=0.05)
tracker = ArucoTracker(aruco_target=target)

# New:
marker_set = ArucoMarkerSet(
    dictionary=cv2.aruco.DICT_4X4_100,
    markers={0: ArucoMarker(0, 0.05)},
)
tracker = ArucoTracker(marker_set=marker_set)
```

- [ ] **Step 5: Run all aruco tests**

Run: `uv run pytest tests/test_aruco_marker.py -v`
Expected: All pass

- [ ] **Step 6: Type check**

Run: `uv run basedpyright src/caliscope/trackers/aruco_tracker.py`
Expected: 0 errors

- [ ] **Step 7: Commit**

```bash
git add src/caliscope/trackers/aruco_tracker.py tests/test_aruco_marker.py
git commit -m "refactor: migrate ArucoTracker from ArucoTarget to ArucoMarkerSet"
```

---

### Task 3: Repository + workspace coordinator migration

**Files:**
- Modify: `src/caliscope/repositories/calibration_targets_repository.py`
- Modify: `src/caliscope/workspace_coordinator.py`

**Interfaces:**
- Consumes: `ArucoMarkerSet`, `ArucoMarker` from Task 1
- Produces: `CalibrationTargetsRepository.load_aruco_marker_set()`, `.save_aruco_marker_set()`, `.aruco_marker_set_exists()`

- [ ] **Step 1: Migrate CalibrationTargetsRepository**

In `src/caliscope/repositories/calibration_targets_repository.py`:
- Change import: `ArucoTarget` → `ArucoMarker, ArucoMarkerSet`
- Rename methods:
  - `load_aruco_target` → `load_aruco_marker_set`, return type `ArucoMarkerSet`
  - `save_aruco_target` → `save_aruco_marker_set`, param type `ArucoMarkerSet`
  - `aruco_target_exists` → `aruco_marker_set_exists`
- Change filename constant: `"aruco_target.toml"` → `"aruco_marker_set.toml"` (4 places)
- In `initialize_defaults`: replace `ArucoTarget.single_marker(...)` with:
  ```python
  default_aruco = ArucoMarkerSet(
      dictionary=cv2.aruco.DICT_4X4_100,
      markers={0: ArucoMarker(marker_id=0, size_m=0.05)},
  )
  ```
- Add legacy-file warning in `initialize_defaults`:
  ```python
  if not self.aruco_marker_set_exists():
      legacy = self._dir / "aruco_target.toml"
      if legacy.exists():
          logger.warning(
              "Found legacy aruco_target.toml; this format is no longer supported. "
              "Please recreate your marker set in the new aruco_marker_set.toml format."
          )
  ```
- Update docstring file layout to mention `aruco_marker_set.toml`

- [ ] **Step 2: Migrate workspace_coordinator.py**

In `src/caliscope/workspace_coordinator.py`:
- Change import: `ArucoTarget` → `ArucoMarker, ArucoMarkerSet`
- In `update_extrinsic_aruco_target` → rename to `update_extrinsic_aruco_marker_set`, change param type, call `save_aruco_marker_set`
- In `create_extrinsic_tracker`:
  - `aruco_target_exists()` → `aruco_marker_set_exists()`
  - Replace `ArucoTarget.single_marker(...)` with `ArucoMarkerSet(...)` construction
  - `save_aruco_target` → `save_aruco_marker_set`
  - `load_aruco_target` → `load_aruco_marker_set`
  - `ArucoTracker(dictionary=target.dictionary, aruco_target=target)` → `ArucoTracker(dictionary=marker_set.dictionary, marker_set=marker_set)`

- [ ] **Step 3: Type check both files**

Run: `uv run basedpyright src/caliscope/repositories/calibration_targets_repository.py src/caliscope/workspace_coordinator.py`
Expected: 0 errors

- [ ] **Step 4: Run targeted tests**

Run: `uv run pytest tests/test_aruco_marker.py tests/ -k "aruco or target or calibration" -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/caliscope/repositories/calibration_targets_repository.py src/caliscope/workspace_coordinator.py
git commit -m "refactor: migrate repository and coordinator to ArucoMarkerSet"
```

---

### Task 4: PnP bootstrap key-widening + legacy stereoCalibrate deletion

**Files:**
- Modify: `src/caliscope/core/bootstrap_pose/pose_network_builder.py:261,289,455-477`
- Delete: `src/caliscope/core/bootstrap_pose/legacy_stereocal_paired_pose_network.py`
- Modify: `src/caliscope/core/bootstrap_pose/build_paired_pose_network.py`
- Modify: `src/caliscope/gui/presenters/extrinsic_calibration_presenter.py` (remove `method="pnp"` kwarg)
- Modify: `src/caliscope/synthetic/explorer/presenter.py` (remove `method="pnp"` kwarg)
- Modify: `tests/test_paired_pose_network.py` (remove `method="pnp"` kwarg)
- Modify: `tests/synthetic/test_unposed_cameras.py` (convert stereocalibrate tests to PnP)

**Interfaces:**
- Consumes: `PoseNetworkBuilder` pipeline
- Produces: `compute_camera_to_object_poses_pnp` returns `dict[tuple[int, int, int], ...]` keyed by `(cam_id, sync_index, object_id)`

- [ ] **Step 1: Widen PnP result key in pose_network_builder.py**

At line 261, stop discarding `object_id`:
```python
# Before:
for (cam_id, sync_index, _object_id), group in grouped:
# After:
for (cam_id, sync_index, object_id), group in grouped:
```

At line 289, include `object_id` in the key:
```python
# Before:
poses[(cam_id, sync_index)] = (R, t, rmse)
# After:
poses[(cam_id, sync_index, object_id)] = (R, t, rmse)
```

Update the return type annotation at line 213:
```python
# Before:
) -> dict[tuple[int, int], tuple[...]]
# After:
) -> dict[tuple[int, int, int], tuple[...]]
```

Update `_camera_to_object_poses` type annotation in `PoseNetworkBuilder.__init__` (line 62-64):
```python
# Before:
self._camera_to_object_poses: (
    dict[tuple[int, int], tuple[NDArray[np.float64], NDArray[np.float64], float]] | None
) = None
# After:
self._camera_to_object_poses: (
    dict[tuple[int, int, int], tuple[NDArray[np.float64], NDArray[np.float64], float]] | None
) = None
```

- [ ] **Step 2: Update compute_relative_poses to match on (sync_index, object_id)**

At lines 454-498, update the function signature and matching logic:
```python
def compute_relative_poses(
    camera_to_object_poses: dict[tuple[int, int, int], tuple[NDArray[np.float64], NDArray[np.float64], float]],
    camera_array: CameraArray,
) -> dict[tuple[tuple[int, int], int], StereoPair]:
    ...
    for cam_id_a, cam_id_b in pairs:
        # Find (sync_index, object_id) tuples where both cameras have poses
        so_a = {(s, o) for c, s, o in camera_to_object_poses.keys() if c == cam_id_a}
        so_b = {(s, o) for c, s, o in camera_to_object_poses.keys() if c == cam_id_b}
        common_so = so_a.intersection(so_b)

        for sync_index, object_id in common_so:
            R_a, t_a, _ = camera_to_object_poses[(cam_id_a, sync_index, object_id)]
            R_b, t_b, _ = camera_to_object_poses[(cam_id_b, sync_index, object_id)]
            ...
```

- [ ] **Step 3: Delete legacy stereoCalibrate path**

```bash
rm src/caliscope/core/bootstrap_pose/legacy_stereocal_paired_pose_network.py
```

- [ ] **Step 4: Simplify build_paired_pose_network.py**

Replace contents of `src/caliscope/core/bootstrap_pose/build_paired_pose_network.py`:
```python
from __future__ import annotations

import logging

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.core.point_data import ImagePoints
from caliscope.core.bootstrap_pose.pose_network_builder import PoseNetworkBuilder

logger = logging.getLogger(__name__)


def build_paired_pose_network(
    image_points: ImagePoints,
    camera_array: CameraArray,
) -> PairedPoseNetwork:
    builder = PoseNetworkBuilder(camera_array, image_points)
    return (
        builder
        .estimate_camera_to_object_poses()
        .estimate_relative_poses()
        .filter_outliers(threshold=1.5)
        .build()
    )
```

- [ ] **Step 5: Remove method="pnp" kwargs from callers**

In `src/caliscope/gui/presenters/extrinsic_calibration_presenter.py`: remove `method="pnp"` from `build_paired_pose_network(...)` call.

In `src/caliscope/synthetic/explorer/presenter.py`: remove `method="pnp"` from `build_paired_pose_network(...)` call.

In `tests/test_paired_pose_network.py`: remove `method="pnp"` from `build_paired_pose_network(...)` call.

- [ ] **Step 6: Convert stereocalibrate tests in test_unposed_cameras.py**

In `tests/synthetic/test_unposed_cameras.py`: change all `method="stereocalibrate"` to remove the `method` kwarg (PnP is now the only path). Remove the import of `build_legacy_stereocal_paired_pose_network` if present.

- [ ] **Step 7: Type check**

Run: `uv run basedpyright src/caliscope/core/bootstrap_pose/pose_network_builder.py src/caliscope/core/bootstrap_pose/build_paired_pose_network.py`
Expected: 0 errors

- [ ] **Step 8: Run PnP-related tests**

Run: `uv run pytest tests/test_paired_pose_network.py tests/synthetic/test_unposed_cameras.py -v`
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add -A src/caliscope/core/bootstrap_pose/ src/caliscope/gui/presenters/extrinsic_calibration_presenter.py src/caliscope/synthetic/explorer/presenter.py tests/test_paired_pose_network.py tests/synthetic/test_unposed_cameras.py
git commit -m "fix: widen PnP key to (cam_id, sync_index, object_id), delete legacy stereoCalibrate"
```

---

### Task 5: Guard align_to_object and scale accuracy for multi-marker

**Files:**
- Modify: `src/caliscope/core/capture_volume.py:585,671`

**Interfaces:**
- Consumes: `CaptureVolume.image_points`, `CaptureVolume.world_points`
- Produces: `ValueError` when multiple `object_id` values present

- [ ] **Step 1: Add multi-object_id guard to compute_volumetric_scale_accuracy**

At the top of `compute_volumetric_scale_accuracy` (after line 597), add:
```python
unique_objects = img_df["object_id"].unique()
if len(unique_objects) > 1:
    raise ValueError(
        f"Scale accuracy requires single-object data, got object_ids {sorted(unique_objects)}. "
        "Multi-marker scale accuracy requires Branch 3 constraint file."
    )
```

- [ ] **Step 2: Add multi-object_id guard to align_to_object**

At the top of `align_to_object` (after getting the data for `sync_index`), add the same guard on the points at that sync_index.

- [ ] **Step 3: Type check**

Run: `uv run basedpyright src/caliscope/core/capture_volume.py`
Expected: 0 errors

- [ ] **Step 4: Commit**

```bash
git add src/caliscope/core/capture_volume.py
git commit -m "guard: align_to_object and scale_accuracy reject multi-marker data"
```

---

### Task 6: GUI migration — summary panel, preview, save-all

**Files:**
- Modify: `src/caliscope/gui/widgets/aruco_target_config_panel.py` → rewrite as `aruco_marker_set_panel.py`
- Modify: `src/caliscope/gui/utils/aruco_preview.py`
- Modify: `src/caliscope/gui/views/project_setup_view.py`

**Interfaces:**
- Consumes: `ArucoMarkerSet` from Task 1, repository from Task 3
- Produces: `ArucoMarkerSetPanel(marker_set: ArucoMarkerSet, targets_dir: Path)` widget, `render_aruco_pixmap(marker_set, marker_id, size)` utility

- [ ] **Step 1: Update aruco_preview.py**

```python
# src/caliscope/gui/utils/aruco_preview.py
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from caliscope.core.aruco_marker import ArucoMarkerSet


def render_aruco_pixmap(marker_set: ArucoMarkerSet, marker_id: int, size: int) -> QPixmap:
    marker = marker_set.markers[marker_id]
    ppm = int(size / marker.size_m * 4.0)
    pixel_size = int(marker.size_m * ppm)
    bgr = marker_set.generate_marker_image(marker_id, pixel_size)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage.copy())
    return pixmap.scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
```

- [ ] **Step 2: Rewrite aruco_target_config_panel.py as summary panel**

Replace `src/caliscope/gui/widgets/aruco_target_config_panel.py` with a new `ArucoMarkerSetPanel` that shows:
- Marker count, IDs, and sizes as a read-only summary label
- "Open Folder" button (opens TOML directory in system file browser)
- "Reload" button (re-reads TOML, emits `config_changed`)
- "Save All Markers" button (saves one PNG per marker)

The panel receives `ArucoMarkerSet` and the targets directory path. No interactive editing — the TOML is the interface.

- [ ] **Step 3: Update project_setup_view.py**

- Change import from `ArucoTargetConfigPanel` to new panel class
- Update `_build_extrinsic_section` to construct new panel
- Update `_on_extrinsic_aruco_changed` — the new panel's `config_changed` signal triggers a reload from the repository
- Update `_update_extrinsic_aruco_preview` to use the new `render_aruco_pixmap` signature
- Replace `_save_aruco_png` with the panel's built-in save-all functionality
- Replace `self._coordinator.update_extrinsic_aruco_target(target)` with `self._coordinator.update_extrinsic_aruco_marker_set(marker_set)`

- [ ] **Step 4: Type check all GUI files**

Run: `uv run basedpyright src/caliscope/gui/widgets/aruco_target_config_panel.py src/caliscope/gui/utils/aruco_preview.py src/caliscope/gui/views/project_setup_view.py`
Expected: 0 errors

- [ ] **Step 5: Commit**

```bash
git add src/caliscope/gui/widgets/aruco_target_config_panel.py src/caliscope/gui/utils/aruco_preview.py src/caliscope/gui/views/project_setup_view.py
git commit -m "refactor: replace ArUco config panel with marker set summary panel"
```

---

### Task 7: Delete ArucoTarget + sweep remaining references

**Files:**
- Delete: `src/caliscope/core/aruco_target.py`
- Delete: `tests/test_aruco_target.py`
- Sweep: any remaining `ArucoTarget` or `aruco_target` import/reference

**Interfaces:**
- Consumes: All previous tasks must be complete
- Produces: Zero references to `ArucoTarget` in the codebase

- [ ] **Step 1: Delete the old files**

```bash
rm src/caliscope/core/aruco_target.py tests/test_aruco_target.py
```

- [ ] **Step 2: Grep for remaining references**

```bash
grep -rn "ArucoTarget\|aruco_target\|from caliscope.core.aruco_target" --include="*.py" src/ tests/
```

Fix any remaining references found.

- [ ] **Step 3: Full test suite**

Run: `xvfb-run --auto-servernum uv run pytest -n auto --dist=loadfile`
Expected: All pass

- [ ] **Step 4: Full type check on changed files**

Run: `uv run basedpyright src/caliscope/core/aruco_marker.py src/caliscope/trackers/aruco_tracker.py src/caliscope/repositories/calibration_targets_repository.py src/caliscope/workspace_coordinator.py src/caliscope/core/bootstrap_pose/pose_network_builder.py src/caliscope/core/bootstrap_pose/build_paired_pose_network.py src/caliscope/core/capture_volume.py src/caliscope/gui/widgets/aruco_target_config_panel.py src/caliscope/gui/utils/aruco_preview.py src/caliscope/gui/views/project_setup_view.py`
Expected: 0 errors

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete ArucoTarget, sweep remaining references"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Every section of `specs/multi-marker-aruco.md` maps to a task. Domain model (T1), TOML format (T1), tracker changes (T2), PnP fix (T4), stereoCalibrate deletion (T4), align_to_object guard (T5), GUI (T6), file deletion (T7), multi-marker filtering test (T2), integration tests (verified via T4+T7 full suite).
- [x] **Placeholder scan:** No TBDs, no "similar to Task N", all steps have code.
- [x] **Type consistency:** `ArucoMarkerSet`/`ArucoMarker` names consistent across tasks. `marker_set` parameter name consistent. `load_aruco_marker_set`/`save_aruco_marker_set`/`aruco_marker_set_exists` method names consistent.
