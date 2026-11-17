"""Multi-object synthetic scenes as camera-local tracks, for track association tests.

Builds scenes on SyntheticScene and converts them to what track_objects and
estimate_keypoints would produce: unlinked ImagePoints carrying track_id,
BoxTracks drawn around each track's keypoints, and the truth of which scene
object each (cam_id, track_id) shows.

Trajectories are built pose by pose rather than with Trajectory.linear or
Trajectory.orbital, because those re-base every frame so the first one is the
world origin, which would start every object at the same place.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.box_tracks import BoxTracks, _covered_area
from caliscope.core.point_data import ImagePoints
from caliscope.core.reidentification_embeddings import ReidentificationEmbeddings
from caliscope.core.track_association import TrackAssociations
from caliscope.synthetic import CalibrationObject, CameraSynthesizer, SceneObject, SE3Pose, SyntheticScene, Trajectory

TrackKey = tuple[int, int]

CAMERA_RING_RADIUS_M = 5.0
CAMERA_HEIGHT_M = 0.9
SPLIT_GAP_FRAMES = 3
DUPLICATE_OFFSET_PX = 2.0
BOX_PADDING_PX = 12.0

_UP = np.array([0.0, 0.0, 1.0])
_DUPLICATE_FRAGMENT = 2

# 17 body-like keypoints in a local frame with the feet at z = 0, spanning
# 0.4 m (x) by 0.3 m (y) by 1.7 m (z), so no camera pair sees a planar cloud.
_BODY_POINTS = np.array(
    [
        [0.00, 0.12, 1.62],  # 0 nose
        [0.00, 0.00, 1.70],  # 1 head top
        [0.00, 0.00, 1.45],  # 2 neck
        [-0.20, 0.00, 1.42],  # 3 left shoulder
        [0.20, 0.00, 1.42],  # 4 right shoulder
        [-0.20, 0.08, 1.12],  # 5 left elbow
        [0.20, -0.05, 1.12],  # 6 right elbow
        [-0.18, 0.15, 0.88],  # 7 left wrist
        [0.18, -0.15, 0.88],  # 8 right wrist
        [0.00, 0.10, 1.25],  # 9 chest
        [-0.12, 0.00, 0.95],  # 10 left hip
        [0.12, 0.00, 0.95],  # 11 right hip
        [-0.12, 0.06, 0.50],  # 12 left knee
        [0.12, -0.04, 0.50],  # 13 right knee
        [-0.12, -0.02, 0.08],  # 14 left ankle
        [0.12, 0.02, 0.08],  # 15 right ankle
        [-0.12, 0.14, 0.00],  # 16 left toe
    ]
)


def body_target(scale: float = 1.0) -> CalibrationObject:
    """A non-planar 17-point constellation the size of a standing person, scaled."""
    return CalibrationObject.from_points(_BODY_POINTS * scale)


def ring_cameras(n: int = 4, drop: Iterable[int] = ()) -> CameraArray:
    """A ring of inward-facing webcams at chest height, optionally with gaps in cam_id."""
    return (
        CameraSynthesizer()
        .add_ring(n=n, radius=CAMERA_RING_RADIUS_M, height=CAMERA_HEIGHT_M)
        .drop_cam_ids(*drop)
        .build()
    )


def _floor_pose(xy: np.ndarray, heading_rad: float) -> SE3Pose:
    return SE3Pose.from_axis_angle(_UP, heading_rad, np.array([xy[0], xy[1], 0.0]))


def walking(start: Sequence[float], end: Sequence[float], n_frames: int) -> Trajectory:
    """Walk in a straight line across the floor, facing the direction of travel."""
    start_xy = np.asarray(start, dtype=float)
    end_xy = np.asarray(end, dtype=float)
    direction = end_xy - start_xy
    heading = float(np.arctan2(direction[1], direction[0]))
    poses = tuple(_floor_pose(start_xy + t * direction, heading) for t in np.linspace(0.0, 1.0, n_frames))
    return Trajectory(poses=poses, origin_frame=0)


def turning_in_place(center: Sequence[float], n_frames: int, radius: float = 0.1) -> Trajectory:
    """Turn one full circle while swaying on a small circle around center."""
    center_xy = np.asarray(center, dtype=float)
    angles = np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False)
    poses = tuple(
        _floor_pose(center_xy + radius * np.array([np.cos(angle), np.sin(angle)]), float(angle)) for angle in angles
    )
    return Trajectory(poses=poses, origin_frame=0)


def standing(position: Sequence[float], n_frames: int, jitter_m: float = 0.003, seed: int = 0) -> Trajectory:
    """Stand still apart from a few millimetres of seeded jitter."""
    rng = np.random.default_rng(seed)
    position_xy = np.asarray(position, dtype=float)
    offsets = rng.normal(0.0, jitter_m, size=(n_frames, 2))
    headings = rng.normal(0.0, 0.01, size=n_frames)
    poses = tuple(_floor_pose(position_xy + offset, float(heading)) for offset, heading in zip(offsets, headings))
    return Trajectory(poses=poses, origin_frame=0)


def build_scene(
    cameras: CameraArray, trajectories: Sequence[Trajectory], scales: Sequence[float] | None = None
) -> SyntheticScene:
    """One body per trajectory; the scene's object_id is the true object (0, 1, ...).

    The scene itself is noiseless; to_tracked adds pixel noise.
    """
    scales = scales if scales is not None else [1.0] * len(trajectories)
    objects = tuple(
        SceneObject(object_id=index, calibration_object=body_target(scale), trajectory=trajectory)
        for index, (trajectory, scale) in enumerate(zip(trajectories, scales))
    )
    return SyntheticScene(camera_array=cameras, objects=objects, pixel_noise_sigma=0.0)


def to_tracked(
    scene: SyntheticScene,
    *,
    splits: Mapping[TrackKey, int] | None = None,
    duplicates: Mapping[TrackKey, tuple[int, int]] | None = None,
    hidden: Iterable[TrackKey] = (),
    sync_offset: int = 0,
    sync_stride: int = 1,
    noise_px: float = 2.0,
    seed: int = 0,
) -> tuple[ImagePoints, BoxTracks, dict[TrackKey, int]]:
    """Convert a scene to unlinked tracked points, full-rate boxes, and the truth.

    Keys of splits, duplicates and hidden are (cam_id, true object).
    splits cuts that track at the given frame, dropping SPLIT_GAP_FRAMES frames
    and giving the rest a new track_id. duplicates adds a second track that
    copies the first over frames [start, stop) shifted by DUPLICATE_OFFSET_PX.
    hidden removes an object from a camera entirely. Frame f becomes
    sync_index sync_offset + sync_stride * f. Each camera numbers its tracks
    from 1, and negative object_id values follow sorted (cam_id, track_id), as
    track_objects assigns them. truth maps (cam_id, track_id) to the true object.
    """
    df = scene.image_points_perfect.df.rename(columns={"object_id": "true_object", "sync_index": "frame"})
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_px, size=(len(df), 2))
    df = df.assign(
        img_loc_x=df["img_loc_x"] + noise[:, 0],
        img_loc_y=df["img_loc_y"] + noise[:, 1],
        obj_loc_x=np.nan,
        obj_loc_y=np.nan,
        obj_loc_z=np.nan,
        fragment=0,
    )

    hidden_pairs = set(hidden)
    visible = [pair not in hidden_pairs for pair in zip(df["cam_id"].tolist(), df["true_object"].tolist())]
    df = df[np.array(visible, dtype=bool)].copy()

    for (cam_id, true_object), cut in (splits or {}).items():
        same_track = (df["cam_id"] == cam_id) & (df["true_object"] == true_object)
        in_gap = same_track & (df["frame"] >= cut) & (df["frame"] < cut + SPLIT_GAP_FRAMES)
        df = df[~in_gap].copy()
        after_gap = (df["cam_id"] == cam_id) & (df["true_object"] == true_object) & (df["frame"] >= cut)
        df.loc[after_gap, "fragment"] = 1

    copies: list[pd.DataFrame] = []
    for (cam_id, true_object), (start, stop) in (duplicates or {}).items():
        same_track = (df["cam_id"] == cam_id) & (df["true_object"] == true_object)
        source = df[same_track & (df["frame"] >= start) & (df["frame"] < stop)]
        shifted_x = source["img_loc_x"] + DUPLICATE_OFFSET_PX
        copies.append(source.assign(img_loc_x=shifted_x, fragment=_DUPLICATE_FRAGMENT))
    df = pd.concat([df, *copies], ignore_index=True)

    tracks = (
        df[["cam_id", "true_object", "fragment"]]
        .drop_duplicates()
        .sort_values(["cam_id", "true_object", "fragment"])
        .reset_index(drop=True)
    )
    tracks["track_id"] = tracks.groupby("cam_id").cumcount() + 1
    tracks["object_id"] = -(np.arange(len(tracks)) + 1)

    df = df.merge(tracks, on=["cam_id", "true_object", "fragment"])
    df["sync_index"] = sync_offset + sync_stride * df["frame"]

    truth = {
        (int(cam_id), int(track_id)): int(true_object)
        for cam_id, track_id, true_object in zip(
            tracks["cam_id"].tolist(), tracks["track_id"].tolist(), tracks["true_object"].tolist()
        )
    }
    point_columns = [
        "sync_index",
        "cam_id",
        "object_id",
        "track_id",
        "keypoint_id",
        "img_loc_x",
        "img_loc_y",
        "obj_loc_x",
        "obj_loc_y",
        "obj_loc_z",
        "frame_time",
    ]
    return ImagePoints(df[point_columns]), _boxes_around_keypoints(df, scene.camera_array), truth


def _boxes_around_keypoints(df: pd.DataFrame, cameras: CameraArray) -> BoxTracks:
    """One padded box per track and frame, spanning that frame's keypoints, clipped to the image."""
    extents = df.groupby(["cam_id", "track_id", "object_id", "frame", "sync_index"], as_index=False).agg(
        x_min=("img_loc_x", "min"),
        x_max=("img_loc_x", "max"),
        y_min=("img_loc_y", "min"),
        y_max=("img_loc_y", "max"),
        frame_time=("frame_time", "first"),
    )
    cam_ids = extents["cam_id"].to_numpy(dtype=np.int64).tolist()
    width = pd.Series([float(cameras[cam_id].size[0]) for cam_id in cam_ids], index=extents.index)
    height = pd.Series([float(cameras[cam_id].size[1]) for cam_id in cam_ids], index=extents.index)
    boxes = pd.DataFrame(
        {
            "sync_index": extents["sync_index"],
            "cam_id": extents["cam_id"],
            "frame_index": extents["frame"],
            "frame_time": extents["frame_time"],
            "track_id": extents["track_id"],
            "object_id": extents["object_id"],
            "box_x1": np.maximum(extents["x_min"] - BOX_PADDING_PX, 0.0),
            "box_y1": np.maximum(extents["y_min"] - BOX_PADDING_PX, 0.0),
            "box_x2": np.minimum(extents["x_max"] + BOX_PADDING_PX, width),
            "box_y2": np.minimum(extents["y_max"] + BOX_PADDING_PX, height),
            "detector_score": 0.9,
            "image_width": width.astype(int),
            "image_height": height.astype(int),
            "rotation_count": 0,
        }
    )
    return BoxTracks(boxes)


def sample_points(points: ImagePoints, boxes: BoxTracks, frames_per_track: int = 12) -> ImagePoints:
    """Keep the points at the boxes sample_for_association keeps, as a user would."""
    kept = boxes.sample_for_association(frames_per_track).df[["cam_id", "sync_index", "track_id"]]
    return ImagePoints(points.df.merge(kept, on=["cam_id", "sync_index", "track_id"]))


def embeddings_by_true_object(points: ImagePoints, truth: Mapping[TrackKey, int]) -> ReidentificationEmbeddings:
    """One embedding per sampled box: boxes of one true object share a vector, different objects are orthogonal."""
    index = points.df[["sync_index", "cam_id", "track_id"]].drop_duplicates().reset_index(drop=True)
    vectors = np.zeros((len(index), max(truth.values()) + 2), dtype=np.float32)
    for row, (cam_id, track_id) in enumerate(zip(index["cam_id"].tolist(), index["track_id"].tolist())):
        vectors[row, truth[int(cam_id), int(track_id)]] = 1.0
    return ReidentificationEmbeddings(index, vectors)


def assert_exclusive(associations: TrackAssociations, boxes: BoxTracks) -> None:
    """Every track in boxes appears once across linked and unresolved, and the links apply cleanly."""
    linked = associations.linked
    unresolved = associations.unresolved
    keys = list(zip(linked["cam_id"].tolist(), linked["track_id"].tolist())) + list(
        zip(unresolved["cam_id"].tolist(), unresolved["track_id"].tolist())
    )
    assert len(keys) == len(set(keys)), "a track appears more than once across linked and unresolved"

    box_df = boxes.df
    box_tracks = set(zip(box_df["cam_id"].tolist(), box_df["track_id"].tolist()))
    assert set(keys) == box_tracks, "the associations do not cover exactly the tracks in the boxes"

    boxes.apply_associations(associations)


def identities_by_object(associations: TrackAssociations, truth: Mapping[TrackKey, int]) -> dict[int, set[int]]:
    """For each linked object_id, the true objects its tracks show."""
    linked = associations.linked
    identities: dict[int, set[int]] = {}
    for object_id, cam_id, track_id in zip(
        linked["object_id"].tolist(), linked["cam_id"].tolist(), linked["track_id"].tolist()
    ):
        identities.setdefault(int(object_id), set()).add(truth[(int(cam_id), int(track_id))])
    return identities


def cameras_by_object(associations: TrackAssociations) -> dict[int, set[int]]:
    """For each linked object_id, the cameras with a linked track."""
    linked = associations.linked
    cameras: dict[int, set[int]] = {}
    for object_id, cam_id in zip(linked["object_id"].tolist(), linked["cam_id"].tolist()):
        cameras.setdefault(int(object_id), set()).add(int(cam_id))
    return cameras


def object_of_track(associations: TrackAssociations, key: TrackKey) -> int | None:
    """The linked object_id of one track, or None when it is not linked."""
    linked = associations.linked
    match = linked[(linked["cam_id"] == key[0]) & (linked["track_id"] == key[1])]
    return int(match["object_id"].iloc[0]) if len(match) else None


def footprint_by_true_object(boxes: BoxTracks, truth: Mapping[TrackKey, int]) -> dict[int, int]:
    """Per true object, the union of its box pixels in each camera, summed over cameras."""
    box_df = boxes.df
    box_df["true_object"] = [
        truth[(cam_id, track_id)] for cam_id, track_id in zip(box_df["cam_id"].tolist(), box_df["track_id"].tolist())
    ]
    footprints: dict[int, int] = {}
    for true_object in sorted(set(box_df["true_object"].tolist())):
        of_object = box_df[box_df["true_object"] == true_object]
        footprints[true_object] = sum(
            _covered_area(of_object[of_object["cam_id"] == cam_id]) for cam_id in sorted(set(of_object["cam_id"]))
        )
    return footprints


def median_box_area(boxes: BoxTracks, truth: Mapping[TrackKey, int], cam_id: int, true_object: int) -> float:
    """Median area of one true object's boxes in one camera."""
    box_df = boxes.df
    keys = [key for key, obj in truth.items() if key[0] == cam_id and obj == true_object]
    rows = box_df[(box_df["cam_id"] == cam_id) & box_df["track_id"].isin([track_id for _, track_id in keys])]
    areas = (rows["box_x2"] - rows["box_x1"]) * (rows["box_y2"] - rows["box_y1"])
    return float(areas.median())
