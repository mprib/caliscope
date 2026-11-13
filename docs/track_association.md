# Cross-Camera Track Association (Experimental)

Person tracking gives each person a track ID that is local to one camera.
Track 3 in camera 0 and track 1 in camera 2 may be the same person, and nothing in the tracks says so.
`associate_tracks` decides which tracks in different cameras show the same person and gives each person one `object_id` across the rig.

It needs cameras with extrinsics.
To calibrate from a single walking subject first, see [Markerless Calibration](markerless_calibration.md).

!!! warning "Experimental"
    This workflow has been checked on two sets of recordings with reference calibrations, described under [What it was tested on](#what-it-was-tested-on).
    Its reliability on other recordings has not been established.
    The function names and arguments may change.

## Run it

Three models take part.
A detector finds person boxes, a pose model finds keypoints inside a box, and a re-identification encoder turns a box crop into a vector that is similar for crops of the same person.

```python
from importlib.resources import files
from pathlib import Path

from caliscope import MODELS_DIR
from caliscope.api import (
    CameraArray,
    associate_tracks,
    estimate_keypoints,
    extract_reidentification_embeddings,
    track_objects,
)
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps
from caliscope.trackers.model_card import ModelCard, ReidentificationCard
from caliscope.trackers.model_download import download_and_extract_model
from caliscope.trackers.onnx_tracker import OnnxTracker
from caliscope.trackers.reidentification_encoder import ReidentificationEncoder

VIDEOS = {
    0: Path("recording/cam_0.mp4"),
    1: Path("recording/cam_1.mp4"),
    2: Path("recording/cam_2.mp4"),
}
cameras = CameraArray.from_toml("capture_volume/camera_array.toml")  # intrinsics and extrinsics
EDGE_MARGIN_PX = 24

# Load the three models, downloading any that are missing.
card_dir = Path(str(files("caliscope.trackers.model_cards")))
pose_card = ModelCard.from_toml(card_dir / "rtmpose_l_halpe26.toml", models_dir=MODELS_DIR)
encoder_card = ReidentificationCard.from_toml(card_dir / "osnet_ain_x1_0_msmt17.toml", models_dir=MODELS_DIR)
for card in (pose_card, pose_card.detector, encoder_card):
    if not card.onnx_exists:
        download_and_extract_model(card, MODELS_DIR)
tracker = OnnxTracker(pose_card)
encoder = ReidentificationEncoder(encoder_card)

# 1. Track every box in every camera.
timestamps = SynchronizedTimestamps.from_anchor_frames(VIDEOS, {cam_id: 0 for cam_id in VIDEOS})
boxes = track_objects(VIDEOS, tracker.detector, timestamps=timestamps, frame_step=1)

# 2. Get keypoints and embeddings at a few moments of each track.
sampled = boxes.sample_for_association()
points = estimate_keypoints(VIDEOS, tracker, sampled, min_confidence=pose_card.confidence_threshold)
embeddings = extract_reidentification_embeddings(VIDEOS, encoder, sampled)

# 3. Decide which tracks are the same person.
associations = associate_tracks(points, cameras, boxes, embeddings)
print(associations.linked)
print(associations.unresolved)

# 4. Get keypoints for every linked person.
linked_boxes = boxes.apply_associations(associations).exclude_edge_boxes(margin_px=EDGE_MARGIN_PX)
linked_points = estimate_keypoints(VIDEOS, tracker, linked_boxes, min_confidence=pose_card.confidence_threshold)
```

`linked_points` holds the keypoints of every linked person, with the same `object_id` in every camera.
Object 0 is the person whose boxes cover the most image area, summed across cameras.

Each numbered step reads the videos again.
The models in steps 2 and 4 run only on the boxes you pass them.
On the one recording that was profiled, decoding the videos took most of the time in every step, so step 2 cost about as much as step 4 despite running on far fewer boxes.

## Read the result

`associate_tracks` returns a `TrackAssociations` with four tables.

| Table | One row per | What it tells you |
|---|---|---|
| `linked` | linked track | Its `object_id`, the number of sampled moments at which it matched another track of that object, and the share of its chances to match that did match |
| `unresolved` | track that was not linked | The reason |
| `pair_evidence` | pair of tracks seen together in different cameras | The sampled moments both were seen at, and how many of those matched |
| `objects` | linked object | The median detector score of its boxes and the median confidence of its sampled keypoints |

Every track appears once, in `linked` or in `unresolved`.
The reasons in `unresolved` are:

- `no keypoints sampled`: the track had no keypoints at the sampled moments.
- `camera has no extrinsics`: the track's camera is not posed.
- `matched no track in another camera`: nothing in another camera agreed with it.
- `conflicts with track N in the same camera`: this track and track N overlap in time and both fit the same person, so neither is linked.
- `too few shared moments to join object N`: it matched an object, but at fewer than three shared moments.

Low values in `objects` suggest the detector tracked something that is not a person.
In one test recording it tracked a treadmill in two cameras.
The two tracks were linked to each other correctly, and that object had a detector score of 0.24 and a keypoint confidence of 0.35, against 0.85 for the person in the same recording.

Save and reload the result with `associations.to_toml(path)` and `TrackAssociations.from_toml(path)`.
`BoxTracks` and `ImagePoints` have `to_csv` and `from_csv`, and the embeddings have `to_npz` and `from_npz`, so you can keep any stage and rerun the ones after it.

## How it decides

The decision has two stages.

**At each sampled moment**, the boxes of all cameras are matched with the method of Dong et al. (2019), Section 3.2.
Every pair of boxes in different cameras gets an affinity between 0 and 1 from two measurements.
The geometric one asks whether the keypoints of each box lie near the epipolar lines of the other box's keypoints.
The re-identification one is the cosine similarity of the two embeddings.
The affinity is the geometric mean of the two, so a pair that fails the geometry cannot match however alike the crops look.
The method then finds the set of matches that agrees around every loop of cameras: if box A matches B and B matches C, then A matches C.

**Across moments**, each pair of tracks is scored by how often it matched out of the moments both tracks were seen.
A pair that matched at more than half of them counts toward a link, and a pair that matched at fewer counts against it.
A pair seen together at fewer than three moments counts for nothing.
Tracks are then merged into objects while the scores between them are positive on balance, taking the clearest merge first.
Two tracks that overlap in time in one camera never join the same object.
When two such tracks fit an object about equally well, both stay unresolved.

The first stage is the published method.
The second stage is Caliscope's own, because the published method works on one moment at a time.
Three details of the first stage also differ from the paper:

- The geometric affinity is a fixed function of the epipolar distance divided by the box size.
  The authors' released code standardizes the pixel distances within each frame, which has no meaning when a frame holds one or two people.
- The paper's "appearance affinity" is called the re-identification affinity here.
- When the matches at a moment put two boxes of one camera in the same group, those two boxes are left unmatched at that moment and the rest of the group is kept.

## The re-identification model

The packaged encoder is OSNet-AIN x1.0 (Zhou et al., 2021), trained by its authors on the MSMT17 person re-identification dataset and published with the [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) library under the MIT license.
Caliscope downloads an [ONNX export](https://huggingface.co/mprib/osnet-ain-x1-0-msmt17-onnx) of the [original weights](https://huggingface.co/kaiyangzhou/osnet).

It is trained on people.
An embedding only supports asking whether two crops show the same individual within one recording.
It does not say who anyone is.

`ReidentificationEncoder` is passed in as an argument, so you can supply another object with an `encode(crops)` method that returns one unit-length vector per crop.
The two similarity thresholds in `associate_tracks` were measured with this encoder.
Another encoder needs its own.

## What it was tested on

Two sources with reference calibrations: the multi-person demo recording of Pose2Sim (four cameras, three people) and 16 trials from the BioCV dataset (nine cameras, one subject, with bystanders in view).

Each recording was run on the full rig and on every subset of two and three cameras, 1,947 cases in all.
In every case, the tracks of each known person that shared at least three sampled moments were linked into one object, and no object mixed a known person with a track seen alongside that person in the same camera.
The same held with the largest allowed epipolar distance set anywhere from 0.08 to 0.60 of the box size.
The packaged value is 0.20.

The thresholds were set on these same recordings, so this is not a held-out test.
In BioCV only the subject is labeled.
The bystander groups were checked by reprojection error and by eye, not against labels.

## References

Dong, J., Jiang, W., Huang, Q., Bao, H., and Zhou, X. (2019). Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views. CVPR.

Zhou, K., Yang, Y., Cavallaro, A., and Xiang, T. (2021). Learning Generalisable Omni-Scale Representations for Person Re-Identification. IEEE TPAMI.

Wei, L., Zhang, S., Gao, W., and Tian, Q. (2018). Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR.
