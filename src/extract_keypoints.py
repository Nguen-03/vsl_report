"""Extract pose (including face points 0-10) and hand landmarks in memory.

Each frame is ordered as 25 pose points, 21 left-hand points, and 21 right-hand
points. The result has shape (60, 201).

Example:
    sequence = extract_keypoints("input.mp4")
    print(sequence.shape)  # (60, 201)
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

FRAMES = 60
RAW_JOINTS = 67
DIMS = 3
UPDATE_DIMS = 2
FEATURES = RAW_JOINTS * DIMS
POSE_LANDMARK_COUNT = 25

FACE_CONNECTION = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)
]
ARMS_CONNECTION = [
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
]
BODY_CONNECTION = [(11, 23), (12, 24), (23, 24)]
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 17),
    (5, 9), (9, 13), (13, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

left, right = 25, 46
HAND_1 = [(u + left, v + left) for u, v in HAND_CONNECTIONS]
HAND_2 = [(u + right, v + right) for u, v in HAND_CONNECTIONS]


def _joints(connections: Sequence[tuple[int, int]]) -> list[int]:
    return sorted({joint for edge in connections for joint in edge})


FACE_JOINTS = _joints(FACE_CONNECTION)
POSE_JOINTS = _joints(ARMS_CONNECTION + BODY_CONNECTION)
HAND_1_JOINTS = _joints(HAND_1)
HAND_2_JOINTS = _joints(HAND_2)
SELECTED_JOINTS = sorted(
    set(FACE_JOINTS + POSE_JOINTS + HAND_1_JOINTS + HAND_2_JOINTS)
)


def _landmark_coordinates(
    landmarks: Sequence[Any] | None,
    indices: Sequence[int],
) -> np.ndarray:
    """Return selected xyz values, using zeros for missing landmarks."""
    coordinates = np.zeros((len(indices), DIMS), dtype=np.float32)
    if landmarks is None:
        return coordinates

    for row, index in enumerate(indices):
        if index < len(landmarks):
            point = landmarks[index]
            coordinates[row] = (point.x, point.y, point.z)
    return coordinates


def normalize_parts(parts: list[np.ndarray]) -> list[np.ndarray]:
    """Normalize pose and hands around the shoulders as in the training pipeline."""
    pose = parts[0]
    left_shoulder, right_shoulder = pose[11], pose[12]
    scale = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    if not np.isfinite(scale) or scale < 1e-6:
        return [np.zeros_like(part) for part in parts]

    center = (left_shoulder + right_shoulder) / 2.0
    normalized = []
    for part in parts:
        part = part.copy()
        if part.any():
            part[:, :3] = (part[:, :3] - center) / scale
        normalized.append(part)
    return normalized


def pick_hands(
    hand_result: Any,
    swap_hands: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left and right hand arrays, correcting labels for unmirrored video."""
    best_score = {"Left": -1.0, "Right": -1.0}
    hands: dict[str, Sequence[Any] | None] = {"Left": None, "Right": None}
    for landmarks, handedness in zip(
        hand_result.hand_landmarks, hand_result.handedness
    ):
        if not handedness:
            continue
        label = handedness[0].category_name
        if label not in hands:
            continue
        score = float(handedness[0].score)
        if swap_hands:
            label = "Right" if label == "Left" else "Left"
        if score > best_score[label]:
            best_score[label] = score
            hands[label] = landmarks
    return (
        _landmark_coordinates(hands["Left"], range(21)),
        _landmark_coordinates(hands["Right"], range(21)),
    )


def landmarks_to_frame(
    pose_result: Any,
    hand_result: Any,
    swap_hands: bool = True,
) -> np.ndarray:
    """Build one frame as 25 pose points followed by left and right hands."""
    pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
    pose = _landmark_coordinates(pose_landmarks, range(POSE_LANDMARK_COUNT))
    left_hand, right_hand = pick_hands(hand_result, swap_hands)
    parts = normalize_parts([pose, left_hand, right_hand])
    frame = np.concatenate([part.reshape(-1) for part in parts])
    if frame.shape != (FEATURES,):
        raise ValueError(f"Expected {FEATURES} values per frame, got {frame.shape}.")
    return frame


def _resample_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("Không đọc được frame nào từ video.")
    indices = np.linspace(0, len(rows) - 1, num=FRAMES).round().astype(np.int64)
    return np.stack([rows[index] for index in indices]).astype(np.float32, copy=False)


def _detect_frame(
    pose_landmarker: PoseLandmarker,
    hand_landmarker: HandLandmarker,
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    previous_timestamp: int,
    mirrored_input: bool,
) -> tuple[np.ndarray, int]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = max(previous_timestamp + 1, int(frame_index * 1000 / fps))
    pose_result = pose_landmarker.detect_for_video(image, timestamp)
    hand_result = hand_landmarker.detect_for_video(image, timestamp)
    return landmarks_to_frame(
        pose_result, hand_result, swap_hands=not mirrored_input
    ), timestamp


def _extract_sequence(
    capture: cv2.VideoCapture,
    pose_landmarker: PoseLandmarker,
    hand_landmarker: HandLandmarker,
    mirrored_input: bool,
) -> np.ndarray:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    previous_timestamp = -1
    rows: list[np.ndarray] = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        row, previous_timestamp = _detect_frame(
            pose_landmarker,
            hand_landmarker,
            frame,
            frame_index,
            fps,
            previous_timestamp,
            mirrored_input,
        )
        rows.append(row)
        frame_index += 1

    if not rows:
        raise ValueError("Không đọc được frame nào từ video.")
    return _resample_rows(rows)


def spatial_normalize(sequences, left_shoulder_idx=11, right_shoulder_idx=12):
    result = []
    for sequence in sequences:
        arr = sequence.reshape(FRAMES, RAW_JOINTS, DIMS)
        left_shoulder = arr[:, left_shoulder_idx, :]
        
        right_shoulder = arr[:, right_shoulder_idx, :]
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        # print(mid_shoulder.shape)
        mid_shoulder_expanded = np.expand_dims(mid_shoulder, axis=1)
        # print(mid_shoulder_expanded)
        arr_shifted = arr - mid_shoulder_expanded
        shoulder_widths = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
        shoulder_widths[shoulder_widths == 0] = 1.0
        shoulder_widths_expanded = np.expand_dims(shoulder_widths, axis=(1, 2))
        arr_normalized = arr_shifted / shoulder_widths_expanded
        result.append(arr_normalized)
    return result

def extract_keypoints(
    video_path: str | Path,
    model_dir: str | Path | None = None,
    *,
    min_conf: float = 0.5,
    mirrored_input: bool = False,
) -> np.ndarray:
    """Extract a fixed (60, 201) float32 array from a video."""
    video_path = Path(video_path)
    if model_dir is None:
        model_dir = Path(__file__).resolve().parents[1] / "mediapipe_models"
    model_dir = Path(model_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    model_paths = {
        "pose": model_dir / "pose_landmarker_full.task",
        "hand": model_dir / "hand_landmarker.task",
    }
    missing_models = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing_models:
        raise FileNotFoundError(
            "Thiếu model MediaPipe (.task): " + ", ".join(missing_models)
        )
    if not 0.0 < min_conf <= 1.0:
        raise ValueError("min_conf phải lớn hơn 0 và không vượt quá 1.")
    if SELECTED_JOINTS != list(range(RAW_JOINTS)):
        raise RuntimeError("Danh sách connection không tạo đủ thứ tự 67 joints.")

    with ExitStack() as stack:
        pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_paths["pose"])),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_conf,
                min_pose_presence_confidence=min_conf,
                min_tracking_confidence=min_conf,
            )
        )
        stack.callback(pose_landmarker.close)
        hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_paths["hand"])),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
                min_tracking_confidence=min_conf,
            )
        )
        stack.callback(hand_landmarker.close)
        capture = cv2.VideoCapture(str(video_path))
        stack.callback(capture.release)
        if not capture.isOpened():
            raise ValueError(f"Không mở được video: {video_path}")
        sequence = _extract_sequence(
            capture, pose_landmarker, hand_landmarker, mirrored_input
        )
    sequence = sequence.reshape(60, 67, 3)
    sequence = sequence[...,:2]
    sequence = sequence.reshape(FRAMES, RAW_JOINTS * UPDATE_DIMS)
    if sequence.shape != (FRAMES, 67 * 2):
        raise ValueError(
            f"Expected sequence shape {(FRAMES, FEATURES)}, got {sequence.shape}."
        )
    # sequence = spatial_normalize(sequence)
    return sequence
import matplotlib.pyplot as plt
def plot_frame(data, frame_idx, title=None, figsize=(10, 7)):
    sequence = np.asarray(data)
    if sequence.shape == (FRAMES, FEATURES):
        sequence = sequence.reshape(FRAMES, RAW_JOINTS, DIMS)
    if sequence.shape == (FRAMES, RAW_JOINTS * UPDATE_DIMS):
        sequence = sequence.reshape(FRAMES, RAW_JOINTS, UPDATE_DIMS)
    # sequence = sequence[:,:,:2]
    # print(sequence.shape)
    if sequence.shape != (FRAMES, RAW_JOINTS, UPDATE_DIMS):
        raise ValueError(
            f"Expected {(FRAMES, RAW_JOINTS, UPDATE_DIMS)} or {(FRAMES, RAW_JOINTS * UPDATE_DIMS)}, "
            f"got {sequence.shape}."
        )
    if not 0 <= frame_idx < FRAMES:
        raise IndexError(f"frame_idx must be in [0, {FRAMES - 1}].")

    points = sequence[frame_idx]
    x, y = points[:, 0], points[:, 1]
    present = np.linalg.norm(points, axis=1) > 1e-8
    _, axis = plt.subplots(figsize=figsize)
    for start, end in FACE_CONNECTION + BODY_CONNECTION + HAND_1 + HAND_2 + ARMS_CONNECTION:
        if present[start] and present[end]:
            axis.plot([x[start], x[end]], [y[start], y[end]])
    for idx in np.flatnonzero(present):
        axis.annotate(
            str(idx), (x[idx], y[idx]), fontsize=7,
            xytext=(3, 3), textcoords="offset points"
        )
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(title or f"Frame {frame_idx}")
    plt.show()


if __name__ == '__main__':
    sequence = extract_keypoints(r"E:\Report\DL\dataset\Video_dl\VID_44.mp4")
    print(sequence.shape)
    for frame_index in range(0, 60, 9):
        plot_frame(sequence, frame_index)