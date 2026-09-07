"""
Video -> model input preprocessing for LipCoordNet.

Two things are produced per video:

1. `frames`  (T, 64, 128, 3) uint8 -- mouth ROI crops, obtained by warping every
   frame onto a canonical 256x256 face template and then cutting a fixed
   160x80 window over the mouth.
2. `coords`  (T, 20, 2) float32 -- the 20 dlib lip landmarks (points 48..67),
   measured on the frame resized to 600x500 and normalised by 600/500.

Both follow the conventions the pretrained LipCoordNet checkpoint was trained
with. See NOTE_ON_COORD_NORMALISATION below -- the upstream inference script
gets this wrong and we deliberately do not copy that bug.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

try:
    import dlib
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dlib is required. Install a prebuilt wheel with:  pip install dlib-bin"
    ) from exc


# --------------------------------------------------------------------------
# Canonical face template (51 points = dlib landmarks 17..67), normalised to
# [0, 1] and then scaled to the alignment canvas size. Taken from the LipNet /
# LipCoordNet reference implementations so that our crops match training.
# --------------------------------------------------------------------------
_TEMPLATE_X = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483,
    0.799124, 0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127,
    0.36688, 0.426036, 0.490127, 0.554217, 0.613373, 0.121737, 0.187122,
    0.265825, 0.334606, 0.260918, 0.182743, 0.645647, 0.714428, 0.793132,
    0.858516, 0.79751, 0.719335, 0.254149, 0.340985, 0.428858, 0.490127,
    0.551395, 0.639268, 0.726104, 0.642159, 0.556721, 0.490127, 0.423532,
    0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874, 0.553364,
    0.490127, 0.42689,
])
_TEMPLATE_Y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625,
    0.587326, 0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758,
    0.179852, 0.231733, 0.245099, 0.244077, 0.231733, 0.179852, 0.178758,
    0.216423, 0.244077, 0.245099, 0.780233, 0.745405, 0.727388, 0.742578,
    0.727388, 0.745405, 0.780233, 0.864805, 0.902192, 0.909281, 0.902192,
    0.864805, 0.784792, 0.778746, 0.785343, 0.778746, 0.784792, 0.824182,
    0.831803, 0.824182,
])

ALIGN_SIZE = 256            # canonical face canvas
MOUTH_W, MOUTH_H = 160, 80  # crop taken from the canvas
OUT_W, OUT_H = 128, 64      # what the network expects

# NOTE_ON_COORD_NORMALISATION
# During training (dataset.MyDataset._load_coords) the lip coordinates were
# divided by a *fixed* 600 x 500 -- the size each frame was resized to before
# landmark extraction. Upstream's inference.py instead divides by the raw video
# resolution (360x288 for GRID), which silently feeds the coordinate GRU inputs
# that are ~1.7x too large. We normalise by 600/500, matching training.
COORD_W, COORD_H = 600, 500


def _face_template(size: int = ALIGN_SIZE, padding: float = 0.25) -> np.ndarray:
    x = (_TEMPLATE_X + padding) / (2 * padding + 1) * size
    y = (_TEMPLATE_Y + padding) / (2 * padding + 1) * size
    return np.stack([x, y], axis=1)


TEMPLATE = _face_template()
# Fixed mouth window on the canonical canvas (identical for every frame).
_MX, _MY = TEMPLATE[-20:].mean(0).astype(np.int32)
CROP_X0, CROP_X1 = _MX - MOUTH_W // 2, _MX + MOUTH_W // 2
CROP_Y0, CROP_Y1 = _MY - MOUTH_H // 2, _MY + MOUTH_H // 2


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping `src` onto `dst`.

    Both are (N, 2). Returns a 2x3 affine matrix suitable for cv2.warpAffine.
    """
    src = src.astype(np.float64).copy()
    dst = dst.astype(np.float64).copy()

    c1, c2 = src.mean(0), dst.mean(0)
    src -= c1
    dst -= c2
    s1, s2 = src.std(), dst.std()
    src /= s1
    dst /= s2

    u, _, vt = np.linalg.svd(src.T @ dst)
    rotation = (u @ vt).T
    scale = s2 / s1

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = c2 - scale * (rotation @ c1)
    return matrix


class LandmarkDetector:
    """dlib 68-point face landmarks, with a light retry on hard frames."""

    def __init__(self, predictor_path: str, upsample: int = 0):
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"Landmark model not found: {predictor_path}\n"
                "Run  python setup_assets.py  to fetch it."
            )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.upsample = upsample

    def __call__(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return (68, 2) landmarks for the largest face, or None."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, self.upsample)
        if len(rects) == 0:
            # Second chance: upsample once (helps with small / distant faces).
            rects = self.detector(gray, self.upsample + 1)
            if len(rects) == 0:
                return None
        rect = max(rects, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, rect)
        return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)


def read_frames(path: str, target_fps: float = 25.0) -> List[np.ndarray]:
    """Decode a video to a list of BGR frames, resampled to `target_fps`."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise IOError(f"No frames decoded from: {path}")

    # Nearest-neighbour resample in time when the source isn't already 25 fps.
    if abs(src_fps - target_fps) > 0.1:
        duration = len(frames) / src_fps
        n_out = max(1, int(round(duration * target_fps)))
        idx = np.clip(
            np.round(np.arange(n_out) * src_fps / target_fps).astype(int),
            0, len(frames) - 1,
        )
        frames = [frames[i] for i in idx]

    return frames


@dataclass
class Preprocessed:
    frames: np.ndarray        # (T, 64, 128, 3) uint8 -- mouth crops
    coords: np.ndarray        # (T, 20, 2) float32 -- normalised lip landmarks
    source: List[np.ndarray]  # original BGR frames (for overlay rendering)
    n_missing: int            # frames where no face was found


def preprocess_video(
    path: str,
    detector: LandmarkDetector,
    target_fps: float = 25.0,
) -> Preprocessed:
    source = read_frames(path, target_fps=target_fps)

    crops: List[np.ndarray] = []
    lips: List[np.ndarray] = []
    last_landmarks: Optional[np.ndarray] = None
    n_missing = 0

    for frame in source:
        landmarks = detector(frame)
        if landmarks is None:
            n_missing += 1
            if last_landmarks is None:
                continue  # drop leading frames until we first see a face
            landmarks = last_landmarks
        last_landmarks = landmarks

        matrix = similarity_transform(landmarks[17:], TEMPLATE)
        canvas = cv2.warpAffine(frame, matrix, (ALIGN_SIZE, ALIGN_SIZE))
        crop = canvas[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
        crop = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        crops.append(crop)

        # Lip-coordinate branch: landmarks measured on a 600x500 resize,
        # exactly as the training-set extractor did.
        resized = cv2.resize(frame, (COORD_W, COORD_H))
        lm = detector(resized)
        if lm is None:
            # Fall back to projecting the full-frame landmarks into 600x500.
            sx = COORD_W / frame.shape[1]
            sy = COORD_H / frame.shape[0]
            lm = landmarks * np.array([sx, sy])
        lip = lm[48:68] / np.array([COORD_W, COORD_H])
        lips.append(lip.astype(np.float32))

    if not crops:
        raise RuntimeError(
            f"No face detected in any frame of {os.path.basename(path)}. "
            "The model needs a reasonably frontal, well-lit face."
        )

    return Preprocessed(
        frames=np.stack(crops).astype(np.uint8),
        coords=np.stack(lips).astype(np.float32),
        source=source,
        n_missing=n_missing,
    )
