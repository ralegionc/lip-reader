"""End-to-end: video file in, sentence out."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch

from .decoding import (
    ctc_beam_decode,
    ctc_greedy_decode,
    snap_to_grid_grammar,
)
from .model import LipCoordNet
from .preprocess import LandmarkDetector, Preprocessed, preprocess_video

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")

DEFAULT_WEIGHTS = os.path.join(ASSETS, "lipcoordnet.pt")
DEFAULT_PREDICTOR = os.path.join(ASSETS, "shape_predictor_68_face_landmarks_GTX.dat")


@dataclass
class Prediction:
    text: str
    raw_text: str
    n_frames: int
    n_missing_faces: int
    seconds: float
    logits: np.ndarray = field(repr=False, default=None)
    preprocessed: Optional[Preprocessed] = field(repr=False, default=None)


class LipReader:
    """Loads the network once; call `.read(path)` per video."""

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        predictor: str = DEFAULT_PREDICTOR,
        device: str = "cpu",
        decoder: str = "greedy",
        beam_size: int = 25,
        grid_grammar: bool = False,
    ):
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"Weights not found: {weights}\nRun  python setup_assets.py  first."
            )
        self.device = torch.device(device)
        self.decoder = decoder
        self.beam_size = beam_size
        self.grid_grammar = grid_grammar

        self.model = LipCoordNet()
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.detector = LandmarkDetector(predictor)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def read(self, path: str, keep_frames: bool = False) -> Prediction:
        started = time.time()
        pre = preprocess_video(path, self.detector)

        video = torch.from_numpy(
            pre.frames.astype(np.float32).transpose(3, 0, 1, 2) / 255.0
        ).unsqueeze(0).to(self.device)
        coords = torch.from_numpy(pre.coords).unsqueeze(0).to(self.device)

        logits = self.model(video, coords)[0].cpu().numpy()

        if self.decoder == "beam":
            raw = ctc_beam_decode(logits, beam_size=self.beam_size)
        else:
            raw = ctc_greedy_decode(logits)

        text = snap_to_grid_grammar(raw) if self.grid_grammar else raw

        return Prediction(
            text=text,
            raw_text=raw,
            n_frames=int(pre.frames.shape[0]),
            n_missing_faces=pre.n_missing,
            seconds=time.time() - started,
            logits=logits,
            preprocessed=pre if keep_frames else None,
        )


# ----------------------------------------------------------------------
# Optional visualisations
# ----------------------------------------------------------------------
def write_mouth_strip(pre: Preprocessed, out_path: str, every: int = 5) -> str:
    """Contact sheet of the aligned mouth crops -- the quickest way to see
    whether preprocessing is behaving."""
    picks = pre.frames[::every]
    strip = np.concatenate(list(picks), axis=1)
    cv2.imwrite(out_path, strip)
    return out_path


def write_annotated_video(
    pre: Preprocessed,
    text: str,
    out_path: str,
    fps: float = 25.0,
) -> str:
    """Re-render the source clip with the predicted sentence burned in."""
    frames: List[np.ndarray] = pre.source
    h, w = frames[0].shape[:2]
    scale = max(1.0, 480 / h)
    w, h = int(w * scale), int(h * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 640.0

    for frame in frames:
        img = cv2.resize(frame, (w, h))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
        org = ((w - tw) // 2, h - max(12, int(0.06 * h)))
        cv2.putText(img, text, org, font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, org, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    return out_path
