#!/usr/bin/env python3
"""
Score the pipeline on GRID clips. Ground truth is recovered from the filename
(GRID encodes the sentence in it, e.g. bbaf2n -> BIN BLUE AT F TWO NOW).

    python evaluate.py samples/*.mpg
    python evaluate.py samples/ --decoder beam --grid-grammar
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from lipreader.decoding import cer, sentence_from_grid_filename, wer
from lipreader.pipeline import DEFAULT_PREDICTOR, DEFAULT_WEIGHTS, LipReader


def collect(patterns):
    paths = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            for ext in ("mpg", "mp4", "avi", "mov"):
                paths.extend(sorted(glob.glob(os.path.join(pattern, f"*.{ext}"))))
        else:
            hits = sorted(glob.glob(pattern))
            paths.extend(hits if hits else [pattern])
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--predictor", default=DEFAULT_PREDICTOR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--decoder", default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam-size", type=int, default=25)
    ap.add_argument("--grid-grammar", action="store_true")
    args = ap.parse_args()

    paths = collect(args.videos)
    if not paths:
        print("error: no videos found", file=sys.stderr)
        return 1

    reader = LipReader(
        weights=args.weights,
        predictor=args.predictor,
        device=args.device,
        decoder=args.decoder,
        beam_size=args.beam_size,
        grid_grammar=args.grid_grammar,
    )

    wers, cers, exact, scored = [], [], 0, 0
    print(f"{'clip':<12} {'truth':<32} {'prediction':<32} {'WER':>6}")
    print("-" * 86)

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        truth = sentence_from_grid_filename(stem)
        pred = reader.read(path)

        if truth is None:
            print(f"{stem:<12} {'(unknown)':<32} {pred.text:<32} {'-':>6}")
            continue

        w, c = wer(pred.text, truth), cer(pred.text, truth)
        wers.append(w)
        cers.append(c)
        scored += 1
        exact += int(pred.text == truth)
        print(f"{stem:<12} {truth:<32} {pred.text:<32} {w:>6.2f}")

    if scored:
        print("-" * 86)
        print(f"clips: {scored}   "
              f"WER: {sum(wers) / scored:.3f}   "
              f"CER: {sum(cers) / scored:.3f}   "
              f"exact sentence match: {exact}/{scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
