#!/usr/bin/env python3
"""
Read lips from a silent video.

    python lipread.py samples/bbaf2n.mpg
    python lipread.py clip.mp4 --decoder beam --annotate out.mp4
    python lipread.py samples/*.mpg --grid-grammar
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from lipreader.pipeline import (
    DEFAULT_PREDICTOR,
    DEFAULT_WEIGHTS,
    LipReader,
    write_annotated_video,
    write_mouth_strip,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Predict the sentence spoken in a silent video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("videos", nargs="+", help="video file(s) or glob pattern(s)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--predictor", default=DEFAULT_PREDICTOR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--decoder", default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam-size", type=int, default=25)
    ap.add_argument(
        "--grid-grammar",
        action="store_true",
        help="snap output onto the 6-slot GRID sentence grammar",
    )
    ap.add_argument("--annotate", metavar="OUT.mp4",
                    help="write the clip with the prediction burned in")
    ap.add_argument("--mouth-strip", metavar="OUT.png",
                    help="write a contact sheet of the aligned mouth crops")
    args = ap.parse_args()

    paths = []
    for pattern in args.videos:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"error: no such file: {missing[0]}", file=sys.stderr)
        return 1

    reader = LipReader(
        weights=args.weights,
        predictor=args.predictor,
        device=args.device,
        decoder=args.decoder,
        beam_size=args.beam_size,
        grid_grammar=args.grid_grammar,
    )

    want_visuals = bool(args.annotate or args.mouth_strip)
    for path in paths:
        pred = reader.read(path, keep_frames=want_visuals)
        name = os.path.basename(path)
        print(f"{name}: {pred.text}")
        detail = f"  ({pred.n_frames} frames, {pred.seconds:.1f}s"
        if pred.n_missing_faces:
            detail += f", {pred.n_missing_faces} frames without a face"
        if args.grid_grammar and pred.raw_text != pred.text:
            detail += f", raw: '{pred.raw_text}'"
        print(detail + ")")

        if args.mouth_strip:
            out = args.mouth_strip if len(paths) == 1 else (
                f"{os.path.splitext(args.mouth_strip)[0]}_{os.path.splitext(name)[0]}.png"
            )
            write_mouth_strip(pred.preprocessed, out)
            print(f"  mouth strip -> {out}")

        if args.annotate:
            out = args.annotate if len(paths) == 1 else (
                f"{os.path.splitext(args.annotate)[0]}_{os.path.splitext(name)[0]}.mp4"
            )
            write_annotated_video(pred.preprocessed, pred.text, out)
            print(f"  annotated video -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
