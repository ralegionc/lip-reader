"""CTC decoding, the GRID grammar, and error metrics."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Class 0 is the CTC blank; classes 1..27 map onto LETTERS.
LETTERS = [
    " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]
BLANK = 0


# --------------------------------------------------------------------------
# Decoders
# --------------------------------------------------------------------------
def ctc_greedy_decode(logits: np.ndarray) -> str:
    """Best-path decode. `logits` is (T, 28)."""
    best = logits.argmax(-1)
    out: List[str] = []
    prev = -1
    for k in best:
        if k != prev and k != BLANK:
            ch = LETTERS[k - 1]
            if not (ch == " " and (not out or out[-1] == " ")):
                out.append(ch)
        prev = k
    return "".join(out).strip()


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _logsumexp(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + np.log1p(np.exp(lo - hi))


def ctc_beam_decode(logits: np.ndarray, beam_size: int = 25) -> str:
    """Prefix beam search over `logits` (T, 28). Slower but usually a little
    better than greedy, especially on out-of-distribution speakers."""
    logp = _log_softmax(logits.astype(np.float64))
    n_classes = logp.shape[1]

    # prefix -> (log prob ending in blank, log prob ending in non-blank)
    beams: Dict[str, Tuple[float, float]] = {"": (0.0, -np.inf)}

    for t in range(logp.shape[0]):
        nxt: Dict[str, Tuple[float, float]] = {}

        def add(prefix: str, pb: float, pnb: float) -> None:
            cb, cnb = nxt.get(prefix, (-np.inf, -np.inf))
            nxt[prefix] = (_logsumexp(cb, pb), _logsumexp(cnb, pnb))

        for prefix, (pb, pnb) in beams.items():
            total = _logsumexp(pb, pnb)
            # Extend with blank -> prefix unchanged.
            add(prefix, total + logp[t, BLANK], -np.inf)

            for k in range(1, n_classes):
                ch = LETTERS[k - 1]
                p = logp[t, k]
                if prefix and prefix[-1] == ch:
                    # Repeat of last char: only extends via a preceding blank.
                    add(prefix, -np.inf, pnb + p)
                    add(prefix + ch, -np.inf, pb + p)
                else:
                    if ch == " " and (not prefix or prefix[-1] == " "):
                        continue  # never emit leading/double spaces
                    add(prefix + ch, -np.inf, total + p)

        beams = dict(
            sorted(
                nxt.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_size]
        )

    best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    return best.strip()


# --------------------------------------------------------------------------
# GRID grammar
#
# Every GRID sentence is exactly six words drawn from fixed slots:
#   command  colour  preposition  letter  digit  adverb
# Snapping a raw prediction onto this grammar cleans up small character errors
# for free. Only sensible when the footage really is GRID-style speech.
# --------------------------------------------------------------------------
GRID_SLOTS: List[List[str]] = [
    ["BIN", "LAY", "PLACE", "SET"],
    ["BLUE", "GREEN", "RED", "WHITE"],
    ["AT", "BY", "IN", "WITH"],
    [c for c in "ABCDEFGHIJKLMNOPQRSTUVXYZ"],          # no 'W'
    ["ZERO", "ONE", "TWO", "THREE", "FOUR",
     "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"],
    ["AGAIN", "NOW", "PLEASE", "SOON"],
]

# Filename encoding used by GRID, e.g. "bbaf2n" -> BIN BLUE AT F TWO NOW
_CMD = {"b": "BIN", "l": "LAY", "p": "PLACE", "s": "SET"}
_COL = {"b": "BLUE", "g": "GREEN", "r": "RED", "w": "WHITE"}
_PRE = {"a": "AT", "b": "BY", "i": "IN", "w": "WITH"}
_DIG = {"z": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR",
        "5": "FIVE", "6": "SIX", "7": "SEVEN", "8": "EIGHT", "9": "NINE"}
_ADV = {"a": "AGAIN", "n": "NOW", "p": "PLEASE", "s": "SOON"}


def sentence_from_grid_filename(stem: str) -> Optional[str]:
    """Recover the ground-truth sentence encoded in a GRID filename."""
    stem = stem.lower()
    if len(stem) != 6:
        return None
    try:
        return " ".join([
            _CMD[stem[0]], _COL[stem[1]], _PRE[stem[2]],
            stem[3].upper(), _DIG[stem[4]], _ADV[stem[5]],
        ])
    except KeyError:
        return None


def _edit_distance(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap_to_grid_grammar(text: str) -> str:
    """Force a prediction onto the nearest valid GRID sentence."""
    words = text.split()
    # Pad or trim to six slots before matching.
    words = (words + [""] * 6)[:6]
    return " ".join(
        min(slot, key=lambda cand: _edit_distance(word, cand))
        for word, slot in zip(words, GRID_SLOTS)
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def wer(prediction: str, truth: str) -> float:
    t = truth.split()
    return _edit_distance(prediction.split(), t) / max(1, len(t))


def cer(prediction: str, truth: str) -> float:
    return _edit_distance(prediction, truth) / max(1, len(truth))
