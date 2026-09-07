#!/usr/bin/env python3
"""
Fetch the two model files this project needs into ./assets/:

  lipcoordnet.pt                              (26 MB)  the trained network
  shape_predictor_68_face_landmarks_GTX.dat   (63 MB)  dlib face landmarks

Both live in the upstream LipCoordNet repository. Run once after cloning:

    python setup_assets.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

REPO = "https://github.com/ffeew/LipCoordNet.git"
RAW = "https://raw.githubusercontent.com/ffeew/LipCoordNet/main"

WEIGHTS_IN_REPO = (
    "pretrain/LipCoordNet_coords_loss_0.025581153109669685"
    "_wer_0.01746208431890914_cer_0.006488426950253695.pt"
)
PREDICTOR_IN_REPO = (
    "lip_coordinate_extraction/shape_predictor_68_face_landmarks_GTX.dat"
)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
TARGETS = {
    WEIGHTS_IN_REPO: "lipcoordnet.pt",
    PREDICTOR_IN_REPO: "shape_predictor_68_face_landmarks_GTX.dat",
}


def _progress(done: int, total: int, label: str) -> None:
    if total <= 0:
        return
    pct = 100 * done / total
    sys.stdout.write(f"\r  {label}: {pct:5.1f}%  ({done / 1e6:.1f} MB)")
    sys.stdout.flush()


def download(url: str, dest: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with open(dest + ".part", "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    _progress(done, total, os.path.basename(dest))
        print()
        os.replace(dest + ".part", dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  direct download failed ({exc})")
        if os.path.exists(dest + ".part"):
            os.remove(dest + ".part")
        return False


def clone_fallback() -> bool:
    """Some networks block raw.githubusercontent; a shallow clone still works."""
    if not shutil.which("git"):
        return False
    print("  falling back to a shallow git clone ...")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO, tmp],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return False
        for src, name in TARGETS.items():
            path = os.path.join(tmp, src)
            if not os.path.exists(path):
                return False
            shutil.copy2(path, os.path.join(ASSETS, name))
    return True


def main() -> int:
    os.makedirs(ASSETS, exist_ok=True)

    missing = {
        src: name for src, name in TARGETS.items()
        if not os.path.exists(os.path.join(ASSETS, name))
    }
    if not missing:
        print("All assets already present in ./assets -- nothing to do.")
        return 0

    print(f"Fetching {len(missing)} file(s) into {ASSETS}")
    failed = []
    for src, name in missing.items():
        url = f"{RAW}/{urllib.parse.quote(src)}"
        if not download(url, os.path.join(ASSETS, name)):
            failed.append(src)

    if failed and not clone_fallback():
        print(
            "\nCould not fetch the assets automatically. Download them by hand from\n"
            f"  {REPO[:-4]}\n"
            "and place them in ./assets as lipcoordnet.pt and "
            "shape_predictor_68_face_landmarks_GTX.dat",
            file=sys.stderr,
        )
        return 1

    print("Done. Try:  python lipread.py samples/bbaf2n.mpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
