# Lip Reader

Sentence-level lip reading from silent video. Feed it a clip of someone talking
with no audio and it predicts what they said, character by character.

```
$ python lipread.py samples/bbaf2n.mpg
bbaf2n.mpg: BIN BLUE AT F TWO NOW
  (75 frames, 4.1s)
```

Runs on CPU in about the length of the clip itself. No GPU required.

## How it works

```
video ──> face landmarks ──┬──> affine-align to canonical face ──> mouth crop ──> 3D CNN ──> BiGRU ──┐
          (dlib, 68 pts)   │         (256×256 template)           (128×64)                           ├──> CTC ──> text
                           └──> 20 lip points, normalised ─────────────────────> BiGRU ──────────────┘
```

Two things make this work better than a naive "crop a box around the mouth"
approach:

**Geometric normalisation.** Every frame is warped onto a canonical face
template using a least-squares similarity transform fitted to 51 facial
landmarks. Head tilt, distance from camera and small pose changes are removed
before the network ever sees the mouth, so the model isn't spending capacity
learning to be pose-invariant.

**A second, explicit geometry stream.** Alongside the pixel crops, the raw 20
lip landmark coordinates are fed through their own bidirectional GRU. Pixels
carry texture (teeth, tongue, shadow); coordinates carry unambiguous shape.
Fusing both is worth a few points of word error rate over pixels alone.

The output layer emits 28 classes (blank + 26 letters + space) per frame and
CTC collapses the frame-level sequence into text, so there's no need for
frame-level alignment labels at training time.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python setup_assets.py          # downloads the two model files into ./assets
python lipread.py samples/bbaf2n.mpg
```

`setup_assets.py` pulls a 26 MB checkpoint and dlib's 63 MB landmark predictor.
They're gitignored, so a fresh clone is small.

## Usage

```bash
# one clip
python lipread.py clip.mp4

# a whole folder, with beam search instead of greedy decoding
python lipread.py "samples/*.mpg" --decoder beam

# burn the prediction into a video, and dump the aligned mouth crops
python lipread.py clip.mp4 --annotate out.mp4 --mouth-strip mouths.png

# score against GRID ground truth (recovered from the filenames)
python evaluate.py samples/
```

Useful flags:

| flag | what it does |
| --- | --- |
| `--decoder beam` | CTC prefix beam search instead of best-path. Slower, slightly better on unfamiliar speakers. |
| `--grid-grammar` | Snap the output onto GRID's six-slot sentence grammar. Only meaningful for GRID-style speech. |
| `--annotate OUT.mp4` | Re-render the clip with the prediction as a subtitle. |
| `--mouth-strip OUT.png` | Contact sheet of the aligned crops — the fastest way to debug preprocessing. |
| `--device cuda` | Use a GPU if you have one. |

As a library:

```python
from lipreader import LipReader

reader = LipReader()                       # loads once, reuse across clips
print(reader.read("clip.mp4").text)
```

## Results

Measured on the ten GRID clips in `samples/`, greedy decoding, CPU:

| metric | value |
| --- | --- |
| word error rate | 5.0% |
| character error rate | 1.7% |
| exact sentence match | 7 / 10 |

All three misses are single-word substitutions between visually identical
mouth shapes — `S`→`N`, `E`→`F`, `IN`→`AT`. These are *viseme collisions*:
the phonemes differ acoustically but the lips do the same thing, so no amount
of image resolution fixes them. It's the fundamental ceiling on silent lip
reading, and the reason human expert lipreaders average only 30–45% word
accuracy on unconstrained speech.

## What it can and can't do

The checkpoint was trained on the **GRID corpus**: 34 speakers, front-facing,
evenly lit, reading six-word sentences from a fixed grammar
(`command colour preposition letter digit adverb`, e.g. "PLACE WHITE IN J
THREE PLEASE"). Expect it to work well on footage that looks like that.

It will do considerably worse on unconstrained speech, off-axis faces, poor
lighting, heavy beards, or fast conversational speed. That's a property of the
training data, not a bug in the pipeline — the preprocessing and inference code
here are vocabulary-agnostic. If you want open-vocabulary lip reading, swap in
a model trained on LRS2/LRS3 (Auto-AVSR, AV-HuBERT) behind the same interface.

## Layout

```
lipread.py            CLI: video in, sentence out
evaluate.py           batch scoring against GRID ground truth
setup_assets.py       fetches the model files
lipreader/
  preprocess.py       face alignment, mouth cropping, lip coordinates
  model.py            LipCoordNet architecture
  decoding.py         CTC greedy + beam decode, GRID grammar, WER/CER
  pipeline.py         glue, plus the annotate/mouth-strip visualisers
samples/              ten GRID clips (ground truth is in the filename)
```

### A note on a fixed upstream bug

Training normalises the lip coordinates by a fixed 600×500 (the size each frame
was resized to before landmark extraction). Upstream's `inference.py` instead
divides by the raw video resolution — 360×288 for GRID — which feeds the
coordinate GRU inputs roughly 1.7× larger than anything it saw during training.
`preprocess.py` normalises by 600×500 to match. See `NOTE_ON_COORD_NORMALISATION`
in that file.

## Credits

- Model architecture and pretrained weights: [LipCoordNet](https://github.com/ffeew/LipCoordNet) by ffeew
- Built on [LipNet](https://arxiv.org/abs/1611.01599) (Assael, Shillingford, Whiteson, de Freitas, 2016)
- Sample clips: [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/) (Cooke et al., 2006), via [rizkiarm/LipNet](https://github.com/rizkiarm/LipNet)
- Face landmarks: [dlib](http://dlib.net/)

## Ethics

Reading speech from video without consent is surveillance. Please use this on
your own footage, on public research datasets, or with the explicit permission
of the people in frame.


## Data

> The raw inputs are not in this repository. [DATA.md](DATA.md) records exactly which files were used, with sizes and fingerprints, and how to re-fetch them.