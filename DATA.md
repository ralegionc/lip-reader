# Data

This repository does not carry its raw inputs. The files listed below were present locally when the results in the README were produced, and were deleted on 2026-10-12 to reclaim **89.22 MB** of disk. Every one of them is a download from a public upstream source, so nothing unique was lost.

This file exists so that a re-download can be *checked* rather than assumed to match. It records the exact inventory, not just the source.

## What was removed

89.22 MB in 2 files.

| Path | Size | Files |
|---|---|---|
| `assets` | 89.22 MB | 2 |

## Restoring it

`shape_predictor_68_face_landmarks_GTX.dat` is the dlib facial landmark predictor and `lipcoordnet.pt` is a pretrained LipCoordNet checkpoint. Both are third-party downloads, neither was trained here.

## Inventory

`data_manifest.csv` carries every file with its exact byte length and a fingerprint. Files of 32 MiB or less have a full `sha256`. Larger files have `headtail256`: a sha256 over the first mebibyte, the last mebibyte and the exact byte length, which catches a truncated or altered download without rereading gigabytes.

| File | Size | Fingerprint |
|---|---|---|
| `assets/shape_predictor_68_face_landmarks_GTX.dat` | 63.36 MB | `headtail256:d31c411e204728b130e1fb23d12efdcc545bbd86f73170150b5b618970a8e9b3` |
| `assets/lipcoordnet.pt` | 25.86 MB | `sha256:403ed48e2d410dcbd161f9572d0dc7778bf4fa5552daa5a903daf9bca43f56bc` |

A source that is live rather than a fixed release will not reproduce these fingerprints. Where that applies it is said above.

