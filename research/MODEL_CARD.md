# Model Card — AQUA-SHIELD detector + verification

## Detector (primary)
- **Architecture:** YOLO11n (Ultralytics), 2.58 M params, 6.3 GFLOPs, 16 MB. Anchor-free single-stage.
- **Backend:** swappable (`ultralytics` | `torchvision`); nothing downstream imports Ultralytics.
- **Training data:** MILCO/NOMBO SSS, survey-year split (train = 2015+2010, 447 objects).
- **Validation (FP filter + calibration fit):** 2017 survey (30 objects).
- **Test (all reported metrics):** 2018+2021 surveys (191 objects, 473 empty frames).
- **Preprocessing profile:** **`none` (raw)** — recorded in the checkpoint's `.meta.json`. Preprocessing measured to hurt SSS (E06/E07).
- **Classes:** MILCO → man-made/mine-like-object; NOMBO → ambiguous/bottom-object-uncertain (never man-made).
- **Augmentation:** no rotation (range geometry fixed); both flips; hue/sat off (1-channel); mosaic 0.3, scale 0.25; brightness 0.4.

### Measured metrics (held-out surveys, IoU 0.3 for object-level)
| Metric | Value |
|---|---|
| mAP50 | 0.116 |
| mAP50-95 | 0.040 |
| Precision (detector only) | 0.247 |
| **Precision (+ learned FP filter)** | **0.322** |
| Recall | 0.10–0.16 |
| Falsely-alarmed empty frames | 37/473 → **25/473** with FP filter |
| MPS inference | 21 ms/frame · 37 fps end-to-end · 640 MB peak |
| ONNX | 10.6 MB · 8.5 ms |

## Verification stack
- **FP filter:** L2-logistic over 10 physical features + raw score, fitted on 2017; recall-floored threshold; per-detection attribution.
- **Calibration:** Platt, fitted on 2017; ECE reported; falls back to `calibrated:false` when unfit.

## Hardware
- Apple M5, 24 GB, macOS 26.5, MPS; CPU fallback; **no CUDA assumed**.

## Licence
- Code MIT; **combined work AGPL-3.0** via the Ultralytics backend. Licence-clean torchvision path implemented but **not trained**. See `LEGAL_AND_LICENSES.md`.

## Failure modes (measured)
- **Large targets missed** (recall 0.000 >2500 px²) — scale-aug bias.
- **Speckle collapse** on the raw model (0% recall retained); mitigated by speckle-aug training (E08, ~41% retained) at a clean-mAP cost.
- **Out-of-domain silence:** no OOD detector; emits confident-looking numbers on unfamiliar sonar.
- **Two labels only:** any unseen object is forced into MILCO/NOMBO.
- **Anomaly branch not functional** (AUROC ~0.5) — no unknown-object detection shipped.

## Intended use / out of scope
- Intended: research/triage aid for SSS surveys, analyst in the loop.
- Out of scope: autonomous tasking, navigation-grade positions, any claim of Indian-waters or ghost-net validation.
