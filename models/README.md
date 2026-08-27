# Models

Checkpoints and fitted verification artefacts live here. Large binaries are
git-ignored — train them, don't commit them.

| File | What it is | Produced by |
|---|---|---|
| `*.pt` | Detector checkpoint | `scripts/train.py` |
| `fp_filter_<tag>.json` | Fitted false-positive filter (weights are human-readable) | `scripts/fit_verification.py` |
| `calibration_<tag>.json` | Fitted Platt calibration | `scripts/fit_verification.py` |
| `verification_fit_<tag>.json` | Full fit record incl. reliability curves | `scripts/fit_verification.py` |
| `*.onnx` | Edge export | `scripts/export_edge.py` |

## Licence warning

Checkpoints trained through the `ultralytics` backend are fine-tuned from
COCO-pretrained `yolo11n.pt`, which Ultralytics distributes under **AGPL-3.0**.
Our checkpoints inherit that. See `../LEGAL_AND_LICENSES.md`.

## Behaviour when these are missing

- **No `.pt`** — the dashboard and the API refuse to start and tell you to train
  one. AQUA-SHIELD never runs on a placeholder or simulated detector.
- **No `fp_filter_*.json`** — falls back to `RuleBasedFilter`, which is explicitly
  labelled a heuristic and reports the rule that fired for every rejection.
- **No `calibration_*.json`** — falls back to `IdentityCalibrator`; every hazard is
  stamped `calibrated: false` and the reports say the number is a raw detector
  score, not a probability.

The fitted filter and calibrator are both fitted **only** on the validation
survey, never on the test surveys.
