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

## Two shipped checkpoints (as of the E09 experiment)

| File | Role | Full-test mAP50 | Full-test recall | Robustness |
|---|---|---|---|---|
| `aquashield_primary.pt` | **PRIMARY** — default, used by the dashboard/API/demo | **0.116** | **0.164** | collapses under speckle/blur/dropout (§12 of README) |
| `aquashield_speckle_robust.pt` | Alternative — offered for degraded-sonar deployments | 0.081 | 0.079 | retains recall under speckle σ=0.25, blur, resolution-loss; still collapses at speckle σ≥1.0; worse than primary under heavy ping-dropout |

**Why the primary didn't change.** E09 was a full, non-truncated retrain of
the speckle-augmentation idea, run specifically to test whether the accuracy
deficit seen in the earlier undertrained run (E08) was a training-budget
artifact. It was not: full convergence (95 epochs, best val at 60) still
scores lower on every held-out-test accuracy metric than the primary model,
and — surprisingly — lower *recall* than even the undertrained E08. The
robustness gain is real but uneven (better on some perturbations, worse on
others), so this is a genuine, relatively stable accuracy/robustness
tradeoff, not something a longer run resolves. Full numbers:
`experiments/e04_e08_e09_final_comparison.json`.

Note: `aquashield_speckle_robust.pt` is ~5.5 MB vs the primary's ~16 MB —
Ultralytics saved this checkpoint in fp16 rather than fp32 (same 2.59M
params; verified by loading and checking `dtype`). Not a different or
smaller architecture.
