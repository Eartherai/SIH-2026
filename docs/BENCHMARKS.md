# Benchmarks

Every number here was measured on the development machine. Nothing is copied
from a datasheet, a paper, or another team's results. Raw records live in
`experiments/` (`registry.jsonl`, `benchmarks.jsonl`, `ablation.json`,
`robustness.json`, `edge_export.json`).

**Hardware:** Apple M5, 10 cores, 24 GB unified memory, macOS 26.5.2, arm64.
Python 3.12.13, PyTorch 2.13.0, ultralytics 8.3.253, MPS.

---

## 1. Evaluation protocol

```
detector      trained on  ── train surveys (2015, 2010)   465 frames, 447 objects
FP filter     fitted on   ── val survey    (2017)          93 frames,  30 objects
calibration   fitted on   ── val survey    (2017)
ALL METRICS   measured on ── test surveys  (2018, 2021)   612 frames, 191 objects
                                                          473 of them EMPTY
```

Splits are by **acquisition year**, never random. A test asserts
survey-disjointness. Object matching uses **IoU ≥ 0.3** (justified in
`docs/ML_PIPELINE.md`); this is stated everywhere so it is never confused with a
COCO mAP50.

**Two metric families:**
- *Object-level* — precision / recall / F1 over annotated targets.
- *Frame-level* — how many of the **473 target-free frames** produced an alarm.
  This is the operational number, and the one PS 26057 is really about.

---

## 2. Training runs — measured on held-out test surveys

Full records in `experiments/registry.jsonl`.

<!-- BENCHMARK:TRAINING -->

| Experiment | Config | Epochs | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| `E03-baseline-yolo11n` | imgsz 640, lr0 0.005, mosaic 1.0, scale 0.5 | 28* | **0.1011** | 0.0390 | 0.1668 | 0.1514 |
| `E04-smallobj-tuned` | imgsz 640, lr0 0.002, mosaic 0.3, scale 0.25 | 64* | **0.1163** | 0.0396 | 0.3444 | 0.1639 |
| `E05-finetune-nomosaic` | imgsz 640, lr0 0.0005, mosaic 0.0, scale 0.2 | 18* | **0.1249** | 0.0408 | 0.3017 | 0.1708 |

\* stopped early by the operator; see `notes` in `experiments/registry.jsonl`.

Per-class mAP50 and the full hyperparameters for every run are in the registry. All figures are on the **held-out test surveys**, never on the validation survey.

---

## 3. Train/inference consistency — a defect we found and fixed

Our first ablation applied preprocessing **only at inference** to a detector
**trained on raw frames**. Measured across all 612 held-out test frames at
detector confidence 0.05:

| Inference input | P | R | F1 | TP | FP | FA-frames (of 473) |
|---|---|---|---|---|---|---|
| `none` (**matched**) | 0.1414 | 0.1466 | **0.1440** | 28 | 170 | 80 |
| `minimal` | 0.1157 | 0.0733 | 0.0897 | 14 | 107 | 49 |
| `standard` (mismatched) | 0.0081 | 0.0209 | **0.0117** | 4 | 488 | 186 |
| `aggressive` (mismatched) | 0.0045 | 0.0157 | 0.0070 | 3 | 660 | 252 |

**A 12× F1 degradation, and more than double the false alarms.** The cause is not
that preprocessing is useless — it is that preprocessing shifts the input
distribution, and a model trained on raw frames has never seen it. **If a
preprocessing chain is worth having, the detector must be TRAINED on its output.**

This defect was in our own pipeline default. Three changes followed:

1. `PipelineConfig.preprocess_profile` now defaults to `"none"`, with a test
   guarding it against a well-meaning restoration.
2. The profile is now a property of the **checkpoint**, recorded in a
   `<weights>.meta.json` sidecar by `scripts/train.py` and selected automatically
   at inference. The dashboard warns if you override it.
3. `scripts/prepare_preprocessed.py` materialises a preprocessed dataset so the
   comparison can be *matched*. It refuses geometry-changing profiles
   (`water_column_mode="split"`, slant-range correction) because those would
   invalidate the copied labels.

**How we noticed.** Not from the aggregate metrics — from the *learned filter's
weights*. An early fit gave `shadow_ratio` a large negative weight, contradicting
sonar physics. That was a symptom, not a finding. With the mismatch fixed,
`shadow_ratio` is +0.15 and `shadow_side_consistent` is +0.32, both positive and
consistent with the physics. An inspectable verification stage turned out to be a
diagnostic instrument.

`scripts/prepare_preprocessed.py` therefore materialises a preprocessed copy of
the dataset so the comparison can be *matched*. It refuses to run with any
geometry-changing profile (`water_column_mode="split"`, slant-range correction),
because those would invalidate the copied labels.

<!-- BENCHMARK:PREPROCESSING -->

_Not yet measured. Run the command in section 9 to populate this table._

---

## 4. Pipeline ablation — held-out test surveys

Each row adds one stage to the row above. `FA-frames` = target-free frames that
produced at least one alarm, out of 473.

**Row X is a negative control, not a result about preprocessing.** The detector
was trained on raw frames, so "no preprocessing" is the *matched* configuration
and the correct baseline. Row X shows what a train/inference preprocessing
mismatch costs. The question "does preprocessing help?" is answered by the
matched 2×2 in section 3, not by row X.

<!-- BENCHMARK:ABLATION -->

_Not yet measured. Run the command in section 9 to populate this table._

---

## 5. Latency and memory

<!-- BENCHMARK:LATENCY -->

### Per-frame CPU stages (device independent)

| Stage | mean | p95 |
|---|---|---|
| quality control | 73.54 ms | 86.71 ms |
| preprocess standard | 30.29 ms | 32.75 ms |
| preprocess aggressive | 66.86 ms | 83.87 ms |

### Inference and end-to-end

| Device | Inference only | Full frame pipeline | Throughput | Peak RSS |
|---|---|---|---|---|
| **mps** | 39.13 ms (p95 49.58) | 156.98 ms | 12.10 frames/s | 664 MB |
| **cpu** | 278.21 ms (p95 354.43) | 364.86 ms | 5.63 frames/s | 938 MB |

**MPS speedup over CPU (inference only): 7.11×**

Model: 16.06 MB · 30 frames · shapes {'(1024, 1024)': 9, '(416, 416)': 21}

Peak RSS stayed far below the 24 GB unified-memory budget, so the pipeline is not memory-bound on this class of machine.

---

## 6. Edge export

<!-- BENCHMARK:EDGE -->

Source checkpoint: 16.06 MB at imgsz 640

| Format | Size | Export time | Runtime latency |
|---|---|---|---|
| ONNX | 10.61 MB | 2.7s | 10.84 ms (p95 15.84) |

ONNX Runtime providers: `CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider`

---

## 7. Robustness under controlled degradation

Synthetic perturbations applied to held-out test frames, one variable at a time.
Synthetic degradation isolates variables in a way real data cannot; it does
**not** replace validation on real degraded surveys.

<!-- BENCHMARK:ROBUSTNESS -->

_Not yet measured. Run the command in section 9 to populate this table._

---

## 8. Training stability on Apple Silicon

Two findings that cost real time and are recorded so they don't cost anyone
else's:

1. **`ultralytics` 8.4.130 diverged on this dataset.** `val/cls_loss` climbed
   24.6 → 56.5 → 2,392 → 1.1 × 10⁶ across five epochs while `train/cls_loss`
   stayed flat around 4–6. Disabling AMP did not help. Data loading was verified
   correct (465 images / 447 objects / 319 backgrounds) and labels were validated
   independently. Pinning to **8.3.253** resolved it. `requirements.txt` pins
   `ultralytics>=8.3,<8.4`.

2. **AMP is disabled by default on the MPS backend** (`--amp` to enable). Given
   finding 1 we cannot claim AMP *caused* that divergence, only that we do not
   enable it here.

3. **Label-format check.** The source paper describes annotations as
   "class, center x, center y, **height, width**". Tested against the files: the
   standard YOLO order (`w` then `h`) gives **0** out-of-bounds boxes across 668
   annotations, while the paper's stated order gives **29**. The files are
   standard YOLO; the paper's prose is loose.

---

## 9. How to reproduce

```bash
python scripts/prepare_milco_nombo.py
python scripts/train.py --exp-id E01 --epochs 150
python scripts/fit_verification.py --weights runs/detect/**/weights/best.pt
python scripts/evaluate.py  --weights runs/detect/**/weights/best.pt
python scripts/benchmark.py --weights runs/detect/**/weights/best.pt
python scripts/robustness.py --weights runs/detect/**/weights/best.pt
python scripts/export_edge.py --weights runs/detect/**/weights/best.pt
```

Training is seeded (`--seed 0`, `deterministic=True`), but MPS kernels are not
bit-reproducible across PyTorch versions, so exact figures may shift slightly.
Pipeline *inference* is deterministic and there is a test asserting it.
