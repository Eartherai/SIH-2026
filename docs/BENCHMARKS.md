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
| `E06-preprocessed-matched` | imgsz 640, lr0 0.002, mosaic 0.3, scale 0.25 | 41* | **0.0318** | 0.0093 | 0.0769 | 0.1089 |

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

| Detector trained on | Inference input | P | R | F1 | FP | FA-frames |
|---|---|---|---|---|---|---|
| raw | raw | 0.2471 | 0.1099 | 0.1522 | 64 | 37/473 |
| raw | preprocessed | 0.0324 | 0.0524 | 0.0400 | 299 | 144/473 |
| preprocessed | preprocessed | 0.0615 | 0.2932 | 0.1016 | 855 | 285/473 |
| preprocessed | raw | 0.1368 | 0.1518 | 0.1439 | 183 | 76/473 |

**Conclusion.** Comparing only the MATCHED cells, training and inferring on raw imagery gives the better F1 (preprocessed 0.1016 vs raw 0.1522); false-alarm frames 285 vs 37 of 473 empty frames. The MISMATCHED cell (raw-trained detector, preprocessed input) scores F1 0.0400 — the cost of applying a preprocessing chain the detector was never trained on.

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

Model: `aquashield_primary.pt` · device `mps` · 612 frames (139 with targets, 473 empty, 191 objects) · match IoU 0.3

| Variant | P | R | F1 | TP | FP | FN | FA-frames | ms/frame |
|---|---|---|---|---|---|---|---|---|
| A. detector only matched preprocessing | 0.2471 | 0.1099 | 0.1522 | 21 | 64 | 170 | 37/473 | 50 |
| B. no tiling control | 0.2526 | 0.1257 | 0.1678 | 24 | 71 | 167 | 43/473 | 38 |
| C. plus rule based fp filter | 0.3000 | 0.0628 | 0.1039 | 12 | 28 | 179 | 18/473 | 48 |
| D. plus learned fp filter | 0.3220 | 0.0995 | 0.1520 | 19 | 40 | 172 | 25/473 | 49 |
| E. full pipeline calibrated | 0.3220 | 0.0995 | 0.1520 | 19 | 40 | 172 | 25/473 | 72 |
| X. mismatched preprocessing control | 0.0102 | 0.0052 | 0.0069 | 1 | 97 | 190 | 57/473 | 84 |

**How to read this table.**

- The **learned FP filter (row D)** is the headline result: it is the stage that raises precision and cuts false alarms while keeping most true positives.
- **Rows D and E are identical on P/R/F1, and that is expected.** Platt calibration is a monotonic transform of the score; it changes the *number reported to the operator*, not which detections are accepted. Its effect is measured as calibration error (section 2 of `scripts/fit_verification.py` output), not as precision.
- **Row X is a negative control**, not a preprocessing result.

**Read this for direction and magnitude, not for a precise ranking of adjacent rows.** With 191 test objects, differences of a few percent are within noise (`docs/LIMITATIONS.md`, §12).

---

## 5. Latency and memory

<!-- BENCHMARK:LATENCY -->

### Per-frame CPU stages (device independent)

| Stage | mean | p95 |
|---|---|---|
| quality control | 31.70 ms | 31.82 ms |
| preprocess standard | 14.11 ms | 14.37 ms |
| preprocess aggressive | 33.99 ms | 34.38 ms |

### Inference and end-to-end

| Device | Inference only | Full frame pipeline | Throughput | Peak RSS |
|---|---|---|---|---|
| **mps** | 21.40 ms (p95 22.28) | 54.77 ms | 37.38 frames/s | 640 MB |
| **cpu** | 82.24 ms (p95 83.21) | 119.00 ms | 17.51 frames/s | 900 MB |

**MPS speedup over CPU (inference only): 3.84×**

Model: 16.07 MB · 30 frames · shapes {'(1024, 1024)': 9, '(416, 416)': 21}

Peak RSS stayed far below the 24 GB unified-memory budget, so the pipeline is not memory-bound on this class of machine.

---

## 6. Edge export

<!-- BENCHMARK:EDGE -->

Source checkpoint: 16.07 MB at imgsz 640

| Format | Size | Export time | Runtime latency |
|---|---|---|---|
| ONNX | 10.61 MB | 1.0s | 8.49 ms (p95 10.68) |

ONNX Runtime providers: `CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider`

---

## 7. Robustness under controlled degradation

Synthetic perturbations applied to held-out test frames, one variable at a time.
Synthetic degradation isolates variables in a way real data cannot; it does
**not** replace validation on real degraded surveys.

<!-- BENCHMARK:ROBUSTNESS -->

120 held-out frames · match IoU 0.3

| Condition | Level | P | R | F1 | Recall retained | FA-frames |
|---|---|---|---|---|---|---|
| baseline | — | 0.2000 | 0.0357 | 0.0606 | 1.00× | 4/60 |
| speckle | 0.25 | 0.0000 | 0.0000 | 0.0000 | 0.00× | 0/60 |
| speckle | 0.5 | 0.0000 | 0.0000 | 0.0000 | 0.00× | 0/60 |
| speckle | 1.0 | 0.0000 | 0.0000 | 0.0000 | 0.00× | 0/60 |
| low contrast | 0.7 | 0.2667 | 0.0476 | 0.0808 | 1.33× | 4/60 |
| low contrast | 0.5 | 0.4000 | 0.0476 | 0.0851 | 1.33× | 2/60 |
| low contrast | 0.3 | 0.2857 | 0.0238 | 0.0440 | 0.67× | 1/60 |
| blur | 3 | 0.4000 | 0.0476 | 0.0851 | 1.33× | 1/60 |
| blur | 5 | 0.3333 | 0.0238 | 0.0444 | 0.67× | 1/60 |
| blur | 9 | 0.0000 | 0.0000 | 0.0000 | 0.00× | 1/60 |
| resolution loss | 0.75 | 0.3000 | 0.0357 | 0.0638 | 1.00× | 2/60 |
| resolution loss | 0.5 | 0.1667 | 0.0119 | 0.0222 | 0.33× | 1/60 |
| resolution loss | 0.25 | 0.0000 | 0.0000 | 0.0000 | 0.00× | 1/60 |
| ping dropout | 0.05 | 0.1429 | 0.0238 | 0.0408 | 0.67× | 6/60 |
| ping dropout | 0.15 | 0.0000 | 0.0000 | 0.0000 | 0.00× | 2/60 |
| ping dropout | 0.3 | 0.0000 | 0.0000 | 0.0000 | 0.00× | 5/60 |
| gain shift | 0.6 | 0.2727 | 0.0357 | 0.0632 | 1.00× | 2/60 |
| gain shift | 1.5 | 0.0909 | 0.0119 | 0.0211 | 0.33× | 3/60 |

> Perturbations are SYNTHETIC and isolate one variable at a time. They do not replace validation on real degraded surveys.

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
