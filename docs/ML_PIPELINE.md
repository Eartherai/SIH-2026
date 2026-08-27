# ML pipeline

## Detector

Backend-agnostic by design (`detection/detector.py`). Nothing downstream imports
Ultralytics, so the AGPL dependency is swappable
(`LEGAL_AND_LICENSES.md`).

- **ultralytics** — YOLO11n, 2.58 M params, 6.3 GFLOPs, 16 MB. AGPL-3.0.
- **torchvision** — BSD-3 alternative. Implemented and interface-tested;
  **not trained**, so no accuracy claim is made for it.

Grayscale frames are replicated to three channels rather than colour-mapped: a
colour map would invent chromatic structure the acoustic data does not contain.

The detector runs at a **deliberately low confidence threshold** (default 0.10).
The design is *recall first, verify second* — it is the verification stage's job
to remove clutter, not the detector's.

## Training

See `research/MODEL_SELECTION.md` for the sonar-domain augmentation policy
(no rotation; both flips; hue/saturation disabled; mosaic reduced because it
downscales already-tiny targets).

Every run appends a row to `experiments/registry.jsonl` containing the dataset,
split, hyperparameters, augmentation, hardware and **measured** test metrics.
Nothing is written unless the run completed and metrics were computed.

## Verification — the part that matters for PS 26057

74% of frames contain no target, so **precision is the binding constraint**. The
verification stage is where AQUA-SHIELD differs from a detector demo.

### Physical features — `confidence/features.py`

Ten descriptors computed from the pixels around each candidate, *independent of
the detector's opinion*:

`target_contrast`, `shadow_ratio`, `shadow_side_consistent`,
`highlight_compactness`, `aspect_ratio`, `edge_straightness`,
`texture_homogeneity`, `background_roughness`, `local_snr`, `size_rank`

Two are worth explaining:

- **`shadow_ratio` / `shadow_side_consistent`.** An object standing proud of the
  seabed blocks the beam and casts a shadow on the *far-range* side of its
  highlight. Ripples and gravel produce bright returns with no isolated shadow.
  We measure the darkness of the strip on each side and, when the nadir column is
  known, check the darker side is the physically correct one. When nadir is
  unknown we return **0.5 — "unknown"** rather than pretending to know the range
  direction.
- **`texture_homogeneity`** is computed *relative to the surrounding background*,
  not against an absolute threshold, because sonar gain varies per survey.

### The learned filter — and a finding that contradicted us

The brief says: *"Do not create arbitrary heuristic rules without testing them."*
So we did not. `confidence/fp_filter.py` fits an L2-regularised logistic model
over the ten features plus the raw detector score, **on the held-out validation
survey**, with class weighting because false positives vastly outnumber true ones.

Fitting it produced a result that **contradicted our own physical prior**:

> `shadow_ratio` received a large **negative** weight.

We expected shadow evidence to indicate a *real* object. In this dataset it does
the opposite — because the strongest dark strips adjacent to a candidate are
usually the **nadir band**, not an object shadow. The model learned that "dark
strip next to the box" is more often the water column than a target.

This is exactly why the filter is fitted rather than hand-tuned. A hand-written
rule of the form *"require a shadow to accept a detection"* — which is what sonar
textbooks suggest and what we would have written — would have **degraded**
precision on this data. The rule-based fallback (`RuleBasedFilter`) contains that
rule and is explicitly labelled a heuristic, not a validated constant.

Every verdict is explainable: the filter reports the top three feature
contributions to the logit for each detection, and rejections carry their reason
into the report and the UI.

### Calibration — `confidence/calibration.py`

A detector score is not a probability. Platt scaling is fitted on the same
held-out validation survey, and reliability (ECE/MCE) is measured before and
after.

When no calibrator is fitted, the pipeline uses `IdentityCalibrator`, every
hazard is stamped `calibrated: false`, a note is attached, and the dashboard and
JSON disclaimer both say the number is a raw score. **We never pass a raw score
off as a probability.**

Both the filter and the calibrator **refuse to fit** on insufficient or
single-class data, returning `fitted: false` with the reason instead of producing
a meaningless model. Unit tested.

## Evaluation

Strict separation:

```
detector      trained on  ── train surveys (2015, 2010)
FP filter     fitted on   ── val survey    (2017)
calibration   fitted on   ── val survey    (2017)
ALL METRICS   measured on ── test surveys  (2018, 2021)   ← never seen before
```

Two metric families (`evaluation/matching.py`):

- **Object-level** — precision / recall / F1 by greedy IoU matching. We use
  **IoU ≥ 0.3**, not the COCO 0.5, because at ~24 px a 3–4 px annotation offset
  (well inside inter-annotator agreement for sonar) drops IoU below 0.5 for a
  visually perfect detection. The threshold is printed in every result so it is
  never mistaken for a COCO mAP50.
- **Frame-level** — how many *empty* frames produced an alarm. This is the number
  that decides whether an operator keeps using the system, and it is the metric
  PS 26057 is really about.

## Not done

- **Hard-negative mining.** Designed but not run. The loop (train → run on empty
  seabed → collect FPs → add → retrain) is straightforward given the 473 empty
  test frames, but it was not executed, so no benefit is claimed.
- **Domain adaptation** across sonar hardware.
- **Trained torchvision baseline.**
