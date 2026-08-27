# SIH submission slides — content

Six slides. Every figure here is generated from `experiments/` and matches
`docs/BENCHMARKS.md`. If you change a number, change it there first — do not put
an unverified figure on a slide.

---

## Slide 1 — Problem & Solution

**Title:** AQUA-SHIELD — Detection → Verification → Localization → Action

**The problem, stated as an engineer would state it**
- Ghost nets keep killing after they are lost: they entangle marine life, destroy
  reefs, and foul propellers.
- Finding them means a human reading thousands of km of side-scan sonar.
- **The real difficulty is not detection.** In our data **74% of sonar frames
  contain no target at all.** A system that fires on a fraction of those is worse
  than useless — the analyst stops trusting it.
- **Precision, not recall, is the binding constraint.**

**Our solution**
A local-first pipeline that detects candidates, **verifies them against physical
evidence**, calibrates confidence, merges repeat sightings into unique hazards,
geolocates them *when metadata allows*, prioritises them, and exports an
actionable report — offline, on a laptop.

**Visual:** one natural-seabed frame, detector-only output vs after verification.

---

## Slide 2 — Idea & Innovation

**What we do NOT claim.** Detection of debris in sonar is not novel. GhostVision
(JMSE 2025) already does detection + georeferencing for derelict gear. We say so
in our repository.

**What is ours**

1. **A learned false-positive filter over physical evidence.** Ten features
   measured from the pixels — shadow coherence, contrast, compactness, relative
   texture — that are *independent of the detector's opinion*, fitted on a
   held-out survey, with per-detection attribution.

   > **Its weights caught a bug in our own pipeline.** An early fit gave acoustic
   > shadow a large *negative* weight — the opposite of the physics. That was not
   > a discovery about sonar; it was a symptom of applying a preprocessing chain
   > the detector had never been trained on. Fixing that mismatch recovered a
   > **12× F1 improvement** and turned both shadow features positive. An
   > inspectable verification stage is a *diagnostic instrument*, not just a
   > classifier — a hand-tuned threshold would have hidden the same defect.

2. **Confidence that means something.** Platt calibration fitted on held-out data;
   when it isn't fitted, every hazard is stamped `calibrated: false`.

3. **Refusal as a feature.** No navigation metadata → no coordinate. `null`, plus
   the reason. A fabricated latitude exports cleanly to CSV and sends a vessel to
   open water.

4. **Confidence ≠ Priority.** "Is it real?" and "should you care?" are different
   questions.

**Visual:** the mismatched-vs-matched F1 table (0.012 → 0.144), and the learned filter's weight chart.

---

## Slide 3 — Technical Architecture

```
RAW SONAR ─▶ QC ─▶ [PREPROCESSING] ─▶ TILING ─▶ DETECTION
                    (off by default —
                     measured, didn't help)
                                                  │
  REPORT ◀─ PRIORITY ◀─ GEOLOCATION ◀─ DEDUP ◀────┤
                                                  │
                     CALIBRATION ◀─ FP FILTER ◀───┘
```

| Stage | Substance |
|---|---|
| QC | dynamic range, speckle index, dropout rows, water-column detection |
| Preprocessing | dropout repair · water-column removal · **Lee** speckle filter · gain normalisation — implemented, **measured, and disabled by default** because it made detection worse (see slide 6) |
| Tiling | overlapping tiles at native resolution; seam duplicates merged by IoU **or** intersection-over-smaller |
| Detection | YOLO11n, 2.58 M params, backend-swappable |
| Verification | 10 physical features → logistic model fitted on held-out survey |
| Geolocation | GeoTIFF affine · per-ping navigation · **or refuse** |

**Stack:** Python 3.12 · PyTorch 2.13 (**MPS**, CPU fallback, no CUDA assumed) ·
OpenCV · pyproj · Streamlit · FastAPI · SQLite · ONNX

**Visual:** the pipeline diagram with the FP-filter stage highlighted.

---

## Slide 4 — Feasibility & Viability

**It already runs.** Measured on an Apple M5 / 24 GB:

| | Measured |
|---|---|
| MPS inference (tiled frame) | 21.4 ms |
| CPU inference | 82.2 ms |
| **MPS speedup** | **3.8×** |
| Throughput | 37.4 frames/s |
| Peak memory | 640 MB |
| ONNX export | 10.61 MB, 8.5 ms |

**Data legitimacy.** Trained on MILCO/NOMBO — real AUV side-scan, **CC BY 4.0**,
DOI `10.6084/m9.figshare.24574879`. Every dependency licence verified
programmatically.

**Evaluated honestly.** Split by **acquisition year**, never randomly.
Train 2015+2010 → calibrate 2017 → test 2018+2021. A random split leaks, because
consecutive frames share seabed, gain settings and often the same object.

**Ablation (612 held-out frames, 473 of them target-free):**

| | Precision | True positives | Frames falsely alarmed |
|---|---|---|---|
| Detector only | 0.247 | 21 | 37 / 473 |
| + hand-written rules | 0.300 | 12 | 18 / 473 |
| **+ learned FP filter** | **0.322** | 19 | **25 / 473** |

**Visual:** the ablation table, detector-only → full pipeline.

---

## Slide 5 — Impact & Benefits

**Measured prototype outcomes** (not projections):
- Falsely-alarmed frames cut from **37 to 25** of 473 target-free frames
  (−32%), while keeping **19 of the 21** true positives the
  detector found
- Precision **0.247 → 0.322** (+30%)
- Hand-written rules reach similar precision but keep only **12** true
  positives — they buy quiet by discarding targets. That gap is the case for
  *fitting* the filter rather than tuning thresholds.
- Throughput **37.4 frames/s** on a laptop (27 ms/frame), peak memory 640 MB
- Deduplication merges repeat sightings into unique hazards; positional
  uncertainty tightens by ~√N over repeat fixes

**Operational benefit**
- The analyst reads a ranked hazard register instead of raw imagery.
- Output lands in QGIS as GeoJSON — a cleanup vessel can be tasked directly.
- Runs offline on survey hardware; no cloud, no per-image cost, no data leaving
  the ship.

**Stated plainly:** we do **not** claim a percentage of analyst time saved. That
requires a user study we have not run. What we can show is a measured reduction in
false alarms and a measured processing rate.

**Visual:** the map with uncertainty circles drawn to true scale.

---

## Slide 6 — Research & References

**Prior art, and what we take from it**

| Work | Licence | Relationship |
|---|---|---|
| GhostVision (JMSE 14(10):951, 2025) | NOASSERTION | Closest system. Not vendored — licence unresolved. |
| PINGMapper (Earth & Space Science, 2022) | MIT | Pipeline shape and output conventions |
| sidescantools | GPL-3.0 | Studied; not vendored |
| AI4Shipwrecks (arXiv 2401.14546) | MIT | Route to a wreck class later |
| MILCO/NOMBO (Data in Brief 53:110132) | **CC BY 4.0** | **Our training data** |

**Method references:** Lee (1980) adaptive speckle filtering · Platt (1999)
probabilistic outputs · Guo et al. (2017) calibration of modern neural networks.

**What we have NOT done — on the slide, deliberately**
- Never detected a ghost net (dataset access-gated, HTTP 403)
- Geolocation accuracy never validated (our data ships no navigation)
- Never run on a Jetson or an AUV
- Ultralytics backend is AGPL-3.0
- Our own sonar preprocessing chain **did not help** when measured properly, and
  ships disabled (mAP50 0.032 preprocessed vs 0.116 raw)
- Detection collapses entirely under added speckle noise

> Judges trust a team that states its limits. Put this on the slide; don't wait to
> be asked.

**Visual:** the prior-art matrix.
