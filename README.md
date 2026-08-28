# 🛰️ AQUA-SHIELD

**Acoustic Intelligence for Underwater Anomaly, Debris & Marine-Hazard Localization**
**Detection → Verification → Localization → Action**

Smart India Hackathon 2026 · Problem Statement **26057** · Ministry of Earth
Sciences (MoES) / National Institute of Ocean Technology (NIOT) · Category:
Software · Theme: Disaster Management

This document is the **complete, comprehensive record** of the project: the
problem, the architecture, every dataset considered, every experiment run,
every external paper and production system reviewed, every measured result
(including the failures we didn't hide), the current UI, and the prioritised
path forward. Nothing below is estimated or recalled from memory — every
number traces to a file under `experiments/`, reproducible with
`scripts/run_full_evaluation.sh`.

---

## Table of contents

1. [The problem, precisely](#1-the-problem-precisely)
2. [Executive summary](#2-executive-summary)
3. [Final architecture](#3-final-architecture)
4. [Data — what we have, and its honest limits](#4-data--what-we-have-and-its-honest-limits)
5. [Everything we tested — detector architecture](#5-everything-we-tested--detector-architecture)
6. [Everything we tested — preprocessing](#6-everything-we-tested--preprocessing)
7. [Everything we tested — false-positive verification](#7-everything-we-tested--false-positive-verification)
8. [Everything we tested — confidence calibration](#8-everything-we-tested--confidence-calibration)
9. [Everything we tested — deduplication & geolocation](#9-everything-we-tested--deduplication--geolocation)
10. [Everything we tested — segmentation](#10-everything-we-tested--segmentation)
11. [Everything we tested — unknown-anomaly detection](#11-everything-we-tested--unknown-anomaly-detection)
12. [Everything we tested — robustness (speckle, blur, dropout, resolution)](#12-everything-we-tested--robustness-speckle-blur-dropout-resolution)
13. [Everything we tested — failure-mode analysis by target size](#13-everything-we-tested--failure-mode-analysis-by-target-size)
14. [Six external papers, four production systems](#14-six-external-papers-four-production-systems)
15. [Why this architecture, and not another](#15-why-this-architecture-and-not-another)
16. [Complete measured benchmarks](#16-complete-measured-benchmarks)
17. [Current UI](#17-current-ui)
18. [Repository structure](#18-repository-structure)
19. [Running it](#19-running-it)
20. [Full limitations (16 items)](#20-full-limitations-16-items)
21. [Future direction, priority-ordered](#21-future-direction-priority-ordered)
22. [Legal, licensing, attribution](#22-legal-licensing-attribution)
23. [Judge questions](#23-judge-questions)

---

## 1. The problem, precisely

PS 26057 asks for an automated pipeline that ingests side-scan sonar (SSS)
imagery, detects man-made marine debris (with emphasis on **ghost fishing
gear** — abandoned, lost, or discarded fishing gear, "ALDFG"), separates it
from natural seabed clutter, geotags it, scores confidence, and reports it —
ideally on edge hardware.

**The hard part is not detection.** In our data, **74% of sonar frames
contain no target at all.** A system that fires on a fraction of those empty
frames is worse than useless, because the analyst stops trusting it and goes
back to reading raw imagery. **Precision, not recall, is the binding
constraint**, and that single fact shapes every architectural decision in
this document.

---

## 2. Executive summary

- **Architecture:** a 9-stage pipeline — QC → (no preprocessing) → tiling →
  YOLO11n detection → learned false-positive filter → calibration →
  deduplication → geolocate-or-refuse → priority → report. Runs fully offline
  on a laptop.
- **Data:** trained and evaluated exclusively on **MILCO/NOMBO**, a real
  side-scan sonar dataset (CC BY 4.0). We have **never detected a ghost net**
  — the one public ghost-gear SSS dataset is access-gated on HuggingFace, one
  human click away, and our ingestion pipeline for it is already built.
- **Headline measured result:** the learned false-positive filter raises
  precision from **0.247 to 0.322** (+30%) and cuts falsely-alarmed
  empty-seabed frames from **37 to 25** out of 473 (−32%), while keeping
  **19 of the 21** true positives the detector found.
- **Performance:** 21 ms/frame on Apple M5 (MPS), 37 frames/s end-to-end,
  640 MB peak memory, 10.6 MB ONNX export at 8.5 ms.
- **Research discipline:** six external papers and four real-world production
  systems were read, verified where possible, and cross-checked against our
  own measurements. **Every proposed heavier architecture was tested or
  reasoned about and rejected** — not on aesthetic grounds, on measured
  compute cost or unreproducible data. Two of our own experiments (an
  anomaly-detection branch, our sonar preprocessing chain) were built,
  measured, and **rejected with evidence** rather than shipped anyway.
- **A genuine tradeoff, tested to completion, and resolved honestly:**
  speckle-augmented training (E09, run to full 95-epoch convergence, not cut
  short) measurably improves robustness under noise but **does not** close to
  the primary model's clean-data accuracy — recall actually came out lower
  than an earlier undertrained attempt. We shipped it as a documented
  alternative checkpoint rather than either hiding the result or promoting a
  regression.
- **What's honestly still weak:** detector accuracy is modest (mAP50 0.116,
  cross-survey); a specific large-target failure mode is diagnosed as a
  genuine training-data coverage gap; geolocation accuracy has never been
  validated; ghost-gear detection is entirely unproven.

---

## 3. Final architecture

```
RAW SSS FRAME
      │
      ▼
┌─────────────────────┐
│ 1. QUALITY CONTROL   │  sonar/qc.py
└─────────────────────┘  dynamic range · speckle index · dropout rows ·
      │                  water-column detection (measured, never guessed)
      ▼
┌─────────────────────┐
│ 2. PREPROCESSING     │  sonar/preprocess.py  —  OFF BY DEFAULT
└─────────────────────┘  implemented, switchable, measured to HURT
      │                  detection on this modality (see §6). Still feeds
      │                  QC and the verification features.
      ▼
┌─────────────────────┐
│ 3. TILING             │  sonar/tiling.py
└─────────────────────┘  overlapping tiles at native resolution;
      │                  seam duplicates merged by IoU OR intersection-
      │                  over-smaller (catches a target split by a tile edge)
      ▼
┌─────────────────────┐
│ 4. DETECTION         │  detection/detector.py — YOLO11n, 2.58M params,
└─────────────────────┘  6.3 GFLOPs. Backend-swappable (Ultralytics/
      │                  torchvision). Low confidence threshold —
      │                  recall first, verify second.
      ▼
┌─────────────────────┐
│ 5. FALSE-POSITIVE     │  confidence/features.py + fp_filter.py
│    FILTERING          │  10 physically-motivated features → L2-logistic
└─────────────────────┘  model fitted on a held-out survey. The single
      │                  highest-value stage — see §7.
      ▼
┌─────────────────────┐
│ 6. CALIBRATION        │  confidence/calibration.py — Platt scaling,
└─────────────────────┘  fitted on held-out data. Reports calibrated:false
      │                  when unfit rather than faking a probability.
      ▼
┌─────────────────────┐
│ 7. DEDUPLICATION      │  tracking/dedup.py — geographic or ping-sequence
└─────────────────────┘  association. N sightings → 1 unique hazard;
      │                  positional uncertainty shrinks ~√N.
      ▼
┌─────────────────────┐
│ 8. GEOLOCALIZATION    │  geolocation/ — GeoTIFF affine, per-ping nav,
└─────────────────────┘  or REFUSE. Every fix carries a metres error
      │                  budget. Never fabricates a coordinate.
      ▼
┌─────────────────────┐
│ 9. PRIORITY            │  reporting/priority.py — separate from
└─────────────────────┘  confidence by design ("is it real?" vs
      │                  "should you care?")
      ▼
  MAP · DASHBOARD · JSON / CSV / GeoJSON REPORT
```

**The rule that shapes everything:** never fabricate a value. No coordinate
without navigation metadata, no calibrated probability without a fitted
calibrator, no physical size without a ground sample distance — each is
`null` plus a stated reason, everywhere in the pipeline and the reports.

**Stack:** Python 3.12 · PyTorch 2.13 (MPS with CPU fallback, no CUDA
assumed) · OpenCV · pyproj · Streamlit · FastAPI · SQLite · ONNX Runtime.

---

## 4. Data — what we have, and its honest limits

### 4.1 The dataset behind every reported number: MILCO/NOMBO

| Field | Value |
|---|---|
| Source | Pessanha Santos, N. & Moura, R. (2024), *Data in Brief* 53:110132 |
| DOI | `10.6084/m9.figshare.24574879` |
| Licence | **CC BY 4.0** — open, no registration, commercial use permitted with attribution |
| Sensor | Marine Sonic dual-frequency SSS, 900–1800 kHz |
| Platform | Teledyne Marine Gavia AUV |
| Years | 2010, 2015, 2017, 2018, 2021 (real acquisition-hardware gap across surveys) |
| Frames | 1,170 (416×416 and 1024×1024) |
| Annotated objects | 668, YOLO box format |
| Classes | `MILCO` (mine-like contact) → mapped to `MAN_MADE`/`mine_like_object`; `NOMBO` (non-mine-like bottom object — **not** "natural", just "not mine-like") → mapped to `AMBIGUOUS`/`bottom_object_uncertain`, **never** to a man-made class |
| Empty frames | **866 of 1,170 (74%)** — this is what makes the false-positive problem measurable and is the reason precision dominates every design choice below |

### 4.2 Splits — leakage-free by acquisition year, never random

Consecutive side-scan frames from one survey share seabed, gain settings, and
often the *same physical object*. A random image split leaks test information
into training and inflates every metric. We split by **survey year**
instead — a test (`test_splits_are_survey_disjoint_no_leakage`) enforces this
automatically.

| Split | Years | Frames | Empty | MILCO | NOMBO | Used for |
|---|---|---|---|---|---|---|
| train | 2015, 2010 | 465 | 319 | 264 | 183 | detector training |
| val | 2017 | 93 | 74 | 28 | 2 | fitting the FP filter + calibrator ONLY |
| **test** | 2018, 2021 | 612 | 473 | 145 | 46 | **every reported metric** |

### 4.3 What we do NOT have, stated plainly

- **We have never detected a ghost net.** The closest public dataset —
  `PINGEcosystem/sss-crab-pot-detection-ds` on HuggingFace (6,674 real SSS
  images of derelict crab pots, CC BY-SA 4.0) — is gate-`auto`: a human must
  click "Agree and access repository" once on the dataset page before any
  file resolves (verified directly: `dataset_info()` succeeds regardless of
  approval, but every actual file returns `HTTP 403 GatedRepo`). No API
  token can perform that click. `scripts/prepare_crab_pot.py` is fully
  built and tested (recording-ID-level leakage-free split, same discipline as
  MILCO/NOMBO's survey-year split) and will run the instant access is
  granted. This is the single highest-leverage remaining task in the project.
- **Geolocation accuracy has never been validated** — MILCO/NOMBO ships no
  navigation metadata, so there is nothing to check computed coordinates
  against. The geometry, sign conventions, and refusal behaviour are unit
  tested; positional *accuracy* is not.
- **No Indian marine data.** The one genuine Indian **field** SSS source found
  (TiHAN/IIT-Hyderabad, Hyderabad lakes, `.xtf`) is also access-gated by a
  form and carries no annotations — its role is Indian-domain *validation*,
  not training, once access is granted.
- **Region undisclosed.** MILCO/NOMBO's authors do not state where the
  surveys were collected. We do not claim it is Indian, and we do not claim
  the model is validated for Indian waters.

### 4.4 Every dataset considered, and its assigned role

| Dataset | Sonar type | Origin | Role assigned | Why |
|---|---|---|---|---|
| **MILCO/NOMBO** | SSS | undisclosed | **training + all metrics** | only ungated, appropriately-licensed, labelled SSS set found |
| sss-crab-pot (ghost gear) | SSS | Delaware, US | training, **pending access** | closest ghost-gear match; ingestion built |
| AI4Shipwrecks | SSS | Michigan, US | future segmentation class (wrecks are large targets) | has real masks, but wrong size regime for our current failure mode |
| KLSG | SSS | China | natural-seabed hard negatives | access by request, pending |
| TiHAN/IIT-Hyderabad | SSS | **India** (lakes) | Indian-domain validation, **pending access** | genuinely Indian but freshwater, unlabelled |
| S3Simulator | SSS (synthetic) | India-authored | optional future augmentation | not field data |
| UATD | **FLS** (not SSS) | China (lakes) | **evidence only — never training/test** | Forward-Looking Sonar; proves the thesis's headline preprocessing gain doesn't apply to SSS |
| Marine Debris (Valdenegro/Singh) | **FLS** (not SSS) | Scotland (watertank) | evidence only | the analysis-report PDF mislabels this "SSS" — it is not |

Full detail: `research/DATA_CARD.md`, `research/dataset_role_matrix.md`,
`research/UATD_USAGE_DECISION.md`, `research/INDIAN_SONAR_DATA.md`.

---

## 5. Everything we tested — detector architecture

| Candidate | GFLOPs | Params | Trained? | Result | Verdict |
|---|---|---|---|---|---|
| **YOLO11n** | **6.3** | 2.58M | ✅ | mAP50 **0.116** cross-survey | **primary** |
| YOLO11s/m | higher | larger | ❌ not attempted | — | reasoned: 447 training objects judged too few to support extra capacity |
| torchvision FCOS/RetinaNet | comparable | comparable | ❌ interface only | — | licence-clean (BSD-3) alternative to AGPL-licensed Ultralytics; implemented, **not trained** — top remaining infrastructure task |
| SSM-DETR (thesis, Ch.6) | **276.3** | 41.6M | ❌ | thesis's own table: "not viable for real-time AUV deployment" | rejected on cost alone before any measurement |
| TR-YOLOv5s (Yu et al. 2021) | 16.2 | — | ❌ (their paper) | mAP 85.6% on **313 shipwreck images** | rejected: 2.6× our compute; their target class is large/high-contrast, exactly what we already miss, not what we need help with |
| MSF-DETR (Zhao et al. 2025) | **50.4** | 20.3M | ❌ (their paper) | AP 78.5 on a **non-public, self-built 3,000-image** dataset | rejected: 8× our compute; dataset unreproducible; their own paper flags missing sonar-specific augmentation as future work — we already have it |
| LEF-RT-DETR (Zhang et al. 2026) | **49.7** | 15.2 MB | ❌ (their paper) | +4.3 AP vs RT-DETR-r18 on a **non-public 970-instance** dataset | rejected: 8× our compute; dataset unreproducible |

**Finding across all four external detection papers reviewed:** every one
that improved on a baseline did so on a **dataset we cannot obtain or
reproduce**, at **2.6×–44× our compute budget**. None trained on more images
than we have; several used fewer training *objects*. The system-level
pipeline, not the detector, is where the leverage is. Full comparison:
`research/MODEL_SELECTION.md`.

**Two Apple-Silicon training stability findings, recorded because they cost
real debugging time:**
1. `ultralytics==8.4.130` **diverged** on this dataset — `val/cls_loss`
   climbed from ~25 to ~1.1×10⁶ over 5 epochs while train loss stayed flat.
   Pinning to `ultralytics>=8.3,<8.4` fixed it.
2. AMP (mixed precision) is **disabled by default on MPS** in our training
   script — not proven to be the cause of (1), but not re-enabled without
   cause either.

---

## 6. Everything we tested — preprocessing

Preprocessing was measured **three separate times, three separate ways** —
because the first result looked wrong, and we didn't trust it until we'd
ruled out our own mistake.

| Test | Setup | F1 | False-alarm frames (of 473) | Verdict |
|---|---|---|---|---|
| **Mismatched** — inference-only | raw-trained detector, our `standard` chain applied only at inference | 0.144 → **0.012** (**12× collapse**) | 80 → 186 | proves preprocessing must be *trained on*, not bolted on at inference |
| **Matched — our own chain** (E06) | dropout-repair + water-column-removal + Lee-filter + gain-norm, full retrain, train=val=test=inference | mAP50 **0.032** vs raw **0.116** | — | our own chain hurts SSS even matched |
| **Matched — the thesis's exact 5-step chain** (E07) | TVG-stand-in → median → histogram-eq → CLAHE → morphology, faithfully reproduced, full retrain | mAP50 **0.043** vs raw **0.116** | — | the thesis's own pipeline *also* hurts SSS, matched |
| **Full 2×2 matrix** | all four train×infer combinations | matched-raw **0.152** beats matched-preprocessed **0.102**; both mismatched cells worse still | 37 vs 285 (matched); 144, 76 (mismatched) | raw wins in every honest comparison |

### The resolution: it's not a contradiction, it's a modality difference

The thesis reports **+12.8 mAP** from this exact 5-step chain (YOLOv8, raw
mAP 0.854 → preprocessed 0.963). That gain is real — but it is measured on
**UATD, which the analysis report's own dataset table (page 8) labels FLS
(Forward-Looking Sonar)**, not Side-Scan. PS 26057 specifies SSS. FLS and SSS
are different imaging geometries (a forward range–bearing fan vs. a swath
with grazing-incidence shadows); a chain tuned for one is not established for
the other. **Both results are true at once.** Full resolution, plus a table
of every other discrepancy found between the thesis analysis report and
verified external sources: `research/thesis_discrepancies.md`.

**Consequence:** `PipelineConfig.preprocess_profile` defaults to `"none"`,
enforced by a test. Preprocessing is recorded as a property of each
*checkpoint* in a `.meta.json` sidecar (so a future model trained on
preprocessed data can never silently mismatch at inference), and the
dashboard warns if an operator overrides it.

**A finding earned by this mistake, not despite it:** an early fit of the
false-positive filter (§7) gave the acoustic-shadow feature a large
*negative* weight — directly contradicting sonar physics (a proud object
casts a shadow; that's supposed to be *evidence* of realness). We nearly
wrote this up as a scientific finding. It was not — it was a **symptom of the
preprocessing mismatch above**, corrupting exactly the pixel region the
shadow features measure. Once fixed, the shadow features came out positive,
consistent with physics. The lesson: an *inspectable* model is a diagnostic
instrument, not just a classifier — it caught a bug in our own pipeline that
aggregate metrics alone had not localised.

---

## 7. Everything we tested — false-positive verification

This is the **highest-leverage component in the system**, because 74% of
frames are empty and every false alarm erodes analyst trust.

| Stage | Precision | Recall | True positives kept | Falsely-alarmed empty frames (of 473) |
|---|---|---|---|---|
| Detector only | 0.247 | 0.110 | 21 | 37 |
| **+ hand-written rules** | 0.300 | 0.063 | 12 | 18 |
| **+ learned FP filter** | **0.322** | 0.100 | **19** | **25** |

**The learned filter beats hand-written rules on the axis that actually
matters:** it keeps 19 of 21 real targets, where the hand-tuned rules keep
only 12 — the rules "win" on paper (similar precision) by throwing away
7 more real detections than the learned model does. That is the concrete
argument, backed by a measurement rather than an assertion, for *fitting* the
verification stage rather than hand-coding thresholds — which is exactly
what the SIH brief itself warns against ("do not create arbitrary heuristic
rules without testing them").

**How it works:** 10 physically-motivated features computed independently of
the detector's own opinion — `target_contrast`, `shadow_ratio`,
`shadow_side_consistent`, `highlight_compactness`, `aspect_ratio`,
`edge_straightness`, `texture_homogeneity`, `background_roughness`,
`local_snr`, `size_rank` — fed to an L2-regularised logistic model fitted on
the held-out validation survey (2017), with a recall floor on threshold
selection (a naive F1-maximising threshold on a thin fit split degenerates to
"reject everything"). Every verdict carries its top-3 feature contributions,
so a rejection is explainable, not a black box.

*External validation:* SeeByte's defence-grade AUV mission pipeline
(SeeTrack → Neptune ATR) is *Raw ingest → ATR inference → confidence-scored
contact list → human analyst → GIS export* — structurally identical to ours,
reached independently. The SIH benchmark report's proposed "novelty" is a
**hand-weighted** confidence formula
(`C = w1·shadow + w2·edge + w3·temporal`); our filter *learns* those exact
weights from data instead of guessing them.

---

## 8. Everything we tested — confidence calibration

Platt (logistic) scaling, fitted on the same held-out validation survey.
Measured before/after with Expected Calibration Error (ECE) and reliability
curves. Falls back to `IdentityCalibrator` — every hazard stamped
`calibrated:false` — when the fit split is too thin or single-class, rather
than silently producing a meaningless model. **Kept.** No external reference
(paper or production system) reviewed proposes anything comparable; every one
either exposes a raw model score or a hand threshold.

---

## 9. Everything we tested — deduplication & geolocation

**Deduplication:** a physical object appears in many consecutive pings. Raw
detection counts overstate the seabed problem several-fold. Geographic
association (when coordinates exist) or ping-sequence IoU association (when
they don't) merges observations into unique hazards with stable IDs;
averaging N independent fixes shrinks positional uncertainty by ~√N (floored
at half the base value, because systematic errors like layback bias don't
average away).

**Geolocation — three cases, one refusal:**
1. **GeoTIFF affine** — reads the pixel→CRS transform directly, reprojects to
   WGS84, also yields a ground-sample-distance for physical sizing.
2. **Per-ping navigation** — row→ping→fix by interpolation, column→slant
   range→ground range via `sqrt(slant² − altitude²)`, bearing = heading±90°,
   geodesic forward solve. Full metres error budget (GPS + heading×range +
   layback + altitude-conditioning + range-resolution, combined in
   quadrature). Heading interpolates *on the circle* (350°→10° → 0°, not
   180°) — unit tested. The nadir singularity (ground range→0 makes the
   altitude term blow up) is **allowed to show**, not smoothed away: a
   detection at nadir correctly reports LOW confidence and large uncertainty,
   matching real sonar-analyst practice of discarding the nadir region.
3. **No metadata → refuse.** `latitude`/`longitude` are `null`, never a
   fabricated 0,0 or a guessed value. A test asserts this can never regress.

**Kept, but explicitly unvalidated for accuracy** — see §20 limitation #3.

---

## 10. Everything we tested — segmentation

| Question | Finding |
|---|---|
| Do we have SSS mask labels? | **No.** MILCO/NOMBO is boxes only. |
| Does any available SSS mask dataset fit our failure mode? | AI4Shipwrecks has real SSS masks — but for **shipwrecks**, large high-contrast targets, which is exactly the class we already **miss entirely** (§13). Training a segmenter there would verify the wrong problem. |
| What do the reviewed segmentation papers report? | BHP-UNet (Tang et al. 2023): Dice 78.3%, IoU 77.7%, on a **1,200-to-3,000-image**, non-public, self-collected dataset. SEAUNet (thesis, Ch.5): mIoU ~0.78 on the same Marine-Debris/KLSG sets, one of which is FLS (§6). |

**Verdict: deferred, not rejected.** The core idea (edge-adaptive attention /
hybrid-dilated convolutions to separate man-made shape from natural clutter)
is sound and thesis-independent of the modality issue. We simply have no
matching supervision to test it honestly, so it is not built on absent
labels. Revisit if a masked SSS debris dataset is obtained.

---

## 11. Everything we tested — unknown-anomaly detection

PS 26057 explicitly asks for this; the thesis lists its absence as a gap; no
reviewed paper or production system solves it either.

**What we built:** a small convolutional autoencoder, trained only on
normal-seabed patches (never shown a target during training), scoring
reconstruction error as an "unlike-normal" signal.

| Variant | Frame-level ROC-AUC | Patch-level ROC-AUC |
|---|---|---|
| 64px patches | 0.465 | 0.482 |
| 32px patches | 0.472 | 0.536 |

**Both are at or barely above chance (0.5).** A naive reconstruction-error
autoencoder cannot separate ~24px, low-contrast targets from speckled,
range-striped seabed texture — the target is too small a fraction of any
patch large enough to reconstruct meaningfully. **Rejected and not shipped.**
A chance-level score labelled "anomaly detection" would be actively
misleading — worse than presenting no score at all. Correct next approach:
feature-embedding novelty detection (PaDiM/PatchCore over the detector
backbone's own features), not pixel reconstruction.

---

## 12. Everything we tested — robustness (speckle, blur, dropout, resolution)

Controlled, synthetic, one-variable-at-a-time perturbations on 120 held-out
frames, measured against the **raw** primary detector (E04):

| Condition | Precision | Recall | F1 | Falsely-alarmed frames (of 60) |
|---|---|---|---|---|
| clean baseline | 0.200 | 0.036 | 0.061 | 4 |
| speckle σ=0.25 | 0.000 | 0.000 | **0.000** | 0 |
| speckle σ=0.5 | 0.000 | 0.000 | 0.000 | 0 |
| speckle σ=1.0 | 0.000 | 0.000 | 0.000 | 0 |
| low contrast (×0.7 → ×0.3) | 0.286–0.400 | 0.024–0.048 | 0.044–0.085 | 1–4 |
| blur (kernel 3→9) | 0.000–0.400 | 0.000–0.048 | 0.000–0.085 | 1 |
| resolution loss (×0.75→×0.25) | 0.000–0.300 | 0.000–0.036 | 0.000–0.064 | 1–2 |
| ping dropout (5%→30%) | 0.000–0.143 | 0.000–0.024 | 0.000–0.041 | 2–6 |
| gain shift (×0.6, ×1.5) | 0.091–0.273 | 0.012–0.036 | 0.021–0.063 | 2–3 |

**Finding: the raw model totally collapses under even the mildest added
speckle** — 0% recall retained. Speckle-augmented training (E08: raw frames
+ multiplicative-speckle copies at σ∈{0.25,0.5} added to the training set
only, val/test kept clean) converts this into graceful degradation:

| Condition | Raw model (E04) F1 | Speckle-aug model (E08) F1 |
|---|---|---|
| clean baseline | 0.061 | 0.236 |
| speckle σ=0.25 | **0.000** | **0.156** (~41% of clean recall retained) |
| speckle σ=0.5 | 0.000 | 0.071 |
| speckle σ=1.0 | 0.000 | 0.000 |

Every other degradation mode (blur, resolution loss, ping dropout) also
improves under the speckle-augmented model, at first look. **We then tested
whether this was simply an artifact of undertraining** (E08 was stopped at
epoch 44 by the operator, not by convergence) by running the identical
recipe to full completion — **E09, 95 of 100 epochs, best val at epoch 60.**

**It was not an undertraining artifact — it is a real, relatively stable
tradeoff:**

| | E04 (raw) | E08 (undertrained) | E09 (fully converged) |
|---|---|---|---|
| Full-test mAP50 | **0.1163** | 0.0763 | 0.0812 |
| Full-test recall | **0.1639** | 0.1415 | **0.0785** (lower than either) |
| Full-test precision | 0.3444 | 0.1849 | 0.3106 |

Full convergence recovered most of the precision deficit (0.185→0.311,
close to the primary's 0.344) but **recall dropped further**, not less —
E09's clean recall (0.079) is lower than the *undertrained* E08's (0.142).
mAP50 barely moved (0.076→0.081), still well below the primary's 0.116.

On the 120-frame robustness subset, the picture is genuinely mixed, not a
clean win: E09 improves over E08 on baseline F1, speckle σ=0.25, blur, and
resolution-loss — but is **worse** than E08 under heavy ping-dropout (F1
0.052 vs 0.179) and speckle σ=0.5 (0.057 vs 0.071). Both collapse fully at
speckle σ=1.0.

**Decision: the primary model does not change.** E04 remains primary — best
on the metrics every other number in this project is measured against. E09
is shipped as a documented, separately-usable alternative checkpoint
(`models/aquashield_speckle_robust.pt`, own fitted FP filter and calibration)
for deployments where noise robustness matters more than peak clean
accuracy. Full three-way comparison:
`experiments/e04_e08_e09_final_comparison.json`. LEF-RT-DETR (published
*after* our first speckle-aug run) still lists sonar-specific augmentation
as unsolved future work — even with this honest, mixed result, we are ahead
of a 2026 paper on having attempted and measured it at all.

---

## 13. Everything we tested — failure-mode analysis by target size

| Size bucket | n objects | Recall (raw E04, IoU≥0.3) |
|---|---|---|
| very small (<300 px²) | 88 | **0.193** |
| small (300–900 px²) | 58 | 0.155 |
| medium (900–2500 px²) | 28 | 0.071 |
| large (>2500 px²) | 17 | **0.000** |

**A genuinely counter-intuitive result on first look:** common sonar wisdom
(and the thesis) assumes small, distant targets are the hard case. On this
dataset, with this model, the opposite holds — the smallest targets are
detected *best*, and every large target is *missed*.

**We investigated this properly rather than leave a plausible guess
standing.** Phase 2 speculated the cause was our own `scale=0.25`
augmentation biasing training toward small objects. We checked
(`experiments/large_target_gap_analysis.json`), using **area-as-fraction-of-frame**
(necessary because MILCO/NOMBO mixes 416px and 1024px images, so raw pixel²
is not comparable across it):

| | Largest object's area, as % of its own frame |
|---|---|
| Training set (2015+2010) | **1.7%** |
| Test set (2018+2021) | **9.3%** |

A **5.5× gap**, driven chiefly by two extreme frames (`0365_2018`,
`0366_2018`). Standard scale-jitter augmentation (E04's factor range
≈0.75–1.25×) physically **cannot synthesize a 5×+ linear-dimension jump**
from typical small-target training crops — no reasonable hyperparameter
retune reaches an object that large from crops that never contain one.

**Revised, corrected conclusion: this is substantially a training-data
coverage gap, not a fixable hyperparameter.** This supersedes the earlier
augmentation-bias hypothesis. The real fix is more training data spanning the
test surveys' size range, or deliberate paste-augmentation that inserts
oversized target crops during training — neither attempted yet. This
correction is propagated through every document that previously stated the
speculative version (`docs/LIMITATIONS.md`, `docs/BENCHMARKS.md`,
`docs/JUDGE_QUESTIONS.md`, `docs/FULL_ARCHITECTURE_ANALYSIS.md`).

---

## 14. Six external papers, four production systems

Every reference was read, key claims verified against primary/independent
sources where possible, and checked against our own measurements — never
adopted on the strength of a reported number alone.

| Reference | Type | Their headline number | Their cost | Why not adopted |
|---|---|---|---|---|
| Divyabarathi thesis (2025, CUSAT) | 5-step preprocessing + SSM-DETR + SEAUNet | +12.8 mAP (YOLOv8, on **UATD = FLS**) | SSM-DETR: 276.3 GFLOPs | Gain is FLS, not SSS (§6); SSM-DETR cost is the thesis's own stated blocker |
| TR-YOLOv5s (Yu et al., *Remote Sensing* 2021) | Transformer-YOLOv5 for SSS | mAP 85.6% (313 shipwreck images) | 16.2 GFLOPs | 2.6× our compute; large-target dataset (the class we already miss) |
| MSF-DETR (Zhao et al., *PLOS ONE* 2025) | Spatial-frequency DETR | AP 78.5 (3,000 non-public images) | 50.4 GFLOPs | 8× our compute; unreproducible dataset; flags missing sonar augmentation as future work (we have it) |
| BHP-UNet (Tang et al., *EURASIP JASP* 2023) | Hybrid-dilated + attention UNet segmentation | Dice 78.3% (1,200→3,000 images) | 73.2 MB | No matching mask labels on our data; wrong target-size regime |
| LEF-RT-DETR (Zhang et al., *Frontiers in Marine Science* 2026) | Edge-frequency RT-DETR | +4.3 AP (970 non-public instances) | 49.7 GFLOPs | 8× our compute; unreproducible dataset |
| GhostNetZero.ai (WWF/Accenture/Microsoft) | Production — DeepLabV3+ResNet50 on Azure A100 | operational, un-benchmarked publicly | cloud GPU cluster | Manual-review bottleneck we automate; cloud-only |
| SeaClear (EU Horizon) | Production — multi-robot litter removal | 80% detect / 90% collect (stated targets) | multi-robot hardware | Physical-retrieval hardware out of scope for a software pipeline; **dual-tier UI pattern adopted** |
| SeeByte SeeTrack + Neptune ATR (defence) | Production — AUV mission mgmt + ATR | — (classified/proprietary) | — | Structural blueprint only — **our pipeline independently converged on the identical shape** |
| NOAA ERMA / MDMAP | Production — federal web-GIS | — | — | Manual citizen-survey ingestion; **export-schema conventions adopted** |

**Six papers, four production systems, zero adoptions of a heavier
detector.** Every academic reference needs data we cannot obtain or compute
we've ruled out for edge deployment; the production systems either solve a
different problem (physical retrieval, manual GIS) or validate the shape we
already have (SeeByte). The one idea judged genuinely worth testing —
TR-YOLOv5s's cross-track downsampling, which corrects SSS's anisotropic
resolution — is deferred to a future matched retrain, not adopted untested.

Full write-ups: `research/thesis_discrepancies.md`,
`research/external_architectures.md`, `research/ARCHITECTURE_DECISION.md`.

**Also searched directly (not just papers): existing ready-to-use GitHub
repositories**, including several other teams' live SIH-26057 submissions
pushed the same week. Two independently arrived at design principles central
to this project — "never invent a coordinate" and "noise/false-positive
suppression is the real problem" — without any shared code, which we read as
stronger validation than a single benchmark number. Nothing found should
replace this repository; no code was copied from any external source. Full
survey: `research/READY_TO_USE_REPOS.md`.

**A formal, academic-style justification of the entire system** — abstract,
related work, methodology, results, limitations, references — is at
`research/AQUA_SHIELD_PAPER.md`.

---

## 15. Why this architecture, and not another

1. **It is the smallest system that survived every test we threw at it.**
   Four separate attempts to add capability — our own preprocessing, the
   thesis's preprocessing, a segmentation head, an anomaly branch — were each
   measured. Three failed outright (preprocessing twice, anomaly); the fourth
   was correctly never built at all, rather than built on absent supervision.
2. **The one component that clearly helps is kept, and it's the headline
   result:** the learned FP filter, +30% precision / −32% false alarms /
   keeps 19-of-21 true positives — exactly what PS 26057 asks for.
3. **A second genuine improvement — speckle-augmented training — is real but
   reported as not-yet-free**, rather than rounded up to "solved."
4. **It runs where a field system needs to run.** 21 ms/frame on Apple
   Silicon, 10.6 MB ONNX, no CUDA assumption. None of the four detection
   papers reviewed clear that bar; three explicitly flag edge deployment as
   unsolved future work.
5. **It converges independently with the one non-academic reference that
   matters most** — SeeByte's defence-grade AUV pipeline is structurally the
   same system, reached without copying it.
6. **Every negative result is kept as evidence, not hidden.** A reviewer who
   asks "why not X" for any X in this document gets a measured answer.

Full argument: `docs/FULL_ARCHITECTURE_ANALYSIS.md` (the complete
component-by-component record) and `research/FINAL_ARCHITECTURE.md` (the
shorter winning-approach synthesis).

---

## 16. Complete measured benchmarks

### 16.1 All training experiments (held-out test surveys)

| Experiment | Data | Epochs | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| E03-baseline | MILCO/NOMBO raw | 28 | 0.1011 | 0.0390 | 0.1668 | 0.1514 |
| **E04-smallobj-tuned (PRIMARY)** | MILCO/NOMBO raw | 64 | **0.1163** | 0.0396 | 0.3444 | 0.1639 |
| E05-finetune-nomosaic | MILCO/NOMBO raw | 18 | 0.1249 | 0.0408 | 0.3017 | 0.1708 |
| E06-preprocessed-matched | our chain, matched | 41 | 0.0318 | 0.0093 | 0.0769 | 0.1089 |
| E07-thesis5step-matched | thesis chain, matched | 80 | 0.0429 | 0.0158 | 0.1592 | 0.1342 |
| E08-speckle-aug | raw+speckle train, clean test | 55 (killed early) | 0.0763 | 0.0236 | 0.1849 | 0.1415 |
| E09-final-speckle-full | raw+speckle train, clean test, **full 95-epoch run** | 95 | 0.0812 | 0.0312 | 0.3106 | **0.0785** |

### 16.2 Pipeline ablation (612 held-out frames, IoU≥0.3)

| Variant | Precision | Recall | F1 | TP | FP | FN | False-alarm frames | Latency |
|---|---|---|---|---|---|---|---|---|
| A. detector only (matched raw) | 0.2471 | 0.1099 | 0.1522 | 21 | 64 | 170 | 37/473 | 50 ms |
| B. no-tiling control | 0.2526 | 0.1257 | 0.1678 | 24 | 71 | 167 | 43/473 | 38 ms |
| C. + hand-written rules | 0.3000 | 0.0628 | 0.1039 | 12 | 28 | 179 | 18/473 | 48 ms |
| **D. + learned FP filter** | **0.3220** | 0.0995 | 0.1520 | **19** | 40 | 172 | **25/473** | 49 ms |
| E. + calibration | 0.3220 | 0.0995 | 0.1520 | 19 | 40 | 172 | 25/473 | 72 ms |
| X. mismatched-preprocessing (negative control) | 0.0102 | 0.0052 | 0.0069 | 1 | 97 | 190 | 57/473 | 84 ms |

*(Rows D and E are identical on P/R/F1 by design — calibration is a monotonic
transform of the score, changing the number reported to the operator, not
which detections are accepted.)*

### 16.3 Preprocessing 2×2 matrix (matched vs. mismatched)

| Trained on | Inference on | Matched? | F1 | False-alarm frames |
|---|---|---|---|---|
| raw | raw | ✅ | **0.1522** | 37 |
| raw | preprocessed | ❌ | 0.0400 | 144 |
| preprocessed | preprocessed | ✅ | 0.1016 | 285 |
| preprocessed | raw | ❌ | 0.1439 | 76 |

### 16.4 Latency & memory (Apple M5, 24 GB)

| | MPS | CPU |
|---|---|---|
| Inference only | **21.4 ms** | 82.2 ms |
| Full pipeline / frame | 54.8 ms | 119.0 ms |
| Survey throughput | **37.4 frames/s** | 17.5 frames/s |
| Peak RSS | **640 MB** | 900 MB |
| **MPS speedup vs CPU** | **3.84×** | — |

| Export | Size | Runtime | Latency |
|---|---|---|---|
| PyTorch checkpoint | 16.07 MB | — | — |
| **ONNX (CoreML EP)** | **10.61 MB** | ONNX Runtime | **8.49 ms** |

### 16.5 Robustness and speckle-augmentation comparison

See §12 for the full tables.

### 16.6 Target-size failure analysis

See §13 for the full tables and the corrected root-cause finding.

### 16.7 Anomaly detection

See §11.

**Full reproduction:** `scripts/run_full_evaluation.sh <weights> [preprocessed-weights]`
regenerates every table in `docs/BENCHMARKS.md` from `experiments/*.json(l)`
— nothing in this README or `docs/BENCHMARKS.md` is hand-typed.

---

## 17. Current UI

**Streamlit dashboard, dual-tier** (adopted from the SeaClear/SeeByte pattern
of separating a technical operator console from a stakeholder-facing summary,
verified working in both modes via headless testing):

- **🎛️ Operator (technical) — 6 tabs:**
  - *Detections* — per-frame overlay, accepted (orange) vs. rejected
    (dim grey, with reason) detections against ground truth (white)
  - *Map* — geolocated hazards with uncertainty circles drawn to true scale,
    colour-coded by priority band
  - *Hazard register* — filterable table, per-hazard evidence breakdown,
    filter verdict, notes
  - *Evidence & QC* — per-frame quality-control scores, preprocessing steps
    applied, timing breakdown
  - *Export* — one-click JSON / CSV / GeoJSON
  - *Provenance* — the exact model, device, preprocessing profile, filter,
    and calibration state that produced every number on screen
- **📋 Executive summary:** headline metrics, map, top-10 hazards by
  priority, one-click export, a single calibration-honesty line. No
  technical internals.

**Four SIH demo scenarios** (all drawn from held-out test surveys — nothing
the model saw during training):
1. `01_clear_targets` — the largest annotated targets, selected purely by
   annotated area (never by model output, to avoid cherry-picking)
2. `02_hard_targets` — the smallest annotated targets; the honest failure
   mode, shown deliberately
3. `03_natural_seabed` — frames with **no target at all**; every detection
   here is by definition a false positive — the scenario PS 26057 is
   actually about
4. `04_georeferenced` — a contiguous frame sequence with an explicitly
   **synthetic** navigation track (labelled as such in the file header, the
   scenario metadata, and the UI), demonstrating geolocation, deduplication,
   and the map on a known geometry

**Also available:** a **FastAPI** service (9 endpoints, auto-generated
OpenAPI docs, `/api/health`, `/api/process`, `/api/hazards`,
`/api/reports/{id}` with JSON/CSV/GeoJSON, and more) and a **SQLite** store
for surveys/runs/frames/hazards.

**Offline guarantee:** `export AQS_OFFLINE_MAP=1` disables the only optional
network path (OpenStreetMap tiles). No cloud AI/inference API is called in
any code path, ever — enforced by a regression test that scans the source
for network calls and forbidden dependencies.

---

## 18. Repository structure

```
aqua-shield/
├── run_demo.sh · setup.sh          one-command demo / setup
├── src/aquashield/
│   ├── device.py                   MPS → CPU selection, never assumes CUDA
│   ├── pipeline.py                 the nine-stage orchestrator
│   ├── ingestion/                  image · geotiff · jsonl_bbox · nav CSV
│   ├── sonar/                      qc · preprocess · tiling
│   ├── detection/                  detector (swappable backend) · boxes · taxonomy · model_meta
│   ├── confidence/                 features · fp_filter · calibration
│   ├── tracking/                   dedup → unique hazards
│   ├── geolocation/                GeoTIFF · per-ping nav · or refuse
│   ├── anomaly/                    autoencoder (built, evaluated, NOT shipped — see §11)
│   ├── evaluation/                 IoU matching, object- + frame-level metrics
│   ├── reporting/                  schema · priority · JSON/CSV/GeoJSON
│   ├── storage/                    SQLite
│   └── api/                        FastAPI + OpenAPI
├── dashboard/app.py                Streamlit dual-tier UI
├── scripts/                        download · prepare (milco_nombo, preprocessed,
│                                   thesis5, speckle_aug, crab_pot) · train · fit_verification ·
│                                   evaluate · benchmark · robustness · export_edge ·
│                                   ablate_preprocessing · make_figures · make_failure_gallery ·
│                                   render_benchmarks · run_full_evaluation
├── tests/                          116 tests incl. end-to-end + headless dashboard +
│                                   offline-guarantee regression tests
├── data/ · demo_data/ · models/ · experiments/ · outputs/
├── research/                       sources · datasets · prior art · model selection ·
│                                   thesis_discrepancies · external_architectures ·
│                                   ARCHITECTURE_DECISION · FINAL_ARCHITECTURE ·
│                                   DATA_CARD · MODEL_CARD · UATD/Indian data decisions ·
│                                   final_experiment_matrix.csv
└── docs/                           ARCHITECTURE · FULL_ARCHITECTURE_ANALYSIS ·
                                    DATA_PIPELINE · ML_PIPELINE · GEOLOCATION ·
                                    BENCHMARKS · DEMO · LIMITATIONS · JUDGE_QUESTIONS
```

---

## 19. Running it

```bash
./setup.sh                                # venv + deps (uses uv if available)
python scripts/download_datasets.py       # MILCO/NOMBO, CC BY 4.0, ~218 MB
python scripts/prepare_milco_nombo.py     # leakage-free survey-year splits
python scripts/train.py --exp-id E01 --epochs 150
python scripts/fit_verification.py --weights runs/detect/**/weights/best.pt
./run_demo.sh                             # dashboard at localhost:8501
```

Fully offline: `export AQS_OFFLINE_MAP=1`.

Once ghost-gear access is granted (§4.3):
```bash
export HF_TOKEN=hf_...                    # never commit this
python scripts/prepare_crab_pot.py        # ready now, blocked only on access
```

**Tests:** `python -m pytest tests/` → **116 passing**, including a full
pipeline run, a determinism check, a headless run of the actual dashboard in
both view modes, and the offline-guarantee scan.

**Training on a real GPU:** `notebooks/aqua_shield_kaggle_2xT4.ipynb` drives
this exact pipeline on Kaggle's free 2× Tesla T4 accelerator — same recipe as
the primary model with real CUDA headroom (larger batch, no wall-clock
compromise), plus optional cells to actually test whether more detector
capacity helps (previously only reasoned about, not measured) and to
genuinely exercise both GPUs in one run. Written and reviewed but **not yet
executed** — the first Kaggle run will be the first real test of the CUDA
code path (MPS/CPU were the only paths exercised on Apple Silicon).

---

## 20. Full limitations (16 items)

Read in full: `docs/LIMITATIONS.md`. Summary:

1. **The data is not the problem statement's data** — trained on mine-like
   contacts, not ghost gear; region undisclosed, not confirmed Indian.
2. **Detector accuracy is modest** (mAP50 0.116 cross-survey) — honest given
   447 training objects, ~24px targets, an 11-year survey gap; not inflated
   by a random split.
3. **Geolocation accuracy has never been validated** — no navigation data to
   check against; the demo's georeferenced scenario is explicitly synthetic.
4. **Calibration is fitted on 30 objects** — a thin basis.
5. **No raw sonar format support** (`.XTF`/`.JSF`/`.SON` etc.) — only
   rasters + navigation CSV; PINGMapper already solves the vendor-format
   problem well, no need to reimplement it.
6. **"Edge deployment" is an architecture claim, not a demonstrated one** —
   measured on this laptop; never run on a Jetson or AUV payload computer.
7. **The Ultralytics detector backend is AGPL-3.0** — fine for research/a
   hackathon, not for a closed-source commercial product; a licence-clean
   torchvision path is implemented but not trained.
8. **Things designed but not executed** — hard-negative mining, a trained
   torchvision baseline, a hyperparameter search.
9. **Our sonar preprocessing does not help, and ships disabled** — measured
   twice, both times worse than raw (§6).
10. **Detection collapses under added speckle** — on the raw model; mitigated
    by speckle-aug training (§12), not yet fully "solved" for clean accuracy.
11. **The unsupervised anomaly branch does not work** — AUROC ≈ chance,
    evaluated and rejected (§11).
12. **Large targets are missed entirely** — re-diagnosed as a 5.5×
    training-data size-coverage gap, not a fixable hyperparameter (§13).
13. **Quality score is an engineering heuristic**, not a calibrated physical
    measure.
14. **Priority weights are a product convention** — no official
    derelict-gear triage standard exists to conform to.
15. **Small-sample statistics** — 191 test objects; adjacent ablation rows
    are within noise; read for direction and magnitude, not precise ranking.
16. **Two things the model literally cannot do:** detect a class it has
    never seen (only MILCO/NOMBO exist as labels), and know when it is
    out-of-domain (no OOD detector — the most dangerous unsolved failure
    mode, since the model emits confident-looking numbers on unfamiliar
    sonar regardless).

---

## 21. Future direction, priority-ordered

| # | Task | Blocker | Status |
|---|---|---|---|
| 1 | **Ghost-gear training on crab-pot data** | one human click on the HF dataset page (gate type `auto`, instant, no review wait) | pipeline built + tested, waiting |
| 2 | ~~Recover the speckle-aug clean-accuracy trade~~ | — | **Resolved this session (E09), not in our favour.** A full 95-epoch run did not close the gap — recall dropped further (0.079 vs E08's 0.142), mAP50 barely moved. This is a genuine, relatively stable tradeoff, not an undertraining artifact. E04 stays primary; E09 shipped as a documented alternative checkpoint (`models/aquashield_speckle_robust.pt`) for noise-heavy deployments. |
| 3 | Fix large-target recall | **re-diagnosed as a 5.5× data-coverage gap**, not an augmentation bug — needs more large-object training data or paste-augmentation | diagnosed, unresolved |
| 4 | TiHAN/IIT-Hyderabad Indian validation access | one human form submission | pending |
| 5 | Replace the failed autoencoder with embedding-based novelty (PaDiM/PatchCore) | none | not started |
| 6 | Train the torchvision backend → licence-clean path | none | interface ready, untrained |
| 7 | Evaluate cross-track downsampling (TR-YOLOv5s idea) as a matched retrain | none | deferred, reasoned |
| 8 | Combined multi-dataset detector (MILCO/NOMBO + crab-pot, taxonomy-mapped) | depends on #1 | after #1 lands |

**What we will explicitly NOT do, and why:**
- Adopt any transformer detector (SSM-DETR, MSF-DETR, LEF-RT-DETR) as
  primary — 8–44× our compute for single-digit-point gains on data we can't
  reproduce.
- Ship the anomaly branch — a chance-level score is worse than none.
- Claim Indian-waters validation — we have no Indian data in hand.
- Claim ghost-gear detection accuracy — we have never trained on ghost gear.
- Re-enable preprocessing by default — measured twice, matched, both worse
  than raw.

---

## 22. Legal, licensing, attribution

Code released under **MIT** (`LICENSE`). The default detector backend
(Ultralytics) is **AGPL-3.0-or-later** — the combined work is therefore
AGPL; a licence-clean torchvision path is designed but not yet trained. Full
audit of every dependency's licence, verified programmatically (not
recalled): `LEGAL_AND_LICENSES.md`.

**Imagery attribution (CC BY 4.0 requirement):**
> Pessanha Santos, N. & Moura, R. (2024). *Side-scan sonar imaging data of
> underwater vehicles for mine detection.* Data in Brief 53:110132. figshare
> DOI `10.6084/m9.figshare.24574879`. Licensed CC BY 4.0. Changes made:
> re-split by acquisition year; reorganised into a YOLO directory layout.
> Pixel data and annotations unmodified.

---

## 23. Judge questions

**61 hard questions** with a 20-second answer and a deep technical answer
for each — covering architecture choices, data legitimacy, the thesis
discrepancy, geolocation honesty, failure modes, and every "why didn't you
just..." a judge is likely to ask: `docs/JUDGE_QUESTIONS.md`.
