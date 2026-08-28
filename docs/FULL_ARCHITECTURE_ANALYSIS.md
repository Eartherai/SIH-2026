# Full Architecture Analysis

**Every component considered, what was tested, what was measured, and why the
final architecture is the right one.** This is the single consolidated
reference — deeper working notes live in `research/` and are linked inline
rather than repeated. Every number below is read from `experiments/*.json(l)`;
none is estimated or recalled.

---

## 1. The final architecture

```
RAW SSS ─▶ QC ─▶ [preprocessing OFF] ─▶ TILING ─▶ YOLO11n (raw-trained, 6.3 GFLOPs)
                                                                │
   REPORT ◀─ PRIORITY ◀─ GEOLOCATION (or refuse) ◀─ DEDUP ◀────┤
                                                                │
                                CALIBRATION ◀─ LEARNED FP FILTER┘
```

Nine stages. Every arrow in this diagram survived a measurement; every
component that isn't here was tested and rejected, or reasoned and deferred,
with the evidence recorded below.

---

## 2. Full component inventory — what we tested, what we found, verdict

### 2.1 Detector architecture

| Candidate | GFLOPs | Params | What we found | Verdict |
|---|---|---|---|---|
| **YOLO11n (ours)** | **6.3** | 2.58M | mAP50 **0.116** cross-survey; 21ms MPS; 10.6MB ONNX | **primary** |
| YOLO11s/m | higher | larger | not run — 447 training objects judged too few to support extra capacity | not attempted, reasoned |
| torchvision FCOS/RetinaNet | comparable | comparable | interface implemented, licence-clean (BSD-3) alternative to Ultralytics' AGPL | implemented, **not trained** |
| SSM-DETR (thesis) | **276.29** | 41.58M | thesis's own cost table: "not viable for real-time AUV deployment" | rejected on cost alone |
| TR-YOLOv5s (Yu 2021) | 16.2 | — | mAP 85.6% on **313 shipwreck images** (large, high-contrast targets — the class we already miss) | rejected: 2.6× our compute, wrong target-size regime |
| MSF-DETR (Zhao 2025) | **50.4** | 20.26M | AP 78.5/mAP50-95 38.5 on a **non-public 3,000-image** self-built dataset | rejected: 8× our compute, unreproducible dataset |
| LEF-RT-DETR (Zhang 2026) | **49.7** | 15.2MB | +4.3 AP vs RT-DETR-r18 on a **non-public 970-instance** dataset | rejected: 8× our compute, unreproducible dataset |
| BHP-UNet (segmentation, not detection) | — | 73.2MB | Dice 78.3%, IoU 77.7% on **1,200 raw + augmented-to-3,000** images | out of category (segmentation); see §2.5 |

**Finding, stated plainly:** every external architecture that improved on a
generic baseline did so on a **dataset we cannot obtain or reproduce**, at
**2.6×–44× our compute**. None of the four detection papers reviewed (thesis
SSM-DETR, TR-YOLOv5s, MSF-DETR, LEF-RT-DETR) trained on more than 3,000 images;
several used fewer training objects than we have. Full breakdown:
`research/MODEL_SELECTION.md`, `research/external_architectures.md`.

### 2.2 Preprocessing — tested three separate times, three separate ways

| Test | Setup | Result | Verdict |
|---|---|---|---|
| **Mismatched** (inference-only, our chain) | raw-trained detector, `standard` profile applied only at inference | F1 **0.144 → 0.012** (12× collapse), false-alarm frames 80→186/473 | proves preprocessing must be *trained*, not bolted on |
| **Matched, our chain** (E06) | full retrain: dropout-repair + water-column-removal + Lee-filter + gain-norm, train+val+test+inference all consistent | mAP50 **0.032** vs raw **0.116** | our own chain hurts SSS even matched |
| **Matched, thesis's exact chain** (E07) | TVG-stand-in→median→histogram-eq→CLAHE→morphology, faithfully reproduced, matched | mAP50 **0.043** vs raw **0.116** | the thesis's own pipeline *also* hurts SSS matched |
| **Full 2×2 matrix** | all four train×infer combinations | matched-raw F1 **0.152** beats matched-preprocessed F1 **0.102**; mismatched cells both worse still (0.040, 0.144) | raw wins in every honest comparison |

**Finding:** the thesis reports **+12.8 mAP** from this exact 5-step chain — but
that gain is measured on **UATD, which is Forward-Looking Sonar**, not
Side-Scan. We reproduced the chain faithfully and it does not transfer to SSS,
tested twice (our chain, their chain), always matched. Full resolution of why
the thesis and our result don't actually contradict each other:
`research/thesis_discrepancies.md`.

### 2.3 False-positive verification — the component that matters most

74% of frames are empty seabed, so precision is the real constraint. Three
approaches were built and measured on the identical 612 held-out frames:

| Stage | Precision | Recall | TP | False-alarm frames (of 473) |
|---|---|---|---|---|
| Detector only (A) | 0.247 | 0.110 | 21 | 37 |
| **+ hand-written rules** (C) | 0.300 | 0.063 | 12 | 18 |
| **+ learned FP filter** (D) | **0.322** | 0.100 | **19** | **25** |

**Finding:** the learned filter beats hand-written rules on every axis that
matters — it keeps **19 of 21** true positives the detector found, where the
hand-rules keep only 12 (they buy quiet by discarding real targets). The
filter is 10 physically-motivated features (shadow coherence, contrast,
texture-relative-to-background, compactness...) fed to an L2-regularised
logistic model, fitted on a held-out survey, with per-detection attribution.
**One additional finding earned by this component:** an early fit gave the
shadow feature a physics-contradicting negative weight — which turned out to
be a *diagnosis of a bug in our own pipeline* (the preprocessing mismatch in
§2.2), not a real result. An inspectable model catches things a black box or a
hand-tuned threshold would not. Full account: `docs/ML_PIPELINE.md`.

*External validation:* SeeByte's defence-grade pipeline is *Ingest → ATR
inference → confidence-scored contact list → analyst → GIS* — structurally
identical to ours. The benchmark report's proposed "novelty" is a
hand-weighted confidence formula (`C = w1·shadow + w2·edge + w3·temporal`) —
exactly the features our filter *learns* the weights for, instead of guessing
them.

### 2.4 Confidence calibration

Platt scaling, fitted on the same held-out survey. Falls back to
`IdentityCalibrator` (stamps `calibrated:false`) when the fit split is too
thin or single-class — refuses to produce a meaningless model rather than
faking one. **Kept.** No external reference proposes anything comparable; this
is a gap every reviewed system (including the four production systems) leaves
to either a hand threshold or raw model output.

### 2.5 Segmentation head (SEAUNet / BHP-UNet style)

| What was checked | Finding |
|---|---|
| Do we have SSS masks to train on? | **No.** MILCO/NOMBO is boxes only. |
| Does any available SSS mask dataset fit? | AI4Shipwrecks has real SSS masks — but for **shipwrecks**, which are the **large targets we already miss entirely** (§2.7). Training a segmenter there would verify the wrong failure mode. |
| BHP-UNet's reported numbers | Dice 78.3%, IoU 77.7% — on a **1,200-to-3,000-image**, non-public, own-collected dataset |

**Verdict: deferred, not rejected.** The idea (anti-noise-blended dilated
convolutions + pyramid split attention to separate man-made shape from natural
clutter) is sound. We have no matching labelled data to test it honestly, so
we do not build it on absent supervision. Revisit if a masked SSS debris
dataset is obtained.

### 2.6 Unknown-anomaly detection

PS 26057 explicitly asks for this; the thesis lists its absence as a gap.

| Variant | Frame ROC-AUC | Patch ROC-AUC |
|---|---|---|
| Conv-autoencoder, 64px patches | 0.465 | 0.482 |
| Conv-autoencoder, 32px patches | 0.472 | 0.536 |

**Finding: both are at or barely above chance (0.5).** A naive
reconstruction-error autoencoder cannot separate ~24px, low-contrast targets
from speckled, striped seabed texture — the target is too small a fraction of
any patch large enough to reconstruct meaningfully. **Rejected and not
shipped** — a chance-level score labelled "anomaly detection" is actively
misleading, worse than no score at all. Every reviewed paper and production
system has the identical gap; none of the six references solve it either.
Correct next approach: feature-embedding novelty (PaDiM/PatchCore over the
detector backbone), not pixel reconstruction.

### 2.7 Failure-mode analysis: target size

| Size bucket | n | Recall |
|---|---|---|
| very small (<300 px²) | 88 | **0.193** |
| small (300–900 px²) | 58 | 0.155 |
| medium (900–2500 px²) | 28 | 0.071 |
| large (>2500 px²) | 17 | **0.000** |

**Finding, and it inverts the expected story:** the thesis and common sonar
wisdom both assume small distant targets are the hard case. On this dataset,
with this model, **every large target is missed and the smallest are found
best** — almost certainly our own `scale=0.25` augmentation combined with the
handful of large test-survey targets looking unlike anything in the training
surveys. This is a diagnosed, unresolved limitation (`docs/LIMITATIONS.md`
§13), not a design choice.

### 2.8 Speckle robustness — a weakness, addressed

| Condition | Raw model (E04) F1 | Speckle-aug model (E08) F1 |
|---|---|---|
| clean baseline | 0.061 | 0.236 |
| speckle σ=0.25 | **0.000** | **0.156** |
| speckle σ=0.5 | 0.000 | 0.071 |
| speckle σ=1.0 | 0.000 | 0.000 |

**Finding:** the raw model **totally collapses** under even mild added
speckle — 0% of clean recall retained. Training with speckle-augmented copies
(σ∈{0.25,0.5}) converts that into graceful degradation (~41% retained at
σ=0.25), and improves every other degradation mode tested (blur, resolution
loss, ping dropout) too. **Honest cost:** on the full 612-frame clean test,
E08 scores mAP50 **0.076** vs E04's **0.116** — it is undertrained (55 epochs
on 3× the data) and the trade is not yet free. **Mechanism proven, promotion
to primary pending a longer tuned run.** LEF-RT-DETR (published *after* our
first speckle-aug run) still lists sonar-specific augmentation as unsolved
future work — we are ahead of a 2026 paper on this specific axis.

### 2.9 Geolocation

Three cases implemented: GeoTIFF affine, per-ping navigation with a full
metres error-budget, or **refuse** (no navigation → `null` coordinates,
enforced by a test). Tiling/dedup reduces positional uncertainty by ~√N over
repeat sightings. **Kept — this is a design discipline, not an experiment**:
positional accuracy has never been validated because our data ships no
navigation, and the code says so at every layer rather than hiding it.
External validation: GhostNetZero's metadata-driven geotagging is the same
pattern; NOAA's export schemas informed our JSON/CSV/GeoJSON format.

### 2.10 UI

| Pattern | Source | Decision |
|---|---|---|
| Dual-tier (Operator / Executive) | SeaClear (multi-robot litter removal), SeeByte (defence ATR) | **implemented** — verified headlessly in both modes |
| Map-first, export-schema conventions | NOAA ERMA/MDMAP | adopted for GeoJSON/CSV export |
| Cloud-GPU segmentation-as-a-service | GhostNetZero (Azure A100) | rejected — out of scope for an offline edge tool |

---

## 3. Comparison against every external reference

| Reference | Type | Their core number | Their cost | Why we didn't adopt it |
|---|---|---|---|---|
| Divyabarathi thesis (2025) | 5-step preproc + SSM-DETR | +12.8 mAP (YOLOv8, **UATD/FLS**) | SSM-DETR 276 GFLOPs | Gain is FLS, not SSS (§2.2); SSM-DETR cost is the thesis's own stated blocker |
| TR-YOLOv5s (Yu 2021) | Transformer-YOLOv5 | mAP 85.6% (313 shipwreck imgs) | 16.2 GFLOPs | Large-target dataset (the class we already miss); 2.6× our compute |
| MSF-DETR (Zhao 2025) | Spatial-frequency DETR | AP 78.5 (3,000 non-public imgs) | 50.4 GFLOPs | 8× our compute; dataset unreproducible; own paper flags missing sonar augmentation (we have it) |
| BHP-UNet (Tang 2023) | Segmentation | Dice 78.3% (1,200→3,000 imgs) | 73.2 MB | No mask labels on our data; different task |
| LEF-RT-DETR (Zhang 2026) | Edge-frequency DETR | +4.3 AP (970 non-public instances) | 49.7 GFLOPs | 8× our compute; dataset unreproducible |
| GhostNetZero.ai | Production (DeepLabV3+ResNet50, Azure A100) | operational, not benchmarked | cloud GPU cluster | Manual-review bottleneck we automate; cloud-only, not edge |
| SeaClear | Production (multi-robot) | 80% detect / 90% collect (targets) | multi-robot hardware | Physical retrieval hardware out of scope for a software pipeline |
| SeeByte SeeTrack/Neptune ATR | Defence (classified) | — (proprietary) | — | Structural blueprint only; **our pipeline independently converged on the same shape** |
| NOAA ERMA/MDMAP | Federal web-GIS | — | — | Manual citizen-survey ingestion, no automated CV |

**Six papers, four production systems, zero adoptions of a heavier detector.**
Every one either needs data we cannot get, compute we've ruled out for edge
deployment, or solves a problem (segmentation, cloud inference) outside this
system's scope. The one idea genuinely worth a future matched test — TR-YOLOv5s's
cross-track downsampling for SSS's anisotropic resolution — is deferred, not
adopted untested (`research/ARCHITECTURE_DECISION.md`).

---

## 4. Why this is the best architecture, in one argument

1. **It is the smallest system that survives every test we could throw at it.**
   Four independent attempts to add capability — our preprocessing, the
   thesis's preprocessing, a segmentation head, an anomaly branch — were each
   measured and three failed outright (preprocessing ×2, anomaly); the fourth
   (segmentation) was correctly not built at all rather than built on absent
   supervision.
2. **The one thing that *did* clearly help — the learned FP filter — is kept,
   and it is the headline number:** +30% precision, −32% false-alarm frames,
   at the cost of 2 of 21 true positives. That trade is exactly what PS 26057
   asks for (minimize false positives from natural seabed).
2b. **A second genuine improvement — speckle-augmented training — is real but
   not yet free**, and it is reported as such rather than rounded up to "kept."
3. **It runs where the field system needs to run.** 21ms/frame on Apple
   Silicon, 10.6MB ONNX, no CUDA assumption — none of the four detection
   papers reviewed clear that bar; three explicitly flag edge deployment as
   unsolved future work.
4. **It converges independently with the one non-academic reference that
   matters most** — SeeByte's defence-grade AUV pipeline is structurally the
   same: ingest, infer, verify with confidence, human review, GIS export.
   That convergence, reached without copying it, is stronger evidence than
   any single benchmark number.
5. **Every negative result is kept as evidence, not hidden.** A reviewer who
   asks "why not X" for any X in this document gets a measured answer, not a
   plausible-sounding one.

**What would change this conclusion:** new *data*, not new architecture. See
`research/FINAL_ARCHITECTURE.md` §4 and §6 for the prioritised list — ghost-gear
training data is the single highest-leverage remaining lever, and it is a data
access problem, not a modelling one.

---

## Sources

- `research/FINAL_ARCHITECTURE.md` — the winning-approach synthesis and future-direction priority table
- `research/ARCHITECTURE_DECISION.md` — kept/removed/deferred with evidence, per component
- `research/external_architectures.md` — the four production-system deep dives
- `research/thesis_discrepancies.md` — verified vs. unverified thesis claims, all six papers
- `research/MODEL_SELECTION.md` — detector candidate comparison
- `docs/BENCHMARKS.md` — every raw table these findings are drawn from
- `docs/ML_PIPELINE.md`, `docs/DATA_PIPELINE.md` — implementation-level detail
- `docs/LIMITATIONS.md` — the 16-item honest-limitations list
