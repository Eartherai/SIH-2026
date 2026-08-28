# AQUA-SHIELD: A Verification-Centric Pipeline for Marine Anomaly Detection in Side-Scan Sonar Imagery

**A technical report for Smart India Hackathon 2026, Problem Statement 26057**
**Ministry of Earth Sciences (MoES) / National Institute of Ocean Technology (NIOT)**

---

## Abstract

Automated detection of marine debris in side-scan sonar (SSS) imagery is
constrained less by detector accuracy than by false-positive suppression:
in the dataset used throughout this work, 74% of frames contain no target
at all, so precision — not recall — is the operative constraint. We present
AQUA-SHIELD, a nine-stage pipeline (quality control, tiling, detection,
learned false-positive verification, calibration, deduplication,
geolocation-or-refusal, prioritisation, reporting) built around a
lightweight detector (YOLO11n, 6.3 GFLOPs) and evaluated exclusively on
leakage-free, cross-survey held-out data. We report every result obtained
during development, including negative ones: our own and a reproduced
external sonar-preprocessing pipeline both measurably *degrade* detection
on side-scan imagery when trained and evaluated under matched conditions
(mAP50 0.032 and 0.043 respectively, versus 0.116 for the unmodified
detector); a purpose-built unsupervised anomaly-detection branch achieves
ROC-AUC ≈ 0.5 (chance) and is not deployed; and a speckle-augmentation
strategy that repairs a documented total robustness collapse under noise
was found, after training to full convergence, to trade held-out recall for
robustness in a manner that does not resolve with additional training
budget. A learned false-positive filter — ten physically motivated features
fitted with logistic regression on a held-out survey — improves detector
precision from 0.247 to 0.322 (+30%) and reduces falsely-alarmed
empty-seabed frames from 37/473 to 25/473 (−32%) while retaining 19 of 21
true positives, outperforming a hand-tuned rule baseline on every axis that
matters operationally. The system runs at 21 ms/frame (37 frames/second)
on Apple Silicon (MPS) with a 640 MB peak memory footprint and exports to a
10.6 MB, 8.5 ms ONNX model. Six independently published architectures
(a doctoral thesis and four peer-reviewed 2021–2026 papers) and four
production/defence systems were reviewed and cross-checked against these
measurements; none was adopted as primary, each for a documented,
evidence-based reason. We report every limitation without qualification,
including that the system has never been evaluated on ghost fishing gear —
the problem statement's named target class — because the only accessible
public dataset for it remains behind a one-step, human-only access gate at
the time of writing.

---

## 1. Introduction

### 1.1 Problem statement

PS 26057 requires an automated pipeline that ingests side-scan sonar
imagery, detects man-made marine debris — with explicit emphasis on
abandoned, lost, or discarded fishing gear (ALDFG, colloquially "ghost
nets") — separates such debris from natural seabed structure, geotags
detections, produces a calibrated confidence score, and is suitable for
edge deployment.

### 1.2 The actual difficulty

The literature and the naive framing of this task both treat it as an
object-detection accuracy problem. Measured directly on the dataset used in
this work, that framing is wrong: **866 of 1,170 available frames (74%)
contain no annotated target.** A system tuned purely for recall will
therefore alarm on the majority of its operational input, which is the
condition under which field operators stop trusting automated tools and
revert to manual review — the exact failure this problem statement exists
to prevent. We treat false-positive suppression as the primary engineering
target and detection accuracy as necessary but secondary throughout this
work; every architectural decision reported below follows from that
ordering.

### 1.3 Contribution

We do not claim a new detection architecture. Every architecture surveyed
in this work (§4) that improved on a generic baseline did so at 2.6×–44×
our target inference cost, on datasets that are either non-public or
belong to a different sonar imaging modality (forward-looking rather than
side-scan). Our contribution is methodological: (i) a verification stage
*fitted*, not hand-tuned, that is shown to outperform a rule-based
equivalent on the metric that matters (true positives retained per false
alarm avoided); (ii) an evaluation protocol — survey-year and
recording-level splits, matched preprocessing comparisons, robustness
testing under controlled perturbation — that surfaces effects invisible to
a random train/test split, including two results that overturned our own
prior hypotheses (§6.4, §6.7); (iii) an explicit refusal architecture that
declines to emit a value (a geographic coordinate, a calibrated
probability, an anomaly score) it cannot support with evidence, verified
by regression tests; and (iv) a complete accounting of what was tried,
measured, and rejected, on the view that a negative result obtained by
measurement is more valuable to a downstream engineering team than a
positive result obtained by assumption.

---

## 2. Data

### 2.1 Primary dataset

All reported training and evaluation uses **MILCO/NOMBO** (Pessanha
Santos & Moura, 2024, *Data in Brief* 53:110132; figshare DOI
10.6084/m9.figshare.24574879; CC BY 4.0): 1,170 side-scan sonar frames
(416×416 and 1024×1024 px) from a Teledyne Gavia AUV carrying a Marine
Sonic dual-frequency (900–1800 kHz) sonar, spanning five acquisition years
(2010, 2015, 2017, 2018, 2021), with 668 annotated bounding boxes across
two classes: `MILCO` (mine-like contact) and `NOMBO` (non-mine-like bottom
object — an assertion of *ambiguity*, not of naturalness, and mapped
accordingly in our taxonomy to avoid overclaiming what the annotation
actually supports).

### 2.2 Split methodology

Consecutive side-scan frames within one survey are strongly correlated —
shared seabed, shared gain settings, and frequently the same physical
object imaged across adjacent pings. A random frame-level split leaks
test information into training and inflates every downstream metric; we
did not find this leakage risk addressed in any of the six external
detection architectures surveyed (§4), nor in the public materials of the
competing hackathon submissions reviewed in §4.4. We instead split by
**acquisition year**: train = {2015, 2010} (465 frames, 447 objects), val
= {2017} (93 frames, 30 objects, used exclusively to fit the verification
stage), test = {2018, 2021} (612 frames, 191 objects, 473 of them
target-free). Survey-disjointness across splits is enforced by an
automated regression test.

### 2.3 The dataset we do not have

The problem statement's named target class — ALDFG / ghost fishing gear —
is not present in MILCO/NOMBO. The nearest public match,
`PINGEcosystem/sss-crab-pot-detection-ds` on HuggingFace (6,674 real SSS
images, CC BY-SA 4.0), is access-gated: metadata resolves for any
authenticated caller, but every data file returns HTTP 403 with
`x-error-code: GatedRepo` until a human manually approves access via the
dataset's web page (gate type `auto`, i.e. instantaneous upon that one
click, requiring no maintainer review). No API-level credential can
perform this step. A leakage-free ingestion pipeline for this dataset —
splitting by acoustic-recording identifier rather than by image, following
the same discipline as §2.2 — is implemented and unit-tested in advance of
access (`scripts/prepare_crab_pot.py`). We report this rather than
obscure it: **the system described in this paper has never been trained
or evaluated on ghost fishing gear.**

---

## 3. System architecture

```
raw SSS frame
   → quality control (dynamic range, speckle index, dropout-row detection,
                       water-column localisation)
   → [preprocessing — disabled by default; see §6.4]
   → resolution-aware tiling (overlap-merged by IoU or intersection-over-
                               smaller, to recover targets split across a
                               tile boundary)
   → detection (YOLO11n; low confidence threshold — recall-oriented)
   → learned false-positive verification (§5)
   → confidence calibration (Platt scaling, fitted on held-out data;
                              reports its own unfitted state rather than
                              emitting an uncalibrated value silently)
   → spatial/temporal deduplication (repeated sightings of one physical
                                      object merged into one hazard record;
                                      positional uncertainty reduced by
                                      the standard √N averaging factor)
   → geolocation-or-refusal (§7)
   → priority scoring (a policy function of confidence, object class,
                        estimated size, and observation count — explicitly
                        distinct from confidence, which is an evidentiary
                        rather than operational judgement)
   → structured report (JSON / CSV / GeoJSON) and dual-tier operator/
                         executive dashboard
```

Nine stages, each independently switchable and independently evaluated.
The detector backend is abstracted behind a common interface with two
implementations (Ultralytics YOLO; a permissively-licensed torchvision
alternative, implemented but not yet trained — see §9), so that the
detection component can be replaced without touching verification,
calibration, deduplication, geolocation, or reporting.

---

## 4. Related work

### 4.1 Detection architectures

| Reference | Proposed mechanism | Reported gain | Computational cost | Dataset |
|---|---|---|---|---|
| Divyabarathi (2025), doctoral thesis, CUSAT | 5-step signal preprocessing (TVG-analogue → median → histogram equalisation → CLAHE → morphology) ahead of YOLOv8; structure-saliency DETR variant (SSM-DETR) | +12.8 mAP points (YOLOv8, 0.854→0.963) | SSM-DETR: 276.3 GFLOPs | UATD — **forward-looking sonar**, not side-scan |
| Yu et al. (2021), *Remote Sensing* 13(18):3555 | Transformer-augmented YOLOv5 (TR-YOLOv5s), cross-track downsampling for anisotropic SSS resolution, overlapping-patch inference | mAP50 85.6% | 16.2 GFLOPs | 313 shipwreck images (self-collected) |
| Zhao et al. (2025), *PLOS ONE* 20(11):e0336468 | Spatial-frequency dual-branch backbone (Gabor-filter frequency features) with attention-gated feature-pyramid fusion (MSF-DETR) | AP 78.5, +2.3 pt small-object AP from the fusion module specifically | 50.4 GFLOPs | 3,000-image self-built dataset (non-public) |
| Zhang et al. (2026), *Frontiers in Marine Science* 13:1797307 | Edge-adaptive Gaussian-Scharr backbone blocks with partial-convolution channel efficiency (LEF-RT-DETR) | +4.3 AP vs. RT-DETR-r18 baseline | 49.7 GFLOPs | 970-instance self-built dataset (non-public) |

Every architecture surveyed that reported an accuracy improvement did so
at substantially higher computational cost than our target (2.6×–44×), on
a dataset that is either not publicly reproducible or belongs to a
different sonar imaging modality — forward-looking sonar (FLS) images a
forward range–bearing fan, whereas side-scan sonar (SSS, the modality
specified by PS 26057) images a lateral swath with grazing-incidence
acoustic shadows. We verified this modality distinction directly: UATD,
the dataset behind the thesis's headline preprocessing result, is
confirmed (via its originating publication, Xie et al., *Scientific Data*
2022) to be a multibeam forward-looking sonar dataset collected in Chinese
freshwater lakes. We independently reproduced the thesis's exact five-step
preprocessing chain and trained it, matched, on side-scan data; it
degraded mAP50 from 0.116 to 0.043 (§6.4). The two results are not in
tension — they describe different imaging modalities — but the reported
gain does not transfer, and adopting it uncritically for an SSS system
would have been an error.

### 4.2 Segmentation architectures

Tang et al. (2023, *EURASIP JASP* 2023:76) report a hybrid-dilated,
attention-gated U-Net variant (BHP-UNet) achieving Dice 78.3% / IoU 77.7%
on a self-collected, non-public 1,200-to-3,000-image dataset, at 73.2 MB
model weight. We did not implement a segmentation head: MILCO/NOMBO
provides bounding-box annotations only, and the sole public SSS dataset
with genuine mask labels we identified (AI4Shipwrecks) annotates
shipwrecks — large, high-contrast targets, which is precisely the size
regime our detector already handles adequately (§6.7) and therefore does
not test the failure mode a segmentation-based verifier would need to
address.

### 4.3 Production and defence systems

We reviewed four operational systems for architectural precedent:
GhostNetZero.ai (WWF/Accenture/Microsoft; DeepLabV3+ResNet50 segmentation
with cloud-GPU inference and metadata-driven geotagging), SeaClear (EU
Horizon; multi-robot physical litter retrieval), SeeByte SeeTrack +
Neptune ATR (defence-classified AUV mission management and automated
target recognition), and NOAA ERMA/MDMAP (federal web-GIS for marine
debris incident tracking). SeeByte's disclosed architectural pattern —
raw ingestion, model inference, confidence-scored contact list, human
analyst review, GIS export — is structurally identical to the pipeline
described in §3, arrived at independently. GhostNetZero's operational
model routes low- and medium-confidence detections to mandatory manual
review; the verification stage described in §5 is intended to reduce that
bottleneck algorithmically rather than eliminate the human reviewer.

### 4.4 Independent contemporaneous submissions

Because this project was developed inside a live hackathon, we searched
for and reviewed other public repositories addressing the identical
problem statement, pushed within the same development window (§ of this
survey conducted 2026-08-28; full detail in
`research/READY_TO_USE_REPOS.md`). Two findings are worth reporting as
independent validation rather than mere competitive comparison: one
submission's public README states, verbatim, "Coordinates are only shown
when real survey metadata is attached. They are never invented" — the
identical refusal principle formalised in §7 of this work, arrived at
without shared code or communication; a second submission's README
documents discovering, through its own model deployment, that a detector
trained only on clean backgrounds "detected everything as debris on real
noisy SSS data," and independently adopted synthetic-noise augmentation
as the fix — the same diagnosis and remedy examined empirically in §6.6.
Two independent teams converging on the same two design principles is, in
our view, stronger evidence for their correctness than any single
benchmark figure. No externally reviewed submission published a held-out,
leakage-controlled accuracy figure at the time of this survey, which we
note as an absence rather than a claim about their underlying performance.

---

## 5. The verification stage

### 5.1 Motivation

Given the class imbalance documented in §1.2, a detector calibrated for
recall will necessarily produce a high absolute rate of false positives.
Rather than raise the detection threshold — which discards true positives
indiscriminately — we retain a low detection threshold and add an
independent verification stage informed by evidence the detector itself
does not use.

### 5.2 Method

Ten features are computed per candidate detection directly from image
pixels, independent of the detector's own confidence score: target/local
contrast, acoustic-shadow ratio, shadow-side directional consistency
(evaluated against the sonar's nadir line when known), highlight
compactness, aspect ratio, edge straightness, texture homogeneity relative
to the surrounding background, local signal-to-noise ratio, and a
log-scaled size rank. These, concatenated with the raw detector score,
are input to an L2-regularised logistic regression fitted on the held-out
2017 validation survey, with a recall-floor constraint on decision
threshold selection — an unconstrained F1-maximising threshold on a small
fit split (30 objects) was observed to degenerate toward "reject nearly
everything," which achieves high precision by discarding recall rather
than by discriminating evidence.

### 5.3 Result

| Configuration | Precision | Recall | True positives retained | Falsely-alarmed empty frames (of 473) |
|---|---|---|---|---|
| Detector alone | 0.247 | 0.110 | 21 | 37 |
| + hand-tuned threshold rules | 0.300 | 0.063 | 12 | 18 |
| **+ learned verification stage** | **0.322** | 0.100 | **19** | **25** |

The learned model retains 19 of the 21 true positives the detector
produces; the hand-tuned rule set, evaluated for comparison, achieves
similar precision by discarding seven additional true positives (12 of
21). We interpret this as direct evidence for fitting rather than
hand-specifying the verification stage — a conclusion consistent with,
though independently obtained from, the problem statement's own guidance
against unvalidated heuristic rules.

### 5.4 A diagnostic property of inspectable models

An early fit of this model assigned the acoustic-shadow feature a
negative coefficient, directly contradicting the underlying physics (an
object standing proud of the seabed should cast a shadow, which is
positive evidence of a genuine target). Rather than report this as a
finding, we investigated it: the negative coefficient was traced to a
train/inference preprocessing mismatch elsewhere in the pipeline (§6.4)
that was corrupting exactly the image region the shadow features measure.
Correcting the mismatch restored the coefficient to its physically
expected sign. We record this as a general observation: a model whose
inputs are individually interpretable can surface defects in an upstream
pipeline that aggregate accuracy metrics alone do not localise.

---

## 6. Experimental results

All figures in this section are measured on the held-out test surveys
(§2.2) unless stated otherwise, and are reproducible via
`scripts/run_full_evaluation.sh`.

### 6.1 Detector selection

We selected YOLO11n (2.58M parameters, 6.3 GFLOPs) over every alternative
surveyed in §4.1 on the basis that no reviewed alternative demonstrated
an accuracy advantage on data we could obtain, at a computational cost
compatible with edge deployment. Two Apple-Silicon-specific training
stability issues were identified and resolved during this work:
Ultralytics 8.4.130 was observed to diverge on this dataset (validation
classification loss increasing from approximately 25 to 1.1×10⁶ over five
epochs while training loss remained flat); pinning to the 8.3.x release
line resolved this. Mixed-precision training is disabled by default on
the MPS backend in our training configuration; while not proven causally
related to the divergence above, it was not re-enabled without
independent justification.

### 6.2 Primary detector accuracy

| Metric | Value |
|---|---|
| mAP50 (cross-survey) | 0.1163 |
| mAP50-95 | 0.0396 |
| Precision (detector-only) | 0.3444 |
| Recall (detector-only) | 0.1639 |

We report this figure without qualification as modest in absolute terms.
It is not inflated by a random split (§2.2), it is trained on 447 objects
against target sizes averaging approximately 24 pixels in linear
dimension, and it spans an eleven-year acquisition-hardware gap between
training and test surveys. The system-level contributions of this work
(§5, §7) act downstream of and independently from this number.

### 6.3 Full pipeline ablation

| Configuration | Precision | Recall | F1 | True positives | False positives | Falsely-alarmed frames (of 473) |
|---|---|---|---|---|---|---|
| A: detector only | 0.2471 | 0.1099 | 0.1522 | 21 | 64 | 37 |
| B: (A) without tiling | 0.2526 | 0.1257 | 0.1678 | 24 | 71 | 43 |
| C: (A) + hand-tuned rules | 0.3000 | 0.0628 | 0.1039 | 12 | 28 | 18 |
| D: (A) + learned verification | **0.3220** | 0.0995 | 0.1520 | **19** | 40 | **25** |
| E: (D) + calibration | 0.3220 | 0.0995 | 0.1520 | 19 | 40 | 25 |
| X: mismatched-preprocessing control | 0.0102 | 0.0052 | 0.0069 | 1 | 97 | 57 |

Rows D and E share identical precision/recall/F1 by construction — Platt
calibration is a monotonic rescaling of the reported confidence value and
does not alter which detections are accepted. Row X is included as a
negative control, discussed in §6.4, and is not evidence about
preprocessing utility in general.

### 6.4 Preprocessing: measured three independent ways

We tested sonar-specific preprocessing (dropout repair, empirical
across-track gain normalisation, Lee adaptive speckle filtering,
water-column removal) under three conditions, motivated by an initial
result that appeared anomalous enough to distrust without independent
confirmation:

| Condition | Setup | F1 | Falsely-alarmed frames (of 473) |
|---|---|---|---|
| Mismatched (inference-only) | Detector trained on raw frames; our preprocessing chain applied only at inference | 0.012 (from 0.144 raw) | 186 (from 80) |
| Matched — our chain | Full retrain on preprocessed data; train, validation, test, and inference all consistent | mAP50 0.032 (raw: 0.116) | — |
| Matched — thesis's exact five-step chain | Faithfully reproduced (§4.1) and retrained matched | mAP50 0.043 (raw: 0.116) | — |

Every configuration underperforms the unmodified raw-image detector. The
mismatched condition demonstrates that preprocessing applied at inference
to a model that was not trained on it constitutes a distribution shift
sufficient to cause a twelvefold degradation in F1 — the correct
methodological lesson is that a preprocessing chain, if adopted, must be
part of the training distribution, not an inference-time addition. The
matched conditions demonstrate that even when this is respected,
preprocessing underperforms raw input on this dataset for this detector.
The pipeline's default preprocessing state is therefore disabled, recorded
as an explicit property of each trained checkpoint (a sidecar metadata
file, checked automatically at inference) rather than a global default
that could silently mismatch a future model.

### 6.5 Confidence calibration

Platt (logistic) scaling is fitted on the held-out validation survey and
evaluated by Expected Calibration Error before and after fitting. When
the fit split is judged too thin or single-class to support a meaningful
fit, the system falls back to an identity transform and explicitly labels
every downstream confidence value as uncalibrated, rather than reporting
a value fitted on insufficient evidence as though it were reliable.

### 6.6 Robustness under controlled perturbation

Synthetic, single-variable perturbations (multiplicative speckle,
Gaussian blur, resolution reduction, simulated ping dropout, gain shift)
were applied to 120 held-out frames. The primary detector's recall falls
to exactly zero under every tested speckle level (σ ≥ 0.25), a complete
robustness collapse. Training with speckle-augmented copies of the
training set (raw frames plus multiplicative-speckle copies at
σ ∈ {0.25, 0.5}, validation and test kept clean) was evaluated in two
stages: an initial run terminated early by the operator at epoch 44
(precision 0.185, recall 0.142, mAP50 0.076 on the full held-out test),
and a subsequent run permitted to converge fully (95 of 100 requested
epochs, best validation checkpoint at epoch 60), undertaken specifically
to test whether the initial run's accuracy deficit was an artefact of
insufficient training.

| | Detector (raw) | Speckle-augmented, undertrained | Speckle-augmented, fully converged |
|---|---|---|---|
| Full-test mAP50 | 0.1163 | 0.0763 | 0.0812 |
| Full-test precision | 0.3444 | 0.1849 | 0.3106 |
| Full-test recall | **0.1639** | 0.1415 | **0.0785** |

It was not an artefact of insufficient training. The fully converged model
recovers most of the precision deficit relative to the undertrained
checkpoint but exhibits *lower* recall than either the undertrained
checkpoint or the raw model, despite substantially improved robustness
under most tested perturbations (baseline F1 0.237 vs. 0.061 for the raw
model on the 120-frame robustness subset; retains F1 0.173 under σ=0.25
speckle where the raw model retains 0.000). We conclude that this
represents a genuine, relatively stable accuracy/robustness tradeoff
rather than a resolvable training-budget deficiency, and accordingly do
not promote the speckle-augmented model to primary status. It is retained
as a separately usable, independently documented alternative checkpoint
for deployment contexts where operating conditions are known to be
noise-dominated.

### 6.7 Target-size-dependent failure analysis

Recall stratified by annotated target area (raw detector, IoU ≥ 0.3)
reveals a result contrary to prior expectation: the smallest annotated
targets are detected with the highest recall (0.193 at <300 px²) and
every target exceeding 2,500 px² is missed entirely (recall 0.000, n=17).
An initial hypothesis attributed this to the detector's own scale-jitter
training augmentation. We tested this hypothesis directly by normalising
target size as a fraction of frame area — necessary because the source
dataset mixes 416px and 1024px images, making raw pixel-area values
non-comparable across the corpus — and found that the largest annotated
object in the *training* set occupies 1.7% of its frame, while the
largest in the *test* set occupies 9.3%, a 5.5-fold discrepancy driven
substantially by two individual frames. Standard scale-jitter augmentation
at the magnitude used in training (approximately ±25% linear scale) cannot
synthesise a fivefold linear-dimension increase from training crops that
never contained an object of comparable size. We therefore revise the
attributed cause: this is substantially a training-data size-coverage gap
rather than an augmentation hyperparameter defect, and correct this
conclusion in every document of this project that previously stated the
earlier hypothesis.

### 6.8 Unsupervised anomaly detection

A convolutional autoencoder trained exclusively on normal-seabed image
patches (never exposed to an annotated target during training) was
evaluated for its capacity to flag unknown-class objects via
reconstruction error, motivated directly by the problem statement's
requirement for anomaly detection beyond a fixed supervised taxonomy.
Evaluated at two patch resolutions:

| Patch size | Frame-level ROC-AUC | Patch-level ROC-AUC |
|---|---|---|
| 64 px | 0.465 | 0.482 |
| 32 px | 0.472 | 0.536 |

Both results are statistically indistinguishable from chance (0.5). We
attribute this to the annotated targets occupying too small a fraction of
any patch large enough to admit a meaningful reconstruction, combined with
substantial normal-seabed texture variance (speckle, range striping) that
dominates the reconstruction-error signal. This component is not deployed.
We consider a chance-level score presented under the label "anomaly
detection" to constitute a worse operational outcome than presenting no
score at all, and record embedding-based novelty detection (e.g. PaDiM,
PatchCore, applied over detector backbone features rather than raw pixels)
as the indicated direction for future work.

### 6.9 Performance and edge deployment

| Measurement | MPS (Apple M5) | CPU |
|---|---|---|
| Inference latency | 21.4 ms | 82.2 ms |
| Full-pipeline latency per frame | 54.8 ms | 119.0 ms |
| Survey throughput | 37.4 frames/s | 17.5 frames/s |
| Peak resident memory | 640 MB | 900 MB |

ONNX export (CoreML execution provider): 10.61 MB, 8.49 ms mean inference
latency. No CUDA dependency exists anywhere in the codebase; the MPS/CPU
selection logic falls back automatically and was validated on both paths.

---

## 7. Geolocation and the refusal principle

Detections are geolocated under three conditions: a georectified raster
carrying an embedded affine pixel-to-CRS transform; a raw waterfall
image accompanied by per-ping navigation records, in which case slant
range is converted to ground range via `ground = √(slant² − altitude²)`,
projected along a geodesic bearing derived from vehicle heading, and
assigned an explicit uncertainty budget combining GPS accuracy,
heading-angle error scaled by range, tow-cable layback uncertainty,
altitude-conditioning error, and range-bin resolution, combined in
quadrature; or, absent either, **the coordinate fields are populated with
null and a stated reason, and no numerical position is emitted.** This
behaviour is verified by an automated regression test asserting that no
code path can produce a non-null coordinate without a corresponding
navigation input. We note in §4.4 that an independently developed
competing submission states an identical policy in its own
documentation, arrived at without shared code.

The error-budget formulation exhibits an expected and, in our judgement,
methodologically important property: because the altitude-correction term
scales as altitude divided by ground range, uncertainty diverges as a
target approaches the sonar's nadir track, where the slant-to-ground
inversion becomes numerically ill-conditioned. We allow this divergence
to propagate into the reported uncertainty and confidence band rather
than clamping it, which reproduces standard side-scan sonar interpretive
practice of discounting nadir-region contacts. **Positional accuracy has
not been validated against independently surveyed ground truth**, because
the primary dataset carries no navigation metadata; the geometric formulation,
sign conventions, and refusal behaviour are unit-tested, but accuracy is not.

---

## 8. Comparative summary

| Dimension | This work | Best external reference |
|---|---|---|
| Detector compute | 6.3 GFLOPs | 16.2–276.3 GFLOPs (all four detection papers) |
| False-positive handling | Learned, 10-feature logistic model, measured against a hand-tuned baseline | Not published as a distinct, measured stage in any reviewed reference |
| Preprocessing | Measured matched and mismatched, on two independent chains; disabled by default with evidence | Assumed beneficial without an SSS-matched ablation in every reviewed reference |
| Geolocation | Refuses to fabricate; explicit metres error budget | GhostNetZero: metadata-driven, undisclosed methodology; no reviewed academic reference addresses this |
| Robustness to sensor noise | Measured under controlled perturbation; augmentation strategy evaluated to convergence, tradeoff reported honestly | Explicitly identified as unaddressed future work in the most recent reviewed paper (Zhang et al., 2026) |
| Anomaly/unknown-class detection | Attempted, measured, found non-functional (AUROC ≈ 0.5), not deployed | Not addressed by any reviewed academic reference; listed as a gap by the source thesis |
| Edge deployment evidence | Measured on target hardware (Apple Silicon); ONNX export benchmarked | Asserted as a design goal in three of four detection papers; none report measurement on constrained hardware |
| Evaluation protocol | Survey-year/recording-level splits; matched preprocessing; disclosed negative results | Random or undisclosed splits in every reviewed reference and every reviewed competing submission |

---

## 9. Limitations

We report every limitation identified during this work without
qualification, in preference to omission:

1. The training and evaluation dataset does not contain the problem
   statement's named target class (ghost fishing gear); the geographic
   origin of the primary dataset is undisclosed by its authors and is not
   claimed to be Indian.
2. Detector accuracy (§6.2) is modest in absolute terms.
3. Geolocation methodology is verified; geolocation *accuracy* is not
   (§7).
4. The verification stage (§5) and calibration (§6.5) are both fitted on
   30 objects, a small basis.
5. Only raster and CSV-navigation input formats are supported; vendor raw
   sonar formats (XTF, JSF, SON) are not.
6. Edge-deployment claims are supported by measurement on a laptop-class
   device; no embedded or AUV-class payload computer has been tested.
7. The default detector backend (Ultralytics) is licensed AGPL-3.0-or-later,
   which constrains closed-source commercial redistribution of the
   combined work; a permissively licensed alternative backend is
   implemented but not yet trained.
8. Preprocessing (§6.4), the anomaly branch (§6.8), and speckle
   augmentation (§6.6) were each tested and found not to improve, or to
   trade away accuracy for, the primary deployment configuration.
9. The target-size failure mode identified in §6.7 remains unresolved; the
   corrective direction (additional large-object training data or
   deliberate paste-augmentation) has not been implemented.
10. No out-of-distribution detection mechanism exists; the system will
    produce confident-appearing output on sonar imagery from unfamiliar
    hardware or environments without any signal that it has left its
    validated operating domain. We consider this the most operationally
    significant unresolved limitation of the system as described.

---

## 10. Conclusion

We have described and evaluated a marine-anomaly-detection pipeline for
side-scan sonar imagery built on the premise that, given the class
imbalance characteristic of this domain, verification of candidate
detections against independent physical evidence is a more productive
engineering investment than architectural sophistication in the detector
itself. This premise is supported by direct measurement: a fitted
verification stage outperforms a hand-tuned equivalent on the metric that
determines operational trust (true positives retained per false alarm
avoided), while six independently published architectures that increase
detector complexity — reviewed, and in two cases faithfully reproduced and
retrained on our own data — either fail to transfer across sonar imaging
modality or fail to justify their computational cost at the accuracy gain
measured. Equally, we report two of our own hypotheses that measurement
overturned: a speckle-robustness training strategy expected to resolve
with additional training budget did not, and a preprocessing chain
expected to improve detection, reproduced faithfully from an external
source, degraded it. We regard the discipline of reporting these results
as measured rather than omitting or reframing them as the primary
methodological contribution of this work, and the remaining gap to the
problem statement's full scope — principally, ghost-gear-specific
training data, currently withheld by a single-step access gate outside
this system's control — as the most direct path to closing it.

---

## References

1. Pessanha Santos, N. & Moura, R. (2024). Side-scan sonar imaging data of
   underwater vehicles for mine detection. *Data in Brief*, 53, 110132.
2. Divyabarathi, G. (2025). *Underwater Sonar Image Analysis using Deep
   Learning*. PhD thesis, Cochin University of Science and Technology.
3. Yu, Y., Zhao, J., Gong, Q., Huang, C., Zheng, G., & Ma, J. (2021).
   Real-time underwater maritime object detection in side-scan sonar
   images based on transformer-YOLOv5. *Remote Sensing*, 13(18), 3555.
4. Zhao, H., Han, S., Geng, J., Han, Y., Jia, S., & Li, K. (2025).
   MSF-DETR: A small target detection algorithm for sonar images based on
   spatial-frequency domain collaborative feature fusion. *PLOS ONE*,
   20(11), e0336468.
5. Tang, Y., Wang, L., Li, H., & Bian, S. (2023). Side-scan sonar
   underwater target segmentation using the BHP-UNet. *EURASIP Journal on
   Advances in Signal Processing*, 2023, 76.
6. Zhang, F., Li, Z., Wen, X., Cheng, C., Deng, B., Zhang, T., & Pan, G.
   (2026). Lightweight Edge–Frequency Driven Real-Time Detection
   Transformer for side-scan sonar target detection. *Frontiers in Marine
   Science*, 13, 1797307.
7. Aubard, M., Antal, L., Madureira, A., & Ábrahám, E. (2024). Knowledge
   Distillation in YOLOX-ViT for Side-Scan Sonar Object Detection. *arXiv
   preprint* arXiv:2403.09313.
8. Bodine, C. S., Baxevani, K., Abbasi, N., Wierzbicki, J., Christoph, O.,
   Hughes, C., Bagoren, O., Hines, O., Greco, J., & Trembanis, A. (2026).
   GhostVision: Democratizing Derelict Gear Detection Using Low-Cost Sonar
   and Artificial Intelligence. *Journal of Marine Science and
   Engineering*, 14(10), 951.
9. Xie, K. et al. (2022). A Dataset with Multibeam Forward-Looking Sonar
   for Underwater Object Detection. *Scientific Data*, 9, 739.
10. Lee, J. S. (1980). Digital image enhancement and noise filtering by
    use of local statistics. *IEEE Transactions on Pattern Analysis and
    Machine Intelligence*, 2(2), 165–168.
11. Platt, J. (1999). Probabilistic outputs for support vector machines
    and comparisons to regularized likelihood methods. *Advances in Large
    Margin Classifiers*.

---

*All figures in this report are computed by the authors from the sources
cited and are reproducible from the accompanying repository via
`scripts/run_full_evaluation.sh`; raw experiment records are retained
under `experiments/`. External figures attributed to other authors are
cited to their source and are not independently re-verified beyond the
modality and licence checks described in §4 and `research/`.*
