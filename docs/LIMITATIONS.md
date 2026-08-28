# Limitations

Written to be used against us. If a judge, a reviewer or an NIOT engineer is
going to find it, it should be here first.

## 1. The data is not the problem statement's data

| PS 26057 asks about | What we trained on |
|---|---|
| Ghost nets, derelict fishing gear | **Mine-like contacts and bottom objects** |
| Shipwrecks, pipes, cylinders | Not represented |
| Indian waters (MoES/NIOT context) | Region undisclosed; not Indian waters |

**Why.** The only open, ungated, clearly-licensed side-scan dataset with
annotations we could obtain is MILCO/NOMBO (CC BY 4.0). The closest ghost-gear
dataset (`PINGEcosystem/sss-crab-pot-detection-ds`) is **access-gated** and
returned HTTP 403.

**Consequence, stated plainly:** *AQUA-SHIELD has never detected a ghost net.*
No claim about ghost-gear accuracy is made anywhere in this repository. What is
demonstrated is the *discrimination task* — separating a man-made target from
ambiguous seabed clutter — which is the transferable part, plus a complete
operational pipeline around it.

## 2. Detector accuracy is modest

Cross-survey generalisation on completely held-out surveys, with only 447
training objects, is genuinely hard. The measured numbers are in
`docs/BENCHMARKS.md` and they are not flattering. They are also not inflated by
a random split, which is the more common failure.

**Why it is low:** 447 training objects; targets averaging 5.7% × 3.3% of the
frame; and an 11-year gap in acquisition hardware/settings between the training
and test surveys.

## 3. Geolocation accuracy has never been validated

The geometry, sign conventions, circular heading interpolation, error-budget
arithmetic and refusal behaviour are all unit tested. **Positional accuracy
against ground truth is not**, because MILCO/NOMBO ships no navigation data.

The demo's `04_georeferenced` scenario uses an explicitly **synthetic** track,
labelled as such in the CSV header, the scenario metadata and the UI.

**What would fix it:** one survey with both sonar and independently surveyed
object positions. This is the single highest-value thing real NIOT data would
unlock.

## 4. Calibration is fitted on 30 objects

The validation survey (2017) contains 30 annotated objects. The Platt calibrator
and the learned FP filter are fitted on the candidates arising from that survey.
That is a **thin** basis. ECE improves markedly on the fit split, but a
calibrator fitted on one small survey may not transfer to a different sonar.

The system reports `calibrated: true/false` per hazard precisely so this is
visible rather than assumed.

## 5. No raw sonar format support

`.XTF`, `.JSF`, `.SON`/`.DAT`, `.sl2` are **not** read. AQUA-SHIELD ingests
rasters (PNG/JPG/TIFF/GeoTIFF) plus a navigation CSV.

This is a deliberate scope decision, not an oversight: PINGMapper (MIT) already
decodes Humminbird/Lowrance well, and scientific formats (EdgeTech, Klein,
Marine Sonic) each need real hardware to test against. Claiming universal sonar
support without a single vendor file to verify would be dishonest.

## 6. "Edge deployment" is an architecture claim, not a demonstrated one

Measured: ONNX export at 10.61 MB running at 10.84 ms under ONNX Runtime on this
M5, and 664 MB peak RSS for the full pipeline.

**Not measured:** any Jetson, any AUV payload computer, any embedded ARM board,
any thermal- or power-constrained environment. We have not run this on hardware
that could go in the water.

## 7. The ultralytics dependency is AGPL-3.0

Fine for a hackathon, research, or an internal NIOT tool. **Not** fine for a
closed-source commercial product. The backend abstraction exists so this can be
swapped, and a BSD-3 torchvision backend is implemented — but it has **not been
trained**, so the licence-clean path is designed, not delivered.

## 8. Things designed but not executed

- **Hard-negative mining.** The loop is specified in `docs/ML_PIPELINE.md` and the
  473 empty test frames make it easy, but it was not run. No benefit is claimed.
- **Trained torchvision baseline.**
- **Segmentation head.** The dataset has boxes, not masks.
- **Hyperparameter search.** E03 → E04 is one reasoned step, not a sweep.
- **Multi-domain model.** One detector, one domain.

## 9. Our sonar preprocessing did not help, and we ship it disabled

We implemented a physically-justified chain — dropout repair, water-column
removal, Lee speckle filtering, across-track gain normalisation, dynamic-range
stretch — and then measured it properly with a matched retrain. It made detection
**substantially worse** (mAP50 0.0318 vs 0.1163 raw). The default profile is now
`none`.

This is an honest negative result and it costs us: a good deal of the sonar-
specific engineering in this repository is, on this dataset with this detector,
not earning its place in the inference path. It remains useful for QC, for the
verification features, and as an inspectable option — but we cannot claim it
improves detection, and we do not.

What we did **not** do is isolate which stage is responsible. A per-stage matched
ablation would likely show some stages help and others hurt. That work is unrun.

## 10. Detection collapses under added speckle

Controlled degradation on 120 held-out frames (`experiments/robustness.json`):
adding multiplicative speckle at even the mildest level tested (sigma 0.25) drove
detections to **zero** — recall 0.036 at baseline, 0.000 under every speckle
level. Blur and resolution loss degrade gracefully up to a point (blur kernel 5,
0.5x resolution) and then fail; ping dropout above 5% also fails. Low contrast and
gain shift are tolerated.

The irony is not lost on us: we implemented a Lee speckle filter precisely for
this, and then measured that applying it makes detection worse overall (§9). The
correct fix is to **train with speckle augmentation** rather than to filter at
inference. We tried it twice: an undertrained attempt (E08) and then a full
95-epoch run (E09) specifically to rule out undertraining as the explanation. It
wasn't — under speckle σ=0.25 the raw model retains 0% of its clean recall while
both augmented variants retain 13–14%, so the robustness weakness is
**measurably, if unevenly, reduced**. But the clean-accuracy cost is **not** a
training-budget artifact: E09's full-test recall (0.079) is *lower* than the
undertrained E08's (0.142), even though precision recovered. This is a real,
relatively stable accuracy/robustness tradeoff. We ship both: E04 stays primary,
E09 is a separately-usable alternative checkpoint for noise-heavy deployments
(`models/aquashield_speckle_robust.pt`, `docs/BENCHMARKS.md` §10.2).

## 11. The unsupervised anomaly branch does not work yet (evaluated, rejected)

PS 26057 asks for anomaly detection; the thesis lists its absence as a gap. We
built the obvious MVP — a convolutional autoencoder trained only on normal seabed,
scoring reconstruction error as an "unlike-normal" signal
(`src/aquashield/anomaly/`, `scripts/train_anomaly.py`). Measured on held-out test:

| Patch size | Frame ROC-AUC | Patch ROC-AUC |
|---|---|---|
| 64 px | 0.465 | 0.482 |
| 32 px | 0.472 | 0.536 |

**Both are at or barely above chance (0.5).** A naive reconstruction-error AE
cannot separate these small (~24 px), low-contrast targets from textured,
speckled, nadir-striped seabed — the target is too small a fraction of the patch
and the normal-seabed error variance swamps it. We **do not ship an anomaly
score**, because a chance-level score presented as anomaly detection would be
worse than none. The code is kept as a runnable, honestly-labelled MVP. The
right next approach is feature-embedding novelty detection (PaDiM / PatchCore over
the detector backbone's features), not pixel reconstruction — logged as future work.

## 12. Quality score is a heuristic

`quality_score` combines dynamic range, dropout, saturation, speckle and usable
area with hand-chosen weights. It is useful for warning an operator and as a soft
input to priority. It is **not** a calibrated measure of anything physical, and
it is labelled as a heuristic in the code.

## 13. Large targets are missed entirely (recall 0.000 on >2500 px²) — investigated and re-diagnosed

Counter to the usual "small targets are hard" framing, our raw model detects the
*smallest* targets best (recall 0.193 at <300 px²) and **every large target**
(17 objects >2500 px²) is **missed** (`docs/BENCHMARKS.md` §10.3).

Phase 2 speculated this was mainly our `scale=0.25` augmentation biasing the
model toward small objects. **We checked properly rather than leave that
speculation standing** (`experiments/large_target_gap_analysis.json`): using
area-as-fraction-of-frame (`area_frac`, needed because MILCO/NOMBO mixes 416px
and 1024px images), the largest **training** object occupies **1.7%** of its
frame; the largest **test** object occupies **9.3%** — a **5.5× gap**, driven
chiefly by two extreme frames (0365_2018, 0366_2018). Standard scale-jitter
augmentation (E04's factor range ≈0.75–1.25×) cannot synthesize a 5×+
linear-dimension jump from typical small-target training crops — it physically
cannot manufacture training exposure to a size regime that doesn't exist in the
source images.

**Revised conclusion:** this is substantially a **training-data coverage gap**,
not primarily a fixable hyperparameter. Augmentation tuning may help at the
margin but will not close it. The real fix is more training data spanning the
same size range as the test surveys, or synthetic paste-augmentation that
deliberately inserts oversized target crops during training — neither
attempted yet.

## 14. Priority weights are a product convention

The priority formula is transparent and adjustable, but the weights and the
class-harm table are **our** choices. No official marine-hazard triage standard
for derelict gear was found during this work, and we do not claim to implement
one.

## 15. Small-sample statistics

The test set holds 191 objects across 612 frames. Differences of a few percent
between pipeline variants are **within noise**. The ablation table should be read
for direction and magnitude, not for precise ordering of adjacent rows.

## 16. Two things the model literally cannot do

- **Detect an object class it has never seen.** A trained ghost net, a container,
  or a pipeline will be reported — if at all — as MILCO or NOMBO, because those
  are the only two labels the head has.
- **Know when it is out of domain.** There is no out-of-distribution detector. On
  sonar from unfamiliar hardware it will still emit confident-looking numbers.
  This is the most dangerous failure mode in deployment and it is not solved.
