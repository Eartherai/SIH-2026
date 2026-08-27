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

## 10. Quality score is a heuristic

`quality_score` combines dynamic range, dropout, saturation, speckle and usable
area with hand-chosen weights. It is useful for warning an operator and as a soft
input to priority. It is **not** a calibrated measure of anything physical, and
it is labelled as a heuristic in the code.

## 11. Priority weights are a product convention

The priority formula is transparent and adjustable, but the weights and the
class-harm table are **our** choices. No official marine-hazard triage standard
for derelict gear was found during this work, and we do not claim to implement
one.

## 12. Small-sample statistics

The test set holds 191 objects across 612 frames. Differences of a few percent
between pipeline variants are **within noise**. The ablation table should be read
for direction and magnitude, not for precise ordering of adjacent rows.

## 13. Two things the model literally cannot do

- **Detect an object class it has never seen.** A trained ghost net, a container,
  or a pipeline will be reported — if at all — as MILCO or NOMBO, because those
  are the only two labels the head has.
- **Know when it is out of domain.** There is no out-of-distribution detector. On
  sonar from unfamiliar hardware it will still emit confident-looking numbers.
  This is the most dangerous failure mode in deployment and it is not solved.
