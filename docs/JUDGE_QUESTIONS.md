# Judge questions

Two answers each: a **20-second** version to say out loud, and a **deep** version
for when they push. Where the honest answer is "we didn't do that", it says so —
a judge who catches you overclaiming stops believing everything else.

---

## A. Architecture & model choice

**1. Why YOLO?**
*20s:* Targets average 24×14 pixels. YOLO11n has a stride-8 detection head that
holds small objects, trains to convergence on 447 objects on a laptop, exports to
ONNX, and runs on Apple Silicon. It was the only candidate that satisfied all
four constraints.
*Deep:* See `research/MODEL_SELECTION.md`. RT-DETR and RF-DETR were rejected as
data-hungry — transformer detectors need far more than 447 objects. Faster R-CNN
adds two-stage cost with no accuracy case at this scale. The real answer is that
the detector is the *replaceable* part: `detection/detector.py` is
backend-agnostic and nothing downstream imports Ultralytics.

**2. Why not segmentation, when the PS allows masks?**
*20s:* Because the dataset has boxes, not masks. Training a segmentation model
would mean inventing supervision we don't have.
*Deep:* AI4Shipwrecks provides real masks and is the correct route later — but for
wrecks, which are large high-contrast targets, i.e. the easy end. PS 26057 is
dominated by small debris against clutter, where the hard part is *deciding
whether a return is man-made*, not delineating its outline to the pixel.

**3. What is actually novel here?**
*20s:* Not the detection. GhostVision already does detection plus georeferencing
for derelict gear, and we say so. What's ours is the verification layer: a
*learned* false-positive filter over physically-motivated features, explicit
confidence calibration, and a refusal to output a coordinate we can't compute.
*Deep:* `research/prior_art_matrix.md` has a claim-by-claim honesty audit,
including the claims we rejected as not-novel.

**4. How is this different from GhostVision?**
*20s:* GhostVision is single-class, single sonar-grade, and its licence is
unresolved (GitHub reports NOASSERTION), so it can't be vendored into a
government deliverable. Its own headline figures — F1 0.512 at recall 0.922 —
show precision is the binding constraint. They tune post-processing; we fit an
inspectable model.
*Deep:* Their published fix raises F1 to ~0.71–0.73 by post-processing
optimisation. Our contribution is making that step a *fitted, auditable* model
with per-detection attribution rather than tuned thresholds.

**5. Why does the detector run at a threshold as low as 0.10?**
*20s:* Deliberate. Recall first, verify second. It's the verification stage's job
to remove clutter, and it can use evidence the detector never saw.
*Deep:* Raising the detector threshold discards candidates irrecoverably. Keeping
them lets an independent, feature-based stage arbitrate — and that stage's
decisions are explainable, which a raised threshold never is.

---

## B. Data & legitimacy

**6. Where did the training data come from?**
*20s:* MILCO/NOMBO — 1,170 real side-scan frames from a Teledyne Gavia AUV,
2010–2021, published in Data in Brief, CC BY 4.0, downloadable without
registration.
*Deep:* DOI `10.6084/m9.figshare.24574879`. Attribution and change statement are
in `data/DATASETS.md` and in every generated report's provenance block.

**7. Is the data legally usable?**
*20s:* Yes. CC BY 4.0 permits commercial use and redistribution with attribution.
Every dependency licence was verified programmatically, not recalled.
*Deep:* `LEGAL_AND_LICENSES.md`. The one real constraint is Ultralytics
(AGPL-3.0), which we flag prominently rather than bury.

**8. Have you actually detected a ghost net?**
*20s:* **No.** The ghost-gear dataset is access-gated and returned HTTP 403. We
built the adapter, we did not get the data, and we make no ghost-net accuracy
claim anywhere.
*Deep:* `PINGEcosystem/sss-crab-pot-detection-ds`, DOI 10.57967/hf/8397. What
transfers is the discrimination task — MILCO vs NOMBO *is* man-made vs ambiguous
— plus the entire pipeline around the detector.

**9. Isn't mine detection a different problem from marine debris?**
*20s:* The object class differs; the hard part doesn't. Both are "is this compact
acoustic return a man-made object or natural seabed structure?" — and 74% of our
frames are empty seabed, which is exactly the false-positive problem the PS names.
*Deep:* NOMBO means "not mine-like", **not** "natural", so we map it to AMBIGUOUS
and never to a man-made subclass (`data/class_mapping.yaml`). Getting that
semantics right is the difference between an honest system and one that quietly
mislabels.

**10. Why not just use more data?**
*20s:* We looked. Most real SSS datasets are access-by-request or gated; the big
marine-debris dataset is optical ROV imagery, not sonar.
*Deep:* `research/sources.md` lists everything reviewed and why each was accepted
or rejected — including a 2025 survey paper used specifically to check we hadn't
missed an open dataset.

---

## C. Method & rigour

**11. How do you know your metrics aren't inflated?**
*20s:* We split by acquisition **year**, not randomly. Train on 2015+2010,
calibrate on 2017, test on 2018+2021. A random split would leak, because
consecutive frames share seabed, gain settings and often the same object.
*Deep:* A test enforces it (`test_splits_are_survey_disjoint_no_leakage`). Our
numbers are lower than a random split would give. That's the point.

**12. What does your confidence number mean?**
*20s:* If `calibrated: true`, it's a Platt-scaled probability fitted on a held-out
survey. If `calibrated: false`, it's a raw detector score and the report, the
JSON disclaimer and the UI all say so.
*Deep:* `confidence/calibration.py`. We measure ECE before and after. Both the
calibrator and the FP filter **refuse to fit** on insufficient or single-class
data rather than produce a meaningless model.

**13. Is the confidence calibrated?**
*20s:* On this dataset, yes, and we report the ECE improvement. But it's fitted on
a survey with 30 objects — a thin basis, and we say so in `docs/LIMITATIONS.md`.

**14. How do you handle rocks and sand ripples?**
*20s:* Ten physical features per candidate — shadow coherence, contrast, highlight
compactness, texture roughness relative to background — fed to a logistic model
fitted on held-out data, with per-detection attribution for every decision.
*Deep:* And the weights earned their keep in an unexpected way. An early fit gave
`shadow_ratio` a large *negative* weight, contradicting the physics. That turned
out to be a **symptom of a bug in our own pipeline** — we were preprocessing at
inference with a chain the detector had never been trained on. Fixing it gave a
12× F1 improvement and turned both shadow features positive. The point is that an
inspectable filter is a diagnostic instrument; a hand-tuned threshold would have
absorbed that defect silently.

**15. How do you handle acoustic shadows?**
*20s:* We measure them on both sides of a candidate and, when the nadir column is
known, check the darker side is the physically correct far-range side. When nadir
is unknown we return "unknown" rather than pretending to know range direction.

**16. What about the water column?**
*20s:* Detected and removed. And detecting it is harder than it sounds — the dark
nadir band is *split* by a bright first-bottom-return spike, so the obvious
"darkest contiguous run" algorithm finds only half of it.
*Deep:* Our first implementation silently missed it on every real frame. The fix
bridges the bright gap; negative controls (uniform image, pure noise) are unit
tested so it can't over-trigger. `docs/DATA_PIPELINE.md` shows the measured profile.

**16b. Do you actually apply that preprocessing?**
*20s:* Only if the detector was **trained** on it. That is the single biggest
lesson from this project — applying our standard chain at inference to a
raw-trained model cost a **12× F1 drop** (0.144 → 0.012) and more than doubled
false-alarm frames. The profile is now a property of the checkpoint, stored in a
sidecar and selected automatically, with a test that stops the bad default coming
back.

**16c. You built a whole sonar preprocessing chain and then disabled it. Why?**
*20s:* Because we measured it and it did not help. Retrained with preprocessing
matched at train and inference time, mAP50 was 0.032 versus 0.116 for the
raw-trained model. Shipping it on by default would have been applying an
operation because it sounds appropriate — which is exactly what the problem
statement warns against.
*Deep:* Our best explanation is that YOLO11n is fine-tuned from COCO weights whose
early layers already extract texture well; denoising and contrast normalisation
destroy cues it can use, and water-column inpainting introduces synthetic
structure. We did **not** isolate which stage is responsible — a per-stage matched
ablation is unrun. The chain is retained as an inspectable option and still feeds
QC and the verification features. It is a genuine negative result and it costs us
some of the sonar-specific engineering in this repo.

**17. Why IoU 0.3 instead of the standard 0.5?**
*20s:* At ~24 px, a 3–4 pixel annotation offset — well within inter-annotator
agreement for sonar — drops IoU below 0.5 for a visually perfect detection. We
print the threshold in every result so it's never confused with a COCO mAP50.

**18. What's your headline metric?**
*20s:* Not mAP. It's the **false-alarm rate on empty frames** — how many of the
473 target-free test frames produced an alarm. That's the number that decides
whether an operator keeps using the system.

---

## D. Geolocation

**19. How do you turn a pixel into a coordinate?**
*20s:* Row → ping → vessel fix. Column → slant range → ground range via altitude
→ geodesic forward solution at heading ± 90°. Or, for a GeoTIFF, straight off the
affine transform.

**20. What if there's no GPS metadata?**
*20s:* We output `null` and say "Geolocation unavailable". We never estimate a
position we can't compute.
*Deep:* This is the most dangerous possible failure — a fabricated latitude looks
like data, exports cleanly, and sends a vessel to open water. There's a test that
asserts coordinates are `None` and never 0.

**21. What's your geolocation error?**
*20s:* Unknown, and we say so. The geometry and the error budget are unit tested;
positional accuracy has **never been validated**, because our dataset ships no
navigation data.
*Deep:* Each fix reports a full budget — GPS, heading × range, layback, altitude
conditioning, range resolution — combined in quadrature. On a plausible geometry
that's ~6 m at mid-swath. Validating it needs a survey with independently
surveyed object positions. That's the first thing we'd want from NIOT.

**22. Your demo shows coordinates. Are they real?**
*20s:* **No — that track is synthetic and labelled as such** in the CSV header,
the scenario metadata and the UI. It exercises the maths on a known geometry. The
other three scenarios have no navigation and correctly report unavailable.

**23. Why is uncertainty huge at the centre of the image?**
*20s:* Because it should be. Ground range = √(slant² − altitude²), so near nadir a
tiny altitude error produces an enormous range error — the inversion is
ill-conditioned. We let that show instead of hiding it, and sonar analysts
discard the nadir region for the same reason.

---

## E. Deduplication & reporting

**24. Why does detection count differ from hazard count?**
*20s:* One physical object appears in many consecutive pings. Reporting raw
detections would overstate the seabed problem several-fold. We cluster into unique
hazards — geographically when we have coordinates, by ping-sequence overlap when
we don't.

**25. Does dedup improve position accuracy?**
*20s:* Yes — averaging N independent fixes cuts random error by about √N. We floor
the improvement at half the base, because systematic terms like layback bias don't
average away.

**26. Confidence vs priority — why two numbers?**
*20s:* They answer different questions. "Is it real?" and "should you care?" A
55%-confidence 12 m net that's well located outranks a 95%-confidence 30 cm blob
with no position. Confidence is evidence; priority is policy.

**27. Is your priority formula a marine standard?**
*20s:* No, and we don't claim one. We looked and found no official derelict-gear
triage standard. The weights are transparent and adjustable per campaign.

---

## F. Deployment & performance

**28. Can it run onboard an AUV?**
*20s:* Architecturally yes; **demonstrated, no.** We've measured ONNX at 10.6 MB
and 10.8 ms on this laptop and 664 MB peak RSS. We have not run it on a Jetson or
any payload computer.
*Deep:* We won't claim Jetson performance we haven't tested. What we can claim is
that nothing in the design blocks it — pure-Python pipeline, ONNX-exportable
model, no cloud dependency, no CUDA assumption.

**29. What's the inference speed?**
*20s:* Measured on an M5: ~39 ms tiled inference on MPS versus ~278 ms on CPU —
7.1× — and ~12 frames/s end-to-end including QC, preprocessing and verification.

**30. What hardware do you need?**
*20s:* A laptop. It was developed on a 24 GB M5 with peak usage under 700 MB. CPU
fallback works; CUDA is used if present but never assumed.

**31. Does it need internet or a cloud API?**
*20s:* No. Set `AQS_OFFLINE_MAP=1` and it makes zero network requests. There are
no calls to OpenAI, Anthropic, Google or any cloud inference service in any code
path.

**32. Can it process a long survey?**
*20s:* Yes — frames stream, nothing loads the whole survey into RAM, and results
go to SQLite. At ~12 frames/s, an 8-hour survey's frames process in well under an
hour.

---

## G. Failure & trust

**33. What happens when the model fails?**
*20s:* It says so. "No confident detections found" rather than a fabricated
result. Rejected candidates stay visible with their reason. Scenario 2 in the demo
exists specifically to show the failure mode.

**34. What's the most dangerous failure mode?**
*20s:* Out-of-domain silence. On sonar from unfamiliar hardware the model will
still emit confident-looking numbers, because there's no out-of-distribution
detector. That's unsolved, and it's in `docs/LIMITATIONS.md`.

**35. How does it behave on an object class it's never seen?**
*20s:* Badly, and predictably — it has two labels, MILCO and NOMBO, so a container
or a pipeline gets forced into one of them. This is why the taxonomy has an
explicit AMBIGUOUS level and why NOMBO maps there rather than to a man-made class.

**36. Can I audit a decision?**
*20s:* Yes. Every hazard carries the top feature contributions that drove the
filter, the raw score, the calibration state, the QC score, and the full
provenance — model, device, preprocessing profile, filter and calibrator.

**37. How do I know you didn't fake the demo?**
*20s:* Every demo frame comes from the held-out test surveys. The dashboard
refuses to start without a real checkpoint. There are 108 tests, including one
that runs the whole dashboard headlessly and one that asserts a repeated run gives
identical results.

---

## H. Scale & the real world

**38. How would this generalise to Indian waters?**
*20s:* Unproven. Different seabed, different sediment, different sonar. The
pipeline is domain-agnostic; the *detector weights* are not. Retraining on NIOT
data is a fine-tuning job, not a redesign.

**39. What would NIOT actually receive?**
*20s:* A local-first repository: dataset prep, training, a fitted verification
stage, a REST API, a dashboard, JSON/CSV/GeoJSON exports that open in QGIS, and
documentation of exactly what is and isn't validated.

**40. What's the honest state of this project?**
*20s:* A working end-to-end prototype with real measured numbers on real sonar,
and a clear list of what hasn't been validated. The pipeline is the contribution;
the detector is the replaceable part, and it needs the right data.

**41. What would you do next, with a week?**
*20s:* Get access to the ghost-gear dataset and retrain — that closes the biggest
gap between what we built and what the PS asks for. Then hard-negative mining on
the 473 empty frames, then a trained torchvision backend to remove the AGPL
constraint.

**42. Why should we believe your numbers when other teams show 95% mAP?**
*20s:* Ask them how they split their data. If frames from one survey appear in
both train and test, their number measures memorisation, not detection. Ours is
lower because it's measured across surveys the model has never seen.

---

## Phase 2 additions (thesis / UATD / Indian data / new experiments)

**43. Why didn't you use SSM-DETR from the thesis?**
*20s:* 276 GFLOPs — the thesis's own cost table rules it out for edge/AUV. And our
bottleneck isn't feature saliency; it's detector recall and a large-target bias, so
a heavier detector wouldn't fix the actual failure.

**44. Why not just copy the thesis's 5-step preprocessing that gave +12.8 mAP?**
*20s:* Because that gain is on **UATD, which is Forward-Looking Sonar**. SIH-26057
is Side-Scan. We reproduced the thesis's actual 5-step pipeline and trained it
matched on SSS — mAP50 0.043 vs 0.116 raw. It hurts SSS. The FLS gain doesn't
transfer.

**45. Then why did preprocessing help them but not you? Are you sure you're right?**
*20s:* Different modality, and we measured it two ways — our own chain (0.032) and
their exact 5-step (0.043), both matched, both below raw (0.116). Their result and
ours are both true; they're just different sonar types. It's in
`research/thesis_discrepancies.md` with the modality evidence.

**46. What's the difference between UATD and SSS, and why does it matter?**
*20s:* UATD is multibeam forward-looking sonar — a forward range–bearing fan.
Side-scan images a swath to the side with grazing-incidence shadows. Different
geometry, different appearance, different shadow physics. A preprocessing chain or
detector tuned to one isn't validated for the other.

**47. Is your Indian dataset actually Indian?**
*20s:* We found exactly one genuine Indian **field** SSS source — TiHAN/IIT-Hyderabad,
Hyderabad lakes. It's access-gated by form and unlabelled, so its role is
Indian-domain *validation*, not training. We do **not** claim "validated for Indian
waters" — we have no Indian data in hand, and it's freshwater lake, not sea.

**48. What's the geographic origin of your training data?**
*20s:* MILCO/NOMBO — the authors don't disclose the region, so we don't claim it's
Indian or anywhere specific. Honest answer: unknown.

**49. Can you prove cross-domain generalization?**
*20s:* Partially — our splits are cross-**survey** (train 2015+2010, test 2018+2021,
11-year hardware gap), which is stronger than random splitting. True cross-**sensor**
generalization we can't prove without a second labelled SSS source; it's stated as
a limitation.

**50. How do you detect an object you've never trained on (anomaly)?**
*20s:* Honestly — right now we don't. We built an autoencoder anomaly branch and
measured it: AUROC ≈ 0.5, chance. Small SSS targets are too small a fraction of a
textured patch for reconstruction error to separate. We rejected it rather than
ship a fake score. The right fix is embedding-based novelty (PaDiM/PatchCore) —
future work.

**51. How does speckle affect your model, and did you fix it?**
*20s:* The raw model collapses — 0% recall retained under σ=0.25 speckle.
Speckle-augmented training partially fixes it — but we ran it to full
convergence specifically to check whether an earlier version's accuracy cost
was just undertraining. It wasn't: the fully-converged model's recall on
clean data (0.079) is actually *lower* than the undertrained one's (0.142),
even though robustness and precision both improved. It's a genuine, stable
tradeoff. We ship both — the accurate model stays primary, the robust one is
a documented alternative checkpoint — rather than pretending one dominates
the other.

**52. Your biggest failure case?**
*20s:* Large targets. Recall is 0.000 on targets over 2500 px² — we miss every big
one, while detecting the smallest best. We checked why properly rather than guess:
the largest test object is 9.3% of its frame, the largest training object only
1.7% — a 5.5× gap no augmentation can synthesize from small crops. It's a
training-data coverage gap, not a hyperparameter bug. It's in the failure gallery
and the limitations, front and centre.

**53. What did Phase 2 actually contribute if the architecture didn't change?**
*20s:* Evidence. Every proposed addition — thesis preprocessing, an anomaly branch,
a segmentation verifier — was tested and **failed to beat the Phase-1 system** on
SSS, or needs data we don't have. The contribution is knowing which boxes should
*not* be on the diagram, and why, with measurements. That's what makes it hard to
attack.

**54. Why no segmentation verifier when the thesis's SEAUNet looked strong?**
*20s:* SEAUNet is trained on mask labels. Our SSS data (MILCO/NOMBO) has boxes, not
masks. The one SSS mask dataset (AI4Shipwrecks) is shipwrecks — large targets, which
is exactly the class we already miss, so it can't verify our small-target failures.
Building a segmenter on absent supervision would be dishonest. Deferred with a reason.

---

## Phase 3 additions (LEF-RT-DETR, ghost-gear data access, final synthesis)

**55. Why didn't you use LEF-RT-DETR — it's a 2026 paper specifically on SSS?**
*20s:* We read it. It's 49.7 GFLOPs — 8× our detector — for a +4.3 AP gain on a
**970-instance, non-public, self-built dataset** we can't reproduce or compare
against. It also explicitly lists sonar-specific augmentation as unsolved future
work, which we already have (speckle-aug, E08).

**56. You said you had ghost-gear data now — do you or don't you?**
*20s:* Correction, stated plainly: I initially reported access based on a
metadata call succeeding, which was wrong — `dataset_info()` lists files for
gated repos regardless of approval; every actual file returns HTTP 403. The
dataset needs one human click ("Agree and access repository," auto-approved, no
review wait) that no API token can perform. The ingestion pipeline is fully
built and tested (`scripts/prepare_crab_pot.py`, leakage-free by recording ID)
and runs the moment that click happens.

**57. After six papers, what's actually different about your submission?**
*20s:* Not the model — every paper we reviewed proposes a heavier detector for a
few AP points, and none of them clears our edge-cost bar. What's different is
we **measured** every tempting shortcut instead of assuming it: the thesis's
preprocessing gain (rejected, it's FLS not SSS), an anomaly branch (rejected,
AUROC ≈ chance), speckle augmentation (kept, and ahead of a Nov-2025 published
paper that still calls it future work). The architecture converges with the
defence-grade blueprint (SeeByte) independently — that convergence is the
validation.

**58. What's your actual ceiling with the data you have?**
*20s:* We're data-limited, not architecture-limited. 447 training objects is the
real ceiling — no detector in six reviewed papers would do dramatically better
with the same data (LEF-RT-DETR needed 871 training images just for AP 51.6 on
3 simple geometric shapes). The system-level pipeline is stronger than the data
currently feeding it. See `research/FINAL_ARCHITECTURE.md`.
