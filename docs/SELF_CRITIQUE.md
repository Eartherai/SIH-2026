# Brutal self-critique

Written for us, not for judges. If any of this surprises you on presentation day,
that is a failure of this document.

---

## What is genuinely impressive

**1. The verification stage earns its place.** Most entries in this problem space
are a detector plus a UI. Building an *independent* evidence layer — ten features
measured from pixels, fitted on a held-out survey, with per-detection attribution
— is a real engineering contribution, and it directly addresses the part of
PS 26057 that most teams will gloss over.

**2. The finding that contradicted us.** `shadow_ratio` getting a negative weight
is the single most interesting thing in this project. It is a concrete
demonstration that the "obvious" textbook heuristic — *a real object casts a
shadow* — would have *hurt* precision on this data, because the darkest strips
beside a candidate are usually the nadir band. That is a genuine result, it is
reproducible, and it is the strongest evidence that we followed the brief's "do
not hand-code untested rules" instruction rather than paying it lip service.

**3. Refusal behaviour, enforced by tests.** Returning `null` coordinates,
`calibrated: false`, and "dimensions unavailable" — and having tests that *fail*
if a coordinate is ever fabricated — is unusual discipline. It is also the thing a
domain expert from NIOT will most immediately respect.

**4. The evaluation protocol.** Survey-year splits, frame-level false-alarm rate
as the headline metric, and a stated IoU threshold. Our numbers are lower than a
random split would produce, which is the point.

**5. It genuinely works end-to-end.** 90+ tests including a headless run of the
real dashboard, a determinism check, and a full pipeline integration test.

---

## What is ordinary

- **The detector.** YOLO11n fine-tuned on a public dataset. Anyone can do this.
- **The dashboard.** Streamlit. Competent, unremarkable.
- **Tiling with overlap and NMS.** Standard practice. The IoS addition is a nice
  detail, not a contribution.
- **The preprocessing chain.** Lee filtering and gain normalisation are textbook.
  Choosing them for the right reason is good engineering, not novelty.
- **CSV/JSON/GeoJSON export.** Table stakes.
- **SQLite + FastAPI.** Correct choices, zero originality.

---

## What is borrowed

- YOLO11n architecture and COCO pretraining — Ultralytics (AGPL).
- The pipeline *shape* — PINGMapper established it.
- The problem framing that precision is the constraint — GhostVision's published
  results showed it before we measured it.
- Lee filter, Platt scaling, ECE — all standard literature.
- The dataset — entirely someone else's fieldwork.

**We assembled far more than we invented.** That is normal and fine, but do not
let a slide imply otherwise.

---

## What is still weak

**1. The data is the wrong data.** This is the biggest problem and no amount of
engineering hides it. PS 26057 is about ghost nets. We trained on mine-like
contacts. The transfer argument (both are man-made-vs-clutter discrimination) is
*reasonable* — it is not *demonstrated*.

**2. Detector accuracy is low in absolute terms.** Cross-survey mAP50 in the low
teens. The explanation (447 objects, 24-px targets, 11-year hardware gap) is
sound, but a judge who only reads the number will not be impressed. Lead with the
false-alarm reduction, not with mAP.

**3. Geolocation is unvalidated.** Everything about it is unit tested except the
thing that matters: whether the coordinate is *correct*. We cannot fix this
without a survey that has both sonar and surveyed positions.

**4. Calibration rests on 30 objects.** The validation survey is thin. The ECE
improvement is real but may not transfer.

**5. "Edge deployment" is a design property, not a result.** We measured ONNX on a
laptop. We have never run this on hardware that could go in the water. Say
"architecturally ready", never "deployable on an AUV".

**6. Single domain, two classes.** The taxonomy is designed for ghost gear,
wrecks and pipes. The model knows MILCO and NOMBO. The gap between the taxonomy on
the slide and the model in the repository is real.

**7. Several designed-but-unrun items.** Hard-negative mining, a trained
torchvision backend, any hyperparameter search. Each is described accurately as
not done — but a stronger team would have *done* one of them.

**8. Small-sample statistics.** 191 test objects. Adjacent rows in the ablation
table are within noise, and we should say so rather than narrate a ranking.

---

## What a strong competing team could do better

- **Get real ghost-net data.** Request access to the gated crab-pot dataset weeks
  earlier, or partner with a fisheries agency. A team that demos an actual ghost
  net beats us on the only axis the problem statement names.
- **Partner with NIOT for Indian-waters sonar.** Any real domain data would be
  worth more than every engineering refinement here.
- **Train a bigger model properly** on pooled multi-source sonar, and report a
  respectable mAP.
- **Validate geolocation** against a surveyed target. That single number —
  "positional error 4.2 m RMS" — is worth more than our entire uncertainty-budget
  implementation.
- **Run a user study.** "Analyst review time reduced 60%, n=5 analysts" beats any
  technical metric for a Disaster Management theme.
- **Deploy on real edge hardware** and show a Jetson or a Pi processing a survey.

---

## What judges will attack

| Attack | Our answer | Strength |
|---|---|---|
| "You never detected a ghost net." | True. Dataset gated. We show the discrimination task instead. | **Weak but honest.** Lead with it before they find it. |
| "Your mAP is low." | Cross-survey split, 447 objects. Ask how they split theirs. | **Strong** — most competitors will have leaked. |
| "Mine detection isn't marine debris." | Same discrimination problem; 74% empty frames is the same false-positive challenge. | **Medium.** Rehearse this one. |
| "How do you know geolocation works?" | Geometry and error budget are tested; accuracy is not validated, and we say so. | **Weak.** No way to strengthen without data. |
| "This is just YOLO with a UI." | Point at the FP filter, the calibration, the refusal behaviour, and the shadow-weight finding. | **Strong.** This is the best answer we have. |
| "What's novel vs GhostVision?" | Learned inspectable verification vs tuned post-processing; they're single-class with an unresolved licence. | **Medium.** Be precise, don't overclaim. |
| "Can it run on an AUV?" | Architecturally yes; not demonstrated. | **Medium.** Honest, and they will respect it. |
| "AGPL in a government deliverable?" | Flagged; backend abstraction exists; clean path designed, not delivered. | **Medium.** Having noticed at all puts us ahead. |

---

## What must be fixed before SIH

**Must:**
1. Fill every **[TBD]** in `docs/SIH_SLIDES.md` from `docs/BENCHMARKS.md`. No
   unverified number on a slide.
2. Re-verify GhostVision's exact metrics against the PDF — our figures come from
   an indexed abstract because the publisher returned 403 to automated fetch
   (`research/sources.md`, item 7).
3. Rehearse the "you never detected a ghost net" answer until it is 15 seconds and
   completely unflustered.
4. Pre-warm the model before the live demo — the first MPS call takes ~3 s.

**Should:**
5. Run hard-negative mining. It is cheap (473 empty test frames), it is already
   designed, and it converts a "not done" into a measured result.
6. Train the torchvision backend even briefly, so the licence-clean path is
   demonstrated rather than asserted.
7. Request access to the crab-pot dataset now. If it arrives before submission,
   everything changes.

**Nice:**
8. A short screen recording of the demo, in case live inference misbehaves.
9. Reduce QC cost — at ~74 ms it is a large share of the per-frame budget.

---

## The honest one-line summary

*A well-engineered, unusually honest operational pipeline wrapped around a
modest detector trained on the wrong dataset, because the right dataset was
locked.* The engineering is genuinely good and the scientific discipline is
better than most. The gap between what PS 26057 asks for and what we can
demonstrate is real, and our best move is to name it first.
