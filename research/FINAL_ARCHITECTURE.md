# Final architecture & winning approach

The synthesis across everything: Phase 1 (baseline system + 9-stage pipeline),
Phase 2 (thesis, UATD, Indian-data research, matched preprocessing/speckle-aug/
anomaly experiments), Phase 2b (TR-YOLOv5s, MSF-DETR, BHP-UNet, GhostNetZero,
SeaClear, SeeByte, NOAA), and now LEF-RT-DETR — **six independent papers plus
four real-world operational systems**, checked against our own measurements
every time. This is the answer to "what should we actually build and present."

---

## 1. The verdict, in one paragraph

Build the **smallest system that survives scrutiny**, not the most sophisticated
one. Every reference — a hackathon thesis, three 2021–2026 detection papers, and
four production/defence systems — independently arrives at the same shape:
**ingest → detect (lightweight) → verify against evidence → calibrate → localise
→ report**, with the verification stage doing the work that separates a demo from
a deployable tool. **Nothing in six papers beat that shape; several tried heavier
detectors and none of it survived an edge-cost check.** Our own measurements
agree, including two results (thesis-5-step, speckle-aug) that we *tested
ourselves rather than assumed* — which is the actual differentiator.

## 2. Final production architecture (unchanged in shape since Phase 1 — now with six independent validations)

```
RAW SSS ─▶ QC ─▶ [preprocessing OFF by default] ─▶ TILING ─▶ YOLO11n (raw-trained)
                                                                   │
   REPORT ◀─ PRIORITY ◀─ GEOLOCATION (or refuse) ◀─ DEDUP ◀────────┤
                                                                   │
                                   CALIBRATION ◀─ LEARNED FP FILTER ┘
```

| Component | Decision | Evidence (this project) | External validation |
|---|---|---|---|
| **Detector: YOLO11n, raw input** | Primary | mAP50 **0.116**, 6.3 GFLOPs, 16 MB, 21 ms/frame MPS | Every paper (TR-YOLOv5s 16.2, LEF-RT-DETR 49.7, MSF-DETR 50.4 GFLOPs) needs 2.5–8× our compute for a few AP points — none clear an edge bar we already clear |
| **No preprocessing at inference** | Kept | Our chain (E06) 0.032, thesis 5-step (E07) 0.043, both **< raw 0.116**, matched | The thesis's own +12.8 mAP gain is on **FLS**, not SSS — doesn't transfer (`thesis_discrepancies.md`) |
| **Tiling + IoU/IoS seam merge** | Kept | targets ~24 px; full-frame loses them | TR-YOLOv5s: identical rationale, 320px/40%-overlap patches |
| **Learned FP filter (not hand rules)** | Kept — **the headline result** | precision 0.247→**0.322**, false-alarm frames 37→**25**/473, keeps 19/21 TPs; beats hand rules (12/21 kept) | SeeByte's "confidence-scored contact list" is the same idea; the benchmark report's hand-weighted formula proposal is what we *learn* instead |
| **Platt calibration** | Kept | ECE improves on fit split; `calibrated:false` when unfit | — |
| **Dedup (geo/sequence)** | Kept | N obs → 1 hazard; √N uncertainty reduction | — |
| **Geolocate-or-refuse** | Kept | never fabricates a coordinate; test-enforced | GhostNetZero: same metadata-driven geotagging pattern |
| **Speckle-augmented training** | **Secondary / promotion candidate** | speckle recall retention **0%→~41%** at σ=0.25; clean mAP trade 0.116→0.076 (undertrained) | LEF-RT-DETR (2026) still lists this as unsolved future work — we're ahead |
| **Anomaly autoencoder** | **Rejected, evidence-based** | AUROC 0.47 (frame) / 0.54 (patch) — chance | Thesis and PS 26057 want this; nobody in the six references solves it either |
| **Segmentation head** | **Deferred, reasoned** | no SSS mask labels on debris-scale targets (AI4Shipwrecks masks are large wrecks) | BHP-UNet/SEAUNet both need mask supervision we don't have |
| **Heavy transformer (SSM-DETR/MSF-DETR/LEF-RT-DETR)** | **Rejected × 3** | our bottleneck is recall/large-target bias, not feature saliency | consistent 8–44× GFLOPs for single-digit AP gains, on datasets we can't even access |
| **Dual-tier UI** | Kept | Operator (6-tab technical) / Executive (decision summary) | SeaClear + SeeByte both split operator vs stakeholder views |

## 3. Best achievable right now, on this dataset, on this M5

**Ceiling, honestly stated:** we are data-limited, not compute- or
architecture-limited. 447 training objects on a single undisclosed-region SSS
survey series is the actual ceiling on detector accuracy — no architecture in the
six papers reviewed would out-run that with the same data (LEF-RT-DETR needed
871 training *images* with 685 targets just to reach AP 51.6 on 3 simple
geometric classes; MSF-DETR used 2,182). **The system-level pipeline is stronger
than the data it's currently running on.**

What we can state as fact on this hardware, this data, today:

- **mAP50 0.116** cross-survey (2018+2021 held out from 2015+2010 training) — low
  in absolute terms, honestly measured, not inflated by a random split.
- **Precision +30%, false alarms −32%** from the learned FP filter alone — this is
  the number that matters for PS 26057's actual ask (74% of frames are empty).
- **21 ms/frame on MPS, 37 fps, 640 MB peak, 10.6 MB ONNX** — genuinely real-time
  and genuinely edge-viable, unlike five of six reviewed detector papers.
- **Speckle robustness is a solved *mechanism*, not yet a solved trade** — the
  fix works, needs a longer run to stop costing clean accuracy.
- **Ghost-gear detection is still zero** — not a modelling problem, a **data
  access** problem (see §4).

## 4. What actually changes the ceiling: data, not architecture

Ranked by expected impact, all confirmed feasible with resources already in hand:

1. **Ghost-gear (crab-pot) dataset access** — 6,674 real SSS images, genuinely on
   the ghost-gear theme PS 26057 names. **Blocked on exactly one human action**:
   visiting the HF dataset page and clicking "Agree and access repository" (gate
   type is `auto` — no waiting for review). `scripts/prepare_crab_pot.py` is
   written, tested (recording-level leakage-free split, same discipline as
   MILCO/NOMBO), and ready to run the instant that click happens. This is the
   single highest-leverage remaining task in the entire project.
2. **Indian-domain validation data** — TiHAN/IIT-Hyderabad SSS (Hyderabad lakes),
   also one human form-submission away (`research/INDIAN_SONAR_DATA.md`).
3. **Longer speckle-aug run** — pure compute, no new data; recovers the clean-mAP
   trade documented in §2.
4. **Large-target recall fix** — pure retraining discipline (rebalance
   scale-augmentation against the cross-survey size distribution); the failure is
   diagnosed (`docs/BENCHMARKS.md` §10.3), the fix is not yet applied.

Nothing above requires new architecture. That is the finding.

## 5. The winning SIH narrative

Not "we built the most advanced sonar detector." **We built the most honestly
verified pipeline**, and we can prove it two ways a judge can check live:

1. **Show the false-positive engine working**: natural-seabed frame → detector
   raises alarms → verification removes them (`docs/images/verification_effect.png`).
2. **Show the failure gallery**: a large target we miss, a false positive we
   catch, a target that survives speckle after augmentation and one that doesn't
   without it (`docs/images/failure_gallery.png`). **State every weakness before
   being asked.**

The differentiator is not a bigger model. It is:
- A pipeline structurally identical to the **defence gold standard** (SeeByte),
  reached independently.
- A **measured**, not assumed, rejection of the two most tempting shortcuts
  (adopting the thesis's FLS preprocessing gain; building an anomaly branch that
  doesn't work) — with the negative results kept as evidence, not hidden.
- **Speckle-robustness training that a November-2025 published paper still lists
  as future work.**
- A refusal architecture (no fabricated coordinates, no fake calibration, no
  anomaly score we don't trust) that a domain expert will recognise as
  operationally serious rather than hackathon-grade.

## 6. Future direction (priority order)

| # | Task | Blocker | Effort |
|---|---|---|---|
| 1 | Ghost-gear training on crab-pot data | **one manual click** (`HF gate`) | ~1 day once unblocked: prep→train→fit→evaluate, pipeline already built |
| 2 | Longer speckle-aug run, promote to primary | none — compute only | few hours on M5 |
| 3 | Fix large-target recall | **re-diagnosed as a 5.5× data-coverage gap, not an augmentation bug** — needs more large-object training data or paste-augmentation | half-day+, data-dependent |
| 4 | Indian-domain validation (TiHAN/IITH) | one manual form | prep pending access |
| 5 | Embedding-based anomaly (PaDiM/PatchCore) to replace the failed AE | none | ~1 day, new subsystem |
| 6 | Train the torchvision backend → licence-clean (non-AGPL) path | none | ~1 day |
| 7 | Cross-track downsampling (TR-YOLOv5s idea), matched retrain | none | half day, follows the preprocessing-testing discipline already established |
| 8 | Multi-dataset combined detector (MILCO/NOMBO + crab-pot, taxonomy-mapped) | depends on #1 | after #1 lands |

## 7. What we will NOT do, and why

- **Adopt any transformer detector** (SSM-DETR, MSF-DETR, LEF-RT-DETR) as
  primary — 8–44× our compute for single-digit AP gains on data we can't
  reproduce, and the thesis/paper authors' own stated bottlenecks (small-sample
  generalisation, edge inference) are exactly our bottlenecks too.
- **Ship the anomaly branch** — AUROC ≈ chance is worse than no score at all.
- **Claim Indian-waters validation** — we have no Indian data in hand.
- **Claim ghost-gear detection accuracy** — we have never trained on ghost gear.
- **Re-enable preprocessing by default** — measured twice (our chain, the
  thesis's chain), both times worse than raw, both times matched.
