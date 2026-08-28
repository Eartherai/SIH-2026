# Prior-art matrix

| Existing system | What it solves | What we reuse | Its limitation for PS 26057 | AQUA-SHIELD contribution |
|---|---|---|---|---|
| **GhostVision** (JMSE 2025, licence NOASSERTION) | Derelict crab-pot detection + georeferencing from low-cost SSS | Nothing (code not vendored — licence unresolved). We adopt its *problem framing*: precision is the binding constraint. | Single class, single sonar grade; precision limited (untuned F1 0.512 @ recall 0.922); licence blocks government reuse | A **learned, feature-based verification stage** whose weights are inspectable, replacing hand-tuned post-processing |
| **PINGMapper** (MIT) | Decoding, per-ping attributes, water-column removal, georectified mosaics | The pipeline *shape*, and its output-naming conventions in our ingestion adapters | Processing only — no detection, classification, confidence or reporting. Recreation-grade hardware only | Detection, verification, calibration, dedup, priority and operational reporting layered on top |
| **sidescantools** (GPL-3.0) | SSS correction and georeferenced export | Nothing (GPL would infect the combined work) | Copyleft; no detection | A permissive-path implementation of the corrections we actually need |
| **AI4Shipwrecks** (MIT site) | Shipwreck **segmentation** benchmark | Nothing yet | Single class; large easy targets; not a debris/clutter problem | Multi-level taxonomy with an explicit AMBIGUOUS class |
| **MILCO/NOMBO** (CC BY 4.0) | Real annotated SSS with a man-made-vs-ambiguous distinction | **Training data, and all reported metrics** | No ghost gear; no navigation data; small (668 objects) | Leakage-free **survey-year** splits + frame-level false-alarm metrics |
| **Manual analyst review** | The current operational method | The interpretation logic (shadow geometry) encoded as measurable features | Slow, fatiguing, inconsistent across analysts | Automated triage that ranks and reports, with the analyst kept in the loop |

## What is genuinely ours, after checking

Assessed honestly against the above:

| Claim | Verdict |
|---|---|
| "AI detects marine debris in sonar" | **Not novel.** GhostVision, and a large literature, already do this. |
| "Detection + georeferencing + mapping" | **Not novel.** GhostVision + PINGMapper already deliver this combination. |
| Learned FP filter over *physically-motivated* features, with per-detection attribution | **Defensible contribution.** Prior systems tune thresholds; we fit and inspect weights — and the weights localised a train/inference preprocessing defect in our own pipeline that aggregate metrics had not (see `docs/ML_PIPELINE.md`). |
| Explicit confidence **calibration** separated from raw detector score, with reliability measured | **Defensible contribution.** Rare in this application area. |
| Refusing to emit a coordinate without metadata, and reporting a per-fix **uncertainty budget** | **Defensible contribution**, and mostly an engineering-honesty choice rather than a research one. |
| Confidence vs **priority** as separate quantities | **Product contribution**, not research novelty. |
| Survey-year-level splitting + frame-level false-alarm rate as the headline metric | **Methodological contribution** — it makes the numbers comparable to operational reality. |
| Edge/onboard deployment | **Not demonstrated.** Runs locally on Apple Silicon; no Jetson or AUV test. Claimed only as an architecture property. |

## Additional references (Phase 2b)

Five more architectures and four real-world operational systems (TR-YOLOv5s, MSF-DETR, BHP-UNet, GhostNetZero, SeaClear, SeeByte, NOAA) are analysed in **`external_architectures.md`**, with the same what-they-do / what-we-already-do / new-to-try / verdict framing. Headline: SeeByte's defence pipeline is identical to ours; MSF-DETR (2025) flags sonar-specific augmentation as a missing differentiator that we already have (E08); every heavy model is edge-hostile vs our 6.3 GFLOPs.
