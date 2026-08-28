# External architectures & real-world systems — analysis and mapping

Five more references (Phase-2b). For each: what it does, **what AQUA-SHIELD
already does**, what is genuinely new to try, and the verdict. Licences/costs
verified where possible; metrics are as-reported by each source (not re-measured
by us).

## The one-line takeaways

1. **SeeByte SeeTrack/Neptune ATR** — the defence gold-standard pipeline is
   *Raw ingest → ATR inference → confidence-scored contact list → analyst review →
   GIS report*. **That is exactly AQUA-SHIELD's pipeline.** Strong architectural
   validation.
2. **MSF-DETR (PLOS ONE 2025)** explicitly lists *"no sonar-specific augmentation
   (speckle, TVG, acoustic shadow, reverberation)"* as a **missing differentiator**.
   **We already did speckle augmentation (E08)** — so on that axis we are ahead of a
   Nov-2025 paper. Its other lesson — *"small-object-aware fusion, not architecture
   complexity, gave the biggest win"* — validates our refusal to build a 50–276
   GFLOP transformer.
3. **The benchmark report's proposed "novelty"** is a *hand-weighted* confidence
   `C = w1·shadow + w2·edge + w3·temporal`. **Our learned FP filter already fits
   those exact weights from data** (shadow, edge, contrast, persistence) instead of
   hand-tuning them — a cleaner, defensible version of their proposal.

## Detection/segmentation papers

| Paper | Core idea | Cost | What we already do | New to try | Verdict |
|---|---|---|---|---|---|
| **TR-YOLOv5s** (Yu 2021, *Remote Sensing*) | Transformer in YOLOv5 + **cross-track downsampling** (anisotropic-resolution fix) + overlapping 320px/40% patches + 8-bit log requant + transfer learning | 16.2 GFLOPs | tiling+overlap (seam-merged), COCO transfer, attention (YOLO11 C2PSA) | **cross-track downsampling** for SSS anisotropy; 8-bit log requant | Downsampling is genuinely SSS-aware and untried — **deferred** (needs matched retrain, per our preprocessing track). Their mAP 85.6% is on **shipwrecks (large targets)** — the exact class we miss. |
| **MSF-DETR** (Zhao 2025, *PLOS ONE*) | Spatial+**frequency (Gabor)** backbone, MAFM attention-fusion FPN, sparse WASSA attention | **50.4 GFLOPs (8× ours)** | multi-scale detection head; empty-frame-heavy test set | frequency-domain features; attention-based scale fusion | **Rejected as primary** (8× our compute, edge-hostile). Their own lesson favours *fusion over complexity*; and they lack speckle aug, which we have. Useful: their Pareto framing; KLSG as cross-domain test. |
| **BHP-UNet** (Tang 2023, *EURASIP JASP*) | UNet + hybrid-dilated conv + **anti-noise blending** + Pyramid-Split-Attention for SSS **segmentation** | 73.2 MB | man-made-vs-natural discrimination via learned FP filter; mosaic/noise-sim aug | segmentation masks for footprint/extent; anti-noise blending loss | Segmentation **deferred** — we have no SSS mask labels (MILCO/NOMBO is boxes; AI4Shipwrecks masks are wrecks = large targets we miss). Anti-noise-blending idea noted for a future seg head. |

## Real-world operational systems (benchmark report)

| System | Architecture | What we borrow | Gap that leaves room for us |
|---|---|---|---|
| **GhostNetZero.ai** (WWF/Accenture/MS) | DeepLabV3+ResNet50 seg on Azure A100; NMEA/GPS/USBL geotag → raster masks | segmentation-+-geotag pattern; metadata-driven geolocation (we do this) | pushes low/med confidence to **manual human review** — we automate that with the learned FP filter |
| **SeaClear** (EU Horizon) | multi-robot sensor fusion; Shore Operation Center web app; 80% detect / 90% collect targets | **dual-tier UI** (operator vs executive) — **now implemented** | heavy physical-retrieval hardware; out of scope for a software pipeline |
| **SeeByte SeeTrack + Neptune ATR** (defence) | Ingest → ATR → probabilistic contacts → analyst → GIS | **the pipeline blueprint — identical to ours** | proprietary/classified; no code/data — structure only |
| **NOAA ERMA / MDMAP** | Web-GIS common operating picture; REST + JSON/CSV export | export-schema + map-first UX patterns | manual citizen surveys; no automated acoustic CV ingest |

## Changes made from this analysis
- **Dual-tier dashboard** (Operator / Executive summary) — SeaClear+SeeByte pattern.
- This document + prior-art matrix updated to log the validation and the two
  concrete borrowables (dual-tier UI ✅; cross-track downsampling — deferred).
- No heavy model adopted: every reference ≥50 GFLOPs or ≥73 MB is edge-hostile
  vs our 6.3 GFLOPs / 10.6 MB ONNX; "fusion not complexity" and our own
  small-object analysis both say the win is in verification and data, not a bigger
  detector.

## Net positioning
AQUA-SHIELD's pipeline **is** the defence-grade blueprint (SeeByte), does the
confidence automation the conservation systems (GhostNetZero) lack, learns the
weighted-evidence confidence the benchmark report only proposes, and already has
the sonar-specific augmentation a 2025 SOTA paper (MSF-DETR) flags as missing —
at a fraction of everyone's compute. The honest gaps remain the ones we already
document: large-target recall, ghost-gear data, and geolocation validation.
