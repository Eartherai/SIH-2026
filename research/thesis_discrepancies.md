# Thesis claims — verification and discrepancies

Per the Phase-2 source-of-truth rule, factual claims about the thesis are ranked:
(1) original thesis PDF, (2) associated published paper, (3) the analysis-report
summary, (4) our inference.

**Availability, stated plainly:** in this session only the **analysis-report PDF**
(`Thesis_Analysis_Report_SIH26057.pdf`, 12 pages, author Divyabarathi G., CUSAT
2025) was provided. The **original thesis PDF was not attached**, and the
associated papers were not fetched in full. Therefore every row below is verified
at best to tier (2)/(3). Claims that require the original thesis to confirm are
marked **UNVERIFIABLE (no primary source)** rather than silently accepted or
rejected. This is a limitation of the inputs, not a judgment on the thesis.

Verified via web (UATD, Marine-Debris dataset provenance, SSM-DETR cost) on
2026-08-28.

---

## The one discrepancy that changes everything: modality

The report's headline recommendation is that the **5-Step Preprocessing** (TVG →
Median → HistEq → CLAHE → Morphology) is **"CRITICAL"** and **"directly
adoptable"** for SIH-26057, citing a **+12.8-point mAP gain** (0.854 → 0.963).

**That gain was measured on UATD.** The report's own dataset table (page 8) lists
UATD as **FLS** — Forward-Looking Sonar. Verified independently: UATD is a
**Multibeam Forward-Looking Sonar** dataset (Tritech Gemini 1200ik), 9,200 images,
collected in **Chinese lakes/shallow water** (Dalian, Maoming), arXiv 2212.00352 /
*Nature Scientific Data* s41597-022-01854-w.

**SIH-26057 specifies Side-Scan Sonar.** FLS and SSS are different imaging
geometries: FLS images a forward fan in range–bearing; SSS images a swath in
range–along-track with grazing-incidence shadows. A preprocessing chain tuned to
one is not established for the other.

This **resolves the apparent contradiction** between the thesis and our Phase-1
result. The thesis: preprocessing helps **FLS** detection. AQUA-SHIELD Phase-1:
the same *kind* of preprocessing **hurts SSS** detection, measured under matched
train/inference (E06: mAP50 0.032 preprocessed vs 0.116 raw). **Both can be true
at once** — they are different modalities. Neither refutes the other.

We are now testing the thesis's *actual* 5-step pipeline on SSS under matched
conditions (experiment **E07**, `docs/BENCHMARKS.md`), so the comparison uses the
thesis chain rather than our own.

---

## Discrepancy table

| # | Topic | Analysis report says | Verified interpretation | Tier |
|---|---|---|---|---|
| 1 | 5-step preprocessing benefit | "+12.8 mAP, CRITICAL, directly adoptable for SIH-26057 (SSS)" | Gain is real **but measured on FLS (UATD)**. No SSS preprocessing benefit is demonstrated in the thesis. Our matched SSS experiments (E06; E07) find it does **not** transfer. Adopt with SSS evidence only. | (2) web-verified modality |
| 2 | "Marine Debris SSS" dataset (page 8) | Labeled **SSS** | **Actually FLS.** It is the Valdenegro-Toro/Singh Marine Debris dataset, ARIS Explorer 3000 **Forward-Looking Sonar**, watertank/turntable, Heriot-Watt Edinburgh. The report's "SSS" label is an error. | (2) web-verified |
| 3 | "Both FLS and SSS used, making techniques directly applicable to SSS" (page 8) | Cross-modality transfer implied | **Overclaim, and internally contradicted:** the report's own limitations (page 10) list *"No Cross-Dataset Generalization — cross-sensor and cross-environment transfer not validated."* The thesis does not demonstrate FLS→SSS transfer. | (3) internal contradiction |
| 4 | Canonical preprocessing = the single 5-step chain | Universal TVG→Median→HistEq→CLAHE→Morph before all models | The Phase-2 prompt states the original thesis uses **different** preprocessing per chapter (normalisation, Wiener, bilateral, CLAHE, Gaussian blending, a separate YOLO framework). The report shows only the 5-step. **Cannot confirm which is canonical without the original thesis.** | **UNVERIFIABLE (no primary)** |
| 5 | Exact detection numbers (YOLOv8 0.854→0.963, ensemble 97.91%, SEAUNet mIoU ~0.78) | Reported to 3 sig figs | Plausible and internally consistent, but **not checked against the primary papers** (IRJMS / SIViP / The Visual Computer). Treat as report-sourced. Do **not** put on a slide as our measurement. | (3) report-sourced |
| 6 | SSM-DETR cost 41.58 M params / 276.29 GFLOPs, "not viable for edge" | Too heavy | **Agrees with our position.** We independently reject SSM-DETR for the edge target. Verified as consistent with the transformer-detector class. | (3), consistent |
| 7 | "No ghost net detection" in the thesis | Listed as a gap | **Confirmed and consistent** with our own Phase-1 finding. The thesis targets mine-like / debris objects, never ALDFG. | (3), consistent |
| 8 | UATD "10 classes (ball, cube, cylinder, tire...)" | 10 classes | Verified: cube, sphere, cylinder, human, tyre, circle-cage, square-cage, metal-barrel, plane, ROV. Report's "ball" ≈ sphere; substance correct. | (2) web-verified |

---

## What this means for our decisions

1. **We do not adopt the 5-step preprocessing on the strength of the thesis.** Its
   evidence is FLS; SIH-26057 is SSS. We test it ourselves on SSS (E07) and let
   the matched measurement decide. Early Phase-1 evidence and the modality argument
   both predict it will not help — but we measure rather than assume, in either
   direction.
2. **We do not treat UATD as an SSS dataset.** Its role is auxiliary at most; see
   `research/UATD_USAGE_DECISION.md`.
3. **We do not repeat the report's "directly applicable to SSS" framing.** The
   thesis's own limitations section contradicts it.
4. **The thesis's genuinely useful, modality-independent ideas** — edge-adaptive
   attention for segmentation (SEAUNet/EAAG), an anomaly branch, edge/structure
   features for verification, and the edge-deployment discipline (skip the 276-GFLOP
   transformer) — are worth testing on SSS, and are evaluated as such in Phase 2.
