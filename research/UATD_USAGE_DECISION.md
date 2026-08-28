# UATD — usage decision (GO / NO-GO per use)

**What UATD is (verified, 2026-08-28):** the Underwater Acoustic Target Detection
dataset. **Multibeam Forward-Looking Sonar (MFLS)**, Tritech Gemini 1200ik,
**9,200 images** (BMP) with XML box annotations, **10 classes** (cube, sphere,
cylinder, human, tyre, circle cage, square cage, metal barrel, plane, ROV),
collected in **Chinese lakes and shallow water** (Dalian; Maoming), depths 4–10 m.
Open-source on figshare `21331143`. arXiv 2212.00352; *Nature Scientific Data*
s41597-022-01854-w.

**The decisive fact:** UATD is **FLS, not SSS**. SIH-26057 specifies Side-Scan
Sonar. UATD is also not Indian. It therefore cannot be a primary training or
evaluation set for this problem, and its labels must never be merged into the SSS
taxonomy.

| Proposed use | Decision | Reasoning |
|---|---|---|
| **A. Supervised SSS training** | **NO-GO** | FLS imaging geometry differs fundamentally from SSS (range–bearing fan vs range–along-track swath with grazing shadows). Training an SSS detector on FLS would teach the wrong appearance model. |
| **B. Auxiliary sonar-domain pretraining** for the SSS detector | **CONDITIONAL GO — but low expected value, not run** | Pretraining on acoustic imagery *could* give a better init than COCO. But YOLO11n is already COCO-pretrained and our bottleneck is 447 SSS objects, not initialisation. A controlled A/B (COCO-init vs UATD-pretrained→SSS-finetune) is the only honest way to claim benefit; it is **not run** this phase, so **no benefit is claimed**. Logged as future work. |
| **C. Generic acoustic representation learning** (self-supervised features transferable to SSS) | **NO-GO for now** | Requires a self-supervised pipeline we have not built, and the FLS→SSS gap makes transfer unproven. Speculative; deferred. |
| **D. Cross-sonar robustness / domain-shift characterisation** | **GO (analysis only)** | UATD is a legitimate *contrast* corpus for documenting how far FLS is from SSS. Used only to argue modality difference in `research/thesis_discrepancies.md` and the cross-sonar note — never as training or test data for reported SSS metrics. |
| **E. Negative control / supplementary SSS data** | **NO-GO** | It is not SSS, so it is not a valid SSS negative either. |

## Bottom line

UATD's role in AQUA-SHIELD is **evidentiary, not operational**: it is the dataset
that proves the thesis's headline preprocessing/detection gains are FLS results,
and therefore that they cannot be assumed to hold for SIH-26057's SSS imagery.
It is **not downloaded into any training or evaluation path**, by design.
