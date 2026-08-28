# Ready-to-use repositories — competitive landscape survey

Searched GitHub directly (not memory) on 2026-08-28 for existing side-scan
sonar / marine-debris / ghost-gear detection repositories — both established
research code and, notably, several **other teams' live SIH-26057
submissions** pushed in the same window as this project. Evaluated for
licence, completeness, and whether anything should be adopted instead of, or
merged into, AQUA-SHIELD.

**Bottom line: nothing found should replace this repository.** Two items are
genuinely useful as future-work leads (one dataset source, one architecture
idea); everything else is either the wrong task, unlicensed, or an
unverified competing submission with no published measurement discipline.

---

## Established research repositories

| Repo | Stars | Licence | What it is | Verdict |
|---|---|---|---|---|
| `remaro-network/KD-YOLOX-ViT` | 96 | **Apache-2.0** | YOLOX + optional ViT layer + knowledge distillation, evaluated on SWDD (Sonar Wall Detection Dataset, Zenodo, 864 train images + 6,243 video frames). Published: Aubard et al., arXiv:2403.09313. | **Wrong task** (underwater tunnel/dam *wall* detection, not debris) — taxonomy doesn't transfer. But two findings corroborate our own: (1) their own README states training took **one week on an RTX 3070 Ti** for 300 epochs — even a real desktop GPU finds this expensive, reinforcing our edge-cost discipline; (2) they report the ViT layer **increased false positives** even as raw detections rose — the same precision-for-recall trade we've measured repeatedly and the reason we invested in a learned FP filter rather than a heavier backbone. |
| `gx-123/SeabedObject-SSS` | 11 | **None** | Real SSS imagery with Aircraft/Human/Shipwreck class folders, 179 MB. | **Cannot be used.** No licence file means all rights reserved by default (GitHub ToS) — using it would require contacting the author for explicit permission, which was not pursued this session. Logged for future follow-up. |
| `firekeepers/DCBD`, `Yang-Code984/FSR_Sonar`, `ahmad-kaif/UnderWaterObjectDetection`, `goreswapnil/Naval-Mine-Detector` | ≤10 | none/unclear | Various shipwreck/mine detection research code. | Checked, not pursued — small, unlicensed or narrowly-scoped, no ghost-gear relevance beyond what MILCO/NOMBO already provides. |

## Other teams' SIH-26057 submissions (found live, same problem statement)

Several repositories explicitly named for this exact hackathon problem
statement were found, pushed within hours to days of this session:

| Repo | Stack | Pushed | Notable |
|---|---|---|---|
| `aditisingh1010/SIH26057-Marine-debris-Detection` | YOLOv8n, FastAPI backend, React+TS+Vite+Leaflet frontend, ONNX export, pytest | 2026-08-28 | **Independently states the identical design rule we use**: *"Coordinates are only shown when real survey metadata is attached. They are never invented."* Also applies preprocessing (bilateral speckle filter) without any published measurement of whether it helps — exactly the untested-preprocessing pattern we tested twice and found harmful. No published accuracy metrics, no cross-survey/leakage discussion in the README. |
| `Dinoman67/sonarvision` | 3-way architecture comparison (YOLOv8n baseline, GhostConv+FastC2f "SS-YOLO", C2f+SE-attention "YOLOv8-ESI"), MIT licence | 2026-08-28 | Built on **NOAA H11833**, a real public-domain hydrographic side-scan sonar survey (verified: NOAA/NCEI hydrographic survey archive is real and includes geo-referenced SSS mosaics). Their own README documents discovering the identical false-positive problem we did: *"Model detected everything as debris on real noisy SSS data"* after training only on clean backgrounds — independently converging on noise-augmentation as the fix, the same conclusion behind our own speckle-augmentation experiment (E08/E09). No published held-out accuracy numbers found. |
| `Kira-Stargazer/Aquadex-AI`, `nikunjdixit-ai/AI-Sonar-Marine-Debris-Detection`, `FobusMDJ/SonarSense` | various | 2026-08-23 to 08-26 | Similarly-scoped submissions; `SonarSense` repo is empty at time of check. Not deeply reviewed — no licence, no published metrics visible from the API/README survey. |

**Reading this honestly:** these are competent, live competing submissions,
not toy projects. Two of them (`aditisingh1010`, `Dinoman67`) **independently
arrived at design principles central to AQUA-SHIELD** — never fabricate a
coordinate, and noise/false-positive suppression is the real problem, not raw
detection. That convergence, reached without any of us seeing each other's
work, is stronger evidence that these are the correct starting principles for
this problem than any single benchmark number would be.

**What none of them appear to have, based on public README/repo survey
(not a full code audit of private branches):** a leakage-free
cross-survey/cross-recording split methodology, a *learned* (as opposed to
hand-coded/heuristic) false-positive filter with measured precision/recall
tradeoffs, a matched preprocessing ablation, a documented negative result, or
a published held-out accuracy number with a stated evaluation protocol. This
is not a claim that their systems perform worse — no independent
benchmark exists to say so — only that AQUA-SHIELD's differentiator remains
what it always was: **measured, not asserted, engineering discipline.**

## New leads for future work

1. **NOAA hydrographic survey side-scan sonar mosaics** (H-series, e.g.
   H11833, H11251) — verified real, via NOAA's National Centers for
   Environmental Information (NCEI) hydrographic data archive. **US federal
   government data is public domain**, making this a legally clean SSS
   source. **Caveat:** these are raw geo-referenced mosaics, not
   pre-annotated for debris detection — the "marine debris" variant used by
   `sonarvision` appears to be their own curation/labelling on raw NOAA
   imagery, and no evidence was found that a pre-labelled version is
   publicly redistributed. Using this would mean a labelling effort, not a
   ready-made dataset. Logged in `research/dataset_role_matrix.md` as a
   future candidate, not adopted this session.
2. **Knowledge distillation** (KD-YOLOX-ViT's core contribution, separate
   from its ViT layer) is architecturally interesting because, unlike every
   other "heavier architecture" idea rejected in this project, **it costs
   nothing extra at inference** — only training uses a larger teacher model.
   Not attempted this session (would require a teacher model and a
   multi-stage training script we don't have), but it is the one idea from
   this survey that doesn't trade edge-deployability for accuracy, and is
   worth a future matched experiment.

## What we did NOT do

We did not clone, copy, or incorporate any code from the repositories above.
Review was limited to public READMEs, file listings, and licence metadata via
the GitHub API — the same due-diligence process applied to every other
external reference in this project (`research/prior_art.md`,
`research/external_architectures.md`).
