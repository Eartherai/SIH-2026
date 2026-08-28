# Data Card

Provenance, geography, sonar type, labels, licence, role, and known domain limits
for every dataset AQUA-SHIELD touched. Sonar type is stated first because it
gates everything: FLS labels never enter the SSS taxonomy.

## Used for training / metrics

### MILCO/NOMBO  — the only dataset behind any reported number
- **Provenance:** Pessanha Santos & Moura (2024), *Data in Brief* 53:110132; figshare `10.6084/m9.figshare.24574879`.
- **Geography:** not disclosed by the authors. **Not established as Indian.**
- **Sonar:** **Side-Scan Sonar**, Marine Sonic dual-frequency 900–1800 kHz, Teledyne Gavia AUV, 2010–2021.
- **Resolution:** 416×416 and 1024×1024.
- **Labels:** 668 boxes over 1,170 frames (74% empty); classes MILCO (mine-like), NOMBO (not-mine-like bottom object).
- **Licence:** **CC BY 4.0** (open, commercial OK with attribution).
- **Role:** supervised training + all held-out metrics; survey-year splits.
- **Domain limits:** no ghost gear; no navigation metadata; small (668 objects); region unknown.

## Evaluated, not in any training/metric path

### UATD — **FLS, evidence-only**
- MFLS (Tritech Gemini 1200ik), 9,200 imgs, 10 classes, **Chinese lakes** (Dalian/Maoming). Open (figshare 21331143). arXiv 2212.00352.
- **Role:** modality-contrast evidence only. Proves the thesis's headline preprocessing gain is FLS. See `UATD_USAGE_DECISION.md`. **NO-GO for training/test.**

### Marine Debris (Valdenegro-Toro/Singh) — **FLS, watertank**
- ARIS Explorer 3000 **Forward-Looking Sonar**, watertank/turntable, **Heriot-Watt, Edinburgh, Scotland**. ~1,868 imgs, 11 debris classes + bg. GitHub `mvaldenegro/marine-debris-fls-datasets`.
- **Role:** debris-morphology / segmentation-pretraining *candidate* only, with a heavy FLS + controlled-tank domain-shift caveat. Not field data. The analysis report's "SSS" label for it is **wrong**.

### sss-crab-pot-detection-ds — SSS, **gated**
- SSS, Delaware bays (US), ~6,674 imgs, Crab-Pot / Maybe-Crab-Pot. **HTTP 403 without approval.** DOI 10.57967/hf/8397. Licence ambiguous (cc-by-sa-4.0 vs README "GPL").
- **Role:** the closest ghost-gear match; adapter ready (`ingestion/jsonl_bbox.py`); **never trained on** (no access).

### AI4Shipwrecks — SSS masks (US)
- 286 SSS images, shipwreck **segmentation** masks, Michigan. MIT (site).
- **Role:** future `shipwreck_structure` segmentation class; large-target verifier data. Not used yet.

### KLSG — SSS seabed (China), by request
- **Role:** natural-seabed hard negatives (rocks, ripples, wreck-like clutter). Access pending.

### TiHAN / IIT-Hyderabad SSS — **Indian, gated, unlabelled**
- **Side-Scan** (SSS-600K), **Hyderabad lakes, India**, `.xtf`, no annotations, form-gated. Freshwater.
- **Role:** Indian-domain **validation / hard negatives**, pending manual access. See `INDIAN_SONAR_DATA.md`. **Not** supervised training (no labels), **not** marine.

### S3Simulator — synthetic SSS (India-authored)
- Synthetic; not field data. Optional future augmentation only.

## Derived datasets we built (all from MILCO/NOMBO, for controlled experiments)
| Name | Transform | Split integrity | Purpose |
|---|---|---|---|
| `milco_nombo_yolo` | survey-year splits of the raw data | leakage-free | training + metrics |
| `milco_nombo_yolo_pp` | our preprocessing chain baked in | same splits | matched preprocessing test (E06) |
| `milco_nombo_yolo_thesis5` | thesis 5-step baked in | same splits | matched thesis-preprocessing test (E07) |
| `milco_nombo_yolo_speckle` | train += speckle copies; val/test clean | same splits | speckle-augmentation test (E08) |
