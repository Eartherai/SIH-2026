# Datasets

## In use: MILCO / NOMBO

| Field | Value |
|---|---|
| Name | Side-scan sonar imaging for mine detection (MILCO/NOMBO) |
| Source | Pessanha Santos, N. & Moura, R. (2024), *Data in Brief* 53:110132 |
| Data DOI | `10.6084/m9.figshare.24574879` |
| Paper DOI | `10.1016/j.dib.2024.110132` |
| Licence | **CC BY 4.0** |
| Access | Public download, no registration |
| Sensor | Marine Sonic dual-frequency side-scan sonar |
| Frequency | 900–1800 kHz |
| Platform | Teledyne Marine Gavia AUV |
| Period | 2010, 2015, 2017, 2018, 2021 |
| Region | Not disclosed by the authors |
| Images | 1,170 (`.jpg`, 416×416 and 1024×1024) |
| Annotations | 668 objects, YOLO `.txt`, one file per image (empty file = no target) |
| Classes | `0 = MILCO` (mine-like contact), `1 = NOMBO` (non-mine-like bottom object) |
| Metadata | **None.** No navigation, no timestamps, no geographic coordinates. |
| Download | `python scripts/download_datasets.py` (~218 MB for the year archives) |
| Storage | ~218 MB compressed, ~230 MB extracted |

### Composition, measured

| Survey year | Frames | Empty frames | MILCO | NOMBO |
|---|---|---|---|---|
| 2010 | 345 | 317 | 22 | 12 |
| 2015 | 120 | 2 | 242 | 171 |
| 2017 | 93 | 74 | 28 | 2 |
| 2018 | 564 | 452 | 96 | 46 |
| 2021 | 48 | 21 | 49 | 0 |
| **Total** | **1,170** | **866 (74%)** | **437** | **231** |

Counted directly from the label files (668 objects in total); reproduce with
`python scripts/prepare_milco_nombo.py`, which writes
`data/splits/milco_nombo_survey_split.json` as the authoritative record.

**The single most important property:** 74% of frames contain **no target at
all**. That is what makes a frame-level false-alarm rate measurable, and it
mirrors real survey conditions. It is also why precision, not recall, is the
hard problem here.

### Splits — survey-year level, no random shuffling

| Split | Years | Frames | Empty | Objects |
|---|---|---|---|---|
| train | 2015, 2010 | 465 | 319 | 447 |
| val | 2017 | 93 | 74 | 30 |
| test | 2018, 2021 | 612 | 473 | 191 |

Rationale: consecutive side-scan frames from one survey share seabed, gain
settings and often the *same physical object*. A random image split leaks test
information into training and inflates every metric. Splitting by acquisition
year — distinct campaigns, years apart, with different hardware settings —
measures what actually matters: generalisation to an **unseen survey**. The
numbers are lower this way. They are also real.

`tests/test_integration.py::test_splits_are_survey_disjoint_no_leakage` enforces
this automatically.

### Suitability for PS 26057 — and where it falls short

**Good fit.** MILCO vs NOMBO *is* the man-made-vs-ambiguous decision the problem
statement asks for. NOMBO means "not mine-like" — it does **not** assert
"natural" — so we map it to `AMBIGUOUS`, never to a man-made subclass
(`data/class_mapping.yaml`).

**Falls short.** No ghost fishing gear. No navigation data. Not Indian waters.
Only 668 annotated objects. Consequences are stated in `docs/LIMITATIONS.md`.

### Attribution (CC BY 4.0 requirement)

> Pessanha Santos, N. & Moura, R. (2024). *Side-scan sonar imaging data of
> underwater vehicles for mine detection.* Data in Brief 53:110132.
> figshare DOI 10.6084/m9.figshare.24574879. Licensed CC BY 4.0.

**Changes made:** re-split by acquisition year; reorganised into a YOLO directory
layout. Pixel data and annotations are unmodified.

---

## Evaluated, not used

### sss-crab-pot-detection-ds — **GATED**

| Field | Value |
|---|---|
| Host | HuggingFace `PINGEcosystem/sss-crab-pot-detection-ds` |
| DOI | `10.57967/hf/8397` |
| Licence | `cc-by-sa-4.0` in repo metadata; README text says "GPL" — **contradictory** |
| Size | ~559 MB, 6,674 images |
| Classes | `Crab-Pot`, `Maybe-Crab-Pot` |
| Format | JSONL metadata, absolute `[x,y,w,h]` boxes, train/valid/test |
| Status | **HTTP 403 without maintainer approval** |

This is the closest public match to the ghost-net theme. Filenames
(`Rec09_..._ss_port_00001`) expose recording id, channel and ping index — usable
for leakage-free splits and for temporal deduplication. An adapter is implemented
(`src/aquashield/ingestion/jsonl_bbox.py`) and a survey-key function is unit
tested, but **no model here has been trained on it**.

### AI4Shipwrecks
286 AUV side-scan images, shipwreck **segmentation**. Right dataset for adding a
`shipwreck_structure` class later; wrong shape for the small-debris-vs-clutter
problem that dominates PS 26057.

### Rejected
- **Seaclear Marine Debris** — optical ROV imagery, not sonar.
- **SeabedObjects-KLSG / KLSG-II, Marine-PULSE** — real SSS wrecks, but access by
  request rather than open download.
- **S3Simulator** — synthetic; cannot substitute for real acquisition noise, but
  useful for future robustness tests.

## Disk budget

| Item | Size |
|---|---|
| MILCO/NOMBO year archives | ~218 MB |
| Extracted | ~230 MB |
| Prepared splits (symlinks) | <1 MB |
| Demo subset (`demo_data/`) | ~5 MB |
| Model checkpoints | ~16 MB each |

`scripts/download_datasets.py` refuses to run with less than 2 GB free.
