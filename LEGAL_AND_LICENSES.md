# Legal & Licensing

Everything AQUA-SHIELD depends on, with its licence verified from the package or
repository metadata rather than from memory. Where a licence is unclear we say so
instead of assuming it is permissive.

Verified on 2026-08-27 from installed package metadata and the GitHub API.

---

## 1. The one that actually constrains you: Ultralytics (AGPL-3.0)

| | |
|---|---|
| Package | `ultralytics` 8.3.253 |
| Licence | **AGPL-3.0-or-later** |
| Where used | `src/aquashield/detection/detector.py`, backend `"ultralytics"` |

**What this means in practice.** AGPL-3.0 is copyleft *and* it triggers on network
use. If you deploy AQUA-SHIELD with the Ultralytics backend as a hosted service,
you must offer the complete corresponding source of the whole combined work to
its users. For a hackathon prototype, an internal NIOT tool, or research, this is
fine. For a closed-source commercial product it is not.

**Mitigation already built in.** The detector layer is deliberately
backend-agnostic. `Detector(backend="torchvision")` runs a BSD-3 model through the
same interface, and nothing downstream — pipeline, verification, geolocation,
reports, dashboard — imports Ultralytics. Swapping the backend does not require
touching any other module.

**Status, stated plainly:** the trained checkpoints shipped here were produced with
Ultralytics, so they inherit AGPL obligations. A licence-clean production path
requires retraining under the torchvision backend. That work is **not done**; see
`docs/LIMITATIONS.md`.

---

## 2. Datasets

### MILCO / NOMBO — **used for training and all reported metrics**

| | |
|---|---|
| Title | Side-scan sonar imaging for mine detection |
| Authors | Nuno Pessanha Santos, Ricardo Moura |
| Paper | *Data in Brief* 53:110132 (2024), DOI `10.1016/j.dib.2024.110132` |
| Data | figshare, DOI `10.6084/m9.figshare.24574879` |
| Licence | **CC BY 4.0** |
| Access | Public, no registration |
| Size | 1,170 images, 512–1024 px, ~218 MB (year archives) |
| Sensor | Marine Sonic dual-frequency SSS, 900–1800 kHz, Teledyne Gavia AUV, 2010–2021 |

**Obligations:** attribution and a statement of changes. Both are discharged in
`README.md`, `data/DATASETS.md`, every generated report's `provenance` block, and
`demo_data/*/scenario.json`. Changes made: re-split by acquisition year,
reorganised into a YOLO directory layout. **Pixel data and annotations are
unmodified.**

CC BY 4.0 permits commercial use and redistribution with attribution, so the
demo subset in `demo_data/` may be redistributed with this repository.

### sss-crab-pot-detection-ds — **NOT used; access could not be obtained**

| | |
|---|---|
| Host | HuggingFace `PINGEcosystem/sss-crab-pot-detection-ds` |
| Licence | `cc-by-sa-4.0` (repo card metadata); the README text says "GPL" — **the two disagree** |
| DOI | `10.57967/hf/8397` |
| Status | **ACCESS-GATED.** Returns HTTP 403 without maintainer approval. |

This is the closest public match to the "ghost nets" theme of PS 26057. We wrote
an adapter (`src/aquashield/ingestion/jsonl_bbox.py`) so it can be used once
access is granted, but **AQUA-SHIELD has not been trained on it and makes no
claims about ghost-gear detection performance.**

Note the licence ambiguity: CC BY-SA 4.0 and GPL are both copyleft but are not
interchangeable, and CC BY-SA would require share-alike on derived datasets.
Resolve this with the maintainers before any redistribution.

---

## 3. Python dependencies

| Package | Version | Licence |
|---|---|---|
| ultralytics | 8.3.253 | **AGPL-3.0-or-later** ⚠ |
| torch | 2.13.0 | BSD-3-Clause |
| torchvision | 0.28.0 | BSD-3-Clause |
| numpy | 1.26.4 | BSD-3-Clause |
| scipy | 1.17.1 | BSD-3-Clause |
| pandas | 3.0.5 | BSD-3-Clause |
| scikit-learn | 1.9.0 | BSD-3-Clause |
| shapely | 2.1.2 | BSD-3-Clause |
| tifffile | 2026.3.3 | BSD-3-Clause |
| opencv-python-headless | 4.11.0.86 | Apache-2.0 |
| streamlit | 1.62.0 | Apache-2.0 |
| folium | 0.20.0 | MIT |
| pyproj | 3.7.2 | MIT |
| plotly | 7.0.0 | MIT |
| fastapi | 0.141.1 | MIT |

Apart from Ultralytics, every dependency is permissively licensed.

---

## 4. Pretrained weights

`yolo11n.pt` (COCO-pretrained) is distributed by Ultralytics under **AGPL-3.0**.
Our checkpoints are fine-tuned from it and therefore inherit that licence.

---

## 5. Prior-art projects we studied but did **not** vendor

No code from any of these is copied into this repository. They informed the
design and are cited in `research/prior_art.md`.

| Project | Licence (verified via GitHub API) | Relationship |
|---|---|---|
| PINGMapper | MIT | Studied. Compatible if vendored later. |
| GhostVision | **NOASSERTION** — no recognised licence file | Studied only. **Do not vendor** until the maintainers clarify terms. |
| sidescantools (sonoware) | GPL-3.0 | Studied only. Vendoring would impose GPL. |
| AI4Shipwrecks | MIT (site repo) | Studied. Dataset terms must be checked separately. |

---

## 6. Map tiles

The dashboard's optional Folium map requests OpenStreetMap tiles (ODbL,
attribution required — Folium renders it automatically). Set
`AQS_OFFLINE_MAP=1` to disable all tile requests; the map then falls back to a
tile-free coordinate plot and the application makes **no network requests at all**.

---

## 7. This project

Released under the MIT Licence (see `LICENSE`) **for the code we wrote**. Note
that combining it with the Ultralytics backend produces a combined work governed
by AGPL-3.0. The MIT grant covers our source; it cannot and does not relicense
Ultralytics.
