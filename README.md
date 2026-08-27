# 🛰️ AQUA-SHIELD

**Acoustic Intelligence for Underwater Anomaly, Debris & Marine-Hazard Localization**

**Detection → Verification → Localization → Action**

Smart India Hackathon 2026 · Problem Statement **26057** · Ministry of Earth
Sciences (MoES) / National Institute of Ocean Technology (NIOT)

---

## The problem

Ghost nets and other man-made debris kill marine life continuously, wreck coral,
and foul vessel propellers. Finding them means towing a side-scan sonar and then
having a human read thousands of kilometres of acoustic imagery — slow, fatiguing,
and inconsistent, because debris hides among rock clusters, sand ripples and
acoustic shadows.

**The measurable difficulty is not "can a model find objects".** In our data,
**74% of sonar frames contain no target at all.** An automated system that fires
on a fraction of those is worse than useless — the analyst stops trusting it and
goes back to reading raw imagery. Precision, not recall, is the binding
constraint.

## What this is

A working, local-first prototype that ingests side-scan sonar, detects candidate
man-made anomalies, **verifies them against physical evidence**, calibrates
confidence, deduplicates repeated sightings into unique hazards, geolocates them
*when the metadata supports it*, prioritises them for an operator, and exports an
actionable report.

It runs offline on a laptop. There are no cloud API calls in any code path.

### The rule that shapes everything

> **Never fabricate a value.**
> No coordinate without navigation metadata. No calibrated probability without a
> fitted calibrator. No physical dimensions without a ground sample distance.
> Each is returned as `null` with a note explaining why.

A fabricated latitude looks exactly like data, exports cleanly to CSV, and sends a
cleanup vessel to open water.

---

## Quick start

```bash
git clone <this-repo> && cd aqua-shield
./setup.sh                                # venv + deps (uses uv if available)
python scripts/download_datasets.py       # MILCO/NOMBO, CC BY 4.0, ~218 MB
python scripts/prepare_milco_nombo.py     # leakage-free survey-level splits
python scripts/train.py --exp-id E01 --epochs 150
python scripts/fit_verification.py --weights runs/detect/**/weights/best.pt
./run_demo.sh                             # dashboard at localhost:8501
```

Fully offline:

```bash
export AQS_OFFLINE_MAP=1 && ./run_demo.sh
```

---

## Architecture

```
RAW SONAR ─▶ QUALITY CONTROL ─▶ [PREPROCESSING] ─▶ TILING ─▶ DETECTION
                                 off by default: measured, didn't help
                                                              │
   REPORT ◀─ PRIORITY ◀─ GEOLOCALIZATION ◀─ DEDUPLICATION ◀───┤
                                                              │
                          CALIBRATION ◀─ FALSE-POSITIVE FILTER ┘
```

Nine stages, each individually switchable — which is what makes the ablation in
`docs/BENCHMARKS.md` possible. Full detail in `docs/ARCHITECTURE.md`.

### The stage that matters most

The false-positive engine (`src/aquashield/confidence/`) computes **ten
physically-motivated features** around each candidate — shadow coherence,
target/background contrast, highlight compactness, texture roughness relative to
the surrounding seabed — that are *independent of the detector's own opinion*, then
fits a logistic model on a **held-out survey**.

We did not hand-write threshold rules, and that turned out to matter — though not
in the way we first thought:

> **The filter's weights exposed a bug in our own pipeline.** In an early fit,
> `shadow_ratio` received a large *negative* weight — the opposite of what sonar
> physics predicts, since a proud object casts a shadow. That looked like a
> surprising scientific result. It was not. It was a symptom: we were applying a
> preprocessing chain at inference that the detector had never been trained on,
> and the mismatch was corrupting exactly that feature. After fixing the
> mismatch, `shadow_ratio` came out **+0.15** and `shadow_side_consistent`
> **+0.32** — both positive, and consistent with the physics.

The lesson we actually take from this is that **a learned, inspectable
verification stage is a diagnostic instrument**, not just a classifier. Its
weights flagged a data-handling defect that our metrics alone had not localised.
A hand-tuned threshold would have silently absorbed the same defect.

Every verdict is explainable: each detection reports the top feature
contributions that drove its accept/reject decision.

---

## Data, and its honest limits

Trained on **MILCO/NOMBO** — 1,170 real side-scan frames from a Teledyne Gavia
AUV, 2010–2021, **CC BY 4.0**, DOI `10.6084/m9.figshare.24574879`.

**Splits are by acquisition year, never random.** Consecutive frames from one
survey share seabed, gain settings, and often the *same physical object*; a random
split leaks and inflates every metric. A test enforces survey-disjointness.

| Split | Surveys | Frames | Empty | Objects |
|---|---|---|---|---|
| train | 2015, 2010 | 465 | 319 | 447 |
| val (fits FP filter + calibration) | 2017 | 93 | 74 | 30 |
| **test (all reported metrics)** | 2018, 2021 | 612 | 473 | 191 |

### What we have **not** done

**AQUA-SHIELD has never detected a ghost net.** The closest public ghost-gear
dataset is access-gated (HTTP 403); we wrote the adapter and did not get the data.
No ghost-gear accuracy claim appears anywhere in this repository. What is
demonstrated is the *discrimination task* — man-made target vs ambiguous seabed
clutter — plus the full operational pipeline around it.

Geolocation accuracy has never been validated against ground truth, because this
dataset ships no navigation data. The demo's georeferenced scenario uses an
explicitly **synthetic** track, labelled as such in the file, the metadata and the
UI.

Full list: **`docs/LIMITATIONS.md`** — read it before believing anything else here.

---

## Measured performance

All figures measured on the development machine (Apple M5, 24 GB, macOS 26.5).
Nothing is copied from a datasheet. See `docs/BENCHMARKS.md` for the full tables
and `experiments/` for the raw records.

| | Measured |
|---|---|
| MPS inference (tiled frame) | **21.4 ms** |
| CPU inference (same) | 82.2 ms |
| **MPS speedup** | **3.8×** |
| End-to-end throughput | 37.4 frames/s (27 ms/frame) |
| Peak RSS (full pipeline) | 640 MB |
| Model | 2.58 M params, 6.3 GFLOPs, 16 MB |
| ONNX export | 10.61 MB, 8.5 ms (ONNX Runtime + CoreML EP) |

### What the verification stage is actually worth

Measured on the 612 held-out test frames (473 of them target-free):

| | Precision | Recall | True positives | False positives | Frames falsely alarmed |
|---|---|---|---|---|---|
| Detector only | 0.247 | 0.110 | 21 | 64 | 37 / 473 |
| + hand-written rules | 0.300 | 0.063 | 12 | 28 | 18 / 473 |
| **+ learned FP filter** | **0.322** | 0.100 | 19 | 40 | **25 / 473** |

The learned filter raises precision by **30%** and removes
**32%** of falsely-alarmed frames while keeping
**19 of the 21** true positives the detector found.

The hand-written rule set reaches similar precision but keeps only 12 true
positives — it buys quiet by throwing away targets. That difference is the
argument for fitting the filter rather than tuning thresholds by hand.

Detection accuracy is reported on **held-out surveys** in `docs/BENCHMARKS.md`.
It is modest, and the reasons are stated. It is not inflated by a random split.

### A defect we found by measuring, and fixed

Preprocessing must match what the detector was **trained** on. Applying our
`standard` profile at inference to a raw-trained model, measured on the 612
held-out test frames:

| Inference input | F1 | True positives | False-alarm frames (of 473) |
|---|---|---|---|
| raw (matched) | **0.1440** | 28 | 80 |
| `standard` profile (mismatched) | 0.0117 | 4 | 186 |

A **12× F1 degradation**. The preprocessing profile is therefore now a property
of the *checkpoint*, recorded in a `.meta.json` sidecar and selected
automatically; the dashboard warns if you override it. A test asserts the
pipeline default stays `none`.

We then asked the harder question properly — *does preprocessing help at all?* —
by retraining on a preprocessed copy of the dataset with an identical recipe, so
train and inference matched:

| Detector | Trained + evaluated on | mAP50 | Precision |
|---|---|---|---|
| E04 | **raw** | **0.1163** | 0.3444 |
| E06 | preprocessed | 0.0318 | 0.0769 |

**It does not help.** A genuine negative result: a good deal of the sonar-specific
preprocessing in this repository does not earn its place in the inference path on
this dataset, and we ship it disabled rather than applying it because it sounds
appropriate. It still feeds quality control and the verification features, and it
remains a switchable, inspectable option. See `docs/LIMITATIONS.md` §9.

---

## Running on Apple Silicon

Developed on an Apple **M5**, 24 GB unified memory, macOS 26.5, arm64. No CUDA
anywhere; MPS with automatic CPU fallback.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
PYTHONPATH=src python -m aquashield.device     # prints the environment report
```

```
OS:            macOS-26.5.2-arm64-arm-64bit
Architecture:  arm64
Python:        3.12.13
PyTorch:       2.13.0
MPS built:     True
MPS available: True
CUDA:          False
Selected:      mps  (apple silicon mps detected)
```

**Two Apple-Silicon-specific findings, recorded because they cost real time:**

1. **`ultralytics` 8.4.130 diverged** on this dataset — `val/cls_loss` climbed to
   ~1.1 × 10⁶ while train loss stayed flat. Disabling AMP did not help. Pinning to
   the stable 8.3 line fixed it. `requirements.txt` pins `ultralytics>=8.3,<8.4`.
2. **AMP is off by default on MPS** (`--amp` to enable). Given finding 1 we cannot
   claim AMP *caused* that divergence, only that we don't enable it here.

Memory discipline: frames stream rather than loading a survey into RAM; batch 16
at 640 px trains comfortably; peak RSS stayed under 700 MB during inference.

---

## Repository

```
aqua-shield/
├── run_demo.sh · setup.sh          one-command demo / setup
├── src/aquashield/
│   ├── device.py                   MPS → CPU, never assumes CUDA
│   ├── pipeline.py                 the nine-stage orchestrator
│   ├── ingestion/                  image · geotiff · jsonl_bbox · nav CSV
│   ├── sonar/                      qc · preprocess · tiling
│   ├── detection/                  detector (swappable backend) · boxes · taxonomy
│   ├── confidence/                 features · fp_filter · calibration
│   ├── tracking/                   dedup → unique hazards
│   ├── geolocation/                GeoTIFF · per-ping nav · or refuse
│   ├── evaluation/                 IoU matching, object- + frame-level metrics
│   ├── reporting/                  schema · priority · JSON/CSV/GeoJSON
│   ├── storage/                    SQLite
│   └── api/                        FastAPI + OpenAPI
├── dashboard/app.py                Streamlit operator UI
├── scripts/                        download · prepare · train · fit · evaluate ·
│                                   benchmark · robustness · export_edge · demo data
├── tests/                          108 tests incl. end-to-end + headless dashboard
├── data/ · demo_data/ · models/ · experiments/ · outputs/
├── research/                       sources · datasets · prior art · model selection
└── docs/                           architecture · pipelines · geolocation ·
                                    benchmarks · demo · limitations · judge Q&A
```

---

## Tests

```bash
source .venv/bin/activate && python -m pytest tests/ -q
```

Covers preprocessing against a synthetic frame with a *known* water column,
target, shadow and dropout row; tile coverage on awkward shapes; IoU/IoS merging;
class-mapping semantics; filter and calibrator refusal on degenerate data;
geolocation sign conventions and circular heading interpolation; CSV/JSON/GeoJSON
schemas; survey-disjointness of the splits; a full pipeline run; a determinism
check; and a headless run of the actual dashboard.

---

## Documentation

| | |
|---|---|
| `docs/ARCHITECTURE.md` | The nine stages, and what we deliberately did *not* build |
| `docs/DATA_PIPELINE.md` | Ingestion, QC, preprocessing, tiling, splits |
| `docs/ML_PIPELINE.md` | Detector, verification, calibration, evaluation protocol |
| `docs/GEOLOCATION.md` | Three cases, the uncertainty budget, the nadir singularity |
| `docs/BENCHMARKS.md` | Every measured number, and the ablation |
| `docs/DEMO.md` | Demo script and what "correct failure" looks like |
| `docs/LIMITATIONS.md` | **Read this one** |
| `docs/JUDGE_QUESTIONS.md` | 42 hard questions with short and deep answers |
| `research/prior_art_matrix.md` | Claim-by-claim novelty audit against existing work |
| `LEGAL_AND_LICENSES.md` | Verified licences, including the AGPL constraint |

---

## Attribution

Imagery: Pessanha Santos, N. & Moura, R. (2024). *Side-scan sonar imaging data of
underwater vehicles for mine detection.* Data in Brief 53:110132.
figshare DOI `10.6084/m9.figshare.24574879`. Licensed **CC BY 4.0**.
*Changes: re-split by acquisition year; reorganised into a YOLO layout. Pixel data
and annotations unmodified.*

Code released under MIT (`LICENSE`). **Note:** the default detector backend uses
Ultralytics (**AGPL-3.0**), so the combined work is AGPL. See
`LEGAL_AND_LICENSES.md`.
