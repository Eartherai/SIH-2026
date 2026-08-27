# Demo guide

## Fastest path

```bash
./run_demo.sh
```

That validates the environment, checks a real model and demo data exist, and
opens the dashboard at http://localhost:8501.

From a clean clone:

```bash
./setup.sh                                  # venv + dependencies
python scripts/download_datasets.py         # MILCO/NOMBO, CC BY 4.0, ~218 MB
python scripts/prepare_milco_nombo.py       # survey-level splits
python scripts/train.py --exp-id E01 --epochs 150
python scripts/fit_verification.py --weights runs/**/weights/best.pt
./run_demo.sh
```

## Offline

```bash
export AQS_OFFLINE_MAP=1
./run_demo.sh
```

With that set the application makes **no network requests at all** — the map
falls back to a tile-free coordinate plot. Inference is local in every mode:
there are no calls to OpenAI, Anthropic, Google, or any cloud inference service,
in any code path.

## The four scenarios, and what each is for

All imagery comes from the **held-out test surveys** (2018, 2021). Nothing in the
demo was seen during training.

### 1 · `01_clear_targets` — the baseline case
The frames carrying the **largest annotated targets** in the test surveys
(median ~3,450 px²). Shows the pipeline working.

### 2 · `02_hard_targets` — the failure mode, on purpose
The frames carrying the **smallest annotated targets** (median ~280 px²). Expect
lower confidence and misses. This scenario exists so the demo cannot be accused
of showing only easy frames. When a judge asks "what happens when it fails?",
open this one.

> **Both sets are selected purely by annotated target area** — a property of the
> data, never of our model's output. Ranking demo frames by anything the detector
> produces would be cherry-picking. (Our first attempt ranked by a contrast
> heuristic; it was a poor proxy, so we replaced it with target size.)

### 3 · `03_natural_seabed` — **the scenario that matters**
Frames with **no annotated target at all**: ripples, rock texture, nadir band,
shadow. *Every hazard reported here is a false positive.*

This is the PS 26057 scenario. Toggle **False-positive filter** off and on in the
sidebar and watch the count change. Rejected candidates stay visible as thin red
boxes with their reason, so nothing is hidden.

### 4 · `04_georeferenced` — geolocation, dedup and the map
A contiguous sequence plus a navigation track. Demonstrates pixel → lat/lon,
positional uncertainty drawn to scale on the map, spatial deduplication
(N observations → 1 hazard), UTM, and GeoJSON export.

> **Say this out loud during the demo:** the navigation track in this scenario is
> **synthetic**. The source dataset ships no positions. It exercises the
> geolocation maths on a known geometry; the coordinates are not real object
> positions. It is labelled as synthetic in the CSV header, the scenario metadata
> and the UI. The other three scenarios have no navigation and correctly report
> *Geolocation unavailable*.

## Suggested 5-minute run

| Time | Do | Say |
|---|---|---|
| 0:00 | Scenario 1 → **Process** | "Thousands of frames per survey are reviewed by hand today. Here are held-out frames the model has never seen." |
| 0:45 | Point at *Raw candidates* vs *Filtered out* | "The detector runs at a deliberately low threshold — recall first. Verification removes the clutter." |
| 1:15 | Switch to **Scenario 3** → Process | "These frames contain nothing. Every box here would be a false alarm." |
| 2:00 | Toggle the FP filter off, reprocess | "This is what a detector-only prototype gives you. This is why precision, not recall, is the real problem." |
| 2:45 | Open a rejected detection's reason | "Every rejection is explainable — these are the features that drove it." |
| 3:15 | **Scenario 4** → Process → **Map** tab | "With navigation, pixels become coordinates. The shaded circles are the actual positional uncertainty, at scale — not decoration." |
| 4:00 | **Hazard register** → open one hazard | "Confidence and priority are different questions. This one is only 55% likely, but it is large, seen nine times, and well located — so it outranks a 95% blob we can't navigate to." |
| 4:30 | **Export** → download CSV/JSON | "This is what the survey team actually receives." |
| 5:00 | **Provenance** tab | "Every number carries the model, device, preprocessing and calibration that produced it." |

## Things that will happen, and are correct

- **"Geolocation unavailable"** in scenarios 1–3. Correct: there is no navigation
  data, and the system refuses to invent a position.
- **"Confidence is a RAW detector score"** if the verification models have not
  been fitted. Correct and honest: run `scripts/fit_verification.py`.
- **"No confident detections found"** on some frames. Correct: an empty frame
  should produce nothing.
- **Physical dimensions missing.** Correct: without a ground sample distance,
  pixels cannot be converted to metres, so the box is reported in pixels.

## If something breaks live

| Symptom | Fix |
|---|---|
| Sidebar warns "profile does not match this checkpoint" | Set the preprocessing profile back to the one the checkpoint was trained on. A mismatch cost a 12× F1 drop when we measured it (`docs/BENCHMARKS.md` §3). |
| "No model checkpoint found" | `python scripts/train.py --exp-id E01 --epochs 150`, or drop a `.pt` into `models/` |
| Map blank / slow | `export AQS_OFFLINE_MAP=1` |
| Dashboard slow on first frame | First MPS call compiles kernels (~3 s). Process once before the demo starts. |
| Port 8501 busy | `streamlit run dashboard/app.py --server.port 8502` |

Pre-warm the model before presenting: process one scenario once, then reload.
