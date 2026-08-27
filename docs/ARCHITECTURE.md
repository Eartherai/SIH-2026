# Architecture

## The positioning

Not "YOLO detects marine debris" — that is reproducible in an afternoon and
already exists publicly (see `research/prior_art.md`). AQUA-SHIELD is:

**Detection → Verification → Localization → Action**

The detector is one stage of nine. The stages that make it *operational* are the
ones that decide whether an analyst can trust and act on the output.

## Pipeline

```
RAW SONAR FRAME
      │
      ▼
┌─────────────────────┐
│ 1. QUALITY CONTROL  │  sonar/qc.py
└─────────────────────┘  dynamic range · speckle index · dropout rows ·
      │                  water-column detection · usable-region ratio
      │                  → quality_score feeds priority, never invents values
      ▼
┌─────────────────────┐
│ 2. PREPROCESSING    │  sonar/preprocess.py
└─────────────────────┘  dropout repair → water-column removal →
      │                  [slant-range corr.] → Lee speckle filter →
      │                  across-track gain norm → dynamic-range stretch
      │                  every stage individually switchable + ablated
      ▼
┌─────────────────────┐
│ 3. TILING           │  sonar/tiling.py
└─────────────────────┘  overlapping tiles at ~native resolution;
      │                  edge tiles shifted inward, never zero-padded
      ▼
┌─────────────────────┐
│ 4. DETECTION        │  detection/detector.py  (backend-agnostic)
└─────────────────────┘  low conf threshold — recall first, verify second
      │                  seam duplicates merged by IoU **or** IoS
      ▼
┌─────────────────────┐
│ 5. FP FILTERING     │  confidence/features.py + fp_filter.py
└─────────────────────┘  10 physical features, independent of the detector;
      │                  learned logistic model fitted on a held-out survey;
      │                  every rejection carries its reason
      ▼
┌─────────────────────┐
│ 6. CALIBRATION      │  confidence/calibration.py
└─────────────────────┘  Platt scaling fitted on the same held-out survey;
      │                  reports `calibrated: false` when not fitted
      ▼
┌─────────────────────┐
│ 7. DEDUPLICATION    │  tracking/dedup.py
└─────────────────────┘  geographic (metres) or ping-sequence (IoU);
      │                  N observations → 1 hazard, stable IDs
      ▼
┌─────────────────────┐
│ 8. GEOLOCALIZATION  │  geolocation/
└─────────────────────┘  GeoTIFF affine · per-ping nav · or REFUSE
      │                  every fix carries an uncertainty budget in metres
      ▼
┌─────────────────────┐
│ 9. PRIORITY         │  reporting/priority.py
└─────────────────────┘  separate from confidence, by design
      │
      ▼
  MAP · DASHBOARD · JSON / CSV / GeoJSON
```

## Why these stages and not others

The architecture in the original brief was challenged rather than accepted.

**Kept, and justified:**
- *Quality control* — its output is a real input to priority, and it tells the
  operator when a low detection count means "clean seabed" versus "bad data".
- *Tiling* — mandatory. Targets are ~24 px; a full waterfall at fixed network
  resolution loses them.
- *FP filtering* — the single highest-value stage. 74% of frames are empty, so
  precision is the binding constraint (`docs/BENCHMARKS.md`).
- *Deduplication* — without it, "hazard count" is meaningless; one object appears
  in many pings.
- *Geolocation with uncertainty* — a coordinate without an error bar is not a
  measurement.

**Deliberately not built:**
- *Evidence fusion as a probabilistic product.* The brief suggested combining
  model, shadow, temporal and quality evidence into one probability. These
  quantities are **not independent** (the detector already looks at shadow and
  contrast), so multiplying them would be statistically wrong. We surface them as
  separate labelled indicators and let one *learned* model weigh the features.
- *A separate temporal-consistency model.* Persistence is already captured by
  `observation_count` from deduplication. A second mechanism would double-count.
- *PostgreSQL/PostGIS.* Unjustified at prototype scale; SQLite/JSON is enough.

## Module map

```
src/aquashield/
├── device.py              MPS → CPU selection, never assumes CUDA
├── pipeline.py            orchestration; every stage switchable
├── ingestion/             image · geotiff · jsonl_bbox · nav CSV
├── sonar/                 qc · preprocess · tiling
├── detection/             detector (ultralytics | torchvision) · boxes · taxonomy
├── confidence/            features · fp_filter · calibration
├── tracking/              dedup → unique hazards
├── geolocation/           reference (GeoTIFF / nav / none) · nav table
├── evaluation/            IoU matching, object- and frame-level metrics
└── reporting/             schema · priority · writers
```

## Design rules enforced throughout

1. **Never fabricate a value.** No coordinate without metadata; no calibrated
   probability without a fitted calibrator; no physical size without a ground
   sample distance. Each is `None` plus a note explaining why.
2. **Backend independence.** Nothing outside `detection/detector.py` imports
   Ultralytics, so the AGPL dependency can be swapped out.
3. **Taxonomy is data, not code.** `data/class_mapping.yaml` decides what a class
   *means*, so retraining cannot silently change report semantics.
4. **Every stage is switchable**, which is what makes the ablation possible.
5. **Provenance travels with the result.** Every report embeds the model, device,
   preprocessing profile, filter and calibration that produced it.
