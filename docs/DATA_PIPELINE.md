# Data pipeline

## Ingestion

| Adapter | Reads | Notes |
|---|---|---|
| `ingestion/image.py` | PNG, JPG, TIFF, BMP, PGM | Normalises to single-channel uint8. 16-bit rasters are scaled by their **observed** percentile range, not by 65535 — otherwise a frame using only the low bits collapses to black. |
| `ingestion/geotiff.py` | GeoTIFF | Returns `(frame, georeference)`; the georeference is `None` when spatial tags are absent. Never fabricated. |
| `ingestion/jsonl_bbox.py` | HuggingFace JSONL bbox datasets | Written for the gated crab-pot dataset. Implemented and unit tested; **never exercised on real data** (see `data/DATASETS.md`). |
| `geolocation/nav.py` | Navigation CSV | Flexible column aliases; skips leading comment banners; refuses a file with no lat/lon rather than guessing. |

**Deliberately not claimed:** raw vendor sonar logs (`.DAT`/`.SON`, `.JSF`,
`.XTF`, `.sl2`) are **not** supported. Universal sonar-format support is a large
piece of work that PINGMapper already does well for Humminbird/Lowrance; the
right move is to call it, not to reimplement it badly. See `docs/LIMITATIONS.md`.

## Quality control — `sonar/qc.py`

Every value is measured from pixels. Where something cannot be determined from
the image alone (true slant range, altitude), the field is `None`, not a guess.

| Measure | Method |
|---|---|
| `dynamic_range` | p99 − p1 |
| `unique_levels` | distinct 8-bit levels |
| `speckle_index` | median σ/μ over the flattest quartile of 16×16 patches |
| `dropout_ratio` | constant rows **plus** anomalously dark, low-variance rows (robust MAD test) |
| `saturated_ratio` | fraction clipped at 0 or 255 |
| `water_column_*` | see below |
| `usable_region_ratio` | 1 − (water column + dead rows) / total |
| `blur_score` | variance of the Laplacian |
| `quality_score` | weighted composite, **explicitly an engineering heuristic** |

### Water-column detection — and why the obvious algorithm fails

Directly beneath the towfish there is no seabed return until the first bottom
echo, so the swath centre is dark. But that dark band is **split by a bright
spike** — the nadir return itself. A naive "find the darkest contiguous run"
therefore finds only half the band.

Measured on a real frame (`0001_2021.jpg`), the centre profile is:

```
0.36 0.33 0.24 0.07 0.06 0.05 0.08 0.10 [0.68] 0.07 0.02 0.01 0.00 0.06 0.12 0.20
                    └── dark ──┘        ^nadir  └──── dark ────┘
```

The implemented algorithm: smooth the column-mean profile → mark columns below
`0.55 × median` → **bridge narrow bright gaps** so the nadir echo does not split
the band → keep the run containing (or nearest) the centre → sanity-check its
width (1–35% of swath) and its distance from centre.

Negative controls are unit tested: uniform images and pure noise must return
`None`. An earlier MAD-based version silently missed the band on every real
frame, which is why the controls exist.

## Preprocessing — `sonar/preprocess.py`

Every stage is individually switchable and is measured in the ablation
(`docs/BENCHMARKS.md`). Order matters and is fixed:

1. **Dropout repair** — linear interpolation across dead ping rows. A dead row is
   a horizontal stripe with two hard synthetic edges, which detectors reliably
   report as a long thin object.
2. **Water-column removal** — `inpaint` (default), `mask`, or `split`. Left in
   place, the nadir band both wastes detector capacity and generates spurious
   elongated targets along its edges.
3. **Slant-range correction** — `ground = sqrt(slant² − alt²)`. **Off by default**
   and *skipped with an explicit note* when altitude is unknown. Applying it with
   a guessed altitude would distort every downstream size and position.
4. **Lee speckle filter** — sonar speckle is *multiplicative*, so a Gaussian blur
   destroys target edges while barely improving SNR. Lee shrinks toward the local
   mean only where local variance is consistent with pure speckle. Unit tested to
   both reduce the speckle index *and* preserve the target/shadow edge.
5. **Across-track gain normalisation** — an empirical stand-in for TVG: divide
   each range bin by its own along-track mean, flattening the range response while
   preserving along-track anomalies (i.e. targets). *Stated limitation:* this is
   not a calibrated TVG inversion and does not recover absolute backscatter.
6. **Dynamic-range stretch** — percentile clip, robust to specular returns.
7. **CLAHE** — optional, off in the default profile.

Profiles: `none`, `minimal`, `standard` (default), `aggressive`.

## Tiling — `sonar/tiling.py`

Overlapping tiles at approximately native resolution. Edge tiles are **shifted
inward rather than zero-padded**, because padding creates a hard synthetic edge
that reads as a man-made linear feature.

Full-coverage is unit tested across four awkward shapes. Seam duplicates are
resolved by `merge_tiled_detections`, which merges on IoU **or**
intersection-over-smaller (IoS). IoS is what catches a seam-clipped fragment
sitting inside a neighbouring tile's full detection — measured IoU 0.50 versus
IoS 1.00 for exactly that case.

## Splits — leakage prevention

Split by **acquisition year**, never randomly. Consecutive side-scan frames from
one survey share seabed, gain settings and often the same physical object; a
random image split leaks test information into training and inflates every
metric.

| Split | Surveys | Frames | Empty | Objects |
|---|---|---|---|---|
| train | 2015, 2010 | 465 | 319 | 447 |
| val | 2017 | 93 | 74 | 30 |
| test | 2018, 2021 | 612 | 473 | 191 |

The validation survey is used **only** to fit the FP filter and the calibrator.
The test surveys are touched only by `scripts/evaluate.py`.

Reproducible via `data/splits/milco_nombo_survey_split.json` plus per-split file
lists. Disjointness *and* survey-disjointness are enforced by tests.
