# Dataset-role matrix

Datasets are **not** blindly combined. Each has one primary role, justified by its
sonar type, domain, and label semantics. Sonar type is the first filter: FLS
labels never enter the SSS taxonomy.

| Dataset | Sonar type | Domain / origin | Labels | Access | Primary role in AQUA-SHIELD |
|---|---|---|---|---|---|
| **MILCO/NOMBO** | **SSS** | AUV survey, region undisclosed | box (MILCO/NOMBO) | open, CC BY 4.0 | **Supervised training + all reported metrics** |
| **AI4Shipwrecks** | SSS | US (Michigan), AUV | segmentation masks (wreck) | open (MIT site) | Future `shipwreck_structure` **segmentation** class; large-target verifier data |
| **KLSG** | SSS | China | image-level (wreck/seabed) | by request | Natural-seabed **hard negatives** (rocks, ripples, wreck-like clutter) — pending access |
| **sss-crab-pot** | SSS | US (Delaware bays) | box (crab-pot) | **gated (HTTP 403)** | **Ghost-gear supervised training** — the closest to PS 26057; adapter ready, access pending |
| **UATD** | **FLS** | China (lakes) | box (10 obj) | open | **Evidence only** (modality contrast); NOT training/test. See `UATD_USAGE_DECISION.md` |
| **Marine Debris (Valdenegro)** | **FLS** | Scotland (watertank/turntable) | seg + box (11 debris) | open (GitHub) | Debris **morphology** reference / segmentation pretraining candidate; **FLS domain-shift caveat**; not field data |
| **TiHAN/IITH SSS** | **SSS** | **India (Hyderabad lakes)** | none | gated (form) | **Indian-domain validation / hard negatives** — pending manual access; unlabelled |
| **S3Simulator** | SSS (synthetic) | India-authored, synthetic | synthetic | open | Optional augmentation/robustness; not field data |

## Roles legend
`supervised training · pretraining · validation · hard negatives · domain
adaptation · external testing · anomaly learning · evidence-only`

## Combination policy
- **Same-modality only** may be pooled for training. FLS and SSS are never mixed
  in a supervised head.
- **Cross-domain evaluation** (train on one SSS source, test on a held-out SSS
  source/site) is preferred over random frame splitting for any generalisation
  claim (see `docs/BENCHMARKS.md`, cross-domain section — pending a second labelled
  SSS source).
- Every dataset actually used records its provenance in every generated report's
  `provenance` block.
