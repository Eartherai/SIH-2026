# Indian sonar data — research and status

Rule applied strictly: a dataset counts as **Indian only if the DATA was
collected in India**. Indian authorship, an Indian institution, or a thesis
written in India does **not** make the data Indian.

Researched 2026-08-28.

## Candidate 1 — TiHAN / IIT Hyderabad Side-Scan Sonar dataset  ⟵ best lead

| Field | Value (verified from tihan.iith.ac.in/TiAND.html) |
|---|---|
| Collection location | **Hyderabad lakes, India** ✅ genuinely Indian |
| Sonar type | **Side-Scan Sonar** ✅ (sensor "SSS-600K", 10–75 m range) |
| Format | `.xtf` (JW Fisher / SonarView compatible) — raw SSS, **not pre-cut images** |
| Annotations | **None mentioned** — appears unlabelled |
| Environment | **Freshwater lakes**, not marine |
| Access | **Gated:** Google-Form request → approval → emailed download link |
| License | "Subject to data usage agreement" (unspecified) |

**Decision:** this is the **only genuinely-Indian SSS source found**, and it is
valuable — but for a **specific, limited role**:

- **NO-GO for supervised training** — no annotations, and freshwater-lake seabed
  is a different domain from marine debris surveys.
- **GO (pathway) for Indian-domain *validation* and *hard-negative* mining** —
  once access is granted, its unlabelled lakebed imagery is exactly the kind of
  natural-clutter data our false-positive engine and (future) anomaly branch
  should be stress-tested against. It would let us make the SIH argument
  *"international training → Indian-domain evaluation"*.
- **Access is a human step.** It requires submitting a form and agreeing to a data
  agreement — I cannot and should not complete that autonomously. Flagged for the
  team.

**Honest framing for SIH:** we may claim an *"Indian-domain validation/adaptation
pathway"* built around this dataset. We may **not** claim *"validated for Indian
waters"* — we have no Indian data in hand, and it is lake, not sea.

## Candidate 2 — TiAND (the terrestrial one)  ✗

`Nitishkr22/TIAND` and the main TiHAN "TiAND" project are a **terrestrial
autonomous-navigation** multimodal dataset (camera + radar, Indian roads). **Not
sonar. Rejected** — a name collision with Candidate 1's sonar sub-dataset.

## Candidate 3 — S3Simulator (SSS simulator)  ~

Indian-authored (arXiv 2408.12833). It is a **synthetic SSS simulator**, not
field data, so it is not "Indian data" by the collection-location rule. Possible
future use for augmentation/robustness only; not pursued this phase.

## Others checked, rejected as Indian

- **UATD** — Chinese (Dalian/Maoming). FLS. Not Indian.
- **MILCO/NOMBO** — region undisclosed by authors; **not** established as Indian.
- **AI4Shipwrecks** — US (Michigan). Not Indian.
- **Marine Debris (Valdenegro)** — Scotland (Heriot-Watt watertank). Not Indian.
- **KLSG** — Chinese SSS. Not Indian.
- **SonarT165** (arXiv 2504.15609, acoustic tracking) — provenance not Indian.

## Summary

There is exactly one credible Indian **field** SSS source (TiHAN/IITH, Hyderabad
lakes). It is access-gated and unlabelled, so its role is Indian-domain
**validation / hard negatives / anomaly stress-testing**, contingent on a manual
access request. No Indian marine SSS with debris/ghost-gear annotations was found
in the public domain.
