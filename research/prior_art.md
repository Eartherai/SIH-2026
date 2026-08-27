# Prior art

What already exists, what it actually does, and what is genuinely left for
AQUA-SHIELD to contribute. Licences were verified through the GitHub API and
PyPI metadata on 2026-08-27, not recalled.

The rule applied throughout: **never claim novelty that is already present in
public software.**

---

## GhostVision — the closest existing system

- Repository: `PINGEcosystem/GhostVision`; PyPI `ghostvision` 1.0.0
- Paper: *GhostVision: Democratizing Derelict Gear Detection Using Low-Cost Sonar
  and Artificial Intelligence*, **Journal of Marine Science and Engineering**
  14(10):951, 2025 (MDPI, open access)
- Licence: **NOASSERTION** — GitHub finds no recognised licence file

**What it does.** A Python interface that detects derelict crab pots ("ghost
pots") in side-scan sonar from low-cost recreation-grade sonar, then georeferences
the detections through PINGMapper. It ships packaged YOLO- and RF-DETR-based
detectors trained via Roboflow, and depends on the wider PING ecosystem
(`pingmapper`, `pingverter`, `pingwizard`, `pingdetect`).

**Reported performance.** Three architectures (YOLOv12, YOLOv26, RF-DETR) trained
on 3,110 manually annotated sonar images. YOLOv12 gave the strongest untuned
operational result (F1 = 0.512, recall = 0.922); after post-processing
optimisation all three converged to F1 ≈ 0.71–0.73.

**Why this matters to us — read this before claiming novelty.** GhostVision
already delivers *detection + georeferencing + mapping for derelict gear*. That
combination is **not** novel and we do not claim it. Two observations remain
genuinely open:

1. Its headline untuned figures (F1 0.512 at recall 0.922) show precision is the
   binding constraint — the same false-positive problem PS 26057 names. Their fix
   is post-processing tuning; ours is an explicitly *learned*, feature-based
   verification stage whose weights are inspectable.
2. It is single-domain (crab pots, recreation-grade sonar) and its licence is
   unresolved, so it cannot be vendored into a government deliverable as-is.

---

## PINGMapper — the sonar-processing reference

- Repository: `CameronBodine/PINGMapper` — **MIT**, actively maintained (91★, last push 2026-08-26)
- Paper: Bodine, Buscombe, Best, Redner & Kaeser, *PING-Mapper: Open-Source
  Software for Automated Benthic Imaging and Mapping Using Recreation-Grade
  Sonar*, Earth and Space Science, 2022, DOI `10.1029/2022EA002469`

**What it does.** Decodes Humminbird® and Lowrance® sonar recordings, exports
per-ping attributes from every channel, removes the water column using the
sonar's own depth sounding, and exports sonogram tiles and georectified mosaics.

**What we take from it.** The *conceptual* pipeline — decode → per-ping
attributes → water-column removal → tiles → georectified mosaic — is the
established, correct shape for SSS processing, and AQUA-SHIELD follows it.
Its `wcp`/`nwcp` and `ss_port`/`ss_star` output naming also told us how real
survey products are organised, which shaped our ingestion adapters.

**What it does not do.** No detection, no classification, no confidence, no
hazard reporting. It is a processing and mapping toolset, not an analysis system.

**Limitation for our use case.** It targets recreation-grade Humminbird/Lowrance
hardware. NIOT-class surveys use scientific/AUV sonar (e.g. Marine Sonic, Klein,
EdgeTech), whose raw formats PINGMapper does not read.

---

## sidescantools (sonoware) — **GPL-3.0**

Processes side-scan data and exports high-resolution georeferenced output.
Studied for its correction chain. **Not vendored:** GPL-3.0 would impose copyleft
on the combined work, which conflicts with the licence-clean path we want to keep
open.

---

## AI4Shipwrecks

- Repository: `umfieldrobotics/AI4Shipwrecks` — MIT (site repo; **dataset terms
  must be checked separately**)
- Paper: *Machine Learning for Shipwreck Segmentation from Side Scan Sonar
  Imagery: Dataset and Benchmark*, arXiv 2401.14546

286 high-resolution AUV side-scan images of shipwreck sites, labelled with
marine-archaeologist consultation, framed as a **segmentation** benchmark.

**Why we did not use it.** It is a single-class (wreck) segmentation dataset;
wrecks are large, high-contrast targets — the *easy* end of the problem. PS 26057
is dominated by small debris against natural clutter. It remains the right dataset
for adding a `shipwreck_structure` class later.

---

## MILCO / NOMBO — the dataset we actually trained on

- Pessanha Santos & Moura, *Data in Brief* 53:110132 (2024); figshare
  `10.6084/m9.figshare.24574879`; **CC BY 4.0**, ungated.

1,170 real side-scan frames from a Teledyne Gavia AUV, 2010–2021, annotated in
YOLO format as MILCO (mine-like contact) or NOMBO (non-mine-like bottom object).

**Why it is a better fit for PS 26057 than it first appears.** The problem
statement's core difficulty is *separating artificial anomalies from natural
seabed structure*. This dataset is built around exactly that distinction: MILCO
vs NOMBO **is** the man-made-vs-ambiguous decision, and 74% of its frames contain
no target at all, which is what makes false-positive rate measurable. It also
spans 11 years of surveys, giving real acquisition-domain shift for an honest
cross-survey split.

**What it is not.** It contains no ghost fishing gear and no navigation data.
Neither ghost-net performance nor real-world geolocation accuracy can be
established from it. See `docs/LIMITATIONS.md`.

---

## Also reviewed

| Work | Note |
|---|---|
| S3Simulator (arXiv 2408.12833) | Synthetic SSS benchmark generator. Useful for future robustness testing; synthetic data cannot substitute for real acquisition noise. |
| Sonar Image Datasets: A Comprehensive Survey (arXiv 2510.03353) | Survey used to confirm we had not missed a better-suited public dataset. |
| SeabedObjects-KLSG / KLSG-II, Marine-PULSE | Real SSS wreck datasets, but access is by request rather than open download. |
| Seaclear Marine Debris | 8,610 images, 40 categories — **optical ROV imagery, not sonar.** Out of scope. |
| OpenSidescan (CIDCO) | Sidescan viewer/annotator. Repository not resolvable via the GitHub API at time of writing. |
