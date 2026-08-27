# Sources

Every external claim in this repository traces to something here. Retrieved
2026-08-27. Where a page could not be retrieved, that is recorded too.

## Datasets

1. **MILCO/NOMBO** — Pessanha Santos, N. & Moura, R. (2024). "Side-scan sonar
   imaging data of underwater vehicles for mine detection." *Data in Brief*
   53:110132. DOI `10.1016/j.dib.2024.110132`
   - Article: https://pmc.ncbi.nlm.nih.gov/articles/PMC10879765/
   - Data (CC BY 4.0, ungated): https://dx.doi.org/10.6084/m9.figshare.24574879
   - Used for: **all training and all reported metrics.**

2. **sss-crab-pot-detection-ds** — PINGEcosystem, HuggingFace. DOI `10.57967/hf/8397`
   - https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds
   - Licence metadata: `cc-by-sa-4.0`; README text says "GPL" (they disagree).
   - **Status: gated — HTTP 403. Not used.**

3. **AI4Shipwrecks** — Sethuraman et al. "Machine Learning for Shipwreck
   Segmentation from Side Scan Sonar Imagery: Dataset and Benchmark."
   arXiv:2401.14546 — https://arxiv.org/pdf/2401.14546
   - Repo: https://github.com/umfieldrobotics/AI4Shipwrecks (MIT)

4. **S3Simulator** — "A benchmarking Side Scan Sonar Simulator dataset for
   Underwater Image Analysis." arXiv:2408.12833 — https://arxiv.org/html/2408.12833v1

5. **Sonar Image Datasets: A Comprehensive Survey of Resources, Challenges, and
   Applications.** arXiv:2510.03353 — https://arxiv.org/pdf/2510.03353
   - Used to confirm no better-suited *open, ungated* dataset was missed.

6. **Seaclear Marine Debris Dataset** — https://pmc.ncbi.nlm.nih.gov/articles/PMC11344804/
   - 8,610 images / 40 classes, but **optical ROV imagery, not sonar.** Out of scope.

## Software / prior art

7. **GhostVision** — "Democratizing Derelict Gear Detection Using Low-Cost Sonar
   and Artificial Intelligence." *J. Mar. Sci. Eng.* 14(10):951, 2025.
   - https://www.mdpi.com/2077-1312/14/10/951 (**returned HTTP 403 to automated
     fetch**; the figures quoted in `prior_art.md` come from the indexed abstract
     and should be re-verified against the PDF before being put in a slide.)
   - Code: https://github.com/PINGEcosystem/GhostVision — licence **NOASSERTION**
   - PyPI: https://pypi.org/project/ghostvision/ (1.0.0)

8. **PINGMapper** — Bodine, C. S., Buscombe, D., Best, R. J., Redner, J. A. &
   Kaeser, A. J. (2022). "PING-Mapper: Open-Source Software for Automated Benthic
   Imaging and Mapping Using Recreation-Grade Sonar." *Earth and Space Science*.
   DOI `10.1029/2022EA002469`
   - https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022EA002469
   - Repo: https://github.com/CameronBodine/PINGMapper — **MIT** (verified via GitHub API)
   - Docs: https://cameronbodine.github.io/PINGMapper/

9. **sidescantools** — https://github.com/sonoware/sidescantools — **GPL-3.0**

## Method references

10. **Lee, J. S. (1980).** "Digital image enhancement and noise filtering by use
    of local statistics." *IEEE TPAMI* 2(2):165–168.
    - Basis of `sonar/preprocess.py::lee_filter`. Chosen because sonar speckle is
      *multiplicative*, which a Gaussian blur handles badly.

11. **Platt, J. (1999).** "Probabilistic outputs for support vector machines and
    comparisons to regularized likelihood methods."
    - Basis of `confidence/calibration.py::PlattCalibrator`.

12. **Guo, C. et al. (2017).** "On Calibration of Modern Neural Networks." ICML.
    - Basis for reporting Expected Calibration Error and reliability curves.

13. **Ultralytics YOLO11** — https://docs.ultralytics.com — **AGPL-3.0-or-later**

## Verified programmatically, not from memory

Licences for PINGMapper (MIT), GhostVision (NOASSERTION), sidescantools
(GPL-3.0) and AI4Shipwrecks (MIT) were read from the GitHub REST API; Python
package licences were read from installed distribution metadata. See
`LEGAL_AND_LICENSES.md`.

## Not verified

- **SIH 2026 official evaluation criteria.** No authoritative current source was
  located during this work. Nothing in this repository claims to satisfy a
  specific SIH scoring rubric.
- GhostVision's exact metric table (see item 7).
