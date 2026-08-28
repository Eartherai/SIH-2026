#!/usr/bin/env python3
"""Download the public sonar datasets AQUA-SHIELD uses.

Only ungated, clearly-licensed data is downloaded automatically. Gated datasets
are reported with instructions rather than being fetched behind an access wall.
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

MILCO_FILES = {           # figshare file ids for DOI 10.6084/m9.figshare.24574879
    "2010.zip": "43169008",
    "2015.zip": "43169002",
    "2017.zip": "43169005",
    "2018.zip": "43169011",
    "2021.zip": "43168999",
}
BASE = "https://ndownloader.figshare.com/files/"


def human(n: int) -> str:
    return f"{n/1e6:.1f} MB"


def download_milco(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print("MILCO/NOMBO side-scan sonar  —  CC BY 4.0  —  figshare 10.6084/m9.figshare.24574879")
    print("  Pessanha Santos, N. & Moura, R. (2024), Data in Brief 53:110132\n")
    total = 0
    for name, fid in MILCO_FILES.items():
        out = dest / name
        if out.exists():
            print(f"  {name:12s} already present ({human(out.stat().st_size)})")
            total += out.stat().st_size
            continue
        print(f"  {name:12s} downloading …", end="", flush=True)
        urllib.request.urlretrieve(BASE + fid, out)
        total += out.stat().st_size
        print(f" {human(out.stat().st_size)}")
    print(f"\n  total {human(total)}")

    for name in MILCO_FILES:
        z = dest / name
        target = dest / f"{z.stem}_x"
        if target.exists():
            continue
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target)
        print(f"  extracted {name} -> {target.name}")


def report_gated() -> None:
    print("\n" + "-" * 72)
    print("GATED DATASET — not downloaded automatically")
    print("-" * 72)
    print("""
sss-crab-pot-detection-ds  (derelict crab pots / ghost fishing gear)
  Host    : HuggingFace, PINGEcosystem/sss-crab-pot-detection-ds
  Licence : CC BY-SA 4.0   DOI 10.57967/hf/8397
  Status  : ACCESS-GATED. The repository returns HTTP 403 until the maintainers
            approve your HuggingFace account.

  This dataset is the closest public match to the 'ghost nets' theme of PS 26057.
  AQUA-SHIELD ships an adapter for it (src/aquashield/ingestion/jsonl_bbox.py)
  but was NOT trained on it, because access could not be obtained.

  Status update (2026-08-28): confirmed still gated even with a valid, authenticated
  HF token -- dataset_info() lists metadata regardless of approval, but every
  actual file (images, metadata.jsonl) returns HTTP 403 x-error-code:GatedRepo.
  The gate type is "auto", meaning NO manual review by the maintainers is needed --
  but a human must still click "Agree and access repository" on the dataset page
  once. No API can do that step.

  To use it:
    1. Open https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds
    2. Click "Agree and access repository" (instant, auto-approved)
    3. export HF_TOKEN=hf_...          (never commit this)
    4. python scripts/prepare_crab_pot.py
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw/milco_nombo")
    ap.add_argument("--skip-milco", action="store_true")
    args = ap.parse_args()

    free = shutil.disk_usage(".").free
    print(f"free disk: {human(free)}\n")
    if free < 2e9:
        print("WARNING: less than 2 GB free. Aborting to avoid filling the disk.")
        return
    if not args.skip_milco:
        download_milco(Path(args.dest))
    report_gated()
    print("Next:  python scripts/prepare_milco_nombo.py")


if __name__ == "__main__":
    main()
