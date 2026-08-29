#!/usr/bin/env python3
"""Crop the raw dashboard screenshots into slide-ready panels.

Boxes are in pixels of the 3000px-wide captures produced by
scripts/capture_dashboard_shots.py. Re-run both scripts together if the
dashboard layout changes.
"""
from pathlib import Path

from PIL import Image

SHOTS = Path(__file__).resolve().parents[1] / "docs" / "images" / "prototype"

# name -> (source file, (left, top, right, bottom))
CROPS = {
    "panel_hero_detection": ("01_detections.png", (750, 940, 2600, 2270)),
    # sonar frame ONLY (no white UI chrome) -- blends into the dark deck
    "panel_sonar": ("01_detections.png", (758, 958, 1998, 2200)),
    "panel_stats": ("01_detections.png", (745, 440, 2850, 650)),
    "panel_register": ("03_register.png", (750, 1120, 2850, 1360)),
    "panel_qc": ("04_evidence.png", (750, 880, 2850, 1560)),
    "panel_geo_table": ("geo_02_map.png", (750, 1950, 2850, 2320)),
}


def main() -> None:
    for name, (src, box) in CROPS.items():
        p = SHOTS / src
        if not p.exists():
            print(f"  MISSING source {src}")
            continue
        im = Image.open(p).crop(box)
        out = SHOTS / f"{name}.png"
        im.save(out)
        print(f"  {out.name}  {im.size}")


if __name__ == "__main__":
    main()
