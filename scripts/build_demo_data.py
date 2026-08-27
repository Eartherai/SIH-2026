#!/usr/bin/env python3
"""Build small, fast, honest demo scenarios for the dashboard and live demo.

Four scenarios, chosen so the demo cannot be dismissed as "one lucky image":

  01_clear_targets      frames whose annotated targets are large and high-contrast
  02_hard_targets       frames whose annotated targets are small / low-contrast
  03_natural_seabed     frames with NO annotated target at all -- the false-positive
                        challenge. A good run should produce few or no hazards here.
  04_georeferenced      a contiguous frame sequence paired with a navigation track,
                        so geolocation, deduplication and the map can be shown.

HONESTY NOTE ON SCENARIO 04
---------------------------
The MILCO/NOMBO archive ships imagery WITHOUT navigation data. The track in
04_georeferenced is therefore SYNTHETIC and is labelled as such in scenario.json,
in the CSV header, and on screen. It exists to exercise and demonstrate the
geolocation maths on a known geometry. The coordinates it produces describe a
fictional survey line and must never be read as the true position of these
objects. Every other scenario runs with no navigation data at all and correctly
reports "Geolocation unavailable".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aquashield.detection.boxes import xywhn_to_xyxy  # noqa: E402

SRC = Path("data/processed/milco_nombo_yolo")
OUT = Path("demo_data")

ATTRIBUTION = (
    "Imagery: Pessanha Santos, N. & Moura, R. (2024), 'Side-scan sonar imaging data "
    "of underwater vehicles for mine detection', Data in Brief 53:110132. "
    "figshare DOI 10.6084/m9.figshare.24574879. Licensed CC BY 4.0."
)


def frame_stats(img_p: Path, lbl_p: Path):
    g = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    h, w = g.shape
    rows = [r.split() for r in lbl_p.read_text().splitlines() if r.strip()] \
        if lbl_p.exists() else []
    if not rows:
        return {"path": img_p, "n": 0, "area": 0.0, "contrast": 0.0}
    arr = np.array([[float(v) for v in r] for r in rows], np.float32)
    boxes = xywhn_to_xyxy(arr[:, 1:5], w, h)
    g01 = g.astype(np.float32) / 255.0
    contrasts, areas = [], []
    for (x0, y0, x1, y1) in boxes.astype(int):
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = g01[y0:y1, x0:x1]
        pad = 20
        ring = g01[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        contrasts.append(float(patch.max() - ring.mean()))
        areas.append(float((x1 - x0) * (y1 - y0)) / (h * w))
    return {"path": img_p, "n": len(boxes),
            "area": float(np.mean(areas)) if areas else 0.0,
            "contrast": float(np.mean(contrasts)) if contrasts else 0.0}


def write_scenario(name: str, title: str, desc: str, short: str, imgs: list[Path],
                   nav_rows: list[dict] | None = None, extra: dict | None = None):
    d = OUT / name
    if d.exists():
        shutil.rmtree(d)
    (d / "images").mkdir(parents=True)
    for p in imgs:
        shutil.copy2(p, d / "images" / p.name)

    if nav_rows:
        with (d / "navigation.csv").open("w", newline="") as f:
            f.write("# SYNTHETIC NAVIGATION - NOT REAL SURVEY POSITIONS.\n")
            f.write("# The source imagery ships without navigation data. This track\n")
            f.write("# exists only to exercise the geolocation maths on a known\n")
            f.write("# geometry. Do not treat the resulting coordinates as real.\n")
            w = csv.DictWriter(f, fieldnames=list(nav_rows[0].keys()))
            w.writeheader()
            w.writerows(nav_rows)

    meta = {
        "scenario": name, "title": title, "description": desc,
        "short_description": short, "frame_count": len(imgs),
        "frames": [p.name for p in imgs],
        "source_dataset": "MILCO/NOMBO side-scan sonar (figshare 24574879)",
        "license": "CC BY 4.0", "attribution": ATTRIBUTION,
        "survey_id": f"AQS-DEMO-{name.split('_')[0]}",
        "navigation": ("SYNTHETIC - see navigation.csv header" if nav_rows
                       else "none - geolocation will report UNAVAILABLE"),
        **(extra or {}),
    }
    (d / "scenario.json").write_text(json.dumps(meta, indent=2))
    print(f"  {name:22s} {len(imgs):3d} frames"
          f"{'  + synthetic nav' if nav_rows else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=8)
    args = ap.parse_args()

    if not SRC.exists():
        print(f"{SRC} not found. Run scripts/prepare_milco_nombo.py first.")
        return
    OUT.mkdir(exist_ok=True)

    # Draw demo frames from the TEST surveys only, so the demo never shows the
    # model frames it was trained on.
    split = SRC / "test"
    stats = []
    for ip in sorted((split / "images").glob("*.jpg")):
        s = frame_stats(ip, (split / "labels" / ip.name).with_suffix(".txt"))
        if s:
            stats.append(s)
    pos = [s for s in stats if s["n"] > 0]
    neg = [s for s in stats if s["n"] == 0]
    print(f"test split: {len(pos)} frames with targets, {len(neg)} empty\n")

    n = args.per_scenario
    clear = sorted(pos, key=lambda s: -(s["contrast"] * math.sqrt(s["area"] + 1e-9)))[:n]
    hard = sorted(pos, key=lambda s: (s["contrast"] * math.sqrt(s["area"] + 1e-9)))[:n]

    write_scenario(
        "01_clear_targets", "Clear man-made targets",
        "Held-out frames whose annotated targets are comparatively large and "
        "high-contrast. The baseline case.",
        "Large, high-contrast targets on held-out surveys.",
        [s["path"] for s in clear])

    write_scenario(
        "02_hard_targets", "Low-contrast / small targets",
        "Held-out frames whose annotated targets are small and poorly separated "
        "from the seabed. Expect lower confidence and some misses - this scenario "
        "exists to show the failure mode honestly, not to hide it.",
        "Small, low-contrast targets. Expect misses.",
        [s["path"] for s in hard])

    write_scenario(
        "03_natural_seabed", "Natural seabed (false-positive challenge)",
        "Held-out frames containing NO annotated target: ripples, rock texture, "
        "nadir band and shadow only. Every hazard reported here is a false "
        "positive. This is the scenario PS 26057 actually cares about.",
        "No targets present. Every hit here is a false positive.",
        [s["path"] for s in neg[: n * 2]])

    # --- georeferenced scenario -------------------------------------------
    seq = [s["path"] for s in sorted(pos, key=lambda s: s["path"].name)[: n * 2]]
    # A plausible survey line off the Tamil Nadu coast, ~2 kn, 0.15 m/ping.
    lat0, lon0, heading = 12.9200, 80.3400, 22.0
    rows = []
    per_frame_pings = 40
    step_m = 0.15
    for i in range(len(seq) * per_frame_pings):
        d = i * step_m
        dlat = d * math.cos(math.radians(heading)) / 111_320.0
        dlon = (d * math.sin(math.radians(heading))
                / (111_320.0 * math.cos(math.radians(lat0))))
        rows.append({"ping": i, "lat": round(lat0 + dlat, 8), "lon": round(lon0 + dlon, 8),
                     "heading": heading, "altitude": 11.0})
    write_scenario(
        "04_georeferenced", "Georeferenced survey line (SYNTHETIC navigation)",
        "A contiguous frame sequence paired with a SYNTHETIC navigation track so "
        "that geolocation, positional uncertainty, spatial deduplication and the "
        "map can be demonstrated. The coordinates describe a fictional survey "
        "line and are NOT the true positions of these objects.",
        "Geolocation + map demo. Navigation track is SYNTHETIC.",
        seq, nav_rows=rows,
        extra={"synthetic_navigation": True,
               "synthetic_navigation_warning":
                   "Coordinates produced from this scenario are fictional. They "
                   "demonstrate the geolocation computation, not real object "
                   "positions.",
               "track": {"start_lat": lat0, "start_lon": lon0,
                         "heading_deg": heading, "ping_spacing_m": step_m,
                         "altitude_m": 11.0}})

    (OUT / "README.md").write_text(
        "# AQUA-SHIELD demo data\n\n"
        "Small curated scenarios for the dashboard and the live demo. All imagery "
        "is drawn from the **held-out test surveys** (2018 and 2021), so nothing "
        "shown in the demo was seen during training.\n\n"
        "| Scenario | Purpose |\n|---|---|\n"
        "| `01_clear_targets` | Large, high-contrast man-made targets |\n"
        "| `02_hard_targets` | Small / low-contrast targets - shows the failure mode |\n"
        "| `03_natural_seabed` | No targets at all; every detection is a false positive |\n"
        "| `04_georeferenced` | Geolocation, dedup and map, using a **synthetic** track |\n\n"
        "## Navigation data\n\n"
        "Only `04_georeferenced` has navigation, and it is **synthetic** - the source "
        "dataset ships no positions. It exercises the geolocation maths on a known "
        "geometry. The other scenarios correctly report *Geolocation unavailable*.\n\n"
        f"## Attribution\n\n{ATTRIBUTION}\n")
    print(f"\nwrote {OUT}/ ({sum(1 for _ in OUT.glob('*/'))} scenarios)")


if __name__ == "__main__":
    main()
