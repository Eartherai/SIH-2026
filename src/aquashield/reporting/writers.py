"""Report writers: JSON (nested, full evidence) and CSV (flat, one row per hazard)."""

from __future__ import annotations

import csv
import json
import platform
import time
from io import StringIO
from pathlib import Path

from .. import __version__
from .schema import CSV_COLUMNS, HazardRecord


def build_report(hazards: list[HazardRecord], *, survey_id: str, summary: dict,
                 provenance: dict) -> dict:
    """Assemble the full nested report.

    `provenance` records exactly which model, dataset, preprocessing profile and
    calibration produced these numbers, so a report can always be traced back to
    the run that made it.
    """
    return {
        "aqua_shield": {
            "version": __version__,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": {"platform": platform.platform(), "machine": platform.machine()},
        },
        "survey": {"survey_id": survey_id, **summary},
        "provenance": provenance,
        "hazards": [h.as_dict() for h in hazards],
        "disclaimer": (
            "Confidence values are produced by an automated detector. Where "
            "'calibrated' is false they are RAW detector scores and must not be "
            "read as probabilities. Coordinates carry an uncertainty in metres; "
            "hazards without a position fix are reported with null coordinates "
            "rather than an estimated position."
        ),
    }


def write_json(report: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str))
    return p


def hazards_to_csv_rows(hazards: list[HazardRecord]) -> list[dict]:
    rows = []
    for h in hazards:
        f = h.flat()
        rows.append({c: f.get(c, "") for c in CSV_COLUMNS})
    return rows


def write_csv(hazards: list[HazardRecord], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(hazards_to_csv_rows(hazards))
    return p


def csv_string(hazards: list[HazardRecord]) -> str:
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    w.writerows(hazards_to_csv_rows(hazards))
    return buf.getvalue()


def write_geojson(hazards: list[HazardRecord], path: str | Path) -> Path:
    """GeoJSON of the geolocated subset, for direct import into QGIS/ArcGIS.

    Hazards without a fix are OMITTED (not placed at 0,0), and the count of
    omitted features is recorded on the FeatureCollection so nothing silently
    disappears.
    """
    feats, skipped = [], 0
    for h in hazards:
        if h.latitude is None or h.longitude is None:
            skipped += 1
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [h.longitude, h.latitude]},
            "properties": {k: v for k, v in h.flat().items() if k not in ("bbox_x0",
                           "bbox_y0", "bbox_x1", "bbox_y1")},
        })
    fc = {"type": "FeatureCollection", "features": feats,
          "aqua_shield_note": f"{skipped} hazard(s) omitted: no position fix available."}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fc, indent=2, default=str))
    return p
