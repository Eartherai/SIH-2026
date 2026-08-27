"""Navigation table ingestion.

Survey navigation arrives in many shapes (PINGMapper CSV exports, vendor track
files, hand-made logs). We accept a CSV with flexible column naming and we are
explicit about what we could and could not find. A missing heading column is
reported, not silently replaced by zero -- a zero heading would rotate every
across-track offset by an arbitrary amount and put hazards in the wrong place.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ALIASES = {
    "latitude": ["lat", "latitude", "lat_deg", "y", "northing_deg", "gps_lat"],
    "longitude": ["lon", "long", "longitude", "lon_deg", "x", "easting_deg", "gps_lon"],
    "heading": ["heading", "hdg", "course", "cog", "heading_deg", "track_deg", "azimuth"],
    "altitude": ["altitude", "alt", "height", "fish_altitude", "altitude_m", "range_to_bottom"],
    "ping": ["ping", "ping_id", "record_num", "index", "chunk_i", "ping_number"],
    "time": ["time", "timestamp", "utc", "datetime", "time_s", "caltime"],
    "depth": ["depth", "water_depth", "depth_m"],
}


def _norm(s: str) -> str:
    return "".join(ch for ch in s.strip().lower() if ch.isalnum() or ch == "_")


class NavTable:
    """Per-ping navigation, indexed so an image row can be mapped to a fix."""

    def __init__(self, rows: list[dict], found: dict[str, str], missing: list[str],
                 source: str = ""):
        self.rows = rows
        self.found = found
        self.missing = missing
        self.source = source
        self.lat = np.array([r["latitude"] for r in rows], np.float64)
        self.lon = np.array([r["longitude"] for r in rows], np.float64)
        self.heading = np.array([r.get("heading", np.nan) for r in rows], np.float64)
        self.altitude = np.array([r.get("altitude", np.nan) for r in rows], np.float64)
        self.ping = np.array([r.get("ping", i) for i, r in enumerate(rows)], np.float64)
        if np.isnan(self.heading).all():
            self.heading = self._derive_heading()
            self.derived_heading = True
        else:
            self.derived_heading = False

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def has_altitude(self) -> bool:
        return bool(np.isfinite(self.altitude).any())

    def _derive_heading(self) -> np.ndarray:
        """Course-over-ground from successive fixes.

        Used only when no heading column exists. This is the vessel's TRACK, not
        the towfish's true heading; in a crosscurrent they differ, so the caller
        should widen heading_accuracy_deg. We flag it via `derived_heading`.
        """
        n = len(self.lat)
        hdg = np.zeros(n)
        if n < 2:
            return hdg
        for i in range(n):
            j = min(i + 1, n - 1)
            k = max(i - 1, 0)
            dlat = math.radians(self.lat[j] - self.lat[k])
            mlat = math.radians((self.lat[j] + self.lat[k]) / 2)
            dlon = math.radians(self.lon[j] - self.lon[k]) * math.cos(mlat)
            hdg[i] = (math.degrees(math.atan2(dlon, dlat)) + 360.0) % 360.0
        return hdg

    def at_row(self, row: float, image_height: int):
        """Map an image row (along-track) to (lat, lon, heading, altitude).

        Row 0 is the FIRST ping in the frame. Linear interpolation between the
        bracketing nav records.
        """
        n = len(self)
        if n == 0:
            return None
        t = float(np.clip(row / max(image_height - 1, 1), 0.0, 1.0)) * (n - 1)
        i0 = int(math.floor(t))
        i1 = min(i0 + 1, n - 1)
        f = t - i0
        lat = float(self.lat[i0] * (1 - f) + self.lat[i1] * f)
        lon = float(self.lon[i0] * (1 - f) + self.lon[i1] * f)
        h0, h1 = self.heading[i0], self.heading[i1]
        # interpolate heading on the circle, not on the number line
        d = ((h1 - h0 + 180.0) % 360.0) - 180.0
        hdg = (h0 + f * d) % 360.0
        a0, a1 = self.altitude[i0], self.altitude[i1]
        alt = None
        if np.isfinite(a0) and np.isfinite(a1):
            alt = float(a0 * (1 - f) + a1 * f)
        elif np.isfinite(a0):
            alt = float(a0)
        return lat, lon, float(hdg), alt

    def describe(self) -> dict:
        return {
            "source": self.source,
            "pings": len(self),
            "columns_found": self.found,
            "columns_missing": self.missing,
            "heading_derived_from_track": self.derived_heading,
            "has_altitude": self.has_altitude,
            "lat_range": [round(float(self.lat.min()), 6), round(float(self.lat.max()), 6)],
            "lon_range": [round(float(self.lon.min()), 6), round(float(self.lon.max()), 6)],
        }


def load_nav_csv(path: str | Path) -> NavTable:
    """Read a navigation CSV, matching columns case-insensitively via ALIASES."""
    p = Path(path)
    # Skip leading comment lines. Survey exports (and our own demo files) often
    # carry provenance/warning banners above the real header row.
    raw = p.read_text().splitlines()
    start = 0
    for i, line in enumerate(raw):
        if line.strip() and not line.lstrip().startswith(("#", ";", "//")):
            start = i
            break
    from io import StringIO
    with StringIO("\n".join(raw[start:])) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{p} has no header row")
        lookup = {_norm(c): c for c in reader.fieldnames}
        found: dict[str, str] = {}
        for canon, names in ALIASES.items():
            for n in names:
                if n in lookup:
                    found[canon] = lookup[n]
                    break
        if "latitude" not in found or "longitude" not in found:
            raise ValueError(
                f"{p}: could not find latitude/longitude columns. "
                f"Header was {reader.fieldnames}. "
                f"Accepted names: {ALIASES['latitude']} / {ALIASES['longitude']}")

        rows: list[dict] = []
        for rec in reader:
            try:
                row = {"latitude": float(rec[found["latitude"]]),
                       "longitude": float(rec[found["longitude"]])}
            except (TypeError, ValueError):
                continue                        # skip unparseable rows, never fabricate
            for key in ("heading", "altitude", "ping", "depth"):
                if key in found:
                    try:
                        row[key] = float(rec[found[key]])
                    except (TypeError, ValueError):
                        pass
            rows.append(row)

    if not rows:
        raise ValueError(f"{p}: no usable navigation rows")
    missing = [k for k in ("heading", "altitude") if k not in found]
    return NavTable(rows, found, missing, source=str(p))
