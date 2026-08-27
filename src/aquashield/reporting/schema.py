"""The hazard record -- the single structure every export is built from.

One definition, used by the JSON writer, the CSV writer, the SQLite store and
the dashboard, so the three can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class HazardRecord:
    # --- identity -----------------------------------------------------------
    hazard_id: str
    survey_id: str

    # --- classification -----------------------------------------------------
    detector_class: str            # what the model actually emitted
    level1: str                    # MAN_MADE | AMBIGUOUS
    level2: str                    # ghost_fishing_gear | mine_like_object | ...

    # --- confidence ---------------------------------------------------------
    raw_detector_score: float      # 0..1, NOT a probability
    confidence_pct: float          # 0..100 after verification + calibration
    confidence_band: str           # LOW | MEDIUM | HIGH | CRITICAL
    calibrated: bool               # False => confidence_pct is not a probability

    # --- priority -----------------------------------------------------------
    priority_score: float
    priority_band: str

    # --- geometry -----------------------------------------------------------
    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    estimated_length_m: float | None = None
    estimated_width_m: float | None = None
    ground_sample_distance_m: float | None = None

    # --- location -----------------------------------------------------------
    latitude: float | None = None
    longitude: float | None = None
    utm_easting: float | None = None
    utm_northing: float | None = None
    utm_zone: str | None = None
    geolocation_method: str = "none"
    geolocation_confidence: str = "UNAVAILABLE"
    geoloc_uncertainty_m: float | None = None

    # --- provenance ---------------------------------------------------------
    frame_ids: list[str] = field(default_factory=list)
    ping_id: int | None = None
    timestamp_utc: str | None = None
    observation_count: int = 1
    association_mode: str = "none"

    # --- evidence -----------------------------------------------------------
    evidence: dict[str, Any] = field(default_factory=dict)
    fp_filter_verdict: str = ""
    frame_quality_score: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def flat(self) -> dict:
        """One flat row for CSV. Nested evidence is summarised, not dumped."""
        d = self.as_dict()
        d.pop("evidence", None)
        d["frame_ids"] = ";".join(self.frame_ids)
        d["notes"] = " | ".join(self.notes)
        ev = self.evidence or {}
        d["evidence_model"] = ev.get("model")
        d["evidence_shadow"] = ev.get("shadow")
        d["evidence_texture"] = ev.get("texture")
        d["evidence_persistence"] = ev.get("persistence")
        d["evidence_data_quality"] = ev.get("data_quality")
        return d


CSV_COLUMNS = [
    "hazard_id", "survey_id", "detector_class", "level1", "level2",
    "confidence_pct", "confidence_band", "calibrated", "raw_detector_score",
    "priority_score", "priority_band",
    "latitude", "longitude", "utm_easting", "utm_northing", "utm_zone",
    "geolocation_method", "geolocation_confidence", "geoloc_uncertainty_m",
    "estimated_length_m", "estimated_width_m", "ground_sample_distance_m",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    "observation_count", "association_mode", "frame_ids", "ping_id", "timestamp_utc",
    "frame_quality_score", "fp_filter_verdict",
    "evidence_model", "evidence_shadow", "evidence_texture",
    "evidence_persistence", "evidence_data_quality", "notes",
]
