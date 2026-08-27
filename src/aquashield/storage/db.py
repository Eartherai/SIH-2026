"""SQLite store for surveys, processing runs, detections and unique hazards.

SQLite deliberately: a survey's worth of hazards is thousands of rows, not
millions, and the file travels with the results. PostGIS would add an
infrastructure dependency for no capability we currently need. If AQUA-SHIELD
ever needs spatial joins across many surveys, that is the point to migrate --
the schema below maps onto PostGIS without change.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS surveys (
    survey_id     TEXT PRIMARY KEY,
    created_utc   TEXT NOT NULL,
    name          TEXT,
    frame_count   INTEGER,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    survey_id     TEXT NOT NULL REFERENCES surveys(survey_id),
    started_utc   TEXT NOT NULL,
    seconds       REAL,
    provenance    TEXT NOT NULL,   -- full JSON: model, device, preprocessing, filter, calibration
    summary       TEXT NOT NULL    -- full JSON survey summary
);

CREATE TABLE IF NOT EXISTS hazards (
    hazard_id        TEXT NOT NULL,
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    survey_id        TEXT NOT NULL,
    detector_class   TEXT,
    level1           TEXT,
    level2           TEXT,
    confidence_pct   REAL,
    confidence_band  TEXT,
    calibrated       INTEGER,
    priority_score   REAL,
    priority_band    TEXT,
    latitude         REAL,          -- NULL when no fix; never 0 as a placeholder
    longitude        REAL,
    geoloc_uncert_m  REAL,
    observation_count INTEGER,
    record           TEXT NOT NULL, -- full JSON HazardRecord
    PRIMARY KEY (run_id, hazard_id)
);

CREATE TABLE IF NOT EXISTS frames (
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    frame_id      TEXT NOT NULL,
    frame_index   INTEGER,
    quality_score REAL,
    n_raw         INTEGER,
    n_accepted    INTEGER,
    n_rejected    INTEGER,
    qc            TEXT,
    PRIMARY KEY (run_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_hazards_survey   ON hazards(survey_id);
CREATE INDEX IF NOT EXISTS idx_hazards_priority ON hazards(priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_hazards_latlon   ON hazards(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_runs_survey      ON runs(survey_id);
"""


class AquaShieldDB:
    def __init__(self, path: str | Path = "outputs/aquashield.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ------------------------------------------------------------------ write
    def save_run(self, result, run_id: str | None = None, name: str | None = None) -> str:
        """Persist a SurveyResult. Returns the run_id."""
        run_id = run_id or f"RUN-{int(time.time()*1000):x}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO surveys VALUES (?,?,?,?,?)",
                      (result.survey_id, now, name or result.survey_id,
                       result.summary.get("frames_processed"), None))
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
                      (run_id, result.survey_id, now,
                       result.summary.get("processing_seconds"),
                       json.dumps(result.provenance, default=str),
                       json.dumps(result.summary, default=str)))
            for h in result.hazards:
                c.execute("INSERT OR REPLACE INTO hazards VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (h.hazard_id, run_id, result.survey_id, h.detector_class,
                           h.level1, h.level2, h.confidence_pct, h.confidence_band,
                           int(h.calibrated), h.priority_score, h.priority_band,
                           h.latitude, h.longitude, h.geoloc_uncertainty_m,
                           h.observation_count, json.dumps(h.as_dict(), default=str)))
            for f in result.frames:
                c.execute("INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?,?,?,?)",
                          (run_id, f.frame_id, f.frame_index, f.qc.get("quality_score"),
                           len(f.raw_detections), len(f.accepted), len(f.rejected),
                           json.dumps(f.qc, default=str)))
        return run_id

    # ------------------------------------------------------------------- read
    def list_surveys(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT s.*, COUNT(DISTINCT r.run_id) AS runs FROM surveys s "
                "LEFT JOIN runs r USING(survey_id) GROUP BY s.survey_id "
                "ORDER BY s.created_utc DESC")]

    def get_survey(self, survey_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM surveys WHERE survey_id=?", (survey_id,)).fetchone()
            if not row:
                return None
            runs = [dict(r) for r in c.execute(
                "SELECT run_id, started_utc, seconds, summary FROM runs "
                "WHERE survey_id=? ORDER BY started_utc DESC", (survey_id,))]
            for r in runs:
                r["summary"] = json.loads(r["summary"])
            return {**dict(row), "runs": runs}

    def list_hazards(self, survey_id: str | None = None, run_id: str | None = None,
                     min_priority: float = 0.0, band: str | None = None,
                     geolocated_only: bool = False, limit: int = 500) -> list[dict]:
        q = "SELECT record FROM hazards WHERE priority_score >= ?"
        args: list = [min_priority]
        if survey_id:
            q += " AND survey_id = ?"
            args.append(survey_id)
        if run_id:
            q += " AND run_id = ?"
            args.append(run_id)
        if band:
            q += " AND priority_band = ?"
            args.append(band)
        if geolocated_only:
            q += " AND latitude IS NOT NULL"
        q += " ORDER BY priority_score DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [json.loads(r["record"]) for r in c.execute(q, args)]

    def get_hazard(self, hazard_id: str, run_id: str | None = None) -> dict | None:
        with self._conn() as c:
            if run_id:
                r = c.execute("SELECT record FROM hazards WHERE hazard_id=? AND run_id=?",
                              (hazard_id, run_id)).fetchone()
            else:
                r = c.execute("SELECT record FROM hazards WHERE hazard_id=? "
                              "ORDER BY rowid DESC LIMIT 1", (hazard_id,)).fetchone()
            return json.loads(r["record"]) if r else None

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not r:
                return None
            d = dict(r)
            d["provenance"] = json.loads(d["provenance"])
            d["summary"] = json.loads(d["summary"])
            d["hazards"] = self.list_hazards(run_id=run_id, limit=10_000)
            return d

    def stats(self) -> dict:
        with self._conn() as c:
            g = lambda q: c.execute(q).fetchone()[0]      # noqa: E731
            return {
                "surveys": g("SELECT COUNT(*) FROM surveys"),
                "runs": g("SELECT COUNT(*) FROM runs"),
                "hazards": g("SELECT COUNT(*) FROM hazards"),
                "geolocated_hazards": g("SELECT COUNT(*) FROM hazards WHERE latitude IS NOT NULL"),
                "high_priority": g("SELECT COUNT(*) FROM hazards WHERE priority_band IN ('HIGH','URGENT')"),
                "db_path": str(self.path),
                "db_size_bytes": self.path.stat().st_size if self.path.exists() else 0,
            }
