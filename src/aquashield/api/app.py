"""AQUA-SHIELD REST API.

Exists so the pipeline can be driven by something other than the dashboard --
a survey processing queue, a shipboard service, or another team's tooling.
OpenAPI docs are generated automatically at /docs.

Runs entirely locally. No outbound calls.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..confidence.calibration import PlattCalibrator
from ..confidence.fp_filter import LearnedFPFilter
from ..detection.detector import Detector
from ..detection.taxonomy import Taxonomy
from ..device import select_device
from ..geolocation import (NavigationReference, NoGeoReference, SonarGeometry,
                           load_nav_csv)
from ..pipeline import AquaShieldPipeline, PipelineConfig
from ..reporting import build_report, csv_string
from ..sonar.preprocess import PROFILES
from ..storage import AquaShieldDB

app = FastAPI(
    title="AQUA-SHIELD API",
    version=__version__,
    description=(
        "Automated detection, verification, localisation and reporting of "
        "man-made anomalies in side-scan sonar imagery.\n\n"
        "**Confidence semantics:** when `calibrated` is false, `confidence_pct` is a "
        "raw detector score, not a probability. **Geolocation:** hazards without "
        "navigation metadata are returned with null coordinates — the service never "
        "estimates a position it cannot compute."
    ),
)

DB = AquaShieldDB(os.environ.get("AQS_DB_PATH", "outputs/aquashield.db"))
_CACHE: dict = {}


# --------------------------------------------------------------------- models
class SurveyIn(BaseModel):
    survey_id: str = Field(..., min_length=1, max_length=64)
    name: str | None = None
    notes: str | None = None


class ProcessOptions(BaseModel):
    preprocess_profile: str = "standard"
    detector_conf: float = Field(0.10, ge=0.0, le=1.0)
    use_fp_filter: bool = True
    tile_size: int = Field(640, ge=64, le=4096)
    tile_overlap: int = Field(128, ge=0, le=2048)
    min_report_confidence_pct: float = Field(0.0, ge=0.0, le=100.0)
    max_range_m: float | None = None
    gps_accuracy_m: float = 5.0
    heading_accuracy_deg: float = 2.0
    layback_uncertainty_m: float = 3.0
    altitude_m: float | None = None


def _model_path() -> str:
    p = os.environ.get("AQS_MODEL_PATH")
    if p and Path(p).exists():
        return p
    for c in [*sorted(Path("models").glob("*.pt")),
              *sorted(Path(".").glob("runs/**/weights/best.pt"))]:
        return str(c)
    raise HTTPException(503, "No trained detector is available. AQUA-SHIELD does not "
                             "run with a placeholder model. Train one with "
                             "scripts/train.py, or set AQS_MODEL_PATH.")


def _pipeline(opts: ProcessOptions) -> AquaShieldPipeline:
    key = (_model_path(), opts.detector_conf, opts.tile_size)
    if key not in _CACHE:
        _CACHE[key] = Detector(key[0], conf=opts.detector_conf)
    det = _CACHE[key]
    filt = LearnedFPFilter.load("models/fp_filter_milco_nombo.json")
    cal = PlattCalibrator.load("models/calibration_milco_nombo.json")
    cfg = PipelineConfig(
        preprocess_profile=opts.preprocess_profile,
        preprocess_config=PROFILES.get(opts.preprocess_profile),
        detector_conf=opts.detector_conf, use_fp_filter=opts.use_fp_filter,
        tile_size=opts.tile_size, tile_overlap=opts.tile_overlap,
        min_report_confidence_pct=opts.min_report_confidence_pct)
    return AquaShieldPipeline(det, cfg, fp_filter=(filt if opts.use_fp_filter else None),
                              calibrator=cal, taxonomy=Taxonomy("milco_nombo"))


# -------------------------------------------------------------------- routes
@app.get("/api/health", tags=["system"])
def health():
    d = select_device("auto")
    try:
        model = _model_path()
    except HTTPException:
        model = None
    return {"status": "ok" if model else "degraded",
            "version": __version__, "device": d.device, "device_reason": d.reason,
            "model_available": model is not None, "model": model,
            "learned_fp_filter": getattr(
                LearnedFPFilter.load("models/fp_filter_milco_nombo.json"), "fitted", False),
            "calibration": getattr(
                PlattCalibrator.load("models/calibration_milco_nombo.json"), "fitted", False),
            "database": DB.stats()}


@app.get("/api/taxonomy", tags=["system"])
def taxonomy():
    t = Taxonomy("milco_nombo")
    return {"source": t.source, "license": t.license, "citation": t.citation,
            "classes": {cid: {"native": e.native_name, "level1": e.level1,
                              "level2": e.level2, "note": e.note}
                        for cid, e in t._by_id.items()}}


@app.post("/api/surveys", status_code=201, tags=["surveys"])
def create_survey(s: SurveyIn):
    with DB._conn() as c:
        c.execute("INSERT OR REPLACE INTO surveys VALUES (?,?,?,?,?)",
                  (s.survey_id, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   s.name or s.survey_id, 0, s.notes))
    return {"survey_id": s.survey_id, "created": True}


@app.get("/api/surveys", tags=["surveys"])
def list_surveys():
    return {"surveys": DB.list_surveys()}


@app.get("/api/surveys/{survey_id}", tags=["surveys"])
def get_survey(survey_id: str):
    s = DB.get_survey(survey_id)
    if not s:
        raise HTTPException(404, f"survey '{survey_id}' not found")
    return s


@app.post("/api/process", tags=["processing"])
async def process(
    files: list[UploadFile] = File(..., description="Sonar frames (PNG/JPG/TIFF)"),
    survey_id: str = Form("SURVEY-ADHOC"),
    navigation: UploadFile | None = File(None, description="Optional navigation CSV"),
    options: str | None = Form(None, description="JSON ProcessOptions"),
):
    import json as _json
    opts = ProcessOptions(**_json.loads(options)) if options else ProcessOptions()

    frames = []
    for f in files:
        raw = await f.read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(400, f"could not decode '{f.filename}'")
        frames.append((Path(f.filename).stem, img))
    if not frames:
        raise HTTPException(400, "no readable frames supplied")

    georef = NoGeoReference()
    if navigation is not None:
        tmp = Path("outputs/_api_nav.csv")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(await navigation.read())
        try:
            nav = load_nav_csv(tmp)
            if opts.max_range_m is None:
                raise HTTPException(
                    400, "navigation supplied but 'max_range_m' is missing. Without the "
                         "per-channel slant range, a pixel column cannot be converted to "
                         "a ground distance, and AQUA-SHIELD will not guess it.")
            georef = NavigationReference(nav, frames[0][1].shape[:2], SonarGeometry(
                max_range_m=opts.max_range_m, gps_accuracy_m=opts.gps_accuracy_m,
                heading_accuracy_deg=opts.heading_accuracy_deg,
                layback_uncertainty_m=opts.layback_uncertainty_m,
                altitude_m=opts.altitude_m))
        except HTTPException:
            raise
        except Exception as e:                                    # noqa: BLE001
            raise HTTPException(400, f"navigation file rejected: {e}") from e

    pipe = _pipeline(opts)
    res = pipe.process_survey(frames, survey_id=survey_id, georef=georef,
                              make_previews=False)
    run_id = DB.save_run(res)
    return {"run_id": run_id, "survey_id": res.survey_id, "summary": res.summary,
            "provenance": res.provenance,
            "hazards": [h.as_dict() for h in res.hazards]}


@app.get("/api/runs/{run_id}", tags=["processing"])
def get_run(run_id: str):
    r = DB.get_run(run_id)
    if not r:
        raise HTTPException(404, f"run '{run_id}' not found")
    return r


@app.get("/api/hazards", tags=["hazards"])
def list_hazards(survey_id: str | None = None, run_id: str | None = None,
                 min_priority: float = 0.0, band: str | None = None,
                 geolocated_only: bool = False, limit: int = 200):
    return {"hazards": DB.list_hazards(survey_id, run_id, min_priority, band,
                                       geolocated_only, limit)}


@app.get("/api/hazards/{hazard_id}", tags=["hazards"])
def get_hazard(hazard_id: str, run_id: str | None = None):
    h = DB.get_hazard(hazard_id, run_id)
    if not h:
        raise HTTPException(404, f"hazard '{hazard_id}' not found")
    return h


@app.get("/api/reports/{run_id}", tags=["reports"])
def get_report(run_id: str, format: str = "json"):
    r = DB.get_run(run_id)
    if not r:
        raise HTTPException(404, f"run '{run_id}' not found")
    from ..reporting.schema import HazardRecord
    recs = [HazardRecord(**{k: v for k, v in h.items()
                            if k in HazardRecord.__dataclass_fields__})
            for h in r["hazards"]]
    if format == "csv":
        return PlainTextResponse(csv_string(recs), media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="{run_id}.csv"'})
    if format == "geojson":
        feats = [{"type": "Feature",
                  "geometry": {"type": "Point", "coordinates": [h.longitude, h.latitude]},
                  "properties": h.flat()} for h in recs if h.latitude is not None]
        skipped = len(recs) - len(feats)
        return JSONResponse({"type": "FeatureCollection", "features": feats,
                             "aqua_shield_note":
                                 f"{skipped} hazard(s) omitted: no position fix."})
    return build_report(recs, survey_id=r["survey_id"], summary=r["summary"],
                        provenance=r["provenance"])
