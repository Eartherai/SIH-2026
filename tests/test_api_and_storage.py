"""SQLite storage and the REST API."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aquashield.reporting.schema import HazardRecord  # noqa: E402
from aquashield.storage import AquaShieldDB  # noqa: E402


class _FakeResult:
    """A minimal SurveyResult stand-in, so storage is testable without a model."""
    def __init__(self, survey_id="S1", n_haz=3, geo=True):
        self.survey_id = survey_id
        self.hazards = [
            HazardRecord(
                hazard_id=f"AQS-{i:05d}", survey_id=survey_id, detector_class="MILCO",
                level1="MAN_MADE", level2="mine_like_object", raw_detector_score=0.6,
                confidence_pct=50.0 + i, confidence_band="MEDIUM", calibrated=True,
                priority_score=40.0 + i * 10, priority_band=("HIGH" if i else "ROUTINE"),
                bbox_x0=1, bbox_y0=2, bbox_x1=30, bbox_y1=40,
                latitude=(12.97 + i * 1e-4 if geo else None),
                longitude=(80.25 if geo else None),
                geoloc_uncertainty_m=(6.0 if geo else None),
                observation_count=i + 1)
            for i in range(n_haz)]
        self.frames = [type("F", (), {
            "frame_id": f"f{i}", "frame_index": i,
            "qc": {"quality_score": 0.9}, "raw_detections": [1, 2, 3],
            "accepted": [1], "rejected": [2, 3]})() for i in range(2)]
        self.summary = {"frames_processed": 2, "processing_seconds": 1.5,
                        "unique_hazards": n_haz}
        self.provenance = {"model_path": "x.pt", "device": "mps"}


@pytest.fixture
def db(tmp_path):
    return AquaShieldDB(tmp_path / "t.db")


class TestStorage:
    def test_save_and_read_back(self, db):
        rid = db.save_run(_FakeResult())
        run = db.get_run(rid)
        assert run["survey_id"] == "S1"
        assert len(run["hazards"]) == 3
        assert run["provenance"]["device"] == "mps"

    def test_hazards_ordered_by_priority(self, db):
        db.save_run(_FakeResult())
        h = db.list_hazards(survey_id="S1")
        scores = [x["priority_score"] for x in h]
        assert scores == sorted(scores, reverse=True)

    def test_priority_and_band_filters(self, db):
        db.save_run(_FakeResult())
        assert len(db.list_hazards(min_priority=55)) == 1
        assert all(x["priority_band"] == "HIGH" for x in db.list_hazards(band="HIGH"))

    def test_geolocated_only_filter_excludes_nulls(self, db):
        db.save_run(_FakeResult(survey_id="GEO", geo=True))
        db.save_run(_FakeResult(survey_id="NOGEO", geo=False))
        got = db.list_hazards(geolocated_only=True)
        assert got and all(x["latitude"] is not None for x in got)
        assert len(got) == 3

    def test_null_coordinates_are_stored_as_null_not_zero(self, db):
        db.save_run(_FakeResult(survey_id="NOGEO", geo=False))
        with db._conn() as c:
            rows = c.execute("SELECT latitude, longitude FROM hazards").fetchall()
        assert all(r["latitude"] is None and r["longitude"] is None for r in rows)

    def test_frames_persisted(self, db):
        rid = db.save_run(_FakeResult())
        with db._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM frames WHERE run_id=?", (rid,)).fetchone()[0]
        assert n == 2

    def test_reruns_do_not_collide(self, db):
        r = _FakeResult()
        a, b = db.save_run(r, run_id="RUN-A"), db.save_run(r, run_id="RUN-B")
        assert a != b
        assert db.stats()["hazards"] == 6      # same hazard ids, different runs
        assert len(db.get_run("RUN-A")["hazards"]) == 3

    def test_missing_lookups_return_none(self, db):
        assert db.get_run("nope") is None
        assert db.get_survey("nope") is None
        assert db.get_hazard("nope") is None

    def test_stats(self, db):
        db.save_run(_FakeResult())
        s = db.stats()
        assert s["surveys"] == 1 and s["runs"] == 1 and s["hazards"] == 3
        assert s["geolocated_hazards"] == 3


class TestAPI:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        monkeypatch.setenv("AQS_DB_PATH", str(tmp_path / "api.db"))
        for m in [k for k in list(sys.modules) if k.startswith("aquashield.api")]:
            del sys.modules[m]
        from fastapi.testclient import TestClient
        from aquashield.api.app import app
        return TestClient(app)

    def test_health_reports_device_and_model_state(self, client):
        d = client.get("/api/health").json()
        assert d["device"] in ("mps", "cpu", "cuda")
        assert "model_available" in d and "database" in d
        assert d["status"] in ("ok", "degraded")

    def test_taxonomy_endpoint_preserves_semantics(self, client):
        c = client.get("/api/taxonomy").json()["classes"]
        assert c["0"]["level1"] == "MAN_MADE"
        assert c["1"]["level1"] == "AMBIGUOUS"     # NOMBO is never man-made

    def test_survey_crud(self, client):
        assert client.post("/api/surveys", json={"survey_id": "S9"}).status_code == 201
        assert "S9" in [s["survey_id"] for s in client.get("/api/surveys").json()["surveys"]]
        assert client.get("/api/surveys/S9").status_code == 200

    def test_unknown_ids_are_404(self, client):
        assert client.get("/api/surveys/nope").status_code == 404
        assert client.get("/api/runs/nope").status_code == 404
        assert client.get("/api/hazards/nope").status_code == 404
        assert client.get("/api/reports/nope").status_code == 404

    def test_undecodable_upload_is_rejected(self, client):
        r = client.post("/api/process",
                        files=[("files", ("bad.jpg", b"not an image", "image/jpeg"))])
        assert r.status_code in (400, 503)

    def test_navigation_without_range_is_refused_not_guessed(self, client):
        """Supplying nav but no slant range must error, never assume a geometry."""
        import io
        img = np.zeros((64, 64), np.uint8)
        import cv2
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        r = client.post(
            "/api/process",
            files=[("files", ("a.jpg", buf.tobytes(), "image/jpeg")),
                   ("navigation", ("n.csv", io.BytesIO(b"lat,lon\n12.9,80.3\n12.91,80.3\n"),
                                   "text/csv"))])
        if r.status_code == 503:
            pytest.skip("no trained model available")
        assert r.status_code == 400
        assert "max_range_m" in r.json()["detail"]

    def test_openapi_schema_is_generated(self, client):
        s = client.get("/openapi.json").json()
        assert s["info"]["title"] == "AQUA-SHIELD API"
        for p in ("/api/health", "/api/process", "/api/hazards", "/api/reports/{run_id}"):
            assert p in s["paths"]
