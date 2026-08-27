"""End-to-end integration, data validation, and a dashboard smoke test.

Tests that need a trained checkpoint SKIP (not fail) when none exists, so a
fresh clone can still run the suite before training.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _weights():
    for p in [*sorted((ROOT / "models").glob("*.pt")),
              *sorted(ROOT.glob("runs/**/weights/best.pt"))]:
        return p
    return None


requires_model = pytest.mark.skipif(_weights() is None,
                                    reason="no trained checkpoint available")
DATA = ROOT / "data" / "processed" / "milco_nombo_yolo"
requires_data = pytest.mark.skipif(not DATA.exists(),
                                   reason="prepared dataset not available")


class TestDataValidation:
    @requires_data
    def test_every_image_has_a_label_file(self):
        for split in ("train", "val", "test"):
            imgs = sorted((DATA / split / "images").glob("*.jpg"))
            assert imgs, f"{split} has no images"
            for ip in imgs:
                lp = (DATA / split / "labels" / ip.name).with_suffix(".txt")
                assert lp.exists(), f"missing label for {ip.name}"

    @requires_data
    def test_all_labels_are_valid_normalised_yolo(self):
        bad = []
        for lp in DATA.glob("*/labels/*.txt"):
            for i, line in enumerate(lp.read_text().splitlines()):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad.append(f"{lp.name}:{i} wrong field count")
                    continue
                c, cx, cy, w, h = (float(v) for v in parts)
                if c not in (0.0, 1.0):
                    bad.append(f"{lp.name}:{i} bad class {c}")
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                    bad.append(f"{lp.name}:{i} out of [0,1]")
                if w <= 0 or h <= 0:
                    bad.append(f"{lp.name}:{i} degenerate box")
                if cx - w / 2 < -1e-3 or cx + w / 2 > 1 + 1e-3:
                    bad.append(f"{lp.name}:{i} box exits frame horizontally")
        assert not bad, f"{len(bad)} invalid labels, first 5: {bad[:5]}"

    @requires_data
    def test_splits_are_disjoint(self):
        sets = {s: {p.name for p in (DATA / s / "images").glob("*.jpg")}
                for s in ("train", "val", "test")}
        assert not sets["train"] & sets["val"]
        assert not sets["train"] & sets["test"]
        assert not sets["val"] & sets["test"]

    @requires_data
    def test_splits_are_survey_disjoint_no_leakage(self):
        """The whole point of the split: no acquisition YEAR may appear in two
        splits, or adjacent frames of one survey would leak across the boundary."""
        def years(s):
            return {p.stem.split("_")[-1] for p in (DATA / s / "images").glob("*.jpg")}
        tr, va, te = years("train"), years("val"), years("test")
        assert not tr & va and not tr & te and not va & te, \
            f"survey-year leakage: train={tr} val={va} test={te}"

    @requires_data
    def test_split_manifest_matches_reality(self):
        mp = ROOT / "data" / "splits" / "milco_nombo_survey_split.json"
        assert mp.exists()
        m = json.loads(mp.read_text())
        for s in ("train", "val", "test"):
            n = len(list((DATA / s / "images").glob("*.jpg")))
            assert m["splits"][s]["frames"] == n


class TestEndToEnd:
    @requires_model
    @requires_data
    def test_full_pipeline_produces_a_valid_report(self, tmp_path):
        sys.path.insert(0, str(ROOT / "src"))
        from aquashield.detection.detector import Detector
        from aquashield.detection.taxonomy import Taxonomy
        from aquashield.pipeline import AquaShieldPipeline, PipelineConfig
        from aquashield.reporting import (CSV_COLUMNS, build_report, write_csv,
                                          write_geojson, write_json)

        imgs = sorted((DATA / "test" / "images").glob("*.jpg"))[:6]
        frames = [(p.stem, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in imgs]
        pipe = AquaShieldPipeline(Detector(str(_weights()), conf=0.10),
                                  PipelineConfig(preprocess_profile="standard"),
                                  taxonomy=Taxonomy("milco_nombo"))
        res = pipe.process_survey(frames, survey_id="TEST-SURVEY")

        assert res.summary["frames_processed"] == len(frames)
        assert len(res.frames) == len(frames)
        # accepted + rejected must account for every raw candidate
        for f in res.frames:
            assert len(f.accepted) + len(f.rejected) == len(f.raw_detections)

        rep = build_report(res.hazards, survey_id=res.survey_id,
                           summary=res.summary, provenance=res.provenance)
        j = json.loads(write_json(rep, tmp_path / "r.json").read_text())
        assert j["survey"]["survey_id"] == "TEST-SURVEY"
        assert "disclaimer" in j
        assert j["provenance"]["device"] in ("mps", "cpu", "cuda")

        c = write_csv(res.hazards, tmp_path / "r.csv")
        assert c.read_text().splitlines()[0].split(",") == CSV_COLUMNS
        write_geojson(res.hazards, tmp_path / "r.geojson")

    @requires_model
    @requires_data
    def test_no_geolocation_means_null_coordinates_never_zeros(self):
        """The single most dangerous failure mode: a fabricated position."""
        sys.path.insert(0, str(ROOT / "src"))
        from aquashield.detection.detector import Detector
        from aquashield.detection.taxonomy import Taxonomy
        from aquashield.pipeline import AquaShieldPipeline, PipelineConfig

        imgs = sorted((DATA / "test" / "images").glob("*.jpg"))[:8]
        frames = [(p.stem, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in imgs]
        pipe = AquaShieldPipeline(Detector(str(_weights()), conf=0.05),
                                  PipelineConfig(), taxonomy=Taxonomy("milco_nombo"))
        res = pipe.process_survey(frames, survey_id="NOGEO")   # no georef supplied
        for h in res.hazards:
            assert h.latitude is None and h.longitude is None
            assert h.geolocation_confidence == "UNAVAILABLE"
            assert any("Geolocation unavailable" in n for n in h.notes)

    @requires_model
    @requires_data
    def test_pipeline_is_deterministic(self):
        """A live demo must not change its answer between runs."""
        sys.path.insert(0, str(ROOT / "src"))
        from aquashield.detection.detector import Detector
        from aquashield.detection.taxonomy import Taxonomy
        from aquashield.pipeline import AquaShieldPipeline, PipelineConfig

        imgs = sorted((DATA / "test" / "images").glob("*.jpg"))[:4]
        frames = [(p.stem, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in imgs]
        det = Detector(str(_weights()), conf=0.10)
        pipe = AquaShieldPipeline(det, PipelineConfig(), taxonomy=Taxonomy("milco_nombo"))
        a = pipe.process_survey(frames, survey_id="D1", make_previews=False)
        b = pipe.process_survey(frames, survey_id="D1", make_previews=False)
        assert [h.hazard_id for h in a.hazards] == [h.hazard_id for h in b.hazards]
        assert [h.confidence_pct for h in a.hazards] == [h.confidence_pct for h in b.hazards]

    @requires_model
    def test_missing_model_raises_a_clear_error(self):
        sys.path.insert(0, str(ROOT / "src"))
        from aquashield.detection.detector import Detector
        with pytest.raises(FileNotFoundError, match="placeholder"):
            Detector("models/definitely_not_here.pt")


class TestDemoAssets:
    def test_demo_scenarios_declare_their_navigation_status(self):
        d = ROOT / "demo_data"
        if not d.exists():
            pytest.skip("demo data not built")
        for sc in sorted(p for p in d.glob("*") if p.is_dir()):
            meta = json.loads((sc / "scenario.json").read_text())
            assert "navigation" in meta and "license" in meta
            if (sc / "navigation.csv").exists():
                # synthetic navigation must be labelled, in the metadata AND the file
                assert meta.get("synthetic_navigation") is True
                assert "SYNTHETIC" in (sc / "navigation.csv").read_text()[:400].upper()


class TestDashboard:
    def test_dashboard_imports_without_running_streamlit(self):
        """UI smoke test: the module must at least parse and resolve its imports."""
        code = (
            "import ast,sys,pathlib;"
            "src=pathlib.Path('dashboard/app.py').read_text();"
            "ast.parse(src);"
            "print('parsed')"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "parsed" in r.stdout


class TestDashboardEndToEnd:
    """Drive the real dashboard headlessly, exactly as an operator would."""

    @requires_model
    def test_demo_mode_processes_and_offers_exports(self):
        pytest.importorskip("streamlit.testing.v1")
        from streamlit.testing.v1 import AppTest

        if not (ROOT / "demo_data").exists():
            pytest.skip("demo data not built")

        at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=300)
        at.run()
        assert not at.exception, at.exception
        assert "AQUA-SHIELD" in [t.value for t in at.title]

        sc = [s for s in at.selectbox if s.label == "Scenario"]
        assert sc, "no demo scenarios offered"
        geo = [o for o in sc[0].options if "04_georeferenced" in str(o)]
        if geo:
            sc[0].set_value(geo[0])
            at.run()

        btn = [b for b in at.button if "Process" in b.label]
        assert btn, "no process button"
        btn[0].click()
        at.run()
        assert not at.exception, at.exception

        labels = {m.label for m in at.metric}
        assert {"Frames", "Raw candidates", "Unique hazards"} <= labels
        dl = [d.label for d in at.get("download_button")]
        assert any("JSON" in d for d in dl) and any("CSV" in d for d in dl)


class TestOfflineGuarantee:
    """AQUA-SHIELD must run with no network access. These tests keep it that way."""

    def test_no_network_calls_in_inference_code(self):
        import re
        bad = re.compile(r"\b(requests\.(get|post)|urlopen|httpx\.|socket\.socket)\b")
        offenders = []
        for f in list((ROOT / "src").rglob("*.py")) + [ROOT / "dashboard" / "app.py"]:
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if bad.search(line):
                    offenders.append(f"{f.relative_to(ROOT)}:{i}")
        assert not offenders, f"network calls in the inference path: {offenders}"

    def test_no_cloud_ai_sdk_dependencies(self):
        req = (ROOT / "requirements.txt").read_text().lower()
        for pkg in ("openai", "anthropic", "google-generativeai", "cohere", "boto3"):
            assert pkg not in req, f"{pkg} must not be a dependency"

    def test_offline_map_flag_is_honoured(self, monkeypatch):
        """With AQS_OFFLINE_MAP=1 the dashboard must not request remote tiles."""
        pytest.importorskip("streamlit.testing.v1")
        if not (ROOT / "demo_data").exists() or _weights() is None:
            pytest.skip("demo data or model unavailable")
        monkeypatch.setenv("AQS_OFFLINE_MAP", "1")
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=300)
        at.run()
        sc = [s for s in at.selectbox if s.label == "Scenario"]
        geo = [o for o in sc[0].options if "04_georeferenced" in str(o)] if sc else []
        if geo:
            sc[0].set_value(geo[0])
            at.run()
        btn = [b for b in at.button if "Process" in b.label]
        btn[0].click()
        at.run()
        assert not at.exception, at.exception
        assert any("Offline map" in str(i.value) for i in at.info), \
            "offline mode did not announce itself"
