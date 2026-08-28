"""Box geometry, taxonomy, features, FP filter, calibration."""
import numpy as np
import pytest

from aquashield.confidence.calibration import (IdentityCalibrator, PlattCalibrator,
                                               band, reliability)
from aquashield.confidence.features import FEATURE_NAMES, extract
from aquashield.confidence.fp_filter import (INPUT_NAMES, LearnedFPFilter,
                                             RuleBasedFilter)
from aquashield.detection.boxes import (ios_matrix, iou_matrix,
                                        merge_tiled_detections, nms, xywhn_to_xyxy)
from aquashield.detection.taxonomy import Taxonomy


class TestBoxes:
    def test_iou_identity_and_disjoint(self):
        a = np.array([[0, 0, 10, 10]], np.float32)
        assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
        b = np.array([[100, 100, 110, 110]], np.float32)
        assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)

    def test_iou_half_overlap(self):
        a = np.array([[0, 0, 10, 10]], np.float32)
        b = np.array([[5, 0, 15, 10]], np.float32)
        assert iou_matrix(a, b)[0, 0] == pytest.approx(1 / 3, abs=1e-5)

    def test_ios_detects_containment_where_iou_fails(self):
        full = np.array([[0, 0, 100, 100]], np.float32)
        frag = np.array([[0, 0, 20, 100]], np.float32)     # seam-clipped fragment
        assert iou_matrix(full, frag)[0, 0] < 0.3
        assert ios_matrix(full, frag)[0, 0] == pytest.approx(1.0)

    def test_nms_removes_duplicates_keeps_distinct(self):
        b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
        keep = nms(b, np.array([0.9, 0.8, 0.7]), 0.5)
        assert len(keep) == 2 and 0 in keep and 2 in keep

    def test_merge_is_class_aware_by_default(self):
        b = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
        s = np.array([0.9, 0.8], np.float32)
        assert len(merge_tiled_detections(b, s, np.array([0, 1]))) == 2
        assert len(merge_tiled_detections(b, s, np.array([0, 0]))) == 1

    def test_merge_handles_empty(self):
        assert merge_tiled_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0)) == []

    def test_xywhn_roundtrip(self):
        out = xywhn_to_xyxy(np.array([[0.5, 0.5, 0.2, 0.4]]), 100, 200)
        assert out[0].tolist() == pytest.approx([40.0, 60.0, 60.0, 140.0])


class TestTaxonomy:
    def test_milco_is_man_made_nombo_is_ambiguous(self):
        t = Taxonomy("milco_nombo")
        assert t[0].level1 == "MAN_MADE" and t[0].level2 == "mine_like_object"
        # NOMBO means "not mine-like", NOT "natural" -- must never be a man-made subclass
        assert t[1].level1 == "AMBIGUOUS"
        assert t[1].level2 == "bottom_object_uncertain"

    def test_unknown_class_id_is_not_invented(self):
        t = Taxonomy("milco_nombo")
        e = t[99]
        assert e.level1 == "AMBIGUOUS" and e.level2 == "unknown_anomaly"
        assert "UNMAPPED" in e.native_name

    def test_unknown_source_raises(self):
        with pytest.raises(KeyError):
            Taxonomy("no_such_dataset")

    def test_license_and_citation_present(self):
        t = Taxonomy("milco_nombo")
        assert "CC BY 4.0" in t.license and t.citation


class TestFeatures:
    def test_all_features_finite_and_named(self, synthetic_waterfall):
        f = extract(synthetic_waterfall, [300, 150, 318, 165], nadir_col=200)
        v = f.vector()
        assert v.shape == (len(FEATURE_NAMES),)
        assert np.isfinite(v).all()

    def test_shadow_is_detected_next_to_the_planted_target(self, synthetic_waterfall):
        # target at cols 300-318 with a dark strip at 318-340 (down-range of nadir=200)
        f = extract(synthetic_waterfall, [300, 150, 318, 165], nadir_col=200)
        assert f.shadow_ratio > 0.3, f"planted shadow not seen (got {f.shadow_ratio})"
        assert f.shadow_side_consistent == 1.0, "shadow should be on the far-range side"

    def test_unknown_nadir_reports_side_as_unknown(self, synthetic_waterfall):
        f = extract(synthetic_waterfall, [300, 150, 318, 165], nadir_col=None)
        assert f.shadow_side_consistent == 0.5, "must not claim to know range direction"

    def test_degenerate_box_does_not_crash(self, synthetic_waterfall):
        assert np.isfinite(extract(synthetic_waterfall, [0, 0, 1, 1]).vector()).all()
        assert np.isfinite(extract(synthetic_waterfall, [-50, -50, 5, 5]).vector()).all()


class TestFPFilter:
    def _synthetic(self, n=500, seed=3):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.3).astype(float)
        X = rng.normal(0, 1, (n, len(INPUT_NAMES)))
        X[:, INPUT_NAMES.index("local_snr")] += 2.2 * y
        return X, y

    def test_learns_the_injected_signal(self):
        X, y = self._synthetic()
        f = LearnedFPFilter().fit(X, y)
        assert f.fitted
        top = max(f.as_dict()["weights"].items(), key=lambda kv: abs(kv[1]))[0]
        assert top == "local_snr"
        assert ((f.proba(X) >= 0.5).astype(float) == y).mean() > 0.75

    def test_refuses_to_fit_on_too_little_data(self):
        X, y = self._synthetic(n=10)
        f = LearnedFPFilter().fit(X, y)
        assert not f.fitted and "error" in f.meta

    def test_refuses_to_fit_single_class(self):
        X, y = self._synthetic()
        f = LearnedFPFilter().fit(X, np.zeros(len(y)))
        assert not f.fitted

    def test_load_missing_file_falls_back_to_rules(self, tmp_path):
        f = LearnedFPFilter.load(tmp_path / "nope.json")
        assert isinstance(f, RuleBasedFilter) and not f.fitted

    def test_roundtrip_save_load_preserves_predictions(self, tmp_path):
        X, y = self._synthetic()
        f = LearnedFPFilter().fit(X, y)
        p = tmp_path / "f.json"
        f.save(p)
        g = LearnedFPFilter.load(p)
        assert np.allclose(f.proba(X), g.proba(X), atol=1e-8), \
            'saved model must reproduce its predictions'

    def test_rule_filter_explains_every_rejection(self):
        v = RuleBasedFilter().predict(np.zeros((1, len(FEATURE_NAMES))), np.array([0.01]))
        assert not v[0].accepted and v[0].reason
        assert v[0].detail["rules_fired"]

    def test_explanations_reference_real_features(self):
        X, y = self._synthetic()
        f = LearnedFPFilter().fit(X, y)
        top = f.top_contributions(X[0], k=3)
        assert all(n in INPUT_NAMES for n, _ in top)


class TestCalibration:
    def test_improves_expected_calibration_error(self):
        rng = np.random.default_rng(7)
        raw = rng.beta(5, 2, 3000)
        y = (rng.random(3000) < raw ** 3).astype(float)
        before = reliability(raw, y).ece
        cal = PlattCalibrator().fit(raw, y)
        after = reliability(cal.transform(raw), y).ece
        assert cal.fitted and after < before / 2

    def test_identity_declares_itself_uncalibrated(self):
        c = IdentityCalibrator()
        assert c.as_dict()["fitted"] is False
        assert np.allclose(c.transform(np.array([0.3, 0.7])), [0.3, 0.7])

    def test_missing_file_returns_identity(self, tmp_path):
        assert isinstance(PlattCalibrator.load(tmp_path / "nope.json"), IdentityCalibrator)

    def test_refuses_single_class_fit(self):
        assert not PlattCalibrator().fit(np.linspace(0.1, 0.9, 50), np.ones(50)).fitted

    @pytest.mark.parametrize("pct,expected",
                             [(95, "CRITICAL"), (70, "HIGH"), (50, "MEDIUM"), (10, "LOW")])
    def test_bands(self, pct, expected):
        assert band(pct) == expected


class TestModelMeta:
    """The preprocessing profile is a property of the CHECKPOINT.

    Applying a chain the detector was never trained on shifted F1 from 0.144 to
    0.012 on the held-out surveys. These tests keep that failure from returning
    silently.
    """

    def test_missing_sidecar_assumes_raw_and_says_so(self, tmp_path):
        from aquashield.detection.model_meta import read_meta
        m = read_meta(tmp_path / "nope.pt")
        assert m["preprocess_profile"] == "none"
        assert m["_assumed"] is True
        assert "mismatch" in m["_note"]

    def test_roundtrip(self, tmp_path):
        from aquashield.detection.model_meta import (preprocess_profile_for_model,
                                                     read_meta, write_meta)
        w = tmp_path / "m.pt"
        w.write_bytes(b"x")
        write_meta(w, preprocess_profile="standard", experiment_id="E99")
        assert preprocess_profile_for_model(w) == "standard"
        assert read_meta(w).get("_assumed") is None
        assert read_meta(w)["experiment_id"] == "E99"

    def test_corrupt_sidecar_falls_back_safely(self, tmp_path):
        from aquashield.detection.model_meta import read_meta
        w = tmp_path / "m.pt"
        w.write_bytes(b"x")
        (tmp_path / "m.meta.json").write_text("{not json")
        assert read_meta(w)["preprocess_profile"] == "none"

    def test_pipeline_default_profile_is_none(self):
        """Guards against someone 'helpfully' restoring a preprocessing default."""
        from aquashield.pipeline import PipelineConfig
        assert PipelineConfig().preprocess_profile == "none"


class TestCrabPotSplitLogic:
    """The crab-pot recording-level split must not leak, same discipline as
    the MILCO/NOMBO survey-year split. Testable without the (gated) dataset
    by exercising survey_key() directly against real observed filenames."""

    def test_survey_key_groups_same_recording(self):
        from aquashield.ingestion.jsonl_bbox import survey_key
        a = survey_key("Rec09_Sensor_Depth_wcp_ss_port_00001_jpg.rf.abc123.jpg")
        b = survey_key("Rec09_Sensor_Depth_wcp_ss_star_00042_png_jpg.rf.def456.jpg")
        c = survey_key("Rec14_Sensor_Depth_wcp_ss_port_00001_jpg.rf.ghi789.jpg")
        assert a == b == "Rec09"
        assert c == "Rec14"
        assert a != c

    def test_prepare_script_does_not_demand_a_token_when_data_already_exists(self, tmp_path, monkeypatch):
        """Regression test: an earlier version of prepare_crab_pot.py checked for
        HF_TOKEN unconditionally, even when the dataset was already downloaded --
        forcing an unnecessary credential requirement on every re-run. The token
        should only be required on the code path that actually needs to call the
        HuggingFace API."""
        import subprocess
        import sys
        from pathlib import Path

        raw = tmp_path / "raw"
        (raw / "train").mkdir(parents=True)
        (raw / "train" / "metadata.jsonl").write_text(
            '{"file_name": "a.jpg", "objects": {"bbox": [], "category": []}}\n')
        # no valid/test metadata -- fine, the loop just skips missing splits

        monkeypatch.delenv("HF_TOKEN", raising=False)
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "prepare_crab_pot.py"),
             "--raw", str(raw), "--out", str(tmp_path / "out")],
            cwd=root, capture_output=True, text=True, timeout=30)
        assert "Set HF_TOKEN" not in result.stdout + result.stderr, (
            "prepare_crab_pot.py demanded a token even though the data was "
            f"already present.\nstdout: {result.stdout}\nstderr: {result.stderr}")
