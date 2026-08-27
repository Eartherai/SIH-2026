"""Geolocation, deduplication, priority, reports."""
import csv
import json
import math

import numpy as np
import pytest

from aquashield.geolocation import (NavigationReference, NoGeoReference,
                                    SonarGeometry, load_nav_csv)
from aquashield.reporting import (CSV_COLUMNS, HazardRecord, build_report,
                                  csv_string, score_priority, write_csv,
                                  write_geojson, write_json)
from aquashield.tracking.dedup import Observation, deduplicate, haversine_m


@pytest.fixture
def nav_csv(tmp_path):
    p = tmp_path / "nav.csv"
    with p.open("w", newline="") as f:
        f.write("# a leading comment banner, as survey exports often carry\n")
        w = csv.writer(f)
        w.writerow(["ping", "lat", "lon", "heading", "altitude"])
        for i in range(100):
            w.writerow([i, 12.9 + i * 9e-6, 80.3, 0.0, 10.0])
    return p


class TestNav:
    def test_skips_comment_banner(self, nav_csv):
        nav = load_nav_csv(nav_csv)
        assert len(nav) == 100 and nav.has_altitude

    def test_derives_heading_when_absent(self, tmp_path):
        p = tmp_path / "n.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lat", "lon"])
            for i in range(20):
                w.writerow([12.9 + i * 1e-5, 80.3])      # due north
        nav = load_nav_csv(p)
        assert nav.derived_heading
        assert nav.heading[5] == pytest.approx(0.0, abs=1.0)
        assert "heading" in nav.describe()["columns_missing"]

    def test_rejects_file_without_coordinates(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError):
            load_nav_csv(p)

    def test_heading_interpolation_wraps_around_north(self, tmp_path):
        p = tmp_path / "n.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lat", "lon", "heading"])
            w.writerow([12.9, 80.3, 350.0])
            w.writerow([12.91, 80.3, 10.0])
        nav = load_nav_csv(p)
        mid = nav.at_row(0.5, 2)[2]           # halfway between 350 and 10 -> 0
        assert min(abs(mid - 0.0), abs(mid - 360.0)) < 1.0


class TestGeolocation:
    def test_case_c_refuses_to_invent_coordinates(self):
        g = NoGeoReference()
        assert g.locate(10, 10) is None
        assert g.describe()["available"] is False

    def test_across_track_offset_has_the_right_sign(self, nav_csv):
        nav = load_nav_csv(nav_csv)
        ref = NavigationReference(nav, (500, 400), SonarGeometry(max_range_m=50.0))
        stbd = ref.locate(390, 250)     # heading 0 (north) -> starboard is EAST
        port = ref.locate(10, 250)
        assert stbd.detail["side"] == "starboard" and port.detail["side"] == "port"
        assert stbd.longitude > port.longitude

    def test_ground_range_is_less_than_slant_range(self, nav_csv):
        nav = load_nav_csv(nav_csv)
        ref = NavigationReference(nav, (500, 400), SonarGeometry(max_range_m=50.0))
        f = ref.locate(390, 250)
        assert f.detail["ground_range_m"] < f.detail["slant_range_m"]

    def test_uncertainty_grows_with_worse_gps(self, nav_csv):
        nav = load_nav_csv(nav_csv)
        a = NavigationReference(nav, (500, 400),
                                SonarGeometry(max_range_m=50.0, gps_accuracy_m=2.0))
        b = NavigationReference(nav, (500, 400),
                                SonarGeometry(max_range_m=50.0, gps_accuracy_m=25.0))
        assert b.locate(390, 250).uncertainty_m > a.locate(390, 250).uncertainty_m

    def test_error_budget_is_reported_in_full(self, nav_csv):
        nav = load_nav_csv(nav_csv)
        ref = NavigationReference(nav, (500, 400),
                                  SonarGeometry(max_range_m=50.0, layback_uncertainty_m=4.0))
        b = ref.locate(390, 250).detail["error_budget_m"]
        assert set(b) == {"gps_m", "heading_m", "layback_m", "altitude_m",
                          "range_resolution_m"}

    def test_nadir_is_flagged_as_ill_conditioned(self, nav_csv):
        """Near nadir, slant->ground inversion blows up. It must NOT be reported
        as a confident fix."""
        nav = load_nav_csv(nav_csv)
        ref = NavigationReference(nav, (500, 400),
                                  SonarGeometry(max_range_m=50.0, altitude_m=10.0))
        centre = ref.locate(200, 250)
        far = ref.locate(395, 250)
        assert centre.uncertainty_m > far.uncertainty_m


class TestDedup:
    def _obs(self, i, lat=None, lon=None, box=(100, 100, 140, 140), cls=0):
        return Observation(f"o{i}", f"f{i}", i, box, cls, "MILCO", 0.7,
                           latitude=lat, longitude=lon,
                           geoloc_uncertainty_m=(8.0 if lat else None))

    def test_geographic_merges_nearby_and_splits_far(self):
        obs = [self._obs(i, 12.97 + i * 1e-5, 80.25) for i in range(4)]
        obs.append(self._obs(9, 12.99, 80.27))
        hz = deduplicate(obs)
        assert len(hz) == 2
        assert max(h.observation_count for h in hz) == 4

    def test_sequence_mode_when_no_coordinates(self):
        hz = deduplicate([self._obs(i) for i in range(3)])
        assert len(hz) == 1 and hz[0].association_mode == "sequence"

    def test_sequence_respects_max_frame_gap(self):
        obs = [self._obs(0), self._obs(50)]
        assert len(deduplicate(obs, max_frame_gap=3)) == 2

    def test_class_aware_by_default(self):
        obs = [self._obs(0, 12.97, 80.25), self._obs(1, 12.97, 80.25, cls=1)]
        assert len(deduplicate(obs)) == 2
        assert len(deduplicate(obs, class_aware=False)) == 1

    def test_averaging_reduces_positional_uncertainty(self):
        obs = [self._obs(i, 12.97 + i * 1e-6, 80.25) for i in range(4)]
        hz = deduplicate(obs)[0]
        assert hz.geoloc_uncertainty_m < 8.0

    def test_hazard_ids_are_stable_across_runs(self):
        obs = [self._obs(i, 12.97 + i * 1e-5, 80.25) for i in range(3)]
        a = [h.hazard_id for h in deduplicate(obs)]
        b = [h.hazard_id for h in deduplicate(list(reversed(obs)))]
        assert a == b

    def test_empty_input(self):
        assert deduplicate([]) == []

    def test_haversine_matches_known_distance(self):
        assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=0.01)


class TestPriority:
    def test_large_ghost_net_outranks_tiny_unlocatable_object(self):
        net = score_priority(confidence_pct=55, level2_class="ghost_fishing_gear",
                             estimated_length_m=12.0, observation_count=9,
                             survey_quality=0.9, geolocated=True,
                             geoloc_uncertainty_m=6)
        blob = score_priority(confidence_pct=95, level2_class="other_man_made",
                              estimated_length_m=0.3, observation_count=1,
                              survey_quality=0.9, geolocated=False,
                              geoloc_uncertainty_m=None)
        assert net.score > blob.score

    def test_unknown_size_scores_neutral_not_zero(self):
        r = score_priority(confidence_pct=50, level2_class="other_man_made",
                           estimated_length_m=None, observation_count=1,
                           survey_quality=1.0, geolocated=True, geoloc_uncertainty_m=5)
        assert r.components["size"] == 0.5

    def test_score_bounded_and_components_reported(self):
        r = score_priority(confidence_pct=100, level2_class="ghost_fishing_gear",
                           estimated_length_m=50.0, observation_count=100,
                           survey_quality=1.0, geolocated=True, geoloc_uncertainty_m=1)
        assert 0 <= r.score <= 100
        assert set(r.components) >= {"confidence", "hazard_class", "size",
                                     "persistence", "actionability"}

    def test_poor_quality_damps_but_does_not_zero(self):
        kw = dict(confidence_pct=80, level2_class="ghost_fishing_gear",
                  estimated_length_m=5.0, observation_count=3, geolocated=True,
                  geoloc_uncertainty_m=5)
        good = score_priority(survey_quality=1.0, **kw).score
        bad = score_priority(survey_quality=0.0, **kw).score
        assert bad == pytest.approx(good * 0.5, rel=1e-6)


class TestReports:
    def _h(self, hid="AQS-00001", lat=12.97, lon=80.25):
        return HazardRecord(hazard_id=hid, survey_id="S1", detector_class="MILCO",
                            level1="MAN_MADE", level2="mine_like_object",
                            raw_detector_score=0.7, confidence_pct=68.0,
                            confidence_band="HIGH", calibrated=True,
                            priority_score=64.0, priority_band="HIGH",
                            bbox_x0=1, bbox_y0=2, bbox_x1=30, bbox_y1=40,
                            latitude=lat, longitude=lon,
                            geoloc_uncertainty_m=(6.0 if lat else None),
                            evidence={"model": 0.7, "shadow": 0.3})

    def test_csv_has_every_declared_column(self, tmp_path):
        p = write_csv([self._h()], tmp_path / "r.csv")
        header = p.read_text().splitlines()[0].split(",")
        assert header == CSV_COLUMNS

    def test_csv_string_matches_file(self, tmp_path):
        """The dashboard download button must emit exactly what write_csv writes.
        Read with newline='' so CRLF is preserved and the comparison is faithful."""
        h = [self._h()]
        p = write_csv(h, tmp_path / "r.csv")
        with open(p, newline="") as f:
            on_disk = f.read()
        assert csv_string(h) == on_disk

    def test_json_carries_provenance_and_disclaimer(self, tmp_path):
        rep = build_report([self._h()], survey_id="S1", summary={"frames_processed": 1},
                           provenance={"model_path": "x.pt"})
        d = json.loads(write_json(rep, tmp_path / "r.json").read_text())
        assert d["provenance"]["model_path"] == "x.pt"
        assert "not be read as probabilities" in d["disclaimer"]

    def test_geojson_omits_unlocated_and_says_so(self, tmp_path):
        p = write_geojson([self._h(), self._h("AQS-2", None, None)], tmp_path / "r.geojson")
        d = json.loads(p.read_text())
        assert len(d["features"]) == 1
        assert "1 hazard(s) omitted" in d["aqua_shield_note"]

    def test_geojson_coordinates_are_lon_lat_order(self, tmp_path):
        d = json.loads(write_geojson([self._h()], tmp_path / "r.geojson").read_text())
        lon, lat = d["features"][0]["geometry"]["coordinates"]
        assert lon == pytest.approx(80.25) and lat == pytest.approx(12.97)
