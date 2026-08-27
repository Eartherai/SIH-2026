"""QC, preprocessing and tiling."""
import numpy as np
import pytest

from aquashield.sonar import qc
from aquashield.sonar.preprocess import (PROFILES, PreprocessConfig, fix_dropouts,
                                         gain_normalize, lee_filter,
                                         normalize_dynamic_range, preprocess,
                                         remove_water_column, slant_range_correct)
from aquashield.sonar.tiling import plan_tiles, tile_image, to_global


class TestQC:
    def test_finds_the_water_column_we_planted(self, synthetic_waterfall):
        r = qc.assess(synthetic_waterfall)
        assert r.water_column_detected
        a, b = r.water_column_bounds
        assert a <= 190 and b >= 210, f"band {a}-{b} misses the planted 185-215"

    def test_no_water_column_on_uniform_or_noise(self):
        assert not qc.assess(np.full((256, 256), 0.4, np.float32)).water_column_detected
        rng = np.random.default_rng(0)
        assert not qc.assess(rng.random((256, 256)).astype(np.float32)).water_column_detected

    def test_detects_dead_ping_row(self, synthetic_waterfall):
        r = qc.assess(synthetic_waterfall)
        assert r.dropout_ratio > 0, "the planted all-zero row was not counted"
        assert r.dropout_ratio < 0.05

    def test_quality_score_is_bounded_and_ordered(self, synthetic_waterfall):
        good = qc.assess(synthetic_waterfall)
        flat = qc.assess(np.full((200, 200), 0.5, np.float32))
        assert 0.0 <= good.quality_score <= 1.0
        assert 0.0 <= flat.quality_score <= 1.0
        assert good.quality_score > flat.quality_score

    def test_report_serialises(self, synthetic_waterfall):
        d = qc.assess(synthetic_waterfall).as_dict()
        assert set(["quality_score", "dropout_ratio", "notes"]).issubset(d)


class TestPreprocess:
    def test_every_stage_is_switchable(self, synthetic_waterfall):
        off = preprocess(synthetic_waterfall, PROFILES["none"])
        assert off.steps_applied == []
        on = preprocess(synthetic_waterfall, PROFILES["aggressive"])
        assert len(on.steps_applied) >= 4

    def test_output_is_uint8_and_same_shape(self, synthetic_waterfall):
        r = preprocess(synthetic_waterfall, PROFILES["standard"])
        assert r.image.dtype == np.uint8
        assert r.image.shape == synthetic_waterfall.shape

    def test_lee_filter_reduces_speckle_but_keeps_the_edge(self, synthetic_waterfall):
        before = qc.speckle_index(synthetic_waterfall)
        after = qc.speckle_index(lee_filter(synthetic_waterfall, 5))
        assert after < before, "Lee filter did not reduce the speckle index"
        # the strong target/shadow boundary must survive
        f = lee_filter(synthetic_waterfall, 5)
        assert f[155, 305] - f[155, 330] > 0.3

    def test_dropout_repair_removes_the_dead_row(self, synthetic_waterfall):
        fixed, info = fix_dropouts(synthetic_waterfall)
        assert info["repaired"] >= 1
        assert fixed[300].std() > 0, "dead row was not repaired"

    def test_water_column_removal_brightens_the_band(self, synthetic_waterfall):
        out, info = remove_water_column(synthetic_waterfall, "inpaint")
        assert info["water_column_detected"]
        a, b = info["bounds"]
        assert out[:, a:b].mean() > synthetic_waterfall[:, a:b].mean()

    def test_gain_normalisation_flattens_range_response(self, synthetic_waterfall):
        before = synthetic_waterfall.mean(axis=0)
        after = gain_normalize(synthetic_waterfall).mean(axis=0)
        assert after.std() < before.std()

    def test_dynamic_range_normalisation_spans_the_range(self, synthetic_waterfall):
        out = normalize_dynamic_range(synthetic_waterfall)
        assert out.min() == pytest.approx(0.0, abs=1e-6)
        assert out.max() == pytest.approx(1.0, abs=1e-6)

    def test_slant_range_correction_is_skipped_without_altitude(self, synthetic_waterfall):
        cfg = PreprocessConfig(slant_range_correction=True, altitude_px=None)
        r = preprocess(synthetic_waterfall, cfg)
        assert any("SKIPPED" in s for s in r.steps_applied), \
            "must refuse to guess an altitude"

    def test_slant_range_correction_runs_with_altitude(self, synthetic_waterfall):
        out = slant_range_correct(synthetic_waterfall, 30.0)
        assert out.shape[0] == synthetic_waterfall.shape[0]
        assert np.isfinite(out).all()


class TestTiling:
    def test_small_image_is_not_tiled(self):
        tiles, plan = tile_image(np.zeros((300, 300), np.uint8), 640, 128)
        assert not plan.tiled and len(tiles) == 1

    @pytest.mark.parametrize("shape", [(2200, 900), (1500, 1500), (700, 4000), (641, 641)])
    def test_tiles_cover_every_pixel(self, shape):
        tiles, plan = tile_image(np.zeros(shape, np.uint8), 640, 128)
        cov = np.zeros(shape, bool)
        for t in tiles:
            cov[t.y0:t.y1, t.x0:t.x1] = True
        assert cov.all(), f"{shape}: {(~cov).sum()} uncovered pixels"

    def test_tiles_never_exceed_tile_size(self):
        tiles, _ = tile_image(np.zeros((2000, 1100), np.uint8), 640, 128)
        for t in tiles:
            assert t.image.shape[0] <= 640 and t.image.shape[1] <= 640

    def test_to_global_shifts_correctly(self):
        b = np.array([[10, 20, 30, 40]], np.float32)
        assert to_global(b, (100, 200)).tolist() == [[110, 220, 130, 240]]

    def test_to_global_handles_empty(self):
        assert to_global(np.zeros((0, 4), np.float32), (5, 5)).shape == (0, 4)

    def test_overlap_must_be_smaller_than_tile(self):
        with pytest.raises(ValueError):
            plan_tiles(1000, 1000, 128, 128)
