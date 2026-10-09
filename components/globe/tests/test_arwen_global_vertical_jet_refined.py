"""The jet-refined vertical grid: the 40-level surface_stretched stack with
its 120 to 400 hPa band re-laid at 23.4 hPa per layer (48 levels).

Floor of 2026-09-04 (upper-air scorecard, GDAS 2026-09-01 00Z on T255):
the 40-level stack's 44 to 60 hPa layers across the jet cost 1.85 m/s
rmsve and -0.53 m/s of speed at 250 hPa before any forecast.  The
candidate keeps every half level of the base stack outside the span
118.73 to 504.19 hPa bit for bit and lays twelve 23.44 hPa layers from
118.73 to 400 hPa with a three-layer geometric taper to the kept 56 hPa
layer below.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from woof.globe.checkpoint import read_checkpoint
from woof.globe.config import (
    VERTICAL_COORDINATES,
    VERTICAL_DEFAULT_NLEV,
    load_config,
)
from woof.globe.runner import run, vertical_grid_receipt
from woof.globe.vertical import (
    JET_BAND_REPORT_PA,
    JET_REFINED_BAND_MAXIMUM_LAYER_PA,
    HybridCoordinate,
    standard_atmosphere_height_m,
)

SMOKE_CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
T255_CONFIG = str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml")
#: The door a user reaches the candidate through: the T255 native verify
#: config with its vertical table changed and nothing else.
JET48_CONFIG = str(_shipped_configs() / "arwen_global_gdas_t255_jet48_24h.toml")
PS_REF = 101_325.0

#: The layout's own pin: SHA-256 of the 49 A then the 49 B half-level
#: values (float64, top to bottom).  A change to the constructor that
#: moves any half level moves this digest and every checkpoint, receipt
#: and export hashed under the 48-level candidate stops resuming.
JET_REFINED_48_AB_SHA256 = (
    "7d3fe9b742413029d5cab5dab90f9f87c36d4a12d53e9176c9f4116de8befb09"
)
#: The T255 native verify config under the candidate (coordinate =
#: "jet_refined", nlev = 48, everything else the shipped config's): the
#: config hash a candidate arm's checkpoints carry.
#: Derived from the T255 config of record's text, so it moves when that
#: text moves: 2026-09-06, the record went bare (the shipped default core,
#: the semi-Lagrangian one at 300 s) and this hash moved from
#: d25bef26c82c846b397875b9c81ac4574b1b8d4d27ac2632f6dd7bca37828894 to
#: ceab416825c7f0653a9cc0d510d6b741883b73ca44412ee67c24dffb8be6a75e;
#: 2026-09-07, the record stopped writing the drain and reads the core's
#: own order 16 at 720 s, and it moved again.
JET_REFINED_T255_CONFIG_HASH = (
    "001b9a2802f37d084aa4a1f9b4d21d052f30fd0d87848d95f341b87291903a83"
)


def _p_half(coordinate, ps=PS_REF):
    return coordinate.a_half_pa + coordinate.b_half * ps


def _ab_digest(coordinate) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(coordinate.a_half_pa, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(coordinate.b_half, dtype=np.float64).tobytes())
    return digest.hexdigest()


def test_the_candidate_is_48_levels_with_the_base_stack_kept_outside_the_span():
    base = HybridCoordinate.surface_stretched(40, 100.0)
    cand = HybridCoordinate.jet_refined(48, 100.0)
    assert cand.nlev == 48
    p_base = _p_half(base)
    p_cand = _p_half(cand)
    # The kept span: 14 half levels from the top (100 Pa to 118.73 hPa)
    # and 21 from the bottom (504.19 hPa to the surface), A and B bit for
    # bit, so the stratosphere, the boundary layer and the first full
    # level (23 m) are the base stack's own.
    assert np.array_equal(cand.a_half_pa[:14], base.a_half_pa[:14])
    assert np.array_equal(cand.b_half[:14], base.b_half[:14])
    assert np.array_equal(cand.a_half_pa[-21:], base.a_half_pa[-21:])
    assert np.array_equal(cand.b_half[-21:], base.b_half[-21:])
    assert p_cand[13] == pytest.approx(11_873.0, abs=1.0)
    assert p_cand[-21] == pytest.approx(50_419.0, abs=1.0)
    assert p_base[13] == p_cand[13] and p_base[-21] == p_cand[-21]
    z1 = standard_atmosphere_height_m(np.sqrt(p_cand[-1] * p_cand[-2])) - standard_atmosphere_height_m(PS_REF)
    assert 15.0 <= z1 <= 35.0
    # Class invariants, explicitly.
    assert np.all(cand.a_half_pa >= 0.0)
    assert np.all((cand.b_half >= 0.0) & (cand.b_half <= 1.0))
    assert cand.a_half_pa[-1] == 0.0 and cand.b_half[-1] == 1.0
    assert cand.a_half_pa[0] == 100.0 and cand.b_half[0] == 0.0
    for ps in (50_000.0, 60_000.0, 80_000.0, PS_REF, 120_000.0):
        assert np.all(np.diff(_p_half(cand, ps)) > 0.0), ps
    assert _ab_digest(cand) == JET_REFINED_48_AB_SHA256


def test_the_band_holds_twelve_layers_of_at_most_25_hpa_and_the_taper_is_smooth():
    cand = HybridCoordinate.jet_refined(48, 100.0)
    p = _p_half(cand)
    dp = np.diff(p)
    band = dp[13:25]
    assert band.size == 12
    np.testing.assert_allclose(band, (40_000.0 - p[13]) / 12.0, rtol=1.0e-12)
    assert np.all(band <= JET_REFINED_BAND_MAXIMUM_LAYER_PA)
    assert p[25] == 40_000.0
    # Taper: three layers growing geometrically from the band thickness to
    # the kept 56.1 hPa layer; no adjacent step above 1.45 anywhere in the
    # stack and none at the 2.5x the plain insertion would have made.
    taper = dp[25:28]
    assert taper.size == 3 and np.all(np.diff(taper) > 0.0)
    ratios = dp[1:] / dp[:-1]
    assert np.all(ratios <= 1.45), ratios.max()
    assert np.all(ratios >= 0.64), ratios.min()
    assert ratios[24] == pytest.approx(ratios[25], rel=1.0e-9)  # geometric taper
    assert ratios[25] == pytest.approx(ratios[26], rel=1.0e-9)
    assert 1.2 < ratios[24] < 1.22 and 1.3 < ratios[27] < 1.4
    # Full levels: twelve in the reported jet band, on the pressures the
    # docstring states.
    p_full = np.sqrt(p[:-1] * p[1:])
    inside = (p_full >= JET_BAND_REPORT_PA[0]) & (p_full <= JET_BAND_REPORT_PA[1])
    assert int(np.count_nonzero(inside)) == 12
    expected = [129.92, 153.44, 176.94, 200.42, 223.90, 247.37, 270.83, 294.29, 317.75, 341.20, 364.65, 388.10]
    np.testing.assert_allclose(p_full[inside] / 100.0, expected, atol=0.006)


def test_the_hybrid_law_of_the_new_half_levels_is_the_base_stacks_own():
    cand = HybridCoordinate.jet_refined(48, 100.0)
    p = _p_half(cand)
    x = np.clip((p - 10_000.0) / (PS_REF - 10_000.0), 0.0, 1.0)
    np.testing.assert_allclose(cand.b_half[14:28], x[14:28] ** 1.2, rtol=1.0e-12)
    np.testing.assert_allclose(cand.a_half_pa[14:28], p[14:28] - cand.b_half[14:28] * PS_REF, atol=1.0e-9)
    # Over a 500 hPa surface the band layers keep at least 10 hPa.
    dp_plateau = np.diff(_p_half(cand, 50_000.0))
    assert np.all(dp_plateau[13:25] > 1_000.0)


def test_describe_reports_the_jet_band_for_both_stacks():
    base = HybridCoordinate.surface_stretched(40, 100.0).describe()
    cand = HybridCoordinate.jet_refined(48, 100.0).describe()
    assert base["jet_band_pa"] == [12_000.0, 40_000.0] == cand["jet_band_pa"]
    assert base["jet_band_thickest_layer_pa"] == pytest.approx(5_981.8, abs=0.1)
    assert cand["jet_band_thickest_layer_pa"] == pytest.approx(2_343.9, abs=0.1)
    assert base["full_levels_in_jet_band"] == 5
    assert cand["full_levels_in_jet_band"] == 12
    assert cand["nlev"] == 48 and len(cand["layers"]) == 48
    assert cand["first_full_level_height_m"] == pytest.approx(base["first_full_level_height_m"])
    assert cand["bottom_layer_thickness_pa"] == base["bottom_layer_thickness_pa"]
    assert cand["top_layer_ln_ratio"] == base["top_layer_ln_ratio"]
    assert cand["full_levels_above_50hpa"] == base["full_levels_above_50hpa"]
    assert cand["pure_pressure_levels"] == base["pure_pressure_levels"]
    heights = [layer["z_full_m_agl"] for layer in cand["layers"]]
    assert heights == sorted(heights, reverse=True)


def test_a_count_whose_band_is_coarser_than_25_hpa_is_refused_naming_48():
    for count in (46, 47):
        with pytest.raises(ValueError, match="nlev = 48 is the smallest count"):
            HybridCoordinate.jet_refined(count, 100.0)
    # More levels are accepted and thin the band further.
    for count, layers in ((49, 13), (56, 20)):
        grid = HybridCoordinate.jet_refined(count, 100.0)
        assert grid.nlev == count
        assert grid.describe()["full_levels_in_jet_band"] == layers
        assert grid.describe()["jet_band_thickest_layer_pa"] < 2_343.95
    with pytest.raises(ValueError, match="band_top_pa < band_bottom_pa"):
        HybridCoordinate.jet_refined(48, 100.0, band_bottom_pa=60_000.0)
    with pytest.raises(ValueError, match="at least one taper layer"):
        HybridCoordinate.jet_refined(48, 100.0, taper_layers=0)


def _vertical_block(text: str, block: str) -> str:
    assert 'coordinate = "surface_stretched"\nnlev = 40\n' in text
    return text.replace('coordinate = "surface_stretched"\nnlev = 40\n', block)


def test_the_candidate_is_selectable_by_name_with_its_own_identity(tmp_path):
    assert "jet_refined" in VERTICAL_COORDINATES
    assert VERTICAL_DEFAULT_NLEV["jet_refined"] == 48
    base_text = Path(T255_CONFIG).read_text(encoding="utf-8")
    named = tmp_path / "named.toml"
    named.write_text(_vertical_block(base_text, 'coordinate = "jet_refined"\n'), encoding="utf-8")
    counted = tmp_path / "counted.toml"
    counted.write_text(_vertical_block(base_text, 'coordinate = "jet_refined"\nnlev = 48\n'), encoding="utf-8")
    forty = load_config(T255_CONFIG)
    cfg = load_config(named)
    assert cfg.vertical_coordinate == "jet_refined"
    assert cfg.vertical.nlev == 48
    assert load_config(counted).config_hash == cfg.config_hash
    assert cfg.config_hash != forty.config_hash
    assert "vertical_coordinate" not in cfg.config_identity
    assert _ab_digest(cfg.vertical) == JET_REFINED_48_AB_SHA256
    # The shipped config keeps its 40-level identity: selectable, not the default.
    assert forty.vertical_coordinate == "surface_stretched" and forty.vertical.nlev == 40
    if not JET_REFINED_T255_CONFIG_HASH.startswith("REPLACED"):
        assert cfg.config_hash == JET_REFINED_T255_CONFIG_HASH
    # 47 under the name is refused through the loader too.
    coarse = tmp_path / "coarse.toml"
    coarse.write_text(_vertical_block(base_text, 'coordinate = "jet_refined"\nnlev = 47\n'), encoding="utf-8")
    with pytest.raises(ValueError, match="nlev = 48 is the smallest count"):
        load_config(coarse)


def test_a_48_level_run_checkpoints_restarts_bit_for_bit_and_states_its_grid(tmp_path):
    base = Path(SMOKE_CONFIG).read_text(encoding="utf-8")
    assert 'coordinate = "pressure_blend"\nnlev = 4\n' in base
    path = tmp_path / "jet.toml"
    path.write_text(
        base.replace('coordinate = "pressure_blend"\nnlev = 4\n', 'coordinate = "jet_refined"\n'),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.vertical.nlev == 48
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    receipt = run(cfg, full)
    assert receipt["vertical"] == vertical_grid_receipt(cfg)
    assert receipt["vertical"]["coordinate"] == "jet_refined"
    assert receipt["vertical"]["nlev"] == 48
    assert receipt["vertical"]["jet_band_thickest_layer_pa"] == pytest.approx(2_343.9, abs=0.1)
    midpoint = full / "arwen_global_step00000002.npz"
    run(cfg, resumed, restart=midpoint)
    meta_a, arrays_a = read_checkpoint(full / "arwen_global_step00000004.npz")
    meta_b, arrays_b = read_checkpoint(resumed / "arwen_global_step00000004.npz")
    assert meta_a["config_hash"] == cfg.config_hash == meta_b["config_hash"]
    # The checkpoint carries the level count in every levelled array's shape.
    assert meta_a["arrays"]["atmosphere__theta"]["shape"][0] == 48
    assert meta_a["arrays"]["atmosphere__qc"]["shape"][0] == 48
    assert arrays_a["atmosphere__theta"].shape[0] == 48
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name
    # A 40-level config never resumes a 48-level archive: the identity differs.
    forty = load_config(
        _write(tmp_path / "forty.toml", base.replace('coordinate = "pressure_blend"\nnlev = 4\n', 'coordinate = "surface_stretched"\n'))
    )
    assert forty.config_hash != cfg.config_hash
    with pytest.raises(ValueError):
        read_checkpoint(midpoint, expected_config_hash=forty.config_hash)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_the_sizing_model_prices_the_candidate_from_its_own_measurement(tmp_path):
    from woof.globe import sizing

    base_text = Path(T255_CONFIG).read_text(encoding="utf-8")
    forty = load_config(T255_CONFIG)
    cand = load_config(_write(tmp_path / "cand.toml", _vertical_block(base_text, 'coordinate = "jet_refined"\n')))
    e40 = sizing.estimate_global_memory(forty)
    e48 = sizing.estimate_global_memory(cand)
    assert e40.nlev == 40 and e48.nlev == 48
    # The candidate's ten-step probe of record on the merged tip of
    # 2026-09-05 (10.13 GiB measured at the allocator; the level set's own
    # 24 h arm on its lane tree read 10.08) is a calibration row, so the
    # candidate at T255 sits inside the measured domain: no extrapolation
    # note, and the figure reads the measurement within the model's own
    # worst residual.
    assert e40.calibrated and e48.calibrated
    assert e48.extrapolation == ()
    assert 48 in sizing.CALIBRATED_DOMAIN["nlev"]
    rows = [p for p in sizing.DEVICE_PEAK_CALIBRATION if int(p["nlev"]) == 48]
    assert len(rows) == 1 and int(rows[0]["truncation"]) == 255
    assert e48.device_peak_bytes == pytest.approx(
        rows[0]["peak_used_bytes"], rel=sizing.worst_calibration_residual_fraction())
    # Every term but the Legendre tables and the fixed remainder scales
    # with the level count; the tables and the surface fields do not.
    assert e48.legendre_table_bytes == e40.legendre_table_bytes
    assert e48.surface_grid_bytes == e40.surface_grid_bytes
    assert e48.fixed_bytes == e40.fixed_bytes
    for name in ("resident_grid_bytes", "working_grid_bytes", "physics_column_bytes", "cumulus_column_bytes"):
        assert getattr(e48, name) == pytest.approx(getattr(e40, name) * 48 / 40, rel=1.0e-6), name
    assert e48.spectral_state_bytes > e40.spectral_state_bytes
    # T255: 8.57 GiB measured at 40 levels and 10.13 at 48, both on the
    # merged tip 565b60153; the THIRTEEN-row fit, which gained a measured
    # T383 rows on 2026-09-06, reads 8.62 and 10.23 for the Eulerian core
    # the rows were measured on.  The config of record now resolves to the
    # semi-Lagrangian core, which holds a second time level the sizer
    # prices (`trajectory_bytes`, six volumes and a plane: 0.265 GiB at 40
    # levels, 0.318 at 48), so the same fit reads 8.89 and 10.55; the
    # ten-step T255 gate on that core measured 8.805 GiB live at one band
    # (RTX 5070 Ti, 2026-09-07) against the 8.89.
    assert e40.device_peak_bytes / 1024 ** 3 == pytest.approx(8.89, abs=0.02)
    assert e48.device_peak_bytes / 1024 ** 3 == pytest.approx(10.55, abs=0.02)
    # T383 (the 35 km target) IS a measured truncation now, so the
    # candidate there is no longer extrapolated in truncation: only the
    # level count is interpolated, between the 40 and 48 the table holds
    # at T255.  It prices at 19.23 GiB at the 12,500-column radiation
    # chunk (18.51 for the Eulerian core; the semi-Lagrangian core's second
    # time level is 0.71 GiB here), still over the 16 GB card the target must fit RESIDENT --
    # which is what the host tier and the band count are for, and what
    # the sizer chooses there with no flag set.
    t383_text = _vertical_block(base_text.replace("truncation = 255", "truncation = 383"), 'coordinate = "jet_refined"\n')
    e383 = sizing.estimate_global_memory(load_config(_write(tmp_path / "t383.toml", t383_text)))
    assert e383.calibrated and e383.extrapolation == ()
    assert e383.device_peak_bytes / 1024 ** 3 == pytest.approx(19.23, abs=0.02)
    e383_small = sizing.estimate_global_memory(load_config(_write(
        tmp_path / "t383s.toml",
        t383_text.replace("radiation_column_chunk = 12500", "radiation_column_chunk = 5000"))))
    assert e383_small.device_peak_bytes / 1024 ** 3 == pytest.approx(17.39, abs=0.02)
    assert 16 * 1000 ** 3 < e383_small.device_peak_bytes < e383.device_peak_bytes


def test_the_verify_config_runs_the_candidate_and_differs_in_its_level_set_alone():
    """The shipped door to the level set is the 40-level verify config with
    the vertical table changed and nothing else, so an arm of each is a
    grade of the level set alone."""
    import dataclasses

    forty = load_config(T255_CONFIG)
    cand = load_config(JET48_CONFIG)
    assert forty.vertical_coordinate == "surface_stretched" and forty.vertical.nlev == 40
    assert cand.vertical_coordinate == "jet_refined" and cand.vertical.nlev == 48
    assert _ab_digest(cand.vertical) == JET_REFINED_48_AB_SHA256
    differing = {
        field.name for field in dataclasses.fields(forty)
        if getattr(forty, field.name) != getattr(cand, field.name)
    }
    assert differing == {"name", "a_half_pa", "b_half", "vertical_coordinate"}
    assert cand.config_hash != forty.config_hash
    assert cand.vertical.describe()["jet_band_thickest_layer_pa"] == pytest.approx(2_343.9, abs=0.1)
