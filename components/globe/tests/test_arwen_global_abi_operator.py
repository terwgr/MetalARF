"""The ABI operator lane: its arithmetic, its refusals and its sidecar
contract, without SimSat, without a GOES granule and without the card.

What is proven here: the scorecard reads a planted bias and a planted
linear distortion both ways from the colocation moments the Rust
`rw_goes colocate` emits; the gate says INCOMPLETE for a class that
measured nothing, PASS inside the bar and FAIL outside it; the read-back
judge accepts a planted change of the right sign only and demands exact
zero from an identity; the level slice selects the upper troposphere; the
simulated-plane sidecar round-trips and refuses a render that is not on
the exact ABI lattice; the lattice tiles are on the visible disk; and the
door refuses a band without a radiance file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import types

import numpy as np
import pytest

from woof.globe import abi_operator as op


def _moments(obs: np.ndarray, sim: np.ndarray) -> dict:
    """The moment block exactly as `rw_goes colocate` writes it."""
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    n = int(obs.size)
    d = sim - obs
    m = {
        "n": n, "sum_obs": float(obs.sum()), "sum_sim": float(sim.sum()),
        "sum_obs_sq": float((obs * obs).sum()), "sum_sim_sq": float((sim * sim).sum()),
        "sum_obs_sim": float((obs * sim).sum()), "min_diff": float(d.min()), "max_diff": float(d.max()),
    }
    so = obs.std()
    ss = sim.std()
    r = float(np.corrcoef(obs, sim)[0, 1]) if so > 0 and ss > 0 else float("nan")
    fit = None
    if ss > 0 and n >= 3:
        slope = float(np.cov(obs, sim, bias=True)[0, 1] / (ss * ss))
        intercept = float(obs.mean() - slope * sim.mean())
        fit = (intercept, slope, float(math.sqrt(max(so * so * (1 - r * r), 0.0))))
    return {
        "n": n, "mean_obs_k": float(obs.mean()), "mean_sim_k": float(sim.mean()),
        "bias_k": float(d.mean()), "rmse_k": float(math.sqrt((d * d).mean())),
        "std_obs_k": float(so), "std_sim_k": float(ss), "correlation": r,
        "linear_fit_intercept_k": fit[0] if fit else None,
        "linear_fit_slope": fit[1] if fit else None,
        "rmse_after_linear_k": fit[2] if fit else None,
        "min_diff_k": float(d.min()), "max_diff_k": float(d.max()),
        "moments": m, "diff_histogram_1k": [0] * 120,
    }


def _colocation(classes: dict, band: int = 13) -> dict:
    return {
        "schema": "gpuwm-da.abi-colocation.v1", "status": "READY", "band": band, "satellite": "G19",
        "scan_start": "2026-09-01T18:00:20.300Z", "scan_end": "2026-09-01T18:09:52.300Z",
        "has_clear_sky_mask": True, "block_pixels": 24, "block_table": "blocks.csv",
        "counts": {"pairs": sum(c["n"] for c in classes.values())},
        "classes": classes,
    }


def test_scorecard_reads_a_planted_bias_and_a_planted_distortion_both_ways():
    rng = np.random.default_rng(7)
    obs = 260.0 + 30.0 * rng.random(5000)
    # A pure +3 K offset: bias +3, rmse 3, slope one, nothing left after the fit.
    warm = _colocation({"both_clear/le60": _moments(obs, obs + 3.0), "all/all": _moments(obs, obs + 3.0)})
    score = op.score_band(warm)
    gate = score["classes"]["both_clear/le60"]
    assert gate["bias_k"] == pytest.approx(3.0)
    assert gate["rmse_k"] == pytest.approx(3.0)
    assert gate["fit_slope"] == pytest.approx(1.0, abs=1e-9)
    assert gate["fit_intercept_k"] == pytest.approx(-3.0, abs=1e-9)
    assert gate["rmse_after_k"] == pytest.approx(0.0, abs=1e-9)
    assert gate["rmse_debiased_k"] == pytest.approx(0.0, abs=1e-9)
    assert score["gate"]["verdict"] == "PASS"
    # The other sign.
    cold = _colocation({"both_clear/le60": _moments(obs, obs - 3.0)})
    assert op.score_band(cold)["classes"]["both_clear/le60"]["bias_k"] == pytest.approx(-3.0)
    # A distortion the fit removes: sim = 1.25 obs - 60 (contrast overstated) plus
    # 1 K of noise.  Debiasing alone leaves the contrast error; the fit removes
    # it to the noise.
    noise = rng.normal(0.0, 1.0, obs.size)
    sim = 1.25 * obs - 60.0 + noise
    distorted = _colocation({"both_clear/le60": _moments(obs, sim)})
    gate = op.score_band(distorted)["classes"]["both_clear/le60"]
    assert gate["fit_slope"] == pytest.approx(0.8, abs=0.02)
    assert gate["rmse_after_k"] == pytest.approx(0.8, abs=0.1)
    assert gate["rmse_debiased_k"] > 2.0 * gate["rmse_after_k"]


def test_gate_is_incomplete_without_pairs_and_fails_outside_the_bar():
    rng = np.random.default_rng(3)
    obs = 270.0 + 20.0 * rng.random(4000)
    # Nothing measured: INCOMPLETE, never PASS.
    empty = _colocation({"both_clear/le60": _moments(obs[:10], obs[:10] + 0.1)})
    verdict = op.score_band(empty)["gate"]
    assert verdict["verdict"] == "INCOMPLETE"
    assert "10 pairs" in verdict["reason"]
    missing = _colocation({"all/all": _moments(obs, obs)})
    assert op.score_band(missing)["gate"]["verdict"] == "INCOMPLETE"
    # 4 K of scatter no line removes: FAIL, with the numbers in the reason.
    noisy = _colocation({"both_clear/le60": _moments(obs, obs + rng.normal(0.0, 4.0, obs.size))})
    verdict = op.score_band(noisy)["gate"]
    assert verdict["verdict"] == "FAIL"
    assert "outside 1.5 K" in verdict["reason"]
    # No entry ships for a failing band; a passing one carries its correction.
    assert op.operator_entry(op.score_band(noisy), 13) is None
    fine = _colocation({"both_clear/le60": _moments(obs, obs + 2.0 + rng.normal(0.0, 0.5, obs.size))})
    passing = op.score_band(fine)
    assert passing["gate"]["verdict"] == "PASS"
    entry = op.operator_entry(passing, 13)
    assert entry["band"] == 13
    # The correction removes the bias in the mean (the slope is diluted a
    # little below one by the noise on the regressor, so the intercept alone
    # is not the planted offset) and leaves the noise as the error.
    gate = passing["classes"]["both_clear/le60"]
    corrected_mean = entry["bias_correction"]["intercept_k"] + entry["bias_correction"]["slope"] * gate["mean_sim_k"]
    assert corrected_mean == pytest.approx(gate["mean_obs_k"], abs=1e-6)
    assert gate["bias_k"] == pytest.approx(2.0, abs=0.05)
    assert entry["observation_error_k"] == pytest.approx(0.5, abs=0.05)
    assert entry["simsat_kwargs"]["geo_navigation"] == "goes-r-abi"
    assert len(entry["assumptions"]) == len(op.ASSUMPTIONS)


def test_an_empty_class_arrives_as_nulls_and_scores_as_nan_not_a_crash():
    # serde writes f64 NaN as JSON null, so an empty class (n = 0) carries
    # null means and correlation; the scorecard must read it, not raise.
    empty = {
        "n": 0, "mean_obs_k": None, "mean_sim_k": None, "bias_k": None, "rmse_k": None,
        "std_obs_k": None, "std_sim_k": None, "correlation": None,
        "linear_fit_intercept_k": None, "linear_fit_slope": None, "rmse_after_linear_k": None,
        "min_diff_k": None, "max_diff_k": None,
        "moments": {"n": 0, "sum_obs": 0.0, "sum_sim": 0.0, "sum_obs_sq": 0.0, "sum_sim_sq": 0.0,
                    "sum_obs_sim": 0.0, "min_diff": None, "max_diff": None},
        "diff_histogram_1k": [0] * 120,
    }
    rng = np.random.default_rng(11)
    obs = 270.0 + 20.0 * rng.random(2000)
    score = op.score_band(_colocation({"both_cloudy/le60": empty, "both_clear/le60": _moments(obs, obs + 1.0)}))
    row = score["classes"]["both_cloudy/le60"]
    assert row["n"] == 0 and math.isnan(row["bias_k"]) and math.isnan(row["correlation"])
    assert score["gate"]["verdict"] == "PASS"


def test_bt_pack_family_reads_through_the_shared_container(tmp_path):
    from woof.globe.obs_pack import (
        BT_SCHEMA_V1, GoesPackError, read_goes_pack)
    from goes_pack_fixtures import write_bt_pack

    bt = np.full((3, 4), 280.0, dtype=np.float32)
    bt[0, 0] = np.nan
    lat = np.linspace(30.0, 31.0, 12, dtype=np.float32).reshape(3, 4)
    lon = np.linspace(-100.0, -99.0, 12, dtype=np.float32).reshape(3, 4)
    bcm = np.zeros((3, 4), dtype=np.float32)
    bcm[1, 1] = 1.0
    path = write_bt_pack(tmp_path / "bt.goespack", bt=bt, rad=bt * 0.3, lat=lat, lon=lon, bcm=bcm)
    pack = read_goes_pack(path, expected_schema=BT_SCHEMA_V1)
    assert pack.schema == BT_SCHEMA_V1
    assert pack.meta["band"] == 13
    assert pack.meta["planck"]["fk2"] == pytest.approx(1395.19)
    assert list(pack.meta["plane_order"]) == ["bt", "rad", "lat", "lon", "bcm", "rad_dqf", "acm_dqf"]
    assert np.isnan(pack.planes["bt"][0, 0]) and pack.planes["bt"][2, 3] == 280.0
    assert pack.planes["bcm"][1, 1] == 1.0
    # The scan angles sit on the 2 km lattice, half a pitch off the origin.
    assert pack.meta["x_scan_rad"][0] == pytest.approx((-2712 + 0.5) * 56e-6)
    # Demanding the CWP family of a BT pack is refused by name.
    with pytest.raises(GoesPackError, match="wrong family"):
        read_goes_pack(path, expected_schema="gpuwm-obs.goes-cwp.v2")


def test_readback_judge_wants_the_sign_and_exact_zero_for_an_identity():
    control = np.linspace(250.0, 300.0, 1000).reshape(20, 50)
    warmer = control + 1.8
    result = op.readback(control, warmer)
    assert result["mean_k"] == pytest.approx(1.8)
    assert result["moved_fraction"] == 1.0
    assert op.judge_calibration(13, "skin_plus", +1, result, minimum_magnitude_k=1.0)["passed"]
    assert not op.judge_calibration(13, "skin_plus", -1, result, minimum_magnitude_k=1.0)["passed"]
    assert not op.judge_calibration(13, "skin_plus", +1, result, minimum_magnitude_k=2.0)["passed"]
    same = op.readback(control, control.copy())
    assert same["identical"] is True and same["rms_k"] == 0.0
    assert op.judge_calibration(13, "identity", 0, same, minimum_magnitude_k=0.0)["passed"]
    assert not op.judge_calibration(13, "identity", 0, result, minimum_magnitude_k=0.0)["passed"]
    # A mask restricts the census; NaN pixels never count.
    masked = control.copy()
    masked[0, :] = np.nan
    mask = np.zeros_like(control, dtype=bool)
    mask[5:10] = True
    partial = op.readback(masked, masked - 0.7, mask=mask)
    assert partial["n"] == 250
    assert partial["mean_k"] == pytest.approx(-0.7)
    with pytest.raises(op.AbiOperatorError, match="shape"):
        op.readback(control, control[:10])


def test_levels_above_selects_the_upper_troposphere():
    # A 10-layer pure-pressure ladder from 1 to 1000 hPa.
    p_half = np.array([100.0, 2000, 5000, 10000, 20000, 30000, 50000, 70000, 85000, 95000, 100000])
    cfg = types.SimpleNamespace(a_half_pa=p_half, b_half=np.zeros_like(p_half))
    sel = op.levels_above(cfg, 50_000.0)
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    assert sel == slice(0, 6)
    assert (p_full[sel] < 50_000.0).all()
    assert p_full[sel.stop] >= 50_000.0
    with pytest.raises(op.AbiOperatorError):
        op.levels_above(cfg, 10.0)


def test_sim_plane_sidecar_round_trips_and_refuses_an_off_lattice_render(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4) + 250.0
    crop = {"sample_angle_urad": 56.0, "x_index_min": -10, "x_index_max": -7,
            "y_index_min": 100, "y_index_max": 102, "nx": 4, "ny": 3}
    plane = op.write_sim_plane(tmp_path / "t.f32", values, band=13, crop=crop, label="t",
                               tape="tape.nc", sensor="goes-r-abi-band13-fm4")
    back, sidecar = op.read_sim_plane(plane.path)
    np.testing.assert_array_equal(back, values)
    assert sidecar["schema"] == op.SIM_PLANE_SCHEMA
    assert sidecar["abi_fixed_grid_crop"]["y_index_max"] == 102
    assert sidecar["finite"] == 12
    assert json.loads(Path(f"{plane.path}.json").read_text())["band"] == 13
    with pytest.raises(op.AbiOperatorError, match="exact ABI lattice"):
        op.write_sim_plane(tmp_path / "u.f32", values, band=13, crop={}, label="u", tape="t", sensor="s")
    with pytest.raises(op.AbiOperatorError, match="crop says"):
        op.write_sim_plane(tmp_path / "v.f32", values, band=13, crop={**crop, "nx": 5}, label="v",
                           tape="t", sensor="s")
    # A mask of the wrong size is refused; the right size is recorded relative to the plane.
    (tmp_path / "t.mask").write_bytes(bytes(11))
    with pytest.raises(op.AbiOperatorError, match="bytes"):
        op.attach_mask(plane, tmp_path / "t.mask")
    (tmp_path / "t.mask").write_bytes(bytes(12))
    op.attach_mask(plane, tmp_path / "t.mask")
    assert json.loads(Path(f"{plane.path}.json").read_text())["mask_plane"] == "t.mask"


def _central_angle_deg(lat: float, lon: float, sub_lon: float = -75.2) -> float:
    lat_r = math.radians(lat)
    dlon = math.radians(lon - sub_lon)
    return math.degrees(math.acos(math.cos(lat_r) * math.cos(dlon)))


def test_default_tiles_keep_every_corner_on_the_visible_disk():
    limb = math.degrees(math.acos(6_371.0 / 42_164.0))
    for label, lat_min, lat_max, lon_min, lon_max in op.DEFAULT_TILES:
        for lat in (lat_min, lat_max):
            for lon in (lon_min, lon_max):
                assert _central_angle_deg(lat, lon) < limb - 2.0, (label, lat, lon)
    # And the tiles cover the window without a gap.
    lats = sorted({t[1] for t in op.DEFAULT_TILES} | {t[2] for t in op.DEFAULT_TILES})
    lons = sorted({t[3] for t in op.DEFAULT_TILES} | {t[4] for t in op.DEFAULT_TILES})
    assert lats == [-62.0, 0.0, 62.0]
    assert lons == [-140.0, -75.0, -10.0]


def test_bands_table_renders_on_the_exact_lattice_and_refuses_unknown_bands(tmp_path):
    for band, spec in op.BANDS.items():
        assert spec.band == band
        assert spec.simsat_kwargs["geo_navigation"] == "goes-r-abi"
        assert spec.simsat_kwargs["resolution"] == "abi2km"
        assert spec.simsat_kwargs["view"] == "geo"
    assert op.BANDS[13].simsat_kwargs["sensor"] == "goes-r-abi-band13-fm4"
    assert op.BANDS[8].simsat_kwargs["band"] == "6.2"
    with pytest.raises(op.AbiOperatorError, match="no operator entry"):
        op.render_tile("tape.nc", 7, tmp_path, label="x", simsat=object())

    # A fake binding whose render carries no lattice crop is refused by name.
    class Geo:
        abi_fixed_grid_crop = None
        science_warnings = []

    fake = types.SimpleNamespace(render_ir=lambda tape, **kw: (np.zeros((2, 2), np.float32), Geo()))
    with pytest.raises(op.AbiOperatorError, match="abi_fixed_grid_crop"):
        op.render_tile("tape.nc", 13, tmp_path, label="x", simsat=fake)

    # And one that does is written with its sidecar.
    class GoodGeo:
        abi_fixed_grid_crop = {"sample_angle_urad": 56.0, "x_index_min": 0, "x_index_max": 1,
                               "y_index_min": 0, "y_index_max": 1, "nx": 2, "ny": 2}
        science_warnings = ["gray absorption"]
        geo_navigation = "goes-r-abi"

    good = types.SimpleNamespace(render_ir=lambda tape, **kw: (np.full((2, 2), 280.0, np.float32), GoodGeo()))
    plane = op.render_tile("tape.nc", 13, tmp_path, label="ok", simsat=good)
    assert plane.shape == (2, 2) and plane.finite == 4
    assert plane.sidecar["science_warnings"] == ["gray absorption"]


def test_run_case_refuses_a_band_without_its_radiance(tmp_path, monkeypatch):
    monkeypatch.setattr(op, "find_rw_goes", lambda explicit=None: Path("rw_goes"))
    cfg = types.SimpleNamespace(config_hash="0" * 64, name="x")
    with pytest.raises(op.AbiOperatorError, match="no L1b radiance file"):
        op.run_case(cfg, tmp_path / "ck.npz", start_date="2026-09-01_18:00:00", out_dir=tmp_path / "o",
                    goes_rad={13: tmp_path / "r13.nc"}, bands=(13, 8))
    with pytest.raises(op.AbiOperatorError, match="no operator entry"):
        op.run_case(cfg, tmp_path / "ck.npz", start_date="2026-09-01_18:00:00", out_dir=tmp_path / "o",
                    goes_rad={7: tmp_path / "r7.nc"}, bands=(7,))


def test_find_rw_goes_names_the_remedy(monkeypatch, tmp_path):
    monkeypatch.delenv("WOOF_RW_GOES", raising=False)
    monkeypatch.setattr(op.shutil, "which", lambda name: None)
    monkeypatch.setattr(op, "__file__", str(tmp_path / "a" / "b" / "c.py"))
    with pytest.raises(op.AbiOperatorError, match="cargo build"):
        op.find_rw_goes(None)
    binary = tmp_path / "rw_goes"
    binary.write_bytes(b"x")
    assert op.find_rw_goes(binary) == binary
    monkeypatch.setenv("WOOF_RW_GOES", str(binary))
    assert op.find_rw_goes(None) == binary


def test_cli_parses_band_files_and_tiles():
    from woof.globe.cli import _parse_band_files, _parse_tiles

    assert _parse_band_files(["13=a.nc", "8=b.nc"], "--goes-rad") == {13: Path("a.nc"), 8: Path("b.nc")}
    with pytest.raises(ValueError, match="BAND=FILE"):
        _parse_band_files(["a.nc"], "--goes-rad")
    assert _parse_tiles(None) == op.DEFAULT_TILES
    assert _parse_tiles(["t,0,10,-20,-5"]) == (("t", 0.0, 10.0, -20.0, -5.0),)
    with pytest.raises(ValueError, match="LABEL,LATMIN"):
        _parse_tiles(["t,0,10"])
