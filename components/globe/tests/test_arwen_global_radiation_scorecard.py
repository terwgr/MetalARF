"""The radiation scorecard and the budget carriers it reads.

Calibration first (every family in both directions, through the module's
own ``calibrate``), then the reader: two native checkpoints become the
exact interval mean, a checkpoint without the held planes is refused by
name, and the reference mapping the cloud-cover score reads is the arms'
own analysis mapping plus the four cover records.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from test_arwen_global_level5_native import (
    FAKE_GLW,
    FAKE_GSW,
    FAKE_OLR,
    _advance,
    _fake_modules,
    _options,
)

from woof.globe import radiation_scorecard as rs
from woof.globe.analysis_initial import (
    PACKAGE_AUTHORITIES_DIR as _package_authorities_dir_value)
from woof.globe.checkpoint import write_checkpoint
from woof.globe.config import load_config
from woof.globe.physics.native_suite import ArwenCudaColumnSuite
from woof.globe.runner import build_model_and_cold_state, run
from woof.globe.water_budget import GLOBAL, LAND, BANDS


# RETIRED 2026-09-09.  Two skips stood here, both patch item 03 of the
# series the carve wrote for the engine's owner: the published
# `np_rrtmgp_hydrometeor_paths` took no `size_bounds`, and
# `np_bound_cloud_sizes` did not exist.  The float64 mirror is now carried at
# `woof.globe.core.npref`, beside the kernels it mirrors, so the reference
# these tests grade against is this package's own -- and it has to be: a
# mirror that is not the kernel it mirrors is a flawed instrument.


CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")

#: The carried source mappings, INSIDE the installed package.  This read used
#: to be `<repo>/woof/authorities`, the engine checkout the model was carved
#: out of, which exists in no install and in this repository not at all.  It
#: was invisible while a conftest hook turned any refusal naming a mapping
#: into a skip; that hook went when the six mappings came into the package,
#: and the read below is where the package actually keeps them.
AUTHORITIES = _package_authorities_dir_value


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def test_every_calibration_family_meets_its_bar():
    report = rs.calibrate()
    readings = report["readings"]

    planes = readings["planted_planes"]
    for sign in ("b=+90", "b=-90"):
        assert abs(planes[sign]["global_error"]) < 1.0e-9
        assert abs(planes[sign]["northern_hemisphere_error_vs_row_quadrature"]) < 1.0e-9
        for name, _lo, _hi in BANDS:
            assert abs(planes[sign]["band_errors_vs_row_quadrature"][name]) < 1.0e-9, name
        # the grid's own discretization of the region edges, not the
        # instrument's: visible at T21's 33 rows, under 0.2 percent of b
        # on the control grid
        assert abs(planes[sign]["region_discretization_error_over_b"]["northern_hemisphere"]) < 0.02
    assert planes["constant_max_error"] < 1.0e-9
    coarse = planes["grid_discretization_error_of_sin2_by_truncation"]["T21 (33 rows)"]
    fine = planes["grid_discretization_error_of_sin2_by_truncation"]["T255 (384 rows, the control grid)"]
    assert abs(coarse["northern_hemisphere"]) > 0.01
    for name, value in fine.items():
        assert abs(value) < 2.0e-3, name

    integrals = readings["planted_time_integrals"]
    for key in ("flux=+250", "flux=-40"):
        for name in ("sw_down_surface", "lw_up_top", "total_cloud_cover"):
            assert abs(integrals[key][name]) < 1.0e-9, (key, name)
        assert abs(integrals[key]["sw_net_surface_reads_zero"]) < 1.0e-9
    assert integrals["stalled_seconds_refused"] is True
    assert integrals["held_fallback"]["sampling"] == rs.HELD_SAMPLING
    assert abs(integrals["held_fallback"]["sw_down_surface_minus_200"]) < 1.0e-9
    assert integrals["held_fallback"]["sw_up_top_incomplete"] == "INCOMPLETE"

    offsets = readings["planted_reference_offsets"]
    for key in ("delta=+0.05", "delta=-0.05", "delta=+0.2", "delta=-0.2"):
        assert abs(offsets[key]["bias_error"]) < 1.0e-12, key
        assert abs(offsets[key]["rmse_error"]) < 1.0e-12, key
        assert abs(offsets[key]["land_bias_error"]) < 1.0e-12, key
    assert abs(offsets["noise sigma 0.1"]["bias"]) < 0.01
    assert offsets["noise sigma 0.1"]["rmse"] == pytest.approx(0.1, rel=0.1)
    assert offsets["offset_on_reference_reads_opposite"] == pytest.approx(-0.05)

    overlap = readings["planted_overlap"]
    for name, value in overlap.items():
        assert abs(value) < 1.0e-12, name

    sizes = readings["planted_sizes"]
    for name, expected in sizes["expected"].items():
        assert sizes["counts"][name] == pytest.approx(expected), name
    assert sizes["carried_path_error"] == 0.0
    assert sizes["untouched_path_error"] == 0.0
    assert sizes["sizes_inside_bounds"] is True
    for name, expected in sizes["weighted_expected"].items():
        assert sizes["weighted_counts"][name] == pytest.approx(expected), name
    assert sizes["unweighted_radiative_equals_in_cloud"] is True

    zonal = readings["planted_zonal_waves"]
    for key in ("b=+37", "b=-37"):
        for name, value in zonal[key].items():
            assert abs(value) < 1.0e-9, (key, name)

    hold = readings["planted_zero_order_hold"]
    for name in ("sw_daytime_bump_1000", "lw_240_plus_10_diurnal"):
        assert abs(hold[name]["integral_error_1_24"]) < 1.0e-9, name
        assert abs(hold[name]["integral_error_12_24"]) < 1.0e-9, name
    # the held-plane fallback is not the applied mean: on a single-longitude
    # daytime bump the hourly trapezoid over-reads by several W/m2 of a 330
    # W/m2 mean, on the nearly constant longwave it under-reads by a tenth
    # of a W/m2; both are recorded so the fallback rows' caveat has a size
    assert 5.0 < hold["sw_daytime_bump_1000"]["held_trapezoid_error_1_24"] < 12.0
    assert 1.0 < hold["sw_daytime_bump_1000"]["held_trapezoid_error_12_24"] < 4.0
    assert -0.3 < hold["lw_240_plus_10_diurnal"]["held_trapezoid_error_1_24"] < 0.0

    regrid = readings["planted_reference_regrid"]
    assert regrid["descending_latitude_max_abs_error"] < 1.0e-4
    assert abs(regrid["descending_latitude_bias"]) < 1.0e-6
    for key in ("+0.05", "-0.05"):
        assert abs(regrid[f"planted_model_offset_{key}_bias_error"]) < 1.0e-6, key
        assert abs(regrid[f"planted_model_offset_{key}_land_bias_error"]) < 1.0e-6, key
    assert regrid["ascending_minus_descending_max_abs"] < 1.0e-12
    assert regrid["minus180_convention_minus_descending_max_abs"] < 1.0e-12


def test_an_absent_carrier_is_incomplete_never_a_number():
    grid = rs.synthetic_grid(21)
    sample = rs.synthetic_sample(grid, time_s=0.0, step=0,
                                 planes={"swdown": 100.0, "gsw": 90.0, "glw": 300.0, "olr": 240.0})
    masks = rs.region_masks(grid, sample.land)
    rows = rs.instantaneous_rows(sample, grid, masks)
    assert rows["sw_down_surface"]["status"] == "measured"
    assert rows["sw_down_surface"][GLOBAL] == pytest.approx(100.0)
    assert rows["sw_up_surface"][GLOBAL] == pytest.approx(10.0)
    for name in ("lw_up_surface", "lw_net_surface", "net_surface", "sw_up_top",
                 "net_top", "atmosphere_net", "total_cloud_cover", "daylit_fraction"):
        assert rows[name]["status"] == "INCOMPLETE", name
        assert GLOBAL not in rows[name]
    # a region with no cells reads None, never a number
    empty = rs.region_masks(grid, np.zeros(grid.shape, dtype=bool))
    rows = rs.instantaneous_rows(sample, grid, empty)
    assert rows["sw_down_surface"][LAND] is None
    assert rows["sw_down_surface"][GLOBAL] == pytest.approx(100.0)


def test_scorecard_window_prefers_the_time_integrals_and_names_the_sampling():
    grid = rs.synthetic_grid(21)
    ny, nx = grid.shape
    planes = {"swdown": 0.0, "gsw": 0.0, "glw": 0.0, "olr": 0.0}
    zeros = {k: np.zeros((ny, nx)) for k in rs.ACCUMULATORS}
    samples = [rs.synthetic_sample(grid, time_s=0.0, step=0, planes=planes,
                                   accumulators=zeros, accumulated_s=0.0)]
    for hour in (1, 2):
        acc = {k: np.full((ny, nx), 3600.0 * hour * 240.0) for k in rs.ACCUMULATORS}
        samples.append(rs.synthetic_sample(grid, time_s=3600.0 * hour, step=72 * hour, planes=planes,
                                           accumulators=acc, accumulated_s=3600.0 * hour))
    result = rs.scorecard(samples, grid, label="planted", window_hours=(1.0, 2.0))
    assert result["run_mean"]["sampling"] == rs.ACCUMULATED_SAMPLING
    assert result["run_mean"]["fluxes"]["lw_up_top"][GLOBAL] == pytest.approx(240.0)
    assert result["window"]["accumulated_s"] == pytest.approx(3600.0)
    assert result["window"]["fluxes"]["sw_down_top"][GLOBAL] == pytest.approx(240.0)
    assert all(i["sampling"] == rs.ACCUMULATED_SAMPLING for i in result["intervals"])
    clim = result["climatology"]["readings"]["lw_up_top"]
    assert clim["status"] == "measured" and clim["inside"] is True
    assert result["climatology"]["caveat"] == rs.CLIMATOLOGY_CAVEAT
    assert "not read" in result["reference"]["fluxes"]
    # the same samples without the integrals fall to the held sampling, and
    # a held plane that is 0 at every checkpoint reads 0, not 240
    held = [replace(s, accumulators={}, accumulated_s=None) for s in samples]
    result = rs.scorecard(held, grid, label="held")
    assert result["run_mean"]["sampling"] == rs.HELD_SAMPLING
    assert result["run_mean"]["fluxes"]["lw_up_top"][GLOBAL] == 0.0


# --------------------------------------------------------------------------
# the reader
# --------------------------------------------------------------------------


def _receipt_for(cfg, model) -> dict:
    grid = model.transform.grid
    return {
        "config_hash": cfg.config_hash,
        "config": {
            "truncation": cfg.truncation,
            "dealias_factor": cfg.dealias_factor,
            "a_half_pa": [float(v) for v in cfg.vertical.a_half_pa],
            "b_half": [float(v) for v in cfg.vertical.b_half],
        },
        "transform": {"nlat": grid.nlat, "nlon": grid.nlon, "radius_m": grid.radius_m},
    }


def test_reader_refuses_a_checkpoint_without_the_held_planes(tmp_path):
    cfg = load_config(CONFIG)
    cfg = replace(cfg, duration_s=2.0 * cfg.dt_s, output_interval_s=2.0 * cfg.dt_s)
    outdir = tmp_path / "run"
    assert run(cfg, outdir)["status"] == "pass"
    model, _state = build_model_and_cold_state(cfg)
    reader = rs.RadiationReader(rs.BudgetGrid.for_shape(model.transform.grid.nlat, model.transform.grid.nlon),
                                optics=False)
    # the cold-start checkpoint is named as such (and skipped by the CLI) ...
    with pytest.raises(rs.ColdStartCheckpoint, match="cold start"):
        reader.sample(outdir / "arwen_global_step00000000.npz")
    # ... a later checkpoint of a run without the planes is refused by name
    with pytest.raises(ValueError, match="carries no held radiation plane"):
        reader.sample(outdir / "arwen_global_step00000002.npz")
    with pytest.raises(SystemExit, match="nothing to score"):
        rs.main(["--run-dir", str(outdir), "--no-optics", "--max-checkpoints", "1"])


def test_reader_turns_two_native_checkpoints_into_the_exact_interval_mean(tmp_path):
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, cfg.dt_s)
    dt = float(exchange.dt_s)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules())
    first = suite.step(exchange)
    second = suite.step(_advance(exchange, first, dt))
    to_numpy = model.transform.backend.to_numpy
    paths = []
    for step, result in ((1, first), (2, second)):
        bundle = replace(state, physics_state=result.physics_state)
        bundle.atmosphere.step = step
        bundle.atmosphere.time_s = step * dt
        paths.append(write_checkpoint(
            tmp_path / f"arwen_global_step{step:08d}.npz", bundle,
            config_hash=cfg.config_hash, to_numpy=to_numpy,
            semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator))
    reader = rs.RadiationReader.from_receipt(_receipt_for(cfg, model))
    reader._truncation = cfg.truncation
    samples = [reader.sample(p) for p in paths]
    assert samples[0].metadata["lwupb_source"] == "surface_emission_formula"
    assert samples[0].accumulated_s == pytest.approx(dt)
    assert samples[1].accumulated_s == pytest.approx(2.0 * dt)
    assert "lwupb" in samples[0].planes and "cldfra_total" in samples[0].planes
    assert samples[0].optics is not None
    optics = samples[0].optics
    for key in ("size_bounding", "size_bounding_before", "ice_visible_optical_depth",
                "coupling", "radii_source", "cloudy_layer_fraction"):
        assert key in optics, key
    for name in rs.SIZE_BOUNDING_FIELDS:
        assert name in optics["size_bounding"] and name in optics["size_bounding_before"], name
    assert optics["size_bounding"]["ice_cells"] == optics["size_bounding_before"]["ice_cells"]
    masks = rs.region_masks(reader.grid, samples[-1].land)
    interval = rs.accumulated_interval(samples[0], samples[1], reader.grid, masks)
    assert interval is not None and interval["sampling"] == rs.ACCUMULATED_SAMPLING
    assert interval["accumulated_s"] == pytest.approx(dt)
    assert interval["fluxes"]["lw_up_top"][GLOBAL] == pytest.approx(FAKE_OLR, rel=1e-6)
    assert interval["fluxes"]["lw_down_surface"][LAND] == pytest.approx(FAKE_GLW, rel=1e-6)
    assert interval["fluxes"]["sw_net_surface"][GLOBAL] == pytest.approx(FAKE_GSW, rel=1e-6)
    result = rs.scorecard(samples, reader.grid, label="fake native")
    assert result["run_mean"]["sampling"] == rs.ACCUMULATED_SAMPLING
    assert result["checkpoints"][0]["radiation_calls"] == 1
    assert result["checkpoints"][0]["lwupb_source"] == "surface_emission_formula"
    assert "cloud_optics" in result["checkpoints"][0]
    lines = rs.summary_lines(result)
    assert any("cloud optics at step" in line for line in lines)


# --------------------------------------------------------------------------
# the reference
# --------------------------------------------------------------------------


def test_a_refused_reference_product_is_incomplete_beside_a_scored_one(monkeypatch, tmp_path):
    """One product the engine refuses does not take the other down, and
    the refusal is named at its checkpoint."""
    import woof.mapped_source as mapped_source

    grid = rs.synthetic_grid(21)
    ny, nx = grid.shape
    lat2d, lon2d = rs.model_lat_lon(grid)

    class Field:
        def __init__(self, values):
            self.values = values

    class Frame:
        valid_time = "2026-09-02T00:00:00Z"
        latitude = np.linspace(-90.0, 90.0, 37)
        longitude = np.arange(0.0, 360.0, 10.0)

        def __init__(self):
            cover = 0.5 + 0.2 * np.sin(np.deg2rad(self.latitude))[:, None] * np.ones((1, 36))
            self.fields = {name: Field(cover) for name in rs.REFERENCE_COVER_FIELDS}

    def fake_decode(mapping, sources):
        # `str()` because the door hands the engine `Path` objects: the
        # decode receipt records the source absolute, so the caller resolves
        # before it calls.  A fake that assumed the old string argument read
        # `"bad" in PosixPath(...)` and died on a TypeError inside itself,
        # which is a test failing at its own stub rather than at the door.
        if "bad" in str(sources[0]):
            raise ValueError("GRIB2 field 608 failed to decode: Section 5 declares 0 data points")
        return [Frame()]

    monkeypatch.setattr(mapped_source, "decode_mapped_source", fake_decode)
    frames = rs.decode_reference_cover(["bad.grb2", "good.grb2"])
    assert frames[0]["status"] == "INCOMPLETE" and "field 608" in frames[0]["reason"]
    assert frames[1]["status"] == "measured"
    planes = {"swdown": 0.0, "gsw": 0.0, "glw": 0.0, "olr": 0.0,
              "cldfra_total": 0.5 + 0.2 * np.sin(np.deg2rad(lat2d))}
    samples = [rs.synthetic_sample(grid, time_s=0.0, step=0, planes=planes),
               rs.synthetic_sample(grid, time_s=3600.0, step=72, planes=planes)]
    result = rs.scorecard(samples, grid, reference_frames={0: frames[0], 72: frames[1]})
    assert result["reference"]["cloud_cover"]["0"]["status"] == "INCOMPLETE"
    scored = result["reference"]["cloud_cover"]["72"]
    assert scored["status"] == "measured"
    assert abs(scored["total_cloud_cover"][GLOBAL]["bias"]) < 0.02
    lines = rs.summary_lines(result)
    assert any("INCOMPLETE" in line and "step 0" in line for line in lines)


def test_the_radii_volumes_are_paired_with_the_tracers_level_for_level():
    """The physics namespace is checkpointed in the column batch's order
    (surface first), the atmosphere in the model's (top first): the view
    flips the radii and refuses a pairing that reads the kernel's no-mass
    radius on the cloud layers."""
    nz, ny, nx = 6, 2, 2
    # a radius volume in batch order: computed (8 um) at batch levels 0-1
    # (the lowest two model levels), the no-mass 25 elsewhere
    effc_batch = np.full((nz, ny, nx), rs.MORRISON_NO_MASS_RADIUS_UM)
    effc_batch[0:2] = 8.0
    qc = np.zeros((nz, ny, nx))
    qc[nz - 2:] = 1.0e-4          # cloud on the lowest two model levels
    flipped = rs.physics_volume_in_atmosphere_order(effc_batch)
    assert np.all(flipped[nz - 2:] == 8.0) and np.all(flipped[: nz - 2] == 25.0)
    cols = lambda v: v.transpose(1, 2, 0).reshape(ny * nx, nz)
    assert rs.radii_pairing_check(cols(flipped), cols(qc), cols(flipped), cols(qc)) == 0.0
    with pytest.raises(ValueError, match="not paired level for level"):
        rs.radii_pairing_check(cols(effc_batch), cols(qc), cols(effc_batch), cols(qc))


def test_reference_mapping_is_the_analysis_mapping_plus_the_cover_records():
    cover = json.loads((AUTHORITIES / rs.REFERENCE_MAPPING).read_text(encoding="utf-8"))
    base = json.loads((AUTHORITIES / "rw-wps-gdas-global-analysis-grib2.mapping.json").read_text(encoding="utf-8"))
    expected = {
        "total_cloud_cover": (1, 10), "low_cloud_cover": (3, 214),
        "middle_cloud_cover": (4, 224), "high_cloud_cover": (5, 234),
    }
    for name, (parameter, level_type) in expected.items():
        selector = cover["fields"][name]["selectors"][0]
        assert (selector["discipline"], selector["category"], selector["parameter"]) == (0, 6, parameter), name
        assert selector["level_type"] == level_type, name
        # instantaneous records only: the time-averaged flux records carry
        # template 4.8 and are not bound
        assert selector["pdt"] == 0, name
        assert cover["fields"][name]["units"] == {"source": "%", "target": "1", "scale": 0.01}
    others = {k: v for k, v in cover["fields"].items() if k not in expected}
    assert others == base["fields"]
    assert cover["schema"] == base["schema"] and cover["format"] == base["format"]
    assert cover["coordinates"] == base["coordinates"]
    assert "interval time semantics" in cover["target"]["name"]
