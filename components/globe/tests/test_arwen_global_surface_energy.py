"""Calibration of the surface-energy instrument in both directions.

Synthetic sides on analytic fields: identical sides read zero, a planted
offset is read back where it was planted and nowhere else, the height
interpolation and the hydrostatic heights are exact on the profiles they
should be exact on, the energy balance closes and reads a planted hole,
and the GFS closure check refuses the wrong ground-flux sign.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import json

import numpy as np
import pytest

from conftest import requires_netcdf_writer  # noqa: E402

from woof.globe import surface_energy as se
from woof.globe.constants import (
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    STEFAN_BOLTZMANN,
)




def _engine_has(module_name: str, symbol: str | None = None) -> bool:
    """Whether the INSTALLED engine carries what this file grades against."""

    import importlib

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    return symbol is None or hasattr(module, symbol)


_ENGINE_HAS_IT = _engine_has("woof.verify.harness.surface_bias", "interpolate_to_tape")

#: The subject of these tests is an engine symbol the carve left on the
#: engine's side of the boundary, supplied as patch item 04 of the series
#: written for the engine's owner.  Against a published engine that does not
#: carry it yet the right outcome is a skip naming the symbol, not a red
#: test whose verdict describes another project's release schedule.
_needs_the_surface_bias_reference = pytest.mark.skipif(
    not _ENGINE_HAS_IT,
    reason="the installed woof does not carry woof.verify.harness.surface_bias"
           ".interpolate_to_tape: patch item 04 of the "
           "series the carve wrote for the engine")



def _grid(dlat, dlon):
    lat = np.arange(-88.0, 89.0, dlat)
    lon = np.arange(0.0, 360.0, dlon)
    return lat, lon


def _analytic(lat2d, lon2d):
    """Fields linear in latitude and longitude (bilinear interpolation of
    a coarser grid reproduces them exactly)."""
    x = np.cos(np.deg2rad(lon2d)) * 0.0 + lon2d / 360.0
    base = {
        "tsk": 300.0 + 0.05 * lat2d + 2.0 * x,
        "t2": 297.0 + 0.05 * lat2d + 2.0 * x,
        "td2": 285.0 + 0.02 * lat2d,
        "q2": 0.008 + 1.0e-5 * lat2d,
        "t80": 296.0 + 0.05 * lat2d + 2.0 * x,
        "t100": 295.8 + 0.05 * lat2d + 2.0 * x,
        "ps": 98000.0 + 10.0 * lat2d,
        "hgt": 300.0 + np.zeros_like(lat2d),
        "land": np.where(np.abs(lat2d) < 70.0, 1.0, 0.0),
        "vegfra": 0.5 + 0.002 * lat2d,
        "z0": 0.1 + np.zeros_like(lat2d),
        "tslb1": 295.0 + 0.04 * lat2d,
        "smois1": 0.25 + 0.001 * lat2d,
        "hfx": 150.0 + lat2d,
        "lh": 120.0 - lat2d,
        "dswrf": 700.0 + np.zeros_like(lat2d),
        "uswrf": 140.0 + np.zeros_like(lat2d),
        "dlwrf": 380.0 + np.zeros_like(lat2d),
        "ulwrf": 460.0 + np.zeros_like(lat2d),
        # The frozen rungs: a pack south of 72S (partial west of 180,
        # full east of it) and north of 75N, snow on the northern land
        # above 55N, a 10 m wind linear in latitude and longitude.
        "seaice": np.where(lat2d <= -72.0, np.where(lon2d < 180.0, 0.6, 1.0),
                           np.where(lat2d >= 75.0, 0.9, 0.0)),
        "snow": np.where((np.abs(lat2d) < 70.0) & (lat2d > 55.0), 20.0, 0.0),
        "u10": 4.0 + 0.02 * lat2d + x,
        "v10": 1.5 + 0.01 * lat2d,
    }
    base["wspd10"] = np.hypot(base["u10"], base["v10"])
    base["rnet"] = base["dswrf"] - base["uswrf"] + base["dlwrf"] - base["ulwrf"]
    base["g"] = base["rnet"] - base["hfx"] - base["lh"]
    base["gflux"] = base["g"]
    base["residual"] = np.zeros_like(lat2d)
    return base


def _reference(dlat=2.0, dlon=2.0):
    lat, lon = _grid(dlat, dlon)
    lon2d, lat2d = np.meshgrid(lon, lat)
    fields = _analytic(lat2d, lon2d)
    state = {"lat": lat, "lon": lon, "fields": fields, "valid_time": "t", "path": "state"}
    flux = {"lat": lat, "lon": lon, "fields": fields, "valid_time": "t", "path": "flux"}
    return state, flux


def _model(dlat=4.0, dlon=4.0, **offsets):
    lat, lon = _grid(dlat, dlon)
    lon2d, lat2d = np.meshgrid(lon, lat)
    fields = _analytic(lat2d, lon2d)
    fields["t1"] = fields["t80"] + 0.6
    fields["z1"] = np.full_like(lat2d, 23.0)
    fields["swdown"] = fields["dswrf"]
    fields["glw"] = fields["dlwrf"]
    fields["albedo"] = fields["uswrf"] / fields["dswrf"]
    fields["emiss"] = np.full_like(lat2d, 0.95)
    # The model's own frozen columns: the pack (its fraction at or above
    # one half) and the ice class on the land south of 60S and on the
    # Greenland box; the surface layer's first-level wind above the 10 m one.
    land = fields["land"] >= 0.5
    lon180 = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    pack = fields["seaice"] >= se.SEA_ICE_THRESHOLD
    fields["landice"] = land & ((lat2d <= -60.0) | ((lat2d >= 60.0) & (lon180 >= -75.0) & (lon180 <= -10.0)))
    fields["frozen"] = pack | fields["landice"]
    fields["wspd1"] = 1.15 * fields["wspd10"]
    for name, value in offsets.items():
        fields[name] = fields[name] + value
    return se.SurfaceSample(step=0, time_s=0.0, lat=lat2d, lon=lon2d, fields=fields)


def _rediagnose_wind(fields, cells, z0_old, z0_new):
    """Move a side's 10 m wind on ``cells`` by the neutral log-law change
    from ``z0_old`` to ``z0_new`` drawn from the same first-level wind (the
    model's z1 and wspd1 = 1.15 wspd10, the analytic sample's rule)."""
    z1 = 23.0
    speed = np.hypot(fields["u10"], fields["v10"])
    wspd1 = 1.15 * speed
    delta = se.log_law_wind(wspd1, z1, z0_new) - se.log_law_wind(wspd1, z1, z0_old)
    scale = np.where(cells, (speed + delta) / speed, 1.0)
    fields["u10"] = fields["u10"] * scale
    fields["v10"] = fields["v10"] * scale
    if "wspd10" in fields:
        fields["wspd10"] = np.hypot(fields["u10"], fields["v10"])


def test_identical_sides_read_zero_on_every_rung():
    state, flux = _reference()
    result = se.localize(_model(), state, flux)
    for region in se.REGIONS:
        entry = result["regions"][region]
        assert entry["n"] > 0
        for key, _unit, _name in se.COMPARED:
            assert entry[key]["bias"] == pytest.approx(0.0, abs=1e-9), key
            assert entry[key]["rmse"] == pytest.approx(0.0, abs=1e-9), key
        d = entry["decomposition_k"]
        for term in ("t2_bias", "column_t80_bias", "surface_layer_d_t2_minus_t80", "skin_d_tsk_minus_t2"):
            assert d[term] == pytest.approx(0.0, abs=1e-9), term
        assert entry["partition"]["model"]["closure_residual_w_m2"] == pytest.approx(0.0, abs=1e-9)
        assert entry["partition"]["gfs"]["closure_residual_w_m2"] == pytest.approx(0.0, abs=1e-9)


def test_a_planted_skin_offset_lands_in_the_skin_term_only():
    state, flux = _reference()
    result = se.localize(_model(tsk=2.0), state, flux)
    d = result["regions"]["conus_land"]["decomposition_k"]
    assert d["tsk_bias"] == pytest.approx(2.0, abs=1e-9)
    assert d["t2_bias"] == pytest.approx(0.0, abs=1e-9)
    assert d["column_t80_bias"] == pytest.approx(0.0, abs=1e-9)
    assert d["surface_layer_d_t2_minus_t80"] == pytest.approx(0.0, abs=1e-9)
    assert d["skin_d_tsk_minus_t2"] == pytest.approx(2.0, abs=1e-9)
    assert result["regions"]["conus_land"]["tsk_k"]["rmse"] == pytest.approx(2.0, abs=1e-9)


def test_a_planted_column_offset_lands_in_the_column_term_only():
    state, flux = _reference()
    result = se.localize(_model(t2=1.5, t80=1.5, t100=1.5, t1=1.5), state, flux)
    d = result["regions"]["conus_land"]["decomposition_k"]
    assert d["t2_bias"] == pytest.approx(1.5, abs=1e-9)
    assert d["column_t80_bias"] == pytest.approx(1.5, abs=1e-9)
    assert d["surface_layer_d_t2_minus_t80"] == pytest.approx(0.0, abs=1e-9)
    assert d["skin_d_tsk_minus_t2"] == pytest.approx(-1.5, abs=1e-9)
    assert se.verdict(result["regions"])["largest_term"] == "column"


def test_a_planted_screen_offset_lands_in_the_surface_layer_term_only():
    state, flux = _reference()
    result = se.localize(_model(t2=-1.0), state, flux)
    d = result["regions"]["nh_midlat_land"]["decomposition_k"]
    assert d["t2_bias"] == pytest.approx(-1.0, abs=1e-9)
    assert d["column_t80_bias"] == pytest.approx(0.0, abs=1e-9)
    assert d["surface_layer_d_t2_minus_t80"] == pytest.approx(-1.0, abs=1e-9)
    assert d["skin_d_tsk_minus_t2"] == pytest.approx(1.0, abs=1e-9)
    assert se.verdict(result["regions"], "nh_midlat_land")["largest_term"] == "surface_layer"


def test_a_planted_flux_partition_shift_is_read_in_both_fluxes():
    state, flux = _reference()
    result = se.localize(_model(hfx=40.0, lh=-40.0), state, flux)
    entry = result["regions"]["global_land"]
    assert entry["hfx_w_m2"]["bias"] == pytest.approx(40.0, abs=1e-9)
    assert entry["lh_w_m2"]["bias"] == pytest.approx(-40.0, abs=1e-9)
    assert entry["rnet_w_m2"]["bias"] == pytest.approx(0.0, abs=1e-9)
    assert entry["partition"]["model"]["closure_residual_w_m2"] == pytest.approx(0.0, abs=1e-9)
    assert entry["partition"]["model"]["bowen_ratio"] > entry["partition"]["gfs"]["bowen_ratio"]


def test_the_energy_balance_reads_a_planted_hole_in_the_ground_flux():
    state, flux = _reference()
    result = se.localize(_model(g=-50.0), state, flux)
    entry = result["regions"]["conus_land"]
    assert entry["g_w_m2"]["bias"] == pytest.approx(-50.0, abs=1e-9)
    assert entry["partition"]["model"]["closure_residual_w_m2"] == pytest.approx(50.0, abs=1e-9)


def test_gfs_ground_flux_with_the_wrong_sign_is_refused():
    state, flux = _reference()
    wrong = dict(flux)
    wrong["fields"] = dict(flux["fields"])
    wrong["fields"]["g"] = -flux["fields"]["g"]
    wrong["fields"]["gflux"] = -flux["fields"]["gflux"]
    with pytest.raises(ValueError, match="does not close"):
        se.localize(_model(), state, wrong)


def test_net_radiation_is_the_balance_noah_closes():
    tsk = np.array([300.0, 280.0])
    rn = se.net_radiation(np.array([600.0, 0.0]), np.array([350.0, 300.0]),
                          np.array([0.2, 0.2]), np.array([0.95, 0.95]), tsk)
    expected = np.array([600.0, 0.0]) * 0.8 + 0.95 * (np.array([350.0, 300.0]) - STEFAN_BOLTZMANN * tsk ** 4)
    assert np.allclose(rn, expected, rtol=0, atol=1e-9)


def test_height_interpolation_is_exact_on_a_linear_profile_and_refuses_above_the_stack():
    nlev, ny, nx = 12, 3, 4
    z = np.zeros((nlev, ny, nx))
    for k in range(nlev):
        z[nlev - 1 - k] = 20.0 + 30.0 * k + 0.5 * np.arange(nx)[None, :]   # 20, 50, 80.5 ... m
    t = 290.0 - 0.0065 * z + 0.01 * np.arange(ny)[None, :, None]
    for target in (80.0, 100.0, 5.0):
        out = se.interpolate_in_height(t, z, target)
        expected = 290.0 - 0.0065 * target + 0.01 * np.arange(ny)[:, None] + 0.0 * np.arange(nx)[None, :]
        assert np.allclose(out, expected, rtol=0, atol=1e-10), target
    with pytest.raises(ValueError, match="above the"):
        se.interpolate_in_height(t, z, 5000.0)


def test_hydrostatic_heights_match_the_isothermal_closed_form():
    nlev, ny, nx = 10, 2, 2
    ps = np.full((ny, nx), 100000.0)
    p_half = np.linspace(20000.0, 100000.0, nlev + 1)[:, None, None] * np.ones((1, ny, nx))
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    t = np.full((nlev, ny, nx), 270.0)
    qv = np.zeros_like(t)
    z = se.hydrostatic_full_level_heights(t, qv, p_half, p_full)
    expected = DRY_AIR_GAS_CONSTANT * 270.0 / GRAVITY_M_S2 * np.log(ps[None] / p_full)
    assert np.allclose(z, expected, rtol=1e-12, atol=1e-9)
    assert np.all(np.diff(z, axis=0) < 0.0)   # top first, decreasing to the surface


def test_dewpoint_inverts_the_saturation_law():
    td = np.array([275.0, 290.0, 300.0])
    p = np.array([100000.0, 95000.0, 101000.0])
    e = se._BOLTON_A * np.exp(se._BOLTON_B * (td - 273.15) / (td - 273.15 + se._BOLTON_C))
    q = se._EPSILON * e / (p - (1.0 - se._EPSILON) * e)
    assert np.allclose(se.dewpoint_from_specific_humidity(q, p), td, rtol=0, atol=1e-8)


def test_the_two_mapping_authorities_load_and_name_their_records():
    """Both products validate, including the masked vegetation record.

    The GFS vegetation record (VEG, discipline 2 category 0 parameter 4)
    carries a bitmap that is off over every water cell, so the mapping keeps
    the mask (`missing.kind = "preserve_mask"` on a SURFACE field) and every
    consumer masks it by name.  woof 2.7.0's own validator restricts that
    policy to soil fields and refused this product whole; the package reads
    every mapping through `mapped_source_compat.load_mapping`, which runs the
    engine's validator and adapts that one rule.
    """

    from woof.globe.analysis_initial import resolve_analysis_mapping
    from woof.globe.mapped_source_compat import load_mapping

    state = load_mapping(resolve_analysis_mapping(se.STATE_MAPPING_ID))
    flux = load_mapping(resolve_analysis_mapping(se.FLUX_MAPPING_ID))
    for short, name in se.STATE_FIELDS.items():
        assert name in state["fields"], name
    sel80 = state["fields"]["air_temperature_80m"]["selectors"][0]
    assert (sel80["level_type"], sel80["level_value"]) == (103, 80)
    assert state["fields"]["vegetation_fraction"]["units"]["scale"] == 0.01
    for short, name in se.FLUX_FIELDS.items():
        assert name in flux["fields"], name
        if name != "land_fraction":
            assert flux["fields"][name]["selectors"][0]["pdt"] == 8, name
    assert not state["target"]["require_lateral_boundaries"]
    assert not flux["target"]["require_lateral_boundaries"]


def test_bias_planes_are_named_for_the_renderer_and_masked_to_land():
    state, flux = _reference()
    model = _model(tsk=1.0)
    planes = se.bias_planes(model, state, flux)
    assert {"TSK_BIAS", "T2_BIAS", "HFX_BIAS", "RNET_BIAS", "TSK_MODEL", "T2_GFS"} <= set(planes)
    land = model.fields["land"] >= 0.5
    assert np.allclose(planes["TSK_BIAS"][land], 1.0, atol=1e-9)
    assert np.all(np.isnan(planes["TSK_BIAS"][~land]))
    assert np.all(np.isfinite(planes["TSK_MODEL"]))


@requires_netcdf_writer
def test_export_carries_the_surface_energy_planes_and_the_instrument_extras(tmp_path):
    """A native-suite checkpoint's energy books reach the render tape in
    WRF's names, LH split land/water, VEGFRA in percent, and the extra
    planes the surface-energy instrument hands the door are stored under
    their own names (the renderer's ``var:<name>`` products)."""
    from netCDF4 import Dataset

    from woof.globe.checkpoint import write_checkpoint
    from woof.globe.config import load_config
    from woof.globe.runner import build_model_and_cold_state
    from woof.globe.wrfout_export import EXPORT_RECEIPT_NAME, export_wrfout

    cfg = load_config(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
    model, state = build_model_and_cold_state(cfg)
    shape = model.transform.grid.shape
    arrays = state.physics_state.arrays
    plants = {"hfx": 120.0, "qfx": 4.0e-5, "noah_grdflx": -60.0, "swdown": 650.0,
              "glw": 340.0, "olr": 250.0, "ust": 0.35, "znt": 0.1, "pblh": 900.0,
              "noah_lh": 80.0}
    for name, value in plants.items():
        arrays[name] = np.full(shape, value, np.float32)
    checkpoint = write_checkpoint(
        tmp_path / "run" / "checkpoint.npz", state,
        config_hash=cfg.config_hash, to_numpy=model.transform.backend.to_numpy,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    seen = {}

    def extras(path, bundle, regrid):
        seen["called"] = True
        return {"T2_BIAS": np.full(shape, 1.5), "TSK_GFS": regrid(np.full(shape, 290.0))}

    tapes = export_wrfout(
        cfg, [checkpoint], tmp_path / "tapes", nlat=18, nlon=36,
        start_date="2026-08-30_18:00:00", extra_planes=extras,
    )
    assert seen["called"]
    with Dataset(tapes[0]) as ds:
        assert np.allclose(np.asarray(ds["HFX"][0]), 120.0)
        assert np.allclose(np.asarray(ds["GRDFLX"][0]), -60.0)
        assert np.allclose(np.asarray(ds["SWDOWN"][0]), 650.0)
        assert np.allclose(np.asarray(ds["GLW"][0]), 340.0)
        land = np.asarray(ds["LANDMASK"][0]) >= 0.5
        lh = np.asarray(ds["LH"][0])
        assert np.allclose(lh[land], 80.0)
        assert np.allclose(lh[~land], 2.5e6 * 4.0e-5, rtol=1e-5)
        vegfra = np.asarray(ds["VEGFRA"][0])
        assert np.all(vegfra >= 0.0) and np.all(vegfra <= 100.0)
        assert np.allclose(np.asarray(ds["T2_BIAS"][0]), 1.5)
        assert np.allclose(np.asarray(ds["TSK_GFS"][0]), 290.0)
    receipt = json.loads((tmp_path / "tapes" / EXPORT_RECEIPT_NAME).read_text(encoding="utf-8"))
    row = receipt["tapes"][0]
    assert "HFX" in row["surface_energy_fields"] and "LH" in row["surface_energy_fields"]
    assert row["extra_planes"] == ["T2_BIAS", "TSK_GFS"]


def test_missing_reference_cells_are_left_out_and_counted():
    """A reference plane decoded under preserve_mask (the GFS vegetation
    fraction, bitmap off over water) is NaN on those cells; the statistic
    is taken over the finite cells only and the count left out is reported,
    never a NaN bias and never a silent zero fill."""
    state, flux = _reference()
    poisoned = dict(state)
    poisoned["fields"] = dict(state["fields"])
    vegfra = state["fields"]["vegfra"].copy()
    lat = state["lat"]
    # Blank a latitude band inside every region's land mask.
    band = (lat >= 34.0) & (lat <= 38.0)
    vegfra[band, :] = np.nan
    poisoned["fields"]["vegfra"] = vegfra
    result = se.localize(_model(), poisoned, flux)
    clean = se.localize(_model(), state, flux)
    for region in se.REGIONS:
        entry = result["regions"][region]["vegfra_1"]
        assert np.isfinite(entry["bias"]) and np.isfinite(entry["rmse"])
        assert entry["bias"] == pytest.approx(0.0, abs=1e-9)
        assert entry["n_missing"] > 0
        assert entry["n"] + entry["n_missing"] == clean["regions"][region]["vegfra_1"]["n"]
        assert clean["regions"][region]["vegfra_1"]["n_missing"] == 0
        # The other rungs are untouched.
        assert result["regions"][region]["t2_k"]["n_missing"] == 0
        assert result["regions"][region]["t2_k"]["n"] == clean["regions"][region]["t2_k"]["n"]
        assert np.isfinite(result["regions"][region]["means"]["gfs"]["vegfra"])
    # An all-missing field refuses by name instead of reading zero.
    poisoned["fields"]["vegfra"] = np.full_like(vegfra, np.nan)
    with pytest.raises(ValueError, match="no finite cell"):
        se.localize(_model(), poisoned, flux)


def test_the_flux_ladder_reads_its_mapping_through_the_package_door(
        monkeypatch, tmp_path) -> None:
    """One door for every mapping document this package reads.

    `decode_gfs_flux` used to call `woof.mapped_source.load_mapping`
    itself, around the adaptation that lets a published engine accept a
    masked SURFACE record.  Four of the six carried mappings declare one;
    the flux document does not, so the only thing keeping this site from
    refusing was the content that document happens to have today, and the
    decode receipt would have said nothing about why.
    """

    from pathlib import Path

    from woof.globe import mapped_source_compat

    seen: list[object] = []

    def door(path, *, _raw=None):
        seen.append(path)
        raise RuntimeError("the package door")

    monkeypatch.setattr(mapped_source_compat, "load_mapping", door)
    with pytest.raises(RuntimeError, match="the package door"):
        se.decode_gfs_flux(tmp_path / "no-such.grib2")
    assert seen and Path(seen[0]).name.endswith(".mapping.json")


def test_flux_record_selectors_pick_the_interval_record_and_nothing_else():
    """The flux ladder reads pdt-8 records by selector through the GRIB2
    inventory: the interval record is picked over its instantaneous twin,
    every named octet must match, and an unnamed key is unconstrained."""
    from woof.globe.analysis_initial import resolve_analysis_mapping
    from woof.globe.mapped_source_compat import load_mapping

    mapping = load_mapping(resolve_analysis_mapping(se.FLUX_MAPPING_ID))
    shtfl = mapping["fields"]["sensible_heat_flux"]["selectors"][0]
    row = {"discipline": "0", "category": "0", "parameter": "11", "level_type": "1",
           "level_value": "0", "pdt": "8", "center": "7", "subcenter": "0",
           "master_table_version": "2", "local_table_version": "1"}
    assert se.record_matches_selector(shtfl, row)
    assert not se.record_matches_selector(shtfl, {**row, "pdt": "0"})
    assert not se.record_matches_selector(shtfl, {**row, "parameter": "10"})
    assert not se.record_matches_selector(shtfl, {**row, "level_value": "2"})
    gflux = mapping["fields"]["ground_heat_flux"]["selectors"][0]
    local = {**row, "discipline": "2", "parameter": "193"}
    assert se.record_matches_selector(gflux, local)
    assert not se.record_matches_selector(gflux, {**local, "center": "98"})
    lhtfl = mapping["fields"]["latent_heat_flux"]["selectors"][0]
    assert se.record_matches_selector(lhtfl, {**row, "parameter": "10", "center": "98"})


def test_the_vegetation_record_keeps_its_bitmap_mask():
    """The GFS VEG record's bitmap is off over water; the state mapping
    decodes it under preserve_mask (NaN where off) so the ladder can read
    the product at all, and the grammar admits the policy on a surface
    field, whichever validator is in front of it."""
    from woof.globe.analysis_initial import resolve_analysis_mapping
    from woof.globe.mapped_source_compat import load_mapping

    state = load_mapping(resolve_analysis_mapping(se.STATE_MAPPING_ID))
    assert state["fields"]["vegetation_fraction"]["missing"] == {"kind": "preserve_mask"}
    assert state["fields"]["vegetation_fraction"]["location"] == "surface"
    for name in ("skin_temperature", "air_temperature_2m", "air_temperature_80m"):
        assert state["fields"][name]["missing"] == {"kind": "reject"}, name


def test_a_latitude_patterned_plant_reads_its_cos_latitude_share_and_rmse():
    """Third family: a plant that is NOT uniform.  +4 K on the skin north of
    40N and nothing south of it.  A uniform plant cannot tell a weighted
    mean from an unweighted one; this one can.  The bias must be the
    cos(latitude)-weighted share of the plant, computed here from the
    analytic masks and the weights alone, the rmse the square root of that
    share times 4 K, and the unweighted mean must differ from both, so what
    is read is the weighting and not a coincidence of the pattern."""
    import math

    state, flux = _reference()
    model = _model()
    lat, lon = model.lat, model.lon
    plant = np.where(lat >= 40.0, 4.0, 0.0)
    model.fields["tsk"] = model.fields["tsk"] + plant
    result = se.localize(model, state, flux)
    # The masks from the analytic land rule (|lat| < 70 on both sides) and
    # the region boxes, not from the instrument.
    land = np.abs(lat) < 70.0
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    masks = {
        "conus_land": land & (lat >= 25) & (lat <= 50) & (lon180 >= -125) & (lon180 <= -65),
        "nh_midlat_land": land & (lat >= 30) & (lat <= 60),
        "global_land": land,
    }
    w = np.cos(np.deg2rad(lat))
    for region, mask in masks.items():
        entry = result["regions"][region]
        assert entry["n"] == int(mask.sum()), region
        share = float(np.sum(w[mask] * (plant[mask] > 0.0)) / np.sum(w[mask]))
        assert 0.0 < share < 1.0, region          # both halves of the plant present
        unweighted = float(np.mean(plant[mask] > 0.0))
        assert abs(unweighted - share) > 1.0e-3, region
        assert entry["tsk_k"]["bias"] == pytest.approx(4.0 * share, abs=1e-9), region
        assert entry["tsk_k"]["mae"] == pytest.approx(4.0 * share, abs=1e-9), region
        assert entry["tsk_k"]["rmse"] == pytest.approx(4.0 * math.sqrt(share), abs=1e-9), region
        assert entry["tsk_k"]["rmse"] > entry["tsk_k"]["bias"]
        d = entry["decomposition_k"]
        assert d["skin_d_tsk_minus_t2"] == pytest.approx(4.0 * share, abs=1e-9), region
        assert d["t2_bias"] == pytest.approx(0.0, abs=1e-9), region
        assert d["column_t80_bias"] == pytest.approx(0.0, abs=1e-9), region
        for key in ("t2_k", "t80_k", "hfx_w_m2", "rnet_w_m2"):
            assert entry[key]["bias"] == pytest.approx(0.0, abs=1e-9), (region, key)


def test_plants_on_the_reference_side_read_back_with_the_opposite_sign():
    """The other direction: the same plants on the GFS side (+2 K skin, +1 K
    at 80 m, +30 W/m2 sensible flux) read as their negatives in the same
    terms, the model's closure stays zero and the GFS closure carries the
    30 W/m2 exactly (inside the sign-convention tolerance, so the product
    is accepted and the residual is reported, not refused)."""
    state, flux = _reference()
    planted_state = dict(state)
    planted_state["fields"] = dict(state["fields"])
    planted_state["fields"]["tsk"] = state["fields"]["tsk"] + 2.0
    planted_state["fields"]["t80"] = state["fields"]["t80"] + 1.0
    planted_flux = dict(flux)
    planted_flux["fields"] = dict(flux["fields"])
    planted_flux["fields"]["hfx"] = flux["fields"]["hfx"] + 30.0
    result = se.localize(_model(), planted_state, planted_flux)
    for region in se.REGIONS:
        entry = result["regions"][region]
        assert entry["tsk_k"]["bias"] == pytest.approx(-2.0, abs=1e-9), region
        assert entry["tsk_k"]["rmse"] == pytest.approx(2.0, abs=1e-9), region
        assert entry["t80_k"]["bias"] == pytest.approx(-1.0, abs=1e-9), region
        assert entry["t2_k"]["bias"] == pytest.approx(0.0, abs=1e-9), region
        assert entry["t100_k"]["bias"] == pytest.approx(0.0, abs=1e-9), region
        assert entry["hfx_w_m2"]["bias"] == pytest.approx(-30.0, abs=1e-9), region
        assert entry["lh_w_m2"]["bias"] == pytest.approx(0.0, abs=1e-9), region
        assert entry["rnet_w_m2"]["bias"] == pytest.approx(0.0, abs=1e-9), region
        d = entry["decomposition_k"]
        assert d["column_t80_bias"] == pytest.approx(-1.0, abs=1e-9), region
        assert d["surface_layer_d_t2_minus_t80"] == pytest.approx(1.0, abs=1e-9), region
        assert d["skin_d_tsk_minus_t2"] == pytest.approx(-2.0, abs=1e-9), region
        assert entry["partition"]["model"]["closure_residual_w_m2"] == pytest.approx(0.0, abs=1e-9), region
        assert entry["partition"]["gfs"]["closure_residual_w_m2"] == pytest.approx(-30.0, abs=1e-9), region
    closure = result["gfs_closure"]["conus_land"]
    assert closure["residual_g_into_soil"] == pytest.approx(-30.0, abs=1e-9)


# ---------------------------------------------------------------------------
# the frozen footprints
# ---------------------------------------------------------------------------


def test_the_frozen_footprints_exist_and_identical_sides_read_zero_on_them():
    state, flux = _reference()
    result = se.localize(_model(), state, flux)
    regions = result["regions"]
    for region in se.FROZEN_REGIONS:
        assert region in regions, region
    populated = [r for r in se.FROZEN_REGIONS if not regions[r].get("empty")]
    for region in ("sea_ice_south", "sea_ice_south_partial", "sea_ice_south_full", "sea_ice_north",
                   "antarctic_land_ice", "greenland_ice", "snow_covered_land", "nh_snow_covered_land"):
        assert region in populated, region
    # No terrain above 2000 m in the analytic planet: the interior is reported empty, not averaged.
    assert regions["antarctic_interior"] == {"n": 0, "empty": True}
    # The analytic pack is 0.9 north of 75N: no partial cells there.
    assert regions["sea_ice_north_partial"]["empty"] and not regions["sea_ice_north_full"].get("empty")
    for region in populated:
        entry = regions[region]
        for key in ("tsk_k", "t2_k", "wspd10_m_s", "seaice_1", "hfx_w_m2", "rnet_w_m2"):
            assert entry[key]["bias"] == pytest.approx(0.0, abs=1e-9), (region, key)
        r = entry["roughness"]
        assert r["n"] == entry["n"] and r["n_left_out"] == 0
        assert r["log_law_wind_effect_m_s"] == pytest.approx(0.0, abs=1e-12), region
        assert r["ln_z0_model_over_gfs"] == pytest.approx(0.0, abs=1e-12), region
        assert r["wspd10_bias_at_gfs_roughness_m_s"] == pytest.approx(0.0, abs=1e-9), region
    # The footprints are the reference's own cells, so the two bins tile the pack.
    south = regions["sea_ice_south"]["n"]
    assert regions["sea_ice_south_partial"]["n"] + regions["sea_ice_south_full"]["n"] == south
    assert regions["sea_ice_south_partial"]["means"]["gfs"]["seaice"] == pytest.approx(0.6)
    assert regions["sea_ice_south_full"]["means"]["gfs"]["seaice"] == pytest.approx(1.0)
    # The three land regions are untouched by the additions (their masks and numbers).
    for region in se.REGIONS:
        assert regions[region]["n"] > 0 and regions[region]["tsk_k"]["bias"] == pytest.approx(0.0, abs=1e-9)


def test_a_skin_offset_planted_on_the_pack_reads_in_the_pack_skin_term_only():
    """A +2 K skin planted on the model's sea-ice columns (and nowhere
    else) is read as +2 K TSK bias and +2 K skin excess on every pack
    footprint, the 2 m and column terms zero there, and nothing on the
    ice sheets, the snow-covered land or the three land regions."""
    state, flux = _reference()
    model = _model()
    pack = model.fields["seaice"] >= se.SEA_ICE_THRESHOLD
    model.fields["tsk"] = model.fields["tsk"] + np.where(pack, 2.0, 0.0)
    result = se.localize(model, state, flux)
    regions = result["regions"]
    for region in ("sea_ice_south", "sea_ice_south_partial", "sea_ice_south_full", "sea_ice_north", "sea_ice_north_full"):
        d = regions[region]["decomposition_k"]
        assert d["tsk_bias"] == pytest.approx(2.0, abs=1e-9), region
        assert d["skin_d_tsk_minus_t2"] == pytest.approx(2.0, abs=1e-9), region
        assert d["t2_bias"] == pytest.approx(0.0, abs=1e-9), region
        assert d["column_t80_bias"] == pytest.approx(0.0, abs=1e-9), region
        assert regions[region]["tsk_k"]["rmse"] == pytest.approx(2.0, abs=1e-9), region
    for region in ("antarctic_land_ice", "greenland_ice", "snow_covered_land", "nh_snow_covered_land") + se.REGIONS:
        assert regions[region]["tsk_k"]["bias"] == pytest.approx(0.0, abs=1e-9), region
        assert regions[region]["decomposition_k"]["skin_d_tsk_minus_t2"] == pytest.approx(0.0, abs=1e-9), region


def test_a_roughness_planted_on_the_model_reads_in_the_wind_by_the_log_law():
    """Ten times the roughness on the model's ice-sheet columns, with the
    model's 10 m wind re-drawn from the same first-level wind by the
    neutral log law: the wind bias on the ice-sheet footprints equals the
    log-law share to rounding, the wind at the reference roughness reads
    zero, ln(z0 ratio) reads ln 10, and the pack and the land regions
    outside the plant read nothing."""
    state, flux = _reference()
    model = _model()
    cells = model.fields["landice"]
    z0_new = np.where(cells, 10.0 * model.fields["z0"], model.fields["z0"])
    _rediagnose_wind(model.fields, cells, model.fields["z0"], z0_new)
    model.fields["z0"] = z0_new
    result = se.localize(model, state, flux)
    regions = result["regions"]
    for region in ("antarctic_land_ice", "greenland_ice"):
        r = regions[region]["roughness"]
        assert r["ln_z0_model_over_gfs"] == pytest.approx(np.log(10.0), abs=1e-12), region
        assert r["log_law_wind_effect_m_s"] < -0.05, region       # a rougher surface reads a slower 10 m wind
        assert r["wspd10_bias_m_s"] == pytest.approx(r["log_law_wind_effect_m_s"], abs=1e-9), region
        assert r["wspd10_bias_at_gfs_roughness_m_s"] == pytest.approx(0.0, abs=1e-9), region
        assert regions[region]["wspd10_m_s"]["bias"] == pytest.approx(r["wspd10_bias_m_s"], abs=1e-9), region
        assert regions[region]["tsk_k"]["bias"] == pytest.approx(0.0, abs=1e-9), region
    # Footprints disjoint from the plant (the 60N row of the Greenland box sits inside nh_midlat_land).
    for region in ("sea_ice_south", "sea_ice_north", "conus_land"):
        r = regions[region]["roughness"]
        assert r["ln_z0_model_over_gfs"] == pytest.approx(0.0, abs=1e-12), region
        assert r["log_law_wind_effect_m_s"] == pytest.approx(0.0, abs=1e-12), region
        assert regions[region]["wspd10_m_s"]["bias"] == pytest.approx(0.0, abs=1e-9), region
    # global_land holds the plant's share only: its wind bias is the area share of the ice sheets' effect.
    assert regions["global_land"]["wspd10_m_s"]["bias"] < 0.0
    assert regions["global_land"]["wspd10_m_s"]["bias"] > regions["antarctic_land_ice"]["wspd10_m_s"]["bias"]


def test_a_roughness_planted_on_the_reference_reads_back_with_the_opposite_sign():
    """The other direction: ten times the roughness on the reference's pack
    cells with the reference's 10 m wind re-drawn by the same law reads a
    positive wind bias on the pack equal to the log-law share, ln(z0 ratio)
    reads -ln 10, and the wind at the reference roughness reads zero."""
    state, flux = _reference()
    planted = dict(state)
    planted["fields"] = dict(state["fields"])
    lat, lon = state["lat"], state["lon"]
    lon2d, lat2d = np.meshgrid(lon, lat)
    cells = planted["fields"]["seaice"] >= se.SEA_ICE_THRESHOLD
    planted["fields"]["z0"] = np.where(cells, 10.0 * state["fields"]["z0"], state["fields"]["z0"])
    _rediagnose_wind(planted["fields"], cells, state["fields"]["z0"], planted["fields"]["z0"])
    result = se.localize(_model(), planted, flux)
    regions = result["regions"]
    for region in ("sea_ice_south", "sea_ice_south_partial", "sea_ice_south_full", "sea_ice_north"):
        r = regions[region]["roughness"]
        assert r["ln_z0_model_over_gfs"] == pytest.approx(-np.log(10.0), abs=1e-12), region
        assert r["log_law_wind_effect_m_s"] > 0.05, region
        assert r["wspd10_bias_m_s"] == pytest.approx(r["log_law_wind_effect_m_s"], abs=1e-9), region
        assert r["wspd10_bias_at_gfs_roughness_m_s"] == pytest.approx(0.0, abs=1e-9), region
    for region in ("antarctic_land_ice", "greenland_ice", "snow_covered_land") + se.REGIONS:
        assert regions[region]["roughness"]["log_law_wind_effect_m_s"] == pytest.approx(0.0, abs=1e-12), region
        assert regions[region]["wspd10_m_s"]["bias"] == pytest.approx(0.0, abs=1e-9), region


def test_the_log_law_is_exact_on_its_closed_form_and_a_degenerate_roughness_is_left_out():
    wspd1 = np.array([10.0, 10.0])
    z1 = np.array([23.0, 23.0])
    assert se.log_law_wind(wspd1, z1, np.array([0.001, 0.01]))[0] == pytest.approx(10.0 * np.log(1.0e4) / np.log(2.3e4))
    assert se.log_law_wind(wspd1, z1, np.array([0.001, 0.01]))[1] == pytest.approx(10.0 * np.log(1.0e3) / np.log(2.3e3))
    m = {"z0": np.array([0.001, 0.0, 0.5]), "z1": np.array([23.0, 23.0, 23.0]),
         "wspd1": np.array([10.0, 10.0, 10.0]), "wspd10": np.array([9.0, 9.0, 9.0])}
    ref = {"z0": np.array([0.01, 0.01, 30.0]), "wspd10": np.array([8.0, 8.0, 8.0])}
    r = se.roughness_reading(m, ref, np.array([True, True, True]), np.ones(3))
    assert r["n"] == 1 and r["n_left_out"] == 2          # a zero roughness and one above z1 are left out
    assert r["ln_z0_model_over_gfs"] == pytest.approx(np.log(0.1))
    assert r["log_law_wind_effect_m_s"] == pytest.approx(
        10.0 * np.log(1.0e4) / np.log(2.3e4) - 10.0 * np.log(1.0e3) / np.log(2.3e3))
    assert r["wspd10_bias_m_s"] == pytest.approx(1.0)


def test_the_snow_bitmap_means_no_snow_on_open_water_only():
    snow = np.array([np.nan, 5.0, np.nan, np.nan])
    land = np.array([0.0, 1.0, 1.0, 0.0])
    ice = np.array([0.0, 0.0, 0.0, 0.7])
    with pytest.raises(ValueError, match="masked on 2 land or sea-ice points"):
        se.snow_with_bitmap_policy(snow, land, ice, "product")
    out = se.snow_with_bitmap_policy(np.array([np.nan, 5.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0]), "product")
    assert out.tolist() == [0.0, 5.0]


def test_the_skin_conduction_is_the_column_step_own_conductance():
    """G read from a checkpoint is g01 (T0 - T1) with the step's own half
    thicknesses over the media at the node midpoints: on 0.3 m of snow
    over 1.5 m of ice the top two nodes are snow, and the reading equals
    the closed form."""
    from woof.globe.physics import frozen_surface as fs

    shape = (1, 1)
    dz = fs.sea_ice_layer_thickness(np.full(shape, 1.5, np.float32), np.full(shape, 0.3, np.float32), np)
    layers = np.stack([np.full(shape, 250.0), np.full(shape, 254.0), np.full(shape, 262.0), np.full(shape, 270.0)]).astype(np.float32)
    g = fs.skin_conduction_w_m2(layers, dz, np.full(shape, 0.3, np.float32), np)
    k0 = fs.SNOW_CONDUCTIVITY_W_M_K
    mid1 = float(dz[0, 0, 0]) + 0.5 * float(dz[1, 0, 0])
    k1 = fs.SNOW_CONDUCTIVITY_W_M_K if mid1 < 0.3 else fs.ICE_CONDUCTIVITY_W_M_K
    g01 = 1.0 / (0.5 * float(dz[0, 0, 0]) / k0 + 0.5 * float(dz[1, 0, 0]) / k1)
    assert float(g[0, 0]) == pytest.approx(g01 * (250.0 - 254.0), rel=1e-5)
    assert float(g[0, 0]) < 0.0      # a skin colder than the node below draws heat upward: G into the column is negative


def test_a_reference_record_masked_over_the_whole_pack_is_reported_missing_not_refused():
    """The GFS vegetation and soil records carry no value over water: on a
    pack footprint every cell of those rungs is masked.  The footprint
    reports the rung missing with the count and keeps its other rungs; a
    land region with the same hole still refuses (its numbers are the
    skin lane's contract)."""
    state, flux = _reference()
    poisoned = dict(state)
    poisoned["fields"] = dict(state["fields"])
    pack = state["fields"]["seaice"] >= se.SEA_ICE_THRESHOLD
    poisoned["fields"]["vegfra"] = np.where(pack, np.nan, state["fields"]["vegfra"])
    result = se.localize(_model(), poisoned, flux)
    entry = result["regions"]["sea_ice_south"]
    assert entry["vegfra_1"]["missing"] and entry["vegfra_1"]["n"] == 0
    assert entry["vegfra_1"]["n_missing"] == entry["n"]
    assert entry["means"]["gfs"]["vegfra"] is None
    assert entry["tsk_k"]["bias"] == pytest.approx(0.0, abs=1e-9)
    assert entry["roughness"]["n"] == entry["n"]
    assert "MISSING" in se.report({**result, "label": "x", "step": 0, "time_s": 0.0,
                                  "verdict": se.verdict(result["regions"])})
    everywhere = dict(state)
    everywhere["fields"] = dict(state["fields"])
    everywhere["fields"]["vegfra"] = np.full_like(state["fields"]["vegfra"], np.nan)
    with pytest.raises(ValueError, match="no finite cell"):
        se.localize(_model(), everywhere, flux)


# ---------------------------------------------------------------------------
# a third synthetic family: mixed plants on the frozen footprints, both sides
# ---------------------------------------------------------------------------


def _pack_shares(model):
    """cos(lat)-weighted shares of the southern pack's partial and full
    bins, from the model sample's own planes (the reference's pack on the
    model grid is the same plane: the model nodes are reference nodes)."""
    weights = np.cos(np.deg2rad(model.lat))
    south = model.lat < 0.0
    partial = (model.fields["seaice"] >= se.SEA_ICE_THRESHOLD) & (model.fields["seaice"] < se.PARTIAL_PACK_FRACTION) & south
    full = (model.fields["seaice"] >= se.PARTIAL_PACK_FRACTION) & south
    w_p, w_f = float(weights[partial].sum()), float(weights[full].sum())
    return partial, full, w_p / (w_p + w_f)


def test_a_skin_plant_on_the_partial_pack_alone_reads_in_its_bin_and_the_pack_reads_its_area_share():
    """+3 K on the model's skin on the southern partial pack only: the
    partial bin reads +3.000 (rmse 3.000), the full bin 0, and the whole
    southern pack the cos(lat)-weighted share of the partial bin times
    3 K in the bias and 3 K times the square root of that share in the
    rmse, computed here from the masks and weights alone; the northern
    pack, the ice sheets and the land regions read 0."""
    state, flux = _reference()
    model = _model()
    partial, full, share = _pack_shares(model)
    assert 0.0 < share < 1.0
    model.fields["tsk"] = model.fields["tsk"] + np.where(partial, 3.0, 0.0)
    regions = se.localize(model, state, flux)["regions"]
    assert regions["sea_ice_south_partial"]["tsk_k"]["bias"] == pytest.approx(3.0, abs=1e-9)
    assert regions["sea_ice_south_partial"]["tsk_k"]["rmse"] == pytest.approx(3.0, abs=1e-9)
    assert regions["sea_ice_south_partial"]["decomposition_k"]["skin_d_tsk_minus_t2"] == pytest.approx(3.0, abs=1e-9)
    assert regions["sea_ice_south_full"]["tsk_k"]["bias"] == pytest.approx(0.0, abs=1e-9)
    assert regions["sea_ice_south"]["tsk_k"]["bias"] == pytest.approx(3.0 * share, abs=1e-9)
    assert regions["sea_ice_south"]["tsk_k"]["rmse"] == pytest.approx(3.0 * np.sqrt(share), abs=1e-9)
    assert regions["sea_ice_south_partial"]["n"] + regions["sea_ice_south_full"]["n"] == regions["sea_ice_south"]["n"]
    for region in ("sea_ice_north", "antarctic_land_ice", "greenland_ice", "snow_covered_land") + se.REGIONS:
        assert regions[region]["tsk_k"]["bias"] == pytest.approx(0.0, abs=1e-9), region


def test_a_wind_excess_beyond_the_log_law_reads_in_the_wind_at_the_reference_roughness_on_both_sides():
    """Roughness times ten on the model's ice sheets with the wind re-drawn
    by the log law AND a further +0.7 m/s planted on the same cells: the
    wind bias is the log-law share plus 0.7, and the wind at the reference
    roughness reads exactly the 0.7 (the decomposition is additive, which
    the pure log-law plants cannot show).  The other direction: the same
    on the reference's pack with a -0.5 m/s excess reads +0.5 at the
    reference roughness on the pack bins."""
    state, flux = _reference()
    model = _model()
    cells = model.fields["landice"]
    z0_new = np.where(cells, 10.0 * model.fields["z0"], model.fields["z0"])
    _rediagnose_wind(model.fields, cells, model.fields["z0"], z0_new)
    model.fields["z0"] = z0_new
    speed = np.hypot(model.fields["u10"], model.fields["v10"])
    scale = np.where(cells, (speed + 0.7) / speed, 1.0)
    model.fields["u10"] = model.fields["u10"] * scale
    model.fields["v10"] = model.fields["v10"] * scale
    model.fields["wspd10"] = np.hypot(model.fields["u10"], model.fields["v10"])
    regions = se.localize(model, state, flux)["regions"]
    for region in ("antarctic_land_ice", "greenland_ice"):
        r = regions[region]["roughness"]
        assert r["ln_z0_model_over_gfs"] == pytest.approx(np.log(10.0), abs=1e-12), region
        assert r["wspd10_bias_at_gfs_roughness_m_s"] == pytest.approx(0.7, abs=1e-9), region
        assert r["wspd10_bias_m_s"] == pytest.approx(r["log_law_wind_effect_m_s"] + 0.7, abs=1e-9), region
        assert regions[region]["wspd10_m_s"]["bias"] == pytest.approx(r["wspd10_bias_m_s"], abs=1e-9), region
    for region in ("sea_ice_south", "sea_ice_north", "conus_land"):
        assert regions[region]["roughness"]["wspd10_bias_at_gfs_roughness_m_s"] == pytest.approx(0.0, abs=1e-9), region
    # the reference side
    planted = dict(state)
    planted["fields"] = dict(state["fields"])
    pack = planted["fields"]["seaice"] >= se.SEA_ICE_THRESHOLD
    planted["fields"]["z0"] = np.where(pack, 10.0 * state["fields"]["z0"], state["fields"]["z0"])
    _rediagnose_wind(planted["fields"], pack, state["fields"]["z0"], planted["fields"]["z0"])
    speed = np.hypot(planted["fields"]["u10"], planted["fields"]["v10"])
    scale = np.where(pack, (speed - 0.5) / speed, 1.0)
    planted["fields"]["u10"] = planted["fields"]["u10"] * scale
    planted["fields"]["v10"] = planted["fields"]["v10"] * scale
    regions = se.localize(_model(), planted, flux)["regions"]
    for region in ("sea_ice_south", "sea_ice_south_partial", "sea_ice_south_full", "sea_ice_north"):
        r = regions[region]["roughness"]
        assert r["ln_z0_model_over_gfs"] == pytest.approx(-np.log(10.0), abs=1e-12), region
        assert r["wspd10_bias_at_gfs_roughness_m_s"] == pytest.approx(0.5, abs=1e-9), region
        assert r["wspd10_bias_m_s"] == pytest.approx(r["log_law_wind_effect_m_s"] + 0.5, abs=1e-9), region
    for region in ("antarctic_land_ice", "greenland_ice", "snow_covered_land") + se.REGIONS:
        assert regions[region]["roughness"]["wspd10_bias_at_gfs_roughness_m_s"] == pytest.approx(0.0, abs=1e-9), region


def test_the_pack_footprints_are_the_reference_cells_whatever_pack_the_model_carries():
    """A model whose sea-ice plane is zero everywhere (an ice-free arm) is
    scored on the same pack cells as one carrying the analysed pack: the
    footprint counts are unchanged and the sea-ice rung reads the
    reference's fraction as the bias."""
    state, flux = _reference()
    with_pack = se.localize(_model(), state, flux)["regions"]
    model = _model()
    model.fields["seaice"] = np.zeros_like(model.fields["seaice"])
    without = se.localize(model, state, flux)["regions"]
    for region in se.FROZEN_REGIONS:
        assert without[region]["n"] == with_pack[region]["n"], region
    assert without["sea_ice_south_partial"]["seaice_1"]["bias"] == pytest.approx(-0.6, abs=1e-9)
    assert without["sea_ice_south_full"]["seaice_1"]["bias"] == pytest.approx(-1.0, abs=1e-9)
    assert without["sea_ice_north"]["seaice_1"]["bias"] == pytest.approx(-0.9, abs=1e-9)
    assert with_pack["sea_ice_south"]["seaice_1"]["bias"] == pytest.approx(0.0, abs=1e-9)
