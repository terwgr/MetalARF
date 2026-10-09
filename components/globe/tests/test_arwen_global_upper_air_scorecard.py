"""The upper-air scorecard instrument: calibration families in both
directions (column interpolation and hydrostatics against closed forms,
planted score differences, the reference against itself, a planted
anomaly rotation, below-ground refusals in both directions, the analysis
regrid) and the checkpoint reader against a state whose fields are
exactly representable at the run's truncation.

Every planted number is read back to float64 rounding; the bars are the
ones the module's CALIBRATION table records.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.globe import upper_air_scorecard as sc

ROUNDOFF_M = 1.0e-9
ROUNDOFF = 1.0e-12


@pytest.fixture(scope="module")
def grid():
    return sc.synthetic_grid(21)


# -- family A: column interpolation and hydrostatics -------------------------


@pytest.mark.parametrize("lapse", [0.0, 30.0, -30.0])
def test_column_interpolation_reads_the_closed_form_at_every_target(grid, lapse):
    columns, truth = sc.synthetic_columns(grid, lapse_k_per_lnp=lapse)
    plf = sc.interpolate_column_fields(**columns)
    for level in sc.LEVELS_PA:
        exact = truth(level)
        inside = plf.valid[level]
        assert np.array_equal(inside, exact["inside"])
        assert inside.all(), "no synthetic column is refused at these targets"
        assert np.max(np.abs(plf.fields["z"][level] - exact["z"])) < ROUNDOFF_M
        assert np.max(np.abs(plf.fields["t"][level] - exact["t"])) < ROUNDOFF
        assert np.max(np.abs(plf.fields["u"][level] - exact["u"])) < ROUNDOFF
        assert np.max(np.abs(plf.fields["v"][level] - exact["v"])) < ROUNDOFF
        assert np.max(np.abs(plf.fields["rh"][level] - exact["rh"])) < 1.0e-10


def test_a_target_between_the_lowest_level_and_the_ground_integrates_from_the_ground(grid):
    columns, truth = sc.synthetic_columns(grid)
    ps = columns["surface_pressure"]
    p_low = columns["p_full"][-1]
    target = float(np.min(0.5 * (ps + p_low)))  # inside every column's bottom sliver or below the lowest level
    plf = sc.interpolate_column_fields(**columns, levels_pa=(target,))
    inside = plf.valid[target]
    assert inside.all()
    exact = truth(target)
    # Columns whose lowest full level lies above the target take the held
    # bottom virtual temperature for the hydrostatic step from the ground;
    # the closed form continues Tv linearly through the 275 Pa sliver, so
    # the held value reads 1e-5 m off it there (the stated approximation).
    assert np.max(np.abs(plf.fields["z"][target] - exact["z"])) < 1.0e-4
    assert np.max(np.abs(plf.fields["t"][target] - exact["t"])) < ROUNDOFF


def test_refusal_below_ground_and_above_top(grid):
    columns, _ = sc.synthetic_columns(grid)
    ps = columns["surface_pressure"]
    above_top = 50.0
    below_ground = float(np.max(ps)) + 10.0
    plf = sc.interpolate_column_fields(**columns, levels_pa=(above_top, below_ground, 50_000.0))
    assert not plf.valid[above_top].any()
    assert not plf.valid[below_ground].any()
    assert plf.valid[50_000.0].all()
    assert np.isnan(plf.fields["z"][below_ground]).all()


# -- family B: planted differences -------------------------------------------


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("target,kind,level,magnitude", [
    ("z500", "z", 50_000.0, 12.0), ("t850", "t", 85_000.0, 1.5), ("rh700", "rh", 70_000.0, 8.0),
])
def test_planted_scalar_offset_reads_back_as_bias_and_rmse(grid, sign, target, kind, level, magnitude):
    reference = sc.synthetic_reference(grid, seed=1)
    model = sc.shifted_copy(reference, {(kind, level): sign * magnitude})
    scores = sc.score_pair(model, reference)[target]["regions"]
    for region, row in scores.items():
        assert row["bias"] == pytest.approx(sign * magnitude, abs=ROUNDOFF_M)
        assert row["rmse"] == pytest.approx(magnitude, abs=ROUNDOFF_M)
        assert row["mae"] == pytest.approx(magnitude, abs=ROUNDOFF_M)
        assert row["masked_fraction"] == pytest.approx(0.0, abs=ROUNDOFF)
        assert row["n"] > 0
    # The other targets are untouched.
    others = sc.score_pair(model, reference)
    for name in others:
        if name == target:
            continue
        for row in others[name]["regions"].values():
            assert abs(row.get("bias", row.get("rmsve"))) < ROUNDOFF


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("target,level", [("w250", 25_000.0), ("w850", 85_000.0)])
def test_planted_vector_offset_reads_back_as_rmsve_and_component_bias(grid, sign, target, level):
    reference = sc.synthetic_reference(grid, seed=2)
    du, dv = sign * 3.0, -sign * 4.0
    model = sc.shifted_copy(reference, {("u", level): du, ("v", level): dv})
    scores = sc.score_pair(model, reference)[target]["regions"]
    for row in scores.values():
        assert row["rmsve"] == pytest.approx(5.0, abs=ROUNDOFF_M)
        assert row["u_bias"] == pytest.approx(du, abs=ROUNDOFF_M)
        assert row["v_bias"] == pytest.approx(dv, abs=ROUNDOFF_M)


def test_planted_texture_adds_in_quadrature(grid):
    reference = sc.synthetic_reference(grid, seed=3)
    rng = np.random.default_rng(5)
    texture = rng.standard_normal(grid.shape)
    texture = texture - grid.global_mean(texture)
    texture = 5.0 * texture / grid.rms(texture)
    model = sc.shifted_copy(reference, {("z", 50_000.0): 12.0 + texture})
    g = sc.score_pair(model, reference)["z500"]["regions"]["global"]
    assert g["bias"] == pytest.approx(12.0, abs=ROUNDOFF_M)
    assert g["rmse"] == pytest.approx(13.0, abs=ROUNDOFF_M)


# -- family C: the reference against itself ----------------------------------


def test_reference_scored_against_itself_reads_zero_and_ac_one(grid):
    reference = sc.synthetic_reference(grid, seed=4)
    scores = sc.score_pair(reference, reference)
    for target in scores.values():
        for row in target["regions"].values():
            if "rmsve" in row:
                assert row["rmsve"] == 0.0 and row["speed_bias"] == 0.0
            else:
                assert row["bias"] == 0.0 and row["rmse"] == 0.0 and row["mae"] == 0.0
                assert row["anomaly_correlation"] == pytest.approx(1.0, abs=ROUNDOFF)


# -- family D: the anomaly correlation of a planted rotation -----------------


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 60.0, 90.0, 120.0, 180.0])
def test_anomaly_correlation_reads_the_planted_rotation(grid, theta_deg):
    reference = sc.synthetic_reference(grid, seed=6)
    lat2, lon2 = grid.mesh()
    eddy = 60.0 * np.cos(3.0 * lon2) * np.cos(lat2)
    quadrature = 60.0 * np.sin(3.0 * lon2) * np.cos(lat2)
    base = reference.fields["z"][50_000.0]
    zonal = sc.zonal_mean_climatology(base, np.ones(base.shape, dtype=bool))
    ref_d = sc.shifted_copy(reference, {("z", 50_000.0): (zonal + eddy) - base})
    th = math.radians(theta_deg)
    model = sc.shifted_copy(ref_d, {("z", 50_000.0): (math.cos(th) - 1.0) * eddy + math.sin(th) * quadrature})
    for region, row in sc.score_pair(model, ref_d)["z500"]["regions"].items():
        assert row["anomaly_correlation"] == pytest.approx(math.cos(th), abs=ROUNDOFF), region


# -- family E: refusals in both directions -----------------------------------


@pytest.mark.parametrize("direction", ["model", "reference"])
def test_a_mountain_masks_the_level_and_a_planted_offset_inside_it_scores_nothing(grid, direction):
    lat2, lon2 = grid.mesh()
    mountain = (np.abs(lat2) < math.radians(15.0)) & (np.cos(lon2) > 0.5)
    columns, _ = sc.synthetic_columns(grid, mountain_pa=40_000.0, mountain_mask=mountain)
    columns_flat, _ = sc.synthetic_columns(grid)
    hilly = sc.interpolate_column_fields(**columns)
    flat = sc.interpolate_column_fields(**columns_flat)
    spoiled = sc.shifted_copy(flat, {("t", 85_000.0): np.where(mountain, 50.0, 0.0), ("rh", 70_000.0): np.where(mountain, 30.0, 0.0)})
    model, ref = (hilly, spoiled) if direction == "model" else (spoiled, hilly)
    scores = sc.score_pair(model, ref)
    planted = float(np.sum(grid.quadrature_weights[:, None] * mountain) / (2.0 * grid.nlon))
    assert planted > 0.02
    for target in ("t850", "w850", "rh700"):
        assert scores[target]["regions"]["global"]["masked_fraction"] == pytest.approx(planted, abs=ROUNDOFF)
    for target in ("z500", "w250"):
        assert scores[target]["regions"]["global"]["masked_fraction"] == pytest.approx(0.0, abs=ROUNDOFF)
    assert scores["t850"]["regions"]["global"]["bias"] == pytest.approx(0.0, abs=ROUNDOFF)
    assert scores["rh700"]["regions"]["global"]["bias"] == pytest.approx(0.0, abs=ROUNDOFF)
    assert scores["t850"]["regions"]["sh_extratropics"]["masked_fraction"] == pytest.approx(0.0, abs=ROUNDOFF)


# -- family F: the analysis regrid -------------------------------------------


@pytest.mark.parametrize("ascending", [False, True])
def test_reference_regrid_reads_a_bilinear_field_exactly(grid, ascending):
    lat_src = np.arange(90.0, -90.001, -1.0)
    lon_src = np.arange(0.0, 360.0, 1.0)
    lon_s, lat_s = np.meshgrid(lon_src, lat_src)
    linear = 2.0 * lat_s + 0.5 * np.where(lon_s > 180.0, 360.0 - lon_s, lon_s)
    if ascending:
        lat_src, linear = lat_src[::-1], linear[::-1]
    levels = np.asarray(sc.LEVELS_PA)
    stack = np.repeat(linear[None], levels.size, axis=0)
    frame = sc.SyntheticFrame(lat_src, lon_src, levels, {
        "geopotential_height": stack, "air_temperature": 250.0 + stack / 10.0,
        "specific_humidity": np.full(stack.shape, 0.001),
        "eastward_wind": stack / 100.0, "northward_wind": -stack / 100.0,
        "surface_pressure": np.full(linear.shape, 101_000.0), "terrain_height": linear,
    })
    plf = sc.reference_fields(frame, grid)
    glat, glon = grid.mesh()
    glat_deg, glon_deg = np.degrees(glat), np.degrees(glon)
    expected = 2.0 * glat_deg + 0.5 * np.where(glon_deg > 180.0, 360.0 - glon_deg, glon_deg)
    for level in sc.LEVELS_PA:
        assert np.max(np.abs(plf.fields["z"][level] - expected)) < ROUNDOFF
        assert np.max(np.abs(plf.fields["t"][level] - 250.0 - expected / 10.0)) < ROUNDOFF
        assert np.max(np.abs(plf.fields["u"][level] - expected / 100.0)) < ROUNDOFF
        assert plf.valid[level].all()
    assert plf.source["kind"] == "analysis"


def test_reference_refuses_a_missing_target_level(grid):
    lat_src = np.arange(90.0, -90.001, -1.0)
    lon_src = np.arange(0.0, 360.0, 1.0)
    shape = (lat_src.size, lon_src.size)
    levels = np.asarray([25_000.0, 50_000.0, 85_000.0])
    stack = np.zeros((levels.size, *shape))
    frame = sc.SyntheticFrame(lat_src, lon_src, levels, {
        "geopotential_height": stack, "air_temperature": stack + 250.0, "specific_humidity": stack + 0.001,
        "eastward_wind": stack, "northward_wind": stack,
        "surface_pressure": np.full(shape, 101_000.0), "terrain_height": np.zeros(shape),
    })
    with pytest.raises(ValueError, match="carries no 70000 Pa level"):
        sc.reference_fields(frame, grid)


def test_reference_masks_below_its_own_surface(grid):
    lat_src = np.arange(90.0, -90.001, -1.0)
    lon_src = np.arange(0.0, 360.0, 1.0)
    lon_s, lat_s = np.meshgrid(lon_src, lat_src)
    shape = lat_s.shape
    levels = np.asarray(sc.LEVELS_PA)
    stack = np.zeros((levels.size, *shape))
    high = np.abs(lat_s) < 10.0
    frame = sc.SyntheticFrame(lat_src, lon_src, levels, {
        "geopotential_height": stack, "air_temperature": stack + 250.0, "specific_humidity": stack + 0.001,
        "eastward_wind": stack, "northward_wind": stack,
        "surface_pressure": np.where(high, 60_000.0, 101_000.0), "terrain_height": np.zeros(shape),
    })
    plf = sc.reference_fields(frame, grid)
    lat2, _ = grid.mesh()
    inside_high = np.abs(np.degrees(lat2)) < 9.0  # rows clear of the bilinear ramp
    assert not plf.valid[85_000.0][inside_high].any()
    assert not plf.valid[70_000.0][inside_high].any()
    assert plf.valid[50_000.0].all()


# -- family G: the checkpoint reader -----------------------------------------


def test_checkpoint_reader_reads_an_exactly_representable_state(tmp_path):
    """An isothermal solid-body-rotation state at T3 (degree-1 wind,
    degree-0 theta and log ps) is exact in the spectral basis, so the
    reader must reproduce it and the hydrostatic height to rounding."""
    from woof.globe.checkpoint import read_checkpoint, write_checkpoint
    from woof.globe.config import load_config
    from woof.globe.constants import GRAVITY_M_S2, KAPPA, REFERENCE_PRESSURE_PA
    from woof.globe.runner import build_model_and_cold_state, build_transform
    from woof.globe.spectral.vector import VorticityDivergenceOperator

    cfg = load_config(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform)
    grid = transform.grid
    lat2, _ = grid.mesh()
    nlev = model.nlev
    ps = 100_000.0
    theta_value = 300.0
    u_field = np.repeat((12.0 * np.cos(lat2))[None], nlev, axis=0)
    v_field = np.zeros_like(u_field)
    vector = VorticityDivergenceOperator(transform)
    vorticity, divergence = vector.vordiv_from_wind(u_field, v_field)
    zeros = np.zeros((nlev, *grid.shape))
    atmosphere = cold.atmosphere.with_fields([
        vorticity, divergence,
        transform.forward(np.full((nlev, *grid.shape), theta_value)),
        transform.forward(np.full(grid.shape, math.log(ps))),
        transform.forward(np.full((nlev, *grid.shape), 0.004)),
    ]).with_grid_tracers({name: zeros.copy() for name in ("qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng")})
    bundle = cold.__class__(atmosphere, cold.surface, cold.physics_state)
    path = tmp_path / "arwen_global_step00000000.npz"
    write_checkpoint(
        path, bundle, config_hash=cfg.config_hash, to_numpy=np.asarray,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    receipt = {
        "config": {
            "truncation": cfg.truncation, "dealias_factor": cfg.dealias_factor,
            "a_half_pa": list(cfg.a_half_pa), "b_half": list(cfg.b_half),
            "native_adapter_options": {"start_time_utc": "2026-01-01T00:00:00Z"},
        },
        "transform": {"nlat": grid.nlat, "nlon": grid.nlon, "radius_m": grid.radius_m},
        "config_hash": cfg.config_hash,
    }
    phi_surface = np.full(grid.shape, GRAVITY_M_S2 * 120.0)
    reader = sc.ModelReader(receipt, phi_surface, levels_pa=(50_000.0, 85_000.0))
    _metadata, arrays = read_checkpoint(path)
    columns = reader.column_fields(arrays)
    assert np.max(np.abs(columns["surface_pressure"] - ps)) < 1.0e-6
    assert np.max(np.abs(columns["u"] - u_field)) < 1.0e-9
    assert np.max(np.abs(columns["v"])) < 1.0e-9
    p_full = columns["p_full"]
    assert np.max(np.abs(columns["temperature"] - theta_value * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA)) < 1.0e-9
    plf = reader.sample(path)
    assert plf.source["valid_time"] == "2026-01-01T00:00:00Z"
    ln_p = np.log(p_full[:, 0, 0])
    for level in (50_000.0, 85_000.0):
        assert plf.valid[level].all()
        assert np.max(np.abs(plf.fields["u"][level] - 12.0 * np.cos(lat2))) < 1.0e-9
        # Temperature at the target is the linear-in-ln p interpolation of
        # the column's own temperatures (a constant-theta column is
        # exponential in ln p, so on this 4-level grid the closed form sits
        # 0.65 K away: the stated floor of the interpolation choice).
        expected_t = float(np.interp(math.log(level), ln_p, columns["temperature"][:, 0, 0]))
        assert np.max(np.abs(plf.fields["t"][level] - expected_t)) < 1.0e-9
        assert np.isfinite(plf.fields["z"][level]).all()
    # Hydrostatic heights of a 300 K constant-theta column over a 120 m
    # surface at 1000 hPa: about 1.4 km at 850 hPa and 5.6 km at 500 hPa.
    assert abs(float(np.mean(plf.fields["z"][85_000.0])) - 1_420.0) < 150.0
    assert abs(float(np.mean(plf.fields["z"][50_000.0])) - 5_620.0) < 300.0


# -- the CLI ------------------------------------------------------------------


def test_calibrate_cli_prints_every_family(tmp_path, capsys):
    out = tmp_path / "cal.json"
    assert sc.main(["--calibrate", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    families = {row["family"] for row in payload["rows"]}
    assert families == {"A1", "A2", "B", "C", "D", "E", "F", "G"}
    a_rows = [row for row in payload["rows"] if row["family"] in ("A1", "A2")]
    for row in a_rows:
        assert row["max_abs_error"]["z"] < ROUNDOFF_M
        assert row["max_abs_error"]["t"] < ROUNDOFF
    d_rows = [row for row in payload["rows"] if row["family"] == "D"]
    for row in d_rows:
        assert row["ac_read"] == pytest.approx(row["ac_expected"], abs=ROUNDOFF)
    for row in payload["rows"]:
        if row["family"] == "E":
            assert row["read_masked_fraction"]["t850"] == pytest.approx(row["planted_masked_fraction_global"], abs=ROUNDOFF)


def test_analysis_series_keeps_one_row_per_hour_and_names_the_rest():
    rows = [
        {"hour": 0.0, "reference": {"kind": "analysis", "product": "gfs"}},
        {"hour": 6.0, "reference": {"kind": "analysis", "product": "gdas"}},
        {"hour": 18.0, "reference": {"kind": "analysis", "product": "gdas"}},
        {"hour": 18.0, "reference": {"kind": "analysis", "product": "gfs"}},
        {"hour": 12.0, "reference": {"kind": "forecast", "product": "gfs"}},
        {"hour": 3.0, "reference": {"kind": "forecast", "product": "gfs"}},
    ]
    primary, extra, forecasts = sc.analysis_series(rows)
    assert [(r["hour"], r["reference"]["product"]) for r in primary] == [(0.0, "gfs"), (6.0, "gdas"), (18.0, "gdas")]
    assert [(r["hour"], r["reference"]["product"]) for r in extra] == [(18.0, "gfs")]
    assert [r["hour"] for r in forecasts] == [3.0, 12.0]


def test_representation_floor_reads_zero_for_a_representable_analysis_and_the_planted_loss(grid):
    """A frame whose columns are linear in ln p round-trips exactly through
    the vertical row (0 to float64 rounding); the spectral row reads
    exactly what the truncation removes from the regridded field (the
    degree-2 pattern loses only the bilinear regrid's own high-degree
    residue, 1e-5 of its rms on the 1 degree source grid); the cold-start
    row, the analysis through the run's own initialization and read back
    as a checkpoint, stays inside that residue on every target; and a
    zonal wave planted on the product's z500 alone reads its rms in all
    three rows, because each rebuilds or truncates the height and the
    wave is the product's.
    A pattern that is not band-limited (cos(lat) cos(2 lon), whose
    latitudinal shape needs every degree) would lose 0.4 percent of its
    rms, so the frame uses cos^2(lat) cos(2 lon), a degree-2 harmonic."""
    from woof.globe.analysis_initial import _global_regridder
    from woof.globe.constants import DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2
    from woof.globe.vertical import HybridCoordinate

    coordinate = HybridCoordinate.surface_stretched(40)
    receipt = {
        "config": {"truncation": grid.truncation, "dealias_factor": 1.5,
                   "a_half_pa": coordinate.a_half_pa.tolist(), "b_half": coordinate.b_half.tolist()},
        "transform": {"nlat": grid.nlat, "nlon": grid.nlon, "radius_m": grid.radius_m},
    }
    transform = sc.build_transform_from_receipt(receipt)
    lat_src = np.arange(90.0, -90.001, -1.0)
    lon_src = np.arange(0.0, 360.0, 1.0)
    lon_s, lat_s = np.meshgrid(np.radians(lon_src), np.radians(lat_src))
    levels = np.asarray([10_000.0, 20_000.0, 25_000.0, 30_000.0, 40_000.0, 50_000.0, 60_000.0, 70_000.0, 85_000.0, 100_000.0])
    x = np.log(levels / 101_325.0)[:, None, None]
    pattern = np.cos(lat_s) ** 2 * np.cos(2.0 * lon_s)  # a degree-2 harmonic: inside T21
    # Temperature and humidity zonally uniform: relative humidity is a
    # nonlinear function of temperature, so a temperature pattern would put
    # harmonics beyond the truncation into the derived RH field.
    t = 290.0 + 0.0 * lat_s + 25.0 * x
    q = np.full(t.shape, 0.002)
    # Two solid-body rotations (degree-1 streamfunctions, exact through the
    # vorticity-divergence operator): about the pole with a speed linear in
    # ln p, and about an equatorial axis.  A uniform zonal wind would not
    # do: its vorticity u tan(lat) / a is singular at the poles.
    u = (10.0 - 15.0 * x) * np.cos(lat_s) + 3.0 * np.sin(lat_s) * np.sin(lon_s)
    v = 3.0 * np.cos(lon_s) + 0.0 * x
    ps = np.full(lat_s.shape, 101_325.0)
    terrain = 100.0 * (1.0 + pattern)
    # geopotential consistent with the hydrostatic integration of Tv linear in ln p
    c0 = 290.0 * (1.0 + 0.61 * 0.002)
    c1 = 25.0 * (1.0 + 0.61 * 0.002)
    z = (GRAVITY_M_S2 * terrain - DRY_AIR_GAS_CONSTANT * (c0 * x + c1 * x * x / 2.0)) / GRAVITY_M_S2
    fields = {
        "geopotential_height": z, "air_temperature": t, "specific_humidity": q,
        "eastward_wind": u, "northward_wind": v, "surface_pressure": ps, "terrain_height": terrain,
    }
    regrid = _global_regridder(lat_src, lon_src, grid)

    def truncation_loss(field):
        on_grid = regrid(field)
        return grid.rms(np.asarray(transform.inverse(transform.forward(on_grid))) - on_grid)

    payload = sc.representation_floor(sc.SyntheticFrame(lat_src, lon_src, levels, fields), receipt)
    assert set(payload["rows"]) == {"vertical", "spectral", "cold_start"}
    residue = {"z500": 1.0e-2, "t850": 1.0e-6, "w250": 1.0e-2, "w850": 1.0e-2, "rh700": 1.0e-6}
    for tag in ("vertical", "spectral", "cold_start"):
        for target, block in payload["rows"][tag].items():
            for region, row in block["regions"].items():
                value = row.get("rmse", row.get("rmsve"))
                bar = 1.0e-6 if tag == "vertical" else residue[target]
                assert value < bar, (tag, target, region, value)
                assert row["masked_fraction"] == pytest.approx(0.0, abs=ROUNDOFF)
    z500 = payload["rows"]["spectral"]["z500"]["regions"]["global"]["rmse"]
    assert z500 == pytest.approx(truncation_loss(z[levels == 50_000.0][0]), rel=1.0e-9, abs=1.0e-12)
    assert 0.0 < z500 < 1.0e-2
    # A zonal wave beyond the truncation planted on the product's z500
    # alone.  Every row rebuilds height hydrostatically from temperature
    # (the vertical and cold-start rows) or truncates it away (the spectral
    # row), so the wave is the product's alone and all three read its rms
    # as the z500 rmse, nothing on any other target.
    wave = 30.0 * np.cos(lat_s) * np.cos(25.0 * lon_s)
    z2 = z.copy()
    z2[levels == 50_000.0] += wave
    payload2 = sc.representation_floor(sc.SyntheticFrame(lat_src, lon_src, levels, {**fields, "geopotential_height": z2}), receipt)
    planted = grid.rms(regrid(wave))
    assert planted > 10.0
    spectral = payload2["rows"]["spectral"]["z500"]["regions"]["global"]["rmse"]
    assert spectral == pytest.approx(truncation_loss(z2[levels == 50_000.0][0]), rel=1.0e-9)
    assert spectral == pytest.approx(planted, rel=1.0e-3)
    assert payload2["rows"]["vertical"]["z500"]["regions"]["global"]["rmse"] == pytest.approx(planted, rel=1.0e-9)
    assert payload2["rows"]["cold_start"]["z500"]["regions"]["global"]["rmse"] == pytest.approx(planted, rel=1.0e-6)
    for target in ("t850", "w250", "w850", "rh700"):
        for tag in ("vertical", "spectral", "cold_start"):
            row = payload2["rows"][tag][target]["regions"]["global"]
            assert row.get("rmse", row.get("rmsve")) < residue[target]


def test_scorecard_table_and_chart(tmp_path):
    grid = sc.synthetic_grid(21)
    reference = sc.synthetic_reference(grid, seed=9)
    rows = []
    for hour, delta in ((0.0, 0.0), (6.0, 4.0), (12.0, 8.0)):
        model = sc.shifted_copy(reference, {("z", 50_000.0): delta})
        rows.append({
            "hour": hour, "step": int(hour), "time_s": hour * 3600.0, "valid_time": "x", "checkpoint": "c",
            "reference": {"kind": "analysis", "product": "gdas", "path": "gdas.f000"},
            "scores": sc.score_pair(model, reference),
        })
    payload = {"schema": sc.SCHEMA, "label": "synthetic", "rows": rows}
    table = sc.scorecard_table([payload], hour=12.0)
    z = [row for row in table if row["target"] == "z500" and row["reading"] == "rmse" and row["region"] == "global"][0]
    assert z["synthetic"] == pytest.approx(8.0, abs=ROUNDOFF_M)
    png = sc.render_chart([payload], tmp_path / "chart.png", title="t")
    assert png.exists() and png.stat().st_size > 1000
    # A second arm with the reference model's own skill rows draws the bar
    # line beside both arms.
    skill = [{"hour": h, "valid_time": None, "forecast": {"kind": "forecast", "product": "gfs"},
              "reference": {"kind": "analysis", "product": "gdas"}, "scores": sc.score_pair(model, reference)}
             for h in (6.0, 12.0)]
    second = {"schema": sc.SCHEMA, "label": "other", "rows": rows[:2], "reference_skill": skill}
    table_png = sc.render_scorecard_table([payload, second], tmp_path / "table.png", hour=6.0)
    assert table_png.exists() and table_png.stat().st_size > 1000
    both_png = sc.render_chart([payload, second], tmp_path / "both.png", title="t")
    assert both_png.exists() and both_png.stat().st_size > 1000
    lines = sc.summary_lines(payload)
    assert len(lines) == 4


def test_spectral_floor_row_truncates_the_product_s_own_field_under_a_mountain(grid):
    """The spectral floor row must read what the truncation removes from
    the product's own field, not what a fill does at a mountain edge: a
    frame whose fields are band-limited everywhere (the product's own
    below-ground continuation included) with a 60 kPa plateau masking the
    850 and 700 hPa targets over a twentieth of the globe reads the regrid
    residue at every target, masked levels included, while the vertical
    round trip over a terrain consistent with the plateau stays exact."""
    from woof.globe.constants import DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2
    from woof.globe.vertical import HybridCoordinate

    coordinate = HybridCoordinate.surface_stretched(40)
    receipt = {
        "config": {"truncation": grid.truncation, "dealias_factor": 1.5,
                   "a_half_pa": coordinate.a_half_pa.tolist(), "b_half": coordinate.b_half.tolist()},
        "transform": {"nlat": grid.nlat, "nlon": grid.nlon, "radius_m": grid.radius_m},
    }
    lat_src = np.arange(90.0, -90.001, -1.0)
    lon_src = np.arange(0.0, 360.0, 1.0)
    lon_s, lat_s = np.meshgrid(np.radians(lon_src), np.radians(lat_src))
    levels = np.asarray([10_000.0, 20_000.0, 25_000.0, 30_000.0, 40_000.0, 50_000.0, 60_000.0, 70_000.0, 85_000.0, 100_000.0])
    x = np.log(levels / 101_325.0)[:, None, None]
    pattern = np.cos(lat_s) ** 2 * np.cos(2.0 * lon_s)
    t = 290.0 + 0.0 * lat_s + 25.0 * x
    q = np.full(t.shape, 0.002)
    u = (10.0 - 15.0 * x) * np.cos(lat_s) + 3.0 * np.sin(lat_s) * np.sin(lon_s)
    v = 3.0 * np.cos(lon_s) + 0.0 * x
    c0 = 290.0 * (1.0 + 0.61 * 0.002)
    c1 = 25.0 * (1.0 + 0.61 * 0.002)
    sea_level_terrain = 100.0 * (1.0 + pattern)

    def height(p):
        xx = np.log(p / 101_325.0)
        return (GRAVITY_M_S2 * sea_level_terrain - DRY_AIR_GAS_CONSTANT * (c0 * xx + c1 * xx * xx / 2.0)) / GRAVITY_M_S2

    plateau = (np.abs(lat_s) < np.radians(15.0)) & (np.cos(lon_s) > 0.5)
    ps = np.where(plateau, 60_000.0, 101_325.0)
    # The terrain sits where the column's own hydrostatic height puts the
    # surface pressure, so the vertical row's integration from it is exact.
    terrain = np.where(plateau, height(ps), sea_level_terrain)
    z = height(levels[:, None, None] * np.ones_like(t))
    fields = {
        "geopotential_height": z, "air_temperature": t, "specific_humidity": q,
        "eastward_wind": u, "northward_wind": v, "surface_pressure": ps, "terrain_height": terrain,
    }
    payload = sc.representation_floor(sc.SyntheticFrame(lat_src, lon_src, levels, fields), receipt)
    residue = {"z500": 1.0e-2, "t850": 1.0e-6, "w250": 1.0e-2, "w850": 1.0e-2, "rh700": 1.0e-6}
    for target in ("t850", "rh700", "w850"):
        for tag in ("vertical", "spectral", "cold_start"):
            assert payload["rows"][tag][target]["regions"]["global"]["masked_fraction"] > 0.02, (tag, target)
    for target in ("z500", "w250"):
        assert payload["rows"]["spectral"][target]["regions"]["global"]["masked_fraction"] == pytest.approx(0.0, abs=ROUNDOFF)
    for target, block in payload["rows"]["spectral"].items():
        for region, row in block["regions"].items():
            value = row.get("rmse", row.get("rmsve"))
            assert value < residue[target], ("spectral", target, region, value)
    for target, block in payload["rows"]["vertical"].items():
        for region, row in block["regions"].items():
            value = row.get("rmse", row.get("rmsve"))
            assert value < 1.0e-6, ("vertical", target, region, value)


def test_model_cache_is_keyed_by_checkpoint_content_not_config_hash_or_name(tmp_path):
    """Two checkpoints of one config hash and one file name but different
    content (two arms of different code on one config) must never share a
    cache entry; the same content in two places must; a checkpoint
    without its self_sha256 is refused by name."""
    import numpy as np

    def write(path, self_sha, config_hash="c" * 64):
        metadata = {"schema": "x", "config_hash": config_hash, "self_sha256": self_sha, "time_s": 0.0, "step": 0}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, __metadata__=np.asarray(json.dumps(metadata)), atmosphere__theta=np.zeros(3))
        return path

    a = write(tmp_path / "arm_a" / "arwen_global_step00000072.npz", "a" * 64)
    b = write(tmp_path / "arm_b" / "arwen_global_step00000072.npz", "b" * 64)
    a_again = write(tmp_path / "arm_a_copy" / "arwen_global_step00000999.npz", "a" * 64)
    phi = "f" * 64
    assert sc.checkpoint_content_digest(a) == "a" * 64
    assert sc.model_cache_path(tmp_path, a, phi) != sc.model_cache_path(tmp_path, b, phi)
    assert sc.model_cache_path(tmp_path, a, phi) == sc.model_cache_path(tmp_path, a_again, phi)
    assert sc.model_cache_path(tmp_path, a, phi) != sc.model_cache_path(tmp_path, a, "0" * 64)
    assert "c" * 12 not in sc.model_cache_path(tmp_path, a, phi).name
    bare = tmp_path / "bare" / "arwen_global_step00000072.npz"
    bare.parent.mkdir()
    np.savez(bare, __metadata__=np.asarray(json.dumps({"schema": "x"})), atmosphere__theta=np.zeros(3))
    with pytest.raises(ValueError, match="carries no self_sha256"):
        sc.model_cache_path(tmp_path, bare, phi)


# -- family G: the vertical round trip on two level sets -------------------


@pytest.mark.parametrize("label, factor_bar", [("surface_stretched_40", (0.036, 0.037)), ("jet_refined_48", (0.0093, 0.0094))])
def test_level_set_family_reads_zero_on_linear_columns_and_the_predicted_jet_loss(label, factor_bar):
    """Family G in both directions on both level sets: columns linear in
    ln p round-trip to rounding; a jet planted with slopes +S and -S about
    250 hPa reads back exactly ``2 S f`` of rmsve with ``f`` the level
    set's own bracket factor (0.0362 on the 40-level stack's 55 hPa
    layers, 0.0094 on the 48-level candidate's 23 hPa layers), nothing on
    any other target, so the instrument reads the level change in the
    direction and by the amount the arithmetic predicts."""
    from woof.globe.vertical import HybridCoordinate

    vertical = HybridCoordinate.surface_stretched(40) if label.endswith("40") else HybridCoordinate.jet_refined(48)
    family = sc.level_set_floor_family(vertical)
    f = family["kink_read_factor_at_target"]
    assert factor_bar[0] < f < factor_bar[1]
    assert f == sc.kink_read_factor(vertical, 25_000.0)
    g = family["rows"]
    for target in g["G1_linear"]["read"].values():
        for row in target.values():
            for value in row.values():
                assert abs(value) < 1.0e-9
    expected = family["rows"]["G2_jet"]["expected"]
    S = family["jet_slope_m_s_per_lnp"]
    assert expected["w250_rmsve"] == pytest.approx(2.0 * S * f, rel=1.0e-12)
    for region, row in g["G2_jet"]["read"]["w250"].items():
        assert row["rmsve"] == pytest.approx(expected["w250_rmsve"], rel=1.0e-9), region
        # the speed bias carries the small meridional component of the planted wind
        assert row["speed_bias"] == pytest.approx(expected["w250_speed_bias"], rel=2.0e-3), region
    for name, target in g["G2_jet"]["read"].items():
        if name == "w250":
            continue
        for row in target.values():
            for value in row.values():
                assert abs(value) < 1.0e-9, name


def test_the_jet_refined_stack_reads_a_quarter_of_the_40_level_floor():
    from woof.globe.vertical import HybridCoordinate

    forty = sc.kink_read_factor(HybridCoordinate.surface_stretched(40), 25_000.0)
    cand = sc.kink_read_factor(HybridCoordinate.jet_refined(48), 25_000.0)
    assert 0.24 < cand / forty < 0.27
    # A full level on the target reads nothing of a kink there.
    coordinate = HybridCoordinate.surface_stretched(40)
    p_half = coordinate.a_half_pa + coordinate.b_half * 101_325.0
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    assert sc.kink_read_factor(coordinate, float(p_full[15])) == 0.0
    with pytest.raises(ValueError, match="outside the full levels"):
        sc.kink_read_factor(coordinate, 10.0)


def test_calibrate_carries_family_g_for_both_level_sets():
    payload = sc.calibrate()
    rows = [row for row in payload["rows"] if row["family"] == "G"]
    assert [row["level_set"] for row in rows] == ["surface_stretched_40", "jet_refined_48"]
    assert [row["nlev"] for row in rows] == [40, 48]
    for row in rows:
        assert row["G1_linear_max_abs_read"] < 1.0e-9
        assert row["G2_jet_other_targets_max_abs_read"] < 1.0e-9
        assert row["G2_jet_w250_read"]["rmsve"] == pytest.approx(row["G2_jet_expected"]["w250_rmsve"], rel=1.0e-9)
    assert rows[1]["G2_jet_w250_read"]["rmsve"] < 0.27 * rows[0]["G2_jet_w250_read"]["rmsve"]


def test_a_config_stands_in_for_a_run_under_the_floor(tmp_path):
    """The receipt built from a config alone carries what the floor and the
    reader take from a run receipt, says it came from a config, and the
    synthetic linear frame round-trips through it on the 48-level
    candidate exactly as on the 40-level stack."""
    from woof.globe.config import load_config

    base = Path(str(_shipped_configs() / "arwen_global_moist_smoke.toml")).read_text(encoding="utf-8")
    text = base.replace('coordinate = "pressure_blend"\nnlev = 4\n', 'coordinate = "jet_refined"\n').replace("truncation = 3", "truncation = 21")
    path = tmp_path / "cand.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    receipt = sc.floor_receipt_from_config(cfg)
    assert receipt["source"] == "config"
    assert receipt["config"]["nlev"] == 48 and receipt["config"]["vertical_coordinate"] == "jet_refined"
    assert receipt["config_hash"] == cfg.config_hash
    from woof.globe.spectral.grid import GaussianGrid

    nlat, nlon = GaussianGrid.shape_for(cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon, dealias_factor=cfg.dealias_factor)
    assert receipt["transform"]["nlat"] == nlat and receipt["transform"]["nlon"] == nlon
    assert receipt["initial"]["provenance"]["mapping"] is None
    transform = sc.build_transform_from_receipt(receipt)
    assert transform.grid.truncation == 21
    grid = transform.grid
    phi = np.zeros(grid.shape)
    reader = sc.ModelReader(receipt, phi)
    assert reader.vertical.nlev == 48 and reader.start_utc is None
    lat_src = np.arange(90.0, -90.001, -1.0)
    lon_src = np.arange(0.0, 360.0, 1.0)
    lon_s, lat_s = np.meshgrid(np.radians(lon_src), np.radians(lat_src))
    levels = np.asarray([10_000.0, 20_000.0, 25_000.0, 30_000.0, 40_000.0, 50_000.0, 70_000.0, 85_000.0, 100_000.0])
    x = np.log(levels / 101_325.0)[:, None, None]
    t = 290.0 + 0.0 * lat_s + 25.0 * x
    q = np.full(t.shape, 0.002)
    u = (10.0 - 15.0 * x) * np.cos(lat_s) + 3.0 * np.sin(lat_s) * np.sin(lon_s)
    v = 3.0 * np.cos(lon_s) + 0.0 * x
    ps = np.full(lat_s.shape, 101_325.0)
    terrain = np.zeros(lat_s.shape)
    from woof.globe.constants import DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2

    c0 = 290.0 * (1.0 + 0.61 * 0.002)
    c1 = 25.0 * (1.0 + 0.61 * 0.002)
    z = -DRY_AIR_GAS_CONSTANT * (c0 * x + c1 * x * x / 2.0) / GRAVITY_M_S2 + 0.0 * lat_s
    frame = sc.SyntheticFrame(lat_src, lon_src, levels, {
        "geopotential_height": z, "air_temperature": t, "specific_humidity": q,
        "eastward_wind": u, "northward_wind": v, "surface_pressure": ps, "terrain_height": terrain,
    })
    payload = sc.representation_floor(frame, receipt)
    assert payload["nlev"] == 48 and payload["receipt_source"] == "config"
    assert payload["vertical_coordinate"] == "jet_refined"
    assert payload["jet_band_thickest_layer_pa"] == pytest.approx(2_343.9, abs=0.1)
    assert 0.0093 < payload["kink_read_factor_250hpa"] < 0.0094
    for target, block in payload["rows"]["vertical"].items():
        for region, row in block["regions"].items():
            assert row.get("rmse", row.get("rmsve")) < 1.0e-6, (target, region)
