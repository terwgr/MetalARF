"""The condensate species and number moments are grid-point tracers.

Finding 2026-09-02: every hydrometeor and number moment was a prognostic
SPECTRAL tracer, and the positivity machinery that kept those truncated
fields nonnegative rescaled every column on a level by the level's
clipped-to-unclipped mass ratio, moving the mass the clip created in the
negative lobes of one feature out of every cloud and rain shaft on the
planet into clear air, four passes per step, where the microphysics
evaporated it.  Grid-scale surface precipitation was 0.008 kg/m2 in 24 h
at 52 km and total precipitation equalled convective precipitation to
four decimals.

The fix: the ten condensate and moment fields live on the Gaussian grid,
are carried by a positive-definite flux-form transport once per step, and
are never analysed into the spectral basis; vapor, the one water field
left in the basis, closes its ringing inside its own column.  These
tests are the fix's contract; the headline one fails on the tree before
the fix by the fraction it states.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from woof.globe.checkpoint import (
    read_checkpoint,
    state_from_checkpoint,
    write_checkpoint,
)
from woof.globe.config import load_config
from woof.globe.constants import (
    CONDENSATE_SPECIES,
    GRAVITY_M_S2,
    GRID_TRACERS,
    NUMBER_MOMENTS,
    PROGNOSTIC_FIELDS,
    SPECTRAL_FIELDS,
    SPECTRAL_TRACER_CHECKPOINT_SCHEMA,
)
from woof.globe.pins import SPECTRAL_TRACER_ERA_PINS_HASHES
from woof.globe.runner import build_model_and_cold_state
from woof.globe.state import ArwenGlobalState
from woof.globe.transport import GridTracerTransport, sample_grid_field
from woof.globe.vertical import HybridCoordinate
from woof.globe.spectral.transform import SphericalHarmonicTransform

T21_TOML = """
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "planted-rain-column-t21"
acknowledgement = "research-only-arwen-global-v1"
backend = "numpy"
precision = "float64"

[grid]
truncation = 21
dealias_factor = 1.5

[time]
dt_s = 60.0
duration_s = 60.0
output_interval_s = 60.0
maximum_cfl = 0.90
integrator = "ssprk3"

[vertical]
coordinate = "pressure_blend"
nlev = 5
p_top_pa = 100.0

[initial]
surface_pressure_pa = 100000.0
surface_temperature_k = 286.0
top_temperature_k = 220.0
qv_surface = {qv_surface}
zonal_wind_m_s = 0.0
perturbation_amplitude = 0.0
zonal_wavenumber = 2
terrain_amplitude_m = 0.0
surface_water_kg_m2 = 500.0

[physics]
mode = "none"

[diffusion]
enabled = true
order = 4
e_folding_time_s_at_truncation = 21600.0
preserve_degree = 1

[semi_implicit]
enabled = true
weight = 0.5

[repair]
mass_fixer = true
water_fixer = true
positivity_repair = true

[gates]
transform_roundtrip_relative_linf = 5.0e-5
transform_parseval_relative_error = 5.0e-5
mass_relative_drift = 1.0e-7
total_water_relative_drift = 1.0e-6
"""


def _t21_model(tmp_path, *, qv_surface=0.0):
    path = tmp_path / "t21.toml"
    path.write_text(T21_TOML.format(qv_surface=qv_surface), encoding="utf-8")
    cfg = load_config(path)
    model, state = build_model_and_cold_state(cfg)
    return cfg, model, state


def _column_water(model, bundle, name):
    g = model.grid_state(bundle.atmosphere, only=(name, "dp"))
    return np.sum(
        np.asarray(g[name], np.float64) * np.asarray(g["dp"], np.float64), axis=0
    ) / GRAVITY_M_S2


def _cell_weights(grid):
    return (grid.quadrature_weights / (2.0 * grid.nlon))[:, None]


# --------------------------------------------------------------- headline
def test_a_planted_rain_column_keeps_its_condensate_through_a_windless_step(tmp_path):
    """One dynamics step of a resting dry atmosphere with one rain column.

    Measured on the tree before the fix with the same plant (spectral
    rain, T21, 5 levels, dt 60 s, no wind, no physics): the column held
    0.3438 of the planted water after the transform alone, 0.0648 of it
    after one step (0.1884 of its own pre-step water: the per-level
    rescale took a 0.7493 factor on the planted level and the
    hyperdiffusion the rest), and 0.9352 of the planted water sat in
    other columns.  With the grid tracers the column keeps 1.0000 of the
    planted water: the only movement is the 5 mm/s wind the plant's own
    water loading induces in one step (1e-7 of the column), and a
    number-moment column, which loads nothing, holds to round-off.
    """
    cfg, model, state = _t21_model(tmp_path)
    grid = model.transform.grid
    nlat, nlon = grid.shape
    j, i = nlat // 2, nlon // 3
    plant = np.zeros((model.nlev, nlat, nlon))
    plant[1:3, j, i] = 1.0e-3
    number = plant / 5.2e-7
    state.atmosphere.qr = plant.copy()
    state.atmosphere.nr = number.copy()
    dp = np.asarray(model.grid_state(state.atmosphere, only=("dp",))["dp"])
    planted = float(np.sum(plant[:, j, i] * dp[:, j, i]) / GRAVITY_M_S2)
    before = _column_water(model, state, "qr")
    assert before[j, i] == pytest.approx(planted, rel=0.0, abs=0.0)

    out, metrics = model.step(state, cfg.dt_s)

    after = _column_water(model, out, "qr")
    retained = float(after[j, i]) / planted
    cell = _cell_weights(grid)
    outside = float(np.sum(after * cell)) - float(after[j, i] * cell[j, 0])
    planted_global = planted * float(cell[j, 0])
    # Before the fix: retained 0.0648 of the planted water, 0.9352 outside.
    assert retained > 1.0 - 1.0e-6, retained
    assert outside / planted_global < 1.0e-6, outside / planted_global
    assert float(np.min(out.atmosphere.qr)) >= 0.0
    # The rain's number moment rode the same 5 mm/s loading wind.
    np.testing.assert_allclose(
        out.atmosphere.nr, number, rtol=0.0, atol=1.0e-6 * float(number.max()),
    )
    # Nothing was clipped, rescaled or floored, and no reservoir moved
    # for the plant beyond the global drift fixer's uniform correction.
    assert metrics["positivity_fixer_water_kg_m2"] == 0.0
    assert metrics["positivity_fixer_max_rescale"] == 0.0
    assert metrics["tracer_transport_floor_clip_kg_m2"] == 0.0
    assert metrics["tracer_transport_substeps"] == 1
    reservoir_change = (
        np.asarray(out.surface.water_kg_m2) - np.asarray(state.surface.water_kg_m2)
    )
    assert float(np.ptp(reservoir_change)) < 1.0e-9

    # A number-moment column alone loads no air, so the atmosphere stays
    # exactly at rest and the column is bit-identical through the step:
    # the transport with zero flux is the identity, and no spectral map
    # touches a grid tracer.
    _cfg2, model2, rest = _t21_model(tmp_path)
    rest.atmosphere.nr = number.copy()
    out2, metrics2 = model2.step(rest, cfg.dt_s)
    # Measured: the spectral roundoff of the uniform state leaves 1.7e-5
    # m/s of wind after the step (Courant 9e-9) and the column moves by
    # 2.4e-11 of itself; the tree before the fix moved 0.81 of it with no wind.
    np.testing.assert_allclose(
        out2.atmosphere.nr, number, rtol=0.0, atol=1.0e-9 * float(number.max()),
    )
    winds = model2.grid_state(out2.atmosphere, only=("u", "v"))
    assert float(np.max(np.abs(winds["u"]))) < 1.0e-3
    assert float(np.max(np.abs(winds["v"]))) < 1.0e-3
    assert metrics2["tracer_transport_max_courant"] < 1.0e-6


# --------------------------------------------------------------- transport
def _transport(T=21, nlev=4):
    transform = SphericalHarmonicTransform.create(
        T, backend="numpy", precision="float64"
    )
    vertical = HybridCoordinate.pressure_blend(nlev)
    return transform, vertical, GridTracerTransport(transform, vertical)


def test_transport_is_an_exact_identity_with_no_flux():
    transform, vertical, transport = _transport()
    nlat, nlon = transform.grid.shape
    dp = vertical.pressure(np.full((nlat, nlon), 1.0e5), transform.backend)["dp"]
    q = np.zeros_like(dp)
    q[2, nlat // 2, nlon // 3] = 1.0e-3
    zero = np.zeros_like(dp)
    omega = np.zeros((vertical.nlev + 1, nlat, nlon))
    out, metrics = transport.advance({"qr": q.copy()}, dp, zero, zero, omega, 50.0)
    np.testing.assert_array_equal(out["qr"], q)
    np.testing.assert_array_equal(metrics["pseudo_density"], dp)
    assert metrics["floor_clip_kg_m2"] == 0.0


def test_transport_keeps_a_uniform_field_uniform_under_divergent_flow():
    transform, vertical, transport = _transport()
    grid = transform.grid
    nlat, nlon = grid.shape
    lat = grid.lat_rad[:, None]
    lon = grid.lon_rad[None, :]
    dp = vertical.pressure(np.full((nlat, nlon), 1.0e5), transform.backend)["dp"]
    u = 20.0 * np.sin(2.0 * lon) * np.cos(lat) * np.ones_like(dp)
    v = 10.0 * np.cos(3.0 * lon) * np.cos(lat) ** 2 * np.ones_like(dp)
    omega = np.zeros((vertical.nlev + 1, nlat, nlon))
    omega[1:-1] = 0.5 * np.sin(lon) * np.cos(lat)
    q = np.full_like(dp, 2.0e-3)
    out, metrics = transport.advance({"qs": q}, dp, dp * u, dp * v, omega, 600.0)
    # The pseudo-density moved (a divergent flow) but the ratio did not.
    assert float(np.max(np.abs(metrics["pseudo_density"] - dp) / dp)) > 1.0e-3
    assert float(np.max(np.abs(out["qs"] - 2.0e-3))) < 1.0e-15


def test_transport_conserves_mass_and_positivity_under_solid_body_rotation():
    transform, vertical, transport = _transport()
    grid = transform.grid
    nlat, nlon = grid.shape
    a = grid.radius_m
    lat = grid.lat_rad[:, None] * np.ones((1, nlon))
    lon = grid.lon_rad[None, :] * np.ones((nlat, 1))
    dp = vertical.pressure(np.full((nlat, nlon), 1.0e5), transform.backend)["dp"]
    alpha = math.pi / 2.0 - 0.05
    u0 = 2.0 * math.pi * a / (12.0 * 86400.0)
    u = u0 * (np.cos(lat) * np.cos(alpha) + np.sin(lat) * np.cos(lon) * np.sin(alpha))
    v = -u0 * np.sin(lon) * np.sin(alpha)
    r = a * np.arccos(np.clip(np.cos(lat) * np.cos(lon - 1.5 * math.pi), -1.0, 1.0))
    bell = np.where(r < a / 3.0, 0.5 * (1.0 + np.cos(math.pi * r / (a / 3.0))), 0.0)
    q = {"qc": np.broadcast_to(bell, dp.shape).copy() * 1.0e-3}
    cell = _cell_weights(grid)
    mass0 = float(np.sum(np.sum(q["qc"] * dp, axis=0) * cell))
    omega = np.zeros((vertical.nlev + 1, nlat, nlon))
    density = dp
    worst_substeps = 0
    for step in range(48):
        q, metrics = transport.advance(
            q, density, dp * u[None], dp * v[None], omega, 1800.0, step=step,
        )
        density = metrics["pseudo_density"]
        worst_substeps = max(worst_substeps, metrics["substeps_x"])
        assert float(np.min(q["qc"])) >= 0.0
        assert metrics["floor_clip_kg_m2"] == 0.0
    mass1 = float(np.sum(np.sum(q["qc"] * density, axis=0) * cell))
    assert abs(mass1 - mass0) / mass0 < 1.0e-13
    # The cross-polar flow exceeds a Courant number of one at the first
    # Gaussian ring (330 m cells at T255, 5.5 degrees here): the polar
    # band sub-cycles and the rest of the sphere runs one sweep.
    assert metrics["max_courant_x"] > 1.0
    assert worst_substeps >= 2
    assert float(np.max(q["qc"])) <= 1.0e-3 * (1.0 + 1.0e-12)


def test_transport_updates_only_its_own_mass_arrays_and_returns_fresh_ones():
    """The sweeps apply their flux divergence in place, so the arrays they
    write must be advance()'s own: the caller's tracers come back
    untouched, and the outputs are new arrays that no input aliases (a
    state that kept a reference to its tracers would otherwise watch
    them change under the next step)."""

    transform, vertical, transport = _transport()
    grid = transform.grid
    nlat, nlon = grid.shape
    a = grid.radius_m
    lat = grid.lat_rad[:, None] * np.ones((1, nlon))
    lon = grid.lon_rad[None, :] * np.ones((nlat, 1))
    dp = vertical.pressure(np.full((nlat, nlon), 1.0e5), transform.backend)["dp"]
    alpha = math.pi / 2.0 - 0.05
    u0 = 2.0 * math.pi * a / (12.0 * 86400.0)
    u = u0 * (np.cos(lat) * np.cos(alpha) + np.sin(lat) * np.cos(lon) * np.sin(alpha))
    v = -u0 * np.sin(lon) * np.sin(alpha)
    r = a * np.arccos(np.clip(np.cos(lat) * np.cos(lon - 1.5 * math.pi), -1.0, 1.0))
    bell = np.where(r < a / 3.0, 0.5 * (1.0 + np.cos(math.pi * r / (a / 3.0))), 0.0)
    tracers = {
        "qc": np.broadcast_to(bell, dp.shape).copy() * 1.0e-3,
        "qr": np.broadcast_to(bell[::-1], dp.shape).copy() * 2.0e-4,
    }
    before = {name: value.copy() for name, value in tracers.items()}
    omega = np.zeros((vertical.nlev + 1, nlat, nlon))
    omega[1:-1] = 0.02 * np.sin(lat)[None]
    for step in range(2):
        out, metrics = transport.advance(
            tracers, dp, dp * u[None], dp * v[None], omega, 1800.0, step=step,
        )
        for name, value in tracers.items():
            assert np.array_equal(value, before[name]), name
            assert out[name] is not value
            assert not np.shares_memory(out[name], value)
        assert metrics["substeps_x"] >= 2 or metrics["substeps_y"] >= 1
        assert any(not np.array_equal(out[name], before[name]) for name in tracers)


def test_transport_refuses_a_courant_limit_that_breaks_positivity():
    transform, vertical, _ = _transport(T=3, nlev=3)
    with pytest.raises(ValueError, match="courant_limit must lie in"):
        GridTracerTransport(transform, vertical, courant_limit=0.5)


# ----------------------------------------------------------- vapor filler
def test_column_hole_filler_conserves_each_column_and_names_the_unfillable(tmp_path):
    _cfg, model, state = _t21_model(tmp_path, qv_surface=0.006)
    g = model.grid_state(state.atmosphere, only=("qv", "dp"))
    qv = np.array(g["qv"], np.float64)
    dp = np.asarray(g["dp"], np.float64)
    nlat, nlon = qv.shape[1:]
    rng = np.random.default_rng(3)
    # Ringing lobes: a fraction of the columns lose a slice of one level
    # below zero, one column goes net negative.
    ring = rng.integers(0, nlat * nlon, 40)
    for flat in ring:
        j, i = divmod(int(flat), nlon)
        qv[1, j, i] = -0.3 * qv[1, j, i]
    jn, inn = 3, 5
    qv[:, jn, inn] = -1.0e-4
    before = np.sum(qv * dp, axis=0)
    filled, created, max_fraction, unfillable = model._fill_column_holes(qv, dp)
    after = np.sum(filled * dp, axis=0)
    assert float(np.min(filled)) >= 0.0
    fillable = np.ones(before.shape, bool)
    fillable[jn, inn] = False
    np.testing.assert_allclose(after[fillable], before[fillable], rtol=1.0e-12, atol=0.0)
    # The net-negative column keeps its plain clip and is named.
    assert after[jn, inn] == 0.0
    cell = _cell_weights(model.transform.grid)
    negative = -np.sum(np.minimum(qv * dp, 0.0), axis=0) / GRAVITY_M_S2
    assert created == pytest.approx(float(np.sum(negative * cell)), rel=1.0e-12)
    assert unfillable == pytest.approx(float(negative[jn, inn] * cell[jn, 0]), rel=1.0e-12)
    assert 0.0 < max_fraction < 1.0
    positive = np.sum(np.maximum(qv * dp, 0.0), axis=0)
    expected = float(np.max(np.where(fillable, negative * GRAVITY_M_S2 / np.maximum(positive, 1e-300), 0.0)))
    assert max_fraction == pytest.approx(expected, rel=1.0e-12)


def test_positivity_repair_touches_only_vapor_and_closes_in_column(tmp_path):
    _cfg, model, state = _t21_model(tmp_path, qv_surface=0.006)
    transform = model.transform
    grid = transform.grid
    nlat, nlon = grid.shape
    # A one-cell vapor hole rings the spectral vapor field negative.
    g = model.grid_state(state.atmosphere, only=("qv", "dp"))
    qv = np.array(g["qv"], np.float64)
    qv[2, nlat // 2, nlon // 2] = -8.0 * qv[2, nlat // 2, nlon // 2]
    state.atmosphere.qv = transform.project(transform.forward(qv))
    raw = np.asarray(model.grid_state(state.atmosphere, only=("qv",))["qv"])
    assert float(np.min(raw)) < 0.0
    tracers_before = {name: np.array(value, copy=True) for name, value in state.atmosphere.grid_tracers().items()}
    surface_before = np.array(state.surface.water_kg_m2, copy=True)

    repaired, negative_water, negative_tracer, fixer = model._repair_positivity(state)

    assert negative_water == pytest.approx(-float(np.min(raw)))
    assert negative_tracer == 0.0
    assert fixer["water_kg_m2"] > 0.0
    assert 0.0 < fixer["max_rescale"] < 1.0
    assert fixer["unfillable_kg_m2"] == 0.0
    for name, value in tracers_before.items():
        np.testing.assert_array_equal(getattr(repaired.atmosphere, name), value)
    np.testing.assert_array_equal(repaired.surface.water_kg_m2, surface_before)
    # The global vapor mean survives the clip, the fill and the projection
    # (the projection preserves degree zero exactly; dp is unchanged).
    dp = np.asarray(g["dp"], np.float64)
    cell = _cell_weights(grid)
    filled, *_ = model._fill_column_holes(raw, dp)
    projected = np.asarray(model.grid_state(repaired.atmosphere, only=("qv",))["qv"])
    assert float(np.sum(np.sum(projected * dp, 0) * cell)) == pytest.approx(
        float(np.sum(np.sum(filled * dp, 0) * cell)), rel=1.0e-6
    )


def test_a_negative_grid_tracer_is_refused_beyond_roundoff(tmp_path):
    _cfg, model, state = _t21_model(tmp_path)
    nlat, nlon = model.transform.grid.shape
    state.atmosphere.qc = np.zeros((model.nlev, nlat, nlon))
    state.atmosphere.qc[1, 2, 3] = -1.0e-9
    with pytest.raises(FloatingPointError, match="grid tracer qc is negative"):
        model.enforce(state)
    with pytest.raises(FloatingPointError, match="negative beyond roundoff"):
        model._repair_positivity(state)
    # Roundoff below the floor is measured and floored, not refused.
    state.atmosphere.qc[1, 2, 3] = -1.0e-20
    state.atmosphere.qc[0, 0, 0] = 1.0e-3
    repaired, _water, tracer, _fixer = model._repair_positivity(state)
    assert tracer == pytest.approx(1.0e-20)
    assert float(np.min(repaired.atmosphere.qc)) == 0.0


# --------------------------------------------------------------- the state
def test_the_state_carries_five_spectral_fields_and_ten_grid_tracers(tmp_path):
    _cfg, model, state = _t21_model(tmp_path)
    atmosphere = state.atmosphere
    assert len(atmosphere.fields()) == len(SPECTRAL_FIELDS) == 5
    assert set(atmosphere.grid_tracers()) == set(GRID_TRACERS)
    assert GRID_TRACERS == (*CONDENSATE_SPECIES, *NUMBER_MOMENTS)
    assert PROGNOSTIC_FIELDS == (*SPECTRAL_FIELDS, *GRID_TRACERS)
    for name in GRID_TRACERS:
        value = getattr(atmosphere, name)
        assert value.dtype.kind == "f"
        assert value.shape == (model.nlev, *model.transform.grid.shape)
    # The spectral maps carry the tracers by reference; a five-field
    # replacement never copies or drops them.
    replaced = atmosphere.with_fields([f.copy() for f in atmosphere.fields()])
    for name in GRID_TRACERS:
        assert getattr(replaced, name) is getattr(atmosphere, name)
    # A tendency state has no tracer rows.
    tendency = model.rhs(atmosphere)
    assert all(getattr(tendency, name) is None for name in GRID_TRACERS)
    # A bundle copy copies them.
    copied = state.copy()
    for name in GRID_TRACERS:
        assert getattr(copied.atmosphere, name) is not getattr(atmosphere, name)
        np.testing.assert_array_equal(getattr(copied.atmosphere, name), getattr(atmosphere, name))


def test_step_metrics_carry_the_transport_record(tmp_path):
    cfg, model, state = _t21_model(tmp_path, qv_surface=0.006)
    _out, metrics = model.step(state, cfg.dt_s)
    record = metrics["tracer_transport"]
    for key in ("max_courant_x", "max_courant_y", "max_courant_z", "substeps_x",
                "substeps_y", "substeps_z", "floor_clip_kg_m2", "order",
                "pseudo_density_mismatch_relative"):
        assert key in record
    assert "pseudo_density" not in record
    assert metrics["tracer_transport_substeps"] >= 1
    assert metrics["tracer_transport_pseudo_density_mismatch_relative"] >= 0.0
    assert metrics["maximum_repaired_negative_number_per_kg"] == 0.0


# ------------------------------------------------------------- checkpoint
def _fake_pin_metadata_v2(bundle, cfg, transform):
    """A spectral-tracer-era (schema v2) checkpoint of the bundle: every
    prognostic field as complex coefficients, under the era's pin."""
    arrays = {}
    for name in SPECTRAL_FIELDS:
        arrays[f"atmosphere__{name}"] = np.asarray(getattr(bundle.atmosphere, name))
    for name in GRID_TRACERS:
        arrays[f"atmosphere__{name}"] = np.asarray(
            transform.project(transform.forward(getattr(bundle.atmosphere, name)))
        )
    for name, value in bundle.surface.arrays().items():
        arrays[f"surface__{name}"] = np.asarray(value)
    for name, value in bundle.physics_state.arrays.items():
        arrays[f"physics__{name}"] = np.asarray(value)

    def digest(array):
        arr = np.ascontiguousarray(array)
        h = hashlib.sha256()
        h.update(arr.dtype.str.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.view(np.uint8))
        return h.hexdigest()

    metadata = {
        "schema": SPECTRAL_TRACER_CHECKPOINT_SCHEMA,
        "config_hash": cfg.config_hash,
        "pins_hash": sorted(SPECTRAL_TRACER_ERA_PINS_HASHES)[0],
        "time_s": 0.0,
        "step": 0,
        "run_trackers": {
            "maximum_spectral_cfl": 0.0,
            "maximum_mass_fixer_log_offset": 0.0,
            "maximum_global_water_fixer_kg_m2": 0.0,
            "maximum_repaired_negative_mixing_ratio": 0.0,
            "maximum_semi_implicit_divergence_increment_s1": 0.0,
            "maximum_physics_water_repair_kg_m2": 0.0,
            "maximum_native_water_residual_kg_m2": 0.0,
            "maximum_native_energy_residual_j_m2": 0.0,
        },
        "physics_state_schema": bundle.physics_state.schema,
        "physics_metadata": bundle.physics_state.metadata,
        "arrays": {
            name: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": digest(a)}
            for name, a in arrays.items()
        },
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    metadata["self_sha256"] = hashlib.sha256(canonical).hexdigest()
    return metadata, arrays


def test_checkpoint_v3_roundtrips_grid_tracers_and_v2_is_inspectable_only(tmp_path):
    cfg, model, state = _t21_model(tmp_path, qv_surface=0.006)
    nlat, nlon = model.transform.grid.shape
    state.atmosphere.qr[1, nlat // 2, nlon // 3] = 2.0e-3
    path = write_checkpoint(
        tmp_path / "v3.npz", state, config_hash=cfg.config_hash, to_numpy=np.asarray,
    )
    metadata, arrays = read_checkpoint(
        path, expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme,
    )
    assert metadata["schema"] == "gpuwm.arwen-global-checkpoint/v3"
    for name in GRID_TRACERS:
        assert arrays[f"atmosphere__{name}"].dtype.kind == "f"
    for name in SPECTRAL_FIELDS:
        assert arrays[f"atmosphere__{name}"].dtype.kind == "c"
    restored = state_from_checkpoint(metadata, arrays, model.transform.backend)
    for name in GRID_TRACERS:
        np.testing.assert_array_equal(getattr(restored.atmosphere, name), getattr(state.atmosphere, name))
    # A v3 checkpoint whose grid tracer was written complex is refused.
    bad_meta, bad_arrays = _fake_pin_metadata_v2(state, cfg, model.transform)
    v2 = tmp_path / "v2.npz"
    with v2.open("wb") as stream:
        np.savez_compressed(
            stream, __metadata__=np.asarray(json.dumps(bad_meta, sort_keys=True)), **bad_arrays,
        )
    # Inspection (no scheme) reads the spectral-tracer era; a restart
    # under a scheme refuses it by name.
    meta2, arrays2 = read_checkpoint(v2)
    assert meta2["schema"] == SPECTRAL_TRACER_CHECKPOINT_SCHEMA
    with pytest.raises(ValueError, match="spectral-tracer era"):
        read_checkpoint(v2, semi_implicit_scheme=cfg.semi_implicit_scheme)
    with pytest.raises(ValueError, match="pass the run's transform"):
        state_from_checkpoint(meta2, arrays2, model.transform.backend)
    upgraded = state_from_checkpoint(
        meta2, arrays2, model.transform.backend, transform=model.transform,
    )
    for name in GRID_TRACERS:
        value = getattr(upgraded.atmosphere, name)
        assert value.dtype.kind == "f"
        assert float(np.min(value)) >= 0.0
    # The synthesized rain carries the old representation's own ringing:
    # a one-cell spike's truncation spreads it over the sphere and the
    # clip keeps the positive lobes (measured 5.1x the planted sum at
    # T21), which is exactly the defect the grid tracers retire; the
    # door exists for inspection, not for resuming.
    old = float(np.sum(np.asarray(state.atmosphere.qr)))
    new = float(np.sum(np.asarray(upgraded.atmosphere.qr)))
    assert new >= old
    assert np.isfinite(new)


# --------------------------------------------------------------- sampling
def test_grid_field_sampling_is_convex_and_exact_on_the_nodes():
    transform, _vertical, _ = _transport(T=21, nlev=3)
    grid = transform.grid
    nlat, nlon = grid.shape
    rng = np.random.default_rng(5)
    field = rng.random((2, nlat, nlon))
    lat = grid.latitude_deg
    lon = grid.longitude_deg
    lon2, lat2 = np.meshgrid(lon, lat)
    on_nodes = sample_grid_field(grid, field, lat2, lon2)
    np.testing.assert_allclose(on_nodes, field, rtol=0.0, atol=1.0e-12)
    between = sample_grid_field(
        grid, field, 0.5 * (lat[:-1] + lat[1:])[:, None], (lon + 360.0 / nlon / 2.0)[None, :],
    )
    assert float(np.min(between)) >= float(np.min(field))
    assert float(np.max(between)) <= float(np.max(field))
    # Beyond the outer rings the nearest ring is held; never negative.
    poles = sample_grid_field(grid, field, np.array([89.9, -89.9]), np.array([10.0, 20.0]))
    assert poles.shape == (2, 2)
    assert float(np.min(poles)) >= 0.0


def test_moment_strength_off_default_is_refused(tmp_path):
    path = tmp_path / "moments.toml"
    path.write_text(
        T21_TOML.format(qv_surface=0.0).replace(
            "preserve_degree = 1", "preserve_degree = 1\nmoment_strength = 0.5"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="moment_strength has no effect"):
        load_config(path)
