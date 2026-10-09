"""The v1 door's surface-pressure increment: what it conserves and what it
removes, measured (audit 2026-09-06 of the -3.3 hPa bias of the cycled
forecast).

Three properties of the analysis step, held on the smoke configuration:

* the global mean of surface pressure is conserved to roundoff whatever
  the sign of the innovation (the door's mass-preserving offset), and
  the dry-air mass with it: no mass leaves or enters the analysis step;
* the uniform part of a network-wide pressure innovation is REMOVED by
  that preservation (a +500 Pa innovation at 64 stations covering the
  globe leaves under 1 Pa in the state, O-A rms within 0.1 Pa of O-B):
  the door analyses gradients, and a bias every station shares is not
  something it can correct.  The report says how much it removed
  (``mass_preservation.uniform_increment_removed_pa``) so the reading is
  never silent;
* a pressure-only analysis changes no potential temperature, so the
  temperature at a model level moves only through the level's pressure
  (kappa T b dlnps), which is the fixed-theta convention of the
  increment and is measured here.

The audit found no mass defect in the analysis step: the physics writes
no surface-pressure tendency, so water the moisture update adds cannot
reach the pressure through the model's own books either.  The pressure
loss of the cycled forecast is the dynamics' response to unbalanced
increments and to the rain the added vapor produces, which these tests
do not pretend to measure.

The refutation of 2026-09-06 added the regional family: a network
covering three percent of the sphere (a CONUS-sized box) is analysed
with the innovation's sign and most of its size inside the box, the
global mean stays, and the uniform component the rule removed is
recorded (at T3 the truncation rings the bump over the sphere, so the
removed mean is larger than the box's share of the innovation; the
number is in the report, never silent).  Read with the initial-state
report's own arms against the CONUS stations at 18 h (model minus
observation, sea-level pressure bias): the cold start -1.31 hPa, the
24 h cycle without the moisture update -1.58 hPa, the cycle with it
-3.34 hPa.  The -3.3 hPa is therefore the inherited analysis bias
(-1.3), the unbalanced hourly pressure increment's forecast response
(-0.3) and the moisture update's (-1.8, the rain-out of the added
vapor); none of it is made in this step, which moves pressure the way
the reports ask and conserves mass both ways.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from woof.globe.assimilate import (
    MASS_PRESERVATION_DIVERGENCE,
    AssimilationOptions,
    _Family,
    _ModelSpace,
    _to_numpy_spectral,
    analyse,
)
from woof.globe.checkpoint import read_checkpoint, state_from_checkpoint
from woof.globe.constants import GRAVITY_M_S2, KAPPA
from woof.globe.obs_table import ObsRow
from woof.globe.runner import build_model_and_cold_state, build_transform

from test_arwen_global_assimilate import spun_up  # noqa: F401 - the fixture rides the import

MOMENT = dt.datetime(2026, 8, 31, 12, tzinfo=dt.timezone.utc)


def _pressure_rows(values, lats, lons):
    return [
        ObsRow("probe", f"P{k}", float(lats[k]), float(lons[k]), 0.0, None, MOMENT,
               "surface_pressure_pa", float(values[k]), 100.0)
        for k in range(len(values))
    ]


def _analysis_of(cfg, checkpoint, delta_pa: float):
    transform = build_transform(cfg)
    model, _ = build_model_and_cold_state(cfg, transform)
    metadata, arrays = read_checkpoint(checkpoint)
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    model.enforce(state)
    terrain = _to_numpy_spectral(
        transform.backend, transform.forward(model.surface_geopotential))
    space = _ModelSpace(
        transform, cfg.vertical, terrain, state.surface,
        cfg.reference_physics.soil_wetness_capacity,
    )
    lats = np.repeat(np.linspace(-60.0, 60.0, 8), 8)
    lons = np.tile(np.arange(0.0, 360.0, 45.0), 8)
    background = space.hx_surface_pressure(
        state.atmosphere, _Family(_pressure_rows(np.zeros(lats.size), lats, lons)))
    rows = _pressure_rows(background + delta_pa, lats, lons)
    options = AssimilationOptions(length_scale_km=4000.0, gate_minimum_count=10**6)
    analysis, report, _ = analyse(
        cfg, model, transform, state, rows, sources=[],
        background={"self_sha256": metadata["self_sha256"], "step": 2, "time_s": 20.0},
        analysis_time=MOMENT, options=options,
    )
    host = transform.backend.to_numpy
    before = model.grid_state(state.atmosphere)
    after = model.grid_state(analysis.atmosphere)
    return report, transform.grid, {
        key: (host(before[key]), host(after[key]))
        for key in ("ps", "temperature", "p_full", "qv", "dp")
    }, (host(state.atmosphere.theta), host(analysis.atmosphere.theta))


@pytest.mark.parametrize("delta_pa", [500.0, -500.0])
def test_the_analysis_step_conserves_total_and_dry_mass(spun_up, delta_pa):
    cfg, checkpoint = spun_up
    report, grid, fields, _ = _analysis_of(cfg, checkpoint, delta_pa)
    ps_b, ps_a = fields["ps"]
    assert grid.global_mean(ps_a) == pytest.approx(grid.global_mean(ps_b), abs=1e-6)
    qv_b, qv_a = fields["qv"]
    dp_b, dp_a = fields["dp"]
    water_b = np.sum(qv_b * dp_b, axis=0) / GRAVITY_M_S2
    water_a = np.sum(qv_a * dp_a, axis=0) / GRAVITY_M_S2
    dry_b = grid.global_mean(ps_b - GRAVITY_M_S2 * water_b)
    dry_a = grid.global_mean(ps_a - GRAVITY_M_S2 * water_a)
    # 1.4e-5 Pa measured; the bar is a thousandth of a pascal.
    assert dry_a == pytest.approx(dry_b, abs=1e-3)
    # The report states the offset it applied and the divergence.
    record = report["mass_preservation"]
    assert record["log_offset"] == report["mass_preserving_log_offset"]
    assert record["divergence"] == MASS_PRESERVATION_DIVERGENCE
    assert record["global_mean_surface_pressure_pa"] == pytest.approx(grid.global_mean(ps_b))


@pytest.mark.parametrize("delta_pa", [500.0, -500.0])
def test_a_network_wide_pressure_innovation_is_removed_and_reported(spun_up, delta_pa):
    """Sixty-four stations covering the globe agree the model is 5 hPa
    off: the spread increment is nearly uniform, the preservation takes
    it out, and under 1 Pa reaches the state.  The report says how much
    was removed, with the sign of the innovation."""
    cfg, checkpoint = spun_up
    report, _, fields, _ = _analysis_of(cfg, checkpoint, delta_pa)
    ps_b, ps_a = fields["ps"]
    assert np.abs(ps_a - ps_b).max() < 1.0
    row = report["variables"]["surface_pressure_pa"]
    assert row["o_minus_b"]["rms"] == pytest.approx(abs(delta_pa), abs=0.01)
    assert abs(row["o_minus_a"]["rms"] - row["o_minus_b"]["rms"]) < 0.1
    removed = report["mass_preservation"]["uniform_increment_removed_pa"]
    # The spread increment asked for most of the innovation everywhere.
    assert np.sign(removed) == np.sign(delta_pa)
    assert 0.9 * abs(delta_pa) < abs(removed) < 1.01 * abs(delta_pa)
    assert report["increment_maxabs"]["log_surface_pressure"] == pytest.approx(
        abs(np.log1p(delta_pa / 1.0e5)), rel=0.05)
    # The scorecard applies its rule to the letter (O-A 499.99 is below O-B
    # 500.00, so the cell reads pass), and the numbers beside the verdict
    # say the analysis took a hundredth of a percent of the innovation:
    # the reader has the removed uniform component above to explain why.
    cell = report["scorecard"]["streams"]["probe"]["variables"]["surface_pressure_pa"]["regions"]["global"]
    assert cell["o_minus_a"]["rms"] / cell["o_minus_b"]["rms"] > 0.999


def test_a_pressure_only_analysis_moves_no_theta(spun_up):
    """The fixed-theta convention: the potential temperature field is
    untouched by a pressure analysis, so temperature at a model level
    changes only through that level's pressure (kappa T b dlnps)."""
    cfg, checkpoint = spun_up
    _, _, fields, (theta_b, theta_a) = _analysis_of(cfg, checkpoint, 500.0)
    assert np.array_equal(theta_b, theta_a)
    t_b, t_a = fields["temperature"]
    p_b, p_a = fields["p_full"]
    expected = t_b * ((p_a / p_b) ** KAPPA - 1.0)
    assert np.allclose(t_a - t_b, expected, atol=1e-9)


def _regional_analysis_of(cfg, checkpoint, delta_pa: float, length_scale_km: float = 1000.0):
    """Sixty-four stations in a 20 by 54 degree box (28N to 48N, 238E to
    292E, three percent of the sphere) agreeing the model is ``delta_pa``
    off there."""
    transform = build_transform(cfg)
    model, _ = build_model_and_cold_state(cfg, transform)
    metadata, arrays = read_checkpoint(checkpoint)
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    model.enforce(state)
    terrain = _to_numpy_spectral(
        transform.backend, transform.forward(model.surface_geopotential))
    space = _ModelSpace(
        transform, cfg.vertical, terrain, state.surface,
        cfg.reference_physics.soil_wetness_capacity,
    )
    lats = np.repeat(np.linspace(28.0, 48.0, 8), 8)
    lons = np.tile(np.linspace(238.0, 292.0, 8), 8)
    background = space.hx_surface_pressure(
        state.atmosphere, _Family(_pressure_rows(np.zeros(lats.size), lats, lons)))
    rows = _pressure_rows(background + delta_pa, lats, lons)
    options = AssimilationOptions(length_scale_km=length_scale_km, gate_minimum_count=10**6)
    analysis, report, _ = analyse(
        cfg, model, transform, state, rows, sources=[],
        background={"self_sha256": metadata["self_sha256"], "step": 2, "time_s": 20.0},
        analysis_time=MOMENT, options=options,
    )
    host = transform.backend.to_numpy
    ps_b = np.asarray(host(model.grid_state(state.atmosphere, only=("ps",))["ps"]), dtype=np.float64)
    model.release_syntheses()
    ps_a = np.asarray(host(model.grid_state(analysis.atmosphere, only=("ps",))["ps"]), dtype=np.float64)
    model.release_syntheses()
    grid = transform.grid
    lon2, lat2 = np.meshgrid(np.asarray(grid.longitude_deg), np.asarray(grid.latitude_deg))
    box = (lat2 >= 28.0) & (lat2 <= 48.0) & (lon2 >= 238.0) & (lon2 <= 292.0)
    return report, grid, ps_b, ps_a, box


@pytest.mark.parametrize("delta_pa", [300.0, -300.0])
def test_a_regional_pressure_innovation_is_applied_with_its_sign_and_the_removed_mean_is_recorded(spun_up, delta_pa):
    """The regional family, both signs: the stations get most of what they
    asked for (measured 288.6 of 300 Pa at a 1,000 km length scale), the
    two box columns carry the innovation's sign and size, the global mean
    of surface pressure is unchanged, and the uniform component the rule
    removed carries the innovation's sign and is recorded (40 Pa here,
    four times the box's 9 Pa share, because the T3 truncation rings the
    bump over the sphere).  An analysis step that made a bias of the
    opposite sign at the stations would fail the first bar."""
    cfg, checkpoint = spun_up
    report, grid, ps_b, ps_a, box = _regional_analysis_of(cfg, checkpoint, delta_pa)
    row = report["variables"]["surface_pressure_pa"]
    applied = row["o_minus_b"]["mean"] - row["o_minus_a"]["mean"]
    assert np.sign(applied) == np.sign(delta_pa)
    assert abs(applied) > 0.9 * abs(delta_pa)
    increment = ps_a - ps_b
    assert box.sum() >= 1
    assert np.sign(increment[box].mean()) == np.sign(delta_pa)
    assert abs(increment[box].mean()) > 0.85 * abs(delta_pa)
    assert grid.global_mean(ps_a) == pytest.approx(grid.global_mean(ps_b), abs=1e-6)
    removed = report["mass_preservation"]["uniform_increment_removed_pa"]
    assert np.sign(removed) == np.sign(delta_pa)
    # The box is three percent of the sphere; what the rule removed is far
    # from the whole innovation (the network-wide family above) and is on
    # the record.
    assert abs(removed) < 0.25 * abs(delta_pa)
    assert report["mass_preservation"]["divergence"] == MASS_PRESERVATION_DIVERGENCE
