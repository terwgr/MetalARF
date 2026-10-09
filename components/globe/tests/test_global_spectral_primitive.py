from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from pathlib import Path

import numpy as np
import pytest

from woof.globe.spectral.config import load_config
from woof.globe.spectral.initial_conditions import primitive_rest_state
from woof.globe.spectral.primitive import PrimitiveDryModel, SigmaCoordinate
from woof.globe.spectral.runner import run
from woof.globe.spectral.transform import SphericalHarmonicTransform


ROOT = Path(__file__).resolve().parents[1]
PRIMITIVE_SMOKE = ROOT / str(_shipped_configs() / "global_spectral_primitive_smoke.toml")


def _model_and_state(*, perturbation: float = 0.0, maximum_cfl: float = 0.65):
    transform = SphericalHarmonicTransform.create(8, precision="float64")
    sigma = SigmaCoordinate(np.asarray([0.0, 0.2, 0.5, 0.75, 1.0]))
    model = PrimitiveDryModel(
        transform,
        sigma,
        integrator="ssprk3",
        mass_fixer=True,
        maximum_cfl=maximum_cfl,
    )
    state = primitive_rest_state(
        transform,
        sigma.full_levels,
        perturbation_k=perturbation,
        zonal_wavenumber=3,
    )
    model.initialize_mass_target(state)
    return transform, model, state


def test_horizontally_uniform_hydrostatic_atmosphere_has_zero_rhs():
    transform, model, state = _model_and_state(perturbation=0.0)
    rhs = model.rhs(state)
    maximum = max(
        float(np.max(np.abs(transform.backend.to_numpy(field))))
        for field in rhs.fields()
    )
    assert maximum < 2.0e-18


def test_hydrostatic_geopotential_increases_upward_and_starts_above_surface():
    transform, model, state = _model_and_state(perturbation=0.0)
    temperature = model.grid_state(state)["temperature"]
    phi = transform.backend.to_numpy(model.hydrostatic_geopotential(temperature))
    column = phi[:, 0, 0]
    assert np.all(np.diff(column) < 0.0)  # array is top -> bottom
    assert column[-1] > 0.0


def test_thermal_wave_steps_finite_and_mass_fixer_closes():
    transform, model, state = _model_and_state(perturbation=0.2)
    initial_mass = model.diagnostics(state)["global_mean_surface_pressure_pa"]
    for _ in range(24):
        state, metadata = model.step(state, 300.0)
        assert metadata["spectral_cfl"] < 0.65
    diag = model.diagnostics(state)
    assert diag["max_wind_m_s"] > 0.0
    assert abs(diag["global_mean_surface_pressure_pa"] - initial_mass) / initial_mass < 2.0e-14
    model.enforce(state)


def test_primitive_cfl_refuses_before_advancing():
    _transform, model, state = _model_and_state(
        perturbation=0.2, maximum_cfl=0.01
    )
    with pytest.raises(ValueError, match="spectral CFL"):
        model.step(state, 300.0)
    assert state.step == 0
    assert state.time_s == 0.0


def test_held_suarez_forcing_is_active_and_finite():
    from woof.globe.spectral.primitive import HeldSuarezForcing

    transform, model, state = _model_and_state(perturbation=0.0)
    forcing = HeldSuarezForcing(enabled=True)
    model.held_suarez = forcing
    grid = model.grid_state(state)
    # The rest state's wind is identically zero, so evaluating the drag on it
    # asserts nothing: deleting the Rayleigh friction entirely would leave
    # that measurement at zero.  Drive it with a wind instead.
    wind = transform.backend.xp.full_like(grid["u"], 10.0)
    thermal, drag_u, drag_v = forcing.tendencies(
        model, grid["temperature"], grid["pressure"], wind, wind
    )
    thermal = transform.backend.to_numpy(thermal)
    assert np.isfinite(thermal).all()
    assert np.max(np.abs(thermal)) > 0.0

    sigma = model.sigma.full_levels
    expected_k_f = (
        np.maximum(0.0, (sigma - forcing.sigma_boundary) / (1.0 - forcing.sigma_boundary))
        / (forcing.surface_drag_days * 86_400.0)
    )
    for component in (drag_u, drag_v):
        drag = transform.backend.to_numpy(component)
        assert np.isfinite(drag).all()
        expected = np.broadcast_to(
            -10.0 * expected_k_f[:, None, None], drag.shape
        )
        np.testing.assert_allclose(drag, expected, rtol=1.0e-14, atol=0.0)
    # Boundary-layer levels drag, free-atmosphere levels do not.
    boundary = sigma > forcing.sigma_boundary
    assert boundary.any() and (~boundary).any()
    drag = transform.backend.to_numpy(drag_u)
    assert np.max(np.abs(drag[boundary])) > 0.0
    assert np.max(np.abs(drag[~boundary])) == 0.0


def test_enforce_admits_the_surface_pressure_the_schema_admits():
    # The state carries ln(ps), so a config sitting exactly on the schema's
    # documented ceiling comes back from synthesis a few ulp above it.
    ceiling = 120_000.0
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    sigma = SigmaCoordinate(np.asarray([0.0, 0.15, 0.4, 0.7, 1.0]))
    model = PrimitiveDryModel(transform, sigma)
    state = primitive_rest_state(
        transform, sigma.full_levels, surface_pressure_pa=ceiling
    )
    ps = transform.backend.to_numpy(model.grid_state(state)["ps"])
    assert ps.max() > ceiling
    assert (ps.max() - ceiling) / ceiling < 1.0e-14
    model.enforce(state)

    # Schema and enforcement read the same ceiling, so neither can drift off
    # the other.
    from woof.globe.spectral import config as config_module
    from woof.globe.spectral import primitive as primitive_module

    assert primitive_module.SURFACE_PRESSURE_CEILING_PA == ceiling
    assert config_module.SURFACE_PRESSURE_CEILING_PA == ceiling

    # The bound still refuses a pressure that is actually out of band.
    diverged = primitive_rest_state(
        transform, sigma.full_levels, surface_pressure_pa=119_000.0
    )
    diverged = diverged.with_fields(
        (
            diverged.vorticity,
            diverged.divergence,
            diverged.temperature,
            transform.add_grid_constant(diverged.log_surface_pressure, 0.1),
        )
    )
    with pytest.raises(FloatingPointError, match="surface pressure outside"):
        model.enforce(diverged)


def test_mass_gate_sees_the_loss_the_mass_fixer_absorbs(tmp_path):
    # With the shipped mass_fixer default on, _fix_mass rescales ln(ps) to the
    # cold-state target every step, so surface_pressure_mass_relative_drift is
    # whatever the fixer forced it to be.  The mass the dynamics actually lost
    # is the offset the fixer had to add back, which is gated against the same
    # per-run mass budget.  Measured on this config: drift 2.9e-16, fixer
    # offset 3.8e-13, so a 1.0e-13 budget separates the two.
    config = tmp_path / "tight-mass-budget.toml"
    config.write_text(
        PRIMITIVE_SMOKE.read_text(encoding="utf-8").replace(
            "mass_relative_drift = 1.0e-10", "mass_relative_drift = 1.0e-13"
        ),
        encoding="utf-8",
        newline="\n",
    )
    receipt = run(load_config(config), tmp_path / "out")
    gates = {row["name"]: row for row in receipt["gates"]}
    assert gates["surface_pressure_mass_relative_drift"]["passed"] is True
    assert gates["maximum_mass_fixer_log_offset"]["passed"] is False
    assert receipt["status"] == "fail"
