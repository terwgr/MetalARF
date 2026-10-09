from __future__ import annotations

import numpy as np
import pytest

from woof.globe.spectral.initial_conditions import williamson2_state
from woof.globe.spectral.shallow_water import ShallowWaterModel
from woof.globe.spectral.transform import SphericalHarmonicTransform


def _max_tendency(model, state) -> float:
    rhs = model.rhs(state)
    return max(
        float(np.max(np.abs(model.transform.backend.to_numpy(field))))
        for field in rhs.fields()
    )


def test_williamson2_alpha_zero_is_discrete_steady_state():
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    model = ShallowWaterModel(transform, integrator="rk4")
    state = williamson2_state(transform, alpha_rad=0.0)
    assert _max_tendency(model, state) < 2.0e-19


def test_williamson2_remains_steady_for_one_day():
    transform = SphericalHarmonicTransform.create(9, precision="float64")
    model = ShallowWaterModel(transform, integrator="rk4", maximum_cfl=0.95)
    state = williamson2_state(transform, alpha_rad=0.0)
    reference = transform.backend.to_numpy(transform.inverse(state.geopotential))
    initial_mass = model.diagnostics(state)["mass_kg_m2_mean"]
    for _ in range(144):
        state = model.step(state, 600.0)
    final = transform.backend.to_numpy(transform.inverse(state.geopotential))
    error = transform.grid.rms(final - reference) / transform.grid.rms(reference)
    mass = model.diagnostics(state)["mass_kg_m2_mean"]
    assert error < 2.0e-13
    assert abs(mass - initial_mass) / initial_mass < 2.0e-14


def test_shallow_water_cfl_refuses_before_advancing():
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    model = ShallowWaterModel(transform, maximum_cfl=0.05)
    state = williamson2_state(transform)
    with pytest.raises(ValueError, match="spectral CFL"):
        model.step(state, 600.0)
    assert state.step == 0
    assert state.time_s == 0.0
