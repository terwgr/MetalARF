"""The radiation budget carriers the native runtime holds and integrates.

Besides WRF's SWDOWN, GLW, GSW and OLR the runtime holds the upward and
downward shortwave at the top of the radiation column, the upward
longwave at the surface and the column cloud cover, and advances a time
integral of every held plane on every physics call, so two checkpoints
give the exact interval-mean flux the model applied.
"""
from __future__ import annotations

import numpy as np
import pytest

from test_arwen_global_level5_native import (
    FAKE_GLW,
    FAKE_GSW,
    FAKE_OLR,
    _advance,
    _exchange,
    _fake_modules,
    _options,
)

from woof.globe.constants import STEFAN_BOLTZMANN
from woof.globe.physics.native_state import RADIATION_ACCUMULATORS
from woof.globe.physics.native_suite import ArwenCudaColumnSuite


def test_native_runtime_integrates_the_held_planes_on_every_call():
    exchange = _exchange()
    dt = float(exchange.dt_s)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules())
    first = suite.step(exchange)
    meta = first.physics_state.metadata
    arrays = first.physics_state.arrays
    assert meta["radiation_calls"] == 1
    assert meta["lwupb_source"] == "surface_emission_formula"
    assert meta["radiation_accumulated_s"] == pytest.approx(dt)
    np.testing.assert_allclose(arrays["acc_lw_up_top_j_m2"], FAKE_OLR * dt, rtol=1e-6)
    np.testing.assert_allclose(arrays["acc_lw_down_surface_j_m2"], FAKE_GLW * dt, rtol=1e-6)
    np.testing.assert_allclose(arrays["acc_sw_up_surface_j_m2"], (0.0 - FAKE_GSW) * dt, rtol=1e-6)
    emissivity = np.asarray(first.surface.emissivity, np.float64)
    skin = np.asarray(first.surface.temperature_k, np.float64)
    expected_lwupb = emissivity * STEFAN_BOLTZMANN * skin ** 4 + (1.0 - emissivity) * FAKE_GLW
    np.testing.assert_allclose(arrays["lwupb"], expected_lwupb, rtol=1e-5)
    np.testing.assert_allclose(arrays["acc_lw_up_surface_j_m2"], expected_lwupb * dt, rtol=1e-5)
    for accumulator in RADIATION_ACCUMULATORS:
        assert accumulator in arrays, accumulator
    # the fake publishes no top-of-column shortwave or cloud cover: the held
    # planes stay at their zero fill and so do their integrals
    for name in ("swupt", "swdnt", "cldfra_total"):
        assert np.all(arrays[name] == 0.0), name
    assert np.all(arrays["acc_total_cloud_cover_s"] == 0.0)
    diagnostics = first.diagnostics
    assert diagnostics["radiation_calls"] == 1.0
    assert diagnostics["mean_surface_upward_longwave_w_m2"] == pytest.approx(float(np.mean(expected_lwupb)), rel=1e-5)

    # the second call in the same radiation bucket does not call the
    # scheme, and the integrals still advance by the held planes
    second = suite.step(_advance(exchange, first, dt))
    meta = second.physics_state.metadata
    assert meta["radiation_calls"] == 1
    assert meta["radiation_accumulated_s"] == pytest.approx(2.0 * dt)
    np.testing.assert_allclose(second.physics_state.arrays["acc_lw_up_top_j_m2"], FAKE_OLR * 2.0 * dt, rtol=1e-6)
    # the next bucket calls the scheme again
    third = suite.step(_advance(exchange, second, float(_options()["radiation_interval_s"])))
    assert third.physics_state.metadata["radiation_calls"] == 2
    assert third.physics_state.metadata["radiation_accumulated_s"] == pytest.approx(3.0 * dt)
