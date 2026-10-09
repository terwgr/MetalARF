"""The fused cupy tracer sweeps return the numpy specification's bits.

Device only (skipped without CuPy or under GPUWM_NO_LOCAL_GPU): each
fused kernel is driven on random and adversarial float32 fields through
both paths of ``GridTracerTransport`` and compared with ``array_equal``,
and the whole ``advance`` is compared fused against unfused.  The CPU
half of the proof (the stacked specification against the per-tracer loop
it replaced) is ``test_arwen_global_transport_stacked.py``; the field
families are shared from there.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.globe.transport import GridTracerTransport
from woof.local_gpu import no_local_gpu

from test_arwen_global_transport_stacked import _fields, _transport


# -- device: the fused kernels against the specification ----------------

@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    if no_local_gpu():
        pytest.skip("GPUWM_NO_LOCAL_GPU is set: no device for the fused sweeps")
    return cp


def _equal(cp, a, b) -> bool:
    return bool(cp.array_equal(cp.asarray(a), cp.asarray(b)))


@pytest.mark.parametrize("family", ["random", "spikes", "uniform_polar"])
@pytest.mark.parametrize("seed", [1, 7])
def test_advance_fused_returns_the_specification_bits(cupy, family, seed):
    cp = cupy
    transform, vertical, fused = _transport(backend="cupy", fused=True)
    plain = GridTracerTransport(transform, vertical, fused=False)
    host = _fields(transform, vertical, seed, family=family)
    tracers = {name: cp.asarray(value) for name, value in host[0].items()}
    dp, dp_u, dp_v, omega = (cp.asarray(value) for value in host[1:])
    # The random family's winds are white noise (60 and 40 m/s standard
    # deviations per cell), not a flow any mass field is consistent with:
    # over 900 s its divergent cells shed several times their mass and the
    # pseudo-density goes negative (measured -6.5e4 Pa at seed 1 on the
    # numpy specification, with NaN tracers), where the two paths' NaN
    # arithmetic legitimately differs.  60 s keeps every full-step face
    # fraction under the positivity bound; the two flow families keep
    # the step the polar sub-cycle was designed on.
    dt_s = 60.0 if family == "random" else 900.0
    for step in (0, 1):
        out_f, met_f = fused.advance(
            {k: v.copy() for k, v in tracers.items()}, dp, dp_u, dp_v, omega, dt_s, step=step,
        )
        out_p, met_p = plain.advance(
            {k: v.copy() for k, v in tracers.items()}, dp, dp_u, dp_v, omega, dt_s, step=step,
        )
        assert bool(cp.all(cp.isfinite(met_p["pseudo_density"])))
        assert float(cp.min(met_p["pseudo_density"])) > 0.0
        assert met_f["order"] == met_p["order"]
        for key in ("substeps_x", "substeps_y", "substeps_z"):
            assert met_f[key] == met_p[key]
        assert _equal(cp, met_f["pseudo_density"], met_p["pseudo_density"])
        for name in tracers:
            assert _equal(cp, out_f[name], out_p[name]), (family, seed, step, name)
        assert met_f["floor_clip_kg_m2"] == met_p["floor_clip_kg_m2"]
        if family == "uniform_polar":
            assert met_f["substeps_x"] >= 2
            # Uniform to roundoff: the polar band's 24 sub-steps leave the
            # ratio within eight float32 ulps of 2e-3 (measured five on the
            # RTX 5090, 2026-09-05).
            bound = 8.0 * float(np.finfo(np.float32).eps) * 2.0e-3
            for name in tracers:
                assert float(cp.max(cp.abs(out_f[name] - np.float32(2.0e-3)))) < bound


def test_each_fused_sweep_matches_the_specification(cupy):
    """The three sweep kernels against the numpy expressions, one launch
    each, on the device."""
    cp = cupy
    transform, vertical, fused = _transport(backend="cupy", fused=True)
    plain = GridTracerTransport(transform, vertical, fused=False)
    host = _fields(transform, vertical, 3, family="random")
    tracers = {name: cp.asarray(value) for name, value in host[0].items()}
    dp, dp_u, dp_v, omega = (cp.asarray(value) for value in host[1:])
    names = list(tracers)
    mass = cp.stack([tracers[n] * dp for n in names])
    dt = np.float32(30.0)
    psi_x = 0.5 * (dp_u + cp.roll(dp_u, -1, axis=-1)) * fused._x_scale
    psi_y = 0.5 * (dp_v[:, :-1, :] + dp_v[:, 1:, :]) * fused._y_face
    psi_z = omega[1:-1]
    m_f, d_f = fused._periodic_step(mass, dp, psi_x, dt)
    m_p, d_p = plain._periodic_step(mass, dp, psi_x, dt)
    assert _equal(cp, m_f, m_p) and _equal(cp, d_f, d_p)
    centre, face, inv_width = fused._meridional_coordinates(slice(0, fused.nlat))
    m_f, d_f = fused._walled_step(mass, dp, psi_y, dt, axis=1, centre=centre, face=face, inv_width=inv_width)
    m_p, d_p = plain._walled_step(
        mass, dp, psi_y, dt, axis=1, centre=plain._mu, face=plain._mu_half, inv_width=plain._y_inv_dmu,
    )
    assert _equal(cp, m_f, m_p) and _equal(cp, d_f, d_p)
    p_half, p_full = fused._pressure_coordinates(dp)
    ones = cp.ones_like(dp)
    m_f, d_f = fused._walled_step(mass, dp, psi_z, dt, axis=0, centre=p_full, face=p_half, inv_width=ones)
    m_p, d_p = plain._walled_step(mass, dp, psi_z, dt, axis=0, centre=p_full, face=p_half, inv_width=ones)
    assert _equal(cp, m_f, m_p) and _equal(cp, d_f, d_p)
