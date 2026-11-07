"""Typed launchers used by the ordinary Metal moist transport stage."""

from __future__ import annotations

import numpy as np

from woof.wrf_exact import ENABLED as WRF_EXACT


def submit_scalar_update(q, q0, tend, c1h, c2h, mu0, mu, dt_eff, *,
                         msft=None, physics=None, fixed=None, lbc_held=None,
                         spec_zone=0, clear_specified_advection=False,
                         clamp=False):
    """Apply WRF's mass-coupled scalar update in the CUDA ufunc order."""
    if WRF_EXACT and (physics is not None or fixed is not None
                      or lbc_held is not None or clear_specified_advection
                      or clamp):
        raise NotImplementedError(
            "Metal strict scalar update requires its source and boundary "
            "tendencies to be prepared in WRF order")
    nz, ny, nx = q.shape
    if not all(0 < dim <= 0x7FFFFFFF for dim in (nz, ny, nx)):
        raise ValueError("Metal moist scalar extents must fit positive int32")
    count = nz * ny * nx
    if count > 0xFFFFFFFF:
        raise ValueError("Metal moist scalar grid does not fit uint indices")
    has_msf, has_physics, has_fixed, has_lbc = (
        msft is not None, physics is not None, fixed is not None,
        lbc_held is not None)
    msft = mu if msft is None else msft
    physics = q0 if physics is None else physics
    fixed = q0 if fixed is None else fixed
    lbc_held = q0 if lbc_held is None else lbc_held
    arrays = {
        "q0": q0, "tend": tend, "c1h": c1h, "c2h": c2h,
        "mu0": mu0, "mu": mu, "msft": msft,
        "physics": physics, "fixed": fixed, "lbc": lbc_held, "q": q,
    }
    out_lo, out_hi = q.byte_offset, q.byte_offset + q.nbytes
    for name, array in arrays.items():
        if name == "q":
            continue
        if (array.buffer.handle == q.buffer.handle
                and out_lo < array.byte_offset + array.nbytes
                and array.byte_offset < out_hi):
            raise ValueError(f"Metal moist scalar output overlaps {name}")
    params = {
        "has_msf": int(has_msf),
        "has_physics": int(has_physics),
        "has_fixed": int(has_fixed),
        "has_lbc": int(has_lbc),
        "spec_zone": int(spec_zone),
        "clear_specified_advection": int(bool(clear_specified_advection)),
        "clamp": int(bool(clamp)),
        "dt_eff": float(np.float32(dt_eff)),
        "nz": nz, "ny": ny, "nx": nx,
    }
    return q.buffer.context.submit_kernel(
        "wrf_exact_moist_scalar_update" if WRF_EXACT else "moist_scalar_update",
        arrays, params, exact_policy=WRF_EXACT,
        grid_size=count)


def submit_time_t_mass(state):
    """Build ``mub2d + mup0`` for the scalar RK update without host arithmetic."""
    if WRF_EXACT:
        from woof.metal.da_state import binary
        out = state.scratch(state.mup0.shape, "moist_mu0")
        out[...] = binary(state.mub2d, state.mup0)
        return out
    from woof.metal.dycore import submit_total_mu

    return submit_total_mu(
        state, perturbation=state.mup0, scratch_slot="moist_mu0")


def submit_strict_tendency(tend, source, msft, has_msf):
    """Combine WRF's mapped advection and prepared source in source order."""
    nz, ny, nx = tend.shape
    return tend.buffer.context.submit_kernel(
        "wrf_exact_moist_tendency", {"tend": tend, "source": source, "msft": msft},
        {"has_msf": int(bool(has_msf)), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=True, grid_size=tend.size)


def submit_strict_forcing_capture(state, tend, mu0):
    """Export the source-selected qv rate using the pre-summed time-t mass."""
    nz, ny, nx = tend.shape
    return tend.buffer.context.submit_kernel(
        "wrf_exact_moist_capture",
        {"tend": tend, "mu0": mu0, "c1h": state.c1h, "c2h": state.c2h,
         "msft": state.msft, "dst": state.rqvften},
        {"has_msf": int(bool(state.has_msf)),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=True, grid_size=tend.size)
