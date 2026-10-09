"""Host boundary calculations used by public nesting verification cases."""

from __future__ import annotations
import numpy as np


def _lbc_weights(width, spec_zone, relax_zone, dt, spec_exp):
    """WRF ``module_bc_em.F:lbc_fcx_gcx`` in float64."""
    fcx = np.zeros(width, dtype=np.float64)
    gcx = np.zeros(width, dtype=np.float64)
    for loop in range(spec_zone + 1, spec_zone + relax_zone + 1):
        index = loop - 1
        if index >= width:
            break
        ramp = (spec_zone + relax_zone - loop) / (relax_zone - 1)
        sponge = np.exp(-(loop - (spec_zone + 1)) * spec_exp)
        fcx[index] = 0.1 / dt * ramp * sponge
        gcx[index] = 1.0 / dt / 50.0 * ramp * sponge
    return fcx, gcx


def np_specified_relaxation(field, tendency, boundary, *, dtbc, dt,
                            spec_zone=1, relax_zone=4, spec_exp=0.0,
                            apply_relax=True):
    """Mirror WRF ``spec_bdytend`` then ``relax_bdytend_core``.

    Boundary side layouts are west/east ``(nz,ny,width)`` and south/north
    ``(nz,width,nx)``; east/north distance zero is the outermost edge.
    The loop bounds preserve WRF's corner ownership (Y sides own corners).
    """
    field = np.asarray(field, dtype=np.float64)
    out = np.asarray(tendency, dtype=np.float64).copy()
    if field.ndim != 3 or out.shape != field.shape:
        raise ValueError("field and tendency must have the same 3-D shape")
    nz, ny, nx = field.shape
    width = boundary.west.value.shape[-1]
    if width < spec_zone + relax_zone:
        raise ValueError("boundary width is smaller than spec_zone + relax_zone")
    fcx, gcx = _lbc_weights(width, spec_zone, relax_zone, dt, spec_exp)

    def forcing(side, index):
        return side.value[index] + dtbc * side.tendency[index]

    # spec_bdytend: Y sides first and own all four corners.
    for d in range(spec_zone):
        js, jn = d, ny - 1 - d
        for k in range(nz):
            for i in range(d, nx - d):
                out[k, js, i] = boundary.south.tendency[k, d, i]
                out[k, jn, i] = boundary.north.tendency[k, d, i]
        iw, ie = d, nx - 1 - d
        for k in range(nz):
            for j in range(d + 1, ny - d - 1):
                out[k, j, iw] = boundary.west.tendency[k, j, d]
                out[k, j, ie] = boundary.east.tendency[k, j, d]

    if not apply_relax:
        return out

    # relax_bdytend_core: spec_zone <= distance < relax_zone.
    for d in range(spec_zone, relax_zone):
        js, jn = d, ny - 1 - d
        for k in range(nz):
            for i in range(d, nx - d):
                side = boundary.south
                f0 = forcing(side, (k, d, i)) - field[k, js, i]
                f1 = forcing(side, (k, d, max(i - 1, 0))) - field[k, js, max(i - 1, 0)]
                f2 = forcing(side, (k, d, min(i + 1, nx - 1))) - field[k, js, min(i + 1, nx - 1)]
                f3 = forcing(side, (k, d - 1, i)) - field[k, js - 1, i]
                f4 = forcing(side, (k, d + 1, i)) - field[k, js + 1, i]
                out[k, js, i] += fcx[d] * f0 - gcx[d] * (f1 + f2 + f3 + f4 - 4.0 * f0)

                side = boundary.north
                f0 = forcing(side, (k, d, i)) - field[k, jn, i]
                f1 = forcing(side, (k, d, max(i - 1, 0))) - field[k, jn, max(i - 1, 0)]
                f2 = forcing(side, (k, d, min(i + 1, nx - 1))) - field[k, jn, min(i + 1, nx - 1)]
                f3 = forcing(side, (k, d - 1, i)) - field[k, jn + 1, i]
                f4 = forcing(side, (k, d + 1, i)) - field[k, jn - 1, i]
                out[k, jn, i] += fcx[d] * f0 - gcx[d] * (f1 + f2 + f3 + f4 - 4.0 * f0)

        iw, ie = d, nx - 1 - d
        for k in range(nz):
            for j in range(d + 1, ny - d - 1):
                side = boundary.west
                f0 = forcing(side, (k, j, d)) - field[k, j, iw]
                f1 = forcing(side, (k, j - 1, d)) - field[k, j - 1, iw]
                f2 = forcing(side, (k, j + 1, d)) - field[k, j + 1, iw]
                f3 = forcing(side, (k, j, d - 1)) - field[k, j, iw - 1]
                f4 = forcing(side, (k, j, d + 1)) - field[k, j, iw + 1]
                out[k, j, iw] += fcx[d] * f0 - gcx[d] * (f1 + f2 + f3 + f4 - 4.0 * f0)

                side = boundary.east
                f0 = forcing(side, (k, j, d)) - field[k, j, ie]
                f1 = forcing(side, (k, j - 1, d)) - field[k, j - 1, ie]
                f2 = forcing(side, (k, j + 1, d)) - field[k, j + 1, ie]
                f3 = forcing(side, (k, j, d - 1)) - field[k, j, ie + 1]
                f4 = forcing(side, (k, j, d + 1)) - field[k, j, ie - 1]
                out[k, j, ie] += fcx[d] * f0 - gcx[d] * (f1 + f2 + f3 + f4 - 4.0 * f0)
    return out


def np_w_relaxation_current(state, *, source_field=None, dtype=np.float32):
    """WRF nested Davies current-side ``mass_weight(w, mut, c1f, c2f)``.

    Authority: ``module_bc_em.F:327-333,1716-1745``.  Unlike force-time
    donor coupling, this comparison field has no ``msfty`` division.
    """
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be np.float32 or np.float64")

    def host(value):
        if hasattr(value, "get"):
            value = value.get()
        return np.asarray(value, dtype=dtype)

    mub = host(state.mub2d)
    mup = host(state.mup)
    mut = np.asarray(mub + mup, dtype=dtype)
    c1f = host(state.c1f)[:, None, None]
    c2f = host(state.c2f)[:, None, None]
    weight = np.asarray(
        np.asarray(c1f * mut[None], dtype=dtype) + c2f, dtype=dtype)
    w = host(state.w if source_field is None else source_field)
    return np.asarray(weight * w, dtype=dtype)
