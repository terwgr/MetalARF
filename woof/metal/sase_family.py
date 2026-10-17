"""Metal submission stages for the SASE-L1 turbulence closure.

The public SASE driver is integrated in ``woof.core.physics``.  This module
owns the typed Metal launches; each stage keeps the source operation order and
uses the closure constants from the existing SASE authority.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray, MetalArrayModule
from .runtime import MetalContext
from woof.core.sase_limits import THREADS_PER_BLOCK
from woof.core.sase_math import (
    BLACKADAR_LAMBDA, CP_AIR, E_MIN, EP2_RV, G_ACCEL, KARMAN,
    P0_REF, RD_AIR, RV_AIR, SVP1, SVP2, SVP3, SVPT0, XLV,
    VENT_DEPTH_CAP, VENT_ENT_COEF, VENT_MB_COEF, VENT_MIN_RUN_CELLS,
    VENT_SAT_ADJUST_ITERS, VENT_SIGW_SHARE, VENT_MASK,
    VENT_QT_STEP_CAP, VENT_THETA_STEP_CAP,
    _solve_tail,
)


_TPB = int(THREADS_PER_BLOCK)


def submit_sase_moist_n2(
        context: MetalContext, theta: MetalArray, qv: MetalArray,
        qc: MetalArray, pressure: MetalArray, n2_dry: MetalArray,
        thickness: MetalArray) -> MetalArray:
    """Compute the SASE-M1 moist N² seam, preserving dry bits off-mask."""
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("SASE theta must be a three-dimensional MetalArray")
    for name, value in (("theta", theta), ("qv", qv), ("qc", qc),
                        ("pressure", pressure), ("n2_dry", n2_dry),
                        ("thickness", thickness)):
        _field(name, value, context, theta.shape)
    nz, ny, nx = theta.shape
    if nz > 128:
        raise ValueError("SASE moist N² supports at most 128 levels")
    constants = MetalArrayModule(context).asarray(np.asarray(
        [RD_AIR / CP_AIR, P0_REF, SVP1, SVP2, SVP3, SVPT0,
         EP2_RV, XLV, RD_AIR, CP_AIR, G_ACCEL], dtype=np.float64))
    out = MetalArrayModule(context).empty(theta.shape, dtype=np.float32)
    context.submit_kernel(
        "sase_moist_n2",
        {"theta": theta, "qv": qv, "qc": qc, "pressure": pressure,
         "n2dry": n2_dry, "out": out, "t": thickness,
         "constants": constants},
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ny * nx, threads_per_group=_TPB)
    return out


def submit_sase_plume_vent_flux(
        context: MetalContext, theta: MetalArray, qv: MetalArray,
        qc: MetalArray, pressure: MetalArray, e_sgs: MetalArray,
        n2_moist: MetalArray, n2_dry: MetalArray, rho1: MetalArray, *,
        f_blend: float, thickness: MetalArray,
        indices: bool = False):
    """Diagnose the conditional shallow-plume flux profiles on Metal."""
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("SASE plume inputs must be 3-D MetalArrays")
    nz, ny, nx = theta.shape
    for name, value in (("theta", theta), ("qv", qv), ("qc", qc),
                        ("pressure", pressure), ("e_sgs", e_sgs),
                        ("n2_moist", n2_moist), ("n2_dry", n2_dry),
                        ("thickness", thickness)):
        _field(name, value, context, theta.shape)
    if (not isinstance(rho1, MetalArray) or rho1.shape != (ny, nx)
            or rho1.dtype != np.dtype(np.float32)
            or rho1.buffer.context is not context):
        raise ValueError("SASE rho1 must be a context-local FP32 surface plane")
    if nz > 128 or not np.isfinite(f_blend) or not 0.0 <= f_blend <= 1.0:
        raise ValueError("invalid SASE plume column depth or blend")
    xp = MetalArrayModule(context)
    fshape = (nz + 1, ny, nx)
    fth, fqv, fqc = (xp.empty(fshape, dtype=np.float32) for _ in range(3))
    idx = xp.empty((7, ny, nx) if indices else (1,), dtype=np.int32)
    constants = np.asarray([
        f_blend, RD_AIR / CP_AIR, P0_REF, SVP1, SVP2, SVP3, SVPT0,
        EP2_RV, XLV / CP_AIR, XLV, CP_AIR, RV_AIR / RD_AIR - 1.0,
        G_ACCEL, E_MIN, VENT_MB_COEF, VENT_ENT_COEF,
        VENT_SIGW_SHARE, VENT_DEPTH_CAP], dtype=np.float64).view(np.uint64)
    const_arr = xp.asarray(constants)
    context.submit_kernel(
        "sase_plume_vent_flux",
        {"theta": theta, "qv": qv, "qc": qc, "pressure": pressure,
         "e": e_sgs, "n2m": n2_moist, "n2d": n2_dry, "rho1": rho1,
         "fth": fth, "fqv": fqv, "fqc": fqc, "idx": idx,
         "t": thickness, "constants": const_arr},
        {"min_run_cells": int(VENT_MIN_RUN_CELLS),
         "sat_iters": int(VENT_SAT_ADJUST_ITERS),
         "per_level": int(VENT_MASK == "per-level-theta-es-v1"),
         "has_idx": int(indices), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=_TPB)
    return (fth, fqv, fqc, idx) if indices else (fth, fqv, fqc)


def submit_sase_vent_deposit_scale(
        context: MetalContext, fth: MetalArray, fqv: MetalArray,
        fqc: MetalArray, rho1: MetalArray, thickness: MetalArray, *,
        dt: float) -> MetalArray:
    """Build the FP64 per-column uniform vent-deposit scale."""
    if not isinstance(fth, MetalArray) or fth.ndim != 3:
        raise TypeError("SASE vent fluxes must be 3-D MetalArrays")
    nzp1, ny, nx = fth.shape
    nz = nzp1 - 1
    for name, value in (("fth", fth), ("fqv", fqv), ("fqc", fqc)):
        _field(name, value, context, (nz + 1, ny, nx))
    _field("thickness", thickness, context, (nz, ny, nx))
    if (not isinstance(rho1, MetalArray) or rho1.shape != (ny, nx)
            or rho1.buffer.context is not context):
        raise ValueError("SASE rho1 must match the vent columns")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("SASE vent dt must be positive and finite")
    xp = MetalArrayModule(context)
    scale = xp.empty((ny, nx), dtype=np.float64)
    constants = np.asarray([dt, VENT_THETA_STEP_CAP,
                            VENT_QT_STEP_CAP], dtype=np.float64)
    context.submit_kernel(
        "sase_vent_deposit_scale",
        {"fth": fth, "fqv": fqv, "fqc": fqc, "rho1": rho1,
         "scale": scale, "t": thickness,
         "constants": xp.asarray(constants)},
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ny * nx, threads_per_group=_TPB)
    return scale


def submit_sase_vent_deposit(
        context: MetalContext, phi: MetalArray, f_row: MetalArray,
        scale: MetalArray, rho1: MetalArray, thickness: MetalArray, *,
        dt: float) -> MetalArray:
    """Apply the uniform cap-scaled explicit flux-form vent deposit."""
    if not isinstance(phi, MetalArray) or phi.ndim != 3:
        raise TypeError("SASE deposit target must be a 3-D MetalArray")
    nz, ny, nx = phi.shape
    _field("vent target", phi, context, phi.shape)
    _field("vent flux", f_row, context, (nz + 1, ny, nx))
    if (not isinstance(scale, MetalArray) or scale.shape != (ny, nx)
            or scale.dtype != np.dtype(np.float64)
            or scale.buffer.context is not context):
        raise ValueError("SASE vent scale must be a context-local FP64 plane")
    _field("thickness", thickness, context, phi.shape)
    _field("rho1", rho1, context, (ny, nx))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("SASE vent dt must be positive and finite")
    constants = np.asarray([dt], dtype=np.float64)
    context.submit_kernel(
        "sase_vent_deposit",
        {"phi": phi, "f": f_row, "scale": scale, "rho1": rho1,
         "t": thickness,
         "constants": MetalArrayModule(context).asarray(constants)},
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=nz * ny * nx, threads_per_group=_TPB)
    return phi


def submit_sase_couple_w_tendency(
        context: MetalContext, dw: MetalArray, c1f: MetalArray,
        c2f: MetalArray, mut: MetalArray, msft: MetalArray,
        rw: MetalArray, *, has_msf: bool, mask_ring: bool) -> MetalArray:
    """Mass-couple SASE's half-level vertical-w rate on Metal."""
    if not isinstance(dw, MetalArray) or dw.ndim != 3:
        raise TypeError("SASE dw must be a three-dimensional MetalArray")
    nz, ny, nx = dw.shape
    for name, value in (("mut", mut), ("msft", msft)):
        _field(name, value, context, (ny, nx))
    for name, value in (("c1f", c1f), ("c2f", c2f)):
        _field(name, value, context, (nz + 1,))
    _field("rw", rw, context, (nz + 1, ny, nx))
    context.submit_kernel(
        "sase_couple_w_tendency",
        {"dw": dw, "c1f": c1f, "c2f": c2f, "mut": mut,
         "msft": msft, "rw": rw},
        {"has_msf": int(has_msf), "mask_ring": int(mask_ring),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=(nz + 1) * ny * nx, threads_per_group=_TPB)
    return rw


def _field(name: str, value, context: MetalContext,
           shape: tuple[int, ...]) -> None:
    if (not isinstance(value, MetalArray) or value.shape != shape
            or value.dtype != np.dtype(np.float32)
            or not value.flags.c_contiguous):
        raise ValueError(
            f"SASE {name} must be contiguous FP32 MetalArray with shape {shape}")
    if value.buffer.context is not context:
        raise ValueError(f"SASE {name} belongs to a different Metal context")


def submit_sase_n2(context: MetalContext, theta: MetalArray, *,
                   dz: float | None = None,
                   dz_col: MetalArray | None = None,
                   out: MetalArray | None = None) -> MetalArray:
    """Compute model-column dry N² on SASE's clamped vertical stencil.

    This is the first Metal SASE stage.  Model callers pass ``dz_col`` as a
    per-cell FP32 field; ``dz`` supports uniform test-box callers.
    """
    if not isinstance(context, MetalContext):
        raise TypeError("Metal SASE requires a MetalContext")
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("SASE theta must be a 3-D MetalArray")
    nz, ny, nx = theta.shape
    if min(nz, ny, nx) <= 0:
        raise ValueError("SASE requires nonempty dimensions")
    _field("theta", theta, context, theta.shape)
    if (dz_col is None) == (dz is None):
        raise ValueError("exactly one of dz and dz_col must be supplied")
    if dz_col is not None:
        _field("dz_col", dz_col, context, theta.shape)
    elif (isinstance(dz, bool) or not np.isfinite(dz)
          or float(dz) <= 0.0):
        raise ValueError("SASE dz must be positive and finite")
    if out is None:
        out = MetalArrayModule(context).empty(theta.shape, dtype=np.float32)
    else:
        _field("out", out, context, theta.shape)

    xp = MetalArrayModule(context)
    if dz_col is not None:
        coeffs = tuple(xp.empty(theta.shape, dtype=np.float32)
                       for _ in range(3))
        context.submit_kernel(
            "sase_ddz_coefficients",
            {"t": dz_col, "cm": coeffs[0], "c0": coeffs[1],
             "cp": coeffs[2]},
            {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
            grid_size=ny * nx, threads_per_group=_TPB)
        zargs = dict(cm=coeffs[0], c0=coeffs[1], cp=coeffs[2],
                     dz=0.0, two_dz=0.0, h_lo=0.0, h_hi=0.0,
                     z_mode=3)
    else:
        # Uniform coefficients are unused; the kernel takes the direct
        # centered/one-sided stencil, as ``launch_n2`` does on CUDA.
        coeffs = tuple(xp.zeros(theta.shape, dtype=np.float32)
                       for _ in range(3))
        zargs = dict(cm=coeffs[0], c0=coeffs[1], cp=coeffs[2],
                     dz=float(dz), two_dz=2.0 * float(dz),
                     h_lo=0.0, h_hi=0.0, z_mode=0)
    params = {"two_dx": 0.0, "two_dy": 0.0,
              **zargs, "g_accel": float(G_ACCEL),
              "nz": nz, "ny": ny, "nx": nx}
    arrays = {"theta": theta, "n2": out,
              "cm": params.pop("cm"), "c0": params.pop("c0"),
              "cp": params.pop("cp")}
    context.submit_kernel(
        "sase_n2", arrays, params, exact_policy=False,
        grid_size=nz * ny * nx, threads_per_group=_TPB)
    return out


def submit_sase_box_filter(context: MetalContext, field: MetalArray,
                           width: int, *, out: MetalArray | None = None
                           ) -> MetalArray:
    """Submit SASE's periodic horizontal test filter (width 2 or 4)."""
    if not isinstance(context, MetalContext):
        raise TypeError("Metal SASE requires a MetalContext")
    if not isinstance(field, MetalArray) or field.ndim != 3:
        raise TypeError("SASE filter input must be a 3-D MetalArray")
    _field("filter input", field, context, field.shape)
    if width not in (2, 4):
        raise ValueError("SASE filter width must be 2 or 4")
    nz, ny, nx = field.shape
    xp = MetalArrayModule(context)
    if out is None:
        out = xp.empty(field.shape, dtype=np.float32)
    else:
        _field("filter output", out, context, field.shape)
    context.submit_kernel(
        "sase_box_filter", {"f": field, "out": out},
        {"width": width, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=nz * ny * nx,
        threads_per_group=_TPB)
    return out


def _make_stencil(context: MetalContext, xp: MetalArrayModule,
                  shape: tuple[int, int, int], *,
                  dz: float | None, dz_col: MetalArray | None):
    nz, ny, nx = shape
    if (dz is None) == (dz_col is None):
        raise ValueError("exactly one of dz and dz_col must be supplied")
    if dz_col is not None:
        _field("dz_col", dz_col, context, shape)
        coeffs = tuple(xp.empty(shape, dtype=np.float32) for _ in range(3))
        context.submit_kernel(
            "sase_ddz_coefficients",
            {"t": dz_col, "cm": coeffs[0], "c0": coeffs[1],
             "cp": coeffs[2]},
            {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
            grid_size=ny * nx, threads_per_group=_TPB)
        return coeffs, (0.0, 0.0, 0.0, 0.0, 3)
    if (isinstance(dz, bool) or not np.isfinite(dz)
            or float(dz) <= 0.0):
        raise ValueError("SASE dz must be positive and finite")
    coeffs = tuple(xp.zeros(shape, dtype=np.float32) for _ in range(3))
    return coeffs, (float(dz), 2.0 * float(dz), 0.0, 0.0, 0)


def submit_sase_strain(context: MetalContext, u: MetalArray,
                       v: MetalArray, w: MetalArray, *, dx: float, dy: float,
                       dz: float | None = None,
                       dz_col: MetalArray | None = None
                       ) -> dict[str, MetalArray]:
    """Centered periodic horizontal and clamped vertical strain tensor."""
    if not isinstance(context, MetalContext):
        raise TypeError("Metal SASE requires a MetalContext")
    if not isinstance(u, MetalArray) or u.ndim != 3:
        raise TypeError("SASE wind fields must be 3-D MetalArrays")
    nz, ny, nx = u.shape
    for name, value in (("u", u), ("v", v), ("w", w)):
        _field(name, value, context, u.shape)
    for name, value in (("dx", dx), ("dy", dy)):
        if (isinstance(value, bool) or not np.isfinite(value)
                or float(value) <= 0.0):
            raise ValueError(f"SASE {name} must be positive and finite")
    xp = MetalArrayModule(context)
    coeffs, (dz_v, two_dz, h_lo, h_hi, z_mode) = _make_stencil(
        context, xp, u.shape, dz=dz, dz_col=dz_col)
    names = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
    out = {name: xp.empty(u.shape, dtype=np.float32) for name in names}
    arrays = {"u": u, "v": v, "w": w,
              **dict(zip(names, out.values())),
              "cm": coeffs[0], "c0": coeffs[1], "cp": coeffs[2]}
    context.submit_kernel(
        "sase_strain", arrays,
        {"two_dx": 2.0 * float(dx), "two_dy": 2.0 * float(dy),
         "dz": dz_v, "two_dz": two_dz, "h_lo": h_lo, "h_hi": h_hi,
         "z_mode": z_mode, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=nz * ny * nx,
        threads_per_group=_TPB)
    return out


def submit_sase_e_hflux(context: MetalContext, e: MetalArray,
                        km: MetalArray, *, dx: float, dy: float,
                        e_min: float) -> tuple[MetalArray, MetalArray]:
    """Build the split step's governed horizontal e-transport fluxes."""
    if not isinstance(e, MetalArray) or e.ndim != 3:
        raise TypeError("SASE e must be a 3-D MetalArray")
    _field("e", e, context, e.shape)
    _field("km", km, context, e.shape)
    nz, ny, nx = e.shape
    xp = MetalArrayModule(context)
    fx, fy = (xp.empty(e.shape, dtype=np.float32) for _ in range(2))
    context.submit_kernel(
        "sase_e_hflux", {"e": e, "km": km, "fx": fx, "fy": fy},
        {"two_dx": 2.0 * float(dx), "two_dy": 2.0 * float(dy),
         "e_min": float(e_min), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=nz * ny * nx,
        threads_per_group=_TPB)
    return fx, fy


def submit_sase_model_stress(
        context: MetalContext, e: MetalArray,
        strain: dict[str, MetalArray], *, c_nu: float, f_blend: float,
        delta_eddy: float, delta_mom: float, ck_over_prt: float,
        e_min: float) -> dict[str, MetalArray]:
    """Submit the original blended SASE stress kernel."""
    if not isinstance(e, MetalArray) or e.ndim != 3:
        raise TypeError("SASE e must be a 3-D MetalArray")
    _field("e", e, context, e.shape)
    names = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
    if set(strain) != set(names):
        raise ValueError("SASE strain must contain the six named tensor fields")
    for name in names:
        _field(name, strain[name], context, e.shape)
    scalars = (c_nu, f_blend, delta_eddy, delta_mom, ck_over_prt, e_min)
    if any(isinstance(x, bool) or not np.isfinite(x) for x in scalars):
        raise ValueError("SASE stress parameters must be finite")
    nz, ny, nx = e.shape
    xp = MetalArrayModule(context)
    outputs = {f"t{i}": xp.empty(e.shape, dtype=np.float32)
               for i in range(6)}
    arrays = {"e": e, **{f"s{i}": strain[name]
                         for i, name in enumerate(names)}, **outputs}
    context.submit_kernel(
        "sase_model_stress", arrays,
        {"c_nu": float(c_nu), "f_blend": float(f_blend),
         "delta_eddy": float(delta_eddy), "delta_mom": float(delta_mom),
         "ck_over_prt": float(ck_over_prt), "e_min": float(e_min),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=e.size, threads_per_group=_TPB)
    return outputs


def submit_sase_scalar_hflux(
        context: MetalContext, scalar: MetalArray, e: MetalArray, *,
        two_dx: float, two_dy: float, kh_coef: float,
        e_min: float) -> tuple[MetalArray, MetalArray]:
    """Submit the e-derived horizontal scalar flux pair."""
    if not isinstance(scalar, MetalArray) or scalar.ndim != 3:
        raise TypeError("SASE scalar must be a 3-D MetalArray")
    _field("scalar", scalar, context, scalar.shape)
    _field("e", e, context, scalar.shape)
    if any(isinstance(x, bool) or not np.isfinite(x)
           for x in (two_dx, two_dy, kh_coef, e_min)):
        raise ValueError("SASE scalar flux parameters must be finite")
    nz, ny, nx = scalar.shape
    xp = MetalArrayModule(context)
    fx, fy = (xp.empty(scalar.shape, dtype=np.float32) for _ in range(2))
    context.submit_kernel(
        "sase_scalar_hflux", {"s": scalar, "e": e, "fx": fx, "fy": fy},
        {"two_dx": float(two_dx), "two_dy": float(two_dy),
         "kh_coef": float(kh_coef), "e_min": float(e_min),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=scalar.size, threads_per_group=_TPB)
    return fx, fy


def submit_sase_vent_flux_diag(
        context: MetalContext, f_row: MetalArray, scale: MetalArray,
        out: MetalArray, *, fac: float = 1.0) -> None:
    """Record the cap-scaled venting flux without changing its inputs."""
    if not isinstance(f_row, MetalArray) or f_row.ndim != 3:
        raise TypeError("SASE vent flux must be a 3-D MetalArray")
    _field("f_row", f_row, context, f_row.shape)
    _field("out", out, context, f_row.shape)
    nzp1, ny, nx = f_row.shape
    if (not isinstance(scale, MetalArray) or scale.shape != (ny, nx)
            or scale.dtype != np.dtype(np.float64)
            or not scale.flags.c_contiguous or scale.buffer.context is not context):
        raise ValueError("SASE vent scale must be a context-local FP64 surface field")
    for source in (f_row, scale):
        if (out.buffer is source.buffer
                and out.byte_offset < source.byte_offset + source.nbytes
                and source.byte_offset < out.byte_offset + out.nbytes):
            raise ValueError("SASE vent flux output must not overlap its inputs")
    if not np.isfinite(fac):
        raise ValueError("SASE vent flux unit factor must be finite")
    context.submit_kernel(
        "sase_vent_flux_diag",
        {"f_row": f_row, "scale": scale, "out": out,
         "constants": MetalArrayModule(context).asarray(np.asarray([fac], np.float64))},
        {"nzp1": nzp1, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=f_row.size, threads_per_group=_TPB)


def submit_sase_diff_flux_diag(
        context: MetalContext, phi: MetalArray, kv: MetalArray,
        rho1: MetalArray, thickness: MetalArray, *, dz: float,
        t_mode: int, kfac: float, fac: float) -> MetalArray:
    """Return the dimensional scalar flux on all vertical interfaces."""
    if not isinstance(phi, MetalArray) or phi.ndim != 3:
        raise TypeError("SASE scalar must be a 3-D MetalArray")
    _field("phi", phi, context, phi.shape)
    _field("kv", kv, context, phi.shape)
    _field("thickness", thickness, context, phi.shape)
    nz, ny, nx = phi.shape
    if (not isinstance(rho1, MetalArray) or rho1.shape != (ny, nx)
            or rho1.dtype != np.dtype(np.float32)
            or not rho1.flags.c_contiguous
            or rho1.buffer.context is not context):
        raise ValueError("SASE rho1 must be a context-local FP32 2-D field")
    if t_mode not in (0, 3):
        raise ValueError("Metal SASE flux diagnostics support t_mode 0 or 3")
    if any(isinstance(x, bool) or not np.isfinite(x)
           for x in (dz, kfac, fac)):
        raise ValueError("SASE flux diagnostic constants must be finite")
    xp = MetalArrayModule(context)
    constants = xp.asarray(np.asarray([kfac, fac], dtype=np.float64))
    out = xp.empty((nz + 1, ny, nx), dtype=np.float32)
    context.submit_kernel(
        "sase_diff_flux_diag",
        {"phi": phi, "kv": kv, "rho1": rho1, "out": out,
         "t": thickness, "constants": constants},
        {"dz": float(dz), "t_mode": int(t_mode),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=(nz + 1) * ny * nx, threads_per_group=_TPB)
    return out


def submit_sase_destagger_w(
        context: MetalContext, w: MetalArray, *,
        out: MetalArray | None = None) -> MetalArray:
    """Average interface vertical velocity to SASE's model levels."""
    if not isinstance(w, MetalArray) or w.ndim != 3 or w.shape[0] < 2:
        raise TypeError("SASE staggered w must be a 3-D MetalArray with nz+1 levels")
    nz, ny, nx = w.shape[0] - 1, w.shape[1], w.shape[2]
    _field("staggered w", w, context, (nz + 1, ny, nx))
    if out is None:
        out = MetalArrayModule(context).empty((nz, ny, nx), dtype=np.float32)
    else:
        _field("destaggered w", out, context, (nz, ny, nx))
    context.submit_kernel(
        "sase_destagger_w", {"w": w, "out": out},
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=nz * ny * nx, threads_per_group=_TPB)
    return out


def submit_sase_surface_source(
        context: MetalContext, *, ust: MetalArray, hfx: MetalArray,
        qfx: MetalArray, theta1: MetalArray, rho1: MetalArray,
        dz1: MetalArray, e_sgs: MetalArray, dt: float,
        ep1: float, cp_air: float, g_accel: float, karman: float) -> None:
    """Deposit surface shear and unstable buoyancy production into e_sgs."""
    if not isinstance(e_sgs, MetalArray) or e_sgs.ndim != 3:
        raise TypeError("SASE e_sgs must be a 3-D MetalArray")
    nz, ny, nx = e_sgs.shape
    surface = (ny, nx)
    _field("e_sgs", e_sgs, context, e_sgs.shape)
    for name, value in (("ust", ust), ("hfx", hfx), ("qfx", qfx),
                        ("theta1", theta1), ("rho1", rho1), ("dz1", dz1)):
        _field(name, value, context, surface)
    values = (dt, ep1, cp_air, g_accel, karman)
    if any(isinstance(x, bool) or not np.isfinite(x) for x in values):
        raise ValueError("SASE surface source constants must be finite")
    context.submit_kernel(
        "sase_surface_source",
        {"ust": ust, "hfx": hfx, "qfx": qfx, "theta1": theta1,
         "rho1": rho1, "dz1": dz1, "e_sgs": e_sgs},
        {"ep1": float(ep1), "cp_air": float(cp_air),
         "g_accel": float(g_accel), "karman": float(karman),
         "dt": float(dt), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=_TPB)


def submit_sase_rate(
        context: MetalContext, a: MetalArray, b: MetalArray,
        out: MetalArray, *, mode: int, inv_dt: float,
        heat_scale: float = 0.0) -> MetalArray:
    """Convert rates, add heating, or form ``a + inv_dt*b`` in place."""
    if mode not in (0, 1, 2):
        raise ValueError("SASE rate mode must be 0 (difference), 1 (heat), or 2 (axpy)")
    if not isinstance(a, MetalArray) or a.ndim != 3:
        raise TypeError("SASE rate inputs must be 3-D MetalArrays")
    for name, value in (("a", a), ("b", b), ("out", out)):
        _field(name, value, context, a.shape)
    if any(isinstance(x, bool) or not np.isfinite(x)
           for x in (inv_dt, heat_scale)):
        raise ValueError("SASE rate constants must be finite")
    nz, ny, nx = a.shape
    context.submit_kernel(
        "sase_rate", {"a": a, "b": b, "out": out},
        {"mode": int(mode), "inv_dt": float(inv_dt),
         "heat_scale": float(heat_scale), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=a.size, threads_per_group=_TPB)
    return out


def submit_sase_structure_partial(
        context: MetalContext, field: MetalArray) -> MetalArray:
    """Accumulate FP64 horizontal structure-function sums by block."""
    if not isinstance(field, MetalArray) or field.ndim != 3:
        raise TypeError("SASE structure field must be a 3-D MetalArray")
    _field("structure field", field, context, field.shape)
    nz, ny, nx = field.shape
    nblocks = (field.size + _TPB - 1) // _TPB
    partials = MetalArrayModule(context).empty((3, nblocks), dtype=np.float64)
    context.submit_kernel(
        "sase_structure_partial", {"f": field, "partials": partials},
        {"nz": nz, "ny": ny, "nx": nx, "nblocks": nblocks},
        exact_policy=False, grid_size=nblocks * _TPB,
        threads_per_group=_TPB)
    return partials


def submit_sase_w_sensor_partial(
        context: MetalContext, w: MetalArray, e: MetalArray, *,
        n2: MetalArray | None, n2_screen: float, e_min: float,
        boundary_width: int = 0) -> MetalArray:
    """Reduce N²-screened w structure sums and interior mean energy."""
    if not isinstance(w, MetalArray) or w.ndim != 3:
        raise TypeError("SASE w must be a 3-D MetalArray")
    _field("w", w, context, w.shape)
    _field("e", e, context, w.shape)
    if n2 is not None:
        _field("n2", n2, context, w.shape)
    bw = int(boundary_width)
    nz, ny, nx = w.shape
    if bw < 0 or (bw and (2 * bw >= ny or 2 * bw >= nx)):
        raise ValueError("SASE boundary width must leave a nonempty interior")
    if any(isinstance(x, bool) or not np.isfinite(x)
           for x in (n2_screen, e_min)):
        raise ValueError("SASE sensor constants must be finite")
    nblocks = (w.size + _TPB - 1) // _TPB
    xp = MetalArrayModule(context)
    partials = xp.empty((5, nblocks), dtype=np.float64)
    n2_arg = e if n2 is None else n2
    context.submit_kernel(
        "sase_w_sensor_partial",
        {"w": w, "e": e, "n2": n2_arg, "partials": partials},
        {"nz": nz, "ny": ny, "nx": nx, "bw": bw,
         "has_n2": int(n2 is not None), "n2_screen": float(n2_screen),
         "e_min": float(e_min), "nblocks": nblocks},
        exact_policy=False, grid_size=nblocks * _TPB,
        threads_per_group=_TPB)
    return partials


def submit_sase_zi_column(
        context: MetalContext, u: MetalArray, v: MetalArray,
        theta: MetalArray, thickness: MetalArray, *, dz: float,
        t_mode: int, rib_crit: float, wspd2_floor: float,
        g_accel: float) -> MetalArray:
    """Diagnose one bulk-Richardson boundary-layer height per column."""
    if not isinstance(u, MetalArray) or u.ndim != 3:
        raise TypeError("SASE winds must be 3-D MetalArrays")
    _field("u", u, context, u.shape)
    for name, value in (("v", v), ("theta", theta),
                        ("thickness", thickness)):
        _field(name, value, context, u.shape)
    nz, ny, nx = u.shape
    if t_mode not in (0, 3):
        raise ValueError("Metal SASE z_i supports t_mode 0 or 3")
    constants = (dz, rib_crit, wspd2_floor, g_accel)
    if any(isinstance(x, bool) or not np.isfinite(x) for x in constants):
        raise ValueError("SASE z_i constants must be finite")
    zi = MetalArrayModule(context).empty((ny, nx), dtype=np.float32)
    context.submit_kernel(
        "sase_zi_column",
        {"u": u, "v": v, "theta": theta, "t": thickness, "zi": zi},
        {"dz": float(dz), "t_mode": int(t_mode),
         "rib_crit": float(rib_crit), "wspd2_floor": float(wspd2_floor),
         "g_accel": float(g_accel), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=_TPB)
    return zi


def submit_sase_thomas_scalar(
        context: MetalContext, phi: MetalArray, kv: MetalArray,
        thickness: MetalArray, *, dz: float, t_mode: int, dt: float,
        kfac: float, floor_val: float = 0.0, has_floor: bool = False,
        sfc_flux: MetalArray | None = None,
        sfc_rho1: MetalArray | None = None,
        sfc_fac: float = 1.0) -> MetalArray:
    """Advance scalar columns with the software FP64 Thomas sweep."""
    if not isinstance(phi, MetalArray) or phi.ndim != 3:
        raise TypeError("SASE scalar must be a 3-D MetalArray")
    _field("phi", phi, context, phi.shape)
    _field("kv", kv, context, phi.shape)
    _field("thickness", thickness, context, phi.shape)
    nz, ny, nx = phi.shape
    if nz > 128:
        raise ValueError("SASE Thomas scalar supports at most 128 levels")
    if t_mode not in (0, 3):
        raise ValueError("Metal SASE Thomas scalar supports t_mode 0 or 3")
    xp = MetalArrayModule(context)
    surface = (ny, nx)
    if sfc_flux is None:
        sfc_flux = xp.zeros(surface, dtype=np.float32)
        has_sfc = False
    else:
        _field("sfc_flux", sfc_flux, context, surface)
        has_sfc = True
    if sfc_rho1 is None:
        sfc_rho1 = xp.ones(surface, dtype=np.float32)
    else:
        _field("sfc_rho1", sfc_rho1, context, surface)
    constants = (dz, dt, kfac, floor_val, sfc_fac)
    if any(isinstance(x, bool) or not np.isfinite(x) for x in constants):
        raise ValueError("SASE Thomas scalar constants must be finite")
    packed = xp.asarray(np.asarray([sfc_fac], dtype=np.float64))
    context.submit_kernel(
        "sase_thomas_scalar",
        {"phi": phi, "kv": kv, "sfc_flux": sfc_flux,
         "sfc_rho1": sfc_rho1, "t": thickness, "constants": packed},
        {"dz": float(dz), "t_mode": int(t_mode), "dt": float(dt),
         "kfac": float(kfac), "floor_val": float(floor_val),
         "has_floor": int(bool(has_floor)), "has_sfc": int(has_sfc),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ny * nx, threads_per_group=_TPB)


def submit_sase_surface_rho1(
        context: MetalContext, pressure: MetalArray,
        temperature: MetalArray, qv: MetalArray, *, rd: float,
        ep1: float, out: MetalArray | None = None) -> MetalArray:
    """Compute the shared lowest-level moist density plane."""
    if not isinstance(pressure, MetalArray) or pressure.ndim != 2:
        raise TypeError("SASE pressure plane must be a 2-D MetalArray")
    ny, nx = pressure.shape
    for name, value in (("pressure", pressure),
                        ("temperature", temperature), ("qv", qv)):
        _field(name, value, context, (ny, nx))
    if out is None:
        out = MetalArrayModule(context).empty((ny, nx), dtype=np.float32)
    else:
        _field("rho1", out, context, (ny, nx))
    if any(isinstance(x, bool) or not np.isfinite(x) for x in (rd, ep1)):
        raise ValueError("SASE density constants must be finite")
    context.submit_kernel(
        "sase_surface_rho1",
        {"pressure": pressure, "temperature": temperature,
         "qv": qv, "rho1": out},
        {"rd": float(rd), "ep1": float(ep1), "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=_TPB)
    return out
    return phi


def submit_sase_thomas_momentum(
        context: MetalContext, u: MetalArray, v: MetalArray,
        w: MetalArray, du: MetalArray, dv: MetalArray, dw: MetalArray,
        kv: MetalArray, thickness: MetalArray, *, dz: float, t_mode: int,
        dt: float, ust: MetalArray | None = None,
        wspd: MetalArray | None = None,
        sfc_floor: float) -> MetalArray:
    """Solve momentum columns and return the three FP64 ledger channels."""
    if not isinstance(u, MetalArray) or u.ndim != 3:
        raise TypeError("SASE winds must be 3-D MetalArrays")
    for name, value in (("u", u), ("v", v), ("w", w), ("du", du),
                        ("dv", dv), ("dw", dw), ("kv", kv),
                        ("thickness", thickness)):
        _field(name, value, context, u.shape)
    nz, ny, nx = u.shape
    ncol = ny * nx
    if nz > 128:
        raise ValueError("SASE momentum Thomas solve supports at most 128 levels")
    if t_mode not in (0, 3):
        raise ValueError("Metal SASE Thomas momentum supports t_mode 0 or 3")
    if wspd is not None and ust is None:
        raise ValueError("SASE wspd requires ust")
    xp = MetalArrayModule(context)
    has_drag = ust is not None
    has_wspd = wspd is not None
    if ust is None:
        ust = xp.zeros((ny, nx), dtype=np.float32)
    else:
        _field("ust", ust, context, (ny, nx))
    if wspd is None:
        wspd = xp.zeros((ny, nx), dtype=np.float32)
    else:
        _field("wspd", wspd, context, (ny, nx))
    if any(isinstance(x, bool) or not np.isfinite(x)
           for x in (dz, dt, sfc_floor)):
        raise ValueError("SASE momentum constants must be finite")
    nblocks = (ncol + _TPB - 1) // _TPB
    partials = xp.empty((3, nblocks), dtype=np.float64)
    constants = xp.asarray(np.asarray([sfc_floor], dtype=np.float64))
    context.submit_kernel(
        "sase_thomas_momentum",
        {"u": u, "v": v, "w": w, "du": du, "dv": dv, "dw": dw,
         "kv": kv, "ust": ust, "wspd": wspd, "t": thickness,
         "partials": partials, "constants": constants},
        {"dz": float(dz), "t_mode": int(t_mode),
         "has_drag": int(has_drag), "has_wspd": int(has_wspd),
         "dt": float(dt), "nz": nz, "ny": ny, "nx": nx,
         "nblocks": nblocks}, exact_policy=False,
        grid_size=nblocks * _TPB, threads_per_group=_TPB)
    return partials


def submit_sase_vertical_channel(
        context: MetalContext, e: MetalArray, theta: MetalArray,
        thickness: MetalArray, *, dz: float, t_mode: int, f_blend: float,
        z0: float, c_kv: float, ls_coef: float, e_min: float,
        g_accel: float, karman: float, blk_lambda: float, c_ks: float,
        cks_exp: float, n2: MetalArray | None = None,
        n2_dry: MetalArray | None = None,
        kv: MetalArray | None = None,
        leps: MetalArray | None = None
        ) -> tuple[MetalArray, MetalArray]:
    """Build SASE vertical diffusivity and dissipation lengths per column."""
    if not isinstance(e, MetalArray) or e.ndim != 3:
        raise TypeError("SASE e must be a 3-D MetalArray")
    _field("e", e, context, e.shape)
    _field("theta", theta, context, e.shape)
    _field("thickness", thickness, context, e.shape)
    if n2 is not None:
        _field("n2", n2, context, e.shape)
    if n2_dry is not None:
        if n2 is None:
            raise ValueError("SASE n2_dry requires n2")
        _field("n2_dry", n2_dry, context, e.shape)
    nz, ny, nx = e.shape
    if nz > 128:
        raise ValueError("SASE vertical channel supports at most 128 levels")
    if t_mode not in (0, 3):
        raise ValueError("Metal SASE vertical channel supports t_mode 0 or 3")
    values = (dz, f_blend, z0, c_kv, ls_coef, e_min, g_accel,
              karman, blk_lambda, c_ks, cks_exp)
    if any(isinstance(x, bool) or not np.isfinite(x) for x in values):
        raise ValueError("SASE vertical channel constants must be finite")
    xp = MetalArrayModule(context)
    n2_arg = e if n2 is None else n2
    n2d_arg = e if n2_dry is None else n2_dry
    if kv is None:
        kv = xp.empty(e.shape, dtype=np.float32)
    else:
        _field("kv", kv, context, e.shape)
    if leps is None:
        leps = xp.empty(e.shape, dtype=np.float32)
    else:
        _field("leps", leps, context, e.shape)
    context.submit_kernel(
        "sase_vertical_channel",
        {"e": e, "n2": n2_arg, "n2d": n2d_arg,
         "theta": theta, "t": thickness, "kv": kv, "leps": leps},
        {"z0": float(z0), "karman": float(karman),
         "blk_lambda": float(blk_lambda), "c_kv": float(c_kv),
         "ls_coef": float(ls_coef), "e_min": float(e_min),
         "f_blend": float(f_blend), "g_accel": float(g_accel),
         "c_ks": float(c_ks), "cks_exp": float(cks_exp),
         "dz": float(dz), "t_mode": int(t_mode),
         "has_n2": int(n2 is not None),
         "has_moist": int(n2_dry is not None),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ny * nx, threads_per_group=_TPB)
    return kv, leps


def submit_sase_split_e_update(
        context: MetalContext, e: MetalArray, heat: MetalArray,
        theta: MetalArray, n2: MetalArray | None, n2_moist: MetalArray | None,
        kv: MetalArray, leps: MetalArray, ph_e: MetalArray,
        ph_heat: MetalArray, pv: MetalArray, fx: MetalArray, fy: MetalArray,
        *, dx: float, dy: float, dz: float | None, dz_col: MetalArray | None,
        dt: float, f_blend: float, delta: float, pr_t: float,
        c_e: float, ls_coef: float, c_es: float, cks_exp: float,
        c_ed: float, g_accel: float, e_min: float,
        stable_dissipation: bool = False,
        taper: MetalArray | None = None,
        lb: MetalArray | None = None) -> MetalArray:
    """Update e and heat through the split source, decay and floor stage."""
    if not isinstance(e, MetalArray) or e.ndim != 3:
        raise TypeError("SASE e must be a 3-D MetalArray")
    for name, value in (("e", e), ("heat", heat), ("theta", theta),
                        ("kv", kv), ("leps", leps), ("ph_e", ph_e),
                        ("ph_heat", ph_heat), ("pv", pv), ("fx", fx),
                        ("fy", fy)):
        _field(name, value, context, e.shape)
    if n2 is not None:
        _field("n2", n2, context, e.shape)
    if n2_moist is not None:
        if n2 is None:
            raise ValueError("SASE n2_moist requires dry n2")
        _field("n2_moist", n2_moist, context, e.shape)
    nz, ny, nx = e.shape
    xp = MetalArrayModule(context)
    coeffs, (dz_v, two_dz, h_lo, h_hi, z_mode) = _make_stencil(
        context, xp, e.shape, dz=dz, dz_col=dz_col)
    if taper is None:
        taper_arg, has_taper = e, 0
    else:
        _field("taper", taper, context, e.shape)
        taper_arg, has_taper = taper, 1
    if lb is None:
        lb_arg, has_ced = leps, 0
    else:
        _field("lb", lb, context, e.shape)
        lb_arg, has_ced = lb, 1
    n2_eff = n2 if n2_moist is None else n2_moist
    n2_arg = e if n2_eff is None else n2_eff
    n2d_arg = e if n2_moist is None else n2
    values = (dx, dy, dt, f_blend, delta, pr_t, c_e, ls_coef, c_es,
              cks_exp, c_ed, g_accel, e_min)
    if any(isinstance(x, bool) or not np.isfinite(x) for x in values):
        raise ValueError("SASE e-update constants must be finite")
    ncell = e.size
    nblocks = (ncell + _TPB - 1) // _TPB
    partials = xp.empty((4, nblocks), dtype=np.float64)
    context.submit_kernel(
        "sase_split_e_update",
        {"e": e, "heat": heat, "theta": theta, "n2": n2_arg,
         "n2d": n2d_arg, "kv": kv, "leps": leps, "lb": lb_arg,
         "ph_e": ph_e, "ph_heat": ph_heat, "pv": pv, "fx": fx,
         "fy": fy, "gtap": taper_arg, "partials": partials,
         "cm": coeffs[0], "c0": coeffs[1], "cp": coeffs[2]},
        {"two_dx": 2.0 * float(dx), "two_dy": 2.0 * float(dy),
         "dz": dz_v, "two_dz": two_dz, "h_lo": h_lo, "h_hi": h_hi,
         "z_mode": int(z_mode), "dt": float(dt),
         "f_blend": float(f_blend), "delta": float(delta),
         "pr_t": float(pr_t), "c_e": float(c_e),
         "ls_coef": float(ls_coef), "c_es": float(c_es),
         "cks_exp": float(cks_exp), "has_ces": int(stable_dissipation),
         "c_ed": float(c_ed), "g_accel": float(g_accel),
         "e_min": float(e_min), "has_ced": int(has_ced),
         "has_n2": int(n2_eff is not None),
         "has_moist": int(n2_moist is not None),
         "has_taper": has_taper, "nblocks": nblocks,
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=nblocks * _TPB, threads_per_group=_TPB)
    return partials


def submit_sase_taper_weights(
        context: MetalContext, thickness: MetalArray, *,
        zdamp: float) -> MetalArray:
    """Build the KDH production-taper weights from column thicknesses."""
    if not isinstance(thickness, MetalArray) or thickness.ndim != 3:
        raise TypeError("SASE thickness must be a 3-D MetalArray")
    _field("thickness", thickness, context, thickness.shape)
    if isinstance(zdamp, bool) or not np.isfinite(zdamp) or zdamp <= 0.0:
        raise ValueError("SASE zdamp must be positive and finite")
    nz, ny, nx = thickness.shape
    out = MetalArrayModule(context).empty(thickness.shape, dtype=np.float32)
    context.submit_kernel(
        "sase_taper_weights", {"t": thickness, "out": out},
        {"zdamp": float(zdamp), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=_TPB)
    return out


def submit_sase_scalar_mix(context: MetalContext, scalar: MetalArray,
                           km: MetalArray, *, dx: float, dy: float,
                           pr_t: float) -> MetalArray:
    """Run the horizontal half of the governed scalar-mix channel."""
    if not isinstance(scalar, MetalArray) or scalar.ndim != 3:
        raise TypeError("SASE scalar must be a 3-D MetalArray")
    _field("scalar", scalar, context, scalar.shape)
    _field("km", km, context, scalar.shape)
    if (isinstance(pr_t, bool) or not np.isfinite(pr_t)
            or float(pr_t) <= 0.0):
        raise ValueError("SASE Prandtl number must be positive and finite")
    nz, ny, nx = scalar.shape
    xp = MetalArrayModule(context)
    fx, fy, out = (xp.empty(scalar.shape, dtype=np.float32)
                   for _ in range(3))
    context.submit_kernel(
        "sase_scalar_hflux_km",
        {"s": scalar, "km": km, "fx": fx, "fy": fy},
        {"two_dx": 2.0 * float(dx), "two_dy": 2.0 * float(dy),
         "kh_fac": 1.0 / float(pr_t), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=nz * ny * nx,
        threads_per_group=_TPB)
    context.submit_kernel(
        "sase_hflux_div", {"fx": fx, "fy": fy, "out": out},
        {"two_dx": 2.0 * float(dx), "two_dy": 2.0 * float(dy),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=nz * ny * nx, threads_per_group=_TPB)
    return out


def submit_sase_model_stress_gov(
        context: MetalContext, e: MetalArray, strain: dict[str, MetalArray], *,
        c_nu: float, f_blend: float, delta: float, c_s: float,
        km_cap: float, nu_eps: float, e_min: float
        ) -> dict[str, MetalArray]:
    """Build the governed SASE stress, diffusivity, and production split."""
    if not isinstance(e, MetalArray) or e.ndim != 3:
        raise TypeError("SASE e must be a 3-D MetalArray")
    _field("e", e, context, e.shape)
    nz, ny, nx = e.shape
    stress_names = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
    if set(strain) != set(stress_names):
        raise ValueError("SASE strain must contain the six named tensor fields")
    for name in stress_names:
        _field(name, strain[name], context, e.shape)
    for name, value in (("c_nu", c_nu), ("f_blend", f_blend),
                        ("delta", delta), ("c_s", c_s),
                        ("km_cap", km_cap), ("nu_eps", nu_eps),
                        ("e_min", e_min)):
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"SASE {name} must be finite")
    xp = MetalArrayModule(context)
    out = {name: xp.empty(e.shape, dtype=np.float32)
           for name in ("t0", "t1", "t2", "t3", "t4", "t5", "km", "r")}
    arrays = {"e": e, **{f"s{i}": strain[name]
                         for i, name in enumerate(stress_names)},
              **{name: out[name] for name in
                 ("t0", "t1", "t2", "t3", "t4", "t5", "km")},
              "r_out": out["r"]}
    context.submit_kernel(
        "sase_model_stress_gov", arrays,
        {"c_nu": float(c_nu), "f_blend": float(f_blend),
         "delta": float(delta), "c_s": float(c_s),
         "km_cap": float(km_cap), "nu_eps": float(nu_eps),
         "e_min": float(e_min), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=e.size, threads_per_group=_TPB)
    return out


def submit_sase_split_tendencies(
        context: MetalContext, stress: dict[str, MetalArray],
        u: MetalArray, v: MetalArray, w: MetalArray, e: MetalArray,
        r: MetalArray, *, dx: float, dy: float, e_min: float
        ) -> dict[str, MetalArray]:
    """Form explicit horizontal tendencies and split production terms."""
    if not isinstance(u, MetalArray) or u.ndim != 3:
        raise TypeError("SASE winds must be 3-D MetalArrays")
    nz, ny, nx = u.shape
    for name, value in (("u", u), ("v", v), ("w", w),
                        ("e", e), ("r", r)):
        _field(name, value, context, u.shape)
    required = ("t0", "t1", "t3", "t4", "t5")
    for name in required:
        _field(name, stress[name], context, u.shape)
    xp = MetalArrayModule(context)
    out = {name: xp.empty(u.shape, dtype=np.float32)
           for name in ("du", "dv", "dw", "ph_e", "ph_heat")}
    arrays = {**{name: stress[name] for name in required},
              "u": u, "v": v, "w": w, "e": e, "r": r, **out}
    context.submit_kernel(
        "sase_split_tendencies", arrays,
        {"two_dx": 2.0 * float(dx), "two_dy": 2.0 * float(dy),
         "e_min": float(e_min), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=nz * ny * nx,
        threads_per_group=_TPB)
    return out


def submit_sase_velocity_products(
        context: MetalContext, u: MetalArray, v: MetalArray,
        w: MetalArray) -> dict[str, MetalArray]:
    """Form the six fine-level velocity products used by the Germano lift."""
    if not isinstance(u, MetalArray) or u.ndim != 3:
        raise TypeError("SASE winds must be 3-D MetalArrays")
    for name, value in (("u", u), ("v", v), ("w", w)):
        _field(name, value, context, u.shape)
    xp = MetalArrayModule(context)
    names = ("uu", "vv", "ww", "uv", "uw", "vw")
    out = {name: xp.empty(u.shape, dtype=np.float32) for name in names}
    arrays = {"u": u, "v": v, "w": w, **out}
    nz, ny, nx = u.shape
    context.submit_kernel(
        "sase_velocity_products", arrays,
        {"nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=u.size, threads_per_group=_TPB)
    return out


def submit_sase_lift_combine(
        context: MetalContext, filtered_products: dict[str, MetalArray],
        filtered_velocity: dict[str, MetalArray]
        ) -> dict[str, MetalArray]:
    """Finish the six Germano lift tensors from their filtered products."""
    product_names = ("fuu", "fvv", "fww", "fuv", "fuw", "fvw")
    velocity_names = ("fu", "fv", "fw")
    if set(filtered_products) != set(product_names):
        raise ValueError("filtered products must be fuu/fvv/fww/fuv/fuw/fvw")
    if set(filtered_velocity) != set(velocity_names):
        raise ValueError("filtered velocity must be fu/fv/fw")
    first = filtered_products["fuu"]
    if not isinstance(first, MetalArray) or first.ndim != 3:
        raise TypeError("SASE filtered fields must be 3-D MetalArrays")
    for name in product_names:
        _field(name, filtered_products[name], context, first.shape)
    for name in velocity_names:
        _field(name, filtered_velocity[name], context, first.shape)
    xp = MetalArrayModule(context)
    names = ("l0", "l1", "l2", "l3", "l4", "l5")
    out = {name: xp.empty(first.shape, dtype=np.float32) for name in names}
    arrays = {**filtered_products, **filtered_velocity, **out}
    nz, ny, nx = first.shape
    context.submit_kernel(
        "sase_lift_combine", arrays,
        {"nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=first.size,
        threads_per_group=_TPB)
    return out


def submit_sase_vertical_production(
        context: MetalContext, u: MetalArray, v: MetalArray, w: MetalArray,
        kv: MetalArray, dz_col: MetalArray) -> MetalArray:
    """Diagnose implicit vertical production from the solved velocities."""
    if not isinstance(u, MetalArray) or u.ndim != 3:
        raise TypeError("SASE winds must be 3-D MetalArrays")
    for name, value in (("u", u), ("v", v), ("w", w),
                        ("kv", kv), ("dz_col", dz_col)):
        _field(name, value, context, u.shape)
    nz, ny, nx = u.shape
    out = MetalArrayModule(context).empty(u.shape, dtype=np.float32)
    context.submit_kernel(
        "sase_vertical_production",
        {"u": u, "v": v, "w": w, "kv": kv, "t": dz_col, "pv": out},
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=nz * ny * nx, threads_per_group=_TPB)
    return out


def submit_sase_blackadar_length(context: MetalContext, dz_col: MetalArray, *,
                                 z0: float) -> MetalArray:
    """Build the geometry-only Blackadar reference length field."""
    if not isinstance(dz_col, MetalArray) or dz_col.ndim != 3:
        raise TypeError("SASE dz_col must be a 3-D MetalArray")
    _field("dz_col", dz_col, context, dz_col.shape)
    if isinstance(z0, bool) or not np.isfinite(z0):
        raise ValueError("SASE z0 must be finite")
    nz, ny, nx = dz_col.shape
    xp = MetalArrayModule(context)
    consts = xp.asarray(np.asarray(
        [z0, KARMAN, BLACKADAR_LAMBDA], dtype=np.float64))
    out = xp.empty(dz_col.shape, dtype=np.float32)
    context.submit_kernel(
        "sase_blackadar_length",
        {"t": dz_col, "constants": consts, "lb": out},
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ny * nx, threads_per_group=_TPB)
    return out


def submit_sase_basis_premultiply(
        context: MetalContext, e: MetalArray,
        strain: dict[str, MetalArray], *, delta: float,
        e_min: float) -> dict[str, MetalArray]:
    """Prepare six deviatoric fine-strain fields for the dynamic solve."""
    if not isinstance(e, MetalArray) or e.ndim != 3:
        raise TypeError("SASE e must be a 3-D MetalArray")
    _field("e", e, context, e.shape)
    names = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
    if set(strain) != set(names):
        raise ValueError("SASE strain must contain the six named tensor fields")
    for name in names:
        _field(name, strain[name], context, e.shape)
    if any(isinstance(x, bool) or not np.isfinite(x)
           for x in (delta, e_min)):
        raise ValueError("SASE delta and E_MIN must be finite")
    xp = MetalArrayModule(context)
    out = {f"p{i}": xp.empty(e.shape, dtype=np.float32) for i in range(6)}
    arrays = {"e": e, **{f"s{i}": strain[name]
                         for i, name in enumerate(names)}, **out}
    nz, ny, nx = e.shape
    context.submit_kernel(
        "sase_basis_premultiply", arrays,
        {"delta": float(delta), "e_min": float(e_min),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=e.size, threads_per_group=_TPB)
    return out


def submit_sase_solve_partial(
        context: MetalContext, e: MetalArray,
        coarse: dict[str, MetalArray], refiltered: dict[str, MetalArray],
        lift: dict[str, MetalArray], *, delta_a: float, delta_b: float,
        e_min: float, boundary_width: int = 0) -> MetalArray:
    """Accumulate one width's FP64 Gram/projection partials."""
    if not isinstance(e, MetalArray) or e.ndim != 3:
        raise TypeError("SASE e must be a 3-D MetalArray")
    _field("e", e, context, e.shape)
    names = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
    for label, values, expected in (
            ("coarse strain", coarse, names),
            ("refiltered strain", refiltered, tuple(f"p{i}" for i in range(6))),
            ("lift", lift, tuple(f"l{i}" for i in range(6)))):
        if set(values) != set(expected):
            raise ValueError(f"SASE {label} fields do not match their six-slot schema")
        for name, value in values.items():
            _field(f"{label} {name}", value, context, e.shape)
    nz, ny, nx = e.shape
    bw = int(boundary_width)
    if bw < 0 or (bw and (2 * bw >= ny or 2 * bw >= nx)):
        raise ValueError("SASE boundary width must leave a nonempty interior")
    xp = MetalArrayModule(context)
    ncell = e.size
    nblocks = (ncell + _TPB - 1) // _TPB
    partials = xp.empty((5, nblocks), dtype=np.float64)
    arrays = {"e": e,
              **{f"sc{i}": coarse[name]
                 for i, name in enumerate(names)},
              **{f"rf{i}": refiltered[f"p{i}"] for i in range(6)},
              **{f"l{i}": lift[f"l{i}"] for i in range(6)},
              "partials": partials}
    context.submit_kernel(
        "sase_solve_partial", arrays,
        {"delta_a": float(delta_a), "delta_b": float(delta_b),
         "e_min": float(e_min), "nz": nz, "ny": ny, "nx": nx,
         "bw": bw, "nblocks": nblocks}, exact_policy=False,
        grid_size=nblocks * _TPB, threads_per_group=_TPB)
    return partials


def submit_sase_dynamic_solve(
        context: MetalContext, u: MetalArray, v: MetalArray, w: MetalArray,
        e: MetalArray, *, dx: float, dy: float, dz: float, delta: float,
        boundary_width: int = 0) -> tuple[float, float]:
    """Run the SASE two-width dynamic solve with authority host tail."""
    if not isinstance(u, MetalArray) or u.ndim != 3:
        raise TypeError("SASE winds must be 3-D MetalArrays")
    for name, value in (("u", u), ("v", v), ("w", w), ("e", e)):
        _field(name, value, context, u.shape)
    nz, ny, nx = u.shape
    bw = int(boundary_width)
    if bw < 0 or (bw and (2 * bw >= ny or 2 * bw >= nx)):
        raise ValueError("SASE boundary width must leave a nonempty interior")
    fine_strain = submit_sase_strain(
        context, u, v, w, dx=dx, dy=dy, dz=dz)
    premul = submit_sase_basis_premultiply(
        context, e, fine_strain, delta=delta, e_min=E_MIN)
    xp = MetalArrayModule(context)
    moments = np.zeros(5, dtype=np.float64)
    for width in (2, 4):
        filtered_velocity = {
            f"f{name}": submit_sase_box_filter(context, value, width)
            for name, value in (("u", u), ("v", v), ("w", w))}
        coarse = submit_sase_strain(
            context, filtered_velocity["fu"], filtered_velocity["fv"],
            filtered_velocity["fw"], dx=dx, dy=dy, dz=dz)
        refiltered = {f"p{i}": submit_sase_box_filter(
            context, premul[f"p{i}"], width) for i in range(6)}
        products = submit_sase_velocity_products(context, u, v, w)
        filtered_products = {
            f"f{name}": submit_sase_box_filter(context, value, width)
            for name, value in products.items()}
        lift = submit_sase_lift_combine(
            context, filtered_products, filtered_velocity)
        partials = submit_sase_solve_partial(
            context, e, coarse, refiltered, lift,
            delta_a=width * float(delta), delta_b=delta,
            e_min=E_MIN, boundary_width=bw)
        moments += partials.to_numpy().sum(axis=1, dtype=np.float64)
    aa, ab, bb, ar, br = (float(x) for x in moments)
    return _solve_tail(np.array([[aa, ab], [ab, bb]], dtype=np.float64),
                       np.array([ar, br], dtype=np.float64))
