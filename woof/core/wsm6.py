"""WSM6 single-moment six-class microphysics (WRF ``mp_physics=6``).

The CUDA kernel owns one complete column and updates qv/qc/qr/qi/qs/qg,
potential temperature, surface rain/snow/graupel, frozen fraction, and the
scheme-native radiation effective radii.  WRF v4.6.1
``phys/physics_mmm/mp_wsm6.F90`` and ``mp_wsm6_effectRad.F90`` are the
primary numerical authorities.

The local process equations and WRF's forward semi-Lagrangian monotone-PLM
sedimentation are ported and checked against direct calls into the v4.6.1
Fortran.  This remains an executable integration port, not a whole-model or
bitwise WRF-parity claim; device and coupled-trajectory evidence are separate
release gates.
"""

from __future__ import annotations

import numpy as np
try:
    import cupy as cp
except ImportError:  # The Metal adapter has no CuPy dependency.
    cp = None

from woof.config import RunConfig
from woof.core import constants as c
from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.state import DTYPE, DomainState

from woof.core.wsm6_constants import (WSM6_DEEP_KMAX, WSM6_SHALLOW_KMAX,
                                       WSM6_VERTICAL_LEVEL_BOUNDS,
                                       wsm6_level_tier)

_COLUMN_TPB = 32
# The ladder itself lives in wsm6_constants (a CuPy-free leaf) so the memory
# preflight can price a WSM6 configuration on a host with no device; these
# names are the adapter's local spelling of it.
_SHALLOW_KMAX = WSM6_SHALLOW_KMAX
_KMAX = WSM6_DEEP_KMAX
VERTICAL_LEVEL_BOUNDS = WSM6_VERTICAL_LEVEL_BOUNDS
_kernel_capacity = wsm6_level_tier


def launch_wsm6(theta, qv, qc, qr, qi, qs, qg, rho, pii, pressure, dz,
                rainnc, rainncv, snownc, snowncv,
                graupelnc, graupelncv, sr, dt: float, *,
                effc, effi, effs, hail_opt: int = 0) -> None:
    """Launch one WSM6 call over contiguous FP32 ``(nz,ny,nx)`` fields."""
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # pragma: no cover - CuPy-only installations
        MetalArray = ()
    if isinstance(theta, MetalArray):
        from woof.metal.wsm6_family import submit_wsm6_column
        return submit_wsm6_column(
            theta=theta, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,
            den=rho, pressure=pressure, pii=pii, dz=dz,
            rainnc=rainnc, rainncv=rainncv, snownc=snownc,
            snowncv=snowncv, graupelnc=graupelnc,
            graupelncv=graupelncv, sr=sr, effc=effc, effi=effi,
            effs=effs, dt=dt, hail_opt=hail_opt)
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WSM6 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    capacity = _kernel_capacity(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if int(hail_opt) not in (0, 1):
        raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "qg": qg, "rho": rho, "pii": pii,
        "pressure": pressure, "dz": dz,
        "effc": effc, "effi": effi, "effs": effs,
    }
    for name, value in volume.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    surface_shape = (ny, nx)
    for name, value in {
            "rainnc": rainnc, "rainncv": rainncv,
            "snownc": snownc, "snowncv": snowncv,
            "graupelnc": graupelnc, "graupelncv": graupelncv,
            "sr": sr}.items():
        if value.shape != surface_shape or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {surface_shape}, got "
                             f"{value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel = (get_kernel("wsm6", "wsm6_column")
              if capacity == _SHALLOW_KMAX else
              get_kernel_int_defines(
                  "wsm6", "wsm6_column", (("WSM6_KMAX", capacity),)))
    kernel(
        (blocks,), (_COLUMN_TPB,),
        (theta, qv, qc, qi, qr, qs, qg, rho, pressure, pii, dz,
         rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
         effc, effi, effs, DTYPE(dt), np.int32(hail_opt),
         np.int32(nz), np.int32(ny), np.int32(nx)))


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run WSM6, and finish retained MP heating."""
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=6 state lacks WSM6 fields: "
                         + ", ".join(missing))
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # pragma: no cover - CuPy-only installations
        MetalArray = ()
    if isinstance(state.p, MetalArray):
        # Fail closed before allocating or mutating prognostic/scratch fields.
        from woof.wrf_exact import ENABLED as WRF_EXACT
        if WRF_EXACT:
            raise NotImplementedError(
                "Metal WSM6 does not implement WRF exact-D arithmetic")
        # Validate all route-limited parameters before preparation writes
        # h_diabatic or allocates the family's scratch fields.
        _validate_metal_call(state, cfg, dt)
        return _apply_metal(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    if cp is None:
        raise RuntimeError("CPU WSM6 execution requires CuPy; use a Metal state")
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "wsm6_theta")
    rho = state.scratch((nz, ny, nx), "wsm6_rho")
    pii = state.scratch((nz, ny, nx), "wsm6_pii")
    dz = state.scratch((nz, ny, nx), "wsm6_dz")
    z8w = state.scratch((nz + 1, ny, nx), "wsm6_z8w")
    theta[...] = thb + state.thp
    rho[...] = 1.0 / state.alt
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")

    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    save_pre_mp_theta(state)
    launch_wsm6(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        rho, pii, state.p, dz, rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, sr, dt, effc=state.effc,
        effi=state.effi, effs=state.effs, hail_opt=cfg.wsm6_hail_opt)
    if refl_10cm_due:
        from woof.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = theta * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


def _apply_metal(state: DomainState, cfg: RunConfig, dt: float, *,
                 refl_10cm_due: bool = False):
    """Device-resident WSM6 preparation, column physics, and heating finish."""
    from woof.core.microphysics import MicrophysicsDiagnostics
    from woof.metal.wsm6_family import (
        submit_wsm6_finish, submit_wsm6_prepare)

    nz, ny, nx = state.p.shape
    shape, surface = (nz, ny, nx), (ny, nx)
    theta = state.scratch(shape, "wsm6_theta")
    rho = state.scratch(shape, "wsm6_rho")
    pii = state.scratch(shape, "wsm6_pii")
    dz = state.scratch(shape, "wsm6_dz")
    z8w = state.scratch((nz + 1, ny, nx), "wsm6_z8w")
    submit_wsm6_prepare(state, theta=theta, den=rho, pii=pii,
                        dz=dz, z8w=z8w)

    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")
    launch_wsm6(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        rho, pii, state.p, dz, rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, sr, dt, effc=state.effc,
        effi=state.effi, effs=state.effs, hail_opt=cfg.wsm6_hail_opt)
    if refl_10cm_due:
        from woof.metal.refl_family import submit_refl_temperature_product
        from woof.metal.wsm6_reflectivity import launch_refl10cm_wsm6
        from woof.core.refl import stash_refl_10cm
        temperature = state.scratch(shape, "refl_t")
        refl = state.scratch(shape, "refl_10cm")
        submit_refl_temperature_product(theta, pii, temperature)
        launch_refl10cm_wsm6(
            state.qv, state.qr, state.qs, state.qg, temperature, state.p,
            refl, hail_opt=cfg.wsm6_hail_opt)
        stash_refl_10cm(state, refl)
    submit_wsm6_finish(state, theta=theta, dt=dt,
                       mp_tend_lim=cfg.mp_tend_lim,
                       no_mp_heating=bool(cfg.no_mp_heating))
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


def _validate_metal_call(state: DomainState, cfg: RunConfig, dt: float) -> None:
    """Fail before preparation can mutate state if WSM6 inputs are unsupported."""
    from woof.metal.array import MetalArray

    nz = int(state.p.shape[0])
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"Metal WSM6 requires 2 <= nz <= {_KMAX}, got {nz}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"Metal WSM6 dt must be finite and positive, got {dt!r}")
    hail_opt = getattr(cfg, "wsm6_hail_opt", 0)
    if not isinstance(hail_opt, (int, np.integer)) or int(hail_opt) not in (0, 1):
        raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt!r}")
    limit = np.float32(np.float32(cfg.mp_tend_lim) * dt32)
    if (not np.isfinite(cfg.mp_tend_lim) or cfg.mp_tend_lim <= 0.0
            or not np.isfinite(limit)):
        raise ValueError("Metal WSM6 heating limit must be finite and positive")
    volume = ("p", "alt", "thp", "qv", "qc", "qr", "qi", "qs", "qg",
              "effc", "effi", "effs", "h_diabatic")
    surface_shape = tuple(state.mup.shape)
    context = state._array_module.context
    for name in volume:
        value = getattr(state, name, None)
        if (not isinstance(value, MetalArray)
                or value.shape != tuple(state.p.shape)
                or value.dtype != DTYPE or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(
                f"Metal WSM6 state.{name} must be contiguous FP32 storage "
                f"with shape {tuple(state.p.shape)}")
    for name, expected in (("thb", ((nz,), (nz, 1, 1), tuple(state.p.shape))),
                           ("phb", ((nz + 1,), (nz + 1, 1, 1),
                                    (nz + 1, *surface_shape)))):
        value = getattr(state, name, None)
        if (not isinstance(value, MetalArray) or value.shape not in expected
                or value.dtype != DTYPE or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(f"Metal WSM6 state.{name} has unsupported storage")
    for name in ("php",):
        value = getattr(state, name, None)
        if (not isinstance(value, MetalArray)
                or value.shape != (nz + 1, *surface_shape)
                or value.dtype != DTYPE or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(f"Metal WSM6 state.{name} has unsupported storage")


__all__ = ["apply", "launch_wsm6"]
