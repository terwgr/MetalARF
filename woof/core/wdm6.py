"""WDM6 double-moment warm-rain microphysics (WRF ``mp_physics=16``).

WDM6 is WSM6's ice half carried unchanged beside a DOUBLE-MOMENT warm rain:
cloud droplet number ``nc``, rain number ``nr`` and a prognostic CCN
reservoir ``nn`` are transported and predicted, rain's PSD becomes a gamma
with shape ``mu = 1`` and a diagnosed intercept, and autoconversion,
accretion, self-collection, evaporation and freezing all move number as
well as mass.

The numerical authority is the byte-frozen WRF v4.6.1
``phys/module_mp_wdm6.F`` in the 1974 reference bundle; the CUDA kernel
(``woof/core/kernels/wdm6.cu``) is transcribed from it line by line and
carries the file:line citations.  The host coefficient block is
``woof/core/wdm6_constants.py``, which imports WSM6's ``_rgmma`` and
``rimed_ice_constants`` rather than copying them, because ``wdm6init``'s
``hail_opt`` arm (:2096-2108) sets exactly the five constants
``mp_wsm6_init`` sets.

WHAT THE ADAPTER OWES THE SCHEME THAT WSM6'S DOES NOT

* ``xland``.  WDM6 picks its autoconversion threshold per column from the
  land/sea mask -- ``qc0`` (maritime, ``xncr0 = 5e7``) where ``slmsk == 2``
  and ``qc1`` (continental, ``xncr1 = 5e8``) elsewhere
  (module_mp_wdm6.F:607-614; the driver's own comment at
  module_microphysics_driver.F:2495 spells the convention "1: land,
  2: water").  woof keeps XLAND on the physics driver's surface field
  dict, so :func:`apply` reads it from there and REFUSES rather than
  substituting a mask: guessing "all land" would silently move the
  continental threshold onto an ocean domain, which is a factor of ten in
  the autoconversion trigger.
* ``nn`` initialisation.  WRF fills the whole CCN array with the namelist
  ``ccn_conc`` on the first time step (:220-227), and ``ccn0`` reaches
  nothing else in the module -- ``wdm62D`` and ``wdm6init`` both declare it
  ``intent(in)`` and never read it.  woof performs that fill once at state
  allocation from ``cfg.wdm6_ccn_conc`` (``woof/core/state.py``), which is
  the NSSL ``qnn`` precedent and is exactly equivalent for a cold start.

EVIDENCE.  This is an implemented-unverified integration port.  No oracle
comparison against the WRF Fortran has been run: the campaign that produced
Shin-Hong's and Grell-Freitas's ULP numbers is the declared next stage for
this scheme too.
"""

from __future__ import annotations

import numpy as np
try:
    import cupy as cp
except Exception:  # Metal and host inspection do not require CuPy.
    cp = None

from woof.config import RunConfig
from woof.core import constants as c
from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.state import DTYPE, DomainState
from woof.core.wdm6_constants import (WDM6_DEEP_KMAX, WDM6_NUMBER_SPECIES,
                                       WDM6_SHALLOW_KMAX,
                                       WDM6_VERTICAL_LEVEL_BOUNDS,
                                       wdm6_level_tier)

_COLUMN_TPB = 32
# The ladder itself lives in wdm6_constants (a CuPy-free leaf) so the memory
# preflight can price a WDM6 configuration on a host with no device; these
# names are the adapter's local spelling of it.
_SHALLOW_KMAX = WDM6_SHALLOW_KMAX
_KMAX = WDM6_DEEP_KMAX
VERTICAL_LEVEL_BOUNDS = WDM6_VERTICAL_LEVEL_BOUNDS
_kernel_capacity = wdm6_level_tier
_KMAX = WDM6_DEEP_KMAX


def launch_wdm6(theta, qv, qc, qr, qi, qs, qg, nn, nc, nr,
                rho, pii, pressure, dz, xland,
                rainnc, rainncv, snownc, snowncv,
                graupelnc, graupelncv, sr, dt: float, *,
                effc, effi, effs, hail_opt: int = 0, count_status=None) -> None:
    """Launch one WDM6 call over contiguous FP32 ``(nz,ny,nx)`` fields."""
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WDM6 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    capacity = _kernel_capacity(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    with np.errstate(over='ignore', under='ignore'):
        device_dt = DTYPE(dt)
    if not np.isfinite(device_dt) or device_dt <= 0:
        raise ValueError(f"WDM6 dt must remain finite and positive in float32, got {dt}")
    if int(hail_opt) not in (0, 1):
        raise ValueError(f"WDM6 hail_opt must be 0 or 1, got {hail_opt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "qg": qg,
        "nn": nn, "nc": nc, "nr": nr,
        "rho": rho, "pii": pii, "pressure": pressure, "dz": dz,
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
            "xland": xland,
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
    # The Metal ABI carries CUDA's uint32 status bits in an int32 resource
    # descriptor.  This is a zero-copy, same-width view: the shader performs
    # atomic_or on those exact bits, and this caller reads and reports them
    # before returning.  Do not omit this resource; WDM6's column preflight
    # contract requires every bad column to report before it is updated.
    from woof.metal.array import MetalArray
    if isinstance(theta, MetalArray):
        from woof.metal.wdm6_family import submit_wdm6_column
        if count_status is None:
            from woof.metal.array import MetalArrayModule
            count_status = MetalArrayModule(theta.buffer.context).zeros(
                (1,), np.uint32)
        if not isinstance(count_status, MetalArray):
            raise TypeError("Metal WDM6 count status must be a MetalArray")
        if count_status.dtype == np.dtype(np.uint32):
            count_status = count_status.view((1,), np.int32)
        if count_status.shape != (1,):
            raise ValueError("Metal WDM6 count status must be one uint32 word")
        count_status.fill(np.int32(0))
        submit_wdm6_column(
            theta=theta, qv=qv, qc=qc, qi=qi, qr=qr, qs=qs, qg=qg,
            nn=nn, nc=nc, nr=nr, den=rho, p=pressure, pii=pii, dz=dz,
            xland=xland, rainnc=rainnc, rainncv=rainncv,
            snownc=snownc, snowncv=snowncv,
            graupelnc=graupelnc, graupelncv=graupelncv, sr=sr,
            effc=effc, effi=effi, effs=effs, dt=device_dt,
            hail_opt=hail_opt, count_status=count_status)
        flags = int(count_status.to_numpy()[0]) & 0xFFFFFFFF
        if flags:
            describe = _wdm6_status_description
            describe(flags)
        return
    if count_status is None:
        if cp is None:
            raise RuntimeError("CUDA WDM6 dispatch requires CuPy")
        count_status = cp.empty((1,), dtype=cp.uint32)
    if count_status.shape != (1,) or count_status.dtype != np.dtype('uint32'):
        raise ValueError('WDM6 count status must be one uint32 word')
    count_status.fill(cp.uint32(0))
    kernel = (get_kernel("wdm6", "wdm6_column")
              if capacity == _SHALLOW_KMAX else
              get_kernel_int_defines(
                  "wdm6", "wdm6_column", (("WDM6_KMAX", capacity),)))
    kernel(
        (blocks,), (_COLUMN_TPB,),
        (theta, qv, qc, qi, qr, qs, qg, nn, nc, nr,
         rho, pressure, pii, dz, xland,
         rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
         effc, effi, effs, device_dt, np.int32(hail_opt),
         np.int32(nz), np.int32(ny), np.int32(nx), count_status))
    from woof.core import health_ledger
    flags = health_ledger.read_status(
        count_status, site='wdm6-counts', describe=_wdm6_status_description)
    if flags:
        _wdm6_status_description(flags)


def _wdm6_status_description(flags):
    if flags & 1:
        raise FloatingPointError('WDM6 substep geometry requires finite positive density, layer thickness and interval; the affected column was not updated')
    raise FloatingPointError('WDM6 substep count is not representable by its signed 32-bit counter for the supplied density/layer thickness and interval; correct those inputs or use a shorter interval. The affected column was not updated')


def column_land_mask(state: DomainState):
    """WRF XLAND for the microphysics call, or a refusal naming the gap.

    Separated from :func:`apply` so the requirement is testable on its own
    and so the refusal text lives beside the reason.  WDM6 is the first
    woof microphysics scheme whose PROCESS RATES read a surface field, and
    a fabricated mask is not a degraded answer -- it is the wrong
    autoconversion threshold everywhere it is wrong.
    """
    driver = getattr(state, "physics", None)
    fields = getattr(driver, "fields", None)
    xland = None if fields is None else fields.get("xland")
    if xland is None:
        raise ValueError(
            "mp_physics=16 (WDM6) needs WRF's XLAND land/sea mask for its "
            "per-column autoconversion threshold (module_mp_wdm6.F:607-614) "
            "and the physics driver carries none; refusing to substitute an "
            "all-land or all-water mask. Construct the domain through "
            "woof.core.physics.initialize_physics, which builds XLAND from "
            "the landmask for every configuration.")
    return xland


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run WDM6, and finish retained MP heating."""
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "nn", "nc", "nr", "effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=16 state lacks WDM6 fields: "
                         + ", ".join(missing))
    xland = column_land_mask(state)
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # pragma: no cover - CUDA-only installations
        MetalArray = ()
    if isinstance(state.p, MetalArray):
        # Fail closed before the preparation kernel captures h_diabatic or
        # creates family scratch. The driver gates the same configuration at
        # construction; these checks also protect direct microphysics.apply.
        from woof.wrf_exact import ENABLED as WRF_EXACT
        if WRF_EXACT:
            raise NotImplementedError(
                "Metal WDM6 does not implement WRF exact-D arithmetic")
        if refl_10cm_due:
            raise NotImplementedError(
                "Metal WDM6 reflectivity diagnostics are not implemented")
        _validate_metal_call(state, cfg, dt, xland=xland)
        return _apply_metal(state, cfg, dt, xland=xland)
    if cp is None:
        raise RuntimeError("CUDA WDM6 execution requires CuPy; use a Metal state")
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "wdm6_theta")
    rho = state.scratch((nz, ny, nx), "wdm6_rho")
    pii = state.scratch((nz, ny, nx), "wdm6_pii")
    dz = state.scratch((nz, ny, nx), "wdm6_dz")
    z8w = state.scratch((nz + 1, ny, nx), "wdm6_z8w")
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
    launch_wdm6(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        state.nn, state.nc, state.nr,
        rho, pii, state.p, dz, cp.ascontiguousarray(xland),
        rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, sr, dt, effc=state.effc,
        effi=state.effi, effs=state.effs, hail_opt=cfg.wdm6_hail_opt,
        count_status=state.scratch((1,), 'wdm6_count_status', dtype=cp.uint32))
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


def _apply_metal(state: DomainState, cfg: RunConfig, dt: float, *, xland):
    """Run WDM6 preparation, column physics, and heating on Metal storage."""
    from woof.core.microphysics import MicrophysicsDiagnostics
    from woof.metal.wsm6_family import (submit_wsm6_finish,
                                         submit_wsm6_prepare)

    nz, ny, nx = map(int, state.p.shape)
    shape, surface = (nz, ny, nx), (ny, nx)
    theta = state.scratch(shape, "wdm6_theta")
    rho = state.scratch(shape, "wdm6_rho")
    pii = state.scratch(shape, "wdm6_pii")
    dz = state.scratch(shape, "wdm6_dz")
    z8w = state.scratch((nz + 1, ny, nx), "wdm6_z8w")
    submit_wsm6_prepare(state, theta=theta, den=rho, pii=pii,
                        dz=dz, z8w=z8w)

    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")
    status = state.scratch((1,), "wdm6_count_status", dtype=np.uint32)
    launch_wdm6(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        state.nn, state.nc, state.nr, rho, pii, state.p, dz, xland,
        rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, dt,
        effc=state.effc, effi=state.effi, effs=state.effs,
        hail_opt=cfg.wdm6_hail_opt, count_status=status)
    submit_wsm6_finish(state, theta=theta, dt=dt,
                       mp_tend_lim=cfg.mp_tend_lim,
                       no_mp_heating=bool(cfg.no_mp_heating))
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


def _validate_metal_call(state: DomainState, cfg: RunConfig, dt: float, *,
                         xland) -> None:
    """Validate the supported Metal state before preparation mutates it."""
    from woof.metal.array import MetalArray

    nz, ny, nx = map(int, state.p.shape)
    if nz < 2 or nz > _KMAX:
        raise ValueError(f"Metal WDM6 requires 2 <= nz <= {_KMAX}, got {nz}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"Metal WDM6 dt must be finite and positive, got {dt!r}")
    hail_opt = getattr(cfg, "wdm6_hail_opt", 0)
    if not isinstance(hail_opt, (int, np.integer)) or int(hail_opt) not in (0, 1):
        raise ValueError(f"WDM6 hail_opt must be 0 or 1, got {hail_opt!r}")
    limit = np.float32(np.float32(cfg.mp_tend_lim) * dt32)
    if (not np.isfinite(cfg.mp_tend_lim) or cfg.mp_tend_lim <= 0.0
            or not np.isfinite(limit)):
        raise ValueError("Metal WDM6 heating limit must be finite and positive")
    context = state._array_module.context
    volume_names = (
        "p", "alt", "thp", "qv", "qc", "qr", "qi", "qs", "qg",
        "nn", "nc", "nr", "effc", "effi", "effs", "h_diabatic")
    for name in volume_names:
        value = getattr(state, name, None)
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != DTYPE or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(
                f"Metal WDM6 state.{name} must be contiguous FP32 storage "
                f"with shape {(nz, ny, nx)}")
    for name, accepted in (
            ("thb", ((nz,), (nz, 1, 1), (nz, ny, nx))),
            ("phb", ((nz + 1,), (nz + 1, 1, 1), (nz + 1, ny, nx)))):
        value = getattr(state, name, None)
        if (not isinstance(value, MetalArray) or value.shape not in accepted
                or value.dtype != DTYPE or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(f"Metal WDM6 state.{name} has unsupported storage")
    for name, expected in (("php", (nz + 1, ny, nx)),
                           ("xland", (ny, nx))):
        value = xland if name == "xland" else getattr(state, name, None)
        if (not isinstance(value, MetalArray) or value.shape != expected
                or value.dtype != DTYPE or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(
                f"Metal WDM6 {name} must be contiguous FP32 storage "
                f"with shape {expected}")


__all__ = ["WDM6_NUMBER_SPECIES", "apply", "column_land_mask", "launch_wdm6"]
