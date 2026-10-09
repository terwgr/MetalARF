"""Morrison two-moment microphysics (WRF ``mp_physics=10``).

The CUDA pipeline stages independent level work around one sedimentation
warp per category across atmospheric columns, carrying WRF's internal substeps.  The
public launcher accepts contiguous FP32 ``(nz, ny, nx)`` fields and updates
potential temperature, six water categories, and their moments in place.
:func:`apply` prepares WRF's microphysics inputs from
:class:`~woof.core.state.DomainState`.

Transcription authority: local WRF v4.6.1
``phys/module_mp_morr_two_moment.F``: wrapper/API lines 563-925, column
scheme lines 929-4062, saturation helper lines 4066-4149.  Float64 mirror:
``tests.physics_reference.npref.np_morrison_column``.
"""

from __future__ import annotations

import numpy as np

from woof.config import RunConfig
from woof.core import constants as c
from woof.core.kernels import get_kernel
from woof.core.morrison_constants import rimed_ice_constants
from woof.core.state import DTYPE, DomainState

_CELL_TPB = 64
_COLUMN_TPB = 32
#: Sedimentation block: one warp per hydrometeor category over the same
#: _COLUMN_TPB columns (kernels/morrison.cu, morrison_sediment_impl).
_SEDIMENT_TPB = 5 * _COLUMN_TPB
_SHALLOW_KMAX = 64
_KMAX = 256
VERTICAL_LEVEL_BOUNDS = (2, _KMAX)

def _metal_empty_like(array):
    from woof.metal.array import MetalArray
    return MetalArray(array.buffer.context.buffer(max(array.nbytes, 4)),
                      array.shape, array.dtype)



_prepare_fields = None


def _get_prepare_fields_kernel(cp):
    """Build the CUDA-only preparation kernel on the first CUDA call."""
    global _prepare_fields
    if _prepare_fields is None:
        _prepare_fields = cp.ElementwiseKernel(
            "raw float32 thb, raw float32 phb, raw float32 thp, raw float32 php, "
            "raw float32 pressure, int32 ncol, int32 thb_full, int32 phb_full, "
            "float32 p0, float32 rcp, float32 gravity",
            "float32 theta, float32 pii, float32 dz",
            r"""
            size_t k = i / ncol;
            size_t below = i;
            size_t above = i + ncol;
            theta = __fadd_rn(thb[thb_full ? i : k], thp[i]);
            pii = powf(__fdiv_rn(pressure[i], p0), rcp);
            float zb = __fdiv_rn(__fadd_rn(phb[phb_full ? below : k], php[below]), gravity);
            float za = __fdiv_rn(__fadd_rn(phb[phb_full ? above : k + 1], php[above]), gravity);
            dz = __fsub_rn(za, zb);
            """,
            "morrison_prepare_fields", options=("--ftz=true",))
    return _prepare_fields

def launch_morrison(theta, qv, qc, qr, qi, qs, qg,
                    nc, nr, ni, ns, ng, rho, pii, pressure, dz,
                    rainnc, rainncv, snownc, snowncv,
                    graupelnc, graupelncv, sr, dt: float,
                    *, effc=None, effr=None, effi=None, effs=None,
                    qrcuten=None, qscuten=None, qicuten=None,
                    morr_rimed_ice: int = 1,
                    _rhoa_scratch=None, _ice_to_snow_scratch=None,
                    _interior_bounds=None) -> None:
    """Launch one WRF Morrison call over an FP32 column batch.

    Atmospheric and hydrometeor arrays are ``(nz, ny, nx)``.  Accumulated,
    per-call, and frozen-fraction precipitation fields are ``(ny, nx)``.
    All prognostic arguments through ``ng`` and all precipitation fields
    are updated in place.  ``rho`` is read-only and retained in the
    WRF-compatible API; WRF's Morrison implementation documents it as
    unused and diagnoses its own density from pressure and temperature
    (source line 589/1325).  The private scratch hook lets the state adapter
    reuse its persistent named density buffer without changing this public
    contract.  ``morr_rimed_ice`` follows WRF's scalar Registry option:
    1 = hail (default), 0 = graupel (Registry.EM_COMMON:2663-2666;
    module_mp_morr_two_moment.F:337-411).
    ``qrcuten``/``qscuten``/``qicuten`` are the optional all-or-none raw KF
    mass rates used by WRF to seed rain/snow/ice number moments before process
    calculations (module_mp_morr_two_moment.F:1327-1343).
    """
    from woof.metal.array import MetalArray
    if isinstance(theta, MetalArray):
        from woof.metal.morrison_family import (
            submit_morrison_process_levels, submit_morrison_sedimentation,
            submit_morrison_finalize, validate_morrison_pipeline)
        from woof.wrf_exact import ENABLED as WRF_EXACT
        if WRF_EXACT:
            raise NotImplementedError(
                "Metal Morrison does not implement WRF exact-D arithmetic")
        shape = theta.shape
        if len(shape) != 3:
            raise ValueError(f"Morrison fields must be 3-D, got {shape}")
        nz, ny, nx = shape
        if nz < 2 or nz > _KMAX:
            raise ValueError(f"Morrison requires 2 <= nz <= {_KMAX}, got {nz}")
        cu_values = (qrcuten, qscuten, qicuten)
        has_cu_tendencies = any(v is not None for v in cu_values)
        if has_cu_tendencies and not all(v is not None for v in cu_values):
            raise ValueError("qrcuten, qscuten, and qicuten must be supplied together")
        for name, value in zip(("qrcuten", "qscuten", "qicuten"), cu_values):
            if value is not None and (value.shape != shape
                    or value.dtype != np.dtype(np.float32)
                    or not value.flags.c_contiguous):
                raise ValueError(f"{name} must be contiguous FP32 with shape {shape}")
        rhoa_scratch = (_metal_empty_like(theta) if _rhoa_scratch is None
                        else _rhoa_scratch)
        ice_to_snow_scratch = (_metal_empty_like(theta)
                               if _ice_to_snow_scratch is None
                               else _ice_to_snow_scratch)
        effective = {"effc":effc,"effr":effr,"effi":effi,"effs":effs}
        for name,value in effective.items():
            if value is None: effective[name]=_metal_empty_like(theta)
        qrcu,qscu,qicu=cu_values if has_cu_tendencies else (rho,rho,rho)
        arrays={"theta":theta,"qv":qv,"qc":qc,"qr":qr,"qi":qi,"qs":qs,
            "qg":qg,"nc":nc,"nr":nr,"ni":ni,"ns":ns,"ng":ng,
            "qrcuten":qrcu,"qscuten":qscu,"qicuten":qicu,
            "rho_in":rhoa_scratch,"pii":pii,"pressure":pressure,
            "ice_to_snow_scratch":ice_to_snow_scratch,
            "effc":effective["effc"],"effi":effective["effi"],"effs":effective["effs"]}
        rimed=rimed_ice_constants(morr_rimed_ice)
        sediment_arrays={"qc":qc,"qr":qr,"qi":qi,"qs":qs,"qg":qg,
            "nc":nc,"nr":nr,"ni":ni,"ns":ns,"ng":ng,
            "theta":theta,"pii":pii,"pressure":pressure,
            "rho_in":rhoa_scratch,"dz":dz,"cloud_nc":effective["effs"]}
        sediment_arrays.update({"rainnc":rainnc,"rainncv":rainncv,
            "snownc":snownc,"snowncv":snowncv,"graupelnc":graupelnc,
            "graupelncv":graupelncv,"sr":sr})
        finalize_arrays={"theta":theta,"qv":qv,"qc":qc,"qr":qr,"qi":qi,
            "qs":qs,"qg":qg,"nc":nc,"nr":nr,"ni":ni,"ns":ns,"ng":ng,
            "rho_in":rhoa_scratch,"pii":pii,"pressure":pressure,
            "ice_to_snow_scratch":ice_to_snow_scratch,
            "effc":effective["effc"],"effi":effective["effi"],
            "effs":effective["effs"],"effr":effective["effr"]}
        validate_morrison_pipeline(arrays,sediment_arrays,finalize_arrays)
        submit_morrison_process_levels(None,arrays,dt=dt,morr_ag=rimed.ag,
            morr_bg=rimed.bg,morr_rhog=rimed.rhog,
            has_cu_tendencies=has_cu_tendencies,
            interior_bounds=_interior_bounds)
        submit_morrison_sedimentation(None,sediment_arrays,dt=dt,
            morr_ag=rimed.ag,morr_bg=rimed.bg,morr_rhog=rimed.rhog,
            interior_bounds=_interior_bounds)
        submit_morrison_finalize(None,finalize_arrays,morr_rhog=rimed.rhog,
            interior_bounds=_interior_bounds)
        return

    # Keep the module usable by the Metal backend, which has its own kernels
    # and must not import CuPy.  A non-Metal call remains the CUDA path and
    # still requires CuPy explicitly.
    import cupy as cp

    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"Morrison fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    minimum, maximum = VERTICAL_LEVEL_BOUNDS
    if nz > maximum:
        raise ValueError(f"nz={nz} exceeds MORR_KMAX={maximum}")
    if nz < minimum:
        raise ValueError(f"Morrison requires nz >= {minimum}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    rimed = rimed_ice_constants(morr_rimed_ice)
    for name, value in (
            ("qv", qv), ("qc", qc), ("qr", qr), ("qi", qi),
            ("qs", qs), ("qg", qg), ("nc", nc), ("nr", nr),
            ("ni", ni), ("ns", ns), ("ng", ng), ("rho", rho),
            ("pii", pii), ("pressure", pressure), ("dz", dz)):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    cu_values = (qrcuten, qscuten, qicuten)
    has_cu_tendencies = any(value is not None for value in cu_values)
    if has_cu_tendencies and not all(value is not None for value in cu_values):
        raise ValueError("qrcuten, qscuten, and qicuten must be supplied "
                         "together")
    if has_cu_tendencies:
        for name, value in zip(("qrcuten", "qscuten", "qicuten"), cu_values):
            if (value.shape != shape or value.dtype != DTYPE
                    or not value.flags.c_contiguous):
                raise ValueError(f"{name} must be contiguous float32 with "
                                 f"shape {shape}")
        qrcu, qscu, qicu = cu_values
    else:
        # The kernel gates every read with has_cu_tendencies.  Reusing a valid
        # pointer avoids allocating a full-volume zero placeholder.
        qrcu = qscu = qicu = rho
    surface_shape = (ny, nx)
    for name, value in (
            ("rainnc", rainnc), ("rainncv", rainncv),
            ("snownc", snownc), ("snowncv", snowncv),
            ("graupelnc", graupelnc), ("graupelncv", graupelncv),
            ("sr", sr)):
        if value.shape != surface_shape or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {surface_shape}, "
                             f"got {value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    if _rhoa_scratch is None:
        rhoa_scratch = cp.empty_like(rho)
    else:
        rhoa_scratch = _rhoa_scratch
        if (rhoa_scratch.shape != shape or rhoa_scratch.dtype != DTYPE
                or not rhoa_scratch.flags.c_contiguous):
            raise ValueError("_rhoa_scratch must be contiguous float32 with "
                             f"shape {shape}")
    if _ice_to_snow_scratch is None:
        ice_to_snow_scratch = cp.empty_like(rho)
    else:
        ice_to_snow_scratch = _ice_to_snow_scratch
        if (ice_to_snow_scratch.shape != shape
                or ice_to_snow_scratch.dtype != DTYPE
                or not ice_to_snow_scratch.flags.c_contiguous):
            raise ValueError("_ice_to_snow_scratch must be contiguous float32 "
                             f"with shape {shape}")

    effective = {"effc": effc, "effr": effr, "effi": effi, "effs": effs}
    for name, value in effective.items():
        if value is None:
            effective[name] = cp.empty_like(theta)
        elif (value.shape != shape or value.dtype != DTYPE
              or not value.flags.c_contiguous):
            raise ValueError(f"{name} must be contiguous float32 with "
                             f"shape {shape}")

    ncell = nz * ny * nx
    ncol = ny * nx
    cell_blocks = (ncell + _CELL_TPB - 1) // _CELL_TPB
    column_blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    process = get_kernel("morrison", "morrison_process_levels")
    sediment_name = ("morrison_sediment_64" if nz <= _SHALLOW_KMAX
                     else "morrison_sediment_256")
    sediment = get_kernel("morrison", sediment_name)
    finalize = get_kernel("morrison", "morrison_finalize_levels")

    process((cell_blocks,), (_CELL_TPB,),
            (theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
             qrcu, qscu, qicu, rhoa_scratch, pii, pressure,
             ice_to_snow_scratch,
             effective["effc"], effective["effi"], effective["effs"],
             DTYPE(rimed.ag), DTYPE(rimed.bg), DTYPE(rimed.rhog),
             DTYPE(dt), np.int32(has_cu_tendencies), np.int32(ncell)))
    sediment((column_blocks,), (_SEDIMENT_TPB,),
             (qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
              effective["effs"], theta, pii, pressure, rhoa_scratch, dz,
              rainnc, rainncv, snownc, snowncv,
              graupelnc, graupelncv, sr, DTYPE(dt),
              DTYPE(rimed.ag), DTYPE(rimed.bg), DTYPE(rimed.rhog),
              np.int32(nz), np.int32(ny), np.int32(nx)))
    finalize((cell_blocks,), (_CELL_TPB,),
             (theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
              rhoa_scratch, pii, pressure, ice_to_snow_scratch,
              effective["effc"], effective["effi"], effective["effs"],
              effective["effr"], DTYPE(rimed.rhog), np.int32(ncell)))


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields and apply Morrison to ``state`` in place.

    On a history step, reflectivity is evaluated after the scheme call from
    its post-call temperature and moments plus the unchanged prepared
    pressure, matching module_mp_morr_two_moment.F:913-914.
    """
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=10 state lacks Morrison fields: "
                         + ", ".join(missing))

    if getattr(state, "_metal_setup_state", False):
        from woof.metal.morrison_family import (
            submit_morrison_prepare_fields, submit_morrison_finish)
        from woof.wrf_exact import ENABLED as WRF_EXACT
        if WRF_EXACT:
            raise NotImplementedError(
                "Metal Morrison does not implement WRF exact-D arithmetic")
        from woof.core.microphysics import _ring_guard_slices
        ring_slices = _ring_guard_slices(state, cfg)
        nz, ny, nx = map(int, state.p.shape)
        interior_bounds = None
        if ring_slices is not None:
            sz = int(cfg.spec_zone)
            j_start, j_stop = min(sz, ny), max(ny - sz, min(sz, ny))
            i_start, i_stop = min(sz, nx), max(nx - sz, min(sz, nx))
            interior_bounds = (j_start, j_stop, i_start, i_stop)
        theta=state.scratch((nz,ny,nx),"morr_theta")
        rho=state.scratch((nz,ny,nx),"morr_rho")
        pii=state.scratch((nz,ny,nx),"morr_pii")
        dz=state.scratch((nz,ny,nx),"morr_dz")
        ice_to_snow=state.scratch((nz,ny,nx),"morr_ice_to_snow")
        submit_morrison_prepare_fields(state,theta=theta,pii=pii,dz=dz,
                                       interior_bounds=interior_bounds)
        surface=(ny,nx)
        rainnc=state.scratch(surface,"mp_rainnc")
        rainncv=state.scratch(surface,"mp_rainncv")
        snownc=state.scratch(surface,"mp_snownc")
        snowncv=state.scratch(surface,"mp_snowncv")
        graupelnc=state.scratch(surface,"mp_graupelnc")
        graupelncv=state.scratch(surface,"mp_graupelncv")
        sr=state.scratch(surface,"mp_sr")
        cu_rates=(getattr(getattr(state,"physics",None),"cu_rates",None)
                  if getattr(cfg,"cu_physics",0) else None)
        launch_morrison(theta,state.qv,state.qc,state.qr,state.qi,state.qs,state.qg,
            state.nc,state.nr,state.ni,state.ns,state.ng,rho,pii,state.p,dz,
            rainnc,rainncv,snownc,snowncv,graupelnc,graupelncv,sr,dt,
            effc=state.effc,effr=state.effr,effi=state.effi,effs=state.effs,
            qrcuten=None if cu_rates is None else cu_rates["rqrcuten"],
            qscuten=None if cu_rates is None else cu_rates["rqscuten"],
            qicuten=None if cu_rates is None else cu_rates["rqicuten"],
            morr_rimed_ice=cfg.morr_rimed_ice,_rhoa_scratch=rho,
            _ice_to_snow_scratch=ice_to_snow,
            _interior_bounds=interior_bounds)
        if refl_10cm_due:
            from woof.metal.refl_family import submit_refl10cm_morrison
            from woof.core.refl import stash_refl_10cm
            refl = state.scratch((nz, ny, nx), "refl_10cm")
            radar_tables = state.scratch((275,), "morr_refl_tables", np.uint64)
            submit_refl10cm_morrison(
                state.qv, state.qr, state.nr, state.qs, state.ns,
                state.qg, state.ng, theta, pii, state.p, refl,
                morr_rimed_ice=cfg.morr_rimed_ice, tables=radar_tables)
            stash_refl_10cm(state, refl)
        submit_morrison_finish(state,theta,dt=dt,mp_tend_lim=cfg.mp_tend_lim,
                              no_mp_heating=bool(cfg.no_mp_heating),
                              interior_bounds=interior_bounds)
        from woof.core.microphysics import MicrophysicsDiagnostics
        return MicrophysicsDiagnostics(
            rainnc=rainnc, rainncv=rainncv, sr=sr,
            snownc=snownc, snowncv=snowncv,
            graupelnc=graupelnc, graupelncv=graupelncv)

    # The non-Metal adapter uses CUDA ufuncs and the CUDA Morrison kernel.
    import cupy as cp

    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "morr_theta")
    rho = state.scratch((nz, ny, nx), "morr_rho")
    pii = state.scratch((nz, ny, nx), "morr_pii")
    dz = state.scratch((nz, ny, nx), "morr_dz")
    ice_to_snow = state.scratch((nz, ny, nx), "morr_ice_to_snow")
    # Match the separate CuPy ufunc roundings, including their FTZ mode.
    _get_prepare_fields_kernel(cp)(
        thb, phb, state.thp, state.php, state.p,
        np.int32(ny * nx), np.int32(state.thb.ndim == 3),
        np.int32(state.phb.ndim == 3),
        DTYPE(c.P0), DTYPE(c.RCP), DTYPE(c.G), theta, pii, dz)

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")
    # Lazy import (matches microphysics.apply's lazy Morrison dispatch):
    # the WRF moist_physics_prep_em/finish_em bracket around the scheme.
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    save_pre_mp_theta(state)
    cu_rates = (getattr(getattr(state, "physics", None), "cu_rates", None)
                if getattr(cfg, "cu_physics", 0) else None)
    launch_morrison(theta, state.qv, state.qc, state.qr,
                    state.qi, state.qs, state.qg,
                    state.nc, state.nr, state.ni, state.ns, state.ng,
                    rho, pii, state.p, dz,
                    rainnc, rainncv, snownc, snowncv,
                    graupelnc, graupelncv, sr, dt,
                    effc=state.effc, effr=state.effr,
                    effi=state.effi, effs=state.effs,
                    qrcuten=(None if cu_rates is None else
                              cu_rates["rqrcuten"]),
                    qscuten=(None if cu_rates is None else
                              cu_rates["rqscuten"]),
                    qicuten=(None if cu_rates is None else
                              cu_rates["rqicuten"]),
                    morr_rimed_ice=cfg.morr_rimed_ice,
                    _rhoa_scratch=rho,
                    _ice_to_snow_scratch=ice_to_snow)
    if refl_10cm_due:
        from woof.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        # WRF wrapper T1D is the scheme-updated absolute temperature while
        # P1D remains the pre-call prepared pressure (F:730, :780, :913-914).
        refl_t[...] = theta * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)
