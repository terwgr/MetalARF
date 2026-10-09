"""Typed launchers for the WRF Morrison process/sediment/finalization path.

The source CUDA unit calls its own ``tgammaf`` builtin. The shader keeps the
Metal call in FP32 and makes the conversion visible, but CUDA-versus-MSL word
and ULP agreement for gamma is unverified; this remains a parity limitation.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from ..wrf_exact import ENABLED as WRF_EXACT


_VOLUME = ("qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng",
           "theta", "pii", "pressure", "rho_in", "dz")
_PRECIP = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
           "graupelncv", "sr")
_FINAL = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr",
          "ni", "ns", "ng", "rho_in", "pii", "pressure",
          "ice_to_snow_scratch", "effc", "effi", "effs", "effr")
_PROCESS = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr",
            "ni", "ns", "ng", "qrcuten", "qscuten", "qicuten",
            "rho_in", "pii", "pressure", "ice_to_snow_scratch",
            "effc", "effi", "effs")

# Status bits returned by submit_morrison_validate_diagnostics.
MORR_DIAG_RAINNC_NONFINITE = 1 << 0
MORR_DIAG_RAINNCV_NONFINITE = 1 << 1
MORR_DIAG_SNOWNC_NONFINITE = 1 << 2
MORR_DIAG_SNOWNCV_NONFINITE = 1 << 3
MORR_DIAG_GRAUPELNC_NONFINITE = 1 << 4
MORR_DIAG_GRAUPELNCV_NONFINITE = 1 << 5
MORR_DIAG_SR_NONFINITE = 1 << 6
MORR_DIAG_RAINNCV_NEGATIVE = 1 << 7
MORR_DIAG_SR_NEGATIVE = 1 << 8
MORR_DIAG_SR_ABOVE_ONE = 1 << 9

def _check_exact():
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal Morrison supports only ordinary non-exact arithmetic")


def _check_state(state):
    _check_exact()
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal Morrison requires a Metal DomainState")


def _check_array(name, array, shape):
    if (not isinstance(array, MetalArray)
            or array.dtype != np.dtype(np.float32)
            or array.shape != tuple(shape)):
        raise TypeError(
            f"{name} must be an FP32 MetalArray with shape {tuple(shape)}")


def _check_mapping(state, arrays, expected, shape):
    if not isinstance(arrays, dict) or set(arrays) != set(expected):
        raise ValueError("Morrison resources do not match the kernel ABI")
    context = state._array_module.context
    for name in expected:
        _check_array(name, arrays[name], shape)
        if arrays[name].buffer.context is not context:
            raise ValueError("Morrison resources must share one Metal context")


def _interior_parameters(interior_bounds, ny, nx):
    """Return the half-open WRF mass-grid interior for a scheme call."""
    if interior_bounds is None:
        j0, j1, i0, i1 = 0, int(ny), 0, int(nx)
    else:
        if (not isinstance(interior_bounds, tuple)
                or len(interior_bounds) != 4
                or any(not isinstance(v, (int, np.integer))
                       for v in interior_bounds)):
            raise TypeError("Morrison interior_bounds must be (j0, j1, i0, i1)")
        j0, j1, i0, i1 = map(int, interior_bounds)
        if not (0 <= j0 <= j1 <= ny and 0 <= i0 <= i1 <= nx):
            raise ValueError("Morrison interior bounds are outside the mass grid")
    return {"j_start": j0, "j_stop": j1,
            "i_start": i0, "i_stop": i1}


def validate_morrison_pipeline(process_arrays, sediment_arrays, finalize_arrays):
    """Validate every pipeline resource before the first state mutation."""
    theta = process_arrays.get("theta") if isinstance(process_arrays, dict) else None
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("Morrison theta must be a three-dimensional MetalArray")
    nz, ny, nx = theta.shape
    if nz < 2 or nz > 256:
        raise ValueError("Morrison requires 2 <= nz <= 256")
    shape, surface = (nz, ny, nx), (ny, nx)
    for arrays, expected in ((process_arrays, _PROCESS),
                             (finalize_arrays, _FINAL)):
        if not isinstance(arrays, dict) or set(arrays) != set(expected):
            raise ValueError("Morrison resources do not match the kernel ABI")
        for name in expected:
            _check_array(name, arrays[name], shape)
            if arrays[name].buffer.context is not theta.buffer.context:
                raise ValueError("Morrison resources must share one Metal context")
    expected_sed = set(_VOLUME + ("cloud_nc",) + _PRECIP)
    if not isinstance(sediment_arrays, dict) or set(sediment_arrays) != expected_sed:
        raise ValueError("Morrison sedimentation resources do not match its ABI")
    for name in _VOLUME + ("cloud_nc",):
        _check_array(name, sediment_arrays[name], shape)
    for name in _PRECIP:
        _check_array(name, sediment_arrays[name], surface)
    if any(sediment_arrays[name].buffer.context is not theta.buffer.context
           for name in expected_sed):
        raise ValueError("Morrison resources must share one Metal context")


def submit_morrison_process_levels(state, arrays, *, dt, morr_ag, morr_bg,
                                  morr_rhog, has_cu_tendencies=False,
                                  interior_bounds=None):
    """Submit one source-ordered FP32 Morrison process update per level."""
    _check_exact()
    if state is not None:
        _check_state(state)
    if not isinstance(arrays.get("theta"), MetalArray) or arrays["theta"].ndim != 3:
        raise TypeError("Morrison theta must be a three-dimensional MetalArray")
    shape = arrays["theta"].shape
    if state is not None:
        _check_mapping(state, arrays, _PROCESS, shape)
    else:
        for name in _PROCESS:
            _check_array(name, arrays[name], shape)
            if arrays[name].buffer.context is not arrays["theta"].buffer.context:
                raise ValueError("Morrison resources must share one Metal context")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0:
        raise ValueError("Morrison dt must be finite and positive")
    context = state._array_module.context if state is not None else arrays["theta"].buffer.context
    bounds = _interior_parameters(interior_bounds, shape[1], shape[2])
    return context.submit_kernel(
        "morrison_process_levels", arrays,
        {"morr_ag": float(np.float32(morr_ag)),
         "morr_bg": float(np.float32(morr_bg)),
         "morr_rhog": float(np.float32(morr_rhog)),
         "dt": float(dt32), "has_cu_tendencies": int(bool(has_cu_tendencies)),
         "nz": shape[0], "ncell": int(arrays["theta"].size),
         "ny": shape[1], "nx": shape[2],
         **bounds},
        exact_policy=False, grid_size=int(arrays["theta"].size))


def submit_morrison_prepare_fields(state, *, theta, pii, dz,
                                   interior_bounds=None):
    """Prepare this call's temperature/Exner/layer depth and saved theta."""
    _check_exact()
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Morrison preparation requires a Metal DomainState")
    if state.h_diabatic is None:
        raise ValueError("Metal Morrison requires allocated h_diabatic")
    nz, ny, nx = map(int, state.p.shape)
    arrays = {"thb": state.thb, "thp": state.thp, "phb": state.phb,
              "php": state.php, "pressure": state.p, "theta": theta,
              "pii": pii, "dz": dz, "h_diabatic": state.h_diabatic}
    ctx = state._array_module.context
    for name, value in arrays.items():
        if not isinstance(value, MetalArray) or value.buffer.context is not ctx:
            raise TypeError(f"Morrison preparation {name} must be a MetalArray on the state context")
    from woof.core import constants as c
    bounds = _interior_parameters(interior_bounds, ny, nx)
    return ctx.submit_kernel(
        "morrison_prepare_fields", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "nzi": nz + 1,
         "thb_full": int(state.thb.ndim == 3),
         "phb_full": int(state.phb.ndim == 3),
         "p0": float(np.float32(c.P0)), "rcp": float(np.float32(c.RCP)),
         "gravity": float(np.float32(c.G)), "ncell": nz * ny * nx,
         **bounds},
        exact_policy=False, grid_size=nz * ny * nx)


def submit_morrison_finish(state, theta, *, dt, mp_tend_lim, no_mp_heating,
                           interior_bounds=None):
    """Apply WRF's retained/clamped moist-physics heating after Morrison."""
    _check_exact()
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Morrison finish requires a Metal DomainState")
    if state.h_diabatic is None:
        raise ValueError("Metal Morrison requires allocated h_diabatic")
    nz, ny, nx = map(int, state.p.shape)
    arrays = {"theta": theta, "thp": state.thp,
              "h_diabatic": state.h_diabatic}
    _check_mapping(state, arrays, tuple(arrays), (nz, ny, nx))
    dt32 = np.float32(dt)
    limit = np.float32(np.float32(mp_tend_lim) * dt32)
    if not np.isfinite(dt32) or dt32 <= 0 or not np.isfinite(limit):
        raise ValueError("Morrison finish requires finite positive dt and limit")
    bounds = _interior_parameters(interior_bounds, ny, nx)
    return state._array_module.context.submit_kernel(
        "morrison_finish", arrays,
        {"dt": float(dt32), "limit": float(limit), "nz": nz,
         "no_mp_heating": int(bool(no_mp_heating)), "ncell": nz * ny * nx,
         "ny": ny, "nx": nx, **bounds},
        exact_policy=False, grid_size=nz * ny * nx)




def submit_morrison_sedimentation(state, arrays, *, dt, morr_ag,
                                 morr_bg, morr_rhog,
                                 interior_bounds=None):
    """Submit one source-sized Morrison sedimentation pass.

    The caller supplies all resident volume/surface arrays. ``nz`` selects
    CUDA's shallow (<=64) or generic (65..256) source specialization.
    """
    _check_exact()
    if state is not None:
        _check_state(state)
    if not isinstance(arrays, dict):
        raise TypeError("Morrison sedimentation arrays must be a mapping")
    if set(arrays) != set(_VOLUME + ("cloud_nc",) + _PRECIP):
        raise ValueError("Morrison sedimentation resources do not match its ABI")
    theta = arrays["theta"]
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("theta must be a three-dimensional MetalArray")
    nz, ny, nx = theta.shape
    if nz < 2 or nz > 256:
        raise ValueError("Morrison sedimentation requires 2 <= nz <= 256")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Morrison dt must be finite and positive")
    for name in _VOLUME + ("cloud_nc",):
        _check_array(name, arrays[name], (nz, ny, nx))
    for name in _PRECIP:
        _check_array(name, arrays[name], (ny, nx))
    if any(arrays[name].buffer.context is not theta.buffer.context
           for name in _VOLUME + ("cloud_nc",) + _PRECIP):
        raise ValueError("Morrison resources must share one Metal context")
    parameters = {
        "dt": float(np.float32(dt)),
        "morr_ag": float(np.float32(morr_ag)),
        "morr_bg": float(np.float32(morr_bg)),
        "morr_rhog": float(np.float32(morr_rhog)),
        "nz": nz, "ny": ny, "nx": nx,
        **_interior_parameters(interior_bounds, ny, nx),
    }
    entrypoint = "morrison_sediment_64" if nz <= 64 else "morrison_sediment_256"
    context = state._array_module.context if state is not None else theta.buffer.context
    return context.submit_kernel(
        entrypoint, arrays, parameters, exact_policy=False,
        grid_size=ny * nx)


def submit_morrison_finalize(state, arrays, *, morr_rhog,
                             interior_bounds=None):
    """Submit Morrison's float-only post-sedimentation conversion/final PSD."""
    _check_exact()
    if state is not None:
        _check_state(state)
    if not isinstance(arrays, dict) or set(arrays) != set(_FINAL):
        raise ValueError("Morrison finalization resources do not match its ABI")
    theta = arrays["theta"]
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("theta must be a three-dimensional MetalArray")
    shape = theta.shape
    for name in _FINAL:
        _check_array(name, arrays[name], shape)
    if any(arrays[name].buffer.context is not theta.buffer.context
           for name in _FINAL):
        raise ValueError("Morrison resources must share one Metal context")
    ncell = theta.size
    context = state._array_module.context if state is not None else theta.buffer.context
    ny, nx = shape[1], shape[2]
    bounds = _interior_parameters(interior_bounds, ny, nx)
    return context.submit_kernel(
        "morrison_finalize_levels", arrays,
        {"morr_rhog": float(np.float32(morr_rhog)), "nz": shape[0],
         "ncell": ncell,
         "ny": ny, "nx": nx, **bounds},
        exact_policy=False, grid_size=ncell)


def submit_morrison_validate_diagnostics(*, rainnc, rainncv, snownc,
                                         snowncv, graupelnc, graupelncv,
                                         sr, status, sr_upper):
    """Validate canonical MP10 surface diagnostics and return one uint32 word.

    Bits 0..6 flag nonfinite values in rainnc, rainncv, snownc, snowncv,
    graupelnc, graupelncv, and sr. Bit 7 flags negative rainncv; bit 8 flags
    negative sr; bit 9 flags sr > ``sr_upper``. Only the status word is read back.
    """
    arrays = {"rainnc": rainnc, "rainncv": rainncv, "snownc": snownc,
              "snowncv": snowncv, "graupelnc": graupelnc,
              "graupelncv": graupelncv, "sr": sr}
    if not isinstance(status, MetalArray) or status.shape != (1,):
        raise TypeError("Morrison diagnostic status must have shape (1,)")
    if (status.dtype != np.dtype(np.uint32)
            or not status.flags.c_contiguous):
        raise TypeError("Morrison diagnostic status must be contiguous uint32")
    first = rainnc
    if not isinstance(first, MetalArray) or first.ndim != 2:
        raise TypeError("Morrison diagnostics must be two-dimensional MetalArrays")
    shape = tuple(first.shape)
    context = status.buffer.context
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray) or array.shape != shape
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous
                or array.buffer.context is not context):
            raise ValueError(
                f"Morrison diagnostic {name} must be contiguous FP32 with "
                f"shape {shape} on the status context")
    ncell = int(np.prod(shape, dtype=np.int64))
    if min(shape) <= 0 or ncell > 0x7fffffff:
        raise ValueError("Morrison diagnostic extents must be positive int32")
    sr_upper32 = np.float32(sr_upper)
    if not np.isfinite(sr_upper32):
        raise ValueError("Morrison SR validation upper bound must be finite FP32")
    status.fill(np.uint32(0))
    context.submit_kernel(
        "morrison_validate_diagnostics", {**arrays, "status": status},
        {"ny": int(shape[0]), "nx": int(shape[1]),
         "ncell": ncell, "sr_upper": float(sr_upper32)},
        exact_policy=False, grid_size=ncell,
        threads_per_group=128)
    return int(status.to_numpy()[0])
