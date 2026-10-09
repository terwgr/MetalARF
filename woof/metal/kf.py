"""Metal launchers for Kain--Fritsch trigger history and column closure."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_COLUMN_TILE = 4096
_COLUMN_VOLUME_RESOURCES = (
    "u", "v", "temperature", "qv", "qc", "pressure", "exner", "dz", "w",
    "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqrcuten",
    "rqscuten", "updraft_out", "downdraft_out",
)
_COLUMN_SURFACE_FLOATS = (
    "rainc", "pratec", "cape_before", "cape_after", "closure_time",
    "nca_seconds",
)
_COLUMN_SURFACE_INTS = (
    "triggered", "shallow_out", "cloud_base", "cloud_top_out",
)


def submit_kf_trigger_history(state, w0avg: MetalArray, tst: np.float32):
    """Advance KF ``W0AVG`` on Metal, once per due cumulus call.

    WRF's KF_eta_CPS first averages adjacent interface vertical velocities,
    then applies ``(old * (TST - 1) + instantaneous) / TST``.  This kernel
    retains that FP32 statement order.  It owns only trigger-memory state;
    the KF_eta_CPS column tendency calculation remains explicitly unsupported.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal KF W0AVG supports ordinary FP32 arithmetic only")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal KF W0AVG requires a Metal DomainState")
    w = getattr(state, "w", None)
    if not isinstance(w, MetalArray) or not isinstance(w0avg, MetalArray):
        raise TypeError("Metal KF W0AVG requires MetalArray inputs")
    nz = int(w0avg.shape[0])
    ny, nx = (int(w0avg.shape[1]), int(w0avg.shape[2]))
    if (w.shape != (nz + 1, ny, nx)
            or w0avg.shape != (nz, ny, nx)
            or w.dtype != np.dtype(np.float32)
            or w0avg.dtype != np.dtype(np.float32)
            or not w.flags.c_contiguous or not w0avg.flags.c_contiguous):
        raise ValueError("Metal KF W0AVG requires contiguous FP32 w(nz+1,ny,nx) and W0AVG(nz,ny,nx)")
    if (w.buffer.context is not state._array_module.context
            or w0avg.buffer.context is not state._array_module.context):
        raise ValueError("Metal KF W0AVG arrays belong to another context")
    if any(dim <= 0 or dim > 0x7FFFFFFF for dim in (nz, ny, nx)):
        raise ValueError("Metal KF W0AVG extents must fit positive int32")
    grid_size = nz * ny * nx
    if grid_size > 0xFFFFFFFF:
        raise ValueError("Metal KF W0AVG grid exceeds the uint grid index")
    tst = np.float32(tst)
    if not np.isfinite(tst) or tst < np.float32(1.0):
        raise ValueError("Metal KF W0AVG TST must be finite and >= 1")
    arrays = {"w": w, "w0avg": w0avg}
    return state._array_module.context.submit_kernel(
        "kf_w0avg_update", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "tst": tst},
        exact_policy=False, grid_size=grid_size)


def submit_kf_columns(state, *, u, v, temperature, qv, qc, pressure,
                      exner, dz, w, dx, dt, cudt, phase_mode, table,
                      specified=False, spec_zone=0):
    """Run the ordinary WRF KF column closure in bounded column tiles.

    The CUDA warp ballot only reorders columns to improve locality; the Metal
    route keeps input order and assigns one thread to each column. The 52-slot
    workspace is allocated for at most ``_COLUMN_TILE`` columns and reused
    across submissions, avoiding a domain-sized scratch allocation.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal KF columns do not implement WRF exact-D arithmetic")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal KF columns require a Metal DomainState")
    if not isinstance(temperature, MetalArray) or temperature.ndim != 3:
        raise TypeError("Metal KF temperature must be a 3-D MetalArray")
    nz, ny, nx = map(int, temperature.shape)
    if nz < 8 or nz > 128:
        raise ValueError(f"Metal KF requires 8 <= nz <= 128, got {nz}")
    ncol = ny * nx
    if (min(ny, nx, ncol) <= 0
            or max(nz, ny, nx, ncol, nz * ncol) > 0x7fffffff):
        raise ValueError("Metal KF dimensions must fit positive int32 indexing")
    if not isinstance(specified, (bool, np.bool_)):
        raise TypeError("Metal KF specified must be a bool")
    spec_zone = int(spec_zone)
    if spec_zone < 0 or (specified and
                         (spec_zone < 1 or min(ny, nx) <= 2 * spec_zone)):
        raise ValueError("Metal KF specified mode requires a positive spec_zone and nonempty interior")
    volume_shape, surface_shape = (nz, ny, nx), (ny, nx)
    arrays = {"u": u, "v": v, "temperature": temperature, "qv": qv,
              "qc": qc, "pressure": pressure, "exner": exner, "dz": dz,
              "w": w}
    context = temperature.buffer.context
    for name in _COLUMN_VOLUME_RESOURCES[:9]:
        value = arrays[name]
        if (not isinstance(value, MetalArray) or value.shape != volume_shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(
                f"Metal KF {name} must be a same-context contiguous FP32 "
                f"array with shape {volume_shape}")
    scalars = tuple(np.float32(x) for x in (dx, dt, cudt))
    if any(not np.isfinite(x) or x <= np.float32(0.0) for x in scalars):
        raise ValueError("Metal KF dx, dt, and cudt must be finite and positive")
    if isinstance(phase_mode, (bool, np.bool_)):
        raise ValueError("Metal KF phase_mode must be an explicit enum")
    try:
        phase_mode = int(phase_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError("Metal KF phase_mode must be an integer enum") from exc
    if phase_mode not in (0, 1, 2, 3):
        raise ValueError(f"invalid Metal KF phase_mode {phase_mode}")

    xp = state._array_module
    metal_table = tuple(xp.asarray(value, dtype=np.float32) for value in
                        (table.temperature, table.qsat, table.thetae_base,
                         table.log_ratio))
    table_names = ("temperature_table", "qsat_table", "thetae_base",
                   "log_ratio")
    outputs = {name: xp.empty(volume_shape, dtype=np.float32) for name in
               ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                "rqrcuten", "rqscuten", "updraft_out", "downdraft_out")}
    outputs.update({name: xp.empty(surface_shape, dtype=np.float32)
                    for name in _COLUMN_SURFACE_FLOATS})
    outputs.update({name: xp.empty(surface_shape, dtype=np.int32)
                    for name in _COLUMN_SURFACE_INTS})
    tile = min(_COLUMN_TILE, ncol)
    workspace = xp.empty((52, nz, tile), dtype=np.float32)
    resources = {**arrays, **outputs,
                 **dict(zip(table_names, metal_table)),
                 "workspace": workspace}
    params_base = {
        "pressure_top": np.float32(table.pressure_top),
        "pressure_reciprocal": np.float32(table.pressure_reciprocal),
        "thetae_reciprocal": np.float32(table.thetae_reciprocal),
        "dx": scalars[0], "dt": scalars[1],
        "cudt": scalars[2], "phase_mode": phase_mode,
        "specified": int(bool(specified)), "spec_zone": spec_zone,
        "nz": nz, "ny": ny, "nx": nx,
    }
    for col0 in range(0, ncol, tile):
        span = min(tile, ncol - col0)
        context.submit_kernel(
            "kf_column", resources,
            {**params_base, "col0": col0, "tile_ncol": span},
            exact_policy=False, grid_size=span, threads_per_group=128)
    return outputs


def submit_kf_driver_merge(
        *, cu_rates: Mapping[str, MetalArray], cu_nca: MetalArray,
        cu_pratec: MetalArray, cu_raincv: MetalArray,
        new_rates: Mapping[str, MetalArray | None],
        nca_seconds: MetalArray, pratec: MetalArray, clock_dt: float,
                           specified: bool = False,
                           spec_zone: int = 0) -> None:
    """Hold-merge one KF result into the driver's persistent Metal arrays.

    The device kernel applies WRF's strict ``cu_nca < 0.5*clock_dt`` gate to
    both volume tendencies and the surface NCA/PRATEC/RAINCV values.  In
    specified mode the ``spec_zone`` ring is left untouched.  ``new_rates``
    must contain all six KF rate names; ``None`` is accepted for optional
    QC/QI/QR/QS rates and writes zero, matching the driver's fallback.
    """
    if not isinstance(new_rates, Mapping):
        raise TypeError("Metal KF merge new_rates must be a mapping")
    required = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                "rqrcuten", "rqscuten")
    if any(name not in new_rates for name in required):
        raise ValueError("Metal KF merge requires all six tendency names")
    if (not isinstance(cu_rates, Mapping)
            or any(name not in cu_rates for name in required)):
        raise ValueError("Metal KF merge cu_rates must contain all six targets")
    if not isinstance(specified, (bool, np.bool_)):
        raise TypeError("Metal KF merge specified must be a bool")
    spec_zone = int(spec_zone)
    clock_dt = np.float32(clock_dt)
    if not np.isfinite(clock_dt) or clock_dt <= np.float32(0.0):
        raise ValueError("Metal KF merge clock_dt must be finite and positive")

    targets = {name: cu_rates[name] for name in required}
    first = targets[required[0]]
    if not isinstance(first, MetalArray) or first.ndim != 3:
        raise TypeError("Metal KF merge targets must be 3-D MetalArrays")
    nz, ny, nx = map(int, first.shape)
    if (min(nz, ny, nx) <= 0
            or max(nz, ny, nx, nz * ny * nx) > 0x7fffffff):
        raise ValueError("Metal KF merge extents must fit positive int32 indexing")
    if spec_zone < 0 or (specified and
                         (spec_zone < 1 or min(ny, nx) <= 2 * spec_zone)):
        raise ValueError("Metal KF merge specified mode requires a positive spec_zone and nonempty interior")
    surface_shape = (ny, nx)
    context = first.buffer.context
    for name, value in targets.items():
        if (not isinstance(value, MetalArray) or value.shape != first.shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous or value.buffer.context is not context):
            raise ValueError(f"Metal KF merge target {name} must be a same-context contiguous FP32 volume")
    surfaces = {"cu_nca": cu_nca, "cu_pratec": cu_pratec,
                "cu_raincv": cu_raincv, "nca_seconds": nca_seconds,
                "pratec": pratec}
    for name, value in surfaces.items():
        if (not isinstance(value, MetalArray) or value.shape != surface_shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous or value.buffer.context is not context):
            raise ValueError(f"Metal KF merge {name} must be a same-context contiguous FP32 surface array")

    optional_names = ("rqccuten", "rqicuten", "rqrcuten", "rqscuten")
    optional_mask = 0
    inputs = {}
    for name in ("rthcuten", "rqvcuten", *optional_names):
        value = new_rates[name]
        if value is None:
            # Keep the resource binding valid; the inactive bit makes the
            # shader avoid reading this alias and write an exact zero.
            inputs[f"new_{name}"] = targets[name]
            continue
        if (not isinstance(value, MetalArray) or value.shape != first.shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous or value.buffer.context is not context):
            raise ValueError(f"Metal KF merge input {name} must be a same-context contiguous FP32 volume")
        inputs[f"new_{name}"] = value
        if name in optional_names:
            optional_mask |= 1 << optional_names.index(name)
    if inputs["new_rthcuten"] is targets["rthcuten"] or inputs["new_rqvcuten"] is targets["rqvcuten"]:
        raise ValueError("Metal KF merge requires new rthcuten and rqvcuten outputs")

    resources = {
        **inputs,
        **{f"cu_{name}": targets[name] for name in required},
        "nca_seconds": nca_seconds, "pratec": pratec,
        "cu_nca": cu_nca, "cu_pratec": cu_pratec,
        "cu_raincv": cu_raincv,
    }
    state_context = context
    # All state storage is owned by the same Metal context.  Any MetalArray
    # inputs above were checked against it before submission.
    params = {"nz": nz, "ny": ny, "nx": nx, "clock_dt": clock_dt,
              "optional_mask": optional_mask,
              "specified": int(bool(specified)), "spec_zone": spec_zone}
    # Surface writes update cu_nca. Keep them in a later dispatch so every
    # volume work item reads the same held NCA value before it is changed.
    state_context.submit_kernel(
        "kf_driver_merge", resources, {**params, "surface_only": 0},
        exact_policy=False, grid_size=nz * ny * nx, threads_per_group=128)
    state_context.submit_kernel(
        "kf_driver_merge", resources, {**params, "surface_only": 1},
        exact_policy=False, grid_size=ny * nx, threads_per_group=128)


def submit_kf_validate_outputs(values, active: int, status: MetalArray) -> int:
    """Check native KF tendency/diagnostic fields on Metal and read one word.

    The output order is the driver's six 3-D rates followed by NCA seconds
    and PRATEC, matching :func:`woof.core.kf.validate_kf_outputs`.
    """
    if len(values) != 8:
        raise ValueError(f"KF Metal validation requires 8 arrays, got {len(values)}")
    if not isinstance(status, MetalArray) or status.shape != (1,):
        raise TypeError("KF Metal validation status must have shape (1,)")
    if status.dtype != np.dtype(np.uint32) or not status.flags.c_contiguous:
        raise TypeError("KF Metal validation status must be contiguous uint32")
    shape3 = tuple(values[0].shape)
    shape2 = tuple(values[6].shape)
    if len(shape3) != 3 or len(shape2) != 2 or shape2 != shape3[1:]:
        raise ValueError("KF Metal validation requires matching 3-D/2-D shapes")
    context = status.buffer.context
    arrays = {name: value for name, value in zip(
        ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqrcuten",
         "rqscuten", "nca_seconds", "pratec"), values)}
    for name, value in arrays.items():
        expected = shape3 if name in (
            "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqrcuten",
            "rqscuten") else shape2
        if (not isinstance(value, MetalArray) or value.shape != expected
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(
                f"KF Metal validation {name} must be a same-context contiguous "
                f"FP32 array with shape {expected}")
    if active < 0 or active > 0xff:
        raise ValueError("KF Metal validation mask must fit eight bits")
    status.fill(np.uint32(0))
    count3 = int(np.prod(shape3, dtype=np.int64))
    count2 = int(np.prod(shape2, dtype=np.int64))
    if max(count3, count2) > 0x7fffffff:
        raise ValueError("KF Metal validation extents must fit positive int32")
    # The MSL validator addresses every output through one linear index.
    # Keep its typed resource shapes one-dimensional; the higher-level
    # validation above has already checked the WRF 3-D/2-D layouts.
    resources = {
        name: value.view((int(value.size),))
        for name, value in arrays.items()
    }
    resources["status"] = status
    context.submit_kernel(
        "kf_validate_outputs", resources,
        {"active": int(active), "count3": count3, "count2": count2},
        exact_policy=False, grid_size=max(count3, count2),
        threads_per_group=128)
    return int(status.to_numpy()[0])


__all__ = ["submit_kf_columns", "submit_kf_trigger_history",
           "submit_kf_validate_outputs"]
