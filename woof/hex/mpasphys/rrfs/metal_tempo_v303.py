"""Metal launch boundary for the strict RRFS TEMPO prepared column."""

import hashlib
import json
import math
from operator import index
from pathlib import Path

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule

from .tempo_v303 import (
    CONSTANT_ARGUMENTS, DIAG_ARGUMENTS, FIELD_ARGUMENTS, MODEL_ARGUMENTS,
    PRECIP_ARGUMENTS, PROFILE_ARGUMENTS, RADAR_ARGUMENTS, STATE_FIELDS, SURFACE_ARGUMENTS,
    TABLE_LAYOUT, TEMPO_FORK_PINS, TEND_ARGUMENTS,
)

_MAX_LEVELS = 128
_TABLE_WORDS = sum(math.prod(shape) for _, shape in TABLE_LAYOUT)
TEMPO_TABLE_WITNESS_SCHEMA = "woof-rrfs-tempo-native-tables/v1"
TEMPO_TABLE_WITNESS_SOURCES = (
    "src/framework/mpas_kind_types.F",
    *("src/core_atmosphere/physics/physics_noaa/TEMPO/src/" + f"module_mp_tempo_{name}.F90"
      for name in ("cfgs", "params", "ml", "utils", "aerosols", "diags", "main", "driver")),
)


def _configuration_flags(flags):
    if len(flags) != 4 or any(flag not in (0, 1) for flag in flags):
        raise ValueError("RRFS TEMPO requires four boolean configuration flags")
    return tuple(map(int, flags))


def _table_buffer(context):
    if _TABLE_WORDS * 8 > min(context.max_buffer_length, context.recommended_working_set_size):
        raise MemoryError("RRFS TEMPO tables exceed the Metal buffer or working-set limit")
    return MetalArrayModule(context).empty(_TABLE_WORDS, np.float64)


def upload_table_witness(context, receipt, *, flags, pin_set, reference_sha256):
    """Upload a portable native table witness bound to an independent digest."""
    flags = _configuration_flags(flags)
    if flags[3]:
        raise NotImplementedError("Metal RRFS TEMPO tables require non-melting reflectivity")
    if not isinstance(reference_sha256, str) or len(reference_sha256) != 64:
        raise ValueError("RRFS TEMPO table witness requires an independent SHA-256")
    from ..native_library_build import source_manifest
    from .framework_init_rrfs import FrameworkInitializationRRFS
    manifest = source_manifest(pin_set)
    if manifest["pins"].get("rrfs-model") not in TEMPO_FORK_PINS:
        raise ValueError("RRFS TEMPO table witness requires a pinned fork")
    receipt = Path(receipt)
    packet = json.loads(receipt.read_text())
    layout = json.dumps(TABLE_LAYOUT, separators=(",", ":")).encode()
    sources = {name: manifest["files"][name]["sha256"] for name in TEMPO_TABLE_WITNESS_SOURCES}
    lookup = {name: manifest["tables"][name]["sha256"]
              for name, *_ in FrameworkInitializationRRFS.TEMPO_LOOKUP_FILES}
    expected = {"schema": TEMPO_TABLE_WITNESS_SCHEMA, "pins": manifest["pins"],
                "flags": list(flags), "rkind": 4, "byte_order": "little", "words": _TABLE_WORDS,
                "layout_sha256": hashlib.sha256(layout).hexdigest(),
                "sources": sources, "lookup_tables": lookup, "reference_sha256": reference_sha256,
                "blob": {"file": "tempo-tables-f64.bin", "sha256": reference_sha256}}
    for key, value in expected.items():
        if packet.get(key) != value:
            raise ValueError(f"RRFS TEMPO table witness differs at {key}")
    path = receipt.with_name(expected["blob"]["file"])
    if path.stat().st_size != _TABLE_WORDS * 8:
        raise ValueError("RRFS TEMPO table witness has a different byte count")
    tables = _table_buffer(context)
    digest = hashlib.sha256()
    offset = 0
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
            tables.buffer.write(block, offset=offset)
            offset += len(block)
    if offset != tables.nbytes or digest.hexdigest() != reference_sha256:
        raise ValueError("RRFS TEMPO table witness bytes differ from the independent reference")
    return tables


def upload_initialized_tables(context, native, *, flags):
    """Transport initialized non-melting native tables in TEMPO blob order."""
    if _configuration_flags(flags)[3]:
        raise NotImplementedError("Metal RRFS TEMPO tables require non-melting reflectivity")
    if native.build.get("source_pin") not in TEMPO_FORK_PINS:
        raise ValueError("RRFS TEMPO tables require a pinned fork framework")
    inactive = {"sbins_radar", "dsbins_radar", "gbins_radar", "dgbins_radar"}
    kinds = tuple(np.dtype(name) for name in ("float32", "float64", "int32", "int64"))
    values = []
    for name, shape in TABLE_LAYOUT:
        item = native.get_table(f"module_mp_tempo_params.{name}")
        if name in inactive:
            if item is not None:
                raise NotImplementedError("Metal RRFS TEMPO tables require non-melting reflectivity")
            values.append((None, math.prod(shape)))
            continue
        if item is None:
            raise ValueError(f"RRFS TEMPO table {name} is uninitialized")
        value, _ = item
        if (value.dtype not in kinds or not value.flags.f_contiguous
                or value.shape + (1,) * (len(shape) - value.ndim) != shape):
            raise ValueError(f"RRFS TEMPO table {name} has a different native kind or extent")
        values.append((value.reshape(-1, order="F"), value.size))
    tables = _table_buffer(context)
    offset = 0
    block_words = 1 << 20
    for value, count in values:
        for start in range(0, count, block_words):
            stop = min(start + block_words, count)
            words = (np.zeros(stop - start, np.float64) if value is None
                     else np.asarray(value[start:stop], dtype=np.float64))
            tables.buffer.write(memoryview(words).cast("B"), offset=offset * 8)
            offset += words.size
    return tables


def _native_dimensions(fields, solve_cells, planes=len(FIELD_ARGUMENTS)):
    if (not isinstance(fields, MetalArray) or fields.ndim != 3
            or fields.shape[0] != planes
            or fields.dtype != np.dtype(np.float32) or not fields.flags.c_contiguous):
        raise TypeError(f"RRFS TEMPO fields must be contiguous Metal float32 ({planes},nz,nc)")
    _, nz, nc = fields.shape
    solve = index(solve_cells)
    if not (1 <= nz <= _MAX_LEVELS and 1 <= solve <= nc
            and nz * nc <= np.iinfo(np.int32).max):
        raise ValueError("RRFS TEMPO solve columns and levels must fit the prepared extent")
    return fields.buffer.context, dict(nz=nz, nc=nc, solve=solve)


def _native_carrier(value, context, shape, name):
    if (not isinstance(value, MetalArray) or value.shape != shape
            or value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous
            or value.buffer.context is not context):
        raise TypeError(f"RRFS TEMPO {name} must be same-context contiguous float32 {shape}")
    return value


def initialize_native_aerosols(nwfa, nifa, nwfa2d, nifa2d, zgrid, area, *,
                               solve_cells, global_nwfa_extrema, global_nifa_extrema,
                               do_restart):
    """Initialize cold-start RRFS profiles and surface sources over solve cells."""
    if do_restart:
        return
    if not isinstance(nwfa, MetalArray) or nwfa.ndim != 2:
        raise TypeError("RRFS TEMPO aerosol profile must be Metal (nz,nc)")
    nz, nc = nwfa.shape
    solve = index(solve_cells)
    if not (2 <= nz <= _MAX_LEVELS and 0 <= solve <= nc <= np.iinfo(np.int32).max):
        raise ValueError("RRFS TEMPO aerosol initialization needs at least two levels and valid solve cells")
    if not solve:
        return
    from .framework_init_rrfs import _log_word
    w_low, w_high = map(np.float32, global_nwfa_extrema)
    i_low, i_high = map(np.float32, global_nifa_extrema)
    nwfa.buffer.context.submit_kernel(
        "hexrrfs_tempo_native_init_aerosols",
        dict(nwfa=nwfa, nifa=nifa, nwfa2d=nwfa2d, nifa2d=nifa2d, zgrid=zgrid, area=area),
        dict(nz=nz, nc=nc, solve=solve, initw=int(w_low == 0 and w_high == 0),
             initi=int(i_low == 0 and i_high == 0),
             log_w=_log_word(np.float32(50e6) / np.float32(300e6)),
             log_i=_log_word(np.float32(0.5e6) / np.float32(1.5e6))),
        exact_policy=True, grid_size=solve)


def prepare_native_columns(model, mesh, diagnostics, constants, prepared, saved, *,
                           solve_cells, aerosol, hail, presence):
    """Update the caller-owned MPAS prepared and initial tendency carriers.

    The caller supplies the existing prepared words for allocated optional
    fields whose scheme is disabled. Upstream leaves those planes untouched
    during preparation and still reads them when their presence bit is set.
    """
    context, dims = _native_dimensions(prepared, solve_cells)
    nz, nc = dims["nz"], dims["nc"]
    presence = index(presence)
    if not (0 <= presence < 32 and (not aerosol or presence & 7 == 7)
            and (not hail or presence & 24 == 24)):
        raise ValueError("RRFS TEMPO enabled schemes require their allocated prepared fields")
    names = (MODEL_ARGUMENTS[:8] + (MODEL_ARGUMENTS[8:11] if aerosol else ())
             + (MODEL_ARGUMENTS[11:13] if hail else ()) + MODEL_ARGUMENTS[13:])
    shape = (nz, nc)
    dummy = prepared[0]
    inputs = {name: _native_carrier(model[name], context, shape, name) if name in names else dummy
              for name in MODEL_ARGUMENTS}
    for name, source, extent in (("zz", mesh, shape), ("zgrid", mesh, (nz + 1, nc)),
                                 ("exner", diagnostics, shape), ("pressure_base", diagnostics, shape),
                                 ("pressure_p", diagnostics, shape)):
        inputs[name] = _native_carrier(source[name], context, extent, name)
    rv, rd = (np.float32(constants[name]) for name in ("R_v", "R_d"))
    if not (np.isfinite(rv) and np.isfinite(rd) and rv > 0 and rd > 0):
        raise ValueError("RRFS TEMPO requires positive finite gas constants")
    _native_carrier(saved, context, (len(TEND_ARGUMENTS), nz, nc), "saved tendencies")
    context.submit_kernel(
        "hexrrfs_tempo_native_from", {**inputs, "prepared": prepared, "saved": saved},
        {**dims, "aerosol": int(bool(aerosol)), "hail": int(bool(hail)),
         "presence": presence, "rv": rv, "rd": rd}, exact_policy=True, grid_size=nz * nc)


def apply_native_emissions(prepared, nwfa2d, *, solve_cells, dt):
    """Apply surface aerosol emissions once before the microphysics substeps."""
    context, dims = _native_dimensions(prepared, solve_cells)
    _native_carrier(nwfa2d, context, (dims["nc"],), "nwfa2d")
    dt = np.float32(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("RRFS TEMPO timestep must be finite and positive in binary32")
    context.submit_kernel(
        "hexrrfs_tempo_native_emissions", {"prepared": prepared, "nwfa2d": nwfa2d},
        {**dims, "aerosol": 1, "dt": dt}, exact_policy=True, grid_size=dims["solve"])


def pack_native_columns(prepared, packed, *, solve_cells, presence, unpack=False):
    """Convert theta to temperature before a substep and reverse it afterwards."""
    context, dims = _native_dimensions(prepared, solve_cells)
    nz, nc = dims["nz"], dims["nc"]
    _native_carrier(packed, context, (len(STATE_FIELDS), nz, nc), "packed columns")
    presence = index(presence)
    if not 0 <= presence < 32:
        raise ValueError("RRFS TEMPO requires the five-bit optional-carrier mask")
    context.submit_kernel(
        "hexrrfs_tempo_native_pack", {"prepared": prepared, "packed": packed},
        {**dims, "presence": presence, "phase": int(bool(unpack))},
        exact_policy=True, grid_size=nz * nc)


def accumulate_native_diagnostics(out, surface, profiles, *, solve_cells):
    """Accumulate substep precipitation and update the latest column profiles."""
    context, dims = _native_dimensions(out, solve_cells, planes=15)
    resources = {f"s_{name}": surface[name] for name in SURFACE_ARGUMENTS[:10]}
    resources.update((f"profile_{name}", profiles[name]) for name in PROFILE_ARGUMENTS)
    context.submit_kernel("hexrrfs_tempo_native_step_diags", {**resources, "out": out},
                          dims, exact_policy=True, grid_size=dims["solve"])


def publish_native_precip(prepared, surface, precip, *, solve_cells, finish, accrue, bucket):
    """Clear step totals or add the source-ordered totals and rain bucket."""
    context, dims = _native_dimensions(prepared, solve_cells)
    resources = {f"s_{name}": surface[name] for name in SURFACE_ARGUMENTS[:10]}
    names = PRECIP_ARGUMENTS[:9] + PRECIP_ARGUMENTS[11:]
    resources.update((f"pr_{name}", precip[name]) for name in names)
    context.submit_kernel(
        "hexrrfs_tempo_native_precip",
        {**resources, "prepared": prepared, "counter": precip["i_rainnc"]},
        {**dims, "finish": int(bool(finish)), "accrue": int(bool(accrue)),
         "bucket": np.float32(bucket)}, exact_policy=True, grid_size=dims["solve"])


def publish_native_tendencies(model, mesh, diagnostics, tendencies, constants, prepared,
                              saved, *, solve_cells, aerosol, hail, dt, top,
                              compmax=None, comp=None):
    """Apply the native vertical cutoff and publish prognostics and tendencies."""
    context, dims = _native_dimensions(prepared, solve_cells)
    nz = dims["nz"]
    top = index(top)
    dt = np.float32(dt)
    if nz < 2 or not 0 <= top <= nz or not np.isfinite(dt) or dt <= 0:
        raise ValueError("RRFS TEMPO publication needs two levels, a valid top and a positive timestep")
    if (compmax is None) != (comp is None):
        raise ValueError("RRFS TEMPO composite reflectivity requires both input and output")
    dummy = prepared[0]
    names = (MODEL_ARGUMENTS[:8] + (MODEL_ARGUMENTS[8:11] if aerosol else ())
             + (MODEL_ARGUMENTS[11:13] if hail else ()) + MODEL_ARGUMENTS[13:15])
    resources = {f"m_{name}": model[name] if name in names else dummy
                 for name in MODEL_ARGUMENTS[:15]}
    resources.update((f"mesh_{name}", mesh[name]) for name in ("zz", "zgrid"))
    resources.update((f"d_{name}", diagnostics[name]) for name in DIAG_ARGUMENTS)
    names = TEND_ARGUMENTS if aerosol else TEND_ARGUMENTS[:9]
    resources.update((f"t_{name}", tendencies[name] if name in names else dummy)
                     for name in TEND_ARGUMENTS)
    resources.update(prepared=prepared, saved=saved,
                     compmax=dummy[0] if compmax is None else compmax,
                     comp=dummy[0] if comp is None else comp)
    physics_constants = dict(zip(("rv", "rd", "rvord", "p0", "rcv", "grav"),
                                 (np.float32(constants[name]) for name in CONSTANT_ARGUMENTS)))
    context.submit_kernel(
        "hexrrfs_tempo_native_to", resources,
        {**dims, **physics_constants, "dt": dt, "top": top, "aerosol": int(bool(aerosol)),
         "hail": int(bool(hail)), "comp_enabled": int(comp is not None)},
        exact_policy=True, grid_size=dims["solve"])


def publish_native_radar(prepared, reflectivity, diagnostics, *, solve_cells, full, hourly):
    """Publish due RRFS radar and relative humidity on the native profile."""
    context, dims = _native_dimensions(prepared, solve_cells)
    status = MetalArrayModule(context).empty(dims["nc"], np.int32)
    device_hourly = isinstance(hourly, MetalArray)
    if device_hourly and hourly.size != 1:
        raise ValueError("RRFS TEMPO hourly switch must be a scalar word")
    resources = {name: diagnostics[name] for name in (*RADAR_ARGUMENTS, "relhum")}
    resources.update(prepared=prepared, refl10cm=reflectivity, status=status,
                     hourly_word=hourly.view((1,)) if device_hourly else status.view((1,)))
    params = {**dims, "hourly": 0 if device_hourly else int(bool(hourly)),
              "full": int(bool(full)), "has_hourly_word": int(device_hourly)}
    if full:
        context.submit_kernel("hexrrfs_tempo_native_radar", resources,
                              {**params, "validate_only": 1},
                              exact_policy=True, grid_size=dims["solve"])
        if np.any(status[:dims["solve"]].to_numpy()):
            raise ValueError("RRFS TEMPO radar needs two distinct layer centers for 1 km interpolation")
    context.submit_kernel(
        "hexrrfs_tempo_native_radar", resources,
        {**params, "validate_only": 0}, exact_policy=True, grid_size=dims["solve"])


def publish_native_rate(step, total, peak, averages, *, solve_cells, dt,
                        rolling=None, limits=(), times=()):
    """Carry the peak rate and native rolling precipitation windows."""
    if not isinstance(step, MetalArray) or step.ndim != 1:
        raise TypeError("RRFS TEMPO precipitation step must be a Metal surface plane")
    nc = step.size
    solve = index(solve_cells)
    dt = np.float32(dt)
    if not 1 <= solve <= nc or not np.isfinite(dt) or dt <= 0:
        raise ValueError("RRFS TEMPO precipitation rate needs solve cells and a positive timestep")
    has_rolling = rolling is not None
    if has_rolling:
        if not isinstance(rolling, MetalArray) or rolling.ndim != 2:
            raise TypeError("RRFS TEMPO rolling precipitation must be a Metal (windows,nc) array")
        longest = rolling.shape[0]
        n1, n5, n10 = map(index, limits)
        time1, time5, time10 = map(np.float32, times)
        if (not all(1 <= n <= longest for n in (n1, n5, n10))
                or not all(np.isfinite(t) and t > 0 for t in (time1, time5, time10))):
            raise ValueError("RRFS TEMPO precipitation window limits and times are invalid")
    else:
        longest = n1 = n5 = n10 = 1
        time1 = time5 = time10 = np.float32(1)
        rolling = step.view((1,))
    step.buffer.context.submit_kernel(
        "hexrrfs_tempo_native_prate",
        {"step": step, "total": total, "peak": peak, "rolling": rolling,
         **{f"ave{n}": averages[f"ave_prate_{n}min"] for n in (1, 5, 10)}},
        {"nc": nc, "solve": solve, "dt": dt, "longest": longest,
         "has_rolling": int(has_rolling), "n1": n1, "n5": n5, "n10": n10,
         "time1": time1, "time5": time5, "time10": time10}, exact_policy=True, grid_size=solve)


def _workspace_sizes(context):
    sizes = getattr(context, "_rrfs_tempo_v303_workspace_sizes", None)
    if sizes is None:
        words = MetalArrayModule(context).empty(2, np.uint64)
        context.submit_kernel(
            "hexrrfs_tempo_workspace_sizes", {"sizes": words}, {"n": 1},
            exact_policy=True, grid_size=1)
        sizes = tuple(int(value) for value in words.to_numpy())
        if any(size < 1 or size > np.iinfo(np.int32).max for size in sizes):
            raise ValueError("RRFS TEMPO returned invalid workspace dimensions")
        context._rrfs_tempo_v303_workspace_sizes = sizes
    return sizes


def launch_columns(fields, tables, *, dt, flags, first_call_main=False,
                   pbl=None, presence, solve_cells=None):
    """Update solve columns and return 15 diagnostic planes with zero halo words."""
    if (not isinstance(fields, MetalArray) or fields.ndim != 3
            or fields.shape[0] != len(STATE_FIELDS)
            or fields.dtype != np.dtype(np.float32) or not fields.flags.c_contiguous):
        raise TypeError("RRFS TEMPO fields must be contiguous Metal float32 (17,nz,ncol)")
    _, nz, ncol = fields.shape
    if not (1 <= nz <= _MAX_LEVELS and 1 <= ncol <= np.iinfo(np.int32).max):
        raise ValueError(f"RRFS TEMPO needs columns and 1 to {_MAX_LEVELS} levels")
    solve = ncol if solve_cells is None else index(solve_cells)
    if not 1 <= solve <= ncol:
        raise ValueError("RRFS TEMPO solve columns must fit the carrier extent")
    context = fields.buffer.context
    if (not isinstance(tables, MetalArray) or tables.shape != (_TABLE_WORDS,)
            or tables.dtype != np.dtype(np.float64) or not tables.flags.c_contiguous
            or tables.buffer.context is not context):
        raise TypeError("RRFS TEMPO needs the same-context native binary64 table layout")
    if pbl is not None and (not isinstance(pbl, MetalArray)
            or pbl.shape != (2, nz, ncol) or pbl.dtype != np.dtype(np.float32)
            or not pbl.flags.c_contiguous or pbl.buffer.context is not context):
        raise TypeError("RRFS TEMPO PBL fields must be same-context float32 (2,nz,ncol)")
    aerosol, hail, subgrid, melting = _configuration_flags(flags)
    if melting:
        raise NotImplementedError("Metal RRFS TEMPO melting reflectivity requires CUDA-equivalent vendor math")
    presence = index(presence)
    if not 0 <= presence < 32:
        raise ValueError("RRFS TEMPO requires the five-bit optional-carrier mask")
    dt = np.float32(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("RRFS TEMPO timestep must be finite and positive in binary32")

    work_words, state_words = _workspace_sizes(context)
    use_pbl = pbl is not None and bool(subgrid)
    bl_words = 2 * nz if use_pbl else 1
    diagnostic_bytes = 15 * nz * ncol * 4
    visible_bytes = fields.nbytes + tables.nbytes + diagnostic_bytes
    if pbl is not None:
        visible_bytes += pbl.nbytes
    per_slot = work_words * 8 + (state_words + bl_words) * 4
    available = context.recommended_working_set_size - visible_bytes
    slots = min(solve, available // per_slot,
                context.max_buffer_length // (work_words * 8),
                context.max_buffer_length // (state_words * 4),
                context.max_buffer_length // (bl_words * 4))
    if slots < 1 or diagnostic_bytes > context.max_buffer_length:
        raise MemoryError("RRFS TEMPO exceeds the Metal workspace budget")

    arrays = MetalArrayModule(context)
    diagnostics = (arrays.empty if solve == ncol else arrays.zeros)((15, nz, ncol), np.float32)
    resources = {
        "fields": fields, "diagnostics": diagnostics, "tables": tables,
        "pbl": pbl if use_pbl else fields.view((1,)),
        "work": arrays.empty((slots, work_words), np.uint64),
        "state": arrays.empty((slots, state_words), np.uint32),
        "blwork": arrays.empty((slots, bl_words), np.float32),
    }
    parameters = {
        "nz": nz, "ncol": ncol, "dt": dt, "aerosol": aerosol, "hail": hail,
        "use_pbl": int(use_pbl), "first": int(bool(first_call_main)),
        "presence": presence, "slots": slots, "work_words": work_words,
        "state_words": state_words, "bl_words": bl_words,
    }
    for offset in range(0, solve, slots):
        batch = min(slots, solve - offset)
        context.submit_kernel(
            "hexrrfs_tempo_columns", resources,
            {**parameters, "offset": offset, "batch": batch},
            exact_policy=True, grid_size=batch)
    return diagnostics
