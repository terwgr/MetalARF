"""Metal submission helpers for integration and full-state health reductions."""

from __future__ import annotations

import numpy as np
from woof.core import constants as c
from .array import MetalArray


_THREADS = 256
_I32_MAX = np.iinfo(np.int32).max


def _require_metal(state):
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal health reductions require a Metal state")


def _i32(name, value):
    value = int(value)
    if value < 0 or value > _I32_MAX:
        raise ValueError(f"Metal health {name} must fit a non-negative int32")
    return value


def submit_health_partial(state, partial, *, cfg=None, boundary_width=None):
    """Submit one 9-word partial row per 256-thread group.

    The caller owns ``partial`` and must retain it until the final fold has
    been submitted.  Indices remain split into exact low/high uint32 payloads
    inside float32 words, matching the CUDA record layout.
    """
    _require_metal(state)
    if state.u.size == 0 or state.w.size == 0 or state.thp.size == 0:
        raise ValueError("zero-size array to health maximum reduction")
    width = 0 if boundary_width is None else int(boundary_width)
    if boundary_width is not None and width <= 0:
        raise ValueError("boundary_width must be positive")
    ny, nx = (int(v) for v in state.w.shape[1:])
    if boundary_width is not None and min(ny, nx) < 2 * width + 1:
        raise ValueError(
            f"boundary_width={width} leaves an empty w interior for {ny} x {nx}")
    sizes = (int(state.u.size), int(state.w.size), int(state.thp.size))
    ncells = int(state.thp.size) if cfg is not None else 0
    ph = state.php if cfg is not None else state.w
    phb = state.phb if cfg is not None else state.w
    nph, nphb = int(ph.size), int(phb.size)
    if max(*sizes, ncells, nph, nphb) > _I32_MAX:
        raise ValueError("Metal health reduction sizes exceed the runtime int32 ABI")
    nblocks = min(256, max(1, (max(sizes) + _THREADS - 1) // _THREADS))
    if tuple(partial.shape) != (nblocks, 9):
        raise ValueError(
            f"partial must have shape {(nblocks, 9)}, got {partial.shape}")
    phb_full = int(cfg is not None and state.phb.ndim == 3)
    gravity = float(np.float32(c.G))
    ph_view = ph.view((int(ph.size),))
    phb_view = phb.view((int(phb.size),))
    parameters = {
        "nu": sizes[0], "nw": sizes[1], "nth": sizes[2],
        "ncells": ncells, "nph": nph, "nphb": nphb,
        "phb_full": phb_full, "width": width,
        "ny": ny, "nx": nx, "nblocks": nblocks, "gravity": gravity,
    }
    return state._array_module.context.submit_kernel(
        "health_partial",
        {"u": state.u.view((sizes[0],)), "w": state.w.view((sizes[1],)),
         "thp": state.thp.view((sizes[2],)), "ph": ph_view,
         "phb": phb_view, "partial": partial},
        parameters, exact_policy=False, grid_size=nblocks * _THREADS,
        threads_per_group=_THREADS)


def submit_health_final(context, partial, result, *, nblocks):
    """Fold partial records to the eight-word stability result."""
    nblocks = _i32("nblocks", nblocks)
    if nblocks == 0:
        raise ValueError("health_final needs at least one partial row")
    if tuple(result.shape) != (8,) or tuple(partial.shape) != (nblocks, 9):
        raise ValueError("health_final needs partial (nblocks, 9) and result (8,)")
    return context.submit_kernel(
        "health_final", {"partial": partial, "result": result},
        {"nblocks": nblocks}, exact_policy=False, grid_size=_THREADS,
        threads_per_group=_THREADS)


def submit_health_partial_tile(tile_state, partial, *, block_offset, bny,
                               bnx, jb, ib, iny, inx, jd, id, dny, dnx,
                               nz, u_extra, width=0, gravity=9.81):
    """Submit one tile-interior record, using domain coordinates for w ties."""
    _require_metal(tile_state)
    phb = getattr(tile_state, "phb", None)
    ph = getattr(tile_state, "php", None)
    have_geo = int(phb is not None and ph is not None)
    phb_full = int(have_geo and phb.ndim == 3)
    if not have_geo:
        ph = phb = tile_state.w
    values = {
        "block_offset": block_offset, "bny": bny, "bnx": bnx,
        "jb": jb, "ib": ib, "iny": iny, "inx": inx, "jd": jd,
        "id": id, "dny": dny, "dnx": dnx, "nz": nz,
        "u_extra": u_extra, "phb_full": phb_full,
        "have_geopotential": have_geo, "width": width,
    }
    q = {name: _i32(name, value) for name, value in values.items()}
    if q["iny"] <= 0 or q["inx"] <= 0 or q["nz"] <= 0:
        raise ValueError("tile health interior and nz must be positive")
    if q["u_extra"] not in (0, 1):
        raise ValueError("u_extra must be 0 or 1")
    nblocks = 256
    if tuple(partial.shape)[1:] != (9,) or partial.shape[0] < q["block_offset"] + nblocks:
        raise ValueError("partial does not have room for tile health rows")
    q.update(nblocks=nblocks, npartial=int(partial.shape[0]),
             nu=int(tile_state.u.size), nw=int(tile_state.w.size),
             nth=int(tile_state.thp.size), nph=int(ph.size),
             nphb=int(phb.size),
             npartial_words=int(partial.size),
             gravity=float(np.float32(gravity)))
    if max(q["nu"], q["nw"], q["nth"], q["nph"], q["nphb"],
           q["npartial_words"]) > _I32_MAX:
        raise ValueError("tile health resource sizes exceed the runtime int32 ABI")
    context = tile_state._array_module.context
    return context.submit_kernel(
        "health_partial_tile",
        {"u": tile_state.u.view((int(tile_state.u.size),)),
         "w": tile_state.w.view((int(tile_state.w.size),)),
         "thp": tile_state.thp.view((int(tile_state.thp.size),)),
         "ph": ph.view((int(ph.size),)), "phb": phb.view((int(phb.size),)),
         "partial": partial.view((int(partial.size),))},
        q, exact_policy=False, grid_size=nblocks * _THREADS,
        threads_per_group=_THREADS)


_HEALTH_BATCH_FIELDS = 30
_HEALTH_CHUNK_TARGET_GROUPS = 8192
_HEALTH_MIN_CHUNK = 4096
_HEALTH_INDEX_LIMIT = 1 << 48


def _health_words(array, name, *, allow_i32=False):
    if not isinstance(array, MetalArray) or not array.flags.c_contiguous:
        raise TypeError(f"Metal health {name} must be a contiguous MetalArray")
    if array.dtype == np.dtype(np.float32):
        return array.view((int(array.size),), np.uint32)
    if allow_i32 and array.dtype == np.dtype(np.int32):
        return array.view((int(array.size),), np.uint32)
    raise TypeError(f"Metal health {name} has unsupported dtype {array.dtype}")


def submit_validate_full_state(state, fields):
    """Batch grouped scans and one final fold, retaining all owners in flight.

    Metadata is uploaded as typed words. State arrays are only zero-copy word
    views; the only readback on success is the final pair of uint64 values.
    """
    _require_metal(state)
    xp = state._array_module
    context = xp.context
    fields = tuple(fields)
    if not fields:
        return None, None
    for field in fields:
        size = int(field.values.size)
        status_bit = int(field.rule.status_bit)
        if status_bit <= 0 or status_bit > np.iinfo(np.uint32).max:
            raise ValueError(
                f"Metal health status bit for {field.name!r} must fit uint32")
        if size >= _HEALTH_INDEX_LIMIT:
            raise ValueError(f"Metal health field {field.name!r} exceeds 48-bit index range")
        _health_words(field.values, field.name, allow_i32=True)
        if field.auxiliary is not None:
            aux = field.auxiliary
            if not isinstance(aux, MetalArray) or aux.dtype != np.dtype(np.float32):
                raise TypeError(f"Metal health auxiliary for {field.name!r} must be float32")
            if not aux.flags.c_contiguous or aux.buffer.context is not context:
                raise ValueError(f"Metal health auxiliary for {field.name!r} is not a contiguous local view")
            if field.aux_mode == "direct":
                if int(aux.size) != size:
                    raise ValueError(f"Metal health direct auxiliary for {field.name!r} must match values")
            elif field.aux_mode == "level":
                if field.plane_size <= 0 or int(aux.size) < (size + field.plane_size - 1) // field.plane_size:
                    raise ValueError(f"Metal health level auxiliary for {field.name!r} is too short")
            else:
                raise ValueError(f"unknown auxiliary mode {field.aux_mode!r}")
    total = sum(int(field.values.size) for field in fields)
    chunk = max(_HEALTH_MIN_CHUNK, total // _HEALTH_CHUNK_TARGET_GROUPS)
    chunk_counts = [(int(field.values.size) + chunk - 1) // chunk for field in fields]
    records = sum(chunk_counts)
    if records == 0:
        return None, None
    if records > np.iinfo(np.int32).max:
        raise ValueError("Metal health grouped scan exceeds the typed dispatch count range")
    # StateHealthValidator is called at the forecast's regular health-gate
    # cadence. Cache both device scratch and the resolved native resource
    # tables on the state so a gate does not allocate ~2 buffers per batch on
    # every step. The identity covers every value that affects the immutable
    # dispatch metadata and resource handles; model values themselves change
    # in place and are deliberately absent from the key.
    def resource_key(array):
        if array is None:
            return None
        return (id(array.buffer), int(array.byte_offset),
                np.dtype(array.dtype).str, int(array.size))

    signature = tuple((
        field.name, resource_key(field.values),
        int(field.rule.status_bit),
        None if field.rule.lower is None else float(np.float32(field.rule.lower)),
        None if field.rule.upper is None else float(np.float32(field.rule.upper)),
        bool(field.rule.strict_lower), resource_key(field.auxiliary),
        field.aux_mode, int(field.plane_size)) for field in fields)
    cache_key = (id(context), signature, chunk, tuple(chunk_counts))
    workspace = getattr(state, "_metal_health_workspace", None)
    if workspace is None or workspace["key"] != cache_key:
        dummy = xp.zeros((1,), dtype=np.uint32)
        partial = xp.empty((records * 2,), dtype=np.uint64)
        result = xp.empty((2,), dtype=np.uint64)
        calls = []
        group_base = 0
        for batch_start in range(0, len(fields), _HEALTH_BATCH_FIELDS):
            batch = fields[batch_start:batch_start + _HEALTH_BATCH_FIELDS]
            batch_counts = chunk_counts[batch_start:batch_start + len(batch)]
            groups = sum(batch_counts)
            if groups == 0:
                continue
            meta32 = np.zeros((120,), dtype=np.uint32)
            meta64 = np.zeros((155,), dtype=np.uint64)
            meta64[:5] = (len(batch), groups, batch_start, chunk, group_base)
            values = {}
            auxiliaries = {}
            prefix = 0
            for slot in range(_HEALTH_BATCH_FIELDS):
                if slot >= len(batch):
                    values[f"v{slot}"] = dummy
                    auxiliaries[f"a{slot}"] = dummy
                    continue
                field = batch[slot]
                size = int(field.values.size)
                flags = int(0 if field.values.dtype == np.dtype(np.float32) else 32)
                lower = field.rule.lower
                upper = field.rule.upper
                if lower is not None: flags |= 1
                if upper is not None: flags |= 2
                if field.rule.strict_lower: flags |= 4
                if field.auxiliary is not None:
                    flags |= 8
                    if field.aux_mode == "level": flags |= 16
                    auxiliaries[f"a{slot}"] = _health_words(
                        field.auxiliary, field.name + " auxiliary")
                else:
                    auxiliaries[f"a{slot}"] = dummy
                values[f"v{slot}"] = _health_words(
                    field.values, field.name, allow_i32=True)
                base = slot * 4
                meta32[base:base + 4] = (
                    flags, field.rule.status_bit,
                    0 if lower is None else np.asarray(
                        np.float32(lower)).view(np.uint32),
                    0 if upper is None else np.asarray(
                        np.float32(upper)).view(np.uint32))
                aux_extent = (0 if field.auxiliary is None
                              else int(field.auxiliary.size))
                plane = int(field.plane_size) if field.aux_mode == "level" else 0
                meta = 5 + slot * 5
                meta64[meta:meta + 5] = (
                    size, aux_extent, plane, prefix, batch_counts[slot])
                prefix += batch_counts[slot]
            meta32_array = xp.asarray(meta32, dtype=np.uint32)
            meta64_array = xp.asarray(meta64, dtype=np.uint64)
            scan_arrays = {**values, **auxiliaries,
                           "meta32": meta32_array,
                           "meta64": meta64_array, "partial": partial}
            call = context.prepare_kernel_call(
                "health_grouped_scan", scan_arrays, {"groups": groups},
                exact_policy=False, grid_size=groups * _THREADS,
                threads_per_group=_THREADS)
            calls.append(call)
            group_base += groups
        final = context.prepare_kernel_call(
            "health_grouped_final", {"partial": partial, "result": result},
            {"records": records}, exact_policy=False, grid_size=_THREADS,
            threads_per_group=_THREADS)
        calls.append(final)
        workspace = {"key": cache_key, "calls": tuple(calls),
                     "result": result, "partial": partial, "dummy": dummy}
        state._metal_health_workspace = workspace
    submission = context.submit_kernel_batch(
        workspace["calls"], label="health-full-state")
    return submission, workspace["result"]


def read_health_scalar(array, flat_index):
    """Read one FP32/INT32 scalar without converting a full field."""
    dtype = np.dtype(array.dtype)
    offset = int(array.byte_offset) + int(flat_index) * dtype.itemsize
    raw = array.buffer.read(dtype.itemsize, offset=offset)
    scalar = np.frombuffer(raw, dtype=dtype, count=1)[0]
    return float(np.float32(scalar))
