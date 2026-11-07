"""Typed launchers for the generated classic Thompson column on Metal."""

from __future__ import annotations

import numpy as np

from .array import MetalArray

_FLOAT = np.dtype(np.float32)
_DOUBLE = np.dtype(np.float64)
_THREADS = 32
TIERS = (64, 128, 256)
#: Float and binary64 level-array words per column, as a multiple of the
#: tier's level capacity plus a constant (tools/thompson_column_metal).
_STRIDE_F = (113, 8)
_STRIDE_D = (71, 0)
#: Largest workspace one launch may take.
WORKSPACE_BYTES = 1 << 28

TABLE_NAMES = (
    "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
    "tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz", "tpi_qcfz", "tni_qcfz",
    "tps_iaus", "tni_iaus", "tpi_ide", "t_Efrw", "t_Efsw",
)
SURFACE_NAMES = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                 "graupelncv", "sr")


def _flat(array, name, shape=None):
    if (not isinstance(array, MetalArray) or array.dtype != _FLOAT
            or not array.flags.c_contiguous):
        raise TypeError(f"Thompson column {name} requires contiguous float32 Metal storage")
    if shape is not None and array.shape != shape:
        raise ValueError(f"Thompson column {name} has shape {array.shape}, expected {shape}")
    return array.view((array.size,), byte_offset=array.byte_offset)


_WORKSPACES = {}


def _module_of(array):
    from .array import MetalArrayModule
    return MetalArrayModule(array.buffer.context)


def _workspace(xp, name, words, dtype):
    """The launch workspace, kept per Metal context and grown on demand."""
    key = (id(xp.context), name)
    held = _WORKSPACES.get(key)
    if held is None or held[0] is not xp.context or held[1].size < words:
        held = (xp.context, xp.empty((words,), dtype))
        _WORKSPACES[key] = held
    return held[1]


def _table(array, name):
    if (not isinstance(array, MetalArray) or array.dtype != _DOUBLE
            or not array.flags.c_contiguous):
        raise TypeError(f"Thompson column table {name} requires contiguous float64 Metal storage")
    return array.view((array.size,), byte_offset=array.byte_offset)


def submit_thompson_column(state_fields, surface, effective_radius, refl, tables,
                           *, nz, dt, diagnostic_due):
    """One generated-column launch over every (ny, nx) column.

    ``state_fields`` maps qv qc qi qr qs qg ni nr temperature pressure w dz to
    (nz, ny, nx) float32 arrays; ``surface`` maps SURFACE_NAMES to (ny, nx)
    arrays; ``effective_radius`` is (effc, effi, effs); ``tables`` maps
    TABLE_NAMES to float64 arrays.
    """
    shape = state_fields["qv"].shape
    if len(shape) != 3 or shape[0] != nz:
        raise ValueError("Thompson column fields must be (nz, ny, nx)")
    columns = shape[1] * shape[2]
    capacity = next((k for k in TIERS if nz <= k), None)
    if capacity is None:
        raise ValueError("generated Thompson column capacity is 256 levels")
    resources = {}
    for name in ("qv", "qc", "qi", "qr", "qs", "qg", "ni", "nr", "temperature",
                 "pressure", "dz"):
        resources[name] = _flat(state_fields[name], name, shape)
    # The column reads w at the bottom face of each mass level; the model's
    # (nz + 1)-face array is passed whole.
    w = state_fields["w"]
    if w.shape not in (shape, (nz + 1,) + shape[1:]):
        raise ValueError(f"Thompson column w has shape {w.shape}")
    resources["w"] = _flat(w, "w")
    for name in SURFACE_NAMES:
        resources[name] = _flat(surface[name], name, shape[1:])
    for name, array in zip(("effc", "effi", "effs"), effective_radius):
        resources[name] = _flat(array, name, shape)
    resources["refl"] = _flat(refl, "refl", shape)
    for name in TABLE_NAMES:
        resources[name] = _table(tables[name], name)
    context = resources["qv"].buffer.context
    dt32 = np.float32(dt)
    xp = _module_of(state_fields["qv"])
    words_f = _STRIDE_F[0] * capacity + _STRIDE_F[1]
    words_d = _STRIDE_D[0] * capacity + _STRIDE_D[1]
    chunk = columns
    if capacity > 64:
        chunk = max(_THREADS, min(columns, WORKSPACE_BYTES // (4 * words_f + 8 * words_d)))
        resources["ws_f"] = _workspace(xp, "ws_f", chunk * words_f, np.float32)
        resources["ws_d"] = _workspace(xp, "ws_d", chunk * words_d, np.float64)
    else:
        resources["ws_f"] = _workspace(xp, "ws_f", 1, np.float32)
        resources["ws_d"] = _workspace(xp, "ws_d", 1, np.float64)
    call = None
    for first in range(0, columns, chunk):
        call = context.submit_kernel(
            f"thompson_column_step_k{capacity}", resources,
            {"diagnostic_due": int(bool(diagnostic_due)), "nz": int(nz),
             "columns": int(columns), "dt": float(dt32), "first": int(first)},
            exact_policy=False, grid_size=min(chunk, columns - first),
            threads_per_group=_THREADS)
    return call


def submit_column_prepare(thb, thp, phb, php, pii, th, temperature, dz,
                          h_diabatic, *, gravity):
    nz, ny, nx = thp.shape
    columns = ny * nx
    resources = {"thb": _flat(thb, "thb"), "thp": _flat(thp, "thp"),
                 "phb": _flat(phb, "phb"), "php": _flat(php, "php"),
                 "pii": _flat(pii, "pii"), "th": _flat(th, "th"),
                 "temperature": _flat(temperature, "temperature"),
                 "dz": _flat(dz, "dz"), "h_diabatic": _flat(h_diabatic, "h_diabatic")}
    return thp.buffer.context.submit_kernel(
        "thompson_column_prepare", resources,
        {"gravity": float(np.float32(gravity)), "size": nz * columns,
         "columns": columns, "thb_full": int(thb.ndim == 3),
         "phb_full": int(phb.ndim == 3)},
        exact_policy=False, grid_size=nz * columns, threads_per_group=256)


def submit_column_finish(temperature, pii, th, thp, h_diabatic, *, limit, dt,
                         no_mp_heating):
    resources = {"temperature": _flat(temperature, "temperature"),
                 "pii": _flat(pii, "pii"), "th": _flat(th, "th"),
                 "thp": _flat(thp, "thp"),
                 "h_diabatic": _flat(h_diabatic, "h_diabatic")}
    return temperature.buffer.context.submit_kernel(
        "thompson_column_finish", resources,
        {"limit": float(np.float32(limit)), "dt": float(np.float32(dt)),
         "no_mp_heating": int(bool(no_mp_heating)), "size": temperature.size},
        exact_policy=False, grid_size=temperature.size, threads_per_group=256)
