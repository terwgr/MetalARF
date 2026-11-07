"""Launch the retained RRFS RUC sea-ice closure on packed Metal work views."""

from operator import index

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule

from .ruc_lsm_8c9f1a8 import ARGUMENTS, COLUMN_EXTENTS, arena_words


_ENTRY = "ice_ruc_ice"
_FIELDS = ARGUMENTS[_ENTRY]
_OPTIONAL = tuple(field["name"] for field in _FIELDS if field["optional"])


def solve_ice(arguments, scalars, tables, tbq, *, solve_cells, present=None,
              workspace=None, entry_ilnb=None, entry_calls=None):
    """Advance prepared ice columns before the caller blends and publishes them.

    Work views retain the full column stride and the source's independent
    gather arrays. Optional precipitation and SPP arrays require full storage
    even when the source presence flag is false.
    """
    anchor = arguments["soilt"]
    if not isinstance(anchor, MetalArray) or anchor.ndim != 1:
        raise TypeError("RRFS RUC ice requires a one-dimensional Metal surface plane")
    nc, solve = anchor.size, index(solve_cells)
    if not 0 <= solve <= nc <= np.iinfo(np.int32).max:
        raise ValueError("RRFS RUC ice solve columns must fit the surface extent")
    if solve == 0:
        return

    resources, params = {}, {}
    for field in _FIELDS:
        name = field["name"]
        if field["dim"] or name == "spp_lsm":
            resources[name] = arguments[name]
        elif name in COLUMN_EXTENTS:
            params[name] = 1  # The retained launcher packs one logical cell per lane.
        else:
            params[name] = scalars[name]
    nsl = index(params["nsl"])
    kms, kme, kts, kte = (index(params[name]) for name in ("kms", "kme", "kts", "kte"))
    if nsl < 3 or not (kms <= 1 <= kme and kms <= kts <= kte <= kme):
        raise ValueError("RRFS RUC ice requires valid soil and atmospheric extents")
    rows = arena_words(_ENTRY, {"nsl": nsl})
    if rows is None or rows * nc >= 2**31 or (kme - kms + 1) * nc >= 2**31:
        raise ValueError("RRFS RUC ice extents exceed the kernel's int32 offsets")
    context = anchor.buffer.context
    arrays = MetalArrayModule(context)
    if workspace is None:
        if rows * nc * 4 > min(context.max_buffer_length, context.recommended_working_set_size):
            raise MemoryError("RRFS RUC ice workspace exceeds the Metal device budget")
        workspace = arrays.empty((rows, nc), np.float32)
    error = arrays.zeros(1, np.int32)
    resources.update(tables=tables, tbq=tbq, workspace=workspace, error=error,
                     entry_ilnb=error if entry_ilnb is None else entry_ilnb,
                     entry_calls=error if entry_calls is None else entry_calls)
    if any(value is not None and value.shape != (nc,) for value in (entry_ilnb, entry_calls)):
        raise ValueError("RRFS RUC ice entry capture must match the column extent")
    if entry_calls is not None:
        entry_calls.fill(0)
    params.update(nc=nc, solve=solve, nlev=kme - kms + 1, arena_words=rows,
                  has_entry=int(entry_ilnb is not None), has_trace=int(entry_calls is not None))
    presence = {} if present is None else present
    params.update((f"has_{name}", int(bool(presence.get(name, True)))) for name in _OPTIONAL)
    context.submit_kernel("hexrrfs_ruc_ice", resources, params, exact_policy=True, grid_size=solve)
    status = int(error.to_numpy()[0])
    if status & 2:
        raise ValueError("RRFS RUC ice SPP profiles must include levels 1 through nsl")
    if status & 1:
        raise RuntimeError("RRFS RUC ice saturation-temperature iteration left its table range")
