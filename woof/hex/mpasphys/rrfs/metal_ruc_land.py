"""Pack and launch the retained RRFS RUC land closure on Metal."""

from operator import index

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule

from .ruc_lsm_8c9f1a8 import ARGUMENTS, COLUMN_EXTENTS, arena_words


_ENTRY = "land_ruc_land"
_ATMOS = ("z3d", "p8w", "t3d", "qv3d", "qc3d", "rho3d")
_SOIL = ("soilmois", "sh2o", "tso", "smfr3d", "keepfr3dflag")
_FORCING = ("rainbl", "frzfrac", "glw", "gsw", "vegfra", "albbck", "tbot", "xland", "xice",
            "shdmin", "shdmax")
_FLAGS = ("spp_lsm", "mosaic_lu", "mosaic_soil")
_PACKED = frozenset(_ATMOS + _SOIL + _FORCING)


def solve_land(arguments, scalars, tables, tbq, *, solve_cells, present=None,
               workspace=None, entry_ilnb=None, entry_calls=None):
    """Advance independent prepared work views using the native tile stop rule."""
    anchor = arguments["soilt"]
    if not isinstance(anchor, MetalArray) or anchor.ndim != 1:
        raise TypeError("RRFS RUC land requires a one-dimensional Metal surface plane")
    nc, solve = anchor.size, index(solve_cells)
    if not 0 <= solve <= nc <= np.iinfo(np.int32).max:
        raise ValueError("RRFS RUC land solve columns must fit the surface extent")
    if solve == 0:
        return

    resources, params = {}, {}
    presence = {} if present is None else present
    for field in ARGUMENTS[_ENTRY]:
        name = field["name"]
        if field["dim"] or name in _FLAGS:
            if name not in _PACKED:
                resources[name] = arguments[name]
        else:
            params[name] = 1 if name in COLUMN_EXTENTS else scalars[name]
        if field["optional"]:
            params[f"has_{name}"] = int(bool(presence.get(name, True)))
    nsl, nlcat, nscat = (index(params[name]) for name in ("nsl", "nlcat", "nscat"))
    kms, kme, kts, kte = (index(params[name]) for name in ("kms", "kme", "kts", "kte"))
    if nsl < 3 or min(nlcat, nscat) < 1 or not (kms <= 1 <= kme and kms <= kts <= kte <= kme):
        raise ValueError("RRFS RUC land requires valid soil, category and atmospheric extents")
    nlev = kme - kms + 1
    rows = arena_words(_ENTRY, {"nsl": nsl, "nlcat": nlcat, "nscat": nscat})
    if rows is None or nc * max(rows, len(_ATMOS) * nlev, len(_SOIL) * nsl, nlcat, nscat) >= 2**31:
        raise ValueError("RRFS RUC land extents exceed the kernel's int32 offsets")
    context = anchor.buffer.context
    arrays = MetalArrayModule(context)
    for name, fields, shape in (("atmos", _ATMOS, (nlev, nc)), ("soil", _SOIL, (nsl, nc)),
                                ("forcing", _FORCING, (nc,))):
        values = [arguments[field] for field in fields]
        if any(not isinstance(value, MetalArray) or value.shape != shape
               or value.dtype != np.float32 or value.buffer.context is not context for value in values):
            raise ValueError(f"RRFS RUC land {name} fields require native packed float32 views")
        packed = resources[name] = arrays.empty((len(fields), *shape), np.float32)
        context.submit_bookkeeping(tuple((value, packed[i]) for i, value in enumerate(values)))
    if workspace is None:
        if rows * nc * 4 > min(context.max_buffer_length, context.recommended_working_set_size):
            raise MemoryError("RRFS RUC land workspace exceeds the Metal device budget")
        workspace = arrays.empty((rows, nc), np.float32)
    for value in (entry_ilnb, entry_calls):
        if value is not None and value.shape != (nc,):
            raise ValueError("RRFS RUC land entry capture must match the column extent")
    error, stop = arrays.zeros(1, np.int32), arrays.empty((2,), np.int32)
    resources.update(tables=tables, tbq=tbq, workspace=workspace, error=error, tile_stop=stop,
                     entry_ilnb=error if entry_ilnb is None else entry_ilnb,
                     entry_calls=error if entry_calls is None else entry_calls)
    params.update(nc=nc, solve=solve, nlev=nlev, arena_words=rows,
                  has_entry=int(entry_ilnb is not None), has_trace=int(entry_calls is not None))
    # The retained Boundary scans its gathered solve columns, before any halo.
    context.submit_kernel("hexrrfs_ruc_land_preflight", {"lakemask": arguments["lakemask"], "stop": stop},
                          {"nc": nc, "solve": solve, "lakemodel": params["lakemodel"]},
                          exact_policy=True, grid_size=1)
    context.submit_kernel("hexrrfs_ruc_land", resources, params, exact_policy=True, grid_size=solve)
    status = int(error.to_numpy()[0])
    if status & 4:
        raise ValueError("RRFS RUC land SPP profiles must include levels 1 through nsl")
    if status:
        raise RuntimeError(f"RRFS RUC land reached a native fatal-error branch ({status})")
    context.submit_bookkeeping(tuple((resources["soil"][i], arguments[name]) for i, name in enumerate(_SOIL)))
    return stop
