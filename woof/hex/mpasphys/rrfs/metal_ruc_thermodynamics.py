"""Native RRFS RUC land and ice saturation-table operators on Metal."""

import numpy as np

from woof.metal.array import MetalArrayModule

from .metal_ruc_boundaries import _launch


def saturation_table(context):
    """Initialize the shared native 5,001-word saturation table once."""
    table = getattr(context, "_rrfs_ruc_tbq", None)
    if table is None:
        table = MetalArrayModule(context).empty(5001, np.float32)
        context.submit_kernel("hexrrfs_ruc_tbq_init", {"tbq": table}, {"n": 1},
                              exact_policy=True, grid_size=1)
        context._rrfs_ruc_tbq = table
    return table


def lookup_saturation(temperature, table, output, *, solve_cells):
    """Apply the shared land and ice interpolation to finite Kelvin inputs."""
    _launch("qsn", temperature, {"tn": temperature, "tbq": table, "qsn": output}, solve_cells)
