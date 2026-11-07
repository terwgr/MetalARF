"""MPAS v8.4.2 momentum-rate projection to resident Metal edges."""
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _i, _require
from .metal_tendencies_v842 import _launch


def project(*, geometry, zonal, meridional, invalid):
    """Project halo-exchanged cell rates; caller supplies native's zero dummy cell."""
    geometry.validate()
    nlev, ncells = zonal.shape
    if ncells != geometry.n_cells:
        raise ValueError('MPAS momentum pool and geometry cell extents differ')
    _require('zonal', zonal, np.float32, (nlev, ncells))
    _require('meridional', meridional, np.float32, (nlev, ncells))
    _require('invalid', invalid, np.int32, (1,))
    context = _context(zonal)
    arrays = _arrays(context)
    shape = (nlev, geometry.n_edges)
    out = arrays.empty(shape, np.float32)
    mass = arrays.ones(shape, np.float32)
    _launch('hexv841_project_edges_v841_f32', geometry.n_edges, context,
            dict(cells=geometry.cells_on_edge, east=geometry.east_cell,
                 north=geometry.north_cell, normal=geometry.edge_normal_vectors,
                 du=zonal, dv=meridional, rho_edge=mass, out_ru=out, invalid=invalid),
            dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(geometry.n_edges)))
    return out
