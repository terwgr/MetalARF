"""Native MPAS wind reconstruction using selected pinned carrier math."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from ..metal_physics_prep_v841 import MetalMpasToPhysGeometryV841
from .metal_tendencies_v842 import _launch


class MetalMpasWinds:
    """Reconstruct resident winds and synchronously read one validation word.

    The CUDA resident path defers that check; this callable checks before
    returning so its caller cannot publish winds from invalid geometry.
    """
    def __init__(self, geometry):
        if not isinstance(geometry,MetalMpasToPhysGeometryV841):raise TypeError('native wind requires sealed Metal physics geometry')
        geometry.validate()
        self.geometry=geometry

    def __call__(self, atmosphere):
        velocity=atmosphere.saved.normal_velocity;g=self.geometry;g.validate()
        context=_context(g._edges_on_cell)
        if velocity.ndim!=2 or velocity.shape[1]!=g.n_edges or velocity.shape[0]<1:raise ValueError('wind velocity must be [level,edge]')
        _require('wind velocity',velocity,np.float32)
        if _context(velocity) is not context:raise ValueError('wind velocity and geometry must share a Metal context')
        fields={'edges':'edges_on_cell','count':'n_edges_on_cell','coeff':'coeffs_reconstruct','lat':'lat_cell','lon':'lon_cell'}
        live={name:getattr(g,'_'+source) for name,source in fields.items()}
        sealed={'sealed_'+name:getattr(g,'_sealed_'+source) for name,source in fields.items()}
        if any(_context(value) is not context for value in (*live.values(),*sealed.values())):raise ValueError('wind geometry must share a Metal context')
        garbage=g.regional_garbage_cell;donor=g.regional_garbage_donor_cell
        if garbage is not None:
            donor=garbage-1 if donor is None else int(donor)
            if not 0<=donor<garbage:raise ValueError('wind garbage donor must be a real cell')
        nz=int(velocity.shape[0]);xp=_arrays(context);shape=(nz,g.n_cells)
        x,y,z,u,v=(xp.empty(shape,np.float32) for _ in range(5));invalid=xp.zeros((1,),np.int32)
        arrays={'velocity':velocity,**live,**sealed,'x':x,'y':y,'z':z,'u':u,'v':v,'invalid':invalid}
        _launch('hexv842_mpas_wind_f32',g.n_cells,context,arrays,dict(nlev=np.int32(nz),ncells=np.int32(g.n_cells),nedges=np.int32(g.n_edges),maxedges=np.int32(g.max_edges)))
        if int(invalid.to_numpy()[0]):raise ValueError('native wind reconstruction failed geometry or input validation')
        if garbage is not None:
            _launch('hexv842_mpas_wind_donor_f32',nz,context,dict(u=u,v=v),dict(nlev=np.int32(nz),ncells=np.int32(g.n_cells),garbage=np.int32(garbage),donor=np.int32(donor)))
        return SimpleNamespace(u_p=u,v_p=v)
