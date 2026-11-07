"""Native WSM6 radar on bounded Metal column workspaces."""
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch


def launch_radar(fields, diagnostics, constants, *, state=None, solve_cells=None, running_max=False):
    nz, nc = fields['qv'].shape
    count = nc if solve_cells is None else int(solve_cells)
    if not 2 <= nz <= 256 or not 0 <= count <= nc or nc < 1:
        raise ValueError('invalid native WSM6 radar extents')
    context = _context(fields['qv'])
    xp = _arrays(context)
    resources = {n: fields[n] for n in ('qv','qr','qs','qg','th','pii','p','zgrid')}
    resources.update(zip(('refl','refl_max','refl_1km','refl_1km_max'),
        (diagnostics[n] for n in ('refl10cm','refl10cm_max','refl10cm_1km','refl10cm_1km_max'))))
    for name,value in tuple(resources.items()):
        shape = (nc,) if name in ('refl_max','refl_1km','refl_1km_max') else (nz+1,nc) if name=='zgrid' else (nz,nc)
        _require(name,value,np.float32,shape)
        if _context(value) is not context:
            raise ValueError('WSM6 radar arrays must share a Metal context')
        resources[name] = value.view((value.size,))
    _require('WSM6 constants',constants,np.float32,(26,))
    if _context(constants) is not context:
        raise ValueError('WSM6 constants must share the column context')
    if state is None:
        state = xp.empty((89,), np.float32)
        _launch('hexv842_wsm6_state_f32',1,context,
                dict(constants=constants,hail=xp.asarray(np.zeros(1,np.int32)),state=state,status=xp.empty((1,),np.int32)),
                dict(count=np.int32(1)))
        stride = 1
    else:
        _require('WSM6 state',state,np.float32,(89*nc,))
        stride = nc
    if not count:
        return
    resources.update(constants=constants,state=state)
    available = context.recommended_working_set_size - sum(v.nbytes for v in resources.values())
    chunk = min(count,32,(available-3)//((nz+3)*(18*4+6*8+3)))
    if chunk < 1:
        raise MemoryError('native WSM6 radar workspace exceeds the Metal working set')
    resources.update(ws_f=xp.empty((chunk*18*(nz+3),),np.float32),
                     ws_d=xp.empty((chunk*6*(nz+3),),np.uint64),
                     ws_b=xp.empty(((chunk*3*(nz+3)+3)//4,),np.uint32))
    for first in range(0,count,chunk):
        _launch('hexv842_wsm6_radar_f32',min(chunk,count-first),context,resources,
            dict(nz=np.int32(nz),nc=np.int32(nc),first=np.int32(first),count=np.int32(min(chunk,count-first)),
                 state_stride=np.int32(stride),running_max=np.int32(running_max)))


def relative_humidity(fields, relhum, *, solve_cells):
    nz,nc=fields['qv'].shape
    resources={n:fields['p' if n=='pres' else n].view((nz*nc,))
               for n in ('th','pii','qv','pres')}
    resources['relhum']=relhum.view((nz*nc,))
    _launch('hexv842_wsm6_relhum_f32',nz*nc,_context(relhum),resources,
            dict(nz=np.int32(nz),nc=np.int32(nc),ns=np.int32(solve_cells)))
