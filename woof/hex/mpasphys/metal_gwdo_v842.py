"""MPAS v8.4.2 YSU orographic gravity-wave drag on Metal."""
from __future__ import annotations
import math
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _i, _require
from .metal_tendencies_v842 import _launch


def run_gwdo(*, prepared, mesh, sfc_input, tend_physics, diag_physics,
             dt_pbl, config_len_disp, constants):
    """Execute the source column operator and publish native tendency/diagnostic pools.

    Inputs are solve-sized RKIND [level,cell] arrays in MPAS order. Scheme
    initialization and finalization carry no private state; Registry pools
    own the persistent tendencies and diagnostics.
    """
    nlev,ncells=tend_physics['rublten'].shape
    if nlev<2 or ncells<1 or not math.isfinite(dt_pbl) or dt_pbl<=0 or not math.isfinite(config_len_disp) or config_len_disp<=0:
        raise ValueError('MPAS GWDO needs positive dt, length, columns and at least two levels')
    shape=(nlev,ncells)
    profiles={'u':prepared['u_p'][:nlev], 'v':prepared['v_p'][:nlev],
              'temperature':prepared['t_p'][:nlev], 'qv':prepared['qv_p'][:nlev],
              'p_mass_dry_hyd':prepared['pres_hydd_p'][:nlev],
              'p_interface_dry_hyd':prepared['pres2_hydd_p'][:nlev+1],
              'exner':prepared['pi_p'][:nlev], 'zmid':prepared['zmid_p'][:nlev],
              'rublten_in':tend_physics['rublten'], 'rvblten_in':tend_physics['rvblten']}
    for name,value in profiles.items():
        _require(name,value,np.float32,(nlev+1,ncells) if name=='p_interface_dry_hyd' else shape)
    static={'mesh_density':mesh['meshDensity'],'var2d':sfc_input['var2d'],'oc1':sfc_input['con'],
            **{name:sfc_input[name] for name in ('oa1','oa2','oa3','oa4','ol1','ol2','ol3','ol4')}}
    for name,value in static.items(): _require(name,value,np.float32,(ncells,))
    if set(constants)!={'gravity','cp','rd','fv','pi'} or not all(math.isfinite(float(x)) and x>0 for x in constants.values()):
        raise ValueError('MPAS GWDO needs positive native physics constants')
    context=_context(profiles['u']); arrays=_arrays(context)
    bytes_needed=4*(12*ncells*(nlev+1)+6*nlev*ncells+2*ncells)
    if bytes_needed > context.recommended_working_set_size:
        raise MemoryError('MPAS GWDO workspace exceeds Metal recommended working set')
    out={name:arrays.empty(shape,np.float32) for name in ('rublten_out','rvblten_out','dtaux3d','dtauy3d','rubldiff','rvbldiff')}
    out.update({name:arrays.empty((ncells,),np.float32) for name in ('dusfcg','dvsfcg')})
    scratch=arrays.empty((12,ncells,nlev+1),np.float32)
    _launch('hexv842_mpas_gwdo_f32',ncells,context,{**profiles,**static,**out,'scratch':scratch},
            dict(nlev=_i(nlev),ncells=_i(ncells),dt=np.float32(dt_pbl),len_disp=np.float32(config_len_disp),
                 **{name:np.float32(value) for name,value in constants.items()}))
    tend_physics['rublten'][...] = out['rublten_out']
    tend_physics['rvblten'][...] = out['rvblten_out']
    for name in ('dtaux3d','dtauy3d','rubldiff','rvbldiff','dusfcg','dvsfcg'):
        diag_physics[name][...] = out[name]
    return out


def gwdo_slot(call):
    if call.scheme != 'bl_ysu_gwdo':
        raise ValueError(f'MPAS Metal GWDO scheme {call.scheme!r} is not implemented')
    return run_gwdo(**call.pools)
