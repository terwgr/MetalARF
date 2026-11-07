"""Native MPAS Thompson conversion boundaries; the full column slot is unfinished."""
import numpy as np
from ..metal_dynamics_v841 import _context, _require
from .metal_tendencies_v842 import _launch
from . import thompson_v842 as native


def launch_conversion(phase,model,mesh,diagnostics,tendencies,fields,constants,*,mp_top_level=None,dt_dyn=None):
    if phase not in ('from','to'):
        raise ValueError('MPAS conversion phase must be from or to')
    nz,nc=fields['qv'].shape
    if not 2<=nz<=native.MAX_LEVELS or nc<1:
        raise ValueError('native Thompson conversion requires 2 through 256 levels and nonempty columns')
    arrays=tuple(model[n] for n in native.MODEL_ARGUMENTS)+tuple(mesh[n] for n in native.MESH_ARGUMENTS)
    arrays+=tuple(diagnostics[n] for n in native.DIAG_ARGUMENTS)+tuple(tendencies[n] for n in native.TENDENCY_ARGUMENTS)
    arrays+=tuple(fields[n] for n in native.FIELD_ARGUMENTS)+tuple(fields[n] for n in ('rho','z'))
    arrays+=tuple(diagnostics[n] for n in ('surface_pressure','tend_sfc_pressure'))
    context=_context(fields['qv']);resources={}
    for name,value in zip(native._CONVERSION_ARRAYS,arrays):
        shape=(nc,) if name in ('d_surface_pressure','d_tend_sfc_pressure') else (nz,nc)
        if name=='g_zgrid':shape=(nz+1,nc)
        if name=='m_w' and value.shape==(nz+1,nc):shape=value.shape
        _require(name,value,np.float32,shape)
        if _context(value) is not context:
            raise ValueError('native Thompson conversion arrays belong to different Metal contexts')
        resources[name]=value.view((value.size,))
    if phase=='to':
        if mp_top_level is None or not 1<=mp_top_level<=nz:
            raise ValueError('native Thompson top level must be a 1-based level inside the tile')
        if dt_dyn is None or not np.isfinite(dt_dyn) or dt_dyn<=0:
            raise ValueError('native Thompson conversion requires positive finite dynamical dt')
    params=dict(nz=np.int32(nz),columns=np.int32(nc),top=np.int32(mp_top_level or nz),dt_dyn=np.float32(dt_dyn or 1))
    params.update({name:np.float32(constants[name]) for name in native.CONVERSION_CONSTANTS})
    _launch('hexv842_mpas_thompson_'+phase+'_f32',nz*nc if phase=='from' else nc,context,resources,params)


def launch_precip(phase,fields,surface,diagnostics,*,l_acrain,bucket,corrections=()):
    if phase not in ('from','to'):
        raise ValueError('native Thompson precipitation phase must be from or to')
    from .driver_v842 import correct_source
    correct_source(native._PRECIP,native.CORRECT_MODE_SOURCE,corrections,'thompson_v842')
    nz,nc=fields['qv'].shape
    if not 2<=nz<=native.MAX_LEVELS or nc<1:
        raise ValueError('native Thompson precipitation requires 2 through 256 levels and nonempty columns')
    resources={name:fields[name] for name in ('qv','rho','dz')}
    resources.update({'work_'+name:surface[name] for name in native.SURFACE_ARGUMENTS[:7]})
    resources.update({'diag_'+name:diagnostics[name] for name in native.PRECIP_ARGUMENTS})
    resources['i_rainnc']=diagnostics['i_rainnc'];context=_context(fields['qv'])
    for name,value in tuple(resources.items()):
        shape=(nz,nc) if name in ('qv','rho','dz') else (nc,)
        dtype=np.int32 if name=='i_rainnc' else np.float32
        _require(name,value,dtype,shape)
        if _context(value) is not context:
            raise ValueError('native Thompson precipitation arrays belong to different Metal contexts')
        resources[name]=value.view((value.size,))
    _launch('hexv842_mpas_thompson_precip_f32',nc,context,resources,
            dict(nz=np.int32(nz),columns=np.int32(nc),phase=np.int32(phase=='to'),l_acrain=np.int32(l_acrain),
                 correct_wb3=np.int32('WB3' in corrections),bucket=np.float32(bucket)))
