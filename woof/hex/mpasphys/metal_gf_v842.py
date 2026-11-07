"""Native MPAS v8.4.2 Grell–Freitas Metal building blocks."""
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch
from . import gf_mpas_v842 as native


def pack_inputs(fields, *, len_disp, dt):
    """Gather native solve-cell profiles and surface inputs into GF column order."""
    u=fields['u']
    if u.ndim!=2 or min(u.shape)<1:raise ValueError('GF requires nonempty level, column profiles')
    nz,n=map(int,u.shape);context=_context(u)
    arrays={name:fields[name] for name in native._PACK_LEVEL_SOURCES}
    arrays['ht']=fields['zgrid'][0]
    arrays.update({name:fields[name] for name in native._PACK_VECTOR_SOURCES[1:]})
    for name,value in arrays.items():
        shape=(nz,n) if name in native._PACK_LEVEL_SOURCES else (n,)
        _require('GF '+name,value,np.float32,shape)
        if _context(value) is not context:raise ValueError('GF arrays must share a Metal context')
    if not np.isfinite(len_disp) or not np.isfinite(dt):raise ValueError('GF scalars must be finite')
    xp=_arrays(context);levels=xp.empty((n,15,nz),np.float32);surface=xp.empty((n,8),np.float32)
    _launch('hexv842_mpas_gf_pack_f32',n*nz,context,dict(arrays,levels=levels,surface=surface),
            dict(n=np.int32(n),nz=np.int32(nz),len_disp=np.float32(len_disp),dt=np.float32(dt)))
    return levels,surface


def saturation_vapour(temperature, output):
    """Run native GF saturation vapour for source-valid positive temperatures.

    The caller must supply positive temperatures; this is not checked on host.
    """
    if temperature.ndim!=1 or temperature.size<1:raise ValueError('GF saturation requires nonempty flat arrays')
    n=temperature.size;context=_context(temperature)
    for name,value in (('temperature',temperature),('output',output)):
        _require('GF '+name,value,np.float32,(n,))
        if _context(value) is not context:raise ValueError('GF arrays must share a Metal context')
    _launch('hexv842_mpas_gf_satvap_f32',n,context,dict(temperature=temperature,output=output),dict(n=np.int32(n)))


def scatter_outputs(levels, surface, integer, tendencies, diagnostics):
    """Publish native GF results to solve-cell tendencies and diagnostics."""
    if levels.ndim!=3 or levels.shape[1]!=6 or min(levels.shape)<1:
        raise ValueError('GF levels require nonempty [column,6,level] storage')
    n,_,nz=map(int,levels.shape);context=_context(levels)
    arrays=dict(levels=levels,surface=surface,integer=integer)
    arrays.update({name:(tendencies if name in native.LEVEL_OUTPUTS[:4] else diagnostics)[name] for name in native._SCATTER_LEVEL_NAMES})
    arrays.update({name:diagnostics[name] for name in native._SCATTER_SURFACE_NAMES+native.INTEGER_OUTPUTS})
    for name,value in arrays.items():
        shape=(n,6,nz) if name=='levels' else (n,6) if name=='surface' else (n,3) if name=='integer' else (nz,n) if name in native._SCATTER_LEVEL_NAMES else (n,)
        dtype=np.int32 if name=='integer' or name in native.INTEGER_OUTPUTS else np.float32
        _require('GF '+name,value,dtype,shape)
        if _context(value) is not context:raise ValueError('GF arrays must share a Metal context')
    _launch('hexv842_mpas_gf_scatter_f32',n*nz,context,arrays,dict(n=np.int32(n),nz=np.int32(nz)))


def run_columns(levels, surface, kpbl, *, itimestep, constants):
    """Execute the native default closures with a source-sized arena."""
    if levels.ndim!=3 or levels.shape[1]!=15:raise ValueError('GF levels require [column,15,level]')
    n,_,nz=map(int,levels.shape)
    if n<1 or nz<16:raise ValueError('GF requires nonempty columns and at least 16 levels')
    context=_context(levels)
    for name,value,dtype,shape in (('levels',levels,np.float32,(n,15,nz)),('surface',surface,np.float32,(n,8)),('kpbl',kpbl,np.int32,(n,))):
        _require('GF '+name,value,dtype,shape)
        if _context(value) is not context:raise ValueError('GF arrays must share a Metal context')
    values=np.asarray(constants,dtype=np.float32)
    if values.shape!=(4,) or not np.all(np.isfinite(values)):raise ValueError('GF requires four finite native constants')
    size=native._scratch_bytes(nz)
    if size>context.max_buffer_length or size>context.recommended_working_set_size//8:raise ValueError('GF source scratch exceeds device workspace budget')
    xp=_arrays(context)
    arrays=dict(levels=levels,surface=surface,kpbl=kpbl,outlev=xp.zeros((n,6,nz),np.float32),outsurface=xp.zeros((n,6),np.float32),outint=xp.zeros((n,3),np.int32),scratch=xp.empty((size//4,),np.uint32),highwater=xp.empty((n,),np.uint32))
    params=dict(n=np.int32(n),nz=np.int32(nz),scratch_words=np.int32(size//4),itimestep=np.int32(itimestep),deep=np.int32(0),shallow=np.int32(8),xlv=values[0],cp=values[1],gravity=values[2],rv=values[3])
    for col in range(n):
        _launch('hexv842_mpas_gf_column_f32',1,context,arrays,dict(params,column=np.int32(col)))
    return arrays['outlev'],arrays['outsurface'],arrays['outint']


def bind_session(provider, call):
    """Bind native solve-cell fields and host manager constants."""
    return native.bind_native_pools(provider,call)


def launch_slot(call):
    """Run GF default closures on native prepared solve-cell pools."""
    if (call.slot,call.scheme)!=('convection',native.SCHEME):raise ValueError('GF requires its native convection slot')
    if (call.options['config_gfconv_closure_deep'],call.options['config_gfconv_closure_shallow'])!=(0,8):raise NotImplementedError('GF Metal currently covers default closure selectors')
    p=call.pools['gf_v842'];n=int(p['nCellsSolve'])
    if n<1:raise ValueError('GF requires at least one solve cell')
    prepared=p['prepared'];mesh=p['mesh'];diag=p['diag_physics'];tend=p['tend_physics'];surface=p['sfc_input']
    nz=tend['rthcuten'].shape[0]
    fields={name:prepared[source][:nz] for name,source in native._PREPARED_FIELDS.items()}
    fields.update({name:tend[name] for name in ('rthdynten','rqvdynten','rthblten','rqvblten','rthratenlw','rthratensw')})
    fields.update({name:diag[name] for name in ('hfx','qfx','gsw','kpbl')})
    fields.update({name:mesh[name] for name in ('meshDensity','areaCell','zgrid')})
    fields['xland']=surface['xland']
    _require('GF active flag',p['cu_act_flag'],np.int32,(n,))
    if _context(p['cu_act_flag']) is not _context(fields['u']):raise ValueError('GF arrays must share a Metal context')
    levels,packed=pack_inputs(fields,len_disp=p['len_disp'],dt=p['dt_dyn'])
    outlev,outsurface,outint=run_columns(levels,packed,fields['kpbl'],itimestep=call.itimestep,constants=p['gf_constants'])
    scatter_outputs(outlev,outsurface,outint,tend,diag)
    p['cu_act_flag'].fill(0)


def update_convection_step1(call):
    """The native first-step update has no Grell–Freitas branch."""
    if call.options['config_convection_scheme'] != native.SCHEME:
        raise ValueError('GF convection hook requires its native scheme')


class MetalMpasGrellFreitas:
    """MPAS convection initialization has no GF branch; held fields live in pools."""
    def bind_pools(self, pools, call):
        return {'gf_v842':bind_session(pools,call)}

    launch_slot=staticmethod(launch_slot)

    def callbacks(self):
        return {'update_convection_step1':update_convection_step1}

    def restart_state(self):
        return {}

    def restore_restart_state(self, state):
        if state:raise ValueError('native GF has no scheme-private restart state')
