"""Native MPAS Thompson column components for Metal."""
import numpy as np
from ..metal_dynamics_v841 import _context, _require
from .metal_tendencies_v842 import _launch


def saturation(pressure, temperature, liquid, ice):
    """Run the native column's liquid and ice saturation helpers."""
    if pressure.ndim != 1 or pressure.size < 1:
        raise ValueError('Thompson saturation requires nonempty flat arrays')
    n=pressure.size;context=_context(pressure)
    for name,value in (('pressure',pressure),('temperature',temperature),('liquid',liquid),('ice',ice)):
        _require('Thompson '+name,value,np.float32,(n,))
        if _context(value) is not context:raise ValueError('Thompson arrays must share a Metal context')
    _launch('hexv842_mpas_thompson_column_saturation_f32',n,context,
            dict(pressure=pressure,temperature=temperature,liquid=liquid,ice=ice),dict(n=np.int32(n)))


def step(fields, surface, tables, *, dt, has_reqc=1, has_reqi=1, has_reqs=1, column_chunk=1, aerosol_aware=False):
    """Run the native Thompson column process on validated resident arrays.

    Workspace sizing bounds each submission against the recommended working
    set. The caller manages other resident allocations and outstanding work.
    """
    from . import thompson_v842 as native
    from ..metal_dynamics_v841 import _arrays
    if tables.get('_file_sha256') != native.TABLE_FILE_SHA256:
        raise ValueError('native Thompson requires certified table identities')
    if aerosol_aware:
        from . import thompson_aero_v842 as aerosol
        if tables.get('_aerosol_auxiliary_sha256') != aerosol.AUXILIARY_SHA256:
            raise ValueError('native aerosol Thompson requires the certified aerosol auxiliary')
        field_names = (*native.FIELD_ARGUMENTS, *aerosol.AEROSOL_FIELDS)
        surface_names = (*native.SURFACE_ARGUMENTS, 'nifa2d', 'nwfa2d')
        kernel = 'hexv842_mpas_thompson_aerosol_column_step_f32'
    else:
        field_names, surface_names = native.FIELD_ARGUMENTS, native.SURFACE_ARGUMENTS
        kernel = 'hexv842_mpas_thompson_column_step_f32'
    qv=fields['qv']
    if qv.ndim!=2 or not 1<=qv.shape[0]<=native.MAX_LEVELS or qv.shape[1]<1:
        raise ValueError('native Thompson requires 1..256 levels and nonempty columns')
    nz,nc=map(int,qv.shape);context=_context(qv)
    inputs={name:fields[name] for name in field_names}
    inputs.update({name:surface[name] for name in surface_names})
    resident_bytes=0
    for name,value in inputs.items():
        shape=(nz,nc) if name in field_names else (nc,)
        _require('Thompson '+name,value,np.float32,shape)
        if _context(value) is not context:raise ValueError('Thompson fields must share a Metal context')
        resident_bytes+=value.size*value.dtype.itemsize
    if aerosol_aware:
        # The five views must cover consecutive regions of one binding. Offsets
        # are expressed in words in the shader, using ulong multiplication.
        packed_names=('nc','nifa','nwfa','nifa2d','nwfa2d')
        packed=[inputs.pop(name) for name in packed_names]
        start=packed[0].byte_offset
        offset=start
        for name,value in zip(packed_names,packed):
            if value.buffer is not packed[0].buffer or value.byte_offset!=offset:
                raise ValueError('aerosol views must be consecutive in one buffer: '+name)
            offset+=value.nbytes
        inputs['aerosol']=packed[0].view(((offset-start)//4,),byte_offset=start)
    for name,shape in native.TABLE_SHAPES.items():
        value=tables[name];dtype=np.float32 if name=='tnccn_act' else np.uint64
        _require('Thompson table '+name,value,dtype,(int(np.prod(shape)),))
        if _context(value) is not context:raise ValueError('Thompson tables must share the Metal context')
        inputs['input_'+name]=value;resident_bytes+=value.size*value.dtype.itemsize
    if not np.isfinite(dt) or dt<=0:raise ValueError('Thompson requires positive finite dt')
    chunk=int(column_chunk)
    if chunk<1:raise ValueError('Thompson column chunk must be positive')
    per_column=(nz+1)*(93*4+64*8+7)
    available=context.recommended_working_set_size-resident_bytes
    chunk=min(chunk,nc,(available-3)//per_column)
    if chunk<1:raise MemoryError('native Thompson column workspace exceeds Metal recommended working set')
    xp=_arrays(context)
    ws_f=xp.empty((chunk*93*(nz+1),),np.float32)
    ws_d=xp.empty((chunk*64*(nz+1),),np.uint64)
    ws_b=xp.empty(((chunk*7*(nz+1)+3)//4,),np.uint32)
    for first in range(0,nc,chunk):
        count=min(chunk,nc-first)
        _launch(kernel,count,context,
                dict(inputs,ws_f=ws_f,ws_d=ws_d,ws_b=ws_b),
                dict(nz=np.int32(nz),columns=np.int32(nc),dt=np.float32(dt),
                     has_reqc=np.int32(has_reqc),has_reqi=np.int32(has_reqi),has_reqs=np.int32(has_reqs),
                     first=np.int32(first),count=np.int32(count)))


def relative_humidity(qv, th, pii, pressure, relhum):
    """Compute the native Thompson relative humidity diagnostic."""
    if qv.ndim != 1 or qv.size < 1:
        raise ValueError('Thompson relative humidity requires nonempty flat arrays')
    n=qv.size;context=_context(qv)
    for name,value in (('qv',qv),('th',th),('pii',pii),('pressure',pressure),('relhum',relhum)):
        _require('Thompson '+name,value,np.float32,(n,))
        if _context(value) is not context:raise ValueError('Thompson arrays must share a Metal context')
    _launch('hexv842_mpas_thompson_column_relhum_f32',n,context,
            dict(qv=qv,th=th,pii=pii,pressure=pressure,relhum=relhum),dict(n=np.int32(n)))


def launch_radar(fields, diagnostics):
    """Run MPAS Thompson's radar operator on the supplied column state."""
    from ..metal_dynamics_v841 import _arrays
    nz, nc = fields['qv'].shape
    if not 2 <= nz <= 256 or nc < 1:
        raise ValueError('native Thompson radar requires 2..256 levels and nonempty columns')
    context = _context(fields['qv'])
    names = ('qv', 'qc', 'qr', 'nr', 'qs', 'qg', 'th', 'pii', 'p', 'z', 'dz')
    resources = {name: fields[name] for name in names}
    resources.update(dict(zip(('refl', 'refl_max', 'refl_1km', 'refl_1km_max'),
        (diagnostics[name] for name in ('refl10cm', 'refl10cm_max', 'refl10cm_1km', 'refl10cm_1km_max')))))
    for name, value in tuple(resources.items()):
        shape = (nc,) if name in ('refl_max', 'refl_1km', 'refl_1km_max') else (nz, nc)
        _require(name, value, np.float32, shape)
        if _context(value) is not context:
            raise ValueError('native radar arrays must share a Metal context')
        resources[name] = value.view((value.size,))
    available = context.recommended_working_set_size - sum(v.nbytes for v in resources.values())
    chunk = min(nc, 32, (available - 3) // ((nz + 1) * (28 * 4 + 4 * 8 + 3)))
    if chunk < 1:
        raise MemoryError('native Thompson radar workspace exceeds the Metal working set')
    xp = _arrays(context)
    resources.update(ws_f=xp.empty((chunk * 28 * (nz + 1),), np.float32),
                     ws_d=xp.empty((chunk * 4 * (nz + 1),), np.uint64),
                     ws_b=xp.empty(((chunk * 3 * (nz + 1) + 3) // 4,), np.uint32))
    for first in range(0, nc, chunk):
        count = min(chunk, nc - first)
        _launch('hexv842_mpas_thompson_column_radar_f32', count, context, resources,
                dict(nz=np.int32(nz), columns=np.int32(nc), first=np.int32(first), count=np.int32(count)))


def launch_slot(call):
    """Run the native Thompson slot and its requested radar diagnostics.

    The caller supplies the native prepared pools and manager substep values.
    Nest-look remains unavailable and is refused before writes.
    """
    from . import thompson_v842 as native
    from .driver_v842 import slot_corrections
    from .metal_thompson_v842 import launch_conversion, launch_precip
    if (call.slot,call.scheme)!=('microphysics',native.SCHEME):
        raise ValueError('MPAS Metal Thompson requires its microphysics slot')
    p=call.pools['thompson_v842']
    if p.get('nest_look') is not None:
        raise NotImplementedError('native Thompson nest look is not yet implemented on Metal')
    if type(p['n_microp']) is not int or p['n_microp']<1:
        raise ValueError('native manager must supply a positive integer microphysics substep count')
    fields=p['fields']
    conversion=(p['model'],p['mesh'],p['diagnostics'],p['tendencies'],fields,p['constants'])
    launch_precip('from',fields,p['surface'],p['precip'],l_acrain=p['l_acrain'],bucket=call.options['config_bucket_rainnc'])
    launch_conversion('from',*conversion)
    for _ in range(p['n_microp']):
        step(fields,p['surface'],p['tables'],dt=p['dt_microp'],**{name:p[name] for name in ('has_reqc','has_reqi','has_reqs')})
    if call.flags.diagnostics:
        launch_radar(fields, p['diagnostics'])
        n = fields['qv'].size
        relative_humidity(*(fields[name].view((n,)) for name in ('qv', 'th', 'pii', 'p')),
                          p['diagnostics']['relhum'].view((n,)))
    launch_precip('to',fields,p['surface'],p['precip'],l_acrain=p['l_acrain'],
                  bucket=call.options['config_bucket_rainnc'],corrections=slot_corrections(call,native.CORRECT_MODE_SOURCE))
    launch_conversion('to',*conversion,mp_top_level=p['mp_top_level'],dt_dyn=call.dt)


def bind_session(session, tables, *, diagnostics_requested=False):
    """Bind native solve-cell Registry and prepared views for a Thompson call."""
    from . import thompson_v842 as native
    pools=session.slot_pools; nz=session.layout.dimensions['nVertLevels']
    state=session.pools.field('state','scalars',level=2)
    model={}
    for name in native.MODEL_ARGUMENTS[:8]:
        index=session.layout.indices['state','index_'+name]
        if not 1<=index<=state.shape[0]:raise ValueError('invalid native scalar index '+name)
        model[name]=state[index-1]
    model.update({name:session.pools.field('state',name,level=2) for name in native.MODEL_ARGUMENTS[8:]})
    mesh={name:pools['mesh'][name] for name in native.MESH_ARGUMENTS}
    diagnostics={name:pools['diag'][name] for name in native.DIAG_ARGUMENTS if name!='rt_diabatic_tend'}
    diagnostics['rt_diabatic_tend']=pools['tend']['rt_diabatic_tend']
    diagnostics['surface_pressure']=pools['diag']['surface_pressure']
    diagnostics['tend_sfc_pressure']=pools['tend']['tend_sfc_pressure']
    if diagnostics_requested:
        diagnostics['relhum'] = pools['diag']['relhum']
        diagnostics.update({name: pools['diag_physics'][name] for name in
                            ('refl10cm', 'refl10cm_max', 'refl10cm_1km', 'refl10cm_1km_max')})
    tendencies={name:pools['tend_physics'][name] for name in native.TENDENCY_ARGUMENTS}
    constants={name:pools['constants'][name] for name in native.CONVERSION_CONSTANTS}
    prepared=pools['prepared'];diag=pools['diag_physics']
    aliases={'pii':'pi_p','p':'pres_p'}; diagnostic=('rainprod','evapprod','re_cloud','re_ice','re_snow')
    fields={name:prepared[aliases.get(name,name+'_p')][:nz]
            for name in (*native.FIELD_ARGUMENTS,'rho','z') if name not in diagnostic}
    fields.update({name:diag[name] for name in diagnostic})
    surface={name:prepared[name+'_p'] for name in native.SURFACE_ARGUMENTS[:7]}
    surface.update(ntc=diag['nt_c'],muc=diag['mu_c'])
    precip={name:diag[name] for name in (*native.PRECIP_ARGUMENTS,'i_rainnc')}
    scalar=pools['scalar_state']
    top=pools['tend_physics']['mp_top_level']
    return dict(model=model,mesh=mesh,diagnostics=diagnostics,tendencies=tendencies,constants=constants,
                fields=fields,surface=surface,precip=precip,tables=tables,
                dt_microp=scalar['dt_microp'],n_microp=int(scalar['n_microp']),
                mp_top_level=int(top.to_numpy().reshape(-1)[0]),l_acrain=bool(scalar['l_acrain']),
                nest_look=getattr(session,'nest_look',None),
                **{name:bool(scalar[name]) for name in ('has_reqc','has_reqi','has_reqs')})


class MetalMpasThompson:
    """Session-owned immutable tables; physical restart state stays in pools."""
    def __init__(self,tables):
        from .thompson_v842 import CORRECT_MODE_SOURCE
        self.tables=tables
        self.corrections=frozenset(CORRECT_MODE_SOURCE)

    def bind_pools(self,pools,call):
        return {'thompson_v842':bind_session(pools.session,self.tables, diagnostics_requested=call.flags.diagnostics)}

    launch_slot=staticmethod(launch_slot)

    def restart_state(self):
        return {}

    def restore_restart_state(self,state):
        if state:raise ValueError('native Thompson has no scheme-private restart state')
