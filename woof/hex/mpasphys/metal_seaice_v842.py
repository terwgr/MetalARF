"""Native MPAS v8.4.2 Noah sea-ice slot on Metal.

This isolated adapter does not register a complete MPAS physics provider.
"""
from __future__ import annotations
import math
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch
from .seaice_v842 import FIELDS, OPTIONAL_FIELDS, _DIAG_FIELDS, _SFC_FIELDS, _SEA_FLUX_FIELDS, options_from_MPAS


class MetalMpasSeaice:
    """Own native module location words and fractional sea-ice correction."""

    corrections = frozenset({'RP3'})

    def __init__(self, context, *, module_locations, i_start, j_index):
        state=np.asarray(module_locations,dtype=np.int32)
        if state.shape != (2,):
            raise ValueError('MPAS sea-ice module locations require two words')
        self.context=context
        self.state=_arrays(context).asarray(state)
        self.i_start=int(i_start)
        self.j_index=int(j_index)

    def restart_state(self):
        return {'module_locations': self.state.to_numpy().copy()}

    def restore_restart_state(self, payload):
        state=np.asarray(payload['module_locations'],dtype=np.int32)
        if state.shape != (2,):
            raise ValueError('MPAS sea-ice module locations require two words')
        self.state[...]=_arrays(self.context).asarray(state)

    def launch_slot(self, call):
        if (call.slot,call.scheme) != ('seaice','sf_noah'):
            raise ValueError('MPAS Metal sea-ice slot currently supports sf_noah')
        opts=options_from_MPAS(call.dt,config_frac_seaice=call.options['config_frac_seaice'])
        if not math.isfinite(opts.dt) or opts.dt<=0:
            raise ValueError('MPAS sea-ice requires a finite positive dt_pbl')
        diag,surface,shared=(call.pools[k] for k in ('diag_physics','sfc_input','seaice_shared'))
        ice=surface['xice']; n=int(ice.size)
        if n<=0: raise ValueError('MPAS sea-ice requires a nonempty solve extent')
        nsoil=int(surface['tslb'].shape[0])
        if not 2<=nsoil<=9: raise ValueError('MPAS Metal sea-ice supports 2 through 9 soil levels')
        microp=call.options['config_microp_scheme']!='off'
        convection=call.options['config_convection_scheme']!='off'
        values={dest:diag[src] for src,dest in _DIAG_FIELDS.items()}
        values.update({dest:surface[src] for src,dest in _SFC_FIELDS.items() if dest!='tslb'})
        values.update({name:diag[name] for name in OPTIONAL_FIELDS})
        if microp: values.update({name:diag[src] for name,src in [('sr_input','sr'),('rainncv','rainncv')]})
        if convection: values['raincv']=diag['raincv']
        missing=[name for name in _SEA_FLUX_FIELDS if name not in shared]
        if missing or not call.options['config_frac_seaice']:
            host_ice=ice.to_numpy()
            if np.any((host_ice>=opts.xice_threshold)&(host_ice<=1.0)):
                raise RuntimeError('MPAS seaice_to_MPAS reads unallocated ocean flux storage at mpas_atmphys_driver_seaice.F:390-398: '+(', '.join(missing) if missing else 'fractional sea ice disabled'))
        rp3 = 'RP3' in getattr(call, 'corrections', frozenset())
        if rp3:
            values['sst'] = surface['sst']
        values.update({name:shared[name] for name in _SEA_FLUX_FIELDS if name in shared})
        for name,value in values.items():
            _require(name,value,np.float32,(n,))
            if _context(value) is not self.context: raise ValueError(f'{name} belongs to another context')
        soil=surface['tslb']
        _require('tslb',soil,np.float32,(nsoil,n))
        for name in ('t3d','qv3d','p8w3d','dz8w'):
            value=shared[name]
            if len(value.shape)!=2 or value.shape[0]<(2 if name=='p8w3d' else 1):
                raise ValueError(f'MPAS sea-ice profile {name} has too few levels')
            _require(name,value,np.float32,(value.shape[0],n))
            if _context(value) is not self.context: raise ValueError(f'{name} belongs to another context')
            values[name]=value[0]
        if _context(soil) is not self.context: raise ValueError('tslb belongs to another context')
        xp=_arrays(self.context)
        if 4*n*(len(FIELDS)+nsoil+5)>self.context.recommended_working_set_size:
            raise MemoryError('MPAS sea-ice workspace exceeds Metal recommended working set')
        fields=xp.zeros((len(FIELDS),n),np.float32)
        for name,value in values.items():
            if name != 'sst': fields[FIELDS.index(name)][...]=value
        soil_work=soil.copy(); p8w2=shared['p8w3d'][1].copy()
        errors=xp.zeros((n,),np.int32); ran=xp.zeros((n,),np.int32)
        state=self.state.copy()
        params={k:np.int32(v) for k,v in dict(n=n,nsoil=nsoil,albedo_opt=opts.seaice_albedo_opt,thickness_opt=opts.seaice_thickness_opt,snowdepth_opt=opts.seaice_snowdepth_opt,frpcpn=opts.frpcpn,optional_noah=7,microp=microp,convection=convection).items()}
        params.update({k:np.float32(v) for k,v in dict(dt=opts.dt,threshold=opts.xice_threshold,albedo_default=opts.seaice_albedo_default,thickness_default=opts.seaice_thickness_default,snowdepth_max=opts.seaice_snowdepth_max,snowdepth_min=opts.seaice_snowdepth_min).items()})
        _launch('hexv842_seaice_from_f32',n,self.context,dict(fields=fields),params)
        if rp3:
            _launch('hexv842_mpas_seaice_rp3_f32', n, self.context,
                dict(fields=fields, sst=values['sst'], skin=surface['skintemp'], ice=ice),
                dict(n=np.int32(n), sea_index=np.int32(FIELDS.index('tsk_sea')),
                     ice_index=np.int32(FIELDS.index('tsk')), threshold=np.float32(opts.xice_threshold)))
        _launch('hexv842_seaice_column_f32',n,self.context,dict(fields=fields,soil=soil_work,p8w2=p8w2,errors=errors,ran=ran),params)
        _launch('hexv842_seaice_state_i32',1,self.context,dict(ran=ran,state=state),dict(n=np.int32(n),i_start=np.int32(self.i_start),j_index=np.int32(self.j_index)))
        failures=errors.to_numpy(); bad=np.flatnonzero(failures)
        if bad.size:
            i=int(bad[0]); raise RuntimeError(f'MPAS sea-ice fatal status {int(failures[i])} at column {self.i_start+i}, j={self.j_index}')
        _launch('hexv842_seaice_to_f32',n,self.context,dict(fields=fields),params)
        for name in ('acsnom','acsnow','chs','chs2','cpm','cqs2','qsfc','qgh','hfx','qfx','lh','sfc_albedo','sfc_emiss','sfcrunoff','z0','znt','grdflx'):
            diag[name][...]=fields[FIELDS.index(_DIAG_FIELDS[name])]
        for name in ('snow','snowc','snowh','skintemp'):
            surface[name][...]=fields[FIELDS.index(_SFC_FIELDS[name])]
        surface['tslb'][...]=soil_work
        for name in OPTIONAL_FIELDS: diag[name][...]=fields[FIELDS.index(name)]
        self.state[...]=state
        return fields
