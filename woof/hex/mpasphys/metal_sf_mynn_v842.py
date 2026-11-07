"""Native MPAS v8.4.2 MYNN surface-layer slot on resident Metal pools."""
from __future__ import annotations
from collections import ChainMap
import math
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch
from .sf_mynn_v842 import CONSTANTS, FIELDS, OUTPUTS, _DIAG_INPUT, _DIAG_NAMES, _PROFILE, _SEA_HANDOFF, _SURFACE


class MetalMpasSfMynn:
    """One session's deterministic stability tables; physical state stays in pools."""

    def __init__(self, context):
        self.context=context
        self.tables=_arrays(context).empty((4004,),np.float32)
        _launch('hexv842_mpas_sf_mynn_tables_f32',1001,context,
                dict(tables=self.tables),dict(n=np.int32(1001)))

    def launch_slot(self, call):
        if (call.slot,call.scheme)!=('surface_layer','sf_mynn'):
            raise ValueError('MPAS Metal surface slot requires sf_mynn')
        pools=call.pools
        mesh,diag,surface,profiles,constants,configs,state=(pools[name] for name in
            ('mesh','diag_physics','sfc_input','atm_input','constants','configs','scalar_state'))
        surface=ChainMap({'psfc':pools['prepared']['psfc_p']},surface)
        n=int(surface['xland'].size)
        if n<=0: raise ValueError('MPAS sf_mynn requires a nonempty solve extent')
        frac=bool(call.options['config_frac_seaice'])
        len_disp=float(configs['config_len_disp']);threshold=float(state['xice_threshold'])
        if not math.isfinite(len_disp) or len_disp<=0 or not math.isfinite(threshold):
            raise ValueError('MPAS sf_mynn needs finite positive length and finite ice threshold')
        required={f'diag.{name}':diag[name] for name in _DIAG_INPUT}
        required['diag.hpbl']=diag['hpbl']
        required.update({f'surface.{name}':surface[name] for name in _SURFACE.values()})
        required['mesh.meshDensity']=mesh['meshDensity']
        for name in OUTPUTS:
            if name not in ('wstar','qstar'):
                target=_DIAG_NAMES.get(name,name)
                required[f'output.{target}']=diag[target]
        if frac:
            required.update({f'surface.{name}':surface[name] for name in ('sst','xice')})
            required['diag.regime']=diag['regime']
            required.update({f'prepared.{name}_sea':pools['prepared'][name+'_sea'] for name in _SEA_HANDOFF})
        for name,value in required.items():
            _require(name,value,np.float32,(n,))
            if _context(value) is not self.context: raise ValueError(f'{name} belongs to another context')
        for name in _PROFILE:
            value=profiles[name]
            if len(value.shape)!=2 or value.shape[0]<2:
                raise ValueError(f'MPAS sf_mynn profile {name} requires at least two levels')
            _require(name,value,np.float32,(value.shape[0],n))
            if _context(value) is not self.context: raise ValueError(f'{name} belongs to another context')
        scalar=np.array([constants[name] for name in CONSTANTS],np.float32)
        if not np.all(np.isfinite(scalar)): raise ValueError('MPAS sf_mynn constants must be finite')
        xp=_arrays(self.context)
        packed=xp.zeros((len(FIELDS),n),np.float32)
        for name in _DIAG_INPUT: packed[FIELDS.index(name)][...]=diag[name]
        packed[FIELDS.index('pblh')][...]=diag['hpbl']
        for name,source in _SURFACE.items(): packed[FIELDS.index(name)][...]=surface[source]
        lower=xp.empty((len(_PROFILE),2,n),np.float32)
        for j,name in enumerate(_PROFILE): lower[j][...]=profiles[name][:2]
        params=dict(n=np.int32(n),first=np.int32(not configs['config_do_restart'] and call.itimestep<=1),len_disp=np.float32(len_disp))
        _launch('hexv842_mpas_sf_mynn_prepare_f32',n,self.context,
                dict(fields=packed,profiles=lower,density=mesh['meshDensity']),params)
        sea=None; hold=None
        if frac:
            sea=xp.empty((len(FIELDS),n),np.float32);hold=diag['regime'].copy()
            _launch('hexv842_mpas_sf_mynn_sea_f32',n,self.context,
                    dict(fields=packed,sea=sea,sst=surface['sst'],ice=surface['xice']),
                    dict(n=np.int32(n),first=params['first'],threshold=np.float32(threshold),
                         rp3=np.int32('RP3' in getattr(call,'corrections',frozenset()))))
        c=xp.asarray(scalar)
        column_params=dict(n=np.int32(n),itimestep=params['first'],isfflx=np.int32(1),spp_pbl=np.int32(0),isftcflx=np.int32(0),iz0tlnd=np.int32(0),present=np.int32(127))
        _launch('hexv842_mpas_sf_mynn_columns_f32',n,self.context,
                dict(fields=packed,constants=c,tables=self.tables),column_params)
        if frac:
            _launch('hexv842_mpas_sf_mynn_columns_f32',n,self.context,
                    dict(fields=sea,constants=c,tables=self.tables),column_params)
            for name in _SEA_HANDOFF: pools['prepared'][name+'_sea'][...]=sea[FIELDS.index(name)]
            _launch('hexv842_mpas_sf_mynn_blend_f32',n,self.context,
                    dict(fields=packed,sea=sea,ice=surface['xice'],regime_hold=hold),
                    dict(n=np.int32(n),threshold=np.float32(threshold)))
        for name in OUTPUTS:
            if name not in ('wstar','qstar'):
                diag[_DIAG_NAMES.get(name,name)][...]=packed[FIELDS.index(name)]
        return diag
