"""MPAS v8.4.2 shortwave slot adapter for native level-major Metal pools."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch
from .metal_rrtmg_sw_v842 import radiation_controls, run_shortwave
from .route_table import route_row
from .rrtmg_sw_v842 import FIELD_NAMES, SCALAR_OUTPUTS


class MetalMpasShortwave:
    """Immutable initialized tables; physics state and alarms live in native pools."""
    def __init__(self, context, tables, *, config=None):
        self.context=context;self.tables=tables
        descriptor = route_row(config)['slots']['radiation_sw']['cuda_entries']['rrtmg_sw']
        self.eot = None
        self.radius_schemes = ('mp_thompson', 'mp_thompson_aerosols', 'mp_wsm6')
        if descriptor['module'] == 'rrfs.radiation_rrfs':
            from .rrfs.radiation_rrfs import (
                DEFAULT_EOT_DAY_OFFSET, DEFAULT_EOT_SECOND_SINE, RADIUS_SCHEMES)
            arguments = descriptor['resources'].get('arguments', {})
            self.eot = (arguments.get('eot_day_offset', DEFAULT_EOT_DAY_OFFSET),
                        arguments.get('eot_second_sine', DEFAULT_EOT_SECOND_SINE))
            self.radius_schemes = RADIUS_SCHEMES

    def restart_state(self):
        return {}

    def restore_restart_state(self,payload):
        if payload:raise ValueError('MPAS SW has no mutable scheme-private restart state')

    def launch_slot(self,call):
        if (call.slot,call.scheme)!=('radiation_sw','rrtmg_sw'):
            raise ValueError('MPAS Metal shortwave slot requires rrtmg_sw')
        p=call.pools;prepared=p['prepared'];diag=p['diag_physics'];surface=p['sfc_input'];mesh=p['mesh']
        constants=p['constants'];scalar=p['scalar_state'];calendar=p['calendar'];opts=call.options
        nc=int(p['nCellsSolve']);nz=int(p['nVertLevels']);ctx=self.context;xp=_arrays(ctx)
        if nc<1 or nz<3:raise ValueError('MPAS SW needs nonempty columns and at least three model levels')
        degrad=np.float32(constants['degrad']);dpd=np.float32(constants['dpd'])
        if not np.isfinite(degrad) or degrad==0:raise ValueError('MPAS SW needs finite nonzero degrad')
        def transpose(source,target,nlev,*,reverse=False,divide=False):
            _require('SW source',source,np.float32,(nc,nlev) if reverse else (nlev,nc))
            _require('SW target',target,np.float32,(nlev,nc) if reverse else (nc,nlev))
            if _context(source) is not ctx or _context(target) is not ctx:raise ValueError('SW array belongs to another context')
            _launch('hexv842_sw_layout_f32',nc*nlev,ctx,dict(src=source,dst=target),dict(nc=np.int32(nc),nl=np.int32(nlev),reverse=np.int32(reverse),divide=np.int32(divide),divisor=degrad))
        def column(source,target,*,divide=False):
            _require('SW column source',source,np.float32,(nc,));_require('SW column target',target,np.float32,(nc,))
            if _context(source) is not ctx or _context(target) is not ctx:raise ValueError('SW column belongs to another context')
            _launch('hexv842_sw_layout_f32',nc,ctx,dict(src=source,dst=target),dict(nc=np.int32(nc),nl=np.int32(1),reverse=np.int32(0),divide=np.int32(divide),divisor=degrad))
        fields=xp.zeros((len(FIELD_NAMES),nc,nz),np.float32)
        aliases=('pres_hyd_p','t_p','dz_p','pi_p','qvrad_p','qcrad_p','qirad_p','qsrad_p')
        for j,name in enumerate(aliases):transpose(prepared[name][:nz],fields[j],nz)
        transpose(diag['cldfrac'][:nz],fields[8],nz)
        microp_scheme = opts['config_microp_scheme'].strip()
        diagnosed = opts['config_microp_re'] and microp_scheme in self.radius_schemes
        if diagnosed:
            for j,name in enumerate(('re_cloud','re_ice','re_snow'),9):transpose(diag[name][:nz],fields[j],nz)
        surf=xp.empty((7,nc),np.float32)
        for j,name in enumerate(('latCell','lonCell')):column(mesh[name],surf[j],divide=True)
        for j,value in enumerate((diag['sfc_albedo'],surface['skintemp'],surface['xland'],surface['xice'],surface['snow']),2):column(value,surf[j])
        pw=xp.empty((nc,nz+1),np.float32);tw=xp.empty((nc,nz+1),np.float32)
        transpose(prepared['pres2_hyd_p'][:nz+1],pw,nz+1);transpose(prepared['t2_p'][:nz+1],tw,nz+1)
        ozone_enabled=bool(opts['config_o3climatology'])
        if ozone_enabled:
            pin=p['atm_input']['pin'];nozn=int(pin.size)
            o3=xp.empty((nc,nozn),np.float32);transpose(diag['o3clim'],o3,nozn)
        else:
            # The fixed-profile branch does not read these ABI placeholders.
            pin=xp.zeros((2,),np.float32)
            o3=xp.zeros((nc,2),np.float32)
        controls=radiation_controls(ctx,seconds=call.time_seconds,interval=scalar['dt_radtsw'],julian=calendar['curr_julday'],degrad=degrad,dpd=dpd)
        names={'none':0,'random':1,'maximum_random':2,'maximum':3,'exponential':4,'exponential_random':5}
        overlap=names.get(opts['config_radt_cld_overlap'].strip(),0)
        decor=int(opts['config_radt_cld_dcorrlen'].strip()=='latitude_varying') if overlap in (4,5) else 0
        aerosol=None
        if microp_scheme == 'mp_thompson_aerosols' or (
                self.eot is not None and microp_scheme == 'mp_tempo'
                and opts['config_tempo_aerosolaware']):
            a=xp.empty((3,nc,nz),np.float32)
            for j,name in enumerate(('qv_p','nwfa_p','nifa_p')):transpose(prepared[name][:nz],a[j],nz)
            aerosol=dict(pressure=fields[0],temperature=fields[1],qv=a[0],dz=fields[2],nwfa=a[1],nifa=a[2],xland=surf[4])
        equation_of_time = None if self.eot is None else (calendar['curr_julday'], *self.eot)
        out=run_shortwave(tables=self.tables,fields=fields,surface=surf,pressure_interfaces=pw,temperature_interfaces=tw,pin=pin,ozone=o3,controls=controls,gmt=calendar['gmt'],degrad=degrad,icloud=1,o3input=2 if ozone_enabled else 0,has_reqc=scalar['has_reqc'],has_reqi=scalar['has_reqi'],has_reqs=scalar['has_reqs'],overlap=overlap,decorrelation=decor,julday=calendar['julday'],aerosol_inputs=aerosol,equation_of_time=equation_of_time)
        for j,name in enumerate(SCALAR_OUTPUTS):column(out['scalars'][j],diag[name])
        transpose(out['tendency'],p['tend_physics']['rthratensw'],nz,reverse=True)
        if aerosol is not None:
            column(out['taod5502d'],diag['taod5502d'])
            transpose(out['taod5503d'],diag['taod5503d'],nz,reverse=True)
        return out
