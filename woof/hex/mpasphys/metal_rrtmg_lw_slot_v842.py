"""Native MPAS longwave slot adapter for level-major Metal pools."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch
from .metal_rrtmg_lw_v842 import run_group, _solver_workspace_words
from .route_table import route_row


class MetalMpasLongwave:
    """Immutable coefficient tables; held radiation fields live in pools."""
    def __init__(self,context,tables,*,config=None):
        self.context=context;self.tables=tables
        self.radius_schemes=('mp_thompson','mp_thompson_aerosols','mp_wsm6')
        descriptor=route_row(config)['slots']['radiation_lw']['cuda_entries']['rrtmg_lw']
        if descriptor['module']=='rrfs.radiation_rrfs':
            from .rrfs.radiation_rrfs import RADIUS_SCHEMES
            self.radius_schemes=RADIUS_SCHEMES

    def restart_state(self):
        return {}

    def restore_restart_state(self,payload):
        if payload:raise ValueError('MPAS LW has no mutable scheme-private restart state')

    def launch_slot(self,call):
        if (call.slot,call.scheme)!=('radiation_lw','rrtmg_lw'):
            raise ValueError('MPAS Metal longwave slot requires rrtmg_lw')
        p=call.pools;prep=p['prepared'];diag=p['diag_physics'];surface=p['sfc_input']
        scalar=p['scalar_state'];calendar=p['calendar'];opts=call.options
        nc=int(p['nCellsSolve']);nm=int(p['nVertLevels']);ctx=self.context;xp=_arrays(ctx)
        if nc<1 or nm<1:raise ValueError('MPAS LW needs nonempty columns and layers')
        scheme=str(opts['config_microp_scheme']).strip();capable=scheme in self.radius_schemes
        radii=bool(opts['config_microp_re']) and capable
        inputs={'p3d':prep['pres_hyd_p'][:nm],'t3d':prep['t_p'][:nm],
                'dz':prep['dz_p'][:nm],'pi':prep['pi_p'][:nm],
                'qv':prep['qvrad_p'][:nm],'qc':prep['qcrad_p'][:nm],
                'qi':prep['qirad_p'][:nm],'qs':prep['qsrad_p'][:nm],
                'cldfra':diag['cldfrac'][:nm],'p8w':prep['pres2_hyd_p'][:nm+1],
                't8w':prep['t2_p'][:nm+1],'tsk':surface['skintemp'],
                'emiss':diag['sfc_emiss'],'xland':surface['xland'],
                'xice':surface['xice'],'snow':surface['snow'],'xlat':prep['xlat_p']}
        inputs.update({name:(diag[name][:nm] if radii else inputs['cldfra']) for name in ('re_cloud','re_ice','re_snow')})
        ozone_enabled=bool(opts['config_o3climatology'])
        pin=p['atm_input']['pin'] if ozone_enabled else xp.zeros((2,),np.float32)
        if pin.ndim!=1 or (ozone_enabled and pin.size<2):raise ValueError('MPAS LW ozone interpolation needs two defined source levels')
        nz=int(pin.size)
        inputs['o3clim']=diag['o3clim'] if ozone_enabled else xp.zeros((nz,nc),np.float32)
        _require('LW ozone pressure',pin,np.float32,(nz,))
        if _context(pin) is not ctx:raise ValueError('MPAS LW ozone pressure belongs to another Metal context')
        for name,value in inputs.items():
            shape=(nc,) if name in ('tsk','emiss','xland','xice','snow','xlat') else (nm+1,nc) if name in ('p8w','t8w') else (nz,nc) if name=='o3clim' else (nm,nc)
            _require('LW '+name,value,np.float32,shape)
            if _context(value) is not ctx:raise ValueError('MPAS LW input belongs to another Metal context')
        names=('glw','olr','lwcf','lwupt','lwuptc','lwdnt','lwdntc','lwupb','lwupbc','lwdnb','lwdnbc')
        dest={name:diag['olrtoa' if name=='olr' else name] for name in names}
        dest['tend_out']=p['tend_physics']['rthratenlw']
        for name in ('re_cloud','re_ice','re_snow'):
            dest[name+'_out']=diag['r'+name] if capable else dest['tend_out']
        for name,value in dest.items():
            _require('LW '+name,value,np.float32,(nm,nc) if name.endswith('_out') else (nc,))
            if _context(value) is not ctx:raise ValueError('MPAS LW output belongs to another Metal context')
        if ozone_enabled:
            ozone_out=diag['o3vmr'];_require('LW ozone diagnostic',ozone_out,np.float32,(nm,nc))
            if _context(ozone_out) is not ctx:raise ValueError('MPAS LW ozone diagnostic belongs to another Metal context')
            upper=xp.zeros((nc,),np.int32);start=xp.empty((1,),np.int32)
            for k in range(nm-1,-1,-1):
                _launch('hexv842_mpas_rrtmg_lw_ozstart_f32',1,ctx,dict(upper=upper,start=start),dict(nc=np.int32(nc),nz=np.int32(nz)))
                _launch('hexv842_mpas_rrtmg_lw_ozlevel_f32',nc,ctx,
                        dict(p3d=inputs['p3d'],pin=pin,ozone=inputs['o3clim'],start=start,upper=upper,o3vmr=ozone_out),
                        dict(nc=np.int32(nc),nz=np.int32(nz),k=np.int32(k)))
        counts=xp.empty((nc,),np.int32)
        _launch('hexv842_mpas_rrtmg_lw_slotcounts_f32',nc,ctx,dict(p8w=inputs['p8w'],counts=counts),dict(nc=np.int32(nc),nm=np.int32(nm)))
        host_counts=counts.to_numpy()
        if np.any(host_counts<4) or np.any(host_counts>128):
            raise ValueError('MPAS LW needs four to 128 defined radiation layers')
        overlap={'none':0,'random':1,'maximum_random':2,'maximum':3,'exponential':4,'exponential_random':5}.get(str(opts['config_radt_cld_overlap']).strip(),0)
        decor=int(overlap>=4 and str(opts['config_radt_cld_dcorrlen']).strip()=='latitude_varying')
        groups=[(int(depth),np.flatnonzero(host_counts==depth).astype(np.int32)) for depth in np.unique(host_counts)]
        for depth,host_indices in groups:
            ng=int(host_indices.size)
            words=_solver_workspace_words(ng,depth)+ng*(12*nm+2*(nm+1)+6+nz+12*depth+2*(depth+1)+depth+4*nm+14)+nc
            if words*4>ctx.recommended_working_set_size:
                raise MemoryError('MPAS LW slot group exceeds Metal recommended working set')
        for depth,host_indices in groups:
            indices=xp.asarray(host_indices);ng=int(indices.size)
            model=xp.empty((ng,12,nm),np.float32);interfaces=xp.empty((ng,2,nm+1),np.float32)
            grouped_surface=xp.empty((ng,6),np.float32);ozone=xp.empty((ng,nz),np.float32)
            _launch('hexv842_mpas_rrtmg_lw_slotpack_f32',ng*nm,ctx,
                    dict(indices=indices,**inputs,model=model,interfaces=interfaces,surface=grouped_surface,ozone=ozone),
                    dict(nc=np.int32(nc),ng=np.int32(ng),nm=np.int32(nm),nz=np.int32(nz),radii=np.int32(radii)))
            result=run_group(model,interfaces,grouped_surface,pin,ozone,self.tables,icloud=1,overlap=overlap,idcor=decor,
                             julday=int(calendar['julday']),reqc=int(scalar['has_reqc']),reqi=int(scalar['has_reqi']),
                             reqs=int(scalar['has_reqs']),o3input=2 if ozone_enabled else 0,presence=15)
            _launch('hexv842_mpas_rrtmg_lw_slotscatter_f32',ng*nm,ctx,dict(indices=indices,**{n:result[n] for n in ('tendency','radii','flux')},**dest),
                    dict(nc=np.int32(nc),ng=np.int32(ng),nm=np.int32(nm),write_radii=np.int32(capable),radii=np.int32(radii)))
        return None
