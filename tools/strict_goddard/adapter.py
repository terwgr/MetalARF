"""Compare the actual Goddard adapter and spectrum publication to host source."""
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/thompson_real_column_parity'))
import host_backend
from tools.goddard_metal.column_probe import fixture
from tools.metal_solar_word_probe import build_reference
from woof.core import kernels
from woof.core.radiation_tables import radiation_table_host_words
from woof.core.radiation_composition import ColumnTableRadiation
from woof.core.rrtmgp import cal_cldfra1
from woof.core.rrtmg_legacy import radconst,_DEGRAD,_DPD
from woof.core.swint import calendar_scalars
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED
assert ENABLED
nz,nc=10,6
inputs,planes,surface,exner=fixture(nz,nc)
pm,t,p,qv,_,dz,rho,*_=planes
qc=planes[8].copy(); zeros=np.zeros((nz,nc),np.float32)
start=datetime(2024,6,20,12,tzinfo=timezone.utc); elapsed=120.; radt=3.
latitude=surface[0]; longitude=np.linspace(-100,100,nc,dtype=np.float32)
julian,xtime,gmt=calendar_scalars(start,elapsed)
declin,solcon=radconst(julian)
with tempfile.TemporaryDirectory() as temp:
    lib,reference=build_reference(Path(temp))
    pair=np.empty(2,np.float32);cosine=np.empty(nc,np.float32);hrang=np.empty(nc,np.float32)
    reference(nc,latitude.ctypes.data,longitude.ctypes.data,julian,np.float32(xtime+np.float32(radt)*np.float32(.5)),gmt,_DEGRAD,_DPD,1,1,declin,pair.ctypes.data,cosine.ctypes.data,hrang.ctypes.data)
    host=host_backend.HostModule('strict_goddard_adapter',kernels.module_source_int_defines('goddard',(('GDD_MAX_LEVELS',nz),)))
    def invoke(name,values,count):
        order=[x.name for x in host.kernels[name]]
        host.get_function(name)(((count+127)//128,),(128,),[values[n] for n in order])
    tw=np.empty((nz+1,nc),np.float32)
    invoke('goddard_interface_temperature',dict(inputs,tw=tw,nz=np.int32(nz),columns=np.int32(nc)),tw.size)
    common={'np':np.int32(nz),'columns':np.int32(nc)}
    table=radiation_table_host_words('goddard-cloud')
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        shape=(nz,1,nc); dev=lambda x:xp.asarray(np.ascontiguousarray(x).reshape(shape))
        state=SimpleNamespace(_metal_setup_state=True,elapsed_seconds=elapsed,fnm=inputs['fnm'],fnp=inputs['fnp'],
            qv=dev(qv),qc=dev(qc),qr=None,qi=None,qs=None,qg=None,qh=None)
        atmosphere={'temperature':dev(t),'pressure':dev(p),'dz':dev(dz),'rho':dev(rho),'exner':dev(exner),
                    'p_interface':xp.asarray(pm.reshape(nz+1,1,nc)),
                    'z_interface':xp.asarray(inputs['z'].reshape(nz+1,1,nc))}
        fields={'tsk':xp.asarray(surface[2].reshape(1,nc)),'emiss':xp.asarray(surface[3].reshape(1,nc)),
                'albedo':xp.asarray(surface[4].reshape(1,nc)),'glw':xp.asarray(np.full((1,nc),300,np.float32))}
        for icloud in (0,1):
            cf=cal_cldfra1(qv,qc,zeros,zeros,t,p,f_qi=False,f_qs=False) if icloud else zeros
            raw=np.concatenate([x.reshape(-1) for x in [pm,t,p,qv,cf,dz,rho,zeros,qc,zeros,zeros,zeros,zeros,*([zeros]*6)]]+
                [latitude,cosine,tw[0],*surface[2:],exner.reshape(-1)])
            band=np.empty((5*nz+15)*nc,np.float32);flags=np.empty(3*nc,np.int32)
            dd=np.empty(nz*nc,np.float32);q=np.empty(6*nz*nc,np.float32);re=np.empty_like(q)
            invoke('goddard_prepare_columns',dict(raw=raw,band=band,flags=flags,dz=dd,q=q,re=re,**common,julday=np.int32(int(julian)+1)),nc)
            optics=np.empty(63*nz*nc,np.float32)
            invoke('goddard_cloud_columns',dict(dz=dd,q=q,re=re,lut=table,output=optics,**common),nz*nc)
            flux=np.empty((6*(nz+1)+4)*nc,np.float32)
            invoke('goddard_bands_columns',dict(input=band,flags=flags,optics=optics,output=flux,**common),nc)
            result=np.empty((2*nz+15)*nc,np.float32)
            invoke('goddard_publish_columns',dict(band=band,flux=flux,raw=raw,output=result,**common,solcon=np.float32(solcon)),nc)
            result=result.reshape(2*nz+15,nc)
            cfg=SimpleNamespace(mp_physics=0,bl_pbl_physics=0,icloud=icloud,icloud_bl=0,use_mp_re=1,radt=radt,radt_minutes=radt,dt=1.)
            for lw,sw in ((True,True),(True,False),(False,True)):
                adapter=ColumnTableRadiation(start,latitude.reshape(1,nc),longitude.reshape(1,nc),column_chunk=3,longwave=lw,shortwave=sw)
                got=adapter(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
                expected={'rthratenlw':result[:nz] if lw else zeros,'rthratensw':result[nz:2*nz] if sw else zeros,
                    'swdown':result[2*nz] if sw else np.zeros(nc,np.float32),
                    'glw':result[2*nz+1] if lw else np.full(nc,300,np.float32),
                    'gsw':result[2*nz+2] if sw else np.zeros(nc,np.float32),
                    'coszen':cosine,'cldfra':cf}
                if lw:expected['olr']=result[2*nz+3]
                if sw:
                    expected['swddir']=result[2*nz+5];expected['swddif']=result[2*nz+6]
                for name,index in zip(ColumnTableRadiation.diagnostics,(4,5,6,7,8,9,10,11,12,13,14)):
                    if (lw if 'LW' in name else sw):expected['diag:'+name]=result[2*nz+index]
                count=diff=0
                for name,exp in expected.items():
                    value=got.flux_diagnostics[name[5:]] if name.startswith('diag:') else getattr(got,name)
                    a=value.to_numpy().reshape(-1).view(np.uint32);b=np.asarray(exp).reshape(-1).view(np.uint32)
                    bad=np.flatnonzero(a!=b);count+=a.size;diff+=len(bad)
                    if len(bad):
                        i=bad[0];print('first difference',icloud,lw,sw,name,i,hex(int(b[i])),hex(int(a[i])))
                print(f'icloud={icloud} lw={lw} sw={sw}: {diff}/{count} word differences')
                if diff or adapter.update_count!=1:raise SystemExit(1)
