"""Production Dudhia caller versus host-compiled CUDA source."""
import hashlib,os
from datetime import datetime, timedelta
from types import SimpleNamespace
import numpy as np
os.environ.setdefault('GPUWM_WRF_EXACT','1')
from tools.thompson_real_column_parity import host_backend
host_backend.install()
from woof.core.kernels import module_source
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

from woof.metal import dudhia_shortwave as metal
from woof.core import dudhia
real_solar = dudhia.wrf_solar_geometry
use_real_solar = os.environ.get("DUDHIA_REAL_SOLAR") == "1"
submitted=[];submit_original=metal.submit_dudhia_shortwave
def submit_capture(**kwargs):
 submitted.append((kwargs['qc'].to_numpy(),kwargs['qi'].to_numpy()))
 return submit_original(**kwargs)
metal.submit_dudhia_shortwave=submit_capture
source=module_source('dudhia_sw');reference=host_backend.HostModule('dudhia_caller_ref',source).get_function('dudhia_sw_column');mynnsource=module_source('rrtmg_legacy_adapter')
original_mynn_sha256=hashlib.sha256(mynnsource.encode()).hexdigest()
for opcode,op in (('add','+'),('sub','-'),('mul','*'),('div','/')):
 old='{ float r; asm("'+opcode+'.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }'
 new='{ volatile float r = a '+op+' b; return r; }'
 if mynnsource.count(old)!=1: raise ValueError('unexpected RLA helper '+opcode)
 mynnsource=mynnsource.replace(old,new)
adapted_mynn_sha256=hashlib.sha256(mynnsource.encode()).hexdigest()
mynnref=host_backend.HostModule('dudhia_mynn_ref',mynnsource).get_function('rla_mynn_gsd41')
original=MetalContext._submit_kernel_batch
def sync(self,*a,**k):
 r=original(self,*a,**k)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
rng=np.random.default_rng(113);cases=words=0
with MetalContext() as ctx:
 xp=MetalArrayModule(ctx);nz,ny,nx=5,3,4;shape=(nz,ny,nx);plane=(ny,nx);ncol=ny*nx
 def rnd(sh,lo,hi):return rng.uniform(lo,hi,sh).astype(np.float32)
 for icloud in (0,1,2):
  for active in (False,True):
   for elapsed in (0.,60.):
    h={'temperature':rnd(shape,220,300),'pressure':rnd(shape,20000,95000),'qv':rnd(shape,0,.02),'qc':rnd(shape,0,.001),'qr':rnd(shape,0,.0002),'qi':rnd(shape,0,.001),'qs':rnd(shape,0,.0002),'qg':rnd(shape,0,.0001),'dz':rnd(shape,100,500),'exner':rnd(shape,.5,1.),'albedo':rnd(plane,.05,.7),'mu':np.resize(np.array([-.1,0,.2,.7],np.float32),plane)}
    h['qv'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view('f')
    h['qc'].fill(0);h['qi'].fill(0);h['qc_bl']=rnd(shape,0,.0001);h['qi_bl']=rnd(shape,0,.0001);h['cldfra_bl']=rnd(shape,0,1);d={k:xp.asarray(v) for k,v in h.items()};atmos={k:d[k] for k in ('temperature','pressure','qv','qc','qi','dz','exner')};state=SimpleNamespace(_metal_setup_state=True,_array_module=xp,elapsed_seconds=elapsed,qc=d['qc'],qr=d['qr'],qi=d['qi'],qs=d['qs'],qg=d['qg']);fields={'albedo':d['albedo'],'glw':xp.asarray(np.full(plane,300,np.float32)),'qc_bl':d['qc_bl'],'qi_bl':d['qi_bl'],'cldfra_bl':d['cldfra_bl']};cfg=SimpleNamespace(bl_pbl_physics=5 if active else 0,icloud_bl=1,bl_mynn_version='gsd_41',dt=60.,radt=5.,radt_minutes=0.,solar_generation='wrf_461');seen=[]
    def solar(*args,**kwargs):
     seen.append(kwargs['hour_offset_seconds'])
     if use_real_solar:
      mu,solcon=real_solar(*args,**kwargs);h['mu']=np.asarray(mu,np.float32);return mu,solcon
     return h['mu'],np.float32(1361)
    dudhia.wrf_solar_geometry=solar;rad=dudhia.DudhiaShortwaveRadiation(datetime(2020,6,1),np.zeros(plane),np.zeros(plane),icloud=icloud)
    submitted.clear();result=rad(atmosphere=atmos,fields=fields,state=state,cfg=cfg);assert rad.update_count==1 and rad.restart_identity['icloud']==icloud and seen==[150.0]
    outputs={k:np.zeros(shape if k=='heating' else plane,np.float32) for k in ('heating','gsw','swdown')};
    if active and icloud != 2:
     h['qc']=h['qc'].copy();h['qi']=h['qi'].copy();n=nz*ny*nx;mynnref(((n+127)//128,),(128,),(np.int64(n),h['qc'],h['qi'],h['qc_bl'],h['cldfra_bl'],h['temperature'],None,np.int32(elapsed==0.),None,None,np.int32(0),np.float32(1e-6),np.float32(1e-8),np.float32(.001)))
    for actual_cloud,expected_cloud in zip(submitted[-1],(h['qc'],h['qi'])):
     np.testing.assert_array_equal(actual_cloud.view('u4'),expected_cloud.view('u4'))
    solcon = real_solar(rad.start_time + timedelta(seconds=elapsed), rad.latitude_deg, rad.longitude_deg, hour_offset_seconds=150., generation='wrf_461')[1] if use_real_solar else 1361
    inputs=[h[k] for k in ('temperature','pressure','qv','qc','qr','qi','qs','qg','dz','exner','mu','albedo')];reference(((ncol+127)//128,),(128,),(*inputs,outputs['heating'],outputs['gsw'],outputs['swdown'],np.float32(solcon),dudhia._wrf_cssca(1.),np.int32(icloud),np.int32(1),np.int32(ncol),np.int32(nz),np.int64(1),np.int64(-ncol),np.int64((nz-1)*ncol)))
    for actual,expected in ((result.rthratensw,outputs['heating']),(result.swdown,outputs['swdown']),(result.gsw,outputs['gsw']),(result.coszen,h['mu'])):
     if os.environ['GPUWM_WRF_EXACT']=='1' or actual is result.coszen:
      np.testing.assert_array_equal(actual.to_numpy().view('u4'),expected.view('u4'));words+=expected.size
    np.testing.assert_array_equal(result.rthratenlw.to_numpy().view('u4'),np.zeros(shape,np.uint32));assert result.glw is fields['glw'];cases+=1
 print('matched',cases,'actual caller cases,',words,'strict/output words; pre-column clouds matched host merge; active GSD merge checked;', ('real upstream solar exercised, not independently verified;' if use_real_solar else 'solar stubbed;'), 'CUDA host source',hashlib.sha256(source.encode()).hexdigest(), 'mynn original/adapted',original_mynn_sha256,adapted_mynn_sha256,'host flags',host_backend.CXX_FLAGS)
