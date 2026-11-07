"""Proposed strict shared MYNN caller versus host-compiled adapter source."""
import hashlib,importlib.util,os,sys
from pathlib import Path
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from tools.thompson_real_column_parity import host_backend
host_backend.install()
from woof.core.kernels import module_source
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.core import mynn_radiation as m
source=module_source('rrtmg_legacy_adapter')
original_sha256=hashlib.sha256(source.encode()).hexdigest()
for opcode,op in (('add','+'),('sub','-'),('mul','*'),('div','/')):
 old='{ float r; asm("'+opcode+'.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }'
 new='{ volatile float r = a '+op+' b; return r; }'
 if source.count(old)!=1: raise ValueError('unexpected RLA helper '+opcode)
 source=source.replace(old,new)
adapted_sha256=hashlib.sha256(source.encode()).hexdigest()
refmodule=host_backend.HostModule('mynn_exact_ref',source);refs={'wrf_461':refmodule.get_function('rla_mynn'),'gsd_41':refmodule.get_function('rla_mynn_gsd41')}
orig=MetalContext._submit_kernel_batch
def sync(self,*a,**kw):
 r=orig(self,*a,**kw)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
rng=np.random.default_rng(291);cases=words=0
with MetalContext() as ctx:
 xp=MetalArrayModule(ctx);shape=(4,13);n=np.prod(shape)
 for version in ('wrf_461','gsd_41'):
  for first in (True,False):
   for has_cf in (True,False):
    for tiny in (False,True):
     q=np.zeros(shape,'f');e=np.zeros(shape,'f');cb=rng.uniform(0,.001,shape).astype('f');ib=rng.uniform(0,.001,shape).astype('f');fb=rng.uniform(0,1,shape).astype('f');t=rng.uniform(210,310,shape).astype('f');cloud=rng.uniform(0,1,shape).astype('f')
     q.flat[:7]=np.array([0,0x80000000,1,0x80000001,0x358637bd,0x358637be,0x358637bf],np.uint32).view('f');e.flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view('f');fb.flat[:3]=np.array([.0009,.001,.0011],np.float32)
     if tiny:cb*=np.float32(2**-120);ib*=np.float32(2**-120)
     qr=q.copy();er=e.copy();cr=cloud.copy() if has_cf else None
     if version=='wrf_461':args=(np.int64(n),qr,er,cb,ib,fb,cr,np.int32(not first),None,None,np.int32(0),np.float32(1e-6),np.float32(1e-8),np.float32(.001))
     else:args=(np.int64(n),qr,er,cb,fb,t,cr,np.int32(first),None,None,np.int32(0),np.float32(1e-6),np.float32(1e-8),np.float32(.001))
     refs[version](((int(n)+127)//128,),(128,),args)
     d=[xp.asarray(x) for x in (q,e,cb,ib,fb,t)];dc=xp.asarray(cloud) if has_cf else None
     m.merge_mynn_bl_clouds(d[0],d[1],dc,qc_bl=d[2],qi_bl=d[3],cldfra_bl=d[4],bl_pbl_physics=5,icloud_bl=1,itimestep=1 if first else 2,bl_mynn_version=version,temperature=d[5])
     for actual,expected in ((d[0],qr),(d[1],er))+(((dc,cr),) if has_cf else ()):
      np.testing.assert_array_equal(actual.to_numpy().view('u4'),expected.view('u4'),err_msg=str((version,first,has_cf,tiny)));words+=expected.size
     cases+=1
 # Exceptional temperatures follow the shared helper's clip expression, not RLA's ternary.
 q=np.zeros((1,4),np.float32);e=q.copy();cb=np.full_like(q,0.0001);ib=cb.copy();fb=np.full_like(q,0.5);t=np.array([np.nan,np.inf,-np.inf,254.0],np.float32).reshape(1,4);qr=q.copy();er=e.copy()
 with np.errstate(invalid='ignore'):
  m.merge_mynn_bl_clouds(qr,er,None,qc_bl=cb,qi_bl=ib,cldfra_bl=fb,bl_pbl_physics=5,icloud_bl=1,itimestep=1,bl_mynn_version='gsd_41',temperature=t)
  d=[xp.asarray(x) for x in (q,e,cb,ib,fb,t)];m.merge_mynn_bl_clouds(d[0],d[1],None,qc_bl=d[2],qi_bl=d[3],cldfra_bl=d[4],bl_pbl_physics=5,icloud_bl=1,itimestep=1,bl_mynn_version='gsd_41',temperature=d[5])
  for actual,expected in ((d[0],qr),(d[1],er)):
   np.testing.assert_array_equal(actual.to_numpy().view('u4'),expected.view('u4'))
 print('exceptional GSD temperatures compared to NumPy shared-helper expression; CuPy source clip reviewed separately')
print('matched',cases,'strict MYNN shared caller cases,',words,'words; host-compiled CUDA adapter, finite physical temperatures; original/adapted',original_sha256,adapted_sha256,'host flags',host_backend.CXX_FLAGS)
