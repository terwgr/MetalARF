"""Compare strict acoustic mass/theta and frame callers with composed CUDA source."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core.acoustic import (submit_metal_acoustic_advance_mu_th,
    submit_metal_acoustic_exact_frame_mu_t, init_wrf_acoustic_muts, WRF_ACOUSTIC_MUTS_SLOT)
from woof.core.kernels import module_source
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
 result=original(self,*args,**kwargs)
 if result is not None:result.wait()
 return result
MetalContext._submit_kernel_batch=sync
source=module_source('acoustic')
start=source.index('#if GPUWM_WRF_EXACT\nstatic __device__ __forceinline__\nreal wrf_total_divergence')
end=source.index('// Forward steps of mu',start)
body=source[start:end]
pre='''#include <stddef.h>\n#include <algorithm>\nusing std::min;using std::max;\n#define __device__\n#define __global__\n#define __forceinline__ inline\n#define GPUWM_WRF_EXACT 1\n#define I3S(k,j,i,ny,nx) (((size_t)(k)*(ny)+(j))*(nx)+(i))\n#define IDX3(k,j,i) (((size_t)(k)*(ny)+(j))*(nx)+(i))\ntypedef float real;struct dim{unsigned x;};dim blockIdx,blockDim,threadIdx;\nstatic float __fdiv_rn(float a,float b){volatile float r=a/b;return r;}\n'''
wrap='''\nextern "C" void run(const real* up,const real* vp,const real* u,const real* v,const real* mup,real* mu,real* muold,const real* rmu,real* mudf,const real* th,real* thp,real* thold,const real* rth,real* ww,const real* wwref,const real* p,real* pold,const real* dnw,const real* rdnw,const real* fnm,const real* fnp,const real* c1,const real* c2,const real* mub,const real* msft,const real* msfu,const real* msfv,real* muts,int mapped,float rdx,float rdy,float dt,int nz,int ny,int nx,int ox,int oy,int zone,int write,int frame){blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(ny*nx);++blockIdx.x)wrf_advance_mu_theta(up,vp,u,v,mup,mu,muold,rmu,mudf,write,th,thp,thold,rth,ww,wwref,p,pold,dnw,rdnw,fnm,fnp,c1,c2,mub,msft,msfu,msfv,mapped,rdx,rdy,dt,nz,ny,nx,ox,oy,zone);if(frame)for(blockIdx.x=0;blockIdx.x<(unsigned)(ny*nx);++blockIdx.x)advance_exact_frame_mu_t(mu,thp,rmu,rth,dt,zone,!ox,nz,ny,nx,muts,mub,mup);}\n'''
class State(SimpleNamespace):
 def scratch(self,shape,slot,dtype=None):
  if slot not in self.slots:self.slots[slot]=self._array_module.asarray(np.full(shape,np.float32(-0.0)))
  return self.slots[slot]
 def existing_scratch(self,slot):return self.slots.get(slot)
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+body+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 fun=ct.CDLL(str(lib)).run;ptr=ct.POINTER(ct.c_float);F=ct.c_float;I=ct.c_int;fun.argtypes=[ptr]*28+[I]+[F]*3+[I]*8
 rng=np.random.default_rng(4471);xp=MetalArrayModule();n=0
 words=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000],np.uint32).view(np.float32).reshape(2,3)
 reference=np.flip(words,axis=1).copy()
 seed=State(_metal_setup_state=True,_array_module=xp,slots={},mub2d=xp.asarray(words),mup=xp.asarray(reference))
 for ref in (None,xp.asarray(words)):
  init_wrf_acoustic_muts(seed,reference=ref)
  expected=words+(reference if ref is None else words)
  np.testing.assert_array_equal(seed.slots[WRF_ACOUSTIC_MUTS_SLOT].to_numpy().view(np.uint32),expected.view(np.uint32))
 for b3 in (0,1):
  for boundary in ('periodic','specified','nested','open_x','open_y'):
   for mapped in (0,1):
    for write in (0,1):
     nz,ny,nx=4,5,6;shape=(nz,ny,nx);full=(nz+1,ny,nx);plane=(ny,nx)
     def rnd(shape):return np.asarray(rng.uniform(-2,2,shape),np.float32)
     h={'u_pp':rnd((nz,ny,nx+1)),'v_pp':rnd((nz,ny+1,nx)),'u':rnd((nz,ny,nx+1)),'v':rnd((nz,ny+1,nx)),'mup':rnd(plane),'mu_pp':rnd(plane),'rmu_t':rnd(plane),'mudf':rnd(plane),'thp':rnd(shape),'th_pp':rnd(shape),'rth_t':rnd(shape),'ww_pp':rnd(full),'ww_ref':rnd(full),'p_pp':rnd(shape),'p_pp_old':rnd(shape),'dnw':rnd((nz,)),'rdnw':rnd((nz,)),'fnm':rnd((nz,)),'fnp':rnd((nz,)),'c1h':rnd((nz,)),'c2h':rnd((nz,)),'mub2d':rnd(plane),'msft':np.asarray(rng.uniform(.8,1.2,plane),np.float32),'msfu':np.asarray(rng.uniform(.8,1.2,(ny,nx+1)),np.float32),'msfv':np.asarray(rng.uniform(.8,1.2,(ny+1,nx)),np.float32),'muts':rnd(plane)}
     h['msfv_inv'] = np.nextafter(np.float32(1.0) / h['msfv'], np.float32(np.inf))
     h['thp'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
     s=State(_metal_setup_state=True,_array_module=xp,slots={},_wrf_acoustic_muts_initialized=True,has_msf=mapped,thb=xp.asarray(np.zeros(shape if b3 else (nz,),np.float32)))
     for name,value in h.items():
      if name not in ('ww_ref','muts'):setattr(s,name,xp.asarray(value))
     s.slots['rk_ww']=xp.asarray(h['ww_ref']);s.slots[WRF_ACOUSTIC_MUTS_SLOT]=xp.asarray(h['muts'])
     cfg=SimpleNamespace(nz=nz,ny=ny,nx=nx,dx=1200.,dy=1700.,specified=boundary=='specified',nested=boundary=='nested',open_x=boundary=='open_x',open_y=boundary=='open_y')
     forced=cfg.specified or cfg.nested;ox=int(cfg.open_x or forced);oy=int(cfg.open_y or forced);zone=int(forced)
     muold=np.full(plane,np.float32(-0.0));thold=np.full(shape,np.float32(-0.0));host={name:value.copy() for name,value in h.items()};pold=host['p_pp_old'];dummy=np.full(plane,np.float32(-0.0))
     names=('u_pp','v_pp','u','v','mup','mu_pp');arr=lambda x:x.ctypes.data_as(ptr)
     seq=[host[x] for x in names]+[muold,host['rmu_t'],host['mudf'] if write else dummy,host['thp'],host['th_pp'],thold,host['rth_t'],host['ww_pp'],host['ww_ref'],host['p_pp'],pold]+[host[x] for x in ('dnw','rdnw','fnm','fnp','c1h','c2h','mub2d','msft','msfu','msfv_inv','muts')]
     fun(*[arr(x) for x in seq],mapped,F(1/cfg.dx),F(1/cfg.dy),F(.7),nz,ny,nx,ox,oy,zone,write,int(forced))
     call=submit_metal_acoustic_advance_mu_th(s,cfg,.7,mudf=s.mudf if write else None,_prepare=True)
     xp.context.submit_prepared_kernel_call(call,label='strict mass theta').wait()
     if forced:
      call=submit_metal_acoustic_exact_frame_mu_t(s,cfg,.7,_prepare=True)
      xp.context.submit_prepared_kernel_call(call,label='strict mass frame').wait()
     comparisons=[(s.mu_pp,host['mu_pp']),(s.th_pp,host['th_pp']),(s.ww_pp,host['ww_pp']),(s.p_pp_old,pold),(s.slots['acoustic_mu_pp_old'],muold),(s.slots['acoustic_th_pp_old'],thold)]
     if write:comparisons.append((s.mudf,host['mudf']))
     if forced:comparisons.append((s.slots[WRF_ACOUSTIC_MUTS_SLOT],host['muts']))
     for actual,expected in comparisons:np.testing.assert_array_equal(actual.to_numpy().view(np.uint32),expected.view(np.uint32),err_msg=str((b3,boundary,mapped,write)))
     n+=1
 print('matched',n,'strict acoustic mass/theta caller and frame cases')
