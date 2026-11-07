"""Compare strict Metal acoustic velocity caller with host CUDA source words."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core.acoustic import prepare_moist_cq,submit_metal_acoustic_advance_uv
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
pd=source[source.index('static __device__ __forceinline__\nreal pdmp'):source.index('// WRF calc_cq')]
body=source[source.index('static __device__\nreal dpn_face'):source.index('#if GPUWM_WRF_EXACT\nstatic __device__ __forceinline__\nreal wrf_',source.index('static __device__\nreal dpn_face'))]
pre='''#include <stddef.h>\n#define __device__\n#define __global__\n#define __forceinline__ inline\n#define GPUWM_WRF_EXACT 1\n#define I3S(k,j,i,ny,nx) (((size_t)(k)*(ny)+(j))*(nx)+(i))\ntypedef float real;struct dim { unsigned x; }; dim blockIdx,blockDim,threadIdx;\nstatic float arn_add(float a,float b){volatile float r=a+b;return r;}\nstatic float arn_mul(float a,float b){volatile float r=a*b;return r;}\n'''
wrap='''\nextern "C" void run(real* u,real* v,const real* ru,const real* rv,const real* pp,const real* po,const real* ph,const real* php,const real* phb,const real* alt,const real* al,const real* pb,const real* mup,const real* mu,const real* mub,const real* c1,const real* c2,const real* fnm,const real* fnp,const real* rdnw,const real* cqu,const real* cqv,int moist,float cf1,float cf2,float cf3,int lid,float rdx,float rdy,float dt,float sm,int zone,int b3,int nz,int ny,int nx,const real* mudf,const real* msfu,const real* msfv,float xs,float ys,int em,int mapped){blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*(ny+1)*(nx+1));++blockIdx.x)advance_uv(u,v,ru,rv,pp,po,ph,php,phb,alt,al,pb,mup,mu,mub,c1,c2,fnm,fnp,rdnw,cqu,cqv,moist,cf1,cf2,cf3,lid,rdx,rdy,dt,sm,zone,b3,nz,ny,nx,mudf,msfu,msfv,xs,ys,em,mapped);}\n'''
class State(SimpleNamespace):
 def scratch(self,shape,slot,dtype=None):
  if slot not in self.slots:self.slots[slot]=self._array_module.zeros(shape,np.float32)
  return self.slots[slot]
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+pd+body+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 fun=ct.CDLL(str(lib)).run;ptr=ct.POINTER(ct.c_float);F=ct.c_float;I=ct.c_int;fun.argtypes=[ptr]*22+[I]+[F]*3+[I]+[F]*4+[I]*5+[ptr]*3+[F]*2+[I]*2
 rng=np.random.default_rng(774);xp=MetalArrayModule();n=0
 for b3 in (0,1):
  for lid in (0,1):
   for boundary in ('periodic','specified','open_x','open_y'):
    for moist in (0,1):
     for em in (0,1):
      nz,ny,nx=4,4,5;shape=(nz,ny,nx);full=(nz+1,ny,nx);plane=(ny,nx)
      def rnd(shape):return np.asarray(rng.uniform(-2,2,shape),np.float32)
      h={'u_pp':rnd((nz,ny,nx+1)),'v_pp':rnd((nz,ny+1,nx)),'ru_t':rnd((nz,ny,nx+1)),'rv_t':rnd((nz,ny+1,nx)),'p_pp':rnd(shape),'p_pp_old':rnd(shape),'ph_pp':rnd(full),'php':rnd(full),'phb':rnd(full if b3 else (nz+1,)),'alt':rnd(shape),'al_pp':rnd(shape),'pb':rnd(shape if b3 else (nz,)),'mup':rnd(plane),'mu_pp':rnd(plane),'mub2d':rnd(plane),'c1h':rnd((nz,)),'c2h':rnd((nz,)),'fnm':rnd((nz,)),'fnp':rnd((nz,)),'rdnw':rnd((nz,)),'mudf':rnd(plane),'msfu':np.asarray(rng.uniform(.8,1.2,(ny,nx+1)),np.float32),'msfv':np.asarray(rng.uniform(.8,1.2,(ny+1,nx)),np.float32)}
      h['msfv_inv'] = np.nextafter(np.float32(1.0) / h['msfv'], np.float32(np.inf))
      h['p_pp'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
      s=State(_metal_setup_state=True,_array_module=xp,slots={},qv=None,has_msf=em,cf1=np.float32(.6),cf2=np.float32(.3),cf3=np.float32(.1),p=xp.asarray(np.zeros(shape,np.float32)),thb=xp.asarray(np.zeros(shape if b3 else (nz,),np.float32)))
      for name,value in h.items():setattr(s,name,xp.asarray(value))
      cfg=SimpleNamespace(nz=nz,ny=ny,nx=nx,dx=1200.,dy=1700.,top_lid=lid,mp_physics=1,moist_cq=True,spec_zone=1,specified=boundary=='specified',nested=False,open_x=boundary=='open_x',open_y=boundary=='open_y',emdiv=.12)
      if moist:
       for name in ('qv','qc','qr'):setattr(s,name,xp.asarray(np.asarray(rng.uniform(0,.01,shape),np.float32)))
      cq=prepare_moist_cq(s,cfg); cqu=cq[0].to_numpy();cqv=cq[1].to_numpy()
      outu=h['u_pp'].copy();outv=h['v_pp'].copy();names=('ru_t','rv_t','p_pp','p_pp_old','ph_pp','php','phb','alt','al_pp','pb','mup','mu_pp','mub2d','c1h','c2h','fnm','fnp','rdnw');arr=lambda x:x.ctypes.data_as(ptr)
      xs=np.float32(np.float32(-cfg.emdiv)*np.float32(cfg.dx));ys=np.float32(np.float32(-cfg.emdiv)*np.float32(cfg.dy));dt=np.float32(.8);sm=np.float32(.17 if em else 0.0)
      fun(arr(outu),arr(outv),*[arr(h[x]) for x in names],arr(cqu),arr(cqv),moist,s.cf1,s.cf2,s.cf3,lid,F(1/cfg.dx),F(1/cfg.dy),dt,sm,int(cfg.specified),b3,nz,ny,nx,arr(h['mudf']),arr(h['msfu']),arr(h['msfv_inv']),xs,ys,em,em)
      if cfg.open_x:
       outu[:,:,0]=h['u_pp'][:,:,0]+dt*h['ru_t'][:,:,0];outu[:,:,-1]=h['u_pp'][:,:,-1]+dt*h['ru_t'][:,:,-1]
      if cfg.open_y:
       outv[:,0,:]=h['v_pp'][:,0,:]+dt*h['rv_t'][:,0,:];outv[:,-1,:]=h['v_pp'][:,-1,:]+dt*h['rv_t'][:,-1,:]
      call=submit_metal_acoustic_advance_uv(s,cfg,float(dt),float(sm),cq=cq,mudf=s.mudf if em else None,_prepare=True)
      xp.context.submit_prepared_kernel_call(call,label='strict velocity comparison').wait()
      for got,want in ((s.u_pp.to_numpy(),outu),(s.v_pp.to_numpy(),outv)):
       np.testing.assert_array_equal(got.view(np.uint32),want.view(np.uint32),err_msg=str((b3,lid,boundary,moist,em)))
      n+=1
 print('matched',n,'strict acoustic velocity caller cases')
