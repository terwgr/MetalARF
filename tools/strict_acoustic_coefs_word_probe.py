"""Compare the strict Metal coefficient caller with host-compiled CUDA source."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core import constants as c
from woof.core.acoustic import prepare_acoustic_coefficients
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
 result=original(self,*args,**kwargs)
 if result is not None:result.wait()
 return result
MetalContext._submit_kernel_batch=sync
source=(Path(__file__).resolve().parents[1]/'woof/core/kernels/acoustic.cu').read_text()
start=source.index('extern "C" __global__\nvoid calc_coefs(');end=source.index('#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION\nstatic __device__',start)
body=source[start:end]
pre='''#include <stddef.h>\n#define __global__\n#define GPUWM_WRF_EXACT 1\n#define G %sf\n#define GAMMA %sf\ntypedef float real; struct dim { unsigned x; }; dim blockIdx,blockDim,threadIdx;\n'''%(repr(c.G),repr(c.GAMMA))
wrapper='''\nextern "C" void run(const real* p,const real* alt,const real* mu,real* c2a,real* a,real* alpha,real* gam,const real* rdn,const real* rdnw,const real* c1h,const real* c2h,const real* c1f,const real* c2f,const real* mub,const real* cqw,int moist,float dt,float eps,int lid,int nz,int ny,int nx){blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(ny*nx);++blockIdx.x)calc_coefs(p,alt,mu,c2a,a,alpha,gam,rdn,rdnw,c1h,c2h,c1f,c2f,mub,cqw,moist,dt,eps,lid,nz,ny,nx);}\n'''
class State(SimpleNamespace):
 def scratch(self,shape,slot,dtype=None):
  if slot not in self.slots:self.slots[slot]=self._array_module.asarray(np.full(shape,np.float32(-0.0)))
  return self.slots[slot]
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+body+wrapper)
 subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 run=ct.CDLL(str(lib)).run;ptr=ct.POINTER(ct.c_float);run.argtypes=[ptr]*15+[ct.c_int,ct.c_float,ct.c_float]+[ct.c_int]*4
 rng=np.random.default_rng(442);xp=MetalArrayModule();cases=0
 for nz in (1,2,5,12):
  for lid in (0,1):
   for moist in (0,1):
    ny,nx=2,3;shape=(nz,ny,nx);full=(nz+1,ny,nx);plane=(ny,nx)
    def pos(shape,lo=0.5,hi=2):return np.asarray(rng.uniform(lo,hi,shape),np.float32)
    h={'p':pos(shape,70000,100000),'alt':pos(shape),'mup':pos(plane),'mub2d':pos(plane,3,5),'rdn':pos((nz,)),'rdnw':pos((nz,)),'c1h':pos((nz,)),'c2h':pos((nz,)),'c1f':pos((nz+1,)),'c2f':pos((nz+1,))}
    h['p'].flat[:min(h['p'].size,4)]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)[:min(h['p'].size,4)]
    state=State(_metal_setup_state=True,_array_module=xp,slots={},qv=None)
    for name,data in h.items():setattr(state,name,xp.asarray(data))
    cfg=SimpleNamespace(nz=nz,ny=ny,nx=nx,mp_physics=1,moist_cq=True,top_lid=lid,epssm=0.1)
    if moist:
     for name in ('qv','qc','qr'):setattr(state,name,xp.asarray(pos(shape,0.0001,.01)))
    result=prepare_acoustic_coefficients(state,cfg,1.3)
    # The production moist-CQ stage runs first and the coefficient call waits.
    cqw=result[6].to_numpy() if moist else np.zeros(full,np.float32)
    out=[np.full(shape if n==0 else full,np.float32(-0.0)) for n in range(4)]
    args=[h['p'],h['alt'],h['mup'],*out,h['rdn'],h['rdnw'],h['c1h'],h['c2h'],h['c1f'],h['c2f'],h['mub2d'],cqw]
    run(*[x.ctypes.data_as(ptr) for x in args],moist,np.float32(1.3),np.float32(.1),lid,nz,ny,nx)
    for name,actual,expected in zip(('c2a','a','alpha','gam'),result[:4],out):
     np.testing.assert_array_equal(actual.to_numpy().view(np.uint32),expected.view(np.uint32),err_msg=str((name,nz,lid,moist)))
    cases+=1
 print('matched',cases,'acoustic coefficient caller cases')
