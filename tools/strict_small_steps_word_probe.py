"""Compare strict small-step callers against the composed dycore CUDA source."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core import constants as c
from woof.core.acoustic import WRF_ACOUSTIC_MUTS_SLOT
from woof.core.dycore import _prepare_small_step_init_launch,_prepare_small_step_finish_launch
from woof.core.kernels import module_source
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
 result=original(self,*args,**kwargs)
 if result is not None:result.wait()
 return result
MetalContext._submit_kernel_batch=sync
source=module_source('dycore')
def extract(name):
 start=source.rindex('extern "C" __global__',0,source.index('void '+name+'('));pos=source.index('{',start);depth=1;end=pos+1
 while depth:
  if source[end]=='{':depth+=1
  if source[end]=='}':depth-=1
  end+=1
 return source[start:end]
names=('u_pp','v_pp','u0','v0','u','v','mup0','mup','mub2d','c1h','c2h','msfu','msfv','w_pp','th_pp','ph_pp','mu_pp','al_pp','p_pp','p_pp_old','w0','w','thp0','thp','php0','php','p','alt','thb','c1f','c2f','rdnw','msft','h_diabatic','muts_wrf')
pre='''#include <stddef.h>\n#include <algorithm>\nusing std::min;using std::max;\n#define __global__\n#define GPUWM_WRF_EXACT 1\n#define GAMMA %sf\n#define I3S(k,j,i,ny,nx) (((size_t)(k)*(ny)+(j))*(nx)+(i))\ntypedef float real;struct dim {unsigned x;};dim blockIdx,blockDim,threadIdx;\nstatic float rn_add(float a,float b){volatile float r=a+b;return r;}\nstatic float rn_sub(float a,float b){volatile float r=a-b;return r;}\nstatic float rn_mul(float a,float b){volatile float r=a*b;return r;}\nstatic float rn_div(float a,float b){volatile float r=a/b;return r;}\n'''%repr(c.GAMMA)
args=','.join('real* '+x for x in names)
wrapper=('extern "C" void run('+args+',int phase,int mapped,int bx,int by,int b3,int nz,int ny,int nx,int rk,float heat,int remove){blockDim.x=1;threadIdx.x=0;if(!phase){for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*(ny+1)*(nx+1));++blockIdx.x)small_step_init_uv(u_pp,v_pp,u0,v0,u,v,mup0,mup,mub2d,c1h,c2h,msfu,msfv,mapped,bx,by,nz,ny,nx,rk);for(blockIdx.x=0;blockIdx.x<(unsigned)(ny*nx);++blockIdx.x)small_step_init_column(w_pp,th_pp,ph_pp,mu_pp,al_pp,p_pp,p_pp_old,w0,w,thp0,thp,php0,php,p,alt,mup0,mup,mub2d,thb,c1h,c2h,c1f,c2f,rdnw,msft,mapped,b3,nz,ny,nx);}else{for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*(ny+1)*(nx+1));++blockIdx.x)small_step_finish_uv(u,v,u_pp,v_pp,mup,mu_pp,mub2d,c1h,c2h,msfu,msfv,mapped,bx,by,nz,ny,nx,muts_wrf);for(blockIdx.x=0;blockIdx.x<(unsigned)(ny*nx);++blockIdx.x)small_step_finish_column(w,thp,php,mup,w_pp,th_pp,ph_pp,mu_pp,mub2d,thb,c1h,c2h,c1f,c2f,msft,h_diabatic,heat,remove,mapped,b3,nz,ny,nx,muts_wrf);}}\n')
class State(SimpleNamespace):
 def existing_scratch(self,slot):return self.slots.get(slot)
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+''.join(extract(n) for n in ('small_step_init_uv','small_step_init_column','small_step_finish_uv','small_step_finish_column'))+wrapper)
 subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 fun=ct.CDLL(str(lib)).run;ptr=ct.POINTER(ct.c_float);I=ct.c_int;F=ct.c_float;fun.argtypes=[ptr]*len(names)+[I]*9+[F,I]
 rng=np.random.default_rng(117);xp=MetalArrayModule();cases=0
 for b3 in (0,1):
  for rk in (1,2,3):
   for boundary in ('periodic','open_x','open_y','specified'):
    for mapped in (0,1):
     for heat in (0.,.4):
      nz,ny,nx=4,3,4;shape=(nz,ny,nx);full=(nz+1,ny,nx);plane=(ny,nx)
      def rnd(shape,scale=1):return np.asarray(rng.uniform(-scale,scale,shape),np.float32)
      def pos(shape,lo=.8,hi=1.2):return np.asarray(rng.uniform(lo,hi,shape),np.float32)
      h={n:rnd((nz,ny,nx+1)) for n in ('u_pp','u0','u')};h.update({n:rnd((nz,ny+1,nx)) for n in ('v_pp','v0','v')});h.update({n:rnd(full) for n in ('w_pp','ph_pp','w0','w','php0','php')});h.update({n:rnd(shape) for n in ('th_pp','al_pp','p_pp','p_pp_old','thp0','thp','h_diabatic')});h.update({n:rnd(plane) for n in ('mup0','mup','mu_pp')});h.update(mub2d=pos(plane,8,10),muts_wrf=pos(plane,4,6),c1h=pos((nz,)),c2h=pos((nz,)),c1f=pos((nz+1,)),c2f=pos((nz+1,)),rdnw=pos((nz,)),msfu=pos((ny,nx+1)),msfv=pos((ny+1,nx)),msft=pos(plane),p=pos(shape,5,8),alt=pos(shape),thb=pos(shape if b3 else (nz,),300,301))
      h['msfv_inv'] = np.nextafter(np.float32(1.0) / h['msfv'], np.float32(np.inf))
      h['thp'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
      state=State(_metal_setup_state=True,_array_module=xp,slots={},_wrf_acoustic_muts_initialized=True,has_msf=mapped)
      for name,value in h.items():
       if name!='muts_wrf':setattr(state,name,xp.asarray(value))
      state.slots[WRF_ACOUSTIC_MUTS_SLOT]=xp.asarray(h['muts_wrf'])
      cfg=SimpleNamespace(open_x=boundary=='open_x',open_y=boundary=='open_y',specified=boundary=='specified',nested=False)
      bx=int(cfg.open_x or cfg.specified);by=int(cfg.open_y or cfg.specified);host={n:v.copy() for n,v in h.items()};arr=lambda x:x.ctypes.data_as(ptr)
      def reference(phase):fun(*[arr(host['msfv_inv'] if n == 'msfv' and phase == 0 else host[n]) for n in names],phase,mapped,bx,by,b3,nz,ny,nx,rk,F(heat),bool(heat))
      reference(0);_prepare_small_step_init_launch(state,cfg,rk_step=rk)()
      for n in ('u_pp','v_pp','w_pp','th_pp','ph_pp','mu_pp','al_pp','p_pp','p_pp_old'):np.testing.assert_array_equal(getattr(state,n).to_numpy().view(np.uint32),host[n].view(np.uint32),err_msg=str(('init',n,b3,rk,boundary,mapped)))
      # Emulate nonzero acoustic increments and a deliberately independent carried MUTS.
      for n in ('u_pp','v_pp','w_pp','th_pp','ph_pp','mu_pp'):
       host[n]=rnd(host[n].shape,.05);getattr(state,n)[...]=xp.asarray(host[n])
      reference(1);_prepare_small_step_finish_launch(state,cfg,heat)()
      for n in ('u','v','w','thp','php','mup'):np.testing.assert_array_equal(getattr(state,n).to_numpy().view(np.uint32),host[n].view(np.uint32),err_msg=str(('finish',n,b3,rk,boundary,mapped,heat)))
      cases+=1
 print('matched',cases,'strict small-step initialization and finish caller cases')
