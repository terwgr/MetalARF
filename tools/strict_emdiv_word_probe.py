"""Compare strict prepared filter and recurrence callers with composed CUDA source."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core.dycore import _prepare_emdiv_filter_launch, _prepare_emdiv_mudf_launch
from woof.core.kernels import module_source
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
 result=original(self,*args,**kwargs)
 if result is not None: result.wait()
 return result
MetalContext._submit_kernel_batch=sync
source=module_source('acoustic')
def extract(name):
 start=source.rindex('extern "C" __global__',0,source.index('void '+name+'(')); pos=source.index('{',start); depth=1; end=pos+1
 while depth:
  if source[end]=='{': depth+=1
  if source[end]=='}': depth-=1
  end+=1
 return source[start:end]
pre='''#include <stddef.h>\n#define __global__\n#define GPUWM_WRF_EXACT 1\n#define I3S(k,j,i,ny,nx) (((size_t)(k)*(ny)+(j))*(nx)+(i))\ntypedef float real;struct dim {unsigned x;};dim blockIdx,blockDim,threadIdx;\nstatic float arn_add(float a,float b){volatile float r=a+b;return r;}\nstatic float arn_sub(float a,float b){volatile float r=a-b;return r;}\nstatic float arn_mul(float a,float b){volatile float r=a*b;return r;}\nstatic float arn_div(float a,float b){volatile float r=a/b;return r;}\nstatic float __fdiv_rn(float a,float b){return arn_div(a,b);}\n'''
wrap='''extern "C" void filter(float*u,float*v,float*d,float*m,float*p,float*c,float*x,float*y,int map,int bx,int by,int forced,int width,int save,int nz,int ny,int nx){blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*(ny+1)*(nx+1));++blockIdx.x)apply_emdiv(u,v,d,m,p,c,x,y,0.0f,0.0f,map,bx,by,forced,width,save,nz,ny,nx);}
extern "C" void recurrence(float*d,float*m,float*p,float dt,int forced,int width,int ny,int nx){blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(ny*nx);++blockIdx.x)update_mudf(d,m,p,dt,forced,forced,width,ny,nx);}
'''
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+extract('apply_emdiv')+extract('update_mudf')+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 dll=ct.CDLL(str(lib));P=ct.POINTER(ct.c_float);I=ct.c_int;F=ct.c_float
 dll.filter.argtypes=[P]*8+[I]*9;dll.recurrence.argtypes=[P]*3+[F]+[I]*4
 arr=lambda a:a.ctypes.data_as(P)
 rng=np.random.default_rng(944);xp=MetalArrayModule();cases=0;words=0;zero_changed=0
 for boundary in ('periodic','open_x','open_y','specified','nested'):
  for mapped in (0,1):
   for save in (0,1):
    for width in (1,2):
     nz,ny,nx=3,5,6;plane=(ny,nx)
     def rand(shape):return rng.uniform(-2,2,shape).astype(np.float32)
     h={'u_pp':rand((nz,ny,nx+1)),'v_pp':rand((nz,ny+1,nx)),'mudf':rand(plane),'mu_pp':rand(plane),'mu_prev':rand(plane),'c1h':rand((nz,)),'msfu':rng.uniform(.5,1.5,(ny,nx+1)).astype(np.float32),'msfv':rng.uniform(.5,1.5,(ny+1,nx)).astype(np.float32)}
     bits=np.array([0,0x80000000,1,0x80000001,0x00800000,0x80800000],np.uint32).view(np.float32)
     for n in ('u_pp','v_pp','mudf','mu_pp','mu_prev'):h[n].flat[:6]=bits
     s=SimpleNamespace(_metal_setup_state=True,_array_module=xp,has_msf=mapped,p=xp.asarray(np.zeros((nz,ny,nx),np.float32)),mup=xp.asarray(np.zeros(plane,np.float32)))
     for n,v in h.items():setattr(s,n,xp.asarray(v))
     cfg=SimpleNamespace(open_x=boundary=='open_x',open_y=boundary=='open_y',specified=boundary=='specified',nested=boundary=='nested',spec_zone=width,emdiv=.12,dx=1200.,dy=1700.)
     forced=int(cfg.specified or cfg.nested);bx=int(cfg.open_x or forced);by=int(cfg.open_y or forced)
     host={n:v.copy() for n,v in h.items()};prev=host['mu_prev'] if save else host['mudf']
     dll.filter(*[arr(host[n]) for n in ('u_pp','v_pp','mudf','mu_pp')],arr(prev),*[arr(host[n]) for n in ('c1h','msfu','msfv')],mapped,bx,by,forced,width,save,nz,ny,nx)
     call=_prepare_emdiv_filter_launch(s,cfg,s.mudf,s.mu_prev if save else None,prepared=True)
     xp.context.submit_prepared_kernel_call(call,label='strict external filter words').wait()
     for n in ('u_pp','v_pp','mu_prev','mudf'):
      np.testing.assert_array_equal(getattr(s,n).to_numpy().view(np.uint32),host[n].view(np.uint32),err_msg=str((n,boundary,mapped,save,width)))
      words+=host[n].size
     zero_changed+=np.count_nonzero((h['u_pp'].view(np.uint32)==0x80000000)&(host['u_pp'].view(np.uint32)==0))
     dt=np.float32(1.3);dll.recurrence(arr(host['mudf']),arr(host['mu_pp']),arr(host['mu_prev']),dt,forced,width,ny,nx)
     _prepare_emdiv_mudf_launch(s,cfg,s.mudf,s.mu_prev,float(dt))()
     np.testing.assert_array_equal(s.mudf.to_numpy().view(np.uint32),host['mudf'].view(np.uint32),err_msg=str(('recurrence',boundary,mapped,save,width)))
     words+=host['mudf'].size;cases+=1
 assert zero_changed>0, 'signed-zero discriminator was not exercised'
 print('matched',cases,'prepared filter and recurrence cases,',words,'words; signed-zero changes',zero_changed)
