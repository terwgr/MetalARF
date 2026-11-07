"""Compare strict Metal geopotential callers with host-compiled CUDA source."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
os.environ.setdefault('GPUWM_WRF_EXACT_BIGSTEP','1')
from woof.core.dycore import (_launch_slow_geopotential, _launch_slow_geopotential_faces,
                              _launch_slow_geopotential_vertical)
from woof.metal.array import MetalArrayModule
root=Path(__file__).resolve().parents[1]
s=(root/'woof/core/kernels/dycore.cu').read_text()
base=s[s.index('static __device__ __forceinline__\nreal base_value'):s.index('static __device__ __forceinline__\nreal pp_value')]
body=s[s.index('static __device__ __forceinline__\nreal ph_total'):s.index('// THE FACE-MASS NEIGHBOUR')]
pos=body.index('void slow_geopotential_faces'); start=body.index('{',pos); depth=1; i=start+1
while depth:
 if body[i]=='{':depth+=1
 if body[i]=='}':depth-=1
 i+=1
body=body[:i]
pre='''#include <stddef.h>\n#include <cmath>\n#include <algorithm>\nusing std::min;\n#define __device__\n#define __global__\n#define __forceinline__ inline\n#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_C_BIGSTEP %s\n#define I3S(k,j,i,ny,nx) (((size_t)(k)*(ny)+(j))*(nx)+(i))\ntypedef float real; const real G=9.81f; struct dim { unsigned x; }; dim blockIdx,blockDim,threadIdx;\nstatic float rn_add(float a,float b){volatile float r=a+b;return r;}\nstatic float rn_sub(float a,float b){volatile float r=a-b;return r;}\nstatic float rn_mul(float a,float b){volatile float r=a*b;return r;}\nstatic float rn_div(float a,float b){volatile float r=a/b;return r;}\n#define __fdiv_rn rn_div\n'''%os.environ['GPUWM_WRF_EXACT_BIGSTEP']
wrap='''\nextern "C" void run(real* t,const real* ww,const real* w,const real* u,const real* v,const real* php,const real* phb,const real* mu,const real* mub,const real* rdnw,const real* fnm,const real* fnp,const real* c1,const real* c2,const real* msft,const real* msfu,const real* msfv,const real* mux,const real* muy,float qx,float qy,float cfn,float cfn1,int mode,int order,int add,int mapped,int bx,int by,int ox,int oy,int specified,int b3,int nz,int ny,int nx) {blockDim.x=1;threadIdx.x=0; for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*ny*nx);++blockIdx.x){ if(mode==2)slow_geopotential_vertical(t,ww,w,php,phb,mu,mub,rdnw,fnm,fnp,c1,c2,msft,mapped,b3,nz,ny,nx); else if((ox||oy)&&!specified)slow_geopotential_open(t,ww,w,u,v,php,phb,mu,mub,rdnw,fnm,fnp,c1,c2,cfn,cfn1,msft,msfu,msfv,mux,muy,qx,qy,mapped,ox,oy,mode==1,add,order,b3,nz,ny,nx); else if(mode==1)slow_geopotential_faces(t,u,v,php,phb,mux,muy,c1,c2,cfn,cfn1,msft,msfu,msfv,qx,qy,mapped,bx,by,specified,order,b3,nz,ny,nx); else slow_geopotential(t,ww,w,u,v,php,phb,mu,mub,rdnw,fnm,fnp,c1,c2,cfn,cfn1,msft,msfu,msfv,qx,qy,mapped,bx,by,specified,order,add,b3,nz,ny,nx); }}\n'''
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+base+body+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 fun=ct.CDLL(str(lib)).run; ptr=ct.POINTER(ct.c_float); F=ct.c_float; I=ct.c_int; fun.argtypes=[ptr]*19+[F]*4+[I]*13
 subnormal_only=os.environ.get('WOOF_GEO_SUBNORMAL')=='1'
 rng=np.random.default_rng(996);xp=MetalArrayModule(); n=0; retained=0
 # periodic, specified, radiative x, radiative y, mixed open and forced
 for mode in ((0,1) if subnormal_only else (0,1,2)):
  for boundary in (('periodic','specified') if subnormal_only else ('periodic','specified','open_x','open_y')):
   if mode==2 and boundary!='periodic':continue
   for order in ((5,) if subnormal_only else (2,5)):
    if mode==2 and order==5:continue
    for b3 in (0,1):
     for mapped in (0,1):
      nz,ny,nx=3,8,8; shape=(nz,ny,nx); full=(nz+1,ny,nx); plane=(ny,nx)
      def rnd(shape):return np.asarray(rng.uniform(-2,2,shape),np.float32)
      h={'rph_t':rnd(full),'ww':rnd(full),'w':rnd(full),'u':rnd((nz,ny,nx+1)),'v':rnd((nz,ny+1,nx)),'php':rnd(full),'phb':rnd(full if b3 else (nz+1,)),'mup':rnd(plane),'mub2d':rnd(plane),'rdnw':rnd((nz,)),'fnm':rnd((nz,)),'fnp':rnd((nz,)),'c1f':rnd((nz+1,)),'c2f':rnd((nz+1,)),'msft':np.asarray(rng.uniform(.8,1.2,plane),np.float32),'msfu':rnd((ny,nx+1)),'msfv':rnd((ny+1,nx)),'mux':rnd((ny,nx+1)),'muy':rnd((ny+1,nx))}
      if subnormal_only:
       h['php']=(rng.uniform(-1,1,full)*np.float64(2**-127)).astype(np.float32)
       h['phb'].fill(0);h['rph_t'].fill(0);h['ww'].fill(0);h['w'].fill(0)
      else:
       h['php'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
      state=SimpleNamespace(_metal_setup_state=True,_array_module=xp,has_msf=mapped,cfn=np.float32(.7),cfn1=np.float32(.3),p=xp.asarray(np.zeros(shape,np.float32)))
      for name,data in h.items():setattr(state,name,xp.asarray(data))
      cfg=SimpleNamespace(h_sca_adv_order=order,open_x=boundary=='open_x',open_y=boundary=='open_y',specified=boundary=='specified',nested=False,dx=1200.,dy=1700.)
      expected=h['rph_t'].copy(); add=mode in (0,2); names=('ww','w','u','v','php','phb','mup','mub2d','rdnw','fnm','fnp','c1f','c2f','msft','msfu','msfv','mux','muy'); arr=lambda x:x.ctypes.data_as(ptr)
      openpath=cfg.open_x or cfg.open_y; qx=.25/cfg.dx if openpath else .25*(1/cfg.dx); qy=.25/cfg.dy if openpath else .25*(1/cfg.dy)
      fun(arr(expected),*[arr(h[x]) for x in names],F(qx),F(qy),F(.7),F(.3),mode,order,add,mapped,int(cfg.open_x or cfg.specified),int(cfg.open_y or cfg.specified),int(cfg.open_x),int(cfg.open_y),int(cfg.specified),b3,nz,ny,nx)
      if mode==0:token=_launch_slow_geopotential(state,cfg,state.ww,add_vertical=True)
      elif mode==1:token=_launch_slow_geopotential_faces(state,cfg,state.mux,state.muy)
      else:token=_launch_slow_geopotential_vertical(state,state.ww)
      token.wait();actual=state.rph_t.to_numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32),err_msg=str((mode,boundary,order,b3,mapped)))
      bits=expected.view(np.uint32)&np.uint32(0x7fffffff)
      retained+=int(np.count_nonzero((bits>0)&(bits<0x00800000)))
      n+=1
 if subnormal_only and not retained:raise AssertionError('no subnormal outputs retained')
 print('matched',n,'geopotential caller cases; retained subnormal words',retained,'bigstep='+os.environ['GPUWM_WRF_EXACT_BIGSTEP'])
