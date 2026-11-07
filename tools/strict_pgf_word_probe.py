"""Compare strict Metal CQ and pressure-gradient callers with host CUDA source."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
os.environ.setdefault('WOOF_WRF_EXACT_DIAGNOSTICS','1')
os.environ.setdefault('GPUWM_WRF_EXACT_BIGSTEP','1')
from woof.core.dycore import _launch_slow_pgf
from woof.core.acoustic import prepare_moist_cq
from woof.metal.array import MetalArrayModule
root=Path(__file__).resolve().parents[1]
dy=(root/'woof/core/kernels/dycore.cu').read_text()
ac=(root/'woof/core/kernels/acoustic.cu').read_text()
pg=dy[dy.index('static __device__ __forceinline__\nreal base_value'):dy.index('static __device__ __forceinline__\nreal q_total')]
cq=ac[ac.index('static __device__ __forceinline__\nreal cq_pair'):ac.index('// Damped p\'\' at full level')]
pre='''#include <stddef.h>\n#define __device__\n#define __global__\n#define __forceinline__ inline\n#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_C_BIGSTEP %s\n#define GPUWM_WRF_EXACT_D_DIAGNOSTICS %s\n#define I3S(k,j,i,ny,nx) (((size_t)(k)*(ny)+(j))*(nx)+(i))\ntypedef float real; struct dim { unsigned x; }; dim blockIdx,blockDim,threadIdx;\nstatic float rn_add(float a,float b){volatile float r=a+b;return r;}\nstatic float rn_sub(float a,float b){volatile float r=a-b;return r;}\nstatic float rn_mul(float a,float b){volatile float r=a*b;return r;}\nstatic float rn_div(float a,float b){volatile float r=a/b;return r;}\nstatic float arn_add(float a,float b){return rn_add(a,b);}\n'''%(os.environ['GPUWM_WRF_EXACT_BIGSTEP'],os.environ['WOOF_WRF_EXACT_DIAGNOSTICS'])
wrap='''\nextern "C" void do_cq(const real* a,const real* b,const real* c,const real* d,const real* e,const real* f,const real* g,real* u,real* v,real* w,int m,int nz,int ny,int nx){blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)((nz+1)*(ny+1)*(nx+1));++blockIdx.x)calc_cq(a,b,c,d,e,f,g,u,v,w,m,nz,ny,nx);}\nextern "C" void do_pg(real* u,real* v,const real* p,const real* pb,const real* al,const real* alt,const real* php,const real* phb,const real* mu,const real* mub,const real* c1,const real* c2,const real* rdnw,const real* fnm,const real* fnp,float cf1,float cf2,float cf3,int lid,float cfn,float cfn1,const real* cqu,const real* cqv,int moist,float rdx,float rdy,float hx,float hy,int bx,int by,int b3,int nz,int ny,int nx){blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*(ny+1)*(nx+1));++blockIdx.x)slow_pgf(u,v,p,pb,al,alt,php,phb,mu,mub,c1,c2,rdnw,fnm,fnp,cf1,cf2,cf3,lid,cfn,cfn1,cqu,cqv,moist,rdx,rdy,hx,hy,bx,by,b3,nz,ny,nx);}\n'''
class State(SimpleNamespace):
 def scratch(self,shape,slot,dtype=None):
  if slot not in self.slots: self.slots[slot]=self._array_module.zeros(shape,np.float32)
  return self.slots[slot]
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp'; lib=Path(tmp)/'ref.dylib'; src.write_text(pre+pg+cq+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 dll=ct.CDLL(str(lib)); ptr=ct.POINTER(ct.c_float); F=ct.c_float; I=ct.c_int
 dll.do_cq.argtypes=[ptr]*10+[I]*4
 dll.do_pg.argtypes=[ptr]*15+[F]*3+[I]+[F]*2+[ptr]*2+[I]+[F]*4+[I]*6
 rng=np.random.default_rng(821); xp=MetalArrayModule(); cases=0
 for b3 in (0,1):
  for boundary in (0,1):
   for lid in (0,1):
    for mode in (0,1,3,6,7,4):
     nz,ny,nx=3,2,3; shape=(nz,ny,nx); plane=(ny,nx)
     def rnd(shape,scale=1): return np.asarray(rng.uniform(-scale,scale,shape),np.float32)
     h={n:rnd(shape) for n in ('p','al','alt','qv','qc','qr','qi','qs','qg','qh')}; h.update(pb=rnd(shape if b3 else (nz,)),php=rnd((nz+1,ny,nx)),phb=rnd((nz+1,ny,nx) if b3 else (nz+1,)),mup=rnd(plane),mub2d=rnd(plane),c1h=rnd((nz,)),c2h=rnd((nz,)),rdnw=rnd((nz,)),fnm=rnd((nz,)),fnp=rnd((nz,)),ru_t=rnd((nz,ny,nx+1)),rv_t=rnd((nz,ny+1,nx)))
     h['p'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
     if mode==4: h['qs'].fill(0);h['qg'].fill(0)
     s=State(_metal_setup_state=True,_array_module=xp,slots={})
     for n,v in h.items():setattr(s,n,xp.asarray(v))
     s.p_perturbation=s.p
     if os.environ['WOOF_WRF_EXACT_DIAGNOSTICS']=='1':s.p=xp.asarray(h['p']+np.float32(10))
     if not mode:s.qv=None
     if mode in (0,1,3):s.qi=None;s.qs=None;s.qg=None
     if mode==4:s.qs=None;s.qg=None
     if mode!=7:s.qh=None
     s.cf1=np.float32(.7);s.cf2=np.float32(.2);s.cf3=np.float32(.1);s.cfn=np.float32(1.1);s.cfn1=np.float32(-.1)
     cfg=SimpleNamespace(nz=nz,ny=ny,nx=nx,mp_physics={0:0,1:0,3:1,6:8,7:18,4:50}[mode],moist_cq=True,dx=1200.,dy=1700.,top_lid=lid,open_x=boundary,open_y=boundary,specified=False,nested=False)
     arr=lambda x:x.ctypes.data_as(ptr)
     u=h['ru_t'].copy();v=h['rv_t'].copy(); cu=np.zeros_like(u);cv=np.zeros_like(v);cw=np.zeros((nz+1,ny,nx),np.float32)
     mass={0:1,1:1,3:3,6:6,7:7,4:6}[mode]
     if mode:dll.do_cq(*[arr(h[n]) for n in ('qv','qc','qr','qi','qs','qg','qh')],arr(cu),arr(cv),arr(cw),mass,nz,ny,nx)
     rdx=1/cfg.dx;rdy=1/cfg.dy
     dll.do_pg(*[arr(x) for x in (u,v,h['p'],h['pb'],h['al'],h['alt'],h['php'],h['phb'],h['mup'],h['mub2d'],h['c1h'],h['c2h'],h['rdnw'],h['fnm'],h['fnp'])],s.cf1,s.cf2,s.cf3,lid,s.cfn,s.cfn1,arr(cu),arr(cv),bool(mode),F(rdx),F(rdy),F(.5*rdx),F(.5*rdy),boundary,boundary,b3,nz,ny,nx)
     # Production CQ allocation and caller, including internally produced CQ.
     token=_launch_slow_pgf(s,cfg);token.wait()
     for got,want in ((s.ru_t.to_numpy(),u),(s.rv_t.to_numpy(),v)):
      np.testing.assert_array_equal(got.view(np.uint32),want.view(np.uint32),err_msg=str((b3,boundary,lid,mode)))
     if mode:
      for name,want in (('acoustic_cqu',cu),('acoustic_cqv',cv),('acoustic_cqw',cw)):
       np.testing.assert_array_equal(s.slots[name].to_numpy().view(np.uint32),want.view(np.uint32),err_msg=str((name,b3,boundary,lid,mode)))
     cases+=1
 print('matched',cases,'caller cases; diagnostics='+os.environ['WOOF_WRF_EXACT_DIAGNOSTICS']+' bigstep='+os.environ['GPUWM_WRF_EXACT_BIGSTEP'])
