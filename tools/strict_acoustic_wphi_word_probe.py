"""Compare strict acoustic w/phi and boundary callers with composed CUDA source."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core import constants as c
from woof.core.acoustic import (submit_metal_acoustic_advance_w_phi,
    submit_metal_acoustic_boundary_frame, WRF_ACOUSTIC_MUTS_SLOT)
from woof.core.kernels import module_source
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
 result=original(self,*args,**kwargs)
 if result is not None:result.wait()
 return result
MetalContext._submit_kernel_batch=sync
root=Path(__file__).resolve().parents[1];source=module_source('acoustic')
a=source.index('static __device__ __forceinline__\nvoid diagnose_p_column(');b=source.index('// Sound-speed factor',a);diag=source[a:b]
a=source.index('#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION\nstatic __device__ __forceinline__\nreal wrf_theta_average');b=source.index('// WRF external-mode filter',a);body=source[a:b]
names=('w_pp','ph_pp','rw_t','rph_t','ww_pp','mu_pp','mu_pp_old','th_pp','th_pp_old','thp','thb','php','phb','alt','c2a','a','alpha','gam','mup','u_pp','v_pp','w_ref','ht','rdn','rdnw','fnm','fnp','c1h','c2h','c1f','c2f','mub2d','cqw','p_pp','al_pp','msft','muts_wrf')
pre='''#include <algorithm>\nusing std::min;using std::max;\n#include "%s"\n#include "%s"\n#define __global__\n#define __forceinline__ inline\n#define GPUWM_WRF_EXACT 1\n#define WPHI_MAX_LEV 257\n#define G %sf\n#define I3S(k,j,i,ny,nx) (((size_t)(k)*(ny)+(j))*(nx)+(i))\ntypedef float real;struct dim{unsigned x;};dim blockIdx,blockDim,threadIdx;\nstatic float arn_add(float a,float b){volatile float r=a+b;return r;}\nstatic float arn_sub(float a,float b){volatile float r=a-b;return r;}\nstatic float arn_mul(float a,float b){volatile float r=a*b;return r;}\nstatic float arn_div(float a,float b){volatile float r=a/b;return r;}\n'''%(root/'tools/glibc_trig_flt32_proof/host_shim.hpp',root/'woof/core/kernels/glibc_trig_flt32.cuh',repr(c.G))
args=','.join('real* '+x for x in names)
common=','.join(names[:33])+',moist,'
scalars='p_pp,al_pp,cf1,cf2,cf3,rdx,rdy,dt,eps,damp,depth,bx,by,zone,b3,lid,nz,ny,nx'
frame_common='ph_pp,w_pp,p_pp,al_pp,rph_t,'
frame_suffix='th_pp,mup,mu_pp,mub2d,rmu_t,php,thp,thb,alt,c2a,rdnw,c1h,c2h,c1f,c2f,dt,frame_zone,b3,nz,ny,nx,muts_wrf'
wrapper=('extern "C" void run('+args+',const real* rmu_t,int moist,float cf1,float cf2,float cf3,float rdx,float rdy,float dt,float eps,float damp,float depth,int bx,int by,int zone,int b3,int lid,int nz,int ny,int nx,int mapped,int frame,int table,int frame_zone){blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(ny*nx);++blockIdx.x){if(mapped)advance_w_phi_msf('+common+'msft,'+scalars+');else advance_w_phi('+common+scalars+');}if(frame)for(blockIdx.x=0;blockIdx.x<(unsigned)(ny*nx);++blockIdx.x){if(table)advance_nested_phi_w('+frame_common+'rw_t,'+frame_suffix+');else advance_specified_phi_w('+frame_common+frame_suffix+');}}\n')
class State(SimpleNamespace):
 def scratch(self,shape,slot,dtype=None):
  if slot not in self.slots:self.slots[slot]=self._array_module.asarray(np.full(shape,np.float32(-0.0)))
  return self.slots[slot]
 def existing_scratch(self,slot):return self.slots.get(slot)
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+diag+body+wrapper)
 subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-fno-fast-math','-dynamiclib',str(src),'-o',str(lib)],check=True)
 fun=ct.CDLL(str(lib)).run;ptr=ct.POINTER(ct.c_float);F=ct.c_float;I=ct.c_int;fun.argtypes=[ptr]*(len(names)+1)+[I]+[F]*9+[I]*12
 rng=np.random.default_rng(842);xp=MetalArrayModule();n=0
 for b3 in (0,1):
  for boundary in ('periodic','specified','nested','open_x','open_y'):
   for mapped in (0,1):
    for lid in (0,1):
     nz,ny,nx=4,5,6;shape=(nz,ny,nx);full=(nz+1,ny,nx);plane=(ny,nx)
     def rnd(shape,scale=1):return np.asarray(rng.uniform(-scale,scale,shape),np.float32)
     def pos(shape,lo=.8,hi=1.2):return np.asarray(rng.uniform(lo,hi,shape),np.float32)
     h={name:rnd(full) for name in ('w_pp','ph_pp','rw_t','rph_t','ww_pp','php','w_ref')}
     h.update({name:rnd(shape) for name in ('th_pp','th_pp_old','thp','p_pp','al_pp')});h.update({name:rnd(plane) for name in ('mu_pp','mu_pp_old','mup','mub2d','rmu_t','ht')})
     h.update(thb=pos(shape if b3 else (nz,),300,301),phb=np.broadcast_to(np.arange(nz+1,dtype=np.float32)[:,None,None]*np.float32(9810),full).copy() if b3 else np.arange(nz+1,dtype=np.float32)*np.float32(9810),alt=pos(shape),c2a=pos(shape),a=rnd(full,.01),alpha=pos(full,.9,1.1),gam=rnd(full,.01),u_pp=rnd((nz,ny,nx+1)),v_pp=rnd((nz,ny+1,nx)),rdn=pos((nz,)),rdnw=pos((nz,)),fnm=pos((nz,)),fnp=pos((nz,)),c1h=pos((nz,)),c2h=pos((nz,)),c1f=pos((nz+1,)),c2f=pos((nz+1,)),cqw=pos(full),msft=pos(plane),muts_wrf=pos(plane,4,5))
     h['mub2d']+=np.float32(8);h['mup']+=np.float32(2)
     h['thp'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
     state=State(_metal_setup_state=True,_array_module=xp,slots={},_wrf_acoustic_muts_initialized=True,has_msf=mapped)
     for name,value in h.items():
      if name not in ('mu_pp_old','th_pp_old','muts_wrf','w_ref','a','alpha','gam','c2a','cqw'):setattr(state,name,xp.asarray(value))
     state.w=xp.asarray(h['w_ref']);state.slots['acoustic_mu_pp_old']=xp.asarray(h['mu_pp_old']);state.slots['acoustic_th_pp_old']=xp.asarray(h['th_pp_old']);state.slots[WRF_ACOUSTIC_MUTS_SLOT]=xp.asarray(h['muts_wrf']);state.cf1=np.float32(.6);state.cf2=np.float32(.3);state.cf3=np.float32(.1)
     cfg=SimpleNamespace(nz=nz,ny=ny,nx=nx,dx=1200.,dy=1700.,specified=boundary=='specified',nested=boundary=='nested',relax_w=False,open_x=boundary=='open_x',open_y=boundary=='open_y',top_lid=lid,epssm=.1,damp_opt=3,dampcoef=.2,zdamp=1800.,moist_cq=True,mp_physics=1,spec_zone=1)
     moist=(b3+mapped+lid)%2; coef=tuple(xp.asarray(h[x]) for x in ('c2a','a','alpha','gam')); cqw=xp.asarray(h['cqw']);cq=(state.u_pp,state.v_pp,cqw,bool(moist))
     host={name:value.copy() for name,value in h.items()}; arr=lambda x:x.ctypes.data_as(ptr)
     forced=cfg.specified or cfg.nested;bx=int(cfg.open_x or forced);by=int(cfg.open_y or forced);dt=np.float32(.8);damp=np.float32(dt*np.float32(cfg.dampcoef))
     fun(*[arr(host[x]) for x in names],arr(host['rmu_t']),moist,state.cf1,state.cf2,state.cf3,F(1/cfg.dx),F(1/cfg.dy),dt,F(cfg.epssm),damp,F(cfg.zdamp),bx,by,int(forced),b3,lid,nz,ny,nx,mapped,int(forced),int(cfg.nested),cfg.spec_zone)
     call=submit_metal_acoustic_advance_w_phi(state,cfg,float(dt),coef,cq=cq,_prepare=True);xp.context.submit_prepared_kernel_call(call,label='strict w phi').wait()
     if forced:
      call=submit_metal_acoustic_boundary_frame(state,cfg,float(dt),coef,_prepare=True);xp.context.submit_prepared_kernel_call(call,label='strict w phi frame').wait()
     for name in ('w_pp','ph_pp','p_pp','al_pp'):np.testing.assert_array_equal(getattr(state,name).to_numpy().view(np.uint32),host[name].view(np.uint32),err_msg=str((name,b3,boundary,mapped,lid,moist)))
     n+=1
 print('matched',n,'strict acoustic w/phi and frame caller cases')
