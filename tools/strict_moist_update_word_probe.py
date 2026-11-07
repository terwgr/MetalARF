"""Compare actual strict scalar update and time-t mass callers with source words."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
import woof.core.moist as moist
from woof.metal.moist_stage import submit_time_t_mass
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
 result=original(self,*args,**kwargs)
 if result is not None:result.wait()
 return result
MetalContext._submit_kernel_batch=sync
class Capture:
 @staticmethod
 def ElementwiseKernel(params,output,body,*args,**kwargs):return body
original_cp=moist.cp;moist.cp=Capture()
bodies=[moist._update_scalar_kernel.__wrapped__(bool(mapped),False,False,False) for mapped in (0,1)]
moist.cp=original_cp
pre='''#include <stddef.h>\ntypedef float T;\nstatic float __fadd_rn(float a,float b){volatile float v=a+b;return v;}\nstatic float __fmul_rn(float a,float b){volatile float v=a*b;return v;}\nstatic float __fdiv_rn(float a,float b){volatile float v=a/b;return v;}\nextern "C" void mass(float *o,const float*a,const float*b,int n){for(int i=0;i<n;i++)o[i]=__fadd_rn(a[i],b[i]);}\n'''
funcs=[]
for mapped,body in enumerate(bodies):
 funcs.append('extern "C" void run%d(float *out,const float* q0s,const float* tends,const float*c1h,const float*c2h,const float*mu0,const float*mu,const float*msft,float dt_eff,int ncol,int n){for(int i=0;i<n;i++){const float q0=q0s[i],tend=tends[i];float q;\n'%mapped+body+'\nout[i]=q;}}\n')
class State(SimpleNamespace):
 def scratch(self,shape,slot):
  if slot not in self.slots:self.slots[slot]=self._array_module.empty(shape,np.float32)
  return self.slots[slot]
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+''.join(funcs));subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 dll=ct.CDLL(str(lib));P=ct.POINTER(ct.c_float);I=ct.c_int;F=ct.c_float;dll.mass.argtypes=[P]*3+[I]
 for name in ('run0','run1'):getattr(dll,name).argtypes=[P]*8+[F,I,I]
 arr=lambda x:x.ctypes.data_as(P);xp=MetalArrayModule();rng=np.random.default_rng(738);cases=words=0
 for mapped in (0,1):
  for dt in (np.float32(.13),np.float32(-.25)):
   for mode in (0,1):
    nz,ny,nx=4,4,5;shape=(nz,ny,nx);plane=(ny,nx)
    def rand(shape,lo=-1,hi=1):return rng.uniform(lo,hi,shape).astype(np.float32)
    h=dict(q0=rand(shape),tend=rand(shape),c1h=rand((nz,),.6,1.2),c2h=rand((nz,),.5,2),mub2d=rand(plane,4,6),mup0=rand(plane),mu=rand(plane,3,7),msft=rand(plane,.5,1.5))
    bits=np.array([0,0x80000000,1,0x80000001,0x00800000,0x80800000],np.uint32).view(np.float32)
    h['q0'].flat[:6]=bits;h['tend'].flat[:6]=bits
    if mode:h['mub2d'].flat[:4]=bits[:4];h['mup0'].flat[:4]=bits[::-1][:4]
    s=State(_metal_setup_state=True,_array_module=xp,slots={})
    for n,v in h.items():setattr(s,n,xp.asarray(v))
    hostmass=np.empty(plane,np.float32);dll.mass(arr(hostmass),arr(h['mub2d']),arr(h['mup0']),hostmass.size)
    mu0=submit_time_t_mass(s);np.testing.assert_array_equal(mu0.to_numpy().view(np.uint32),hostmass.view(np.uint32))
    out=np.empty(shape,np.float32);getattr(dll,'run%d'%mapped)(arr(out),arr(h['q0']),arr(h['tend']),arr(h['c1h']),arr(h['c2h']),arr(hostmass),arr(h['mu']),arr(h['msft']),dt,ny*nx,out.size)
    q=xp.empty(shape,np.float32);moist._update_scalar_in_place(q,s.q0,s.tend,s.c1h,s.c2h,mu0,s.mu,float(dt),msft=s.msft if mapped else None,clamp=False)
    np.testing.assert_array_equal(q.to_numpy().view(np.uint32),out.view(np.uint32),err_msg=str((mapped,dt,mode)))
    words+=out.size+hostmass.size;cases+=1
 print('matched',cases,'strict scalar-update and time-t-mass caller cases,',words,'words')
# Execute the same source aggregation caller on NumPy and Metal, including
# absent operands and cancellation of signed subnormals.
class Physics:
 def __init__(self,value):self.value=value
 def scalar_for(self,name):return self.value
host_state=SimpleNamespace(_metal_setup_state=False)
metal_state=SimpleNamespace(_metal_setup_state=True)
count=0
for has_physics in (False,True):
 for has_fixed in (False,True):
  shape=(3,4,5)
  physics=np.resize(np.array([0,0x80000000,1,0x80000001,0x3f800000,0xbf800000],np.uint32),shape).view(np.float32)
  fixed=np.resize(np.array([0x80000000,1,0x80000001,1,0xbf800000,0x3f800000],np.uint32),shape).view(np.float32)
  host=np.empty(shape,np.float32);device=xp.empty(shape,np.float32)
  moist._wrf_scalar_source(host_state,'qv',Physics(physics) if has_physics else None,{'qv':fixed} if has_fixed else None,host)
  moist._wrf_scalar_source(metal_state,'qv',Physics(xp.asarray(physics)) if has_physics else None,{'qv':xp.asarray(fixed)} if has_fixed else None,device)
  np.testing.assert_array_equal(device.to_numpy().view(np.uint32),host.view(np.uint32))
  count+=host.size
print('matched four strict source accumulation caller cases,',count,'words')
# Exercise the real strict species caller through its PD fold, while replacing
# the separate limiter with a capture so its carried mass can be checked.
class PDState(State):
 def existing_scratch(self,slot):return self.carried
state=PDState(_metal_setup_state=True,_array_module=xp,slots={},has_msf=False)
shape=(4,4,5);plane=shape[1:];q0=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)*np.float32(.0001)
q0.flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
source=np.resize(np.array([0x80000000,1,0x80000001,0x3a83126f],np.uint32),shape).view(np.float32)
mu0=np.full(plane,4.25,np.float32);mu=np.full(plane,5.25,np.float32);c1=np.array([.5,.8,1.,1.2],np.float32);c2=np.ones(4,np.float32)
state.c1h=xp.asarray(c1);state.c2h=xp.asarray(c2);state.msft=xp.asarray(np.ones(plane,np.float32));state.carried=xp.asarray(np.full(plane,9.0,np.float32))
q=xp.asarray(q0);q0_device=xp.asarray(q0);tend=xp.empty(shape,np.float32);mu0_device=xp.asarray(mu0);mu_device=xp.asarray(mu)
seen={}
def capture(q,q0_eff,ru,rv,ww,mass,*args,**kwargs):
 seen['fold']=q0_eff.to_numpy();seen['mass']=mass
old_flux,old_renorm,old_orders=moist.launch_pd_fluxes,moist.launch_pd_renorm_apply,moist.vertical_orders
moist.launch_pd_fluxes=capture;moist.launch_pd_renorm_apply=lambda *args,**kwargs:None;moist.vertical_orders=lambda cfg:(5,5)
cfg=SimpleNamespace(specified=False,nested=False,dx=1000.,dy=1000.)
moist._advance_species_wrf(state,cfg,'qc',q,q0_device,tend,None,ru=None,rv=None,ww=None,implicit=None,pd=True,bufs=(None,)*6,mu0=mu0_device,mu=mu_device,mu0_row=mu0_device,mu_row=mu_device,chm0=None,dt_eff=.2,final=True,physics_tendencies=Physics(xp.asarray(source)),fixed_tendencies=None,recompute=False,apply_scalar_lbc=None,export_advective_forcing=False,boundary_x=False,boundary_y=False)
moist.launch_pd_fluxes,moist.launch_pd_renorm_apply,moist.vertical_orders=old_flux,old_renorm,old_orders
source_ref=np.zeros(shape,np.float32);source_ref+=source
expected=np.empty(shape,np.float32);dummy=np.ones(plane,np.float32)
dll.run0(arr(expected),arr(q0),arr(source_ref),arr(c1),arr(c2),arr(mu0),arr(mu0),arr(dummy),F(.2),plane[0]*plane[1],expected.size)
np.testing.assert_array_equal(seen['fold'].view(np.uint32),expected.view(np.uint32))
assert seen['mass'] is state.carried
print('matched strict species PD-fold caller,',expected.size,'words; limiter received carried acoustic mass')
# Actual non-PD species caller, with the separate advection producer seeded
# at its output boundary; compare the mapped/source combination and update.
old_advection=moist.launch_flux_div_scalar
adv=np.resize(np.array([0x80000000,1,0x80000001,0x3a83126f,0xba83126f],np.uint32),shape).view(np.float32)
def seed_advection(q,ru,rv,ww,tend,*args,**kwargs):tend[...]=xp.asarray(adv)
moist.launch_flux_div_scalar=seed_advection
for mapped in (False,True):
 state.has_msf=mapped
 map_host=np.resize(np.array([.5,1.,2.,1.3],np.float32),plane)
 state.msft=xp.asarray(map_host)
 q=xp.asarray(q0);tend=xp.empty(shape,np.float32)
 moist._advance_species_wrf(state,cfg,'qc',q,q0_device,tend,None,ru=None,rv=None,ww=None,implicit=None,pd=False,bufs=None,mu0=mu0_device,mu=mu_device,mu0_row=mu0_device,mu_row=mu_device,chm0=None,dt_eff=.2,final=False,physics_tendencies=Physics(xp.asarray(source)),fixed_tendencies=None,recompute=False,apply_scalar_lbc=None,export_advective_forcing=False,boundary_x=False,boundary_y=False)
 combined=adv.copy()
 if mapped:combined*=map_host[None]
 combined+=source_ref
 expected=np.empty(shape,np.float32)
 dll.run0(arr(expected),arr(q0),arr(combined),arr(c1),arr(c2),arr(mu0),arr(mu),arr(dummy),F(.2),plane[0]*plane[1],expected.size)
 np.testing.assert_array_equal(tend.to_numpy().view(np.uint32),combined.view(np.uint32))
 np.testing.assert_array_equal(q.to_numpy().view(np.uint32),expected.view(np.uint32))
moist.launch_flux_div_scalar=old_advection
print('matched two strict non-PD species callers,',2*expected.size*2,'words; advection seeded at producer boundary')
# Capture the pure advective qv rate with the exact time-t mass. GF's
# unmapped branch is a bit-preserving copy; mapped GF uses shared root glue.
for mapped in (False,True):
 state.has_msf=mapped;state.rqvften=xp.empty(shape,np.float32)
 state.msft=xp.asarray(map_host)
 forcing_cfg=SimpleNamespace(cu_physics=0)
 moist._capture_advective_qv_forcing(state,xp.asarray(adv),mu0_device,cfg=forcing_cfg)
 expected=adv/(c1[:,None,None]*mu0[None]+c2[:,None,None])
 if mapped:expected*=map_host[None]
 np.testing.assert_array_equal(state.rqvften.to_numpy().view(np.uint32),expected.view(np.uint32))
for mapped in (False,True):
 state.has_msf=mapped;state.rqvften=xp.empty(shape,np.float32)
 moist._capture_advective_qv_forcing(state,xp.asarray(adv),mu0_device,cfg=SimpleNamespace(cu_physics=3))
 expected=np.multiply(adv,map_host[None],dtype=np.float32) if mapped else adv.copy()
 np.testing.assert_array_equal(state.rqvften.to_numpy().view(np.uint32),expected.view(np.uint32))
print('matched four strict qv forcing capture callers,',4*adv.size,'words; mapped GF uses shared strict glue')
