"""Compare strict PD caller outputs with composed CUDA source compiled on host."""
import ctypes as ct
import os,re,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.wrf_exact import ADVECTION_ENABLED
from woof.core.moist import launch_pd_fluxes,launch_pd_renorm_apply
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from tools.pd_wrf_exact.generate import retained,KERNELS
from tools.mp28_metal.parse import def_name
from tools.thompson_real_column_parity.host_backend import parse_kernels
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
 r=original(self,*args,**kwargs)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
items=retained(ADVECTION_ENABLED);param=parse_kernels('\n'.join(items));defs=[];wrappers=[]
for d in items:
 name=def_name(d)
 if name in KERNELS:
  head=d[:d.index('{')].replace('extern "C" __global__','extern "C"').replace('void '+name+'(', 'void host_'+name+'(')
  params=','.join(p.name for p in param[name]);extent='nz' if name=='pd_renorm_apply' else 'nz+1';jextent='ny' if name=='pd_renorm_apply' else 'ny+1';iextent='nx' if name=='pd_renorm_apply' else 'nx+1'
  wrappers.append(head+'{blockDim.x=1;threadIdx.x=0;for(blockIdx.z=0;blockIdx.z<(unsigned)('+extent+');++blockIdx.z)for(blockIdx.y=0;blockIdx.y<(unsigned)('+jextent+');++blockIdx.y)for(blockIdx.x=0;blockIdx.x<(unsigned)('+iextent+');++blockIdx.x)'+name+'('+params+');}\n')
  d=re.sub(r'unsigned int active = __ballot_sync\(0xffffffffu,\s*i < nx && j < ny && k < nz\);','',d)
  d=re.sub(r'int lane = threadIdx.x & 31;\s*real sc_l = __shfl_up_sync\(active, sc, 1\);\s*real sc_r = __shfl_down_sync\(active, sc, 1\);\s*bool have_l = lane > 0 && \(active & \(1u << \(lane - 1\)\)\);\s*bool have_r = lane < 31 && \(active & \(1u << \(lane \+ 1\)\)\);','real sc_l=sc, sc_r=sc; bool have_l=false, have_r=false;',d)
 d=d.replace("if (fo > ph_low)", "if (fo > ph_low && (++limiter_count, true))")
 d=d.replace("if (cr > 1.0f || cr < -1.0f)", "if ((cr > 1.0f || cr < -1.0f) && (++sl_count, true))")
 defs.append(d)
pre='''extern "C" {int limiter_count=0, sl_count=0;}\n#include <stddef.h>\n#include <algorithm>\n#include <cmath>\nusing std::min;using std::max;\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct dims {unsigned x,y,z;};dims blockIdx,blockDim,threadIdx;\nstatic float __fdiv_rn(float a,float b){volatile float v=a/b;return v;}\nstatic double __ddiv_rn(double a,double b){volatile double v=a/b;return v;}\nstatic float __double2float_rn(double a){volatile float v=(float)a;return v;}\n'''
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+'\n'.join(defs)+'\n'+''.join(wrappers));subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 dll=ct.CDLL(str(lib));P=ct.POINTER(ct.c_float);F=ct.c_float;I=ct.c_int
 for n in KERNELS:getattr(dll,'host_'+n).argtypes=[P if p.pointer else F if p.base in ('real','float') else I for p in param[n]]
 rng=np.random.default_rng(520);xp=MetalArrayModule();cases=words=0
 for boundary in (False,True):
  for mapped in (False,True):
   for vorder in (3,5):
    nz,ny,nx=7,8,8;shape=(nz,ny,nx);plane=(ny,nx)
    def rnd(shape,lo=-1,hi=1):return rng.uniform(lo,hi,shape).astype(np.float32)
    h=dict(q=rnd(shape,0,.01),q0=rnd(shape,0,.01),ru=rnd((nz,ny,nx+1),-3,3),rv=rnd((nz,ny+1,nx),-3,3),rw=rnd((nz+1,ny,nx),-75,75),mut=rnd(plane,4,6),mub=rnd(plane,3,5),mup_old=rnd(plane,-.5,.5),c1h=rnd((nz,),.5,1),c2h=rnd((nz,),.5,1),rdnw=rnd((nz,),-2,-1),fnm=rnd((nz,),.4,.6),fnp=rnd((nz,),.4,.6),msft=rnd(plane,.8,1.2),tend=rnd(shape,-.01,.01))
    h['q0'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
    for n,sh in [('fxl',(nz,ny,nx+1)),('fxc',(nz,ny,nx+1)),('fyl',(nz,ny+1,nx)),('fyc',(nz,ny+1,nx)),('fzl',(nz+1,ny,nx)),('fzc',(nz+1,ny,nx))]:h[n]=rnd(sh)
    h['mu_old']=np.add(h['mub'],h['mup_old'],dtype=np.float32)
    state=SimpleNamespace(_metal_setup_state=True,_array_module=xp)
    d={n:xp.asarray(a) for n,a in h.items()};state.c1h=d['c1h'];state.c2h=d['c2h'];state.rdnw=d['rdnw'];state.fnm=d['fnm'];state.fnp=d['fnp'];state.msft=d['msft'];state.mub2d=d['mub'];state.mup0=d['mup_old']
    values=dict(h,dx=1.3,dy=1.7,dt=.2,dx_inv=np.float32(1/1.3),dy_inv=np.float32(1/1.7),nz=nz,ny=ny,nx=nx,has_msf=int(mapped),open_x=int(boundary),open_y=int(boundary),vorder=vorder,tend_out=h['tend'])
    def host(name):
     args=[values[p.name].ctypes.data_as(P) if p.pointer else F(values[p.name]) if p.base in ('real','float') else I(values[p.name]) for p in param[name]]
     getattr(dll,'host_'+name)(*args)
    host('pd_fluxes')
    if vorder==5:host('pd_vertical_sl')
    launch_pd_fluxes(d['q'],d['q0'],d['ru'],d['rv'],d['rw'],d['mut'],state,1.3,1.7,.2,*[d[n] for n in ('fxl','fxc','fyl','fyc','fzl','fzc')],msft=d['msft'],has_msf=mapped,open_x=boundary,open_y=boundary,vorder=vorder)
    for n in ('fxl','fxc','fyl','fyc','fzl','fzc'):
     np.testing.assert_array_equal(d[n].to_numpy().view(np.uint32),h[n].view(np.uint32),err_msg=str(('flux',n,ADVECTION_ENABLED,boundary,mapped,vorder)));words+=h[n].size
    host('pd_renorm_apply');launch_pd_renorm_apply(d['q0'],d['mu_old'],*[d[n] for n in ('fxl','fxc','fyl','fyc','fzl','fzc')],d['tend'],state,1.3,1.7,.2,msft=d['msft'],has_msf=mapped,open_x=boundary,open_y=boundary)
    np.testing.assert_array_equal(d['tend'].to_numpy().view(np.uint32),h['tend'].view(np.uint32),err_msg=str(('renorm',ADVECTION_ENABLED,boundary,mapped,vorder)));words+=h['tend'].size;cases+=1
 limiter=I.in_dll(dll,'limiter_count').value;sl=I.in_dll(dll,'sl_count').value
 assert limiter>0 and sl>0, (limiter,sl)
 print('matched',cases,'strict PD flux/SL and renorm cases,',words,'words; advection control',int(ADVECTION_ENABLED),'limiter calls',limiter,'multi-cell SL faces',sl)
 # Exercise the real final periodic scalar-stage caller with every producer,
 # including the carried acoustic mass, against the same host CUDA kernels.
 from woof.core.moist import advance_scalars_stage
 class Stage(SimpleNamespace):
  def scratch(self,shape,slot):
   if slot not in self.slots:self.slots[slot]=self._array_module.empty(shape,np.float32)
   return self.slots[slot]
  def existing_scratch(self,slot):return self.carried
 nz,ny,nx=7,8,8;shape=(nz,ny,nx);plane=(ny,nx)
 q_initial=h['q'].copy();old=h['q0'].copy();mup=h['mup_old']+np.float32(.25)
 mu0=np.add(h['mub'],h['mup_old'],dtype=np.float32);mu=np.add(h['mub'],mup,dtype=np.float32)
 chm0=h['c1h'][:,None,None]*mu0[None]+h['c2h'][:,None,None]
 chm=h['c1h'][:,None,None]*mu[None]+h['c2h'][:,None,None]
 qeff=((chm0*old)+(np.float32(.2)*np.zeros(shape,np.float32)))/chm0
 h['q0']=qeff.copy();h['mu_old']=mu0;h['tend']=np.zeros(shape,np.float32)
 for n in ('fxl','fxc','fyl','fyc','fzl','fzc'):h[n].fill(0)
 values=dict(h,dx=1.3,dy=1.7,dt=.2,dx_inv=np.float32(1/1.3),dy_inv=np.float32(1/1.7),nz=nz,ny=ny,nx=nx,has_msf=1,open_x=0,open_y=0,vorder=5,tend_out=h['tend'])
 host('pd_fluxes');host('pd_vertical_sl');host('pd_renorm_apply')
 mapped_tend=h['tend']*h['msft'][None]
 expected=((chm0*qeff)+(np.float32(.2)*mapped_tend))/chm
 s=Stage(_metal_setup_state=True,_array_module=xp,slots={},p=xp.empty(shape,np.float32),qv=xp.asarray(q_initial),qv0=xp.asarray(old),qc=None,qr=None,has_msf=True,rqvften=None,mub2d=xp.asarray(h['mub']),mup0=xp.asarray(h['mup_old']),mup=xp.asarray(mup),carried=xp.asarray(h['mut']),c1h=xp.asarray(h['c1h']),c2h=xp.asarray(h['c2h']),rdnw=xp.asarray(h['rdnw']),fnm=xp.asarray(h['fnm']),fnp=xp.asarray(h['fnp']),msft=xp.asarray(h['msft']))
 cfg=SimpleNamespace(open_x=False,open_y=False,specified=False,nested=False,moist_adv_opt=1,scalar_adv_opt=1,cu_physics=0,v_sca_adv_order=5,dx=1.3,dy=1.7)
 advance_scalars_stage(s,cfg,xp.asarray(h['ru']),xp.asarray(h['rv']),xp.asarray(h['rw']),.2,True)
 np.testing.assert_array_equal(s.qv.to_numpy().view(np.uint32),expected.view(np.uint32),err_msg='full periodic strict PD stage')
 for n in ('fxl','fxc','fyl','fyc','fzl','fzc'):
  np.testing.assert_array_equal(s.slots['pd_'+n].to_numpy().view(np.uint32),h[n].view(np.uint32),err_msg='stage '+n)
 print('matched actual periodic strict final scalar stage,',expected.size,'scalar words and six flux arrays')
