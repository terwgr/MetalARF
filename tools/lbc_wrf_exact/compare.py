"""Compare strict Metal boundary relaxation caller with composed CUDA words."""
import ctypes as ct
import os,re,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.wrf_exact import BIGSTEP_ENABLED
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.lbc_state import submit_state_specified_relaxation,_perimeter_count
from tools.lbc_wrf_exact.generate import retained
from tools.mp28_metal.parse import def_name
from tools.thompson_real_column_parity.host_backend import parse_kernels
orig=MetalContext._submit_kernel_batch
def sync(self,*a,**k):
 r=orig(self,*a,**k)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
items=retained(BIGSTEP_ENABLED);params=parse_kernels('\n'.join(items))['state_specified_relaxation']
defs='\n'.join(items);kernel=next(x for x in items if def_name(x)=='state_specified_relaxation');head=kernel[:kernel.index('{')].replace('extern "C" __global__','extern "C"').replace('void state_specified_relaxation(', 'void host_relax(');names=','.join(p.name for p in params)
wrap=head+'{blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*frame_count);++blockIdx.x)state_specified_relaxation('+names+');}\n'
pre='''#include <stddef.h>\n#include <algorithm>\nusing std::min;using std::max;\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct dims{unsigned x;};dims blockIdx,blockDim,threadIdx;\nstatic float __fadd_rn(float a,float b){volatile float v=a+b;return v;}\nstatic float __fsub_rn(float a,float b){volatile float v=a-b;return v;}\nstatic float __fmul_rn(float a,float b){volatile float v=a*b;return v;}\nstatic float __fdiv_rn(float a,float b){volatile float v=a/b;return v;}\n'''
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+defs+'\n'+wrap);subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 fun=ct.CDLL(str(lib)).host_relax;P=ct.POINTER(ct.c_float);F=ct.c_float;I=ct.c_int;fun.argtypes=[P if p.pointer else F if p.base in ('real','float') else I for p in params]
 xp=MetalArrayModule();rng=np.random.default_rng(947);cases=words=inverse_differences=0
 for field in ('qv','u','v','theta','w'):
  for mapped in (0,1):
   for sides in (15,5,1):
    nz,mny,mnx=3,(4 if sides==1 else 8),8;ny=mny+(field=='v');nx=mnx+(field=='u');width=4;spec=1;relax=3 if sides==1 else 2;frame_width=min(relax,(min(ny,nx)+1)//2);shape=(nz,ny,nx)
    def rnd(shape,lo=-1,hi=1):return rng.uniform(lo,hi,shape).astype(np.float32)
    host={'field_tend':rnd(shape),'held':rnd(shape),'mub2d':rnd((mny,mnx),5,7),'mup':rnd((mny,mnx)),'source_mup':rnd((mny,mnx)),'source':rnd(shape),'thb':rnd((nz,),299,301),'c1h':rnd((nz,),.6,1.2),'c2h':rnd((nz,),.5,1.5),'c1f':rnd((nz,),.6,1.2),'c2f':rnd((nz,),.5,1.5),'msft':rnd((mny,mnx),.8,1.2),'msfu':rnd((mny,mnx+1),.8,1.2),'msfv':rnd((mny+1,mnx),.8,1.2),'fcx':rnd((width,),.01,.2),'gcx':rnd((width,),.01,.1)}
    host['msfv_inv'] = np.nextafter(np.float32(1.) / host['msfv'], np.float32(np.inf))
    host['field_tend'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
    for n,sh in [('west',(nz,ny,width)),('east',(nz,ny,width)),('south',(nz,width,nx)),('north',(nz,width,nx))]:host[n]=rnd(sh);host[n+'_t']=rnd(sh)
    d={n:xp.asarray(a) for n,a in host.items()};state=SimpleNamespace(_metal_setup_state=True,_array_module=xp,has_msf=mapped,mub2d=d['mub2d'],mup=d['mup'],thb=d['thb'],c1h=d['c1h'],c2h=d['c2h'],c1f=d['c1f'],c2f=d['c2f'],msft=d['msft'],msfu=d['msfu'],msfv=d['msfv'],msfv_inv=d['msfv_inv'])
    target={'theta':'thp'}.get(field,field);setattr(state,target,d['source'])
    boundary=SimpleNamespace(**{n:SimpleNamespace(value=d[n],tendency=d[n+'_t']) for n in ('west','east','south','north')})
    kind={'u':0,'v':1,'theta':2,'qv':5,'w':6}[field];ref=host['field_tend'].copy();values=dict(host,msfv=host['msfv_inv'],field_tend=ref,mup=host['source_mup'],dtbc=np.float32(.13),width=width,spec_zone=spec,relax_zone=relax,apply_relax=1,clear_specified=0,add_held=1,divide_msf=int(field=='theta'),has_msf=mapped,thb_3d=0,kind=kind,nz=nz,ny=ny,nx=nx,frame_count=_perimeter_count(ny,nx,frame_width),relax_sides=sides)
    for n in ('u','v','w','thp','php','scalar'):values[n]=host['source'] if n==(field if field in ('u','v','w') else 'thp' if field=='theta' else 'scalar') else host['field_tend']
    args=[values[p.name].ctypes.data_as(P) if p.pointer else F(values[p.name]) if p.base in ('real','float') else I(values[p.name]) for p in params];fun(*args)
    if field=='v' and mapped:
     oldref=host['field_tend'].copy(); oldvalues=dict(values,field_tend=oldref,msfv=np.float32(1.)/host['msfv'])
     oldargs=[oldvalues[p.name].ctypes.data_as(P) if p.pointer else F(oldvalues[p.name]) if p.base in ('real','float') else I(oldvalues[p.name]) for p in params];fun(*oldargs)
     inverse_differences+=int(np.count_nonzero(ref.view(np.uint32)!=oldref.view(np.uint32)))
    interior=(slice(None),slice(frame_width,ny-frame_width),slice(frame_width,nx-frame_width))
    if field=='theta':ref[interior]/=host['msft'][None,frame_width:ny-frame_width,frame_width:nx-frame_width]
    ref[interior]+=host['held'][interior]
    submit_state_specified_relaxation(state,field,d['field_tend'],boundary,dtbc=.13,width=width,spec_zone=spec,relax_zone=relax,apply_relax=True,fcx=d['fcx'],gcx=d['gcx'],held=d['held'],divide_msf=(field=='theta'),source_field=d['source'],source_mup=d['source_mup'],relax_sides=sides)
    np.testing.assert_array_equal(d['field_tend'].to_numpy().view(np.uint32),ref.view(np.uint32),err_msg=str((field,mapped,sides,BIGSTEP_ENABLED)))
    words+=ref.size;cases+=1
 assert inverse_differences > 0, 'stored inverse fixture does not discriminate recomputation'
 print('matched',cases,'strict LBC relaxation callers,',words,'words; bigstep control',int(BIGSTEP_ENABLED),'inverse differing words',inverse_differences)
 # Actual specified qv stage: retain the real scalar boundary caller and
 # source kernel; seed only the independent advection producer at its output.
 import woof.core.moist as moist
 import woof.ingest.lateral_bc as lbc
 import woof.boundary_fields as contract
 class Stage(SimpleNamespace):
  def scratch(self,shape,slot):
   if slot not in self.slots:self.slots[slot]=xp.empty(shape,np.float32)
   return self.slots[slot]
  def existing_scratch(self,slot):return getattr(self,"carried",None)
 nz,ny,nx=3,8,8;width=4;spec=1;relax=2;shape=(nz,ny,nx);plane=(ny,nx)
 def rnd(sh,lo=-1,hi=1):return rng.uniform(lo,hi,sh).astype(np.float32)
 hs={'qv0':rnd(shape,.001,.01),'qv':rnd(shape,.001,.01),'mub2d':rnd(plane,5,7),'mup':rnd(plane),'mup0':rnd(plane),'c1h':rnd((nz,),.6,1.2),'c2h':rnd((nz,),.5,1.5),'c1f':rnd((nz,),.6,1.2),'c2f':rnd((nz,),.5,1.5),'thb':rnd((nz,),299,301),'msft':rnd(plane,.8,1.2),'msfu':rnd((ny,nx+1),.8,1.2),'msfv':rnd((ny+1,nx),.8,1.2),'fcx':rnd((width,),.01,.2),'gcx':rnd((width,),.01,.1)}
 hs['msfv_inv'] = np.nextafter(np.float32(1.) / hs['msfv'], np.float32(np.inf))
 for n,sh in [('west',(nz,ny,width)),('east',(nz,ny,width)),('south',(nz,width,nx)),('north',(nz,width,nx))]:hs[n]=rnd(sh);hs[n+'_t']=rnd(sh)
 ds={n:xp.asarray(x) for n,x in hs.items()};boundary=SimpleNamespace(**{n:SimpleNamespace(value=ds[n],tendency=ds[n+'_t']) for n in ('west','east','south','north')});interval=SimpleNamespace(fields={'qv':boundary})
 state=Stage(_metal_setup_state=True,_array_module=xp,slots={},lateral_boundaries=object(),has_msf=True,p=xp.empty(shape,np.float32),qc=None,qr=None,rqvften=None,**{n:ds[n] for n in ('qv','qv0','mub2d','mup','mup0','c1h','c2h','c1f','c2f','thb','msft','msfu','msfv','msfv_inv')})
 cfg=SimpleNamespace(open_x=False,open_y=False,specified=True,nested=False,spec_zone=spec,relax_zone=relax,moist_adv_opt=0,scalar_adv_opt=0,cu_physics=0,mp_physics=0,dx=1.,dy=1.)
 adv=np.resize(np.array([0x80000000,1,0x80000001,0x3a83126f],np.uint32),shape).view(np.float32)
 held=np.zeros(shape,np.float32);vals={**hs,'field_tend':held,'held':held,'mup':hs['mup0'],'u':held,'v':held,'w':held,'thp':held,'php':held,'scalar':hs['qv0'],'dtbc':np.float32(.13),'width':width,'spec_zone':spec,'relax_zone':relax,'apply_relax':1,'clear_specified':0,'add_held':0,'divide_msf':0,'has_msf':1,'thb_3d':0,'kind':5,'nz':nz,'ny':ny,'nx':nx,'frame_count':_perimeter_count(ny,nx,relax),'relax_sides':15}
 args=[vals[p.name].ctypes.data_as(P) if p.pointer else F(vals[p.name]) if p.base in ('real','float') else I(vals[p.name]) for p in params];fun(*args)
 base_adv=adv.copy();base_adv[:,:spec,:]=0;base_adv[:,-spec:,:]=0;base_adv[:,spec:-spec,:spec]=0;base_adv[:,spec:-spec,-spec:]=0
 combined=base_adv*hs['msft'][None];combined+=held
 mu0=hs['mub2d']+hs['mup0'];mu=hs['mub2d']+hs['mup'];old=hs['c1h'][:,None,None]*mu0[None]+hs['c2h'][:,None,None];new=hs['c1h'][:,None,None]*mu[None]+hs['c2h'][:,None,None]
 expected=(old*hs['qv0']+np.float32(.2)*combined)/new
 old_active,old_weights,old_mask,old_fields,old_adv=lbc._active_device_interval,lbc._resident_weights,lbc._relax_side_mask,contract.potential_external_scalar_fields,moist.launch_flux_div_scalar
 lbc._active_device_interval=lambda state,cfg:(interval,.13,.2,0.)
 lbc._resident_weights=lambda *a,**k:(ds['fcx'],ds['gcx'])
 lbc._relax_side_mask=lambda boundaries:15
 contract.potential_external_scalar_fields=lambda cfg:('qv',)
 moist.launch_flux_div_scalar=lambda q,ru,rv,ww,tend,*a,**k:tend.__setitem__(Ellipsis,xp.asarray(adv))
 ru=xp.asarray(np.zeros((nz,ny,nx+1),np.float32));rv=xp.asarray(np.zeros((nz,ny+1,nx),np.float32));ww=xp.asarray(np.zeros((nz+1,ny,nx),np.float32))
 moist.advance_scalars_stage(state,cfg,ru,rv,ww,.2,False,apply_relax=True)
 np.testing.assert_array_equal(state.slots['lbc_qv_held'].to_numpy().view(np.uint32),held.view(np.uint32),err_msg='captured source')
 np.testing.assert_array_equal(state.qv.to_numpy().view(np.uint32),expected.view(np.uint32),err_msg='specified qv stage')
 # Later RK stages must retain the same boundary source instead of recapturing.
 moist.advance_scalars_stage(state,cfg,ru,rv,ww,.2,False,apply_relax=False)
 np.testing.assert_array_equal(state.qv.to_numpy().view(np.uint32),expected.view(np.uint32),err_msg='held qv source')
 lbc._active_device_interval,lbc._resident_weights,lbc._relax_side_mask,contract.potential_external_scalar_fields,moist.launch_flux_div_scalar=old_active,old_weights,old_mask,old_fields,old_adv
 print('matched actual specified qv stage capture and held reuse,',2*expected.size+held.size,'words; advection seeded')
 # Final specified stage uses the held source, carried MUTS, all PD producers,
 # ring exclusion and uncoupling; compare against source-compiled PD kernels.
 from tools.pd_wrf_exact.generate import retained as pd_retained,KERNELS as PD_KERNELS
 from woof.wrf_exact import ADVECTION_ENABLED
 pd_items=pd_retained(ADVECTION_ENABLED);pd_params=parse_kernels('\n'.join(pd_items));pd_defs=[];pd_wrap=[]
 for item in pd_items:
  name=def_name(item)
  if name in PD_KERNELS:
   h=item[:item.index('{')].replace('extern "C" __global__','extern "C"').replace('void '+name+'(', 'void host_'+name+'(')
   ns=','.join(p.name for p in pd_params[name]);ex='nx' if name=='pd_renorm_apply' else 'nx+1';ey='ny' if name=='pd_renorm_apply' else 'ny+1';ez='nz' if name=='pd_renorm_apply' else 'nz+1'
   pd_wrap.append(h+'{blockDim.x=1;threadIdx.x=0;for(blockIdx.z=0;blockIdx.z<(unsigned)('+ez+');++blockIdx.z)for(blockIdx.y=0;blockIdx.y<(unsigned)('+ey+');++blockIdx.y)for(blockIdx.x=0;blockIdx.x<(unsigned)('+ex+');++blockIdx.x)'+name+'('+ns+');}\n')
   item=re.sub(r'unsigned int active = __ballot_sync\(0xffffffffu,\s*i < nx && j < ny && k < nz\);','',item)
   item=re.sub(r'int lane = threadIdx.x & 31;\s*real sc_l = __shfl_up_sync\(active, sc, 1\);\s*real sc_r = __shfl_down_sync\(active, sc, 1\);\s*bool have_l = lane > 0 && \(active & \(1u << \(lane - 1\)\)\);\s*bool have_r = lane < 31 && \(active & \(1u << \(lane \+ 1\)\)\);','real sc_l=sc, sc_r=sc; bool have_l=false, have_r=false;',item)
  pd_defs.append(item)
 pd_pre='''#include <stddef.h>\n#include <algorithm>\n#include <cmath>\nusing std::min;using std::max;\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct dims{unsigned x,y,z;};dims blockIdx,blockDim,threadIdx;\nstatic float __fdiv_rn(float a,float b){volatile float v=a/b;return v;}\nstatic double __ddiv_rn(double a,double b){volatile double v=a/b;return v;}\nstatic float __double2float_rn(double a){volatile float v=(float)a;return v;}\n'''
 pd_src=Path(tmp)/'pd.cpp';pd_lib=Path(tmp)/'pd.dylib';pd_src.write_text(pd_pre+'\n'.join(pd_defs)+'\n'+''.join(pd_wrap));subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(pd_src),'-o',str(pd_lib)],check=True);pd_dll=ct.CDLL(str(pd_lib))
 for n in PD_KERNELS:getattr(pd_dll,'host_'+n).argtypes=[P if p.pointer else F if p.base in ('real','float') else I for p in pd_params[n]]
 carried=rnd(plane,3,7);rdnw=rnd((nz,),-2,-1);fnm=rnd((nz,),.4,.6);fnp=rnd((nz,),.4,.6);u=rnd((nz,ny,nx+1),-2,2);v=rnd((nz,ny+1,nx),-2,2);w=rnd((nz+1,ny,nx),-30,30)
 state.carried=xp.asarray(carried);state.rdnw=xp.asarray(rdnw);state.fnm=xp.asarray(fnm);state.fnp=xp.asarray(fnp);cfg.moist_adv_opt=1;cfg.v_sca_adv_order=3
 qprev=state.qv.to_numpy();qeff=((old*hs['qv0'])+(np.float32(.2)*held))/old
 pp={'q':qprev,'q0':qeff,'ru':u,'rv':v,'rw':w,'mut':carried,'mu_old':mu0,'mub':hs['mub2d'],'mup_old':hs['mup0'],'c1h':hs['c1h'],'c2h':hs['c2h'],'rdnw':rdnw,'fnm':fnm,'fnp':fnp,'msft':hs['msft'],'tend_out':np.zeros(shape,np.float32),'dx':1.,'dy':1.,'dx_inv':1.,'dy_inv':1.,'dt':.2,'nz':nz,'ny':ny,'nx':nx,'has_msf':1,'open_x':1,'open_y':1,'vorder':3}
 for n,sh in [('fxl',(nz,ny,nx+1)),('fxc',(nz,ny,nx+1)),('fyl',(nz,ny+1,nx)),('fyc',(nz,ny+1,nx)),('fzl',(nz+1,ny,nx)),('fzc',(nz+1,ny,nx))]:pp[n]=np.zeros(sh,np.float32)
 def pd_host(name):
  args=[pp[p.name].ctypes.data_as(P) if p.pointer else F(pp[p.name]) if p.base in ('real','float') else I(pp[p.name]) for p in pd_params[name]]
  getattr(pd_dll,'host_'+name)(*args)
 pd_host('pd_fluxes');pd_host('pd_renorm_apply')
 pt=pp['tend_out'];pt[:,:spec,:]=0;pt[:,-spec:,:]=0;pt[:,spec:-spec,:spec]=0;pt[:,spec:-spec,-spec:]=0
 final_expected=(old*qeff+np.float32(.2)*(pt*hs['msft'][None]))/new
 lbc._active_device_interval=lambda state,cfg:(interval,.13,.2,0.);lbc._resident_weights=lambda *a,**k:(ds['fcx'],ds['gcx']);lbc._relax_side_mask=lambda boundaries:15;contract.potential_external_scalar_fields=lambda cfg:('qv',)
 moist.advance_scalars_stage(state,cfg,xp.asarray(u),xp.asarray(v),xp.asarray(w),.2,True,apply_relax=False)
 np.testing.assert_array_equal(state.qv.to_numpy().view(np.uint32),final_expected.view(np.uint32),err_msg='specified final PD stage')
 for n in ('fxl','fxc','fyl','fyc','fzl','fzc'):np.testing.assert_array_equal(state.slots['pd_'+n].to_numpy().view(np.uint32),pp[n].view(np.uint32),err_msg='specified PD '+n)
 lbc._active_device_interval,lbc._resident_weights,lbc._relax_side_mask,contract.potential_external_scalar_fields=old_active,old_weights,old_mask,old_fields
 print('matched actual specified final qv PD stage,',final_expected.size,'scalar words and six flux arrays; advection control',int(ADVECTION_ENABLED))
 # Nested scalar source is recomputed from the saved scalar and time-t mass.
 cfg.specified=False;cfg.nested=True;cfg.moist_adv_opt=0
 lbc._active_device_interval=lambda state,cfg:(interval,.13,.2,0.);lbc._resident_weights=lambda *a,**k:(ds['fcx'],ds['gcx']);lbc._relax_side_mask=lambda boundaries:15
 moist.launch_flux_div_scalar=lambda q,ru,rv,ww,tend,*a,**k:tend.__setitem__(Ellipsis,xp.asarray(adv))
 moist.advance_scalars_stage(state,cfg,ru,rv,ww,.2,False,apply_relax=False)
 np.testing.assert_array_equal(state.qv.to_numpy().view(np.uint32),expected.view(np.uint32),err_msg='nested time-t source')
 lbc._active_device_interval,lbc._resident_weights,lbc._relax_side_mask,moist.launch_flux_div_scalar=old_active,old_weights,old_mask,old_adv
 print('matched actual nested qv stage,',expected.size,'scalar words; advection seeded')
 # A supplied hydrometeor recomputes its boundary source rather than using
 # qv's held slot, and is excluded from flow-dependent overwrite.
 cfg.specified=True;cfg.nested=False;state.qc=xp.asarray(hs['qv']);state.qc0=xp.asarray(hs['qv0']);interval.fields['qc']=boundary
 lbc._active_device_interval=lambda state,cfg:(interval,.13,.2,0.);lbc._resident_weights=lambda *a,**k:(ds['fcx'],ds['gcx']);lbc._relax_side_mask=lambda boundaries:15;contract.potential_external_scalar_fields=lambda cfg:('qv',)
 moist.launch_flux_div_scalar=lambda q,ru,rv,ww,tend,*a,**k:tend.__setitem__(Ellipsis,xp.asarray(adv))
 moist.advance_scalars_stage(state,cfg,ru,rv,ww,.2,False,apply_relax=False)
 np.testing.assert_array_equal(state.qc.to_numpy().view(np.uint32),expected.view(np.uint32),err_msg='specified hydrometeor source')
 lbc._active_device_interval,lbc._resident_weights,lbc._relax_side_mask,contract.potential_external_scalar_fields,moist.launch_flux_div_scalar=old_active,old_weights,old_mask,old_fields,old_adv
 print('matched actual specified hydrometeor recompute,',expected.size,'scalar words; advection seeded')
