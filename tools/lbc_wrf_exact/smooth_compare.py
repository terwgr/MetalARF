import ctypes as ct,os,subprocess,tempfile
from types import SimpleNamespace
from pathlib import Path
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core.kernels import module_source
from tools.mp28_metal.parse import closure,split_top,def_name
from tools.thompson_real_column_parity.host_backend import parse_kernels
from woof.core.nest_interp import register_nest,smoother,smoother_parent_window,feedback_parent_bounds
from woof.core.nest import NestCoupler
from woof.core.nest_operands import NestWindowSource
from woof.core.streamed_state import CanonicalStoreState
from woof.metal.nest_family import submit_strict_feedback_clamp
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
orig=MetalContext._submit_kernel_batch
def sync(self,*a,**kw):
 r=orig(self,*a,**kw)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
pre=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=module_source('nest'),text=True,capture_output=True,check=True).stdout
names=('nest_smooth_j','nest_smooth_i','nest_copy_fcn');items=closure(split_top(pre),names)[0];params=parse_kernels('\n'.join(items));wrappers=[]
for name in names:
 kernel=next(x for x in items if def_name(x)==name);head=kernel[:kernel.index('{')].replace('extern "C" __global__','extern "C"').replace('void '+name+'(', 'void host_'+name+'(');args=','.join(p.name for p in params[name]);count='nz*(niw+2)*(njw+2)' if name.endswith('_j') else 'nz*niw*njw' if name.endswith('_i') else 'nz*loop_ni*loop_nj';wrappers.append(head+'{blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)('+count+');++blockIdx.x)'+name+'('+args+');}\n')
pre='''#include <stddef.h>\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct D{unsigned x;};D blockIdx,blockDim,threadIdx;\n'''
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';p.write_text(pre+'\n'.join(items)+'\n'+''.join(wrappers));subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(p),'-o',str(lib)],check=True);dll=ct.CDLL(str(lib));P=ct.POINTER(ct.c_float)
 for name in names:getattr(dll,'host_'+name).argtypes=[P if p.pointer else ct.c_float if p.base=='float' else ct.c_int for p in params[name]]
 xp=MetalArrayModule();rng=np.random.default_rng(449);cases=words=0
 for ratio in (2,3):
  for stagger in ('','x','y'):
   reg=register_nest(nri=ratio,nrj=ratio,i_parent_start=4,j_parent_start=4,child_nx=24,child_ny=24,parent_nx=24,parent_ny=24,stagger=stagger,wrapper='interp');i0,j0,niw,njw=smoother_parent_window(reg)
   for option in (1,2):
    shape=(2,reg.nyp,reg.nxp);a=rng.uniform(-1,1,shape).astype('f');a[:,j0,i0:i0+4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view('f');ref=a.copy();scratch=np.zeros_like(a);vals=dict(cfld=ref,scr=scratch,mode=0 if option==1 else 1,i0=i0,j0=j0,niw=niw,njw=njw,nz=2,nyp=reg.nyp,nxp=reg.nxp)
    for xnu in ((0.,) if option==1 else (np.float32(.5),np.float32(-.52))):
     vals['xnu']=xnu
     for name in names[:2]:
      args=[vals[p.name].ctypes.data_as(P) if p.pointer else ct.c_float(vals[p.name]) if p.base=='float' else ct.c_int(vals[p.name]) for p in params[name]];getattr(dll,'host_'+name)(*args)
    d=xp.asarray(a);ds=xp.empty(shape,np.float32);smoother(d,reg,smooth_option=option,scratch=ds);got=d.to_numpy();np.testing.assert_array_equal(got.view('u4'),ref.view('u4'),err_msg=str((ratio,stagger,option)));cases+=1;words+=ref.size
    backing=xp.empty((a.size+37,),np.float32);template=SimpleNamespace(_metal_setup_state=True,_array_module=xp,qv=d);cfg=SimpleNamespace(nx=reg.nxp,ny=reg.nyp,nz=2);store={'state/qv':d,'scratch/nest_parent_field':backing};state=CanonicalStoreState(template,cfg,store=store,geography={},scalars={},inventory={'state/qv':d},geography_inventory={});owner=SimpleNamespace(store=store,template_state=template,_geography={});state._streamed_domain=owner;source=NestWindowSource(state);field=source.array('qv');slot_owner=SimpleNamespace(child_node=SimpleNamespace(state=state),slot_shapes={'nest_parent_field':backing.shape},slot_dtypes={'nest_parent_field':np.float32});scratch_slot=NestCoupler._scratch(slot_owner,'nest_parent_field');assert field is d and scratch_slot is backing and scratch_slot.buffer.context is field.buffer.context and scratch_slot.size >= field.size;d[...] = a;smoother(field,reg,smooth_option=option,scratch=scratch_slot);np.testing.assert_array_equal(d.to_numpy().view('u4'),ref.view('u4'),err_msg=str(('canonical',ratio,stagger,option)))
 for ratio in (2,3):
  reg=register_nest(nri=ratio,nrj=ratio,i_parent_start=4,j_parent_start=4,child_nx=24,child_ny=24,parent_nx=24,parent_ny=24,stagger='',wrapper='interp');i0,j0,niw,njw=smoother_parent_window(reg);ilo,ihi,jlo,jhi=feedback_parent_bounds(reg,spec_zone=1)
  for option in (1,2):
   parent0={k:rng.uniform(-1,1,(reg.nyp,reg.nxp)).astype('f') for k in ('mup','qv')};child0={k:rng.uniform(-1,1,(reg.nyc,reg.nxc)).astype('f') for k in ('mup','qv')};refs={k:v.copy() for k,v in parent0.items()}
   for kind in ('mup','qv'):
    vals=dict(cfld=refs[kind],nfld=child0[kind],ipos=4,jpos=4,nri=ratio,nrj=ratio,spec_zone=1,xstag=0,ystag=0,nz=1,nyp=reg.nyp,nxp=reg.nxp,nyc=reg.nyc,nxc=reg.nxc,parent_i0=0,parent_j0=0,child_i0=0,child_j0=0,loop_i0=ilo,loop_j0=jlo,loop_ni=ihi-ilo+1,loop_nj=jhi-jlo+1)
    def call(name,values):
     args=[values[p.name].ctypes.data_as(P) if p.pointer else ct.c_float(values[p.name]) if p.base=='float' else ct.c_int(values[p.name]) for p in params[name]];getattr(dll,'host_'+name)(*args)
    call('nest_copy_fcn',vals);scratch=np.zeros_like(refs[kind]);sv=dict(cfld=refs[kind],scr=scratch,mode=0 if option==1 else 1,i0=i0,j0=j0,niw=niw,njw=njw,nz=1,nyp=reg.nyp,nxp=reg.nxp)
    for xnu in ((0.,) if option==1 else (np.float32(.5),np.float32(-.52))):
     sv['xnu']=xnu
     for name in names[:2]:call(name,sv)
    if kind=='qv':
     rect=refs[kind][j0:j0+njw,i0:i0+niw];np.maximum(rect,np.float32(0),out=rect)
   parent=SimpleNamespace(_metal_setup_state=True,_array_module=xp,mup=xp.asarray(parent0['mup']),qv=xp.asarray(parent0['qv'][None]));child=SimpleNamespace(_metal_setup_state=True,_array_module=xp,mup=xp.asarray(child0['mup']),qv=xp.asarray(child0['qv'][None]));node=SimpleNamespace(state=child,parent=SimpleNamespace(state=parent),clock=SimpleNamespace(ticks=7),cfg=SimpleNamespace(run=SimpleNamespace(spec_zone=1)));scratch=xp.empty((reg.nyp*reg.nxp,),np.float32)
   coupler=SimpleNamespace(child_node=node,registrations={'m':reg},_transaction_chunk_shape=lambda:(6,6),feedback=1,_prepared_feedback={'ticks':7,'kinds':['mu','qv']},_bind_geometry=lambda:None,_feedback_child_is_bounded=lambda:False,microphysics_reverse_transition=None,_reverse_diagnoses=lambda kind:False,smooth_option=option,feedback_sync_bytes=0,feedback_host_scratch_bytes=0,feedback_count=0,_scratch=lambda name:scratch)
   coupler._restrict_windowed=lambda *a,**kw:NestCoupler._restrict_windowed(coupler,*a,**kw);NestCoupler.feedback_commit(coupler,node)
   for kind in ('mup','qv'):np.testing.assert_array_equal(getattr(parent,kind).to_numpy().reshape(reg.nyp,reg.nxp).view('u4'),refs[kind].view('u4'),err_msg=str((ratio,option,kind)))
   def canonical(mup,qv,backing=None):
    md=xp.asarray(mup);qd=xp.asarray(qv[None]);template=SimpleNamespace(_metal_setup_state=True,_array_module=xp,mup=md,qv=qd);store={'state/mup':md,'state/qv':qd};inventory={'state/mup':md,'state/qv':qd}
    if backing is not None:store['scratch/nest_parent_field']=backing
    cfg=SimpleNamespace(nx=mup.shape[1],ny=mup.shape[0],nz=1);state=CanonicalStoreState(template,cfg,store=store,geography={},scalars={},inventory=inventory,geography_inventory={});state._streamed_domain=SimpleNamespace(store=store,template_state=template,_geography={});return state,store
   parent,pstore=canonical(parent0['mup'],parent0['qv']);backing=xp.empty((reg.nyp*reg.nxp+37,),np.float32);child,cstore=canonical(child0['mup'],child0['qv'],backing);node.state=child;node.parent.state=parent;coupler._feedback_child_is_bounded=lambda:True;coupler.slot_shapes={'nest_parent_field':backing.shape};coupler.slot_dtypes={'nest_parent_field':np.float32};coupler._scratch=lambda name:NestCoupler._scratch(coupler,name);coupler.feedback_count=0
   NestCoupler.feedback_commit(coupler,node)
   for kind in ('mup','qv'):np.testing.assert_array_equal(pstore['state/'+kind].to_numpy().reshape(reg.nyp,reg.nxp).view('u4'),refs[kind].view('u4'),err_msg=str(('canonical feedback',ratio,option,kind)))
   assert coupler.feedback_count==1 and child.existing_scratch('nest_parent_field') is backing
 print('matched 4 resident and 4 canonical device-store feedback transactions with both smoothers and qv clamp')
 a=np.array([0,0x80000000,1,0x80000001,0xbf800000,0x7fc12345,0x7f800000],np.uint32).view('f').reshape(1,1,7);d=xp.asarray(a);submit_strict_feedback_clamp(d.buffer.context,parent=d,i0=0,j0=0,niw=7,njw=1);np.testing.assert_array_equal(d.to_numpy().view('u4'),np.maximum(a,np.float32(0)).view('u4'));print('matched',cases,'smoother callers,',words,'words; clamp signed/NaN words matched')
