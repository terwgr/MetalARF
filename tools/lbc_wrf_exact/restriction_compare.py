import ctypes as ct, os, subprocess, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core.kernels import module_source
from tools.mp28_metal.parse import closure,split_top
from woof.core.nest import NestCoupler
from woof.core.nest_operands import NestWindowSource
from woof.core.nest_interp import feedback_parent_bounds
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
orig=MetalContext._submit_kernel_batch
def sync(self,*a,**kw):
 r=orig(self,*a,**kw)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
src=module_source('nest'); pre=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=src,text=True,capture_output=True,check=True).stdout
items=closure(split_top(pre),('nest_copy_fcn',))[0]
pre='''#include <stddef.h>\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct D{unsigned x;}; D blockIdx,blockDim,threadIdx;\n'''
wrapper='''extern "C" void host_copy(float* a,const float* b,int ip,int jp,int ri,int rj,int sz,int xs,int ys,int nz,int ny,int nx,int cy,int cx,int ilo,int jlo,int ni,int nj){blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*ni*nj);++blockIdx.x)nest_copy_fcn(a,b,ip,jp,ri,rj,sz,xs,ys,nz,ny,nx,cy,cx,0,0,0,0,ilo,jlo,ni,nj);}\n'''
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';p.write_text(pre+'\n'.join(items)+'\n'+wrapper);subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(p),'-o',str(lib)],check=True);dll=ct.CDLL(str(lib));P=ct.POINTER(ct.c_float);dll.host_copy.argtypes=[P,P]+[ct.c_int]*16
 xp=MetalArrayModule();rng=np.random.default_rng(31);cases=words=0
 for ratio in (2,3):
  for field in ('mu','u','v','t','qv'):
   for b3 in (0,1):
    nz=1 if field=='mu' else 2; ny,nx=8+(field=='v'),8+(field=='u');cy,cx=15+(field=='v'),15+(field=='u'); shape=(nz,ny,nx);childshape=(nz,cy,cx)
    target=rng.uniform(-1,1,shape).astype('f'); child=rng.uniform(-1,1,childshape).astype('f');child[:,3,3:7]=np.array([0,0x80000000,1,0x80000001],np.uint32).view('f');pt=rng.uniform(295,305,(nz,ny,nx) if b3 else (nz,)).astype('f');cth=rng.uniform(295,305,(nz,cy,cx) if b3 else (nz,)).astype('f')
    reg=SimpleNamespace(nri=ratio,nrj=ratio,i_parent_start=2,j_parent_start=2,nxc=cx,nyc=cy,nxp=nx,nyp=ny,xstag=field=='u',ystag=field=='v');ilo,ihi,jlo,jhi=feedback_parent_bounds(reg,spec_zone=1)
    ref=target.copy();inp=child.copy()
    if field=='t': inp=np.subtract(np.add(inp,cth if b3 else cth[:,None,None],dtype=np.float32),np.float32(300),dtype=np.float32)
    dll.host_copy(ref.ctypes.data_as(P),inp.ctypes.data_as(P),2,2,ratio,ratio,1,int(reg.xstag),int(reg.ystag),nz,ny,nx,cy,cx,ilo,jlo,ihi-ilo+1,jhi-jlo+1)
    if field=='t':
     sl=(slice(None),slice(jlo,jhi+1),slice(ilo,ihi+1));base=pt[sl] if b3 else pt[:,None,None];ref[sl]=np.subtract(np.add(ref[sl],np.float32(300),dtype=np.float32),base,dtype=np.float32)
    attr={'mu':'mup','t':'thp'}.get(field,field);parent=SimpleNamespace(_metal_setup_state=True,_array_module=xp,thb=xp.asarray(pt));cstate=SimpleNamespace(_metal_setup_state=True,_array_module=xp,thb=xp.asarray(cth));setattr(parent,attr,xp.asarray(target));setattr(cstate,attr,xp.asarray(child));node=SimpleNamespace(cfg=SimpleNamespace(run=SimpleNamespace(spec_zone=1)));coupler=SimpleNamespace(child_node=node,registrations={('x' if field=='u' else 'y' if field=='v' else 'm'):reg},_transaction_chunk_shape=lambda:(6,6))
    NestCoupler._restrict_windowed(coupler,field,NestWindowSource(cstate),NestWindowSource(parent));got=getattr(parent,attr).to_numpy();
    np.testing.assert_array_equal(got.view('u4'),ref.view('u4'),err_msg=str((ratio,field,b3)));cases+=1;words+=got.size
    if field == 't':
     mu=rng.uniform(-1,1,(ny,nx)).astype('f'); cmu=rng.uniform(-1,1,(cy,cx)).astype('f'); mref=mu.copy()
     dll.host_copy(mref.ctypes.data_as(P),cmu.ctypes.data_as(P),2,2,ratio,ratio,1,0,0,1,ny,nx,cy,cx,ilo,jlo,ihi-ilo+1,jhi-jlo+1)
     parent.mup=xp.asarray(mu);parent.thp=xp.asarray(target);cstate.mup=xp.asarray(cmu)
     node.state=cstate;node.parent=SimpleNamespace(state=parent);node.clock=SimpleNamespace(ticks=7)
     coupler.feedback=1;coupler._prepared_feedback={'ticks':7,'kinds':['mu','t']};coupler._bind_geometry=lambda:None;coupler._feedback_child_is_bounded=lambda:False;coupler.microphysics_reverse_transition=None;coupler._reverse_diagnoses=lambda kind:False;coupler.smooth_option=0;coupler.feedback_sync_bytes=0;coupler.feedback_host_scratch_bytes=0;coupler.feedback_count=0;coupler._restrict_windowed=lambda *a,**kw:NestCoupler._restrict_windowed(coupler,*a,**kw)
     NestCoupler.feedback_commit(coupler,node)
     np.testing.assert_array_equal(parent.mup.to_numpy().view('u4'),mref.view('u4'))
     np.testing.assert_array_equal(parent.thp.to_numpy().view('u4'),ref.view('u4'))
     assert coupler.feedback_count==1
 print('matched',cases,'actual windowed feedback callers,',words,'words')
