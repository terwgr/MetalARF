"""Compare strict nesting coupling caller with composed CUDA words."""
import ctypes as ct,os,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.wrf_exact import BIGSTEP_ENABLED
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.lbc_state import submit_uncouple_feedback_field
from tools.lbc_wrf_exact.generate_uncouple import retained,KERNELS
from tools.mp28_metal.parse import def_name
from tools.thompson_real_column_parity.host_backend import parse_kernels
orig=MetalContext._submit_kernel_batch
def sync(self,*a,**k):
 r=orig(self,*a,**k)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
items=retained(BIGSTEP_ENABLED);params=parse_kernels('\n'.join(items));defs='\n'.join(items);wrappers=[]
for name in KERNELS:
 kernel=next(x for x in items if def_name(x)==name);head=kernel[:kernel.index('{')].replace('extern "C" __global__','extern "C"').replace('void '+name+'(', 'void host_'+name+'(');names=','.join(p.name for p in params[name]);count='nz*(j_hi-j_lo+1)*(i_hi-i_lo+1)';wrappers.append(head+'{blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)('+count+');++blockIdx.x)'+name+'('+names+');}\n')
pre='''#include <stddef.h>\n#include <algorithm>\nusing std::min;using std::max;\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct dims{unsigned x;};dims blockIdx,blockDim,threadIdx;\nstatic float __fadd_rn(float a,float b){volatile float v=a+b;return v;}\nstatic float __fsub_rn(float a,float b){volatile float v=a-b;return v;}\nstatic float __fmul_rn(float a,float b){volatile float v=a*b;return v;}\nstatic float __fdiv_rn(float a,float b){volatile float v=a/b;return v;}\n'''
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+defs+'\n'+''.join(wrappers));subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True);dll=ct.CDLL(str(lib));P=ct.POINTER(ct.c_float);F=ct.c_float;I=ct.c_int
 for n in KERNELS:getattr(dll,'host_'+n).argtypes=[P if p.pointer else F if p.base in ('real','float') else I for p in params[n]]
 xp=MetalArrayModule();rng=np.random.default_rng(738);cases=words=0
 for field in ('u','v','t','ph','w','qv'):
  for mapped in (0,1):
   for b3 in (0,1):
    for mode in ('interior','edge'):
     nz,mny,mnx=(1 if field=='mu' else 3),7,8;ny=mny+(field=='v');nx=mnx+(field=='u');shape=(nz,ny,nx)
     def rnd(shape,lo=-1,hi=1):return rng.uniform(lo,hi,shape).astype(np.float32)
     h={'target':rnd(shape),'out':rnd(shape),'mub2d':rnd((mny,mnx),5,7),'mup':rnd((mny,mnx)),'thb':rnd((nz,mny,mnx) if b3 else (nz,),299,301),'c1h':rnd((nz,),.6,1.2),'c2h':rnd((nz,),.5,1.5),'c1f':rnd((nz,),.6,1.2),'c2f':rnd((nz,),.5,1.5),'msft':rnd((mny,mnx),.8,1.2),'msfu':rnd((mny,mnx+1),.8,1.2),'msfv':rnd((mny+1,mnx),.8,1.2)}
     if field=='mu':h['target']=h['mup'].reshape(shape).copy()
     h['target'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
     h['out'][:,2,2:6]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
     if field=='mu':h['mup']=h['target'].reshape((mny,mnx)).copy()
     d={n:xp.asarray(v) for n,v in h.items()};state=SimpleNamespace(_metal_setup_state=True,_array_module=xp,has_msf=mapped,**{n:d[n] for n in ('mub2d','mup','thb','c1h','c2h','c1f','c2f','msft','msfu','msfv')})
     if field!='mu':setattr(state,{'t':'thp','ph':'php'}.get(field,field),d['target'])
     name='uncouple_feedback_field';ref=h['target'].copy();kind={'u':0,'v':1,'t':2,'ph':3,'w':6,'qv':5}[field]; reg=SimpleNamespace(xstag=field=='u',ystag=field=='v',nxc=nx*3,nyc=ny*3,nri=3,nrj=3,i_parent_start=1 if mode=='edge' else 2,j_parent_start=1 if mode=='edge' else 2); from woof.core.nest_interp import feedback_parent_bounds; i_lo,i_hi,j_lo,j_hi=feedback_parent_bounds(reg,spec_zone=0 if mode=='edge' else 1); i_hi=min(i_hi,nx-1); j_hi=min(j_hi,ny-1); reg.nxc=(i_hi+1-reg.i_parent_start+(0 if reg.xstag else 1)+(0 if mode=='edge' else 1))*3+(1 if reg.xstag else 0); reg.nyc=(j_hi+1-reg.j_parent_start+(0 if reg.ystag else 1)+(0 if mode=='edge' else 1))*3+(1 if reg.ystag else 0); i_lo,i_hi,j_lo,j_hi=feedback_parent_bounds(reg,spec_zone=0 if mode=='edge' else 1)
     vals=dict(h,target=ref,coupled=h['out'],has_msf=mapped,thb_3d=b3,kind=kind,nz=nz,ny=ny,nx=nx,mny=mny,mnx=mnx,i_lo=i_lo,i_hi=i_hi,j_lo=j_lo,j_hi=j_hi)
     args=[vals[p.name].ctypes.data_as(P) if p.pointer else F(vals[p.name]) if p.base in ('real','float') else I(vals[p.name]) for p in params[name]];getattr(dll,'host_'+name)(*args)
     submit_uncouple_feedback_field(state,field,d['out'],reg,spec_zone=0 if mode=='edge' else 1)
     np.testing.assert_array_equal(d['target'].to_numpy().view(np.uint32),ref.view(np.uint32),err_msg=str((field,mapped,b3,mode,BIGSTEP_ENABLED)))
     words+=ref.size;cases+=1
 print('matched',cases,'strict feedback uncoupling callers,',words,'words; bigstep control',int(BIGSTEP_ENABLED))
