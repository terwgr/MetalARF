"""Compare strict boundary finalization caller to selected CUDA source words."""
import ctypes as ct,os,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.wrf_exact import BIGSTEP_ENABLED
from woof.ingest.lateral_bc import _launch_finalize_field
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.lbc_state import _perimeter_count
from tools.lbc_wrf_exact.generate_final import retained
from tools.mp28_metal.parse import def_name
from tools.thompson_real_column_parity.host_backend import parse_kernels
orig=MetalContext._submit_kernel_batch
def sync(self,*a,**k):
 r=orig(self,*a,**k)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
items=retained(BIGSTEP_ENABLED);params=parse_kernels('\n'.join(items))['finalize_state_field'];kernel=next(x for x in items if def_name(x)=='finalize_state_field');head=kernel[:kernel.index('{')].replace('extern "C" __global__','extern "C"').replace('void finalize_state_field(', 'void host_final(');names=','.join(p.name for p in params);wrap=head+'{blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*ny*nx);++blockIdx.x)finalize_state_field('+names+');}\n'
pre='''#include <stddef.h>\n#include <algorithm>\n#include <cmath>\nusing std::min;using std::max;using std::isnan;\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct dims{unsigned x;};dims blockIdx,blockDim,threadIdx;\nstatic float __fadd_rn(float a,float b){volatile float v=a+b;return v;}\nstatic float __fsub_rn(float a,float b){volatile float v=a-b;return v;}\nstatic float __fmul_rn(float a,float b){volatile float v=a*b;return v;}\nstatic float __fdiv_rn(float a,float b){volatile float v=a/b;return v;}\n'''
class State(SimpleNamespace):
 def existing_scratch(self,slot):return self.carried
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+'\n'.join(items)+'\n'+wrap);subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 fun=ct.CDLL(str(lib)).host_final;P=ct.POINTER(ct.c_float);F=ct.c_float;I=ct.c_int;fun.argtypes=[P if p.pointer else F if p.base in ('real','float') else I for p in params]
 xp=MetalArrayModule();rng=np.random.default_rng(672);cases=words=0
 for field in ('u','v','theta','phi','w','qv'):
  for mapped in (0,1):
   for b3 in (0,1):
    for has_muts in (0,1):
     nz,mny,mnx=3,7,8;ny=mny+(field=='v');nx=mnx+(field=='u');width=2;spec=1;shape=(nz,ny,nx);frame=_perimeter_count(mny,mnx,spec)
     def rnd(shape,lo=-1,hi=1):return rng.uniform(lo,hi,shape).astype(np.float32)
     host={'target':rnd(shape),'old_mup_frame':rnd((frame,)),'mub2d':rnd((mny,mnx),5,7),'mup':rnd((mny,mnx)),'muts_wrf':rnd((mny,mnx),3,4),'thb':rnd((nz,mny,mnx) if b3 else (nz,),299,301),'c1h':rnd((nz,),.6,1.2),'c2h':rnd((nz,),.5,1.5),'c1f':rnd((nz,),.6,1.2),'c2f':rnd((nz,),.5,1.5),'msft':rnd((mny,mnx),.8,1.2),'msfu':rnd((mny,mnx+1),.8,1.2),'msfv':rnd((mny+1,mnx),.8,1.2)}
     host['target'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
     for n,sh in [('west',(nz,ny,width)),('east',(nz,ny,width)),('south',(nz,width,nx)),('north',(nz,width,nx))]:host[n]=rnd(sh);host[n+'_t']=rnd(sh)
     d={n:xp.asarray(a) for n,a in host.items()};state=State(_metal_setup_state=True,_array_module=xp,has_msf=mapped,carried=d['muts_wrf'] if has_muts else None,**{n:d[n] for n in ('mub2d','mup','thb','c1h','c2h','c1f','c2f','msft','msfu','msfv')});setattr(state,{'theta':'thp','phi':'php'}.get(field,field),d['target']);boundary=SimpleNamespace(**{n:SimpleNamespace(value=d[n],tendency=d[n+'_t']) for n in ('west','east','south','north')})
     kind={'u':0,'v':1,'theta':2,'phi':3,'w':6,'qv':5}[field];ref=host['target'].copy();vals=dict(host,target=ref,scalar_unused=ref,dtbc=np.float32(.23),width=width,spec_zone=spec,has_msf=mapped,thb_3d=b3,kind=kind,nz=nz,ny=ny,nx=nx,mny=mny,mnx=mnx)
     args=[(ct.cast(0,P) if p.name=='muts_wrf' and not has_muts else vals[p.name].ctypes.data_as(P)) if p.pointer else F(vals[p.name]) if p.base in ('real','float') else I(vals[p.name]) for p in params];fun(*args)
     _launch_finalize_field(state,field,boundary,d['old_mup_frame'],.23,spec)
     np.testing.assert_array_equal(d['target'].to_numpy().view(np.uint32),ref.view(np.uint32),err_msg=str((field,mapped,b3,has_muts,BIGSTEP_ENABLED)))
     words+=ref.size;cases+=1
 print('matched',cases,'strict final-boundary callers,',words,'words; bigstep control',int(BIGSTEP_ENABLED),'with and without carried MUTS')
