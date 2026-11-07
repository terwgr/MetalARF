"""Compare flow-boundary caller against composed CUDA source words."""
import ctypes as ct,os,subprocess,tempfile
from pathlib import Path
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core.kernels import module_source
from woof.ingest.lateral_bc import apply_flow_dependent_boundaries
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from tools.mp28_metal.parse import closure,def_name,split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels
orig=MetalContext._submit_kernel_batch
def sync(self,*a,**k):
 r=orig(self,*a,**k)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
source='#define GPUWM_WRF_EXACT 1\n'+module_source('lbc_flow');preprocessed=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=source,text=True,capture_output=True,check=True).stdout
items=closure(split_top(preprocessed),('flow_dependent_batch',))[0];params=parse_kernels('\n'.join(items))['flow_dependent_batch'];kernel=next(x for x in items if def_name(x)=='flow_dependent_batch');head=kernel[:kernel.index('{')].replace('extern "C" __global__','extern "C"').replace('void flow_dependent_batch(', 'void host_flow(');names=','.join(p.name for p in params)
wrap=head+'{blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*frame_count);++blockIdx.x)flow_dependent_batch('+names+');}\n'
pre='''#include <stddef.h>\n#include <algorithm>\nusing std::min;using std::max;\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct dims{unsigned x;};dims blockIdx,blockDim,threadIdx;\n'''
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+'\n'.join(items)+'\n'+wrap);subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 fun=ct.CDLL(str(lib)).host_flow;P=ct.POINTER(ct.c_float);F=ct.c_float;I=ct.c_int;fun.argtypes=[P if p.pointer else F if p.base in ('real','float') else I for p in params]
 xp=MetalArrayModule();rng=np.random.default_rng(148);cases=words=0
 for nfield in (1,3,9):
  for width in (1,2):
   for inflow in (0.,.25):
    nz,ny,nx=3,7,8;shape=(nz,ny,nx)
    fields=[rng.uniform(-2,2,shape).astype(np.float32) for _ in range(nfield)]
    for f in fields:f.flat[:6]=np.array([0,0x80000000,1,0x80000001,0x7fc01234,0x3f800000],np.uint32).view(np.float32)
    u=np.resize(np.array([0,0x80000000,1,0x80000001,0x3f800000,0xbf800000,0x7fc00000],np.uint32),(nz,ny,nx+1)).view(np.float32);v=np.resize(np.array([0x80000001,1,0,0x80000000,0xbf800000,0x3f800000,0x7fc00000],np.uint32),(nz,ny+1,nx)).view(np.float32)
    host=[f.copy() for f in fields];padded=host+[host[-1]]*(9-nfield);dev=[xp.asarray(f) for f in fields];du=xp.asarray(u);dv=xp.asarray(v);frame=sum(2*(nx-2*d)+2*max(ny-2*d-2,0) for d in range(width))
    vals={**{'f'+str(i):padded[i] for i in range(9)},'u_flux':u,'v_flux':v,'nfield':nfield,'spec_zone':width,'nz':nz,'ny':ny,'nx':nx,'frame_count':frame,'inflow_value':inflow}
    args=[vals[p.name].ctypes.data_as(P) if p.pointer else F(vals[p.name]) if p.base in ('real','float') else I(vals[p.name]) for p in params];fun(*args)
    apply_flow_dependent_boundaries(dev,du,dv,width,inflow_value=inflow)
    for expected,got in zip(host,dev):np.testing.assert_array_equal(got.to_numpy().view(np.uint32),expected.view(np.uint32),err_msg=str((nfield,width,inflow)))
    words+=nfield*np.prod(shape);cases+=1
 print('matched',cases,'strict flow-boundary callers,',words,'words; signed subnormal fluxes and NaNs included')
