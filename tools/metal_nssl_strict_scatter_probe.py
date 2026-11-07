"""Compare the actual strict NSSL scatter caller with selected CUDA source."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.wrf_exact import ENABLED, effective_options
if not ENABLED: raise SystemExit('GPUWM_WRF_EXACT=1 is required')
from woof.core.kernels import module_source, module_options
from woof.core.nssl2_driver_support import NSSL2DriverWorkspace, scatter_nssl2_driver_workspace
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

src=module_source('nssl2_driver_support')
a=src.index('enum Nssl2DriverField');b=src.index('};',a)+2
i=src.index('extern "C" __global__ void nssl2_driver_scatter(')
body=src[i:];names=('qv','qc','qr','qi','qs','qg','qh','qndrop','qnr','qni','qns','qng','qnh','qnn','qvolg','qvolh')
pre='#include <cstddef>\n#define __global__\n#define __restrict__\nstruct D {int x;}; D blockDim,blockIdx,threadIdx;\n'+src[a:b]+'\n'
args=','.join('float *'+x for x in names);fields=','.join(names)
wrap='\nextern "C" void run(const float *rho,const float *state,'+args+',int n,int ccn){blockDim.x=1;threadIdx.x=0;for(int x=0;x<n;++x){blockIdx.x=x;nssl2_driver_scatter(rho,state,'+fields+',n,ccn);}}\n'
with tempfile.TemporaryDirectory(prefix='woof-nssl-scatter-') as d:
    d=Path(d);(d/'source.cpp').write_text(pre+body+wrap)
    subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(d/'source.cpp'),'-o',str(d/'source.dylib')],check=True)
    f=ctypes.CDLL(str(d/'source.dylib')).run;fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*18+[ctypes.c_int,ctypes.c_int];ptr=lambda x:x.ctypes.data_as(fp)
    nz,ny,nx=2,1,2048;shape=(nz,ny,nx);n=np.prod(shape);rng=np.random.default_rng(118)
    state=rng.uniform(0,100,(16,*shape)).astype(np.float32);rho=rng.uniform(.4,1.4,shape).astype(np.float32)
    state.reshape(16,-1)[:,:4]=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).tiny],np.float32)
    fields=[rng.uniform(-100,100,shape).astype(np.float32) for _ in names];cases={}
    with MetalContext() as context:
        context.batch_kernel_calls=False;submit=context._submit_kernel_batch
        def sync(*a,**kw):
            x=submit(*a,**kw);x.wait();return x
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        ws=NSSL2DriverWorkspace(state=xp.asarray(state),category_surface_export=xp.zeros((5,ny,nx)),shape=shape)
        for ccn in (False,True):
            ref=[x.copy() for x in fields];f(*map(ptr,[rho,state,*ref]),int(n),int(ccn));dev=[xp.asarray(x) for x in fields]
            scatter_nssl2_driver_workspace(ws,xp.asarray(rho),*dev,predicted_ccn=ccn)
            got=[x.to_numpy() for x in dev]
            cases[str(ccn)]={name:{'words':int(a.size),'different':int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))} for name,a,b in zip(names,ref,got)}
    print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_driver_support')),'cases':cases},indent=2))
    if any(c['different'] for group in cases.values() for c in group.values()):raise SystemExit(1)
