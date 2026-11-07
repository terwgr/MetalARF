"""Compare strict concentration radii with host-compiled selected CUDA source."""
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
from woof.core.kernels import module_source,module_options
from woof.core.nssl2_radiation import launch_effective_radius_concentration
from woof.metal.nssl2_diagnostics import submit_nssl2_effective_radius_concentration
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
src=module_source('nssl2_diagnostics'); start=src.index('extern "C" __global__ void nssl2_effective_radius_concentration(')
brace=src.index('{',start); depth=1; end=brace+1
while depth:
    depth+=(src[end]=='{')-(src[end]=='}');end+=1
shim=ROOT/'tools/thompson_real_column_parity/cuda_host_shim.h'; glibc=ROOT/'woof/core/kernels/glibc_flt32.cuh'
wrap='''\nextern "C" void run(const float *rho,const float *qc,const float *nc,const float *qi,const float *ni,const float *qs,const float *ns,float *rc,float *ri,float *rs,int n){for(int i=0;i<n;++i){blockIdx.x=i;nssl2_effective_radius_concentration(rho,qc,nc,qi,ni,qs,ns,rc,ri,rs,n);}}\n'''
with tempfile.TemporaryDirectory(prefix='woof-nssl-radii-') as tmp:
    tmp=Path(tmp);(tmp/'source.cpp').write_text(f'#include "{shim}"\n#include "{glibc}"\n'+src[start:end]+wrap)
    subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(tmp/'source.cpp'),'-o',str(tmp/'source.dylib')],check=True)
    f=ctypes.CDLL(str(tmp/'source.dylib')).run;fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*10+[ctypes.c_int];ptr=lambda x:x.ctypes.data_as(fp)
    shape=(4,1,1024); n=int(np.prod(shape)); rng=np.random.default_rng(6101)
    rho=rng.uniform(.3,1.5,shape).astype(np.float32)
    mass=[np.power(10.,rng.uniform(-15,-2,shape)).astype(np.float32) for _ in range(3)]
    number=[np.power(10.,rng.uniform(-10,10,shape)).astype(np.float32) for _ in range(3)]
    for x in mass+number: x.flat[:5]=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).tiny,-1.],np.float32)
    inputs=[rho,mass[0],number[0],mass[1],number[1],mass[2],number[2]]
    expected=[np.empty(shape,np.float32) for _ in range(3)];f(*map(ptr,inputs+expected),n)
    with MetalContext() as context:
        context.batch_kernel_calls=False;submit=context._submit_kernel_batch
        def sync(*a,**kw):
            r=submit(*a,**kw);r.wait();return r
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        dev=[xp.asarray(x) for x in inputs]; out=[xp.zeros(shape) for _ in range(3)]
        launch_effective_radius_concentration(*dev,*out)
        actual=[x.to_numpy() for x in out]
    diffs=[int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32))) for x,y in zip(expected,actual)]
    active=[int(np.count_nonzero((x>np.float32(1e-13)) & (y>np.float32(1e-8)))) for x,y in zip(mass,number)]
    print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_diagnostics')),'words':3*n,'different':diffs,'active':active},indent=2))
    if any(diffs):raise SystemExit(1)
