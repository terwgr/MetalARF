"""Compare the strict NUCOND prepass with host-compiled selected CUDA source."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from woof.wrf_exact import ENABLED, effective_options
if not ENABLED:
    raise SystemExit('GPUWM_WRF_EXACT=1 is required')
from woof.core.kernels import module_source, module_options
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.nssl2_nucond import submit_nssl2_nucond_supersaturation

src = module_source('nssl2_nucond')
a = src.index('namespace {', src.index('WRF v4.6.1 NSSL option-18 NUCOND'))
b = src.index('__device__ __forceinline__ float dtabqvs', a)
c = src.index('extern "C" __global__ void nssl2_nucond_supersaturation(', b)
d = src.index('extern "C" __global__ void nssl2_nucond_default(', c)
shim = ROOT / 'tools/thompson_real_column_parity/cuda_host_shim.h'
glibc = ROOT / 'woof/core/kernels/glibc_flt32.cuh'
wrap = '''\nextern "C" void run(const float *t,const float *p,const float *e,const float *q,float *s,int n) {\n  for(int i=0;i<n;++i) {blockIdx.x=i; nssl2_nucond_supersaturation(t,p,e,q,s,n);}\n}\n'''
with tempfile.TemporaryDirectory(prefix='woof-nssl-ss-') as tmp:
    path = Path(tmp)
    (path/'source.cpp').write_text(f'#include "{shim}"\n#include "{glibc}"\n'+src[a:b]+'\n}\n'+src[c:d]+wrap)
    subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(path/'source.cpp'),'-o',str(path/'source.dylib')],check=True)
    f = ctypes.CDLL(str(path/'source.dylib')).run
    fp = ctypes.POINTER(ctypes.c_float)
    f.argtypes = [fp]*5+[ctypes.c_int]
    ptr = lambda x: x.ctypes.data_as(fp)
    shape=(4,1,2048); n=int(np.prod(shape)); rng=np.random.default_rng(101)
    t=rng.uniform(190,320,shape).astype(np.float32)
    p=rng.uniform(20000,105000,shape).astype(np.float32)
    e=rng.uniform(.6,1.1,shape).astype(np.float32)
    q=rng.uniform(-.001,.03,shape).astype(np.float32)
    q.flat[:6]=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)), -np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).tiny,-np.finfo(np.float32).tiny],np.float32)
    expected=np.empty(shape,np.float32)
    f(*map(ptr,(t,p,e,q,expected)),n)
    with MetalContext() as context:
        context.batch_kernel_calls=False
        submit=context._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs); result.wait(); return result
        context._submit_kernel_batch=sync
        xp=MetalArrayModule(context)
        arrays=[xp.asarray(x) for x in (t,p,e,q)]
        out=xp.zeros(shape)
        submit_nssl2_nucond_supersaturation(context,*arrays,out,nz=shape[0],ny=shape[1],nx=shape[2])
        actual=out.to_numpy()
    different=int(np.count_nonzero(expected.view(np.uint32)!=actual.view(np.uint32)))
    print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_nucond')),'words':n,'different':different},indent=2))
    if different: raise SystemExit(1)
