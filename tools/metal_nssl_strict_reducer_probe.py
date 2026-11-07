"""Compare the strict NSSL reducer's Metal caller with selected CUDA source."""
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
from woof.core.nssl2_driver_support import NSSL2DriverWorkspace, reduce_nssl2_precipitation
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

src=module_source('nssl2_driver_support')
start=src.index('extern "C" __global__ void nssl2_driver_reduce_precipitation(')
end=src.index('extern "C" __global__ void nssl2_driver_scatter(',start)
body=src[start:end]
pre='''#include <cstddef>\n#define __global__\n#define __restrict__\nstruct D {int x;}; D blockDim,blockIdx,threadIdx;\nstatic float __fdiv_rn(float a,float b){volatile float x=a/b;return x;}\n'''
wrap='''\nextern "C" void run(const float *e,const float *rho,float *a,float *b,float *c,float *d,float *f,float *g,float *h,float *i,float *j,float dt,int n){blockDim.x=1;threadIdx.x=0;for(int x=0;x<n;++x){blockIdx.x=x;nssl2_driver_reduce_precipitation(e,rho,a,b,c,d,f,g,h,i,j,dt,n);}}\n'''
with tempfile.TemporaryDirectory(prefix='woof-nssl-reducer-') as d:
    d=Path(d);(d/'source.cpp').write_text(pre+body+wrap)
    subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(d/'source.cpp'),'-o',str(d/'source.dylib')],check=True)
    f=ctypes.CDLL(str(d/'source.dylib')).run;fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*11+[ctypes.c_float,ctypes.c_int]
    ptr=lambda x:x.ctypes.data_as(fp)
    n=4096;nz=2;shape=(nz,1,n);rng=np.random.default_rng(117)
    e=rng.uniform(0,1.e-4,(5,1,n)).astype(np.float32);rho=rng.uniform(.4,1.4,shape).astype(np.float32)
    fields=[rng.uniform(0,.1,(1,n)).astype(np.float32) for _ in range(9)]
    # Include positive/negative zero and gradual-underflow discriminators.
    e[:,0,:4]=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).tiny],np.float32)
    for x in fields:x[0,:4]=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).tiny],np.float32)
    ref=[x.copy() for x in fields];dt=np.float32(.75)
    f(*map(ptr,[e,rho,*ref]),dt,n)
    with MetalContext() as context:
        context.batch_kernel_calls=False;submit=context._submit_kernel_batch
        def sync(*a,**kw):
            x=submit(*a,**kw);x.wait();return x
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        dev=[xp.asarray(x) for x in fields]
        ws=NSSL2DriverWorkspace(state=xp.zeros((16,*shape)),category_surface_export=xp.asarray(e),shape=shape)
        reduce_nssl2_precipitation(ws,*dev,air_density=xp.asarray(rho),dt_s=dt)
        got=[x.to_numpy() for x in dev]
    names=('rainnc','rainncv','snownc','snowncv','graupelnc','graupelncv','hailnc','hailncv','sr')
    cases={name:{'words':int(a.size),'different':int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))} for name,a,b in zip(names,ref,got)}
    print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_driver_support')),'cases':cases},indent=2))
    if any(c['different'] for c in cases.values()):raise SystemExit(1)
