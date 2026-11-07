"""Compare strict fused-GS preparation with host-compiled selected CUDA source."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from woof.wrf_exact import ENABLED, effective_options
if not ENABLED: raise SystemExit('GPUWM_WRF_EXACT=1 is required')
from woof.core.kernels import module_source, module_options
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.nssl2_fused_gs import submit_nssl2_prepare_fused_gs
src=module_source('nssl2_fused_gs')
constants='\n'.join(re.search(r'^constexpr (?:float|int) '+name+r'\s*=.*$',src,re.M).group(0) for name in ('FQSAT','NQSAT','CAW','CBW','CAI','CBI'))
a=src.index('__device__ __forceinline__ int ltemq_of(');b=src.index('// delbk / delabk',a)
c=src.index('__device__ __forceinline__ float prepass_qvs(');d=src.index('extern "C" __global__ void nssl2_fused_gs(',c)
shim=ROOT/'tools/thompson_real_column_parity/cuda_host_shim.h'; glibc=ROOT/'woof/core/kernels/glibc_flt32.cuh'
wrap='''\nextern "C" void run(float *t,float *target,const float *v,const float *th,const float *rho,const float *p,const float *e,int n){for(int i=0;i<n;++i){blockIdx.x=i;nssl2_prepare_fused_gs(t,target,v,th,rho,p,e,n);}}\n'''
with tempfile.TemporaryDirectory(prefix='woof-nssl-gs-prep-') as tmp:
    tmp=Path(tmp)
    (tmp/'source.cpp').write_text(f'#include "{shim}"\n#include "{glibc}"\nnamespace {{\n'+constants+'\n'+src[a:b]+src[c:d]+wrap)
    subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(tmp/'source.cpp'),'-o',str(tmp/'source.dylib')],check=True)
    f=ctypes.CDLL(str(tmp/'source.dylib')).run; fp=ctypes.POINTER(ctypes.c_float); f.argtypes=[fp]*7+[ctypes.c_int]; ptr=lambda x:x.ctypes.data_as(fp)
    nz,ncol=4,2048; shape=(nz,ncol); n=nz*ncol; rng=np.random.default_rng(2109)
    theta=rng.uniform(210,300,shape).astype(np.float32); exner=rng.uniform(.95,1.05,shape).astype(np.float32)
    rho=rng.uniform(.3,1.4,shape).astype(np.float32); pressure=rng.uniform(25000,105000,shape).astype(np.float32)
    state=np.zeros((16,*shape),np.float32); state[0]=rng.uniform(-.005,.05,shape).astype(np.float32)
    state[0].flat[:4]=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),-np.nextafter(np.float32(0),np.float32(1))],np.float32)
    t=np.empty(shape,np.float32); target=np.empty(shape,np.float32)
    f(*map(ptr,(t,target,state[0],theta,rho,pressure,exner)),n)
    with MetalContext() as context:
        context.batch_kernel_calls=False; submit=context._submit_kernel_batch
        def sync(*a,**kw):
            result=submit(*a,**kw); result.wait(); return result
        context._submit_kernel_batch=sync; xp=MetalArrayModule(context)
        dt=xp.zeros(shape); di=xp.zeros(shape)
        ds,dh,dr,dp,de=[xp.asarray(x) for x in (state,theta,rho,pressure,exner)]
        submit_nssl2_prepare_fused_gs(context,dt,di,ds,dh,dr,dp,de,nz=nz,ncol=ncol)
        got_t=dt.to_numpy(); got_target=di.to_numpy()
    differences=[int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32))) for x,y in ((t,got_t),(target,got_target))]
    print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_fused_gs')),'words':2*n,'different':differences,'active_targets':int(np.count_nonzero(target))},indent=2))
    if any(differences): raise SystemExit(1)
