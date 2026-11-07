"""Compare strict Registry radii and radiation caller to selected source."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.wrf_exact import ENABLED,effective_options
if not ENABLED: raise SystemExit('GPUWM_WRF_EXACT=1 is required')
from woof.core.kernels import module_source,module_options
from woof.core.nssl2_radiation import update_effective_radii
from woof.metal.nssl2_diagnostics import submit_nssl2_effective_radius_concentration
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
src=module_source('nssl2');start=src.index('extern "C" __global__ void nssl2_effective_radius(');end=src.index('// Numerical authority is module_mp_nssl_2mom.F:5168',start)
shim=ROOT/'tools/thompson_real_column_parity/cuda_host_shim.h';glibc=ROOT/'woof/core/kernels/glibc_flt32.cuh'
wrap='''\nextern "C" void run(const float *rho,const float *qc,const float *nc,const float *qi,const float *ni,const float *qs,const float *ns,float *rc,float *ri,float *rs,int n){for(int i=0;i<n;++i){blockIdx.x=i;nssl2_effective_radius(rho,qc,nc,qi,ni,qs,ns,rc,ri,rs,n);}}\nextern "C" void micron(const float *in,float *out,int n){for(int i=0;i<n;++i){volatile float v=in[i]*1.0e6f;out[i]=v;}}\n'''
with tempfile.TemporaryDirectory(prefix='woof-nssl-registry-radii-') as tmp:
    tmp=Path(tmp);(tmp/'source.cpp').write_text(f'#include "{shim}"\n#include "{glibc}"\n'+src[start:end]+wrap)
    subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(tmp/'source.cpp'),'-o',str(tmp/'source.dylib')],check=True)
    lib=ctypes.CDLL(str(tmp/'source.dylib'));f=lib.run;fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*10+[ctypes.c_int];lib.micron.argtypes=[fp,fp,ctypes.c_int];ptr=lambda x:x.ctypes.data_as(fp)
    shape=(4,1,1024);n=int(np.prod(shape));rng=np.random.default_rng(7121)
    rho=rng.uniform(.3,1.5,shape).astype(np.float32);mass=[np.power(10.,rng.uniform(-15,-2,shape)).astype(np.float32) for _ in range(3)];number=[np.power(10.,rng.uniform(-10,10,shape)).astype(np.float32) for _ in range(3)]
    for x in mass+number:x.flat[:4]=[0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).tiny]
    inputs=[rho,mass[0],number[0],mass[1],number[1],mass[2],number[2]]
    metres=[np.empty(shape,np.float32) for _ in range(3)];f(*map(ptr,inputs+metres),n)
    micros=[np.empty(shape,np.float32) for _ in range(3)]
    for a,b in zip(metres,micros):lib.micron(ptr(a),ptr(b),n)
    with MetalContext() as ctx:
        ctx.batch_kernel_calls=False;submit=ctx._submit_kernel_batch
        def sync(*a,**kw):
            r=submit(*a,**kw);r.wait();return r
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);dev=[xp.asarray(x) for x in inputs];out=[xp.zeros(shape) for _ in range(3)]
        submit_nssl2_effective_radius_concentration(ctx,*dev,*out,registry_unit_numbers=True)
        actual_m=[x.to_numpy() for x in out]
        state=SimpleNamespace(qc=dev[1],qndrop=dev[2],qi=dev[3],qni=dev[4],qs=dev[5],qns=dev[6],effc=out[0],effi=out[1],effs=out[2])
        assert update_effective_radii(state,air_density=dev[0],radiation_due=False) is False
        assert update_effective_radii(state,air_density=dev[0],radiation_due=True,validate_values=True) is True
        actual_u=[x.to_numpy() for x in out]
    def differences(a,b):return [int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32))) for x,y in zip(a,b)]
    dm,du=differences(metres,actual_m),differences(micros,actual_u)
    active=[int(np.count_nonzero((a>np.float32(1e-13)) & ((b*rho)>np.float32(1e-8)))) for a,b in zip(mass,number)]
    print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2')),'metre_words':3*n,'metre_different':dm,'micron_words':3*n,'micron_different':du,'active':active},indent=2))
    if any(dm+du):raise SystemExit(1)
