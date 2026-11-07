"""Compare strict NSSL hot gather with the selected CUDA source."""
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
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.nssl2_driver_transport import submit_nssl2_driver_gather_initialize

src=module_source('nssl2_driver_support')
a=src.index('enum Nssl2DriverField');b=src.index('// WRF v4.6.1 module_mp_nssl_2mom.F:4242',a)
body=src[a:b]
pre='''#include <cstddef>\n#include <cmath>\n#define __global__\n#define __device__\n#define __forceinline__ inline\n#define __restrict__\nstruct D {int x;}; D blockDim,blockIdx,threadIdx;\nstatic float __fdiv_rn(float a,float b){volatile float x=a/b;return x;}\n'''
names=('air_density','qv','qc','qr','qi','qs','qg','qh','qndrop','qnr','qni','qns','qng','qnh','qnn','qvolg','qvolh','qrcuten','qscuten','qicuten','qccuten')
args=','.join('const float *'+x for x in names);fields=','.join(names)
wrap='\nextern "C" void run('+args+',float *state,float dt,int n,int ccn){blockDim.x=1;threadIdx.x=0;for(int x=0;x<n;++x){blockIdx.x=x;nssl2_driver_gather_initialize('+fields+',state,dt,0,0,n,ccn);}}\n'
with tempfile.TemporaryDirectory(prefix='woof-nssl-gather-') as d:
    d=Path(d);(d/'source.cpp').write_text(pre+body+wrap)
    subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(d/'source.cpp'),'-o',str(d/'source.dylib')],check=True)
    f=ctypes.CDLL(str(d/'source.dylib')).run;fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*22+[ctypes.c_float,ctypes.c_int,ctypes.c_int];ptr=lambda x:x.ctypes.data_as(fp)
    nz,ny,nx=2,1,2048;shape=(nz,ny,nx);n=np.prod(shape);rng=np.random.default_rng(119)
    inputs={name:rng.uniform(0,1,shape).astype(np.float32) for name in names}
    inputs['air_density']=rng.uniform(.4,1.4,shape).astype(np.float32)
    for name in names[1:]:inputs[name].reshape(-1)[:4]=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).tiny],np.float32)
    cases={}
    with MetalContext() as context:
        context.batch_kernel_calls=False;submit=context._submit_kernel_batch
        def sync(*a,**kw):
            x=submit(*a,**kw);x.wait();return x
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context);dev={name:xp.asarray(value) for name,value in inputs.items()}
        for ccn in (False,True):
            ref=np.empty((16,*shape),np.float32);f(*map(ptr,[*inputs.values(),ref]),np.float32(.75),int(n),int(ccn))
            state=xp.zeros((16,*shape));submit_nssl2_driver_gather_initialize(context,state,**dev,dt_s=.75,first_step=False,cu_used=False,predicted_ccn=ccn)
            got=state.to_numpy();cases[str(ccn)]={'words':int(ref.size),'different':int(np.count_nonzero(ref.view(np.uint32)!=got.view(np.uint32)))}
        for first,cu in ((True,False),(False,True),(True,True)):
            state=xp.asarray(np.full((16,*shape),-23,np.float32))
            try:submit_nssl2_driver_gather_initialize(context,state,**dev,dt_s=.75,first_step=first,cu_used=cu)
            except NotImplementedError:pass
            else:raise AssertionError('unsupported power path was not refused')
            assert np.all(state.to_numpy()==np.float32(-23))
    print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_driver_support')),'hot_cases':cases,'power_paths_refused':3},indent=2))
    if any(c['different'] for c in cases.values()):raise SystemExit(1)
