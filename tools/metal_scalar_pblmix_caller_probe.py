"""Compare the actual shared Metal caller with the selected CUDA source graph."""
import ctypes
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from woof.core.physics import PhysicsDriver
from woof.config import scalar_pblmix_species
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

parser=argparse.ArgumentParser()
parser.add_argument('--fixture', type=Path, required=True)
parser.add_argument('--poison-rebuilt-scratch', action='store_true')
args=parser.parse_args()
assert hashlib.sha256(args.fixture.read_bytes()).hexdigest() == 'ee2e3b0ff34569db72da628aa448163050cf609af6d0645f256f728161bde0aa'

# Compile the actual CUDA column body, replacing only its PTX RN primitives
# with volatile host FP32 operations. This is a source-graph comparison, not
# a claim of having run the NVIDIA backend.
src = (ROOT / 'woof/core/kernels/mynn_scalar_mix.cu').read_text()
body = src[src.index('template<class T> struct MynnColumn'):src.index('// ===========================================================================\n// One stock qn')]
header = '''#include <cstddef>\nusing real=float;\n#define __device__\n#define __global__\n#define __restrict__\nstruct Dim {int x;}; Dim blockIdx, blockDim, threadIdx;\nstatic float add(float a,float b){volatile float x=a+b;return x;}\nstatic float sub(float a,float b){volatile float x=a-b;return x;}\nstatic float mul(float a,float b){volatile float x=a*b;return x;}\nstatic float divr(float a,float b){volatile float x=a/b;return x;}\n#define SMX_ADD(a,b) add(a,b)\n#define SMX_SUB(a,b) sub(a,b)\n#define SMX_MUL(a,b) mul(a,b)\n#define SMX_DIV(a,b) divr(a,b)\n'''
wrapper = '''\nextern "C" void run(const float *q,const float *dz,const float *rho,const float *kh,float *dt,float *solved,float *rate,float *scratch,int nz,int nc){blockDim.x=1;threadIdx.x=0;for(int i=0;i<nc;++i){blockIdx.x=i;scalar_pblmix_columns(q,dz,rho,kh,dt,solved,rate,scratch,nz,nc);}}\n'''
with tempfile.TemporaryDirectory(prefix='woof-scalar-pblmix-') as d:
    path = Path(d); (path/'source.cpp').write_text(header+body+wrapper)
    subprocess.run(['clang++','-std=c++17','-O0','-fPIC','-shared','-fno-fast-math','-ffp-contract=off',str(path/'source.cpp'),'-o',str(path/'source.dylib')],check=True)
    host=ctypes.CDLL(str(path/'source.dylib')).run
    fp=ctypes.POINTER(ctypes.c_float)
    host.argtypes=[fp]*8+[ctypes.c_int,ctypes.c_int]
    ptr=lambda x:x.ctypes.data_as(fp)
    rng=np.random.default_rng(112); nz,ny,nx=7,4,5; shape=(nz,ny,nx); nc=ny*nx
    rand=lambda lo,hi,shape: rng.uniform(lo,hi,shape).astype(np.float32)
    dz=rand(20,250,shape); rho=rand(.5,1.2,shape); kh=rand(0,20,shape)
    c1=rand(.2,.8,(nz,)); c2=rand(5,50,(nz,)); mut=rand(400,700,(ny,nx)); dt=np.full(nc,30,np.float32)
    with MetalContext() as context:
        context.batch_kernel_calls=False
        submit=context._submit_kernel_batch
        def sync(*a,**kw):
            result=submit(*a,**kw);result.wait();return result
        context._submit_kernel_batch=sync
        xp=MetalArrayModule(context)
        class State:
            _metal_setup_state=True
            _array_module=xp
            has_msf=False
            def __init__(self):
                self.p=xp.zeros(shape);self.c1h=xp.asarray(c1);self.c2h=xp.asarray(c2);self._scratch={}
            def scratch(self,shape,slot,dtype=None):
                if slot not in self._scratch:self._scratch[slot]=xp.zeros(shape,dtype or np.float32)
                if args.poison_rebuilt_scratch and slot.startswith('scalar_pblmix_metal_'):
                    self._scratch[slot][...] = np.float32(np.nan)
                return self._scratch[slot]
            def total_mu(self):return xp.asarray(mut)
        state=State()
        class Driver:
            _couple_scalar_pblmix=PhysicsDriver._couple_scalar_pblmix
            _scalar_pblmix_rates=PhysicsDriver._scalar_pblmix_rates
        driver=Driver();driver.state=state;driver.fields={'exch_h':xp.asarray(kh)};driver.bldt_seconds=30
        atmosphere={'dz':xp.asarray(dz),'rho':xp.asarray(rho)}
        results=[]
        for pbl,mp in ((pbl,mp) for pbl in (1,2,5,9,11)
                       for mp in (0,8,9,10,16,18,28,50,900)
                       if not (pbl == 5 and mp == 28)):
            for masked in (False,True):
                cfg=SimpleNamespace(scalar_pblmix=1,bl_pbl_physics=pbl,mp_physics=mp,specified=masked and pbl != 9,nested=masked and pbl == 9)
                names=scalar_pblmix_species(cfg); inputs={}
                for name in names:
                    inputs[name]=rand(0,2000,shape);setattr(state,name,xp.asarray(inputs[name]))
                marker=object();driver.pbl_tendencies=SimpleNamespace(extra_scalars={'unchanged':marker, **{name:marker for name in names}})
                driver._couple_scalar_pblmix(cfg,atmosphere)
                assert driver.pbl_tendencies.extra_scalars['unchanged'] is marker
                for name in names:
                    q=inputs[name];solved=np.empty(shape,np.float32);rate=np.empty(shape,np.float32);scratch=np.empty((5*nz+1,nc),np.float32)
                    host(*map(ptr,(q,dz,rho,kh,dt,solved,rate,scratch)),nz,nc)
                    expected=(c1[:,None,None]*mut[None,:,:]+c2[:,None,None])*rate
                    if masked:
                        expected[:,:,0]=0;expected[:,:,-1]=0;expected[:,0,:]=0;expected[:,-1,:]=0
                    actual=driver.pbl_tendencies.extra_scalars[name].to_numpy()
                    differing=int(np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32)))
                    results.append({'pbl':pbl,'mp':mp,'masked':masked,'species':name,'words':q.size,'differing':differing})
        for pbl,mp in ((0,28),(900,28),(5,28)):
            cfg=SimpleNamespace(scalar_pblmix=1,bl_pbl_physics=pbl,mp_physics=mp,specified=False,nested=False)
            existing={'nc': object()};driver.pbl_tendencies=SimpleNamespace(extra_scalars=existing)
            driver._couple_scalar_pblmix(cfg,atmosphere)
            assert driver.pbl_tendencies.extra_scalars is existing
        # Each WRF fixture column gets unit mass coupling so the driver result
        # can be compared directly to the unmodified Fortran tendency.
        fixture=np.load(args.fixture)
        class FixtureState(State):
            def __init__(self):
                self.p=xp.zeros((50,1,1));self.c1h=xp.asarray(np.zeros(50,np.float32));self.c2h=xp.asarray(np.ones(50,np.float32));self._scratch={}
            def total_mu(self):return xp.asarray(np.ones((1,1),np.float32))
        driver.state=FixtureState()
        cfg=SimpleNamespace(scalar_pblmix=1,bl_pbl_physics=1,mp_physics=8,specified=False,nested=False)
        for case in range(8):
            for species in range(4):
                column=lambda name: fixture[name][case,species].reshape(50,1,1).copy()
                driver.state.ni=xp.asarray(column('qn'))
                driver.fields={'exch_h':xp.asarray(column('exch_h'))}
                driver.bldt_seconds=float(fixture['dt'][case,species,0])
                driver.pbl_tendencies=SimpleNamespace(extra_scalars={})
                driver._couple_scalar_pblmix(cfg,{'dz':xp.asarray(column('dz')),'rho':xp.asarray(column('rho'))})
                actual=driver.pbl_tendencies.extra_scalars['ni'].to_numpy().reshape(50)
                expected=fixture['tendency'][case,species]
                differing=int(np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32)))
                results.append({'wrf_case':case,'column':species,'words':50,'differing':differing})
        print(json.dumps(results,indent=2))
        if any(r['differing'] for r in results):raise SystemExit(1)
