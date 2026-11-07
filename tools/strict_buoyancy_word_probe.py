"""Compare the Metal caller with host-compiled strict CUDA source words."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np

os.environ['GPUWM_WRF_EXACT'] = '1'
os.environ.setdefault('WOOF_WRF_EXACT_DIAGNOSTICS', '1')
from woof.core.dycore import _launch_slow_buoyancy
from woof.metal.dycore import submit_slow_buoyancy
from woof.metal.array import MetalArrayModule

source = (Path(__file__).resolve().parents[1] / 'woof/core/kernels/dycore.cu').read_text()
base = source[source.index('static __device__ __forceinline__\nreal base_value'):source.index('static __device__ __forceinline__\nreal mut_value')]
pp = source[source.index('static __device__ __forceinline__\nreal pp_value'):source.index('static __device__ __forceinline__\nreal dpn_value')]
body = source[source.index('#if GPUWM_WRF_EXACT\n// WRF calc_cq'):source.index('static __device__ __forceinline__\nreal ph_total')]
prelude = '''#include <stddef.h>\n#define __device__\n#define __global__\n#define __forceinline__ inline\n#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_D_DIAGNOSTICS %s\ntypedef float real; const float G=9.81f;\nstruct dim { unsigned x; }; dim blockIdx,blockDim,threadIdx;\nstatic float rn_add(float a,float b){volatile float r=a+b;return r;}\nstatic float rn_sub(float a,float b){volatile float r=a-b;return r;}\nstatic float rn_mul(float a,float b){volatile float r=a*b;return r;}\nstatic float rn_div(float a,float b){volatile float r=a/b;return r;}\n''' % os.environ['WOOF_WRF_EXACT_DIAGNOSTICS']
wrapper = '''\nextern "C" void run(real* rw,const real* p,const real* pb,const real* mu,const real* mub,const real* qv,const real* qc,const real* qr,const real* qi,const real* qs,const real* qg,const real* qh,const real* rdn,const real* rdnw,const real* c1,const real* c2,const real* ms,int moist,int map,int b3,int nz,int ny,int nx){blockDim.x=1; threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)(nz*ny*nx);++blockIdx.x)slow_buoyancy(rw,p,pb,mu,mub,qv,qc,qr,qi,qs,qg,qh,rdn,rdnw,c1,c2,ms,moist,map,b3,nz,ny,nx);}\n'''
with tempfile.TemporaryDirectory() as tmp:
    cpp=Path(tmp)/'ref.cpp'; lib=Path(tmp)/'ref.dylib'; cpp.write_text(prelude+base+pp+body+wrapper)
    subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(cpp),'-o',str(lib)],check=True)
    run=ct.CDLL(str(lib)).run
    ptr=ct.POINTER(ct.c_float); run.argtypes=[ptr]*17+[ct.c_int]*6
    rng=np.random.default_rng(734); xp=MetalArrayModule()
    for b3 in (0,1):
      for mode in range(4):
       for mapped in (0,1):
        nz,ny,nx=4,2,3; shape=(nz,ny,nx); plane=(ny,nx)
        def rand(shape, scale=1): return np.asarray(rng.uniform(-scale,scale,shape),np.float32)
        h={'rw_t':rand((nz+1,ny,nx)), 'p':rand(shape,100), 'pb':rand(shape if b3 else (nz,),10), 'mup':rand(plane,5),'mub2d':rand(plane,5),'rdn':rand((nz,)), 'rdnw':rand((nz,)), 'c1f':rand((nz+1,)), 'c2f':rand((nz+1,)), 'msft':np.asarray(rng.uniform(.8,1.2,plane),np.float32)}
        for name in ('qv','qc','qr','qi','qs','qg','qh'): h[name]=rand(shape,.01)
        h['p'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
        h['qv'].flat[:4]=np.array([0x80000000,0,1,0x80000001],np.uint32).view(np.float32)
        # Missing snow/graupel are passed as the zero plane by the CUDA caller.
        if mode>=2 and not mapped: h['qs']=np.zeros(shape,np.float32); h['qg']=np.zeros(shape,np.float32)
        if mode == 0 and not mapped:
            for name in ('rw_t','pb','mup','mub2d','c1f','c2f'): h[name].fill(0)
            h['rdn'].fill(1); h['rdnw'].fill(1)
            h['p'].flat[:]=np.resize(np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32),h['p'].size)
        expected=h['rw_t'].copy(); names=['p','pb','mup','mub2d','qv','qc','qr','qi','qs','qg','qh','rdn','rdnw','c1f','c2f','msft']
        refs=[expected]+[h[n] for n in names]; run(*[a.ctypes.data_as(ptr) for a in refs],mode,mapped,b3,nz,ny,nx)
        state=SimpleNamespace(_metal_setup_state=True,_array_module=xp,has_msf=mapped)
        for n,v in h.items(): setattr(state,n,xp.asarray(v))
        state.phb=xp.asarray(np.zeros(shape if b3 else (nz,),np.float32))
        state.p_perturbation=state.p
        if os.environ["WOOF_WRF_EXACT_DIAGNOSTICS"] == "1":
            state.p=xp.asarray(h["p"] + np.float32(11))
        if not mode: state.qv=None
        if mode<2: state.qi=None; state.qs=None; state.qg=None
        elif not mapped: state.qs=None; state.qg=None
        if mode<3: state.qh=None
        token=_launch_slow_buoyancy(state,SimpleNamespace()); token.wait()
        actual=state.rw_t.to_numpy(); mismatch=np.flatnonzero(actual.view(np.uint32)!=expected.view(np.uint32))
        if mismatch.size: raise AssertionError((b3,mode,mapped,mismatch.tolist(),actual.ravel()[mismatch],expected.ravel()[mismatch]))
        state.rw_t=xp.asarray(h["rw_t"])
        token=submit_slow_buoyancy(state,SimpleNamespace()); token.wait()
        np.testing.assert_array_equal(state.rw_t.to_numpy().view(np.uint32), expected.view(np.uint32))
    print('matched 16 cases, literal words including untouched bottom and top; diagnostics='+os.environ['WOOF_WRF_EXACT_DIAGNOSTICS'])
