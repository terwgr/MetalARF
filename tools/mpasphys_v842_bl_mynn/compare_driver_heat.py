"""Compare the native MYNN dissipation-heat stage with selected host source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT, MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import driver_heat
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    m=re.search(r'__device__ __noinline__ void f_bl_mynn_run\s*\(.*?\)\s*\{',SOURCE,re.S);assert m
    s=SOURCE[m.end():];a=s.index('// bl_mynn.F90:973');body=s[a:s.index('// bl_mynn.F90:981',a)]
    helper='''
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_pow(float a,float b){return mynn_ieee_result(a,b,gfk_pow(a,b));}
__device__ float mynn_exp(float a){return mynn_ieee_result(a,a,gfk_exp(a));}
'''
    names=('qke1','el_pbl1','p1','diss_heat')
    src='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+files('woof.core.kernels').joinpath('glibc_flt32.cuh').read_text()+helper
    src+=''.join(f'#define {n}(i) {n}_ptr[col*nl+(i)-1]\n' for n in names)
    src+='extern "C" __global__ void host_heat(int nc,int nl,float cp,'+','.join('float* '+n+'_ptr' for n in names)+'){int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;int kts=1,kte=nl,k;float b1=24.f;\n'+body+'}\n'
    host=HostModule('mpas_v842_mynn_driver_heat',src).get_function('host_heat');rng=np.random.default_rng(842);nc=127;nl=17
    f={'qke1':rng.uniform(.0002,10,(nc,nl)).astype(np.float32),'el_pbl1':rng.uniform(.01,500,(nc,nl)).astype(np.float32),'p1':rng.uniform(1000,110000,(nc,nl)).astype(np.float32),'diss_heat':rng.uniform(-10,10,(nc,nl)).astype(np.float32)}
    f['qke1'][0,:4]=[1e-4,150.,1e-4,150.];f['el_pbl1'][0,:4]=[.01,.01,500.,500.];f['p1'][0,:4]=[1.,1.,100000.,100000.]
    expected={n:v.copy() for n,v in f.items()};host(((nc+127)//128,),(128,),(nc,nl,np.float32(1004),*[expected[n] for n in names]))
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);actual={n:xp.asarray(v.copy()) for n,v in f.items()}
        driver_heat(actual,cp=1004.)
        for n in names:
            got=actual[n].to_numpy();want=expected[n];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
            if bad.size:raise AssertionError(f'{n}: {bad.size} differences at {bad[0]} got {got.flat[bad[0]]} want {want.flat[bad[0]]}')
    print(f'MYNN dissipation heat matches selected source for {nc} columns',flush=True)
if __name__=='__main__':main()
