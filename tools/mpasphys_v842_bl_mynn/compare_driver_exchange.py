"""Compare native MYNN exchange coefficients with selected host source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT, MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import driver_exchange
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    m=re.search(r'__device__ __noinline__ void f_retrieve_exchange_coeffs\s*\(.*?\)\s*\{',SOURCE,re.S);assert m
    s=SOURCE[m.end():];a=s.index('// bl_mynn_subroutines.F90:4331');b=s.index('// MPAS physics_mmm/bl_mynn_subroutines.F90:4344',a);body=s[a:b].strip();assert body.endswith('}');body=body[:-1]
    names=('dfm','dfh','dz','k_m','k_h')
    src='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT
    src+=''.join(f'#define {n}(i) {n}_ptr[col*nl+(i)-1]\n' for n in names)
    src+='extern "C" __global__ void host_exchange(int nc,int nl,'+','.join('float* '+n+'_ptr' for n in names)+'){int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;int kts=1,kte=nl,k;float dzk;\n'+body+'}\n'
    host=HostModule('mpas_v842_mynn_driver_exchange',src).get_function('host_exchange');rng=np.random.default_rng(842);nc=31;nl=17
    f={n:rng.uniform(.01,500,(nc,nl)).astype(np.float32) for n in names}
    f['dfm'][0,1]=np.array([1],np.uint32).view(np.float32)[0]
    expected={n:v.copy() for n,v in f.items()};host(((nc+127)//128,),(128,),(nc,nl,*[expected[n] for n in names]))
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);actual={n:xp.asarray(v.copy()) for n,v in f.items()}
        driver_exchange(actual)
        for n in names:
            got=actual[n].to_numpy();want=expected[n];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
            if bad.size:raise AssertionError(f'{n}: {bad.size} differences at {bad[0]} got {got.flat[bad[0]]} want {want.flat[bad[0]]}')
    print(f'MYNN exchange coefficients match selected source for {nc} columns',flush=True)
if __name__=='__main__':main()
