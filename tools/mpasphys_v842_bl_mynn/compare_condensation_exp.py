"""Compare native MYNN gfk_exp FMA graph to the selected host source."""
from importlib.resources import files
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    src=CUDA_FTZ_HELPERS+files('woof.core.kernels').joinpath('glibc_flt32.cuh').read_text()
    src+='\nextern "C" __global__ void host_exp(int n,const float* x,float* y){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=gfk_exp(x[i]);}\n'
    host=HostModule('mpas_v842_mynn_exp_fma',src).get_function('host_exp')
    rng=np.random.default_rng(842);values=rng.uniform(-104,89,12000).astype(np.float32)
    edges=np.array([0,0x80000000,1,0x80000001,0x4202422f,0xc27c65d9,0x42b00000,0xc2b00000,0x42b17218,0x42b17219,0xc2cff1b4,0xc2cff1b5,0x7f800000,0xff800000,0x7fc12345,0xffc12345],np.uint32)
    bits=np.concatenate((values.view(np.uint32),edges));values=bits.view(np.float32);n=len(values);expected=np.empty(n,np.float32)
    host(((n+127)//128,),(128,),(n,values,expected))
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);actual=xp.empty((n,),np.float32)
        _launch('hexv842_mpas_bl_mynn_condensation_exp_f32',n,ctx,{'input':xp.asarray(values),'output':actual},{'count':np.int32(n)})
        got=actual.to_numpy();bad=np.flatnonzero(got.view(np.uint32)!=expected.view(np.uint32))
        if bad.size:raise AssertionError(f'MYNN exp {bad.size} differences at input {bits[bad[0]]:08x}, actual {got.view(np.uint32)[bad[0]]:08x}, expected {expected.view(np.uint32)[bad[0]]:08x}')
    print(f'MYNN FMA exponential matches source for {n} finite/boundary/special words',flush=True)

if __name__=='__main__':main()
