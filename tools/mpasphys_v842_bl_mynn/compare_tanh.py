"""Compare pinned CORE-MATH tanhf with the MYNN Metal primitive."""
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.resources_build import _vendored_maths,_cuda_core_math
from woof.hex.mpasphys.metal_bl_mynn_v842 import tanh_values
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    source='#define p12_expect(a,b) (a)\n'+_cuda_core_math(_vendored_maths()['core-math/tanhf.c'],'tanhf')
    source+='''\nextern "C" __global__ void host_tanh(int n,const float* x,float* y){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=p12_core_tanhf::cr_tanhf(x[i]);}\n'''
    host=HostModule('mpas_v842_mynn_tanh',source).get_function('host_tanh')
    rng=np.random.default_rng(842);bits=rng.integers(0,0xff800000,12000,dtype=np.uint32)
    bits=bits[(bits&0x7f800000)!=0x7f800000]
    edges=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x32ffffff,0x33000000,0x33000001,0x397fffff,0x39800000,0x39800001,0x41102cb2,0x41102cb3,0x41102cb4,0x7f7fffff,0x7f800000,0xff800000,0x7f812345,0xff812345,0x7fc12345,0xffc12345],np.uint32)
    bits=np.concatenate((bits,edges,edges^np.uint32(0x80000000)));values=bits.view(np.float32);n=len(values);expected=np.empty(n,np.float32)
    host(((n+127)//128,),(128,),(n,values,expected))
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);actual=xp.empty((n,),np.float32)
        tanh_values(xp.asarray(values),actual)
        got=actual.to_numpy();bad=np.flatnonzero(got.view(np.uint32)!=expected.view(np.uint32))
        if bad.size:raise AssertionError(f'MYNN tanh {bad.size} diffs at {bad[0]}, input bits {bits[bad[0]]:08x}, actual {got.view(np.uint32)[bad[0]]:08x}, expected {expected.view(np.uint32)[bad[0]]:08x}')
    print(f'MYNN CORE-MATH tanhf matches source for {n} finite/boundary/infinite/NaN words',flush=True)

if __name__=='__main__':main()
