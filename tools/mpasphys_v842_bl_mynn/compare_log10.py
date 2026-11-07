"""Compare pinned CORE-MATH log10f with the MYNN Metal primitive."""
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.resources_build import _vendored_maths,_cuda_core_math
from woof.hex.mpasphys.metal_bl_mynn_v842 import log10_values
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    source='#define p12_expect(a,b) (a)\n'+_cuda_core_math(_vendored_maths()['core-math/log10f.c'],'log10f')
    source+='''\nextern "C" __global__ void host_log10(int n,const float* x,float* y){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=p12_core_log10f::cr_log10f(x[i]);}\n'''
    host=HostModule('mpas_v842_mynn_log10',source).get_function('host_log10')
    rng=np.random.default_rng(842);bits=rng.integers(0,0xff800000,12000,dtype=np.uint32)
    bits=bits[(bits&0x7f800000)!=0x7f800000]
    edges=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f7fffff,0x3f800000,0x3f800001,0x411fffff,0x41200000,0x41200001,0x4e6e6b28,0x501502f9,0x42c80000,0x7f7fffff,0x7f800000,0xff800000,0x7f812345,0xff812345,0x7fc12345,0xffc12345],np.uint32)
    transitions=np.asarray([np.uint32(0x3f800000+min(0x7fffff,max(0,(j<<17)-65536+delta))) for j in range(65) for delta in (-1,0,1)],np.uint32)
    bits=np.concatenate((bits,edges,edges^np.uint32(0x80000000),transitions));values=bits.view(np.float32);n=len(values);expected=np.empty(n,np.float32)
    host(((n+127)//128,),(128,),(n,values,expected))
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);actual=xp.empty((n,),np.float32)
        log10_values(xp.asarray(values),actual)
        got=actual.to_numpy();bad=np.flatnonzero(got.view(np.uint32)!=expected.view(np.uint32))
        if bad.size:
            for i in bad[:12]:print(f'{bits[i]:08x} j={((int(bits[i])&0x7fffff)+65536)>>17} got={got[i]} want={expected[i]}',flush=True)
            raise AssertionError(f'MYNN log10 {bad.size} differences')
    print(f'MYNN CORE-MATH log10f matches source for {n} finite/boundary/infinite/NaN words',flush=True)

if __name__=='__main__':main()
