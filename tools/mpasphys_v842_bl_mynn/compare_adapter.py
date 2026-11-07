"""Compare MPAS MYNN pre/post/radiation to selected compiled source."""
from itertools import product
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.native_powf import CUDA_NATIVE_POWF
from woof.hex.mpasphys.bl_mynn_v842 import ADAPTER_CUDA_SOURCE, MYNN_IEEE_ARITHMETIC_SUPPORT
from woof.hex.mpasphys.metal_bl_mynn_v842 import pre,post,radiation,grid_spacing
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    start=ADAPTER_CUDA_SOURCE.index('extern "C" __global__ void mynn_pre(')
    stop=ADAPTER_CUDA_SOURCE.index('// MPAS mpas_atmphys_driver_pbl.F:454-485',start)
    source='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+ADAPTER_CUDA_SOURCE[start:stop]
    # The radiation kernel is independent of the driver layout helper.
    a=ADAPTER_CUDA_SOURCE.index('extern "C" __global__ void mynn_radiation(')
    b=ADAPTER_CUDA_SOURCE.index('// MPAS driver_pbl.F:384-389',a)
    host=HostModule('mpas_v842_bl_mynn_adapter',source+ADAPTER_CUDA_SOURCE[a:b])
    n=256;rng=np.random.default_rng(842)
    mix={name:rng.uniform(-.04,.04,n).astype(np.float32) for name in ('qv','qc','qi','qs')}
    tends={name:rng.uniform(-.0005,.0005,n).astype(np.float32) for name in ('dqv','dqc','dqi','dqs')}
    # Include valid positive/negative subnormals, signed zero, infinities and NaN payloads.
    bits=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc12345,0xffc12345,0x7f812345],np.uint32)
    for i,value in enumerate(mix.values()):value[:len(bits)]=np.roll(bits,i).view(np.float32)
    for i,value in enumerate(tends.values()):value[:len(bits)]=np.roll(bits,i+3).view(np.float32)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for flags in product((0,1),repeat=3):
            for inverse in (False,True):
                expected={name:rng.uniform(.1,.9,n).astype(np.float32) for name in ('sqv','sqc','sqi','sqs')}
                actual={name:xp.asarray(value.copy()) for name,value in expected.items()}
                host.get_function('mynn_pre_rp1' if inverse else 'mynn_pre')(((n+127)//128,),(128,),(n,*flags,*mix.values(),*expected.values()))
                pre({name:xp.asarray(value) for name,value in mix.items()},actual,flags,exact_inverse=inverse)
                for name in expected:
                    got=actual[name].to_numpy();bad=np.flatnonzero(got.view(np.uint32)!=expected[name].view(np.uint32))
                    if bad.size:raise AssertionError(f'pre {flags} inverse={inverse} {name}: {bad.size} diffs first {bad[0]} {got[bad[0]]!r} {expected[name][bad[0]]!r}')
                expected={name:value.copy() for name,value in tends.items()};actual={name:xp.asarray(value.copy()) for name,value in tends.items()}
                host.get_function('mynn_post_rp1' if inverse else 'mynn_post')(((n+127)//128,),(128,),(n,*flags,np.float32(60),*mix.values(),*expected.values()))
                post({name:xp.asarray(value) for name,value in mix.items()},actual,flags,60,exact_inverse=inverse)
                for name in expected:
                    got=actual[name].to_numpy();bad=np.flatnonzero(got.view(np.uint32)!=expected[name].view(np.uint32))
                    if bad.size:raise AssertionError(f'post {flags} inverse={inverse} {name}: {bad.size} diffs first {bad[0]} {got[bad[0]]!r} {expected[name][bad[0]]!r}')
        lw=mix['qv'];sw=mix['qc'];expected=np.empty(n,np.float32)
        host.get_function('mynn_radiation')(((n+127)//128,),(128,),(n,lw,sw,expected))
        actual=xp.empty((n,),np.float32);radiation(xp.asarray(lw),xp.asarray(sw),actual)
        if not np.array_equal(actual.to_numpy().view(np.uint32),expected.view(np.uint32)):raise AssertionError('radiation mismatch')
    start=ADAPTER_CUDA_SOURCE.index('extern "C" __global__ void mynn_dx(')
    source='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+CUDA_NATIVE_POWF+MYNN_IEEE_ARITHMETIC_SUPPORT+ADAPTER_CUDA_SOURCE[start:]
    kernel=HostModule('mpas_v842_bl_mynn_dx',source).get_function('mynn_dx')
    density=np.exp2(np.linspace(-125,125,2048,dtype=np.float32)).astype(np.float32)
    density[0]=np.array(0x00800000,np.uint32).view(np.float32);density[-1]=np.array(0x7f7fffff,np.uint32).view(np.float32)
    expected=np.empty_like(density)
    kernel(((len(density)+127)//128,),(128,),(len(density),np.float32(4500),density,expected))
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        out=xp.zeros(density.shape,np.float32);grid_spacing(xp.asarray(density),4500,out)
        bad=np.flatnonzero(out.to_numpy().view(np.uint32)!=expected.view(np.uint32))
        if bad.size:raise AssertionError(f'dx differs in {bad.size} words, first {bad[0]}')
        for bits in (0,1,0x80000000,0xbf800000,0x7f800000,0x7fc12345):
            bad_density=density.copy();bad_density[2]=np.array(bits,np.uint32).view(np.float32)
            saved=out.to_numpy().copy()
            try:grid_spacing(xp.asarray(bad_density),4500,out)
            except ValueError:pass
            else:raise AssertionError('exceptional native pow fallback was not refused')
            if not np.array_equal(out.to_numpy().view(np.uint32),saved.view(np.uint32)):raise AssertionError('invalid dx published')
    print('MYNN native moisture and radiation match source; positive normal grid spacing matches source and exceptional fallback is refused',flush=True)

if __name__=='__main__':main()
