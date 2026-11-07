"""Compare native MYNN surface and scale-aware factors to selected source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import stability
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def host_module():
    def function(name):
        m=re.search(r'__device__ __noinline__ (?:float|void) '+name+r'\s*\(',SOURCE);assert m
        i=SOURCE.index('{',m.end());j=i+1;depth=1
        while depth:
            if SOURCE[j]=='{':depth+=1
            elif SOURCE[j]=='}':depth-=1
            j+=1
        return SOURCE[m.start():j]
    helper='''
struct MynnArena {};
struct MynnFrame {MynnFrame(MynnArena&){} };
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}
__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_pow(float a,float b){return mynn_ieee_result(a,b,gfk_pow(a,b));}
__device__ float mynn_log(float a){return mynn_ieee_result(a,a,gfk_log(a));}
__device__ float mynn_atan(float a){return mynn_ieee_result(a,a,glibc_atanf(a));}
template<int N> __device__ float mynn_ipow(float a){static_assert(N==2);return mpas_mul(a,a);}
'''
    root=files('woof.core.kernels')
    source='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+root.joinpath('glibc_flt32.cuh').read_text()+root.joinpath('glibc_trig_flt32.cuh').read_text()+helper+'\nstruct Column {MynnArena &arena;static constexpr float cphm_unst=16.f,cphh_unst=16.f;\n'+'\n'.join(function(n) for n in ('f_phim','f_phih','f_scale_aware'))+'\n};\n'
    source+='''extern "C" __global__ void host_stability(int n,const float* zet,const float* dx,const float* pblh,float* phim,float* phih,float* psig_bl,float* psig_shcu){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;MynnArena a;Column col{a};col.f_scale_aware(dx[i],pblh[i],psig_bl[i],psig_shcu[i]);phim[i]=col.f_phim(zet[i]);phih[i]=col.f_phih(zet[i]);}'''
    return HostModule('mpas_v842_mynn_stability',source).get_function('host_stability')


def main():
    host=host_module();n=257;rng=np.random.default_rng(842)
    zet=rng.uniform(-2,2,n).astype(np.float32);zet[:7]=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x3f800000],np.uint32).view(np.float32)
    dx=rng.uniform(10,10000,n).astype(np.float32);pblh=rng.uniform(20,5000,n).astype(np.float32)
    zet[7:9]=np.array([0x7f7fffff,0xff7fffff],np.uint32).view(np.float32)
    dx[7:9]=np.array([0x7f7fffff,0x00800000],np.uint32).view(np.float32)
    pblh[7:9]=np.array([0x00800000,0x7f7fffff],np.uint32).view(np.float32)
    names=('phim','phih','psig_bl','psig_shcu');expected={name:np.empty(n,np.float32) for name in names}
    host(((n+127)//128,),(128,),(n,zet,dx,pblh,*expected.values()))
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        actual={name:xp.empty((n,),np.float32) for name in names}
        stability(xp.asarray(zet),xp.asarray(dx),xp.asarray(pblh),actual)
        for name in names:
            got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
            if bad.size:raise AssertionError(f'{name} {bad.size} differences at {bad[0]} actual {got[bad[0]]} expected {want[bad[0]]}')
        badzet=zet.copy();badzet[0]=np.inf
        try:stability(xp.asarray(badzet),xp.asarray(dx),xp.asarray(pblh),actual)
        except ValueError:pass
        else:raise AssertionError('nonfinite stability input accepted')
        for name in names:
            if not np.array_equal(actual[name].to_numpy().view(np.uint32),expected[name].view(np.uint32)):raise AssertionError('refusal changed '+name)
    print('MYNN surface and scale-aware factors match selected source for stable, unstable and subnormal states',flush=True)

if __name__=='__main__':main()
