"""Compare native MYNN EDMF plume condensation with host source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import edmf_condensation
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

def extract(name,ret):
    m=re.search(r'__device__ __noinline__ '+ret+' '+name+r'\s*\(.*?\)\s*\{',SOURCE,re.S);assert m
    j=m.end();d=1
    while d:
        if SOURCE[j]=='{':d+=1
        elif SOURCE[j]=='}':d-=1
        j+=1
    return SOURCE[m.start():j]

def main():
    helper='''
struct MynnArena{};struct MynnFrame{MynnFrame(MynnArena&) {}};
struct MynnText {const char* p; MynnText& operator=(const char* q){p=q;return *this;} bool operator==(const char* q){return *p==*q;}};
template<class T> struct MynnOptional {bool present()const{return false;} operator T()const{return T{"b"};}};
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_abs(float a){return __uint_as_float(__float_as_uint(a)&0x7fffffffu);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}
__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_pow(float a,float b){return mynn_ieee_result(a,b,gfk_pow(a,b));}
'''
    names=('qt','thl','p','zagl','thv','qc');params=('p1000mb','rcp','xlvcp','t0c','rvovrd')
    const='\n'.join(re.search(r'static constexpr float '+k+r'\s*=.*?;',SOURCE).group() for k in ['j'+str(i) for i in range(9)]+['k'+str(i) for i in range(9)]+['tice'])
    src='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+files('woof.core.kernels').joinpath('glibc_flt32.cuh').read_text()+helper+'struct Column {MynnArena &arena;'+''.join('float '+n+';' for n in params)+const+extract('f_qsat_blend','float')+extract('f_condensation_edmf','void')+'};\n'
    decl=','.join('float* '+n for n in names);par=','.join('float '+n for n in params)
    src+=f'extern "C" __global__ void host_edmf(int n,{par},{decl}){{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;MynnArena a;Column obj{{a,'+','.join(params)+'};obj.f_condensation_edmf('+','.join(n+'[i]' for n in names)+');}\n'
    host=HostModule('mpas_v842_mynn_edmf_cond',src).get_function('host_edmf');rng=np.random.default_rng(842);n=2048
    data={'qt':rng.uniform(0,.03,n).astype(np.float32),'thl':rng.uniform(230,310,n).astype(np.float32),'p':rng.uniform(20000,100000,n).astype(np.float32),'zagl':rng.uniform(0,20000,n).astype(np.float32),'thv':rng.uniform(200,300,n).astype(np.float32),'qc':rng.uniform(0,.002,n).astype(np.float32)}
    data['zagl'][:5]=np.array([0,99.99999,100,100.00001,200],np.float32)
    values=dict(p1000mb=100000.,rcp=287./1004.,xlvcp=2490.,t0c=273.15,rvovrd=461./287.)
    expected={n:v.copy() for n,v in data.items()}
    host(((n+127)//128,),(128,),(n,*[np.float32(values[k]) for k in params],*[expected[k] for k in names]))
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);actual={k:xp.asarray(v.copy()) for k,v in data.items()}
        edmf_condensation(actual,constants=values)
        for key in names:
            got=actual[key].to_numpy();want=expected[key];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
            if bad.size:raise AssertionError(f'{key}: {bad.size} differences at {bad[0]} got {got[bad[0]]} want {want[bad[0]]}')
    print(f'MYNN EDMF plume condensation matches selected source for {n} cases',flush=True)

if __name__=='__main__':main()
