"""Compare native default MYNN mixing length with its selected source branch."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.resources_build import _vendored_maths,_cuda_core_math
from woof.hex.mpasphys.metal_bl_mynn_v842 import mixing_length2
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def host_module():
    m=re.search(r'__device__ __noinline__ void f_mym_length\s*\(.*?\)\s*\{',SOURCE,re.S);assert m
    start=m.end();j=start;depth=1
    while depth:
        if SOURCE[j]=='{':depth+=1
        elif SOURCE[j]=='}':depth-=1
        j+=1
    full=SOURCE[start:j-1];sw=full.index('switch (bl_mynn_mixlength)');case=full.index('case 2: {',sw)+len('case 2: {');end=full.index('break; } }',case)
    selected=SOURCE[m.start():start]+full[:sw]+full[case:end]+'\n}'
    helper='''
template<int R> struct MynnIndices {int value[R];};
struct MynnArena {float mem[8192];int offset;};
struct MynnFrame {MynnArena& a;int saved;MynnFrame(MynnArena& v):a(v),saved(v.offset){} ~MynnFrame(){a.offset=saved;}};
template<class T,int R> struct MynnArray {T* data;int count;void rebase(MynnIndices<R>){};T& operator()(int i){return data[i-1];}};
template<class T,int R> MynnArray<T,R> mynn_array(MynnArena& a,MynnIndices<R> lo,MynnIndices<R> hi){int n=hi.value[0]-lo.value[0]+1;T* p=a.mem+a.offset;a.offset+=n;return {p,n};}
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}
__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_sqrt(float a){return mynn_ieee_result(a,a,mpas_sqrt(a));}
__device__ float mynn_pow(float a,float b){return mynn_ieee_result(a,b,gfk_pow(a,b));}
__device__ float mynn_tanh(float a){return mynn_ieee_result(a,a,p12_core_tanhf::cr_tanhf(a));}
template<int N> __device__ float mynn_ipow(float a){static_assert(N==2);return mpas_mul(a,a);}
'''
    root=files('woof.core.kernels');core=_cuda_core_math(_vendored_maths()['core-math/tanhf.c'],'tanhf')
    source='#define __CUDA_ARCH__ 1\n#define p12_expect(a,b) (a)\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+root.joinpath('glibc_flt32.cuh').read_text()+core+helper+'\nstruct Column {MynnArena &arena;float grav,gtr,karman;static constexpr float qmin=0.f,zmax=1.f,onethird=1.f/3.f;\n'+selected+'\n};\n'
    names=('dz','u1','v1','qke','dtv','cldfra_bl1d','edmf_w1','edmf_a1','zw','zi','rmo','fltv','psig_bl','qkw','el')
    decl=','.join('float* '+n for n in names);arr=lambda n:'MynnArray<float,1>{'+n+'+col*nl,nl}'
    zw='MynnArray<float,1>{zw+col*(nl+1),nl+1}'
    args=['1','nl','1.f',arr('dz'),'1000.f',zw,'rmo[col]','0.f','fltv[col]','0.f',arr('dz'),arr('dz'),arr('u1'),arr('v1'),arr('qke'),arr('dtv'),arr('el'),'zi[col]',arr('dz'),arr('qkw'),'psig_bl[col]',arr('cldfra_bl1d'),'2',arr('edmf_w1'),arr('edmf_a1')]
    source+=f'extern "C" __global__ void host_length(int nc,int nl,float gravity,float karman,{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena a{{{{}},0}};Column obj{{a,gravity,mpas_div(gravity,300.f),karman}};obj.f_mym_length('+','.join(args)+');}\n'
    return HostModule('mpas_v842_mynn_length2',source).get_function('host_length'),names


def main():
    host,names=host_module();rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (2,3,8,33):
            nc=7;shape=(nc,nl);fields={name:rng.uniform(.01,.4,shape).astype(np.float32) for name in names if name not in ('zw','zi','rmo','fltv','psig_bl')}
            fields['dz']=rng.uniform(20,180,shape).astype(np.float32);fields['u1']*=20;fields['v1']*=20
            fields['dtv']=rng.uniform(-.02,.02,shape).astype(np.float32)
            fields['zw']=np.broadcast_to(np.linspace(0,5000,nl+1,dtype=np.float32),(nc,nl+1)).copy()
            fields['zi']=np.linspace(100,1200,nc,dtype=np.float32);fields['rmo']=np.array([0,-.001,.001,-.01,.01,0,.0001],np.float32)
            fields['fltv']=np.array([0,.01,-.01,.05,.2,-.02,.001],np.float32);fields['psig_bl']=np.linspace(.1,.95,nc,dtype=np.float32)
            expected={n:v.copy() for n,v in fields.items()};actual={n:xp.asarray(v.copy()) for n,v in fields.items()}
            host((1,),(128,),(nc,nl,np.float32(9.81),np.float32(.4),*[expected[n] for n in names]))
            mixing_length2(actual,gravity=9.81,karman=.4)
            for name in names:
                got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'levels={nl} {name}: {bad.size} differences at {bad[0]} actual {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
            before={n:actual[n].to_numpy().copy() for n in ('qkw','el')};actual['zw']=xp.zeros((nc,nl+1),np.float32)
            try:mixing_length2(actual,gravity=9.81,karman=.4)
            except ValueError:pass
            else:raise AssertionError('unsafe mixing-length height scan accepted')
            for n in before:
                if not np.array_equal(actual[n].to_numpy().view(np.uint32),before[n].view(np.uint32)):raise AssertionError('invalid scan mutated '+n)
            print(f'native MYNN mixing length selector 2 levels={nl} matches source; unsafe scan refused',flush=True)

if __name__=='__main__':main()
