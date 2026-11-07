"""Compare the native five-pass MYNN initialization to selected host source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT, MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.resources_build import _vendored_maths,_cuda_core_math
from woof.hex.mpasphys.metal_bl_mynn_v842 import initialize
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def selected(name):
    m=re.search(r'__device__ __noinline__ void '+name+r'\s*\(.*?\)\s*\{',SOURCE,re.S);assert m
    j=m.end();depth=1
    while depth:
        if SOURCE[j]=='{':depth+=1
        elif SOURCE[j]=='}':depth-=1
        j+=1
    if name!='f_mym_length':return SOURCE[m.start():j]
    full=SOURCE[m.end():j-1];sw=full.index('switch (bl_mynn_mixlength)');case=full.index('case 2: {',sw)+len('case 2: {');end=full.index('break; } }',case)
    return SOURCE[m.start():m.end()]+full[:sw]+full[case:end]+'\n}'


def host_module():
    helper='''
template<int R> struct MynnIndices {int value[R];};
struct MynnArena {float mem[32768];int offset;};
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
    constants=('b1','b2','twothirds','qkemin','pr','g1','c2','c3','c5','a1','c1','a2','g2','ckmod')
    const='\n'.join(re.search(r'static constexpr float '+k+r'\s*=.*?;',SOURCE).group() for k in constants)
    src='#define __CUDA_ARCH__ 1\n#define p12_expect(a,b) (a)\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+root.joinpath('glibc_flt32.cuh').read_text()+core+helper+'\nstruct Column {MynnArena &arena;float grav,gtr,karman,tv0;static constexpr float qmin=0.f,zmax=1.f,onethird=1.f/3.f;\n'+const+'\n'+''.join(selected(n)+'\n' for n in ('f_mym_level2','f_mym_length','f_mym_initialize'))+'};\n'
    names=('dz','u','v','thl','qw','theta','thetav','cldfra_bl1d','edmf_w1','edmf_a1','sh','sm','el','qke','tsq','qsq','cov','zw','zi','rmo','ust','psig_bl')
    decl=','.join('float* '+n for n in names);arr=lambda n:'MynnArray<float,1>{'+n+'+col*nl,nl}';zw='MynnArray<float,1>{zw+col*(nl+1),nl+1}'
    args=['1','nl','1.f',arr('dz'),'1000.f',zw,arr('u'),arr('v'),arr('thl'),arr('qw'),'zi[col]',arr('theta'),arr('thetav'),arr('sh'),arr('sm'),'ust[col]','rmo[col]',arr('el'),arr('qke'),arr('tsq'),arr('qsq'),arr('cov'),'psig_bl[col]',arr('cldfra_bl1d'),'2',arr('edmf_w1'),arr('edmf_a1'),'bool(flag)','0',arr('dz')]
    src+=f'extern "C" __global__ void host_init(int nc,int nl,int flag,{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena a{{{{}},0}};Column obj{{a,9.81f,mpas_div(9.81f,300.f),.4f,mpas_mul(.61f,300.f)}};obj.f_mym_initialize('+','.join(args)+');}\n'
    return HostModule('mpas_v842_mynn_initialize',src).get_function('host_init'),names


def main():
    host,names=host_module();rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (2,3,8,33):
            nc=7;shape=(nc,nl)
            f={n:rng.uniform(.01,.4,shape).astype(np.float32) for n in names if n not in ('zw','zi','rmo','ust','psig_bl')}
            f['dz']=rng.uniform(20,180,shape).astype(np.float32);f['u']*=20;f['v']*=20
            for n in ('thl','theta','thetav'):f[n]=rng.uniform(270,310,shape).astype(np.float32)
            f['zw']=np.broadcast_to(np.linspace(0,5000,nl+1,dtype=np.float32),(nc,nl+1)).copy()
            f['zi']=np.linspace(100,1200,nc,dtype=np.float32);f['rmo']=np.array([0,-.001,.001,-.01,.01,0,.0001],np.float32)
            f['ust']=np.linspace(.05,.6,nc,dtype=np.float32);f['psig_bl']=np.linspace(.1,.95,nc,dtype=np.float32)
            for flag in (False,True):
                expected={n:v.copy() for n,v in f.items()};actual={n:xp.asarray(v.copy()) for n,v in f.items()}
                host((1,),(128,),(nc,nl,int(flag),*[expected[n] for n in names]))
                initialize(actual,gravity=9.81,karman=.4,ep1=.61,initialize_qke=flag)
                for name in names:
                    got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                    if bad.size:raise AssertionError(f'levels={nl} flag={flag} {name}: {bad.size} differences at {bad[0]} actual {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
                before={n:actual[n].to_numpy().copy() for n in ('sh','sm','el','qke','tsq','qsq','cov')}
                actual['zw']=xp.zeros((nc,nl+1),np.float32)
                try:initialize(actual,gravity=9.81,karman=.4,ep1=.61,initialize_qke=flag)
                except ValueError:pass
                else:raise AssertionError('unsafe initialization height scan accepted')
                for n in before:
                    if not np.array_equal(actual[n].to_numpy().view(np.uint32),before[n].view(np.uint32)):raise AssertionError('invalid scan mutated '+n)
                print(f'native MYNN initialization levels={nl} qke={flag} matches selected source; invalid scan refused',flush=True)

if __name__=='__main__':main()
