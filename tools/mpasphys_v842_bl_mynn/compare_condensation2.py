"""Compare native MYNN cloud-PDF selector 2 with selected host source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.resources_build import _vendored_maths,_cuda_core_math
from woof.hex.mpasphys.metal_bl_mynn_v842 import condensation2
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def selected(name,ret='void'):
    m=re.search(r'__device__ __noinline__ '+ret+' '+name+r'\s*\(.*?\)\s*\{',SOURCE,re.S);assert m
    j=m.end();depth=1
    while depth:
        if SOURCE[j]=='{':depth+=1
        elif SOURCE[j]=='}':depth-=1
        j+=1
    s=SOURCE[m.start():j]
    if name=='f_mym_condensation':
        sw=s.index('switch (bl_mynn_cloudpdf)');case=s.index('case 2: case (-(2)): {',sw)+len('case 2: case (-(2)): {');end=s.index('break; } }',case)
        s=s[:sw]+s[case:end]+s[end+len('break; } }'):]
    return s


def host_module():
    helper='''
template<int R> struct MynnIndices {int value[R];};
struct MynnArena {float mem[32768];int offset;};
struct MynnFrame {MynnArena& a;int saved;MynnFrame(MynnArena& v):a(v),saved(v.offset){} ~MynnFrame(){a.offset=saved;}};
template<class T,int R> struct MynnArray {T* data;int count;void rebase(MynnIndices<R>){};T& operator()(int i){return data[i-1];}};
template<class T,int R> MynnArray<T,R> mynn_array(MynnArena& a,MynnIndices<R> lo,MynnIndices<R> hi){int n=hi.value[0]-lo.value[0]+1;T* p=a.mem+a.offset;a.offset+=n;return {p,n};}
struct MynnText {const char* p; MynnText& operator=(const char* q){p=q;return *this;} bool operator==(const char* q){return *p==*q;}};
template<class T> struct MynnOptional {bool present()const{return false;} operator T()const{return T{"b"};}};
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}
__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ int mynn_max(int a,int b){return a>b?a:b;}
__device__ float mynn_sqrt(float a){return mynn_ieee_result(a,a,mpas_sqrt(a));}
__device__ double mynn_sqrt(double a){return mynn_ieee_result(a,a,__dsqrt_rn(a));}
__device__ double mynn_real64(float a){return (double)a;}
__device__ float mynn_real32(double a){return (float)a;}
__device__ float mynn_pow(float a,float b){return mynn_ieee_result(a,b,gfk_pow(a,b));}
__device__ float mynn_exp(float a){return mynn_ieee_result(a,a,gfk_exp(a));}
__device__ float mynn_log10(float a){return mynn_ieee_result(a,a,p12_core_log10f::cr_log10f(a));}
__device__ float mynn_atan(float a){return mynn_ieee_result(a,a,glibc_atanf(a));}
template<int N> __device__ float mynn_ipow(float a){static_assert(N==2);return mpas_mul(a,a);}
'''
    root=files('woof.core.kernels');core=_cuda_core_math(_vendored_maths()['core-math/log10f.c'],'log10f')
    consts=['j'+str(i) for i in range(9)]+['k'+str(i) for i in range(9)]+['tice','tliq']
    const='\n'.join(re.search(r'static constexpr float '+k+r'\s*=.*?;',SOURCE).group() for k in consts)
    names=('dz','zw','thl','qw','qv','qc','qi','qs','p','exner','tsq','qsq','cov','sh','el','th','rstoch_col','qc_bl1d','qi_bl1d','cldfra_bl1d','vt','vq','sgm','xland','pblh1','hfx1','rmo')
    scalars=('cp','cpv','cice','cliq','ep_2','r_d','r_v','xls','xlv','t0c','xlvcp','ep1')
    src='#define __CUDA_ARCH__ 1\n#define p12_expect(a,b) (a)\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+root.joinpath('glibc_flt32.cuh').read_text()+root.joinpath('glibc_trig_flt32.cuh').read_text()+core+helper+'\nstruct Column {MynnArena &arena;'+''.join('float '+n+';' for n in scalars)+'float tv0;\n'+const+'\n'+''.join(selected(n,'float')+'\n' for n in ('f_qsat_blend','f_xl_blend'))+selected('f_mym_condensation')+'};\n'
    decl=','.join('float* '+n for n in names);params=','.join('float '+n for n in scalars);arr=lambda n:'MynnArray<float,1>{'+n+'+col*nl,nl}'
    args=['1','nl','1000.f',arr('dz'),arr('zw'),'xland[col]',arr('thl'),arr('qw'),arr('qv'),arr('qc'),arr('qi'),arr('qs'),arr('p'),arr('exner'),arr('tsq'),arr('qsq'),arr('cov'),arr('sh'),arr('el'),'2',arr('qc_bl1d'),arr('qi_bl1d'),arr('cldfra_bl1d'),'pblh1[col]','hfx1[col]',arr('vt'),arr('vq'),arr('th'),arr('sgm'),'rmo[col]','0',arr('rstoch_col')]
    src+=f'extern "C" __global__ void host_cond(int nc,int nl,{params},{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena a{{{{}},0}};Column obj{{a,'+','.join(scalars)+',mpas_mul(ep1,300.f)};obj.f_mym_condensation('+','.join(args)+');}\n'
    return HostModule('mpas_v842_mynn_condensation2',src).get_function('host_cond'),names,scalars


def main():
    host,names,scalars=host_module();rng=np.random.default_rng(842)
    constants=dict(cp=1004.,cpv=1850.,cice=2106.,cliq=4190.,ep_2=.622,r_d=287.,r_v=461.,xls=2.83e6,xlv=2.5e6,t0c=273.15,xlvcp=2.5e6/1004.,ep1=.61)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (3,4,8,33):
            nc=7;shape=(nc,nl);f={n:rng.uniform(.01,.4,shape).astype(np.float32) for n in names if n not in ('xland','pblh1','hfx1','rmo')}
            f['dz']=rng.uniform(50,150,shape).astype(np.float32);f['zw']=np.broadcast_to(np.linspace(0,12000,nl,dtype=np.float32),shape).copy()
            f['p']=np.broadcast_to(np.linspace(100000,18000,nl,dtype=np.float32),shape).copy();f['exner']=np.ones(shape,np.float32)
            f['th']=rng.uniform(235,295,shape).astype(np.float32)
            f['th'][:3,:]=np.array([240.,269.,273.15],np.float32)[:,None]
            f['thl']=f['th'].copy();f['qw']=rng.uniform(.0001,.012,shape).astype(np.float32)
            for n in ('qv','qc','qi','qs'):f[n]=rng.uniform(1.e-8,2.e-4,shape).astype(np.float32)
            f['qsq']=rng.uniform(1.e-9,5.e-6,shape).astype(np.float32);f['el']=rng.uniform(2,60,shape).astype(np.float32)
            f['xland']=np.array([1,2,1,2,1,2,1],np.float32);f['pblh1']=np.linspace(300,1500,nc,dtype=np.float32);f['hfx1']=np.linspace(-20,200,nc,dtype=np.float32);f['rmo']=np.linspace(-.01,.01,nc,dtype=np.float32)
            expected={n:v.copy() for n,v in f.items()};actual={n:xp.asarray(v.copy()) for n,v in f.items()}
            host((1,),(128,),(nc,nl,*[np.float32(constants[n]) for n in scalars],*[expected[n] for n in names]))
            condensation2(actual,constants=constants)
            for name in names:
                got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'levels={nl} {name}: {bad.size} differences at {bad[0]} actual {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
            print(f'native MYNN cloud-PDF 2 condensation levels={nl} matches selected source',flush=True)

if __name__=='__main__':main()
