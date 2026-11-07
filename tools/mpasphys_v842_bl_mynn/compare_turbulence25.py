"""Compare native MYNN closure 2.5 turbulence to selected host source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT, MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.resources_build import _vendored_maths,_cuda_core_math
from woof.hex.mpasphys.metal_bl_mynn_v842 import turbulence25
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


def drop_block(s,needle,keep_else=False):
    i=s.index(needle);o=s.index('{',i);j=o+1;depth=1
    while depth:
        if s[j]=='{':depth+=1
        elif s[j]=='}':depth-=1
        j+=1
    if keep_else:
        tail=re.match(r'\s*else\s*\{',s[j:]);assert tail
        o2=j+tail.end()-1;k=o2+1;depth=1
        while depth:
            if s[k]=='{':depth+=1
            elif s[k]=='}':depth-=1
            k+=1
        return s[:i]+s[o2+1:k-1]+s[k:]
    return s[:i]+s[j:]

def turbulence_source():
    s=selected('f_mym_turbulence')
    s=drop_block(s,'if (((closure) >= (3.0f)))',True)
    s=drop_block(s,'if (bl_mynn_tkebudget)')
    s=drop_block(s,'if (((spp_pbl) == (1)))')
    while 'if (debug_code)' in s:s=drop_block(s,'if (debug_code)')
    return s

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
__device__ double mynn_real64(float a){return (double)a;}
__device__ float mynn_real32(double a){return (float)a;}
__device__ double mynn_dadd(double a,double b){return mynn_ieee_result(a,b,__dadd_rn(a,b));}
__device__ double mynn_dsub(double a,double b){return mynn_ieee_result(a,b,__dsub_rn(a,b));}
__device__ double mynn_dmul(double a,double b){return mynn_ieee_result(a,b,__dmul_rn(a,b));}
__device__ double mynn_ddiv(double a,double b){return mynn_ieee_result(a,b,__ddiv_rn(a,b));}
__device__ double mynn_sqrt(double a){return mynn_ieee_result(a,a,__dsqrt_rn(a));}
__device__ double mynn_neg(double a){return -a;}
__device__ double mynn_max(double a,double b){return a>b?a:b;}
template<int N> __device__ double mynn_ipow(double a){static_assert(N==2);return mynn_dmul(a,a);}
template<int N> __device__ float mynn_ipow(float a){static_assert(N==2);return mpas_mul(a,a);}
'''
    root=files('woof.core.kernels');core=_cuda_core_math(_vendored_maths()['core-math/tanhf.c'],'tanhf')
    constants=('b1','b2','twothirds','qkemin','pr','g1','c2','c3','c5','a1','c1','a2','g2','ckmod','cc2','cc3','e1c','e2c','e3c','e4c','e5c')
    const='\n'.join(re.search(r'static constexpr float '+k+r'\s*=.*?;',SOURCE).group() for k in constants)
    src='#define __CUDA_ARCH__ 1\n#define p12_expect(a,b) (a)\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+root.joinpath('glibc_flt32.cuh').read_text()+core+helper+'\nstruct Column {MynnArena &arena;float grav,gtr,karman,tv0;static constexpr float qmin=0.f,zmax=1.f,onethird=1.f/3.f;\n'+const+'\n'+''.join(selected(n)+'\n' for n in ('f_mym_level2','f_mym_length'))+turbulence_source()+'};\n'
    names=('dz','u','v','thl','thetav','qw','ql','vt','vq','qke','tsq','qsq','cov','theta','cldfra_bl1d','edmf_w1','edmf_a1','tkeprodtd','sh','sm','el','dfm','dfh','dfq','tcd','qcd','pdk','pdt','pdq','pdc','qshear1d','qbuoy1d','zw','zi','rmo','fltv','psig_bl')
    decl=','.join('float* '+n for n in names);arr=lambda n:'MynnArray<float,1>{'+n+'+col*nl,nl}';zw='MynnArray<float,1>{zw+col*(nl+1),nl+1}'
    args=['1','nl','1.f','2.5f',arr('dz'),'1000.f',zw,arr('u'),arr('v'),arr('thl'),arr('thetav'),arr('ql'),arr('qw'),arr('qke'),arr('tsq'),arr('qsq'),arr('cov'),arr('vt'),arr('vq'),'rmo[col]','0.f','fltv[col]','0.f','zi[col]',arr('theta'),arr('sh'),arr('sm'),arr('el'),arr('dfm'),arr('dfh'),arr('dfq'),arr('tcd'),arr('qcd'),arr('pdk'),arr('pdt'),arr('pdq'),arr('pdc'),arr('dz'),arr('qshear1d'),arr('qbuoy1d'),arr('dz'),'false','psig_bl[col]','0.f',arr('cldfra_bl1d'),'2',arr('edmf_w1'),arr('edmf_a1'),arr('tkeprodtd'),'0',arr('dz')]
    src+=f'extern "C" __global__ void host_init(int nc,int nl,int flag,{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena a{{{{}},0}};Column obj{{a,9.81f,mpas_div(9.81f,300.f),.4f,mpas_mul(.61f,300.f)}};obj.f_mym_turbulence('+','.join(args)+');}\n'
    return HostModule('mpas_v842_mynn_turbulence25',src).get_function('host_init'),names


def main():
    host,names=host_module();rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (2,3,8,33):
            nc=7;shape=(nc,nl)
            f={n:rng.uniform(.01,.4,shape).astype(np.float32) for n in names if n not in ('zw','zi','rmo','fltv','psig_bl')}
            f['dz']=rng.uniform(20,180,shape).astype(np.float32);f['u']*=20;f['v']*=20
            for n in ('thl','theta','thetav'):f[n]=rng.uniform(270,310,shape).astype(np.float32)
            f['zw']=np.broadcast_to(np.linspace(0,5000,nl+1,dtype=np.float32),(nc,nl+1)).copy()
            f['zi']=np.linspace(100,1200,nc,dtype=np.float32);f['rmo']=np.array([0,-.001,.001,-.01,.01,0,.0001],np.float32)
            f['fltv']=np.array([0,.01,-.01,.05,.2,-.02,.001],np.float32);f['psig_bl']=np.linspace(.1,.95,nc,dtype=np.float32)
            for flag in (False,):
                expected={n:v.copy() for n,v in f.items()};actual={n:xp.asarray(v.copy()) for n,v in f.items()}
                host((1,),(128,),(nc,nl,int(flag),*[expected[n] for n in names]))
                turbulence25(actual,gravity=9.81,karman=.4,ep1=.61)
                for name in names:
                    got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                    if bad.size:raise AssertionError(f'levels={nl} flag={flag} {name}: {bad.size} differences at {bad[0]} actual {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
                before={n:actual[n].to_numpy().copy() for n in ('sh','sm','el','dfm','dfh','dfq','tcd','qcd','pdk','pdt','pdq','pdc')}
                actual['zw']=xp.zeros((nc,nl+1),np.float32)
                try:turbulence25(actual,gravity=9.81,karman=.4,ep1=.61)
                except ValueError:pass
                else:raise AssertionError('unsafe turbulence height scan accepted')
                for n in before:
                    if not np.array_equal(actual[n].to_numpy().view(np.uint32),before[n].view(np.uint32)):raise AssertionError('invalid scan mutated '+n)
                print(f'native MYNN turbulence closure 2.5 levels={nl} matches selected source; invalid scan refused',flush=True)

if __name__=='__main__':main()
