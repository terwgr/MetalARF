"""Compare native MYNN tendencies, solves and moisture carry with host source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import tendencies
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
    s=re.sub(r'for \(int mynn_element=0; mynn_element < \((\w+)\.slice\(MynnRange\{.*?\}\)\)\.size\(\); \+\+mynn_element\) \{ \1\.slice\(MynnRange\{.*?\}\)\.linear\(mynn_element\) = 0.0f; \}',r'for(int i=kts;i<=kte;i++) { \1(i)=0.0f; }',s)
    return m.group(),s


def host_module():
    helper='''
template<int R> struct MynnIndices {int value[R];};
struct MynnArena {float mem[65536];int offset;};
struct MynnFrame {MynnArena& a;int saved;MynnFrame(MynnArena& v):a(v),saved(v.offset){} ~MynnFrame(){a.offset=saved;}};
template<class T,int R> struct MynnArray {T* data;int count;void rebase(MynnIndices<R>){};T& operator()(int i){return data[i-1];} int size(){return count;} T& linear(int i){return data[i];}};
template<class T,int R> MynnArray<T,R> mynn_array(MynnArena& a,MynnIndices<R> lo,MynnIndices<R> hi){int n=hi.value[0]-lo.value[0]+1;T* p=a.mem+a.offset;a.offset+=n;return {p,n};}
struct MynnText {const char* p; MynnText& operator=(const char* q){p=q;return *this;} bool operator==(const char* q){return *p==*q;}};
template<class T> struct MynnOptional {bool present()const{return false;} operator T()const{return T{"b"};}};
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}
__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_sqrt(float a){return mynn_ieee_result(a,a,mpas_sqrt(a));}
template<int N> __device__ float mynn_ipow(float a){static_assert(N==2);return mpas_mul(a,a);}
'''
    sig,tend=selected('f_mynn_tendencies');profiles=re.findall(r'MynnArray<float, 1> (\w+)',sig)
    columns=('psfc','ust','flt','flq','flqv','flqc','wspd','uoce','voce')
    flags=('flag_qc','flag_qi','flag_qnc','flag_qni','flag_qs','flag_qnwfa','flag_qnifa','flag_qnbca','flag_ozone','bl_mynn_cloudmix','bl_mynn_mixqt','bl_mynn_edmf_mom','bl_mynn_mixscalars')
    constants=('delt','r_d','p608','xlvcp','xlscp','t0c')
    const='\n'.join(re.search(r'static constexpr float '+k+r'\s*=.*?;',SOURCE).group() for k in ['j'+str(i) for i in range(9)]+['k'+str(i) for i in range(9)]+['tice'])
    src='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+helper+'struct Column{MynnArena &arena;float r_d,p608,xlvcp,xlscp,t0c;static constexpr bool debug_code=false;'+const+'\n'+selected('f_qsat_blend','float')[1]+selected('f_tridiag2')[1]+selected('f_moisture_check')[1]+tend+'};\n'
    decl=','.join('float* '+n for n in profiles+list(columns));constargs=','.join('float '+n for n in constants);flagargs=','.join('int '+n for n in flags)
    raw=sig[sig.index('(')+1:sig.rindex(')')];args=[]
    for arg in re.split(r',(?![^<]*>)',raw):
        a=arg.strip();name=a.split()[-1]
        if name=='kts':args.append('1')
        elif name=='kte':args.append('nl')
        elif name in profiles:args.append('MynnArray<float,1>{'+name+'+col*nl,nl}')
        elif name in columns:args.append(name+'[col]')
        elif name=='delt':args.append(name)
        elif name in flags:args.append('bool('+name+')')
        else:raise ValueError(a)
    src+=f'extern "C" __global__ void host_tend(int nc,int nl,{constargs},{flagargs},{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena a{{{{}},0}};Column obj{{a,r_d,p608,xlvcp,xlscp,t0c}};obj.f_mynn_tendencies('+','.join(args)+');}\n'
    return HostModule('mpas_v842_mynn_tendencies',src).get_function('host_tend'),profiles,columns,flags,constants


def main():
    host,profiles,columns,flags,constants=host_module();rng=np.random.default_rng(842)
    const=dict(delt=30.,r_d=287.,p608=.61,xlvcp=2490.,xlscp=2819.,t0c=273.15)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (2,3,8,17):
            nc=5;shape=(nc,nl);fields={n:rng.uniform(0,.0001,shape).astype(np.float32) for n in profiles}
            fields['dz']=rng.uniform(60,150,shape).astype(np.float32);fields['rho']=rng.uniform(.5,1.2,shape).astype(np.float32)
            fields['p']=np.broadcast_to(np.linspace(100000,30000,nl,dtype=np.float32),shape).copy();fields['exner']=rng.uniform(.7,1,shape).astype(np.float32)
            for n in ('th','tk','thl'):fields[n]=rng.uniform(250,300,shape).astype(np.float32)
            for n in ('u','v'):fields[n]=rng.uniform(2,10,shape).astype(np.float32)
            for n in ('qv','sqv','sqw'):fields[n]=rng.uniform(.002,.01,shape).astype(np.float32)
            for n in ('dfh','dfm','dfq'):fields[n]=rng.uniform(.001,.05,shape).astype(np.float32)
            for n in columns:fields[n]=np.full(nc,.1,np.float32)
            fields['psfc']=np.linspace(90000,100000,nc,dtype=np.float32);fields['ust']=np.linspace(.1,.5,nc,dtype=np.float32);fields['wspd']=np.linspace(2,10,nc,dtype=np.float32)
            for mode in ('none','cloud','all','mixqt'):
                opts={n:False for n in flags}
                if mode!='none':opts.update(bl_mynn_cloudmix=True,flag_qc=True,flag_qi=True)
                if mode=='all':opts.update({n:True for n in flags if n not in ('bl_mynn_mixqt','bl_mynn_edmf_mom')})
                if mode=='mixqt':opts['bl_mynn_mixqt']=True
                expected={n:v.copy() for n,v in fields.items()};actual={n:xp.asarray(v.copy()) for n,v in fields.items()}
                host((1,),(128,),(nc,nl,*[np.float32(const[n]) for n in constants],*[int(opts[n]) for n in flags],*[expected[n] for n in profiles+list(columns)]))
                tendencies(actual,constants=const,flags=opts)
                for name in profiles+list(columns):
                    got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                    if bad.size:raise AssertionError(f'levels={nl} mode={mode} {name}: {bad.size} differences at {bad[0]} actual {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
                print(f'native MYNN tendencies levels={nl} mode={mode} matches selected source',flush=True)

if __name__=='__main__':main()
