"""Compare native MYNN eight-plume mass flux with selected host source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.resources_build import _vendored_maths,_cuda_core_math
from woof.hex.mpasphys.metal_bl_mynn_v842 import dmp
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

def selected(name,ret='void'):
    m=re.search(r'__device__ __noinline__ '+ret+' '+name+r'\s*\(.*?\)\s*\{',SOURCE,re.S);assert m
    j=m.end();depth=1
    while depth:
        if SOURCE[j]=='{':depth+=1
        elif SOURCE[j]=='}':depth-=1
        j+=1
    return m.group(),SOURCE[m.start():j]
def remove(s,needle):
    i=s.index(needle);o=s.index('{',i);j=o+1;d=1
    while d:
        if s[j]=='{':d+=1
        elif s[j]=='}':d-=1
        j+=1
    return s[:i]+s[j:]
def host_module():
    sig,body=selected('f_dmp_mf')
    while 'if (mix_chem)' in body:body=remove(body,'if (mix_chem)')
    while 'if (((debug_mf) == (1)))' in body:body=remove(body,'if (((debug_mf) == (1)))')
    body=re.sub(r'MynnArray<float, [123]> (?:chemn|upchem|edmf_chem) = mynn_array[^\n]*\n','',body)
    helper='''
template<int R> struct MynnIndices{int value[R];};struct MynnRange{int lo,hi;};
struct MynnArena{unsigned int mem[1048576];int offset;};
struct MynnFrame{MynnArena &a;int save;MynnFrame(MynnArena &v):a(v),save(v.offset){}~MynnFrame(){a.offset=save;}};
template<class T,int R> struct MynnArray{
T* data;int n1,n2;int lo[2],hi[2];
void rebase(MynnIndices<R> x){lo[0]=x.value[0];if(R>1)lo[1]=x.value[1];}
T& operator()(int i){return data[i-lo[0]];}
T& operator()(int i,int j){return data[i-lo[0]+n1*(j-lo[1])];}
int size(){return n1*n2;}T& linear(int i){return data[i];}
MynnArray<T,1> slice(MynnRange x){return {data+(x.lo-lo[0]),x.hi-x.lo+1,1,{x.lo,1},{x.hi,1}};}
};
template<class T,int R> MynnArray<T,R> mynn_array(MynnArena &a,MynnIndices<R> lo,MynnIndices<R> hi){int n1=hi.value[0]-lo.value[0]+1,n2=R>1?hi.value[1]-lo.value[1]+1:1;T* ptr=(T*)(a.mem+a.offset);a.offset+=n1*n2;return {ptr,n1,n2,{lo.value[0],R>1?lo.value[1]:1},{hi.value[0],R>1?hi.value[1]:1}};}
template<class T> T mynn_maxval(MynnArray<T,1> x){T a=x(1);for(int i=2;i<=x.n1;i++)a=a>x(i)?a:x(i);return a;}
struct MynnText{const char* p;MynnText& operator=(const char* q){p=q;return *this;}bool operator==(const char* q){return *p==*q;}};
template<class T> struct MynnOptional {bool present()const{return false;}operator T()const{return T{"b"};}};
template<> struct MynnOptional<bool>{bool value;operator bool()const{return value;}};
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_abs(float a){return __uint_as_float(__float_as_uint(a)&0x7fffffffu);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}__device__ int mynn_min(int a,int b){return a<b?a:b;}
__device__ float mynn_max(float a,float b){return a>b?a:b;}__device__ int mynn_max(int a,int b){return a>b?a:b;}
__device__ float mynn_sqrt(float a){return mynn_ieee_result(a,a,mpas_sqrt(a));}
__device__ float mynn_pow(float a,float b){return mynn_ieee_result(a,b,gfk_pow(a,b));}
__device__ float mynn_exp(float a){return mynn_ieee_result(a,a,gfk_exp(a));}
__device__ float mynn_tanh(float a){return mynn_ieee_result(a,a,p12_core_tanhf::cr_tanhf(a));}
__device__ float mynn_atan(float a){return mynn_ieee_result(a,a,glibc_atanf(a));}
template<int N> __device__ float mynn_ipow(float a){static_assert(N==2);return mpas_mul(a,a);}
'''
    profiles=re.findall(r'MynnArray<float, 1> (\w+)',sig);interfaces={'zw'}|{n for n in profiles if n.startswith('s_aw')}
    columns=('ust','flt','fltv','flq','flqv','pblh','dx','landsea','ts','psig_shcu');outcols=('maxwidth','maxmf','ztop','ktop')
    constants=('dt','p1000mb','rcp','xlvcp','t0c','rvovrd','r_v','ep_2','ep_3','cice','cliq','cpv','xls','xlv','cp','p608','gravity','ep1');flags=('momentum_opt','tke_opt','scalar_opt')
    static='\n'.join(re.search(r'static constexpr float '+k+r'\s*=.*?;',SOURCE).group() for k in ['j'+str(i) for i in range(9)]+['k'+str(i) for i in range(9)]+['tice','onethird'])
    lib=files('woof.core.kernels');core=_cuda_core_math(_vendored_maths()['core-math/tanhf.c'],'tanhf')
    members=('p1000mb','rcp','xlvcp','t0c','rvovrd','r_v','ep_2','ep_3','cice','cliq','cpv','xls','xlv','cp','p608','grav','gtr','tv0')
    src='#define __CUDA_ARCH__ 1\n#define p12_expect(a,b) (a)\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+lib.joinpath('glibc_flt32.cuh').read_text()+lib.joinpath('glibc_trig_flt32.cuh').read_text()+core+helper+'\nstruct Column {MynnArena &arena;'+''.join('float '+n+';' for n in members)+'static constexpr bool env_subs=false;'+static+'\n'+''.join(selected(n,ret)[1] for n,ret in (('f_esat_blend','float'),('f_qsat_blend','float'),('f_xl_blend','float'),('f_condensation_edmf','void')))+body+'};\n'
    decl=','.join('float* '+n for n in profiles+list(columns)+list(outcols[:3]))+',int* kpbl,int* ktop';cargs=','.join('float '+n for n in constants);fargs=','.join('int '+n for n in flags)
    raw=sig[sig.index('(')+1:sig.rindex(')')];args=[]
    for arg in re.split(r',(?![^<]*>)',raw):
        a=arg.strip();name=a.split()[-1].lstrip('&')
        if name=='ii':args.append('col+1')
        elif name=='kts':args.append('1')
        elif name=='kte':args.append('nl')
        elif name in profiles:
            z='nl+1' if name in interfaces else 'nl';args.append('MynnArray<float,1>{'+name+'+col*('+z+'),'+z+',1,{1,1},{'+z+',1}}')
        elif name in columns+outcols+('kpbl',):args.append(name+'[col]')
        elif name=='dt':args.append('dt')
        elif name in flags:args.append('bool('+name+')')
        elif name in ('nchem','spp_pbl'):args.append('0')
        elif name in ('chem1','s_awchem'):args.append('MynnArray<float,2>{nullptr,0,0,{1,1},{0,0}}')
        elif name=='mix_chem':args.append('false')
        elif name.startswith('f_q'):args.append('MynnOptional<bool>{false}')
        else:raise ValueError(a)
    init=['p1000mb','rcp','xlvcp','t0c','rvovrd','r_v','ep_2','ep_3','cice','cliq','cpv','xls','xlv','cp','p608','gravity','mpas_div(gravity,300.f)','mpas_mul(ep1,300.f)']
    src+=f'extern "C" __global__ void host_dmp(int nc,int nl,{cargs},{fargs},{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena a{{{{}},0}};Column obj{{a,'+','.join(init)+'};obj.f_dmp_mf('+','.join(args)+');}\n'
    return HostModule('mpas_v842_mynn_dmp',src).get_function('host_dmp'),profiles,interfaces,columns,outcols,constants,flags

def main():
    host,profiles,interfaces,columns,outcols,constants,flags=host_module();rng=np.random.default_rng(842)
    const=dict(dt=30.,p1000mb=100000.,rcp=287./1004.,xlvcp=2490.,t0c=273.15,rvovrd=461./287.,r_v=461.,ep_2=.622,ep_3=.378,cice=2106.,cliq=4190.,cpv=1850.,xls=2.83e6,xlv=2.5e6,cp=1004.,p608=.61,gravity=9.81,ep1=.61)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (8,17,33):
            nc=4;f={n:rng.uniform(.001,.01,(nc,nl+1 if n in interfaces else nl)).astype(np.float32) for n in profiles}
            f['zw']=np.broadcast_to(np.arange(nl+1,dtype=np.float32)*100,(nc,nl+1)).copy();f['dz']=np.full((nc,nl),100,np.float32)
            f['p']=np.broadcast_to(np.linspace(100000,70000,nl,dtype=np.float32),(nc,nl)).copy();f['rho']=np.full((nc,nl),1.1,np.float32)
            f['thv']=np.broadcast_to(np.linspace(302,298,nl,dtype=np.float32),(nc,nl)).copy()
            for n in ('th','thl','tk'):f[n]=np.broadcast_to(np.linspace(301,297,nl,dtype=np.float32),(nc,nl)).copy()
            for n in ('u','v'):f[n]=rng.uniform(2,7,(nc,nl)).astype(np.float32)
            f['w']=np.zeros((nc,nl),np.float32);f['w'][0,:]=np.nextafter(np.float32(0),np.float32(-1));f['exner']=np.full((nc,nl),.96,np.float32)
            for n in ('qt','qv'):f[n]=rng.uniform(.006,.012,(nc,nl)).astype(np.float32)
            f['qc']=rng.uniform(0,.0001,(nc,nl)).astype(np.float32);f['sgm']=np.full((nc,nl),.001,np.float32)
            f['ust']=np.full(nc,.4,np.float32);f['flt']=np.array([.03,.02,.04,.025],np.float32);f['fltv']=np.array([0,.002,.08,.04],np.float32)
            for n in ('flq','flqv'):f[n]=np.full(nc,1.e-4,np.float32)
            f['pblh']=np.full(nc,500,np.float32);f['dx']=np.full(nc,3000,np.float32);f['landsea']=np.array([1.5,2,1,2],np.float32)
            f['ts']=np.full(nc,305,np.float32);f['psig_shcu']=np.full(nc,.5,np.float32);f['kpbl']=np.full(nc,min(5,nl-1),np.int32)
            for n in outcols:f[n]=np.zeros(nc,np.int32 if n=='ktop' else np.float32)
            for mode in ('default','scalar','momentum','tke'):
                opts={'momentum_opt':mode=='momentum','tke_opt':mode=='tke','scalar_opt':mode in ('scalar','momentum','tke')}
                expected={n:v.copy() for n,v in f.items()};actual={n:xp.asarray(v.copy()) for n,v in f.items()}
                host((1,),(128,),(nc,nl,*[np.float32(const[n]) for n in constants],*[int(opts[n]) for n in flags],*[expected[n] for n in profiles+list(columns)+list(outcols[:3])],expected['kpbl'],expected['ktop']))
                dmp(actual,constants=const,options=opts)
                for n in profiles+list(columns)+list(outcols)+['kpbl']:
                    got=actual[n].to_numpy();want=expected[n];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                    if bad.size:raise AssertionError(f'levels={nl} mode={mode} {n}: {bad.size} differences at {bad[0]} actual {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
                print(f'native MYNN plume levels={nl} mode={mode} source match; maxmf={expected["maxmf"]}',flush=True)

if __name__=='__main__':main()
