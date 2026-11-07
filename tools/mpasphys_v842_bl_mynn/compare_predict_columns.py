"""Compare native MMM MYNN prediction including its internal solves."""
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import predict_columns
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def host_module():
    names=('dz','el','dfq','rho','pdk','pdt','pdq','pdc','qke','tsq','qsq','cov','s_aw','s_awqke','qwt1d','qdiss1d')
    def function(name):
        m=re.search(r'__device__ __noinline__ void '+name+r'\s*\(',SOURCE);assert m
        i=SOURCE.index('{',m.end());j=i+1;depth=1
        while depth:
            if SOURCE[j]=='{':depth+=1
            elif SOURCE[j]=='}':depth-=1
            j+=1
        return SOURCE[m.start():j]
    helper='''
template<int R> struct MynnIndices {int value[R];};
struct MynnArena {float mem[8192];int offset;};
struct MynnFrame {MynnArena& a;int saved;MynnFrame(MynnArena& v):a(v),saved(v.offset){} ~MynnFrame(){a.offset=saved;}};
template<class T,int R> struct MynnArray {T* data;int count;void rebase(MynnIndices<R>){};int size(){return count;} T& linear(int i){return data[i];} T& operator()(int i){return data[i-1];}};
template<class T,int R> MynnArray<T,R> mynn_array(MynnArena& a,MynnIndices<R> lo,MynnIndices<R> hi){int n=hi.value[0]-lo.value[0]+1;T* p=a.mem+a.offset;a.offset+=n;return {p,n};}
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}
__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_sqrt(float a){return mynn_ieee_result(a,a,mpas_sqrt(a));}
template<int N> __device__ float mynn_ipow(float a){float x=mpas_mul(a,a);return N==2?x:mpas_mul(x,a);}
'''
    decl=','.join('float* '+n for n in names)
    arg=lambda name:'MynnArray<float,1>{'+name+'+col*'+('(nl+1),nl+1' if name in ('s_aw','s_awqke') else 'nl,nl')+'}'
    call=','.join(['1','nl','closure','delt',arg('dz'),'ust[col]','flt[col]','flq[col]','pmz[col]','phh[col]']+[arg(n) for n in names[1:14]]+['bool(edmf_tke)',arg('qwt1d'),arg('qdiss1d'),'false'])
    source='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+helper+'\nstruct Column {MynnArena &arena;float karman;static constexpr float b1=24.f,b2=15.f,sqfac=3.f;\n'+function('f_tridiag2')+'\n'+function('f_mym_predict')+'\n};\n'
    source+=f'extern "C" __global__ void host_predict(int nc,int nl,float closure,float delt,float karman,int edmf_tke,const float* ust,const float* flt,const float* flq,const float* pmz,const float* phh,{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena arena{{{{}},0}};Column obj{{arena,karman}};obj.f_mym_predict({call});}}\n'
    return HostModule('mpas_v842_mynn_predict',source).get_function('host_predict'),names


def main():
    host,names=host_module();rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (2,3,17,33):
            nc=7;shape=(nc,nl)
            for closure in (2.5,2.9,3.0):
                for edmf in (False,True):
                    fields={n:rng.uniform(.001,.1,(nc,nl+1) if n in ('s_aw','s_awqke') else shape).astype(np.float32) for n in names}
                    fields.update(ust=np.linspace(.1,.5,nc,dtype=np.float32),flt=np.linspace(.01,.08,nc,dtype=np.float32),flq=np.linspace(.00001,.0002,nc,dtype=np.float32),pmz=np.linspace(.5,1.5,nc,dtype=np.float32),phh=np.linspace(.7,1.7,nc,dtype=np.float32))
                    fields['dz']=rng.uniform(20,200,shape).astype(np.float32);fields['el']=rng.uniform(10,100,shape).astype(np.float32)
                    fields['rho']=rng.uniform(.5,1.3,shape).astype(np.float32);fields['dfq']=rng.uniform(.01,1,shape).astype(np.float32)
                    fields['qke'][0,0]=np.array(0x00010000,np.uint32).view(np.float32)
                    expected={n:x.copy() for n,x in fields.items()};actual={n:xp.asarray(x.copy()) for n,x in fields.items()}
                    options=dict(closure=closure,delt=60,ust=.3,flt=.02,flq=.0001,pmz=1,phh=1,karman=.4,edmf_tke=edmf,tkebudget=False)
                    host((1,),(128,),(nc,nl,*[np.float32(options[x]) for x in ('closure','delt','karman')],int(edmf),*[expected[n] for n in ('ust','flt','flq','pmz','phh')],*[expected[n] for n in names]))
                    predict_columns(actual,**{n:options[n] for n in ('closure','delt','karman','edmf_tke','tkebudget')})
                    for name in (*names,'ust','flt','flq','pmz','phh'):
                        got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                        if bad.size:raise AssertionError(f'levels={nl} closure={closure} edmf={edmf} {name}: {bad.size} differences, first {bad[0]}')
                    options['tkebudget']=True
                    try:predict_columns(actual,**{n:options[n] for n in ('closure','delt','karman','edmf_tke','tkebudget')})
                    except ValueError:pass
                    else:raise AssertionError('uninitialized native budget was not refused')
                    for name in (*names,'ust','flt','flq','pmz','phh'):
                        if not np.array_equal(actual[name].to_numpy().view(np.uint32),expected[name].view(np.uint32)):raise AssertionError('refusal mutated '+name)
                    print(f'MYNN predictor levels={nl} closure={closure} edmf={edmf} matches selected source',flush=True)

if __name__=='__main__':main()
