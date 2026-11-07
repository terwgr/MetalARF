"""Compare native MYNN cold/warm driver preparation with selected source."""
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import driver_pre
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

def main():
    m=re.search(r'__device__ __noinline__ void f_bl_mynn_run\s*\(.*?\)\s*\{',SOURCE,re.S);assert m;s=SOURCE[m.end():]
    def region(a,b):
        i=s.index('// bl_mynn.F90:'+str(a));return s[i:s.index('// bl_mynn.F90:'+str(b),i)]
    body=region(634,664)+region(681,696)+'if(cold){\n'+region(710,722)+region(730,733)+'\n}'
    inputs=('dz1','th1','sqv1','sqc1','sqi1','sqs1','ex1')
    outputs=('qv1','qc1','qi1','qs1','sqw','thl','thetav','qke1','sh1','sm1','el_pbl1','tsq1','qsq1','cov1','cldfra_bl1','qc_bl1','qi_bl1')
    names=inputs+outputs+('ust1','zw');decl=','.join('float* '+n for n in names)
    macros=''.join(f'#define {n}(i) {n}_ptr[col*nl+(i)-1]\n' for n in inputs+outputs)
    decl=','.join('float* '+n+'_ptr' for n in names)
    macros+='#define zw(i) zw_ptr[col*(nl+1)+(i)-1]\n'
    src='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+'\n__device__ float mynn_max(float a,float b){return a>b?a:b;}\n'+macros
    src+=f'extern "C" __global__ void host_pre(int nc,int nl,int cold,{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;int k,kts=1,kte=nl;float ust1=ust1_ptr[col],xlvcp=2490.f,xlscp=2819.f;\n'+body+'\n}\n'
    host=HostModule('mpas_v842_mynn_driver_pre',src).get_function('host_pre');rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (2,3,8,33):
            nc=7;fields={n:rng.uniform(.001,.2,(nc,nl)).astype(np.float32) for n in inputs+outputs}
            fields['dz1']=rng.uniform(20,200,(nc,nl)).astype(np.float32);fields['th1']=rng.uniform(240,310,(nc,nl)).astype(np.float32);fields['ex1']=rng.uniform(.6,1,(nc,nl)).astype(np.float32)
            fields['ust1']=np.linspace(.1,.7,nc,dtype=np.float32);fields['zw']=rng.uniform(1,5,(nc,nl+1)).astype(np.float32)
            for cold in (False,True):
                expected={n:v.copy() for n,v in fields.items()};actual={n:xp.asarray(v.copy()) for n,v in fields.items()}
                host((1,),(128,),(nc,nl,int(cold),*[expected[n] for n in names]));driver_pre(actual,xlvcp=2490,xlscp=2819,cold=cold)
                for n in names:
                    got=actual[n].to_numpy();want=expected[n];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                    if bad.size:raise AssertionError(f'levels={nl} cold={cold} {n}: {bad.size} differences at {bad[0]}')
                print(f'MYNN driver preparation levels={nl} cold={cold} matches selected source',flush=True)
if __name__=='__main__':main()
