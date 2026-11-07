"""Compare native MYNN moisture correction against selected host source."""
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import moisture_check
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    m=re.search(r'__device__ __noinline__ void f_moisture_check\s*\(.*?\)\s*\{',SOURCE,re.S);j=m.end();depth=1
    while depth:
        if SOURCE[j]=='{':depth+=1
        elif SOURCE[j]=='}':depth-=1
        j+=1
    body=SOURCE[m.start():j]
    helper='''
template<int R> struct MynnIndices{int value[R];};struct MynnArena{};struct MynnFrame{MynnFrame(MynnArena&) {}};
template<class T,int R> struct MynnArray{T* data;void rebase(MynnIndices<R>){};T& operator()(int i){return data[i-1];}};
__device__ float mynn_max(float a,float b){return a>b?a:b;}
'''
    names=('dp','exner','qv','qc','qi','qs','th','dqv','dqc','dqi','dqs','dth')
    decl=','.join('float* '+n for n in names);args=','.join('MynnArray<float,1>{'+n+'+col*nl}' for n in names)
    src=CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+helper+'struct Column{MynnArena &arena;float xlvcp,xlscp;'+body+'};\n'
    src+=f'extern "C" __global__ void host_moisture(int nc,int nl,{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena a;Column obj{{a,2490.f,2819.f}};obj.f_moisture_check(nl,30.f,'+args+');}\n'
    host=HostModule('mpas_v842_mynn_moisture',src).get_function('host_moisture');rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (1,2,8,33):
            nc=7;shape=(nc,nl);fields={n:rng.uniform(-.002,.003,shape).astype(np.float32) for n in names}
            fields['dp']=rng.uniform(50,500,shape).astype(np.float32);fields['exner']=rng.uniform(.5,1,shape).astype(np.float32);fields['th']=rng.uniform(240,310,shape).astype(np.float32)
            fields['qv'][0,:]=np.float32(1.e-20);fields['qv'][1,:]=np.float32(-1.e-22);fields['qv'][2,:]=np.nextafter(np.float32(0),np.float32(1));fields['qc'][3,:]=np.float32(-1.e-5);fields['qi'][4,:]=np.float32(-2.e-5);fields['qs'][5,:]=np.float32(-3.e-5)
            expected={n:v.copy() for n,v in fields.items()};actual={n:xp.asarray(v.copy()) for n,v in fields.items()}
            host((1,),(128,),(nc,nl,*[expected[n] for n in names]));moisture_check(actual,delt=30,xlvcp=2490,xlscp=2819)
            for name in names:
                got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'levels={nl} {name}: {bad.size} differences at {bad[0]} actual {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
            print(f'native MYNN moisture correction levels={nl} matches selected source',flush=True)

if __name__=='__main__':main()
