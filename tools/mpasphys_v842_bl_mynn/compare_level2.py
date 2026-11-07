"""Compare the native MYNN level-2 operator with its selected source."""
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import level2
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def host_module():
    names=('dz','u','v','thl','thetav','qw','ql','vt','vq','dtl','dqw','dtv','gm','gh','sm','sh')
    constants=('b1','b2','pr','g1','c2','c3','c5','a1','c1','a2','g2','ckmod')
    declarations=[]
    for key in constants:
        m=re.search(r'static constexpr float '+key+r'\s*=.*?;',SOURCE);assert m
        declarations.append(m.group())
    m=re.search(r'__device__ __noinline__ void f_mym_level2\s*\(',SOURCE);assert m
    i=SOURCE.index('{',m.end());j=i+1;depth=1
    while depth:
        if SOURCE[j]=='{':depth+=1
        elif SOURCE[j]=='}':depth-=1
        j+=1
    helper='''
template<int R> struct MynnIndices {int value[R];};
struct MynnArena {};
struct MynnFrame {MynnFrame(MynnArena&){} };
template<class T,int R> struct MynnArray {T* data;void rebase(MynnIndices<R>){};T& operator()(int i){return data[i-1];}};
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}
__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_sqrt(float a){return mynn_ieee_result(a,a,mpas_sqrt(a));}
template<int N> __device__ float mynn_ipow(float a){static_assert(N==2);return mpas_mul(a,a);}
'''
    decl=','.join('float* '+n for n in names);args=','.join('MynnArray<float,1>{'+n+'+col*nl}' for n in names)
    source='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+helper+'\nstruct Column {MynnArena &arena;float gtr,tv0;\n'+'\n'.join(declarations)+'\n'+SOURCE[m.start():j]+'\n};\n'
    source+=f'extern "C" __global__ void host_level2(int nc,int nl,float gravity,float ep1,{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena arena;Column obj{{arena,mpas_div(gravity,300.f),mpas_mul(ep1,300.f)}};obj.f_mym_level2(1,nl,{args});}}\n'
    return HostModule('mpas_v842_mynn_level2',source).get_function('host_level2'),names


def main():
    host,names=host_module();rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (1,2,3,17,33):
            nc=7;shape=(nc,nl)
            fields={name:rng.uniform(-.1,.1,shape).astype(np.float32) for name in names}
            fields['dz']=rng.uniform(20,200,shape).astype(np.float32)
            fields['thl']+=300;fields['thetav']+=300
            fields['u']*=100;fields['v']*=100
            if nl>1:
                fields['vt'][0,1]=np.array(0x00010000,np.uint32).view(np.float32)
                fields['u'][1,1]=np.array(0x7fc12345,np.uint32).view(np.float32)
            expected={key:value.copy() for key,value in fields.items()};actual={key:xp.asarray(value.copy()) for key,value in fields.items()}
            host((1,),(128,),(nc,nl,np.float32(9.81),np.float32(.608),*[expected[n] for n in names]))
            level2(actual,gravity=9.81,ep1=.608)
            for name in names:
                got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'levels {nl}, {name}: {bad.size} diffs at {bad[0]}, actual {got.flat[bad[0]]}, expected {want.flat[bad[0]]}')
            print(f'MYNN native level2 levels {nl} matches selected source',flush=True)

if __name__=='__main__':main()
