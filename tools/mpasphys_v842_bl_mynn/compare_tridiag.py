"""Compare native MYNN tridiagonal bodies on host and Metal."""
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA
from woof.hex.mpasphys.metal_bl_mynn_v842 import tridiag
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def host_module():
    helper='''
template<int R> struct MynnIndices {int value[R];};
struct MynnArena {float mem[2048];int offset;};
struct MynnFrame { MynnFrame(MynnArena&){} };
template<class T,int R> struct MynnArray {T* data;void rebase(MynnIndices<R>){};T& operator()(int i){return data[i-1];}};
template<class T,int R> MynnArray<T,R> mynn_array(MynnArena& arena,MynnIndices<R> lo,MynnIndices<R> hi) {int count=hi.value[0]-lo.value[0]+1;T* p=arena.mem+arena.offset;arena.offset+=count;return {p};}
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
'''
    bodies=[]
    for name in ('f_tridiag','f_tridiag2','f_tridiag3'):
        m=re.search(r'__device__ __noinline__ void '+name+r'\s*\(',MYNN_MODULE_CUDA);assert m
        i=MYNN_MODULE_CUDA.index('{',m.end());j=i+1;depth=1
        while depth:
            if MYNN_MODULE_CUDA[j]=='{':depth+=1
            elif MYNN_MODULE_CUDA[j]=='}':depth-=1
            j+=1
        bodies.append(MYNN_MODULE_CUDA[m.start():j])
    source='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+helper+'\nstruct Column {MynnArena &arena;\n'+'\n'.join(bodies)+'\n};\n'
    for variant in (1,2,3):
        fn='f_tridiag'+('' if variant==1 else str(variant));names=['a','b','c','d']+(['x'] if variant!=1 else [])
        decl=','.join('float* '+x for x in names);args=','.join('MynnArray<float,1>{'+x+'+col*nl}' for x in names)
        source+=f'extern "C" __global__ void host_tri{variant}(int nc,int nl,{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena arena{{{{}},0}};Column obj{{arena}};obj.{fn}(nl,{args});}}\n'
    return HostModule('mpas_v842_mynn_tridiag',source)


def main():
    host=host_module();rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for n in (1,2,3,17,33):
            nc=7;shape=(nc,n)
            for variant in (1,2,3):
                values={name:rng.uniform(-.2,.2,shape).astype(np.float32) for name in ('a','c','d','x')}
                values['b']=rng.uniform(1,4,shape).astype(np.float32)
                values['d'][0,0]=np.array(0x00010000,np.uint32).view(np.float32)
                values['c'][1,0]=np.array(0x80000001,np.uint32).view(np.float32)
                expected={key:value.copy() for key,value in values.items()}
                actual={key:xp.asarray(value.copy()) for key,value in values.items()}
                names=('a','b','c','d')+(('x',) if variant!=1 else ())
                host.get_function('host_tri'+str(variant))((1,),(128,),(nc,n,*[expected[k] for k in names]))
                tridiag(variant,*[actual[k] for k in names])
                for name in names:
                    got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                    if bad.size:raise AssertionError(f'variant {variant}, levels {n}, {name}: {bad.size} diffs at {bad[0]}')
                print(f'MYNN tridiag {variant} levels {n} matches selected source',flush=True)

if __name__=='__main__':main()
