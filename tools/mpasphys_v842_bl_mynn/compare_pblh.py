"""Compare the native MYNN boundary-layer height routine with selected source."""
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.resources_build import _vendored_maths,_cuda_core_math
from woof.hex.mpasphys.metal_bl_mynn_v842 import pbl_height
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def host_module():
    m=re.search(r'__device__ __noinline__ void f_get_pblh\s*\(',SOURCE);assert m
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
__device__ float mynn_min(float a,float b){return a<b?a:b;}
__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_tanh(float x){return mynn_ieee_result(x,x,p12_core_tanhf::cr_tanhf(x));}
'''
    core=_cuda_core_math(_vendored_maths()['core-math/tanhf.c'],'tanhf')
    source='#define __CUDA_ARCH__ 1\n#define p12_expect(a,b) (a)\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+core+helper+'\nstruct Column {MynnArena& arena;\n'+SOURCE[m.start():j]+'\n};\n'
    source+='''extern "C" __global__ void host_pblh(int nc,int nl,const float* th,const float* qke,const float* zw,const float* dz,const float* land,float* zi,int* kzi){int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;MynnArena arena;Column obj{arena};int o=col*nl;obj.f_get_pblh(1,nl,zi[col],{(float*)th+o},{(float*)qke+o},{(float*)zw+o},{(float*)dz+o},land[col],kzi[col]);}'''
    return HostModule('mpas_v842_mynn_pblh',source).get_function('host_pblh')


def main():
    host=host_module();rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nl in (3,4,8,33):
            nc=7;z=np.broadcast_to(np.linspace(0,900,nl,dtype=np.float32),(nc,nl)).copy()
            dz=rng.uniform(30,180,(nc,nl)).astype(np.float32);theta=(rng.uniform(-.1,.1,(nc,nl))+np.linspace(290,295,nl)).astype(np.float32)
            qke=rng.uniform(.001,.2,(nc,nl)).astype(np.float32);qke[0,0]=.01;qke[1,0]=.1
            land=np.array([1,2,1.5,1,2,1,2],np.float32);zi=np.zeros(nc,np.float32);kzi=np.zeros(nc,np.int32)
            host((1,),(128,),(nc,nl,theta,qke,z,dz,land,zi,kzi))
            a=xp.empty((nc,),np.float32);b=xp.empty((nc,),np.int32)
            inputs=[xp.asarray(v) for v in (theta,qke,z,dz,land)]
            pbl_height(*inputs,a,b)
            for name,got,want in (('zi',a.to_numpy(),zi),('kzi',b.to_numpy(),kzi)):
                bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'levels={nl} {name} differs {bad.size} first {bad[0]} {got[bad[0]]} vs {want[bad[0]]}')
            badz=np.zeros_like(z);before=a.to_numpy().copy();beforek=b.to_numpy().copy()
            try:pbl_height(inputs[0],inputs[1],xp.asarray(badz),inputs[3],inputs[4],a,b)
            except ValueError:pass
            else:raise AssertionError('source out-of-bounds scan not refused')
            if not np.array_equal(a.to_numpy().view(np.uint32),before.view(np.uint32)) or not np.array_equal(b.to_numpy(),beforek):raise AssertionError('invalid scan published outputs')
            print(f'MYNN PBL height levels {nl} matches selected source; unbounded scan refused',flush=True)

if __name__=='__main__':main()
