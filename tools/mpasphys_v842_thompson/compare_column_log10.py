"""Compare the pinned native Thompson log10f helper to selected source."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=(ROOT/'woof/hex/data/mpasphys-maths/ubm_log10f.cuh').read_text()+'''
extern "C" __global__ void check(const float* input,float* output,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)output[i]=ubm_log10f(input[i]);}
'''
host=HostModule('native_thompson_log10',source).get_function('check')
rng=np.random.default_rng(842);words=rng.integers(0,2**32,size=10000,dtype=np.uint32)
boundaries=np.array([0,0x80000000,1,0x007fffff,0x00800000,0x7f800000,0xff800000,0x7fc00001,0x7f800001,0x3f800000,0x41200000],np.uint32)
for j in range(65):
    center=0x3f800000+j*(1<<17)
    boundaries=np.concatenate((boundaries,np.array([center-1,center,center+1],np.uint32)))
words=np.concatenate((words,boundaries));values=words.view(np.float32);n=len(words);expected=np.empty(n,np.float32)
host(((n+127)//128,),(128,),(values,expected,np.int32(n)))
with MetalContext(batch_kernel_calls=False) as ctx:
    submit=ctx._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);out=xp.empty((n,),np.float32)
    _launch('hexv842_mpas_thompson_column_log10_f32',n,ctx,dict(input=xp.asarray(values),output=out),dict(n=np.int32(n)))
    got=out.to_numpy().view(np.uint32);ref=expected.view(np.uint32);bad=np.flatnonzero(got!=ref)
    print('log10 words',n,'differences',len(bad),[(hex(int(words[i])),hex(int(got[i])),hex(int(ref[i]))) for i in bad[:10]],flush=True)
    if len(bad):raise SystemExit(1)
