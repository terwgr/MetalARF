"""Compare native GF saturation-vapour arithmetic with selected source."""
from pathlib import Path
import os,re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.core import kernels
from woof.hex.mpasphys import gf_mpas_v842 as native
from woof.hex.mpasphys.metal_gf_v842 import saturation_vapour
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
support=native._SUPPORT.replace('#define __global__\n','').replace('#define __device__\n','')
support=re.sub(r'inline float (__fadd_rn|__fsub_rn|__fmul_rn|__fdiv_rn|__fsqrt_rn|gfk_log|gfk_pow)\([^\n]*\n','',support)
s=native._SCHEME_SOURCE;start=s.index('__device__ GFReal satvap (GFWorkspace &workspace, GFReal * temp2)\n{');brace=s.index('{',start);depth=0
for i in range(brace,len(s)):
    depth+=(s[i]=='{')-(s[i]=='}')
    if depth==0:end=i+1;break
header=Path(kernels.__file__).with_name('glibc_flt32.cuh').read_text()
source=header+'\n#define GF_HOST_WITNESS\n'+support+'\n'+s[start:end]+'''
extern "C" __global__ void check(const float* t,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;GFWorkspace ws={nullptr,0,0,0,0};GFReal x=t[i];out[i]=(float)satvap(ws,&x);}
'''
host=HostModule('native_gf_satvap',source).get_function('check')
t=np.array([150,200,220,253.15,253.15498,253.155,253.15502,273.15,300,373.16,400,1e-38,1e-40],np.float32)
if os.environ.get("GF_SPECIAL_WORDS"):t=np.concatenate((t,np.array([0.0,-0.0,np.inf,-np.inf,np.nan],np.float32)))
n=len(t);ref=np.zeros_like(t)
host(((n+127)//128,),(128,),(t,ref,np.int32(n)))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kw):
        result=submit(*args,**kw);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context);out=xp.empty((n,),np.float32)
    saturation_vapour(xp.asarray(t),out);got=out.to_numpy();ix=np.flatnonzero(got.view(np.uint32)!=ref.view(np.uint32))
    print('satvap differences',len(ix),'/',n,[(int(i),hex(int(got.view(np.uint32)[i])),hex(int(ref.view(np.uint32)[i]))) for i in ix],flush=True)
    if len(ix):raise SystemExit(1)
