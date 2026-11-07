"""Compare native LW optical-depth and diffusivity stages with selected source."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import optical_depth,diffusivity
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=(ROOT/'woof/core/kernels/rrtmg_lw_chain.cu').read_text()
def extract(name):
    m=re.search(r'extern "C" __global__ void '+name+r'\s*\(',source)
    i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return source[m.start():j]
header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()+'''
#define NGPTLW 140
#define NBNDLW 16
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define rlw_exp(x) gfk_exp(x)
'''
host=HostModule('mpas_v842_lw_transfer',header+'\n'.join(extract(n) for n in ('rlw_taut','rlw_rtrn_secdiff')))
nc,nl=4,9
rng=np.random.default_rng(842)
taug=rng.uniform(0,2,(nc,nl,140)).astype(np.float32);taua=rng.uniform(0,.3,(nc,nl,16)).astype(np.float32)
taug[0,0,:4]=np.arange(1,5,dtype=np.uint32).view(np.float32);taua[0,0,:4]=np.arange(1,5,dtype=np.uint32).view(np.float32)
ngb=(np.arange(140)%16+1).astype(np.int32);pwv=np.array([0,.05,1,20],np.float32)
a0=np.linspace(1.4,1.9,16,dtype=np.float32);a1=np.linspace(-.2,.3,16,dtype=np.float32);a2=np.linspace(-.5,.1,16,dtype=np.float32)
taut=np.empty_like(taug);secdiff=np.empty((nc,16),np.float32)
host.get_function('rlw_taut')(((nc*nl*140+127)//128,),(128,),(np.int32(nc),np.int32(nl),taug,taua,ngb,taut))
host.get_function('rlw_rtrn_secdiff')((1,),(128,),(np.int32(nc),pwv,a0,a1,a2,secdiff))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    dt=xp.empty(taut.shape,np.float32);ds=xp.empty(secdiff.shape,np.float32)
    optical_depth(*map(xp.asarray,(taug,taua,ngb)),dt);diffusivity(*map(xp.asarray,(pwv,a0,a1,a2)),ds)
    for name,got,expected in (('taut',dt,taut),('secdiff',ds,secdiff)):
        diff=np.count_nonzero(got.to_numpy().view(np.uint32)!=expected.view(np.uint32))
        print(name,'words',expected.size,'differences',diff)
        if diff:raise SystemExit(1)
