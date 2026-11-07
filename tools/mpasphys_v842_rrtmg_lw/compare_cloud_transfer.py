"""Compare native LW cloud-transfer preparation with selected source."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.core.rrtmg_lw import _lw_coalesced_source
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import cloud_transfer
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=_lw_coalesced_source((ROOT/'woof/core/kernels/rrtmg_lw_chain.cu').read_text())
m=re.search(r'extern "C" __global__ void rlw_rtrn_prol_coalesced\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()+'''
#define NGPTLW 140
#define NBNDLW 16
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define rlw_exp(x) gfk_exp(x)
'''
host=HostModule('mpas_v842_lw_cloud_transfer',header+source[m.start():j]).get_function('rlw_rtrn_prol_coalesced')
nc,nl=3,9;rng=np.random.default_rng(842)
cloud=(rng.random((nc,nl,140))>.55).astype(np.float32);cloud[0,0,:]=0
cloud[0,1,:]=1;tau=rng.uniform(0,5,cloud.shape).astype(np.float32)
tau[0,1,0]=np.array([1],np.uint32).view(np.float32)[0]
secdiff=rng.uniform(1.5,1.8,(nc,16)).astype(np.float32);ngb=(np.arange(140)%16+1).astype(np.int32)
od=np.empty_like(cloud);ef=np.empty_like(cloud);flags=np.zeros((nc,nl),np.int32)
host(((nc*nl+127)//128,),(128,),(np.int32(nc),np.int32(nl),cloud,tau,secdiff,ngb,od,ef,flags))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    do=xp.empty(od.shape,np.float32);de=xp.empty(ef.shape,np.float32);df=xp.zeros(flags.shape,np.int32)
    cloud_transfer(*map(xp.asarray,(cloud,tau,secdiff,ngb)),do,de,df)
    for name,got,expected in (('odcld',do,od),('efclfrac',de,ef),('icldlyr',df,flags)):
        diff=np.count_nonzero(got.to_numpy().view(np.uint32)!=expected.view(np.uint32))
        print(name,'words',expected.size,'differences',diff)
        if diff:raise SystemExit(1)
