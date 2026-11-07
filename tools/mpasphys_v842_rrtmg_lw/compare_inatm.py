"""Compare native RRTMG-LW INATM against the selected source kernel."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import inatm
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=(ROOT/'woof/core/kernels/rrtmg_lw_chain.cu').read_text()
part=source.split('extern "C" __global__ void rlw_inatm(',1)[1]
source='extern "C" __global__ void rlw_inatm('+part[:part.index('\n}')+2]
header='''#define MXMOL 38
#define MAXXSEC 4
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
'''
host=HostModule('mpas_v842_lw_inatm',header+source).get_function('rlw_inatm')
rng=np.random.default_rng(842)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*a,**k):
        result=submit(*a,**k);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    for nl in (1,8,49):
        n=4
        p=np.tile(np.linspace(1000,20,nl+1,dtype=np.float32),(n,1))
        w=rng.uniform(.000001,.0008,(n,38,nl)).astype(np.float32);w[:,0,:]=.005
        w[:,7:,:]=rng.uniform(.01,.1,(n,31,nl)).astype(np.float32)
        x=rng.uniform(1e-11,1e-8,(n,4,nl)).astype(np.float32)
        x.view(np.uint32).flat[::31]=np.uint32(1)
        expected=[w.copy(),x.copy(),np.empty((n,nl),np.float32),np.empty((n,nl),np.float32),np.empty(n,np.float32)]
        grav=np.float32(9.8066);avogad=np.float32(6.02214208470173e23)
        host((1,),(128,),(np.int32(n),np.int32(nl),p,grav,avogad,*expected))
        dp=xp.asarray(p);dw=xp.asarray(w);dx=xp.asarray(x);cold=xp.empty((n,nl),np.float32);broad=xp.empty((n,nl),np.float32);pw=xp.empty((n,),np.float32)
        inatm(dp,dw,dx,cold,broad,pw,grav=grav,avogad=avogad)
        bad=0;words=0
        for name,actual,reference in zip(('wkl','wx','coldry','wbrodl','pwvcm'),(dw,dx,cold,broad,pw),expected):
            got=actual.to_numpy();indices=np.flatnonzero(got.view(np.uint32)!=reference.view(np.uint32));bad+=len(indices);words+=got.size
            if len(indices):print(name,'mismatches',len(indices),'first',indices[:8].tolist())
        print('levels',nl,'words',words,'differences',bad)
        if bad:raise SystemExit(1)
