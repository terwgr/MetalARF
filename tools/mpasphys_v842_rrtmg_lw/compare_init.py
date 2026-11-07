"""Compare native RRTMG-LW table initialization against selected source."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_lw_v842 import _INIT_SOURCE
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
static=[]
for name in ('ngc','ngn','ngm','wt'):
    static.append(re.search(r'extern "C" __device__ const (?:float|int) mpas_static_'+name+r'\[\d+\] = \{.*?\};',_INIT_SOURCE,re.S).group())
source='\n'.join(static)+'\n'+_INIT_SOURCE[_INIT_SOURCE.index('extern "C" __global__ void mpas_init_rwgt'):]
header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()+'''
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
#define rlw_exp(x) gfk_exp(x)
'''
host=HostModule('mpas_v842_lw_init',header+source)
nweights=np.empty(256,np.float32)
host.get_function('mpas_init_rwgt')((1,),(1,),(nweights,))
ntau=np.empty(10001,np.float32);nexp=np.empty_like(ntau);ntfn=np.empty_like(ntau)
host.get_function('mpas_init_tables')((79,),(128,),(ntau,nexp,ntfn))
rng=np.random.default_rng(842)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*a,**k):
        result=submit(*a,**k);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    weights=xp.empty((256,),np.float32)
    _launch('hexv842_mpas_rrtmg_lw_rwgt_f32',1,context,dict(rwgt=weights),dict(n=np.int32(1)))
    tau=xp.empty((10001,),np.float32);exp=xp.empty((10001,),np.float32);tfn=xp.empty((10001,),np.float32)
    _launch('hexv842_mpas_rrtmg_lw_tables_f32',10001,context,dict(tau=tau,exp_table=exp,tfn=tfn),dict(n=np.int32(10001)))
    count=0
    for name,actual,expected in [('rwgt',weights,nweights),('tau',tau,ntau),('exp',exp,nexp),('tfn',tfn,ntfn)]:
        got=actual.to_numpy();bad=np.flatnonzero(got.view(np.uint32)!=expected.view(np.uint32))
        print(name,'mismatches',len(bad),'/',len(expected),'indices',bad[:12].tolist());count+=len(bad)
    ngc=(10,12,16,14,16,8,12,8,12,6,8,8,4,2,2,2)
    words=0
    for band,groups in enumerate(ngc):
        for first in (0,1):
            for weighted in (0,1):
                points=65;raw=rng.uniform(-1,1,(points*16,)).astype(np.float32)
                # Exercise retained binary32 subnormal inputs in reductions.
                raw.view(np.uint32)[::113]=np.uint32(1)
                expected=np.empty(points*groups,np.float32)
                host.get_function('mpas_cmbgb')(((len(expected)+127)//128,),(128,),
                    (np.int32(band),np.int32(points),np.int32(first),np.int32(weighted),raw,nweights,expected))
                output=xp.empty(expected.shape,np.float32)
                _launch('hexv842_mpas_rrtmg_lw_cmbgb_f32',expected.size,context,
                    dict(raw=xp.asarray(raw),rwgt=weights,out=output),
                    dict(band=np.int32(band),points=np.int32(points),first=np.int32(first),weighted=np.int32(weighted)))
                got=output.to_numpy();bad=np.flatnonzero(got.view(np.uint32)!=expected.view(np.uint32));count+=len(bad);words+=len(expected)
                if len(bad):print('band',band+1,'first',first,'weighted',weighted,'mismatch',len(bad),'indices',bad[:5].tolist())
    print('reduction words',words,'total differences',count)
    if count:raise SystemExit(1)
