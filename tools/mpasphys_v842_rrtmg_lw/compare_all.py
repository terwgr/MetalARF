"""Compare the complete native RRTMG-LW initializer with selected source."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys import rrtmg_lw_v842 as native
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import initialize_tables
from woof.core import rrtmg_lw as legacy
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
static=[]
for name in ('ngc','ngn','ngm','wt'):
    static.append(re.search(r'extern "C" __device__ const (?:float|int) mpas_static_'+name+r'\[\d+\] = \{.*?\};',native._INIT_SOURCE,re.S).group())
source='\n'.join(static)+'\n'+native._INIT_SOURCE[native._INIT_SOURCE.index('extern "C" __global__ void mpas_init_rwgt'):]
header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()+'''
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
#define rlw_exp(x) gfk_exp(x)
'''
host=HostModule('mpas_v842_lw_all_init',header+source)
weights=np.empty(256,np.float32);host.get_function('mpas_init_rwgt')((1,),(1,),(weights,))
rng=np.random.default_rng(842);raw={}
for band,dest,name,points,first,weighted in native._REDUCTIONS:
    module=raw.setdefault(f'rrlw_kg{band:02d}',{})
    if name not in module:module[name]=rng.uniform(.001,1,(points*16,)).astype(np.float32)
    assert module[name].shape==(points*16,)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*a,**k):
        result=submit(*a,**k);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    device={key:{name:xp.asarray(value) for name,value in row.items()} for key,row in raw.items()}
    result=initialize_tables(device,context)
    bad=0;words=0
    for band,dest,name,points,first,weighted in native._REDUCTIONS:
        size=points*int(legacy.NGC[band-1]);expected=np.empty(size,np.float32)
        host.get_function('mpas_cmbgb')(((size+127)//128,),(128,),
            (np.int32(band-1),np.int32(points),np.int32(first),np.int32(weighted),raw[f'rrlw_kg{band:02d}'][name],weights,expected))
        actual=result['reduced'][dest].to_numpy();indices=np.flatnonzero(actual.view(np.uint32)!=expected.view(np.uint32))
        words+=size;bad+=len(indices)
        if len(indices):print(dest,'mismatches',len(indices),'first',indices[:5].tolist())
    assert np.array_equal(result['rwgt'].to_numpy().view(np.uint32),weights.view(np.uint32))
    for band,names in legacy.GPU_BAND_TABS.items():
        assert all(v is result['reduced'][f'kg{band:02d}/{name}'] for name,v in zip(names,result['band'][band]))
    print('native initializer reduced arrays',len(native._REDUCTIONS),'words',words,'differences',bad)
    if bad:raise SystemExit(1)
