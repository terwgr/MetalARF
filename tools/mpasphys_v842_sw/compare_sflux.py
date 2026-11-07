"""Compare native solar-flux kernel against the selected source body."""
from pathlib import Path
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_tables
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    root=Path(__file__).resolve().parents[2]
    header=(root/'woof/metal/shaders/rrtmg_legacy_sw_tables.metalh').read_text().replace('as_type<float>(', '__uint_as_float(')
    a=CUDA_SOURCE.index('struct RswSpec {');b=CUDA_SOURCE.index('// Fortran left-to-right 8-term interpolation',a)
    c=CUDA_SOURCE.index('__device__ void rsw_sfluxzen_body(');d=CUDA_SOURCE.index('// cldprmc_sw: one thread',c)
    pre='''typedef float real;
#define RSW_NGPT 112
#define RSW_NBND 14
#define AD(a,b) __fadd_rn(a,b)
#define SU(a,b) __fsub_rn(a,b)
#define MU(a,b) __fmul_rn(a,b)
#define DV(a,b) __fdiv_rn(a,b)
'''
    src=HostModule('mpas_v842_sw_sflux',pre+header+CUDA_SOURCE[a:b]+CUDA_SOURCE[c:d]).get_function('rsw_sfluxzen_b')
    fixture=np.load(root/'tools/rrtmg_wrf461_oracle/sw_fixtures/fixtures_shallow.npz');nc,nl=2,21
    inputs={}
    for name in ('jp','jt','jt1','colh2o','colco2','colch4','colo2','colo3','fac00','fac01','fac10','fac11'):
        inputs[name]=np.stack([fixture['c00301/setcoef/'+name],fixture['c00302/setcoef/'+name]])
    inputs['laytrop']=np.array([nl,10],np.int32);inputs['laysolfr']=np.stack((np.full(14,nl,np.int32),np.full(14,12,np.int32)))
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);tables=initialize_tables(ctx)
        hosttab=tables.packed.to_numpy();hostngb=tables.ngb.to_numpy();expected=np.empty((nc,112),np.float32)
        order=('jp','jt','jt1','colh2o','colco2','colch4','colo2','colo3','fac00','fac01','fac10','fac11')
        src((2,),(128,),(nc,nl,inputs['laytrop'],hosttab,hostngb,inputs['laysolfr'],*[inputs[n] for n in order],expected))
        out=xp.empty((nc,112),np.float32);arrays={n:xp.asarray(v) for n,v in inputs.items()}
        arrays.update(tab=tables.packed,ngb=tables.ngb,sfluxzen=out)
        ctx.submit_kernel('hexv842_sw_sfluxzen_f32',arrays,dict(ncol=np.int32(nc),nl=np.int32(nl),ntab=np.int32(tables.packed.size)),exact_policy=False,grid_size=nc*112,threads_per_group=128)
        got=out.to_numpy();bad=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32))
        if bad:raise AssertionError(f'native solar flux {bad} differing words')
    print('native SW solar flux matches selected source, both empty and present upper loops',flush=True)

if __name__=='__main__':main()
