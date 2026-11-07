"""Compare MPAS radiation and precipitation bucket updates to selected source."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.surface_update_v842 import _CUDA,RADIATION_FIELDS
from woof.hex.mpasphys.metal_surface_v842 import update_radiation_diagnostics,update_precipitation_buckets
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    helpers=_CUDA[_CUDA.index('__device__ float add('):_CUDA.index('// MPAS source: mpas_atmphys_update_surface.F:89-108.\nextern')]
    body=_CUDA[_CUDA.index('extern "C" __global__ void accumulate('):]
    kernel=HostModule('mpas_v842_buckets',helpers+body).get_function('accumulate')
    rng=np.random.default_rng(420); nc=9; solve=7; its=2; ite=6
    fields={}
    for n in RADIATION_FIELDS:
        fields[n]=rng.uniform(0,.5,nc).astype(np.float32)
        fields['ac'+n]=rng.uniform(0,1.2,nc).astype(np.float32)
        fields['i_ac'+n]=rng.integers(0,4,nc,dtype=np.int32)
    for suffix,rate in (('c','cuprec'),('nc','rainncv')):
        fields[rate]=rng.uniform(0,.5,nc).astype(np.float32)
        fields['rain'+suffix]=rng.uniform(0,1.2,nc).astype(np.float32)
        fields['i_rain'+suffix]=rng.integers(0,4,nc,dtype=np.int32)
    ref={n:v.copy() for n,v in fields.items()}
    cfg={'config_bucket_radt':1.,'config_bucket_rainc':1.,'config_bucket_rainnc':1.}
    with MetalContext() as context:
        xp=MetalArrayModule(context); diag={n:xp.asarray(v) for n,v in fields.items()}
        for ringing in (False,True):
            for n in RADIATION_FIELDS:
                kernel((1,),(128,),(nc,its-1,ite,np.float32(2),np.float32(1),int(ringing),ref[n],ref['ac'+n],ref['i_ac'+n]))
            for suffix,rate,dt in (('c','cuprec',2.),('nc','rainncv',1.)):
                kernel((1,),(128,),(nc,its-1,ite,np.float32(dt),np.float32(1),int(ringing),ref[rate],ref['rain'+suffix],ref['i_rain'+suffix]))
            update_radiation_diagnostics(configs=cfg,diag=diag,its=its,ite=ite,n_cells_solve=solve,dt_dyn=2,ringing=ringing)
            update_precipitation_buckets(configs=cfg,diag=diag,its=its,ite=ite,n_cells_solve=solve,dt_dyn=2,ringing=ringing)
            for n,value in ref.items():
                actual=diag[n].to_numpy()
                if not np.array_equal(actual.view(np.uint32),value.view(np.uint32)):
                    raise AssertionError(f'{n} differs from selected source')
    print('radiation and precipitation bucket updates match host-selected source words')

if __name__=='__main__': main()
