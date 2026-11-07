"""Compare active v842 cloud-fraction slot with host-compiled selected source."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.radshared_v842 import CUDA_SOURCE
from woof.hex.mpasphys.metal_cloud_v842 import cloudiness_slot, cloud_restart_state, restore_cloud_state
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    start=CUDA_SOURCE.index('extern "C" __global__ void cloud_from(')
    end=CUDA_SOURCE.index('// MPAS module_mp_thompson_cldfra3.F',start)
    header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()
    host=HostModule('mpas_v842_cloud_fraction',header+'\n'+CUDA_SOURCE[start:end])
    cloud_from=host.get_function('cloud_from'); simple=host.get_function('cloud_simple')
    rng=np.random.default_rng(8424); nz,nc=5,7
    shape=(nz,nc)
    profiles={'qv':rng.uniform(.001,.025,shape).astype(np.float32),
              'qc':rng.uniform(0,.001,shape).astype(np.float32),
              'qi':rng.uniform(0,.0003,shape).astype(np.float32),
              'qs':rng.uniform(0,.0004,shape).astype(np.float32),
              't':rng.uniform(245,300,shape).astype(np.float32),
              'pres':rng.uniform(50000,99000,shape).astype(np.float32)}
    profiles['qc'][0,0]=profiles['qi'][0,0]=profiles['qs'][0,0]=0
    profiles['qv'][1,1]=np.float32(.5)
    density=rng.uniform(.5,3,(nc,)).astype(np.float32)
    land=rng.uniform(1,2,(nc,)).astype(np.float32)
    local={n:np.ascontiguousarray(profiles[n].T) for n in profiles}
    carriers=[np.empty((nc,nz),np.float32) for _ in range(5)]
    dx=np.empty(nc,np.float32); xland=np.empty(nc,np.float32)
    cloud_from((1,),(128,),(nc,nz,np.float32(12000),density,land,
        *(local[n] for n in ('qv','qc','qi','qs')),dx,xland,*carriers))
    simple((1,),(128,),(nc*nz,1,1,1,*carriers[:4],local['t'],local['pres'],carriers[4]))
    expected={n:np.ascontiguousarray(v.T) for n,v in zip(('qvrad','qcrad','qirad','qsrad','cldfrac'),carriers)}
    expected.update(dx=dx,xland=xland)
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        d={n:xp.asarray(v) for n,v in profiles.items()}
        diag=xp.asarray(np.full(shape,-1,np.float32))
        rad={'ozone_state':'preserve'}
        published={n:xp.asarray(np.full(v.shape,-2,np.float32)) for n,v in expected.items()}
        pools={'prepared':d,'mesh':{'meshDensity':xp.asarray(density)},
               'sfc_input':{'xland':xp.asarray(land)},'config_len_disp':12000,
               'diag_physics':{'cldfrac':diag},'radshared':rad,'radiation_prepared':published}
        out=cloudiness_slot(SimpleNamespace(scheme='cld_fraction',pools=pools))
        for name,ref in expected.items():
            got=out[name].to_numpy()
            unequal=np.count_nonzero(got.view(np.uint32)!=ref.view(np.uint32))
            if unequal: raise AssertionError(f'{name}: {unequal} differing words; max abs {np.max(np.abs(got-ref))}')
        if not np.array_equal(diag.to_numpy().view(np.uint32),expected['cldfrac'].view(np.uint32)):
            raise AssertionError('diagnostic differs')
        if rad['ozone_state']!='preserve': raise AssertionError('ozone state changed')
        for name,ref in expected.items():
            if not np.array_equal(published[name].to_numpy().view(np.uint32),ref.view(np.uint32)):
                raise AssertionError(f'prepared radiation differs: {name}')
        checkpoint=cloud_restart_state(rad)
        for name in out: out[name].fill(-7)
        restore_cloud_state(rad,checkpoint,context)
        for name,ref in expected.items():
            if not np.array_equal(rad[name].to_numpy().view(np.uint32),ref.view(np.uint32)):
                raise AssertionError(f'cloud carrier restore differs: {name}')
        if rad['ozone_state']!='preserve': raise AssertionError('ozone state changed at restore')
        c=expected['cldfrac']; print(f'active cloud fraction matches selected source: min={c.min():.5g}, max={c.max():.5g}, nonzero={np.count_nonzero(c)}')

if __name__=='__main__': main()
