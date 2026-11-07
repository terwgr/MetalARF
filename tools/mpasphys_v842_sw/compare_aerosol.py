"""Compare the live native Thompson SW aerosol path to selected source."""
from importlib.resources import files
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import thompson_aerosols
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    root=files('woof.core.kernels');a=CUDA_SOURCE.index('__device__ const float MPAS_RAOD')
    b=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw.F:10026-10664',a)
    c=CUDA_SOURCE.index('__device__ float mpas_sw_rh(');d=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw.F:3113-3167',c)
    source=root.joinpath('glibc_flt32.cuh').read_text()+CUDA_SOURCE[a:b]+CUDA_SOURCE[c:d]
    host=HostModule('mpas_v842_sw_aerosol',source)
    nc,nz=7,6;nl=nz+1;rng=np.random.default_rng(842)
    pressure=rng.uniform(30000,100000,(nc,nz)).astype(np.float32);temperature=rng.uniform(230,305,(nc,nz)).astype(np.float32)
    qv=rng.uniform(.0001,.02,(nc,nz)).astype(np.float32);dz=rng.uniform(50,500,(nc,nz)).astype(np.float32)
    nwfa=rng.uniform(1e6,5e8,(nc,nz)).astype(np.float32);nifa=rng.uniform(1e2,1e6,(nc,nz)).astype(np.float32)
    xland=np.array([1,2,1,2,1,2,1],np.float32)
    aod3=np.empty((nc,nz),np.float32);aod2=np.empty(nc,np.float32);typ=np.empty(nc,np.int32)
    host.get_function('mpas_sw_gt_aod')((1,),(128,),(nc,nz,pressure,temperature,qv,dz,nwfa,nifa,xland,aod3,aod2,typ))
    ht=np.zeros(nc,np.float32);unused=np.zeros(nc,np.float32);tau=np.empty((nc,14,nl),np.float32);ssa=np.empty_like(tau);asy=np.empty_like(tau)
    host.get_function('mpas_sw_aerosol')((1,),(128,),(nc,nz,nl,1,2,3,3,3,np.float32(.12),np.float32(1.3),np.float32(.85),np.float32(.9),0,pressure,temperature,qv,dz,ht,typ,aod2,aod3,unused,unused,unused,tau,ssa,asy))
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        actual=thompson_aerosols(**{n:xp.asarray(v) for n,v in dict(pressure=pressure,temperature=temperature,qv=qv,dz=dz,nwfa=nwfa,nifa=nifa,xland=xland).items()})
        for name,expected in [('taod5503d',aod3),('taod5502d',aod2),('taer_type',typ),('tauaer',tau),('ssaaer',ssa),('asyaer',asy)]:
            got=actual[name].to_numpy();bad=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32))
            if bad:raise AssertionError(f'{name} differs at {bad} words')
    print('native Thompson SW aerosols match selected source for mixed land/ocean columns',flush=True)

if __name__=='__main__':main()
