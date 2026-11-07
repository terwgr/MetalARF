"""Compare native SW McICA against selected sequential KISS source."""
from importlib.resources import files
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import mcica_subcolumns
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    root=files('woof.core.kernels')
    a=CUDA_SOURCE.index('__device__ float mpas_kiss(');b=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_lw.F:1751-1818',a)
    c=CUDA_SOURCE.index('extern "C" __global__ void mpas_sw_mcica(');d=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw.F:2016-2048',c)
    source=root.joinpath('glibc_flt32.cuh').read_text()+CUDA_SOURCE[a:b]+CUDA_SOURCE[c:d]
    kernel=HostModule('mpas_v842_sw_mcica',source).get_function('mpas_sw_mcica')
    nc,nl=7,7;rng=np.random.default_rng(842)
    layers=np.zeros((15,nc,nl),np.float32)
    layers[0]=np.linspace(975.123,200.789,nl,dtype=np.float32)[None,:]+rng.uniform(-1,1,(nc,nl)).astype(np.float32)
    layers[8]=rng.uniform(0,1,(nc,nl));layers[8,0,1::2]=0
    for i in (9,10,11):layers[i]=rng.uniform(0,20,(nc,nl))
    height=np.broadcast_to(np.linspace(50,15000,nl,dtype=np.float32),(nc,nl)).copy()
    latitude=np.linspace(-70,70,nc,dtype=np.float32)
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        dl=xp.asarray(layers);dh=xp.asarray(height);dlat=xp.asarray(latitude)
        for overlap in range(6):
            for decor in (0,1):
                out=np.zeros((8,nc,nl,112),np.float32);out[5]=1
                kernel((1,),(128,),(nc,nl,overlap,decor,180,0,1,layers[0],layers[8],layers[9],layers[10],layers[11],height,latitude,layers[0],out))
                actual=mcica_subcolumns(layers=dl,height=dh,latitude=dlat,overlap=overlap,decorrelation=decor,julday=180).to_numpy()
                bad=np.count_nonzero(actual.view(np.uint32)!=out.view(np.uint32))
                if bad:raise AssertionError(f'McICA overlap {overlap} decor {decor}: {bad} words')
    print('native SW McICA matches selected KISS source for all overlap and decorrelation selectors',flush=True)

if __name__=='__main__':main()
