"""Compare source SW output conversion for heterogeneous day/night columns."""
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import convert_outputs
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    start=CUDA_SOURCE.index('extern "C" __global__ void mpas_sw_outputs(')
    end=CUDA_SOURCE.index('// MPAS mpas_atmphys_driver_radiation_sw.F:874,899.',start)
    header='#define FADD(a,b) __fadd_rn(a,b)\n#define FSUB(a,b) __fsub_rn(a,b)\n#define FMUL(a,b) __fmul_rn(a,b)\n#define FDIV(a,b) __fdiv_rn(a,b)\n'
    src=HostModule('mpas_v842_sw_outputs',header+CUDA_SOURCE[start:end]).get_function('mpas_sw_outputs')
    nc,nz=131,5;rng=np.random.default_rng(842)
    cosine=rng.uniform(.05,1,nc).astype(np.float32);cosine[::3]=-0.2;cosine[1]=0
    cosine[2]=np.array(1,np.uint32).view(np.float32);cosine[3]=np.array(0x80000001,np.uint32).view(np.float32)
    values={'cosine':cosine,'exner':rng.uniform(.6,1,(nc,nz)).astype(np.float32)}
    values.update({n:rng.uniform(1,500,(nc,nz+2)).astype(np.float32) for n in ('up','down','upc','downc','direct')})
    values['heat']=rng.uniform(-8,8,(nc,nz+1)).astype(np.float32)
    values['scalars']=rng.uniform(-1,1,(14,nc)).astype(np.float32)
    values['tendency']=rng.uniform(-1,1,(nc,nz)).astype(np.float32)
    expected={n:x.copy() for n,x in values.items()}
    order=('cosine','exner','up','down','upc','downc','direct','heat','scalars','tendency')
    src(((nc+127)//128,),(128,),(nc,nz,*[expected[n] for n in order]))
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        device={n:xp.asarray(x) for n,x in values.items()}
        convert_outputs(**device)
        for n in ('scalars','tendency'):
            got=device[n].to_numpy();different=np.count_nonzero(got.view(np.uint32)!=expected[n].view(np.uint32))
            if different:raise AssertionError(f'{n}: {different} different words')
    print('SW outputs match selected source for 131 day/night columns and five levels',flush=True)

if __name__=='__main__':main()
