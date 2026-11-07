"""Compare native shortwave time and solar controls with selected source."""
from importlib.resources import files
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import radiation_controls
from woof.metal.runtime import MetalContext


def main():
    root=files('woof.core.kernels')
    start=CUDA_SOURCE.index('extern "C" __global__ void mpas_sw_driver_times(')
    end=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw_aerosols.F:890-921',start)
    source=root.joinpath('glibc_flt32.cuh').read_text()+root.joinpath('glibc_trig_flt32.cuh').read_text()+CUDA_SOURCE[start:end]
    host=HostModule('mpas_v842_sw_controls',source)
    times=host.get_function('mpas_sw_driver_times'); solar=host.get_function('mpas_sw_radconst')
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync
        for day in (1,40,79,80,81,172,266,365,366):
            for seconds in (0,5432.5,86400):
                args=dict(seconds=seconds,interval=900,julian=day,degrad=np.float32(np.pi/180),dpd=np.float32(360/365))
                expected=np.empty(4,np.float32)
                times((1,),(1,),(np.float32(seconds),np.float32(900),expected[:2]))
                solar((1,),(1,),(np.float32(day),args['degrad'],args['dpd'],expected[2:]))
                actual=radiation_controls(ctx,**args).to_numpy()
                if not np.array_equal(actual.view(np.uint32),expected.view(np.uint32)):
                    raise AssertionError(f'controls {day}/{seconds}: {actual} != {expected}')
    print('native SW time and solar controls match selected source, 27 calendar cases',flush=True)

if __name__=='__main__':main()
