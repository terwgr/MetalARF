"""Compare native SW table initialization to selected host-compiled source."""
from importlib.resources import files
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE, STATIC_LAYOUT
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_raw_coefficients, initialize_tables, _RAW_NAMES
from woof.core.rrtmg_sw import _CUDA_TABLE_ARRAYS
from woof.hex.mpasphys.metal_radiation_tables_v842 import load_raw_tables
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    header=files('woof.core.kernels').joinpath('glibc_flt32.cuh').read_text()
    start=CUDA_SOURCE.index('__device__ const float MPAS_STATIC_TABLES[6936]')
    src=header+'\n#define RSW_C_BPADE __uint_as_float(1080440641u)\n'+CUDA_SOURCE[start:]
    host=HostModule('mpas_v842_sw_init',src)
    weights=np.empty(224,np.float32);static=np.empty(6936,np.float32);exp=np.empty(10001,np.float32)
    host.get_function('mpas_sw_weights')((1,),(128,),(weights,))
    host.get_function('mpas_sw_static_tables')((55,),(128,),(static,))
    host.get_function('mpas_sw_exp_table')((79,),(128,),(exp,))
    table=load_raw_tables("sw");raw={};shapes={}
    for _,band,var in _CUDA_TABLE_ARRAYS:
        key=f'kg{band}/{_RAW_NAMES[var]}'
        data=table[f'rrsw_kg{band}'][_RAW_NAMES[var]]
        shapes[key]=data.shape;raw[key]=np.ascontiguousarray(data.ravel(order='F'))
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            result=original(*a,**kw);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        actual=initialize_tables(ctx)
        packed=actual.packed.to_numpy(); errors=0; count=0
        for define,band,var in _CUDA_TABLE_ARRAYS:
            key=f'kg{band}/{_RAW_NAMES[var]}';first=var in ('sfluxref','rayla')
            rows=raw[key].size//16
            from woof.hex.mpasphys.rrtmg_sw_v842 import NGC
            n=rows*NGC[band-16];expected=np.empty(n,np.float32)
            host.get_function('mpas_sw_reduce_coefficients')(((n+127)//128,),(128,),
                (band-16,rows,int(first),int(var=='sfluxref'),n,raw[key],weights,expected))
            got=packed[actual.offsets[define]:actual.offsets[define]+n]
            bad=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32));errors+=bad;count+=n
            if bad:print('reduction mismatch',key,int(bad),flush=True)
        for name,got,expected in [('weights',actual.weights.to_numpy(),weights),('static',packed[actual.static_start:actual.exp_start],static),('exp',packed[actual.exp_start:],exp)]:
            bad=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32));errors+=bad;count+=len(expected)
            if bad:print(name,'mismatches',int(bad),flush=True)
        print(f'native SW init compared {count} source words: {errors} differences',flush=True)
        if errors:raise AssertionError('native SW initialization mismatch')

if __name__=='__main__':main()
