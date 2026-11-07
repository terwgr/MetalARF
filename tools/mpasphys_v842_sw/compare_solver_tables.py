"""Compare native initialized table words with the existing solver packing."""
import numpy as np
from woof.core.rrtmg_sw import load_sw_tables,_pack_cuda_tables
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_tables
from woof.metal.runtime import MetalContext


def main():
    expected,_=_pack_cuda_tables(load_sw_tables())
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync
        actual=initialize_tables(ctx).packed.to_numpy()
    if actual.shape!=expected.shape:raise AssertionError(f'{actual.shape} != {expected.shape}')
    bad=np.flatnonzero(actual.view(np.uint32)!=expected.view(np.uint32))
    print('native vs existing solver packed tables mismatched:',len(bad),'of',len(actual),flush=True)
    if bad.size:
        for x in bad[:20]:print(int(x),actual[x],expected[x])
        raise AssertionError('table mismatch')

if __name__=='__main__':main()
