"""Exercise the native interface-prefix layout on Metal."""
import numpy as np
from woof.hex.mpasphys.metal_bl_mynn_v842 import interface_prefix
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    rng=np.random.default_rng(842)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for nc,nl in ((1,1),(7,3),(31,33)):
            words=rng.integers(0,2**32,size=(nc,nl+1),dtype=np.uint32);src=words.view(np.float32)
            dst=xp.empty((nc,nl),np.float32);interface_prefix(xp.asarray(src),dst)
            if not np.array_equal(dst.to_numpy().view(np.uint32),words[:,:nl]):raise AssertionError((nc,nl))
    print('MYNN interface-prefix copy preserves all words',flush=True)
if __name__=='__main__':main()
