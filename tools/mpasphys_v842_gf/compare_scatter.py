"""Check GF output coupling word-for-word across all native result fields."""
import numpy as np
from woof.hex.mpasphys import gf_mpas_v842 as native
from woof.hex.mpasphys.metal_gf_v842 import scatter_outputs
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
n,nz=4,5;rng=np.random.default_rng(842)
levels=rng.integers(0,2**32,(n,6,nz),dtype=np.uint32).view(np.float32)
surface=rng.integers(0,2**32,(n,6),dtype=np.uint32).view(np.float32)
integer=rng.integers(-100,100,(n,3),dtype=np.int32)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kw):
        result=submit(*args,**kw);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    tend={name:xp.empty((nz,n),np.float32) for name in native.LEVEL_OUTPUTS[:4]}
    diag={name:xp.empty((nz,n),np.float32) for name in native._SCATTER_LEVEL_NAMES[4:]}
    diag.update({name:xp.empty((n,),np.float32) for name in native._SCATTER_SURFACE_NAMES})
    diag.update({name:xp.empty((n,),np.int32) for name in native.INTEGER_OUTPUTS})
    scatter_outputs(xp.asarray(levels),xp.asarray(surface),xp.asarray(integer),tend,diag)
    words=0
    for i,name in enumerate(native._SCATTER_LEVEL_NAMES):
        got=(tend if name in tend else diag)[name].to_numpy().view(np.uint32);ref=levels[:,i,:].T.view(np.uint32)
        assert np.array_equal(got,ref),name;words+=got.size
    for src,names in ((surface,native._SCATTER_SURFACE_NAMES),(integer,native.INTEGER_OUTPUTS)):
        for i,name in enumerate(names):
            got=diag[name].to_numpy().view(np.uint32);ref=src[:,i].view(np.uint32)
            assert np.array_equal(got,ref),name;words+=got.size
    print('GF native scatter words',words,'differences 0')
