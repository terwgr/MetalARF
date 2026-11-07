"""Compare day-column solver inputs to native wrapper expressions."""
import numpy as np
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    n=131;cosine=np.linspace(.01,1,n,dtype=np.float32);cosine[0]=np.array(1,np.uint32).view(np.float32)
    albedo=np.linspace(.05,.9,n,dtype=np.float32);controls=np.array([0,15,.2,1370],np.float32)
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        out={'prmu0':xp.empty((n,),np.float32),**{k:xp.empty((n,14),np.float32) for k in ('adjflux','albdir','albdif')}}
        _launch('hexv842_sw_solver_inputs_f32',n,ctx,dict(cosine=xp.asarray(cosine),albedo=xp.asarray(albedo),controls=xp.asarray(controls),**out),dict(nc=np.int32(n)))
        ref={'prmu0':np.where(cosine<=np.float32(1e-10),np.float32(1e-10),cosine),
             'adjflux':np.full((n,14),np.float32(np.float32(1.)*np.float32(controls[3]/np.array(1152059146,np.uint32).view(np.float32))),np.float32),
             'albdir':np.broadcast_to(albedo[:,None],(n,14)).copy(),'albdif':np.broadcast_to(albedo[:,None],(n,14)).copy()}
        for name in out:
            if not np.array_equal(out[name].to_numpy().view(np.uint32),ref[name].view(np.uint32)):raise AssertionError(name)
    print('sunlit SW solver input words match native wrapper expressions',flush=True)

if __name__=='__main__':main()
