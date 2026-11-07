"""Check native day selection and KISS pressure STOP word boundaries."""
import numpy as np
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    words=np.array([0,0x80000000,1,0x80000001,0x3f800000,0xbf800000,0x7fc00000],np.uint32)
    cosine=words.view(np.float32);nc=len(cosine);nl=4
    play=np.zeros((nc,nl),np.float32);play[:,0]=np.array([1000,1000,1000,np.nextafter(np.float32(1000),np.float32(np.inf)),np.nextafter(np.float32(1000),np.float32(0)),0,1],np.float32)
    play[:,1]=np.array([1000,np.nextafter(np.float32(1000),np.float32(np.inf)),np.nextafter(np.float32(1000),np.float32(0)),1000,1000,np.array(1,np.uint32).view(np.float32),0],np.float32)
    expected=np.flatnonzero(cosine>np.float32(0)).astype(np.int32)
    invalid=(np.float32(play[:,0]*np.float32(100))<np.float32(play[:,1]*np.float32(100))).astype(np.int32)
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        indices=xp.empty((nc,),np.int32);count=xp.zeros((1,),np.int32)
        _launch('hexv842_sw_day_select_i32',1,ctx,dict(cosine=xp.asarray(cosine),indices=indices,count=count),dict(nc=np.int32(nc)))
        n=int(count.to_numpy()[0]);got=indices.to_numpy()[:n]
        if not np.array_equal(got,expected):raise AssertionError(f'day selection {got} != {expected}')
        bad=xp.zeros((nc,),np.int32)
        _launch('hexv842_sw_kiss_guard_i32',nc,ctx,dict(play=xp.asarray(play),invalid=bad),dict(nc=np.int32(nc),nl=np.int32(nl)))
        if not np.array_equal(bad.to_numpy(),invalid):raise AssertionError('KISS pressure equality/neighbor boundary')
    print('native day/zero/subnormal/NaN selection and pressure equality boundaries pass',flush=True)

if __name__=='__main__':main()
