"""Strict direct-normal radiation driver operation and refusal boundary."""
from unittest.mock import patch
import numpy as np
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.bandwidth_family import direct_normal_flux
from woof.wrf_exact import ENABLED
assert ENABLED
cos=np.array([-1.,-0.,0.,1.e-45,1.e-7,1.e-6,.1,1.],np.float32)
direct=np.array([2.,2.,2.,1.e-40,1.e-7,1.e-6,3.,500.],np.float32)
with np.errstate(all='ignore'):
    expected=direct/np.maximum(cos,np.float32(1.e-6))
    expected[cos<=np.float32(0)]=np.float32(0)
with MetalContext() as context:
    xp=MetalArrayModule(context)
    d=xp.asarray(direct);c=xp.asarray(cos);out=xp.asarray(np.full(cos.shape,-99,np.float32))
    direct_normal_flux(d,c,out)
    got=out.to_numpy()
    assert np.array_equal(got.view(np.uint32),expected.view(np.uint32))
    invalid=xp.asarray(np.full(cos.shape,np.nan,np.float32))
    before=out.to_numpy().view(np.uint32).copy()
    with patch.object(context,'submit_kernel') as submit:
        try:direct_normal_flux(invalid,c,out)
        except ValueError as error:assert 'finite inputs' in str(error)
        else:raise AssertionError('nonfinite input accepted')
        assert submit.call_count==0
    assert np.array_equal(out.to_numpy().view(np.uint32),before)
    huge=xp.asarray(np.full(cos.shape,np.finfo(np.float32).max,np.float32))
    try:direct_normal_flux(huge,c,out)
    except ValueError as error:assert 'nonfinite result' in str(error)
    else:raise AssertionError('nonfinite result published')
    assert np.array_equal(out.to_numpy().view(np.uint32),before)
print(f'direct normal: 0/{got.size} word differences; both refusal paths preserve output')
