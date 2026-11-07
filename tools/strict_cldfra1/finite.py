"""Check strict cloud-fraction nonfinite refusal at its direct caller."""
from unittest.mock import patch
import numpy as np
from woof.core.rrtmgp import cal_cldfra1
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED
assert ENABLED
with MetalContext() as context:
    xp=MetalArrayModule(context)
    def device(values):
        return tuple(xp.asarray(np.array([[value]],np.float32)) for value in values)
    invalid=device((float('nan'),.001,0,0,280,100000))
    with patch.object(context,'submit_kernel') as submit:
        try:
            cal_cldfra1(*invalid,f_qi=False,f_qs=False)
        except ValueError as error:
            assert 'finite qv' in str(error)
        else: raise AssertionError('nonfinite input accepted')
        assert submit.call_count == 0
    # Finite inputs that make the source cloud-fraction expression nonfinite.
    singular=device((.01,.001,0,0,35.86,0))
    try:
        cal_cldfra1(*singular,f_qi=False,f_qs=False)
    except ValueError as error:
        assert 'nonfinite result' in str(error)
    else: raise AssertionError('nonfinite result published')
print('strict cloud fraction refused nonfinite input before dispatch and result before publication')
