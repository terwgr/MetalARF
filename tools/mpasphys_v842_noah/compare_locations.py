"""Compare the ordered native Noah module-location update."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.noah_v842 import _CORE_CUDA,_STATE_CUDA
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source='extern "C" __device__ void noah_finish_module_locations('+_CORE_CUDA.split('extern "C" __device__ void noah_finish_module_locations(',1)[1]
fn=HostModule('mpas_noah_locations',source+_STATE_CUDA).get_function('finish_module_locations')
locations=np.array([[i,1] for i in range(1,8)],np.int32)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*a,**k):
        result=submit(*a,**k);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    initial=np.array([77,88,99,111],np.int32)
    for routes in (np.array([1,1,1,2,0,0,1],np.int32),np.zeros(7,np.int32)):
        expected=initial.copy()
        fn((1,),(1,),(locations,routes,expected,np.int32(len(routes))))
        state=xp.asarray(initial)
        _launch('hexv842_mpas_noah_locations_i32',1,context,
                dict(locations=xp.asarray(locations),routes=xp.asarray(routes),state=state),dict(n=np.int32(len(routes))))
        actual=state.to_numpy()
        print('state',actual.tolist(),'expected',expected.tolist())
        assert np.array_equal(actual,expected)
