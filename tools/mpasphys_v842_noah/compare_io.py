"""Compare level-first Metal pool conversions with selected native source."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.noah_v842 import _MPAS_IO_CUDA,_MPAS_INPUT_FIELDS,_MPAS_OUTPUT_FIELDS,_device_math_source
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

host=HostModule('mpas_noah_pool_io',_device_math_source()+_MPAS_IO_CUDA)
from_host=host.get_function('noah_from_mpas');to_host=host.get_function('noah_to_mpas')
n=7;rng=np.random.default_rng(42);inputs={}
for name,kind in _MPAS_INPUT_FIELDS.items():
    width={'atmosphere':2,'pressure':2,'soil':4}.get(kind)
    shape=(n,width) if width else (n,)
    inputs[name]=rng.uniform(.05,.8,shape).astype(np.float32)
    if kind=='integer_surface':inputs[name]=np.arange(1,n+1,dtype=np.int32)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*a,**k):
        result=submit(*a,**k);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    arrays={name:xp.asarray(value.T.copy() if value.ndim==2 else value) for name,value in inputs.items()}
    for micro,convection in ((0,0),(1,1)):
        x=np.empty((n,79),np.float32);ix=np.empty((n,7),np.int32)
        params=(np.int32(2),np.int32(2),np.int32(0),np.int32(n),np.float32(60),np.float32(.02),np.int32(micro),np.int32(convection),np.int32(13),np.int32(15),np.int32(2),np.int32(1))
        from_host((1,),(128,),(x,ix,*inputs.values(),*params))
        dx=xp.empty((n,79),np.float32);di=xp.empty((n,7),np.int32)
        locations=xp.empty((n,2),np.int32)
        fields=dict(floats=dx,integers=di,locations=locations,**arrays)
        options=dict(n=np.int32(n),dt=np.float32(60),xice_threshold=np.float32(.02),micro=np.int32(micro),convection=np.int32(convection),isurban=np.int32(13),isice=np.int32(15),step=np.int32(2),usemonalb=np.int32(1))
        _launch('hexv842_mpas_noah_from_f32',n,context,fields,options)
        got=dx.to_numpy();got_i=di.to_numpy()
        bad=np.argwhere(got.view(np.uint32)!=x.view(np.uint32));bad_i=np.argwhere(got_i!=ix)
        print('from',micro,convection,'float mismatches',bad.tolist(),'int mismatches',bad_i.tolist())
        assert not bad.size and not bad_i.size
        assert np.array_equal(locations.to_numpy(),np.array([[i+1,1] for i in range(n)],np.int32))
        # Publish altered driver words; nonzero status must preserve each pool.
        published=x.copy();published+=np.float32(.125)
        status=np.array([0,0,3,0,0,2,0],np.int32)
        host_out={name:inputs[name].copy() for name in _MPAS_OUTPUT_FIELDS}
        device_out={name:xp.asarray(v.T.copy() if v.ndim==2 else v) for name,v in host_out.items()}
        to_host((1,),(128,),(published,status,*host_out.values(),np.int32(0),np.int32(n)))
        _launch('hexv842_mpas_noah_to_f32',n,context,dict(floats=xp.asarray(published),status=xp.asarray(status),**device_out),dict(n=np.int32(n)))
        for name,expected in host_out.items():
            got=device_out[name].to_numpy();got=got.T if got.ndim==2 else got
            assert np.array_equal(got.view(np.uint32),expected.view(np.uint32)),name
        print('to',micro,convection,'matched',sum(v.size for v in host_out.values()),'words')
