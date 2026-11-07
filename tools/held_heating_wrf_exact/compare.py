"""Compare actual retained-heating caller with its selected CUDA source."""
import hashlib,json,os
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from tools.thompson_real_column_parity import host_backend
host_backend.install()
from woof.core.kernels import module_source
from woof.core.dycore import add_h_diabatic_tendency
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
original_submit=MetalContext._submit_kernel_batch
def synchronous_submit(self,*args,**kwargs):
 result=original_submit(self,*args,**kwargs)
 if result is not None:result.wait()
 return result
MetalContext._submit_kernel_batch=synchronous_submit
source=module_source('held_heating')
reference=host_backend.HostModule('strict_held_heating_reference',source).get_function('add_held_heating')
rng=np.random.default_rng(127)
cases=words=subnormals=0
with MetalContext() as context:
 xp=MetalArrayModule(context)
 for nx in (7,11):
  for mapped in (False,True):
   for tiny in (False,True):
    nz,ny=5,3;shape=(nz,ny,nx);plane=(ny,nx)
    def rnd(shape,low=-1,high=1):return rng.uniform(low,high,shape).astype(np.float32)
    data=dict(rth_t=rnd(shape),h_diabatic=rnd(shape),mub2d=rnd(plane,3,5),mup=rnd(plane),c1h=rnd((nz,),.5,1),c2h=rnd((nz,),.1,1),msft=rnd(plane,.5,1.5))
    if tiny:
     for name in ('rth_t','h_diabatic'):data[name]*=np.float32(2**-128)
     data['rth_t'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
    expected=data['rth_t'].copy()
    args=[data[n].copy() for n in ('h_diabatic','mub2d','mup','c1h','c2h','msft')]
    reference(((expected.size+127)//128,), (128,), (expected,*args,np.uint64(expected.size),np.int32(ny*nx),np.int32(mapped)))
    state=SimpleNamespace(_metal_setup_state=True,_array_module=xp,has_msf=mapped,**{n:xp.asarray(a.copy()) for n,a in data.items()})
    add_h_diabatic_tendency(state)
    actual=state.rth_t.to_numpy()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32),err_msg=str((nx,mapped,tiny)))
    cases+=1;words+=actual.size
    subnormals+=np.count_nonzero((expected.view(np.uint32)&0x7f800000)==0)
print(json.dumps(dict(cases=cases,words=words,different_words=0,subnormal_or_zero_words=int(subnormals),source_sha256=hashlib.sha256(source.encode()).hexdigest())))
