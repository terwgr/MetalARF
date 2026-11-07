"""Actual Metal Dudhia submitter versus selected host-compiled CUDA source."""
import hashlib,json,os
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from tools.thompson_real_column_parity import host_backend
host_backend.install()
from woof.core.kernels import module_source
from woof.core.dudhia import _wrf_cssca
from woof.metal.dudhia_shortwave import submit_dudhia_shortwave
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
source=module_source('dudhia_sw')
reference=host_backend.HostModule('strict_dudhia_reference',source).get_function('dudhia_sw_column')
original_submit=MetalContext._submit_kernel_batch
def synchronous_submit(self,*args,**kwargs):
 result=original_submit(self,*args,**kwargs)
 if result is not None:result.wait()
 return result
MetalContext._submit_kernel_batch=synchronous_submit
rng=np.random.default_rng(812)
cases=words=0
with MetalContext() as context:
 xp=MetalArrayModule(context)
 for nz in (5,41):
  for cloudy in (0,1):
   for tiny in (False,True):
    ny,nx=3,7;shape=(nz,ny,nx);plane=(ny,nx);ncol=ny*nx
    def rnd(shape,lo,hi):return rng.uniform(lo,hi,shape).astype(np.float32)
    data=dict(temperature=rnd(shape,210,305),pressure=np.broadcast_to(np.linspace(95000,9000,nz,dtype=np.float32)[:,None,None],shape).copy(),qv=rnd(shape,0,.02),qc=rnd(shape,0,.001),qr=rnd(shape,0,.0002),qi=rnd(shape,0,.001),qs=rnd(shape,0,.0002),qg=rnd(shape,0,.0001),dz=rnd(shape,100,600),exner=rnd(shape,.5,1.05),cosine_zenith=np.resize(np.array([-0.3,0,1e-10,0.1,0.2,0.5,0.9],np.float32),plane),albedo=rnd(plane,.03,.85))
    if tiny:
     for name in ('qv','qc','qr','qi','qs','qg'):data[name]*=np.float32(2**-120)
     data['qv'].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view('f')
    for scat in (0., .25, 1., 2., 10.):
     outputs={name:np.zeros(shape if name=='heating' else plane,np.float32) for name in ('heating','gsw','swdown')}
     arrays=[data[name].copy() for name in ('temperature','pressure','qv','qc','qr','qi','qs','qg','dz','exner','cosine_zenith','albedo')]
     reference(((ncol+127)//128,),(128,),(*arrays,outputs['heating'],outputs['gsw'],outputs['swdown'],np.float32(1361),_wrf_cssca(scat),np.int32(cloudy),np.int32(1),np.int32(ncol),np.int32(nz),np.int64(1),np.int64(-ncol),np.int64((nz-1)*ncol)))
     device={name:xp.asarray(a.copy()) for name,a in data.items()}
     device.update({name:xp.empty(shape if name=='heating' else plane,np.float32) for name in ('heating','gsw','swdown','coszen')})
     submit_dudhia_shortwave(**device,solcon=1361.,swrad_scat=scat,icloud=cloudy)
     for name,expected in {**outputs,'coszen':data['cosine_zenith']}.items():
      actual=device[name].to_numpy()
      np.testing.assert_array_equal(actual.view('u4'),expected.view('u4'),err_msg=str((nz,cloudy,tiny,scat,name)))
      words+=actual.size
     cases+=1
print(json.dumps(dict(cases=cases,words=words,different_words=0,source_sha256=hashlib.sha256(source.encode()).hexdigest(),reference='selected CUDA source compiled on host; actual outputs Metal GPU')))
