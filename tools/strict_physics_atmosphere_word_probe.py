"""Compare actual strict phy_prep callers with the retained shared source."""
import os
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.core import physics
from woof.core.state import DomainState
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

class State(SimpleNamespace):
 def total_mu(self):return DomainState.total_mu(self)
 def total_theta(self):return DomainState.total_theta(self)
 def scratch(self,shape,slot):
  if slot not in self.slots:self.slots[slot]=self._array_module.zeros(shape,np.float32)
  return self.slots[slot]

rng=np.random.default_rng(920)
words=cases=ordinary_rho_differences=0
with MetalContext() as context:
 xp=MetalArrayModule(context)
 for base3d in (False,True):
  for all_species in (False,True):
   for subnormal in (False,True):
    nz,ny,nx=5,3,7;shape=(nz,ny,nx);plane=(ny,nx);faces=(nz+1,ny,nx)
    def rnd(shape,lo,hi):return rng.uniform(lo,hi,shape).astype(np.float32)
    data=dict(p=rnd(shape,10000,100000),thb=rnd(shape if base3d else (nz,),280,310),
     thp=rnd(shape,-1,1),u=rnd((nz,ny,nx+1),-10,10),v=rnd((nz,ny+1,nx),-10,10),
     phb=rnd(faces if base3d else (nz+1,),1,5000),php=rnd(faces,-1,1),
     mub2d=rnd(plane,80000,90000),mup=rnd(plane,-100,100),
     c1h=rnd((nz,),.5,1),c2h=rnd((nz,),1,10),dnw=-np.full(nz,.2,np.float32),
     alt=rnd(shape,.8,1.5))
    if subnormal:
     bits=np.resize(np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff],np.uint32),data['u'].shape)
     data['u']=bits.view(np.float32).copy()
     data['v'].flat[:6]=bits.flat[:6].copy().view(np.float32)
     data['alt'].flat[:4]=np.array([8e37,1e38,2e38,3e38],np.float32)
    for name in ('qv','qc','qr','qi','qs','qg','qh'):
     data[name]=rnd(shape,0,.02) if all_species or name=='qv' else None
    if subnormal:
     data['qv'].flat[:6]=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff],np.uint32).view(np.float32)
    host=State(**{n:None if a is None else a.copy() for n,a in data.items()},
      _metal_setup_state=False,_array_module=np,slots={},p_top=np.float32(5000))
    device=State(**{n:None if a is None else xp.asarray(a.copy()) for n,a in data.items()},
      _metal_setup_state=True,_array_module=xp,slots={},p_top=np.float32(5000))
    saved=physics.cp;physics.cp=np
    try:expected=physics._prepare_atmosphere(host)
    finally:physics.cp=saved
    actual=physics._prepare_atmosphere(device)
    assert set(actual)==set(expected)
    for name in expected:
     a=actual[name].to_numpy();e=np.asarray(expected[name])
     assert a.shape==e.shape and a.dtype==e.dtype,(name,a.shape,e.shape)
     np.testing.assert_array_equal(a.view(np.uint32),e.view(np.uint32),err_msg=str((base3d,all_species,subnormal,name)))
     words+=e.size
    ordinary=(np.float32(1)+data['qv'])/data['alt']
    ordinary_rho_differences+=np.count_nonzero(ordinary.view(np.uint32)!=expected['rho'].view(np.uint32))
    cases+=1
assert ordinary_rho_differences>0,'rho negative control is not discriminating'
print('matched',cases,'actual strict physics atmosphere callers,',words,'words; ordinary rho differs',ordinary_rho_differences,'words')
