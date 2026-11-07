import numpy as np
from unittest.mock import patch
from tools.goddard_metal.column_probe import fixture
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.goddard_family import interface_temperature, solve_chunk
import woof.metal.goddard_family as family
from woof.core.radiation_tables import radiation_table_host_words
nz,nc=10,6
inputs,planes,surface,exner=fixture(nz,nc)
names=('pw','temperature','pressure','qv','cldfra','dz','rho','qi','qc','qr','qs','qg','qh','re_ice','re_cloud','re_rain','re_snow','re_graupel','re_hail')
fields=dict(zip(names,planes)); fields.update(latitude=surface[0],cosine=surface[1],tsk=surface[2],emiss=surface[3],albedo=surface[4],exner=exner)
fields['pressure']=np.full_like(fields['pressure'],90000.);fields['cosine']=np.ones_like(fields['cosine'])
with MetalContext() as ctx:
 xp=MetalArrayModule(ctx); d={n:xp.asarray(v) for n,v in fields.items()}
 d['tw']=interface_temperature(d['temperature'],xp.asarray(inputs['z']),xp.asarray(inputs['fnm']),xp.asarray(inputs['fnp']),levels=nz,columns=nc)
 a=xp.asarray(np.full((nz,nc),-99,np.float32));b=xp.asarray(np.full((nz,nc),-99,np.float32));c=xp.asarray(np.full((15,nc),-99,np.float32));table=xp.asarray(radiation_table_host_words('goddard-cloud'))
 original=family._submit; calls=[]; indices=[]
 def record(name,*args,**kwargs):
  calls.append(name); result=original(name,*args,**kwargs)
  if name=='goddard_prepare_columns':indices.append(args[0]['flags'].to_numpy().reshape(3,nc).copy())
  return result
 for pressure,expected in ((np.full((nz,nc),90000.,np.float32),(1,1)),
                  (np.broadcast_to(np.array([1000,1000,1000,900,800,400,500,700,200,100],np.float32)[:,None]*100,(nz,nc)).copy(),(5,3)),
                  (np.broadcast_to(np.array([1000,1000,1000,1000,1000,900,550,100,100,100],np.float32)[:,None]*100,(nz,nc)).copy(),(4,4))):
  d['pressure']=xp.asarray(pressure);calls.clear()
  with patch.object(family,'_submit',side_effect=record):
   try:solve_chunk(d,table,levels=nz,full_columns=nc,first=0,columns=nc,julday=172,solcon=1361.,direct_radii=True,heat_lw=a,heat_sw=b,surface=c)
   except ValueError as e:assert 'cloud indices' in str(e)
   else:raise AssertionError('unsafe indices accepted')
  assert calls==['goddard_pack_grid','goddard_prepare_columns'],calls
  assert np.all(indices[-1][0]==expected[0]) and np.all(indices[-1][1]==expected[1])
  for x in (a,b,c):assert np.all(x.to_numpy()==np.float32(-99))
 print('top-layer, reversed and equal active cloud indices refused before solve and output scatter')
