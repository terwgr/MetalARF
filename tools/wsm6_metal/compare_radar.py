"""Compare Metal start-state WSM6 radar against its host-built CUDA source."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity import host_backend
from tools.thompson_real_column_parity.host_backend import HostModule
host_backend._POINTER_DTYPES.update(Wsm6State=np.uint8,Wsm6Workspace=np.uint8)
from woof.hex.mpasphys import wsm6_refl_start_v842 as source
from woof.hex.mpasphys.metal_wsm6_radar_v842 import launch_radar, relative_humidity
from woof.hex.mpasphys import wsm6_v842 as native
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

nz,nc=8,35
text=source.source(source.device_math_header(),nz,numerics='strict')
text=text.replace('#include <cupy/complex.cuh>','#include <complex>\nnamespace thrust {using std::complex;using std::sqrt;}')
host=HostModule('wsm6_start_radar',text+native.WSM6_RELHUM_SOURCE)
sizes=np.zeros(2,np.uint64)
host.get_function('wsm6_start_sizes')((1,),(1,),(sizes,))
c=np.array([9.806,1004.5,1870.,287.,461.6,273.15,461.6/287-1,287/461.6,0,0,0,1e-12,2.834e6,2.5e6,3.34e5,1000,100,4190,2106,610.78,1e-5,2e-5,4e-5,5e-5,1e-4,1e-3],np.float32)
template=np.zeros(int(sizes[1]),np.uint8);status=np.zeros(1,np.int32)
host.get_function('wsm6_start_template')((1,),(1,),(c,np.int32(0),template,status))
assert status[0]==0
fields={name:np.full((nz,nc),value,np.float32) for name,value in dict(qv=.005,qr=.0001,qs=.00002,qg=.00001,pii=1,p=90000).items()}
fields['th']=np.broadcast_to(np.array([296,289,280,274,270,265,250,230],np.float32)[:,None],(nz,nc)).copy()
fields['zgrid']=np.broadcast_to(np.arange(nz+1,dtype=np.float32)[:,None]*500,(nz+1,nc)).copy()
for col in range(nc):
 for name in ('qr','qs','qg'):
  if col%4==0 or (col%4==1 and name!='qr') or (col%4==2 and name=='qr'):
   fields[name][:,col]=0
  fields[name][:,col]*=np.float32(1+col/10)
targets=('refl10cm','refl10cm_max','refl10cm_1km','refl10cm_1km_max')
expected={n:np.full((nz,nc) if n=='refl10cm' else (nc,),-35,np.float32) for n in targets}
t=fields['th']*fields['pii']
workspace=np.zeros(nc*int(sizes[0]),np.uint8)
host.get_function('wsm6_start_radar')(((nc+31)//32,),(32,),
 (np.int32(nz),np.int32(nc),template,t,*(fields[n] for n in ('qv','qr','qs','qg','p','zgrid')),*(expected[n] for n in targets[:3]),workspace))
expected[targets[3]][:]=expected[targets[2]]
wanted_rh=np.zeros((nz,nc),np.float32)
solve=nc-2
host.get_function('wsm6_relhum')(((nz*solve+63)//64,),(64,),
 (fields['th'],fields['pii'],fields['qv'],fields['p'],wanted_rh,np.int32(nz),np.int32(nc),np.int32(solve)))
with MetalContext(batch_kernel_calls=False) as context:
 submit=context._submit_kernel_batch
 def synchronous(*args,**kwargs):
  result=submit(*args,**kwargs);result.wait();return result
 context._submit_kernel_batch=synchronous
 xp=MetalArrayModule(context)
 inputs={n:xp.asarray(v) for n,v in fields.items()}
 out={n:xp.asarray(np.full_like(v,-35)) for n,v in expected.items()}
 launch_radar(inputs,out,xp.asarray(c))
 bad=0
 for name,want in expected.items():
  got=out[name].to_numpy();idx=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32));bad+=len(idx)
  print(name,want.size,'words',len(idx),'differences',[(int(i),float(got.flat[i]),float(want.flat[i])) for i in idx[:5]],flush=True)
 rh=xp.asarray(np.full_like(wanted_rh,-999))
 relative_humidity(inputs,rh,solve_cells=solve)
 idx=np.flatnonzero(rh.to_numpy().view(np.uint32)!=wanted_rh.view(np.uint32));bad+=len(idx)
 print('relative humidity',wanted_rh.size,'words',len(idx),'differences; halos zero',flush=True)
 for n,want in fields.items(): assert inputs[n].to_numpy().tobytes()==want.tobytes(),n
 assert bad==0,bad
