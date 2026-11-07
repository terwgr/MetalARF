"""Compare the actual sea-ice caller and ownership mask with selected CUDA source.

The reference is host-compiled source, not an NVIDIA device. Synthetic columns
exercise masked water, existing ice, warm water, new freezing and complete melt.
"""
import json, hashlib, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.thompson_real_column_parity import host_backend
host_backend.install()
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from woof.ocean import seaice
from woof.core.surface_ocean import _ice_owned
root=Path.cwd()
source=(root/'woof/ocean/seaice/thermodynamics.cu').read_text()
ref=host_backend.HostModule('seaice_coupled',source).get_function('ice_step')
keys=('xice','ice_volume','ice_snow_mass','ice_skin','ice_ocean_mask','ice_rain','ice_snowfall','ice_growth','ice_melt','ice_energy','ice_energy_error','ice_ocean_exchange','ice_ocean_heat','ice_ocean_coupled','ice_thickness_change','ice_concentration_change','ice_thickness','ice_snow_depth','tsk','sst','ice_air','ice_qair','psfc','chs','swdown','glw','hfx','lh','qfx','qsfc','albedo','emiss','snow','snowh','znt')
n=7
h={k:np.zeros((1,n),np.float32) for k in keys}
for k,v in dict(xice=[0,.8,0,0,.8,.8,.8],ice_volume=[0,.8,0,0,.8,.8,.000001],ice_ocean_mask=[0,1,1,1,1,1,1],sst=[280,271.35,280,271.34,271.35,271.35,280],ice_air=[270]*n,ice_qair=[.003]*n,psfc=[98000]*n,chs=[.01]*n,swdown=[100]*n,glw=[250]*n,ice_ocean_coupled=[9]*n,ice_skin=[270]*n,tsk=[270]*n).items():h[k][...]=v
h['xland']=np.array([[1,2,2,2,1,2,2]],np.float32)
h['glw'][0,-1]=1000
h['ice_air'][0,3]=250
h['glw'][0,3]=180
h['swdown'][0,3]=0
expected={k:v.copy() for k,v in h.items()}
ref((1,),(128,),(*(expected[k] for k in keys),np.float32(60),np.float32(2),np.float32(.1),np.int32(n)))
with MetalContext() as ctx:
 ctx.batch_kernel_calls=False
 original=ctx._submit_kernel_batch
 def sync(*a,**kw):
  result=original(*a,**kw)
  if result is not None:result.wait()
  return result
 ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
 f={k:xp.asarray(v) for k,v in h.items()}
 driver=SimpleNamespace(fields=f,bldt_seconds=60)
 cfg=SimpleNamespace(seaice_ocean_heat_flux=2,seaice_new_ice_thickness=.1)
 seaice.step(driver,{'temperature':f['ice_air'].view((1,1,n)),'qv':f['ice_qair'].view((1,1,n))},cfg)
 result={}
 for k in keys:
  a=f[k].to_numpy();b=expected[k];result[k]=dict(different_words=int(np.count_nonzero(a.view('u4')!=b.view('u4'))),max_abs=float(np.max(np.abs(a-b))))
 assert not any(v['different_words'] for v in result.values()),result
 np.testing.assert_array_equal(expected['ice_ocean_coupled'],np.array([[9,1,0,1,1,1,1]],np.float32))
 np.testing.assert_array_equal(f['ice_ocean_coupled'].to_numpy(),expected['ice_ocean_coupled'])
 owned=_ice_owned(f).to_numpy();want=((expected['ice_ocean_mask']>0)&((expected['ice_ocean_coupled']>0)|(expected['xland']<=1.5))).astype(np.float32)
 np.testing.assert_array_equal(owned,want)
 out=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),host_flags=host_backend.CXX_FLAGS,fields=result,coupled=expected['ice_ocean_coupled'].tolist(),owned=owned.tolist())
 Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
