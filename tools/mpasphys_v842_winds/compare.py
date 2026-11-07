"""Compare native wind geometry and words with its selected source kernel."""
from types import SimpleNamespace
import os
import sys
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_physics_prep_v841 import _CUDA_SOURCE
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.math_v842 import bind_common_carrier_math
from woof.hex.metal_physics_prep_v841 import MetalMpasToPhysGeometryV841
from woof.hex.mpasphys.metal_winds_v842 import MetalMpasWinds
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

start=_CUDA_SOURCE.index('extern "C" __global__ void reconstruct_winds_v841_f32(')
end=_CUDA_SOURCE.index('extern "C" __global__ void prep_noahmp_sounding_v841_f32(',start)
source=bind_common_carrier_math(CUDA_FTZ_HELPERS+'\n'+_CUDA_SOURCE[start:end])
host=HostModule('native_wind_reconstruction',source).get_function('reconstruct_winds_v841_f32')
lat=np.array([0.,-0.,.7,-1.2,np.array(1,np.uint32).view(np.float32)],np.float32)
lon=np.array([-0.,np.array(1,np.uint32).view(np.float32),2.1,-3.,.2],np.float32)
edges=np.array([[0,1,-1],[1,2,3],[2,4,-1],[3,4,5],[-1,-1,-1]],np.int32)
count=np.array([2,3,2,3,0],np.int32);coeff=np.zeros((5,3,3),np.float32)
coeff[0,:2]=[[.25,-.1,.3],[-.2,.4,.15]];coeff[1]=[[.1,.2,-.3],[.2,-.3,.4],[-.4,.2,.1]];coeff[2,:2]=[[.5,.1,.2],[-.3,.1,.25]];coeff[3]=[[.2,-.1,.15],[.1,.2,-.3],[-.2,.4,.2]]
mesh={'latCell':lat,'lonCell':lon,'nEdgesOnCell':count,'edgesOnCell':edges,'coeffs_reconstruct':coeff,'cellsOnEdge':np.zeros((6,2),np.int32),'garbage_cell':4,'regional_garbage_donor_cell':2,'on_a_sphere':True}
velocity=np.array([[.3,-.2,.1,2.,-1.,.5],[.01,3.,-.7,.9,.6,-2.],[2.,1.,.2,-.3,.1,5.]],np.float32)
nlev,ne=velocity.shape;nc=len(lat);width=edges.shape[1]
x=np.zeros((nlev,nc),np.float32);y=x.copy();z=x.copy();u=x.copy();v=x.copy();invalid=np.zeros(1,np.int32)
host((1,),(128,),(np.int32(nlev),np.int32(nc),np.int32(ne),np.int32(width),velocity,edges,count,coeff,lat,lon,edges.copy(),count.copy(),coeff.copy(),lat.copy(),lon.copy(),x,y,z,u,v,invalid))
if invalid[0]:raise AssertionError('selected source rejected valid geometry')
u[:,-1]=u[:,2];v[:,-1]=v[:,2]
if os.environ.get('WINDS_HOST_ONLY'):
    print('pinned source host wind words',u.size+v.size,'invalid',int(invalid[0]),flush=True);sys.exit(0)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context);geometry=MetalMpasToPhysGeometryV841.from_host(context,mesh)
    adapter=MetalMpasWinds(geometry);atmosphere=SimpleNamespace(saved=SimpleNamespace(normal_velocity=xp.asarray(velocity)))
    result=adapter(atmosphere);bad=0
    for name,value,expected in (('u',result.u_p,u),('v',result.v_p,v)):
        got=value.to_numpy().view(np.uint32);want=expected.view(np.uint32);idx=np.flatnonzero(got!=want);bad+=len(idx)
        print(name,'differences',len(idx),'/',want.size,[(int(i),hex(int(want.flat[i])),hex(int(got.flat[i]))) for i in idx[:8]],flush=True)
    geometry._lat_cell.fill(0.)
    try:adapter(atmosphere)
    except ValueError as error:
        if 'validation' not in str(error):raise
    else:raise AssertionError('changed live geometry must be refused')
    print('sealed geometry mutation refused; regional donor copied',flush=True)
    if bad:raise SystemExit(1)
