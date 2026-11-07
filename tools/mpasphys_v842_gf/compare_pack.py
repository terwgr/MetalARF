"""Compare native GF's Metal gather against its selected source boundary."""
from pathlib import Path
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.core import kernels
from woof.hex.mpasphys import gf_mpas_v842 as native
from woof.hex.mpasphys.metal_gf_v842 import pack_inputs
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
header=Path(kernels.__file__).with_name('glibc_flt32.cuh').read_text()
source=header+'\n#define GF_HOST_WITNESS\n'+native._PACK_SURFACE+'''
extern "C" __global__ void check(int n,const float* ht,const float* hfx,const float* qfx,const float* xland,const float* gsw,const float* density,const float* area,const float* lw,const float* sw,int nz,float len_disp,float dt,float* surface,float* radiation){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*nz)return;int c=i%n,k=i/n;
 radiation[i]=__fadd_rn(lw[i],sw[i]);if(k==0)gf_surface_column(ht[c],hfx[c],qfx[c],xland[c],gsw[c],density[c],area[c],len_disp,dt,surface+8*c);
}
'''
host=HostModule('native_gf_pack',source).get_function('check')
nz,n=5,4;rng=np.random.default_rng(842)
fields={name:rng.uniform(-.01,.01,(nz,n)).astype(np.float32) for name in native._PACK_LEVEL_SOURCES}
fields['rthratenlw'].flat[0]=np.array(1,np.uint32).view(np.float32);fields['rthratensw'].flat[0]=np.array(2,np.uint32).view(np.float32)
for name in native._PACK_VECTOR_SOURCES[1:]:fields[name]=rng.uniform(.1,2,n).astype(np.float32)
fields['meshDensity'][:]=[.25,1,2,4];fields['zgrid']=rng.uniform(0,10,(nz+1,n)).astype(np.float32)
surface=np.empty((n,8),np.float32);radiation=np.empty((nz,n),np.float32)
args=(np.int32(n),fields['zgrid'][0],*(fields[name] for name in native._PACK_VECTOR_SOURCES[1:]),fields['rthratenlw'],fields['rthratensw'],np.int32(nz),np.float32(1000),np.float32(60),surface,radiation)
host(((n*nz+127)//128,),(128,),args)
expected=np.empty((n,15,nz),np.float32)
for i,name in enumerate(native.LEVEL_INPUTS):
    src={'rthften':'rthdynten','rqvften':'rqvdynten'}.get(name,name)
    expected[:,i,:]=(radiation if name=='rthraten' else fields[src]).T
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kw):
        result=submit(*args,**kw);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    levels,output=pack_inputs({name:xp.asarray(value) for name,value in fields.items()},len_disp=1000,dt=60)
    bad=0
    for name,got,ref in (('levels',levels.to_numpy(),expected),('surface',output.to_numpy(),surface)):
        ix=np.flatnonzero(got.view(np.uint32)!=ref.view(np.uint32));bad+=len(ix)
        print(name,'differences',len(ix),'/',ref.size,[(int(i),hex(int(got.view(np.uint32).flat[i])),hex(int(ref.view(np.uint32).flat[i]))) for i in ix[:6]],flush=True)
    if bad:raise SystemExit(1)
