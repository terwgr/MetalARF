"""Compare native Thompson saturation methods with selected CUDA source."""
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys import thompson_v842 as native
from woof.hex.mpasphys.metal_thompson_column_v842 import saturation
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

def method(name):
    start=native._COLUMN.index('__device__ __forceinline__ float '+name+'(')
    return native._COLUMN[start:native._body_end(native._COLUMN,native._COLUMN.index('{',start))]
support=native._SUPPORT[:native._SUPPORT.index('// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.\n__device__ float tc_abs')]
source=support+'struct ThompsonColumn {\n'+method('rslf')+'\n'+method('rsif')+'''\n};
extern "C" __global__ void check(const float* p,const float* t,float* a,float* b,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;ThompsonColumn m;a[i]=m.rslf(p[i],t[i]);b[i]=m.rsif(p[i],t[i]);}
'''
host=HostModule('native_thompson_saturation',source).get_function('check')
p=np.array([100000,70000,20000,1,0,-1,np.inf,1e-40,1e-38,95000,95000],np.float32)
t=np.array([300,273.16,193.16,220,220,220,220,220,220,np.nan,-np.inf],np.float32)
n=len(p);a=np.zeros(n,np.float32);b=a.copy();host((1,),(128,),(p,t,a,b,np.int32(n)))
with MetalContext(batch_kernel_calls=False) as ctx:
    submit=ctx._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);da=xp.empty((n,),np.float32);db=xp.empty((n,),np.float32)
    saturation(xp.asarray(p),xp.asarray(t),da,db)
    bad=0
    for name,out,ref in (('liquid',da,a),('ice',db,b)):
        got=out.to_numpy();idx=np.flatnonzero(got.view(np.uint32)!=ref.view(np.uint32));bad+=len(idx)
        print(name,'differences',len(idx),'/',n,[(int(i),hex(int(got.view(np.uint32)[i])),hex(int(ref.view(np.uint32)[i]))) for i in idx],flush=True)
    if bad:raise SystemExit(1)
