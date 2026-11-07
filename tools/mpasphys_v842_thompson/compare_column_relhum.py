"""Compare native Thompson relative humidity against selected source."""
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys import thompson_v842 as native
from woof.hex.mpasphys.metal_thompson_column_v842 import relative_humidity
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

def method(name):
    start=native._COLUMN.index('__device__ __forceinline__ float '+name+'(')
    return native._COLUMN[start:native._body_end(native._COLUMN,native._COLUMN.index('{',start))]
support=native._SUPPORT[:native._SUPPORT.index('// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.\n__device__ float tc_abs')]
start=native._PRECIP.index('extern "C" __global__ void mpas_thompson_relhum(')
source=support+'struct ThompsonColumn {\n'+method('rslf')+'\n'+method('rsif')+'\n};\n'+native._PRECIP[start:native._body_end(native._PRECIP,native._PRECIP.index('{',start))]
host=HostModule('native_thompson_relhum',source).get_function('mpas_thompson_relhum')
qv=np.array([.01,.005,.001,0,1e-40,.005,.001],np.float32);th=np.array([300,273.16,240,210,260,275,273.16],np.float32);pii=np.array([1,1,.95,1,1,.99,1],np.float32);p=np.array([100000,70000,60000,50000,80000,95000,90000],np.float32)
n=len(qv);ref=np.zeros(n,np.float32);host((1,),(128,),(qv,th,pii,p,ref,np.int32(n)))
with MetalContext(batch_kernel_calls=False) as ctx:
    submit=ctx._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);out=xp.empty((n,),np.float32)
    relative_humidity(xp.asarray(qv),xp.asarray(th),xp.asarray(pii),xp.asarray(p),out)
    got=out.to_numpy();ix=np.flatnonzero(got.view(np.uint32)!=ref.view(np.uint32));print('relative humidity differences',len(ix),'/',n,[(int(i),hex(int(got.view(np.uint32)[i])),hex(int(ref.view(np.uint32)[i]))) for i in ix],flush=True)
    if len(ix):raise SystemExit(1)
