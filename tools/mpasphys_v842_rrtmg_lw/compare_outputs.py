"""Compare native LW layer-count and output stages with selected source."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_lw_v842 import _DRIVER_SOURCE as source
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import layer_counts,outputs
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
macros=source[source.index('#define M(f,k)'):source.index('// MPAS: module_ra_rrtmg_lw.F:11954-11983.')]
def extract(name):
    m=re.search(r'(?:__device__ void|extern "C" __global__ void) '+name+r'\s*\(',source)
    i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return source[m.start():j]
header='''#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
'''
host=HostModule('mpas_v842_lw_outputs',header+macros+'\n'+'\n'.join(extract(n) for n in ('mpas_layer_counts','mpas_lw_outputs_one','mpas_lw_outputs')))
count_host=host.get_function('mpas_layer_counts');output_host=host.get_function('mpas_lw_outputs')
nc,nm,nl=4,4,10
interfaces=np.zeros((nc,2,nm+1),np.float32);interfaces[:,0,-1]=[0,1990,2000,2200]
model=np.ones((nc,12,nm),np.float32);model[:,7,:]=[0,.2,1,0];model[1,7,3]=np.array([1],np.uint32).view(np.float32)[0];model[:,11,:]=[.8,.9,1,1.1]
atmosphere=(np.arange(nc*12*nl,dtype=np.float32).reshape(nc,12,nl)+1)/100
arr=[(np.arange(nc*(nl+1),dtype=np.float32).reshape(nc,nl+1)+j)/5 for j in (1,2,3,4)]
heating=(np.arange(nc*nl,dtype=np.float32).reshape(nc,nl)-5)/13
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*a,**k):
        result=submit(*a,**k);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    expected=np.zeros(nc,np.int32);count_host((1,),(128,),(np.int32(nc),np.int32(nm),interfaces,expected))
    di=xp.asarray(interfaces);dc=xp.empty(expected.shape,np.int32);layer_counts(di,dc)
    got=dc.to_numpy();assert np.array_equal(got,expected),(got,expected)
    print('layer counts',got.tolist(),'0/',nc,'differences')
    dm,da,*rest=map(xp.asarray,[model,atmosphere,*arr,heating])
    for icloud in (0,1):
        t=np.empty((nc,nm),np.float32);r=np.empty((nc,3,nm),np.float32);f=np.empty((nc,11),np.float32)
        output_host((1,),(128,),(*map(np.int32,(nc,nm,nl,icloud)),model,atmosphere,*arr,heating,t,r,f))
        dt=xp.empty(t.shape,np.float32);dr=xp.empty(r.shape,np.float32);df=xp.empty(f.shape,np.float32)
        outputs(dm,da,*rest,dt,dr,df,icloud=icloud)
        diffs=sum(np.count_nonzero(x.to_numpy().view(np.uint32)!=y.view(np.uint32)) for x,y in ((dt,t),(dr,r),(df,f)))
        print('icloud',icloud,'words',t.size+r.size+f.size,'differences',diffs)
        if diffs:raise SystemExit(1)
