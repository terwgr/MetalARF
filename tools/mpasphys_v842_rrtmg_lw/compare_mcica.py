"""Compare the native longwave KISS wrapper with its selected source."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_lw_v842 import _DRIVER_SOURCE as source,_RANDOM_SOURCE
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import mcica
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
macros=source[source.index('#define M(f,k)'):source.index('// MPAS: module_ra_rrtmg_lw.F:11954-11983.')]
def extract(name):
    m=re.search(r'(?:__device__ (?:float|void)|extern "C" __global__ void) '+name+r'\s*\(',source)
    i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return source[m.start():j]
header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()+'''
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
#define rlw_exp(x) gfk_exp(x)
'''
host=HostModule('mpas_v842_lw_mcica',header+_RANDOM_SOURCE+macros+'\n'+'\n'.join(extract(n) for n in ('mpas_kiss','mpas_mcica_make','mpas_lw_mcica'))).get_function('mpas_lw_mcica')
nc,nl=3,8
rng=np.random.default_rng(842)
a=np.zeros((nc,12,nl),np.float32);a[:,0,:]=np.linspace(910.123,2.1,nl,dtype=np.float32);a[:,2,:]=np.arange(nl,dtype=np.float32)*500
a[:,5,:]=rng.uniform(0,1,(nc,nl));a[0,5,:2]=0;a[1,5,2]=np.array([1],np.uint32).view(np.float32)[0]
a[:,6:9,:]=rng.uniform(.001,.7,(nc,3,nl));surface=np.zeros((nc,6),np.float32);surface[:,5]=[-.6,0,.7]
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context);da=xp.asarray(a);ds=xp.asarray(surface)
    for overlap in range(6):
        for idcor,julday in ((0,100),(1,100),(1,250)):
            expected=[np.zeros((nc,nl,140),np.float32) for _ in range(4)]
            host((1,),(128,),(*map(np.int32,(nc,nl,overlap,idcor,julday)),surface,a,*expected))
            got=[xp.asarray(x) for x in expected] if overlap==0 else [xp.empty(x.shape,np.float32) for x in expected]
            mcica(da,ds,*got,overlap=overlap,idcor=idcor,julday=julday)
            diff=sum(np.count_nonzero(x.to_numpy().view(np.uint32)!=y.view(np.uint32)) for x,y in zip(got,expected))
            print('overlap',overlap,'idcor',idcor,'julday',julday,'words',sum(x.size for x in expected),'differences',diff)
            if diff:raise SystemExit(1)
