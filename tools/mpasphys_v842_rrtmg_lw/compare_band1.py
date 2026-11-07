"""Compare native LW first band with selected source and explicit tables."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_lw_v842 import _solver_source
from woof.core.rrtmg_lw import GPU_BAND_TABS
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import band1
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=_solver_source();m=re.search(r'extern "C" __global__ void rlw_taugb1\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
macros=source[source.index('#define SL_PAVEL 0'):source.index('extern "C" __global__ void rlw_taugb1(')]
tables=GPU_BAND_TABS[1];chosen=source[m.start():j]
chosen=chosen.replace('const float* const* __restrict__ tabs,',','.join('const float* __restrict__ in_'+n for n in tables)+',')
for ix,n in enumerate(tables):chosen=chosen.replace(f'const float* {n} = tabs[{ix}];',f'const float* {n} = in_{n};')
header='''#define NGPTLW 140
#define MAXXSEC 4
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
'''
host=HostModule('mpas_v842_lw_band1',header+macros+chosen).get_function('rlw_taugb1')
nc,nl=2,5;fs=np.ones((33,nc,nl),np.float32);isv=np.ones((6,nc,nl),np.int32)
fs[0]=np.array([900,400,150,30,5],np.float32);fs[1]=2e23;fs[2:10]=.3;fs[10:14]=.25;fs[26:33]=.1
isv[0]=np.array([2,4,10,16,20],np.int32);isv[1]=2;isv[2]=2;isv[3]=3;isv[4]=2;isv[5]=3
laytrop=np.full(nc,3,np.int32);wx=np.zeros((nc,4,nl),np.float32);chi=np.ones(413,np.float32)
rng=np.random.default_rng(842);sizes=(650,2350,100,40,190,190,10,10)
tabs={n:rng.uniform(.001,.5,size).astype(np.float32) for n,size in zip(tables,sizes)}
taug=np.zeros((nc,nl,140),np.float32);fracs=np.zeros_like(taug)
host(((nc*nl+127)//128,),(128,),(np.int32(nc),np.int32(nl),laytrop,fs,isv,wx,chi,np.float32(.999999),*[tabs[n] for n in tables],taug,fracs))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    actual=xp.zeros(taug.shape,np.float32);af=xp.zeros(fracs.shape,np.float32)
    band1(*map(xp.asarray,(laytrop,fs,isv,wx,chi)),{n:xp.asarray(v) for n,v in tabs.items()},actual,af,oneminus=np.float32(.999999))
    bad=0
    for name,got,expected in (('taug',actual,taug),('fracs',af,fracs)):
        diff=np.count_nonzero(got.to_numpy().view(np.uint32)!=expected.view(np.uint32));bad+=diff
        print(name,'words',expected.size,'differences',diff)
    if bad:raise SystemExit(1)
