"""Compare native LW ordered accumulation and final flux/heating stages."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.core.rrtmg_lw import _lw_coalesced_source
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import transfer_accumulate,transfer_final,_static_words
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=_lw_coalesced_source((ROOT/'woof/core/kernels/rrtmg_lw_chain.cu').read_text())
def extract(name):
    m=re.search(r'extern "C" __global__ void '+name+r'\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return source[m.start():j]
header='''#define NGPTLW 140
#define NBNDLW 16
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
'''
host=HostModule('mpas_v842_lw_flux',header+extract('rlw_rtrn_accum_coalesced')+extract('rlw_rtrn_final_coalesced'))
nc,nl=2,5;rng=np.random.default_rng(842);shape=(nc,nl,140)
inputs={n:rng.uniform(0,5,shape).astype(np.float32) for n in ('radld_p','radclrd_p','radlu_p','radclru_p')}
inputs['radld_p'][0,0,0]=np.array([1],np.uint32).view(np.float32)[0]
inputs['iclddn_p']=(rng.random(shape)>.5).astype(np.uint8)
inputs.update({n:rng.uniform(0,5,(nc,140)).astype(np.float32) for n in ('radlu_sfc','radclru_sfc')})
tables={'ngs':_static_words('ngs',16,True),'delwave':_static_words('delwave',16,False),
        'fluxfac':np.float32(62831.85546875),'heatfac':np.float32(8.434944152832031)}
fluxnames=('totuflux','totdflux','totuclfl','totdclfl');out={n:np.zeros((nc,nl+1),np.float32) for n in fluxnames}
out.update({n:np.zeros((nc,nl+1),np.float32) for n in ('fnet','fnetc','htr','htrc')})
pz=np.tile(np.array([1000,800,600,400,200,30],np.float32),(nc,1))
host.get_function('rlw_rtrn_accum_coalesced')(((nc*(nl+1)+127)//128,),(128,),(*map(np.int32,(nc,nl)),*[inputs[n] for n in ('radld_p','radclrd_p','radlu_p','radclru_p','iclddn_p','radlu_sfc','radclru_sfc')],tables['ngs'],tables['delwave'],np.float32(.5),*[out[n] for n in fluxnames]))
accum={n:v.copy() for n,v in out.items()}
host.get_function('rlw_rtrn_final_coalesced')((1,),(128,),(*map(np.int32,(nc,nl)),pz,tables['fluxfac'],tables['heatfac'],*[out[n] for n in (*fluxnames,'fnet','fnetc','htr','htrc')]))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    di={n:xp.asarray(v.astype(np.int32) if n=='iclddn_p' else v) for n,v in inputs.items()};dt={n:xp.asarray(v) if isinstance(v,np.ndarray) else v for n,v in tables.items()}
    dout={n:xp.zeros(v.shape,np.float32) for n,v in out.items()}
    transfer_accumulate(di,dt,dout)
    for n in fluxnames:
        if not np.array_equal(dout[n].to_numpy().view(np.uint32),accum[n].view(np.uint32)):raise AssertionError('accum '+n)
    transfer_final(xp.asarray(pz),dt,dout)
    total=bad=0
    for name,expected in out.items():
        got=dout[name].to_numpy();diff=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32));total+=expected.size;bad+=diff
        if diff:print(name,'differences',diff)
    print('flux and heating words',total,'differences',bad)
    if bad:raise SystemExit(1)
