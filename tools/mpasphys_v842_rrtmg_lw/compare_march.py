"""Compare native LW transfer march with selected coalesced source."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.core.rrtmg_lw import _lw_coalesced_source
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import transfer_march
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=_lw_coalesced_source((ROOT/'woof/core/kernels/rrtmg_lw_chain.cu').read_text())
m=re.search(r'extern "C" __global__ void rlw_rtrn_march_coalesced\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
header='''#define NGPTLW 140
#define NBNDLW 16
#define RLW_MAXLAY 128
#define RLW_REC6 0.166667f
#define RLW_TBLINT 10000.0f
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
'''
host=HostModule('mpas_v842_lw_march',header+source[m.start():j]).get_function('rlw_rtrn_march_coalesced')
nc,nl=2,5;rng=np.random.default_rng(842);gshape=(nc,nl,140)
inputs={n:rng.uniform(.01,.9,gshape).astype(np.float32) for n in ('cldfmc','odcld','efclfrac','fracs','taut')}
inputs['cldfmc']=(inputs['cldfmc']>.5).astype(np.float32);inputs['icldlyr']=np.array([[0,1,1,0,1],[1,0,1,1,0]],np.int32)
inputs['taut'][:,:,0]=0;inputs['taut'][:,:,1]=.01;inputs['taut'][:,:,2]=.1
inputs['taut'][0,0,3]=np.array(0x80000001,np.uint32).view(np.float32)
inputs.update({n:rng.uniform(.8,1.8,(nc,16)).astype(np.float32) for n in ('secdiff','semiss','plankbnd')})
inputs['planklay']=rng.uniform(1,20,(nc,nl,16)).astype(np.float32);inputs['planklev']=rng.uniform(1,20,(nc,nl+1,16)).astype(np.float32)
tables={'ngb':(np.arange(140)%16+1).astype(np.int32),'bpade':np.float32(3.5971224308013916)}
tables['tau_tbl']=np.linspace(0,20,10001,dtype=np.float32)
tables['exp_tbl']=np.exp(-tables['tau_tbl'].astype(np.float64)).astype(np.float32)
tables['tfn_tbl']=np.linspace(0,.8,10001,dtype=np.float32)
outputs={n:np.zeros(gshape,np.float32) for n in ('radld_p','radclrd_p','radlu_p','radclru_p')}
outputs['iclddn_p']=np.zeros(gshape,np.uint8)
outputs.update({n:np.zeros((nc,140),np.float32) for n in ('radlu_sfc','radclru_sfc')})
order=('cldfmc','odcld','efclfrac','icldlyr','secdiff','semiss','planklay','planklev','plankbnd','fracs','taut')
host(((nc*140+127)//128,),(128,),(*map(np.int32,(nc,nl)),*[inputs[n] for n in order],tables['tau_tbl'],tables['exp_tbl'],tables['tfn_tbl'],tables['bpade'],tables['ngb'],*[outputs[n] for n in ('radld_p','radclrd_p','radlu_p','radclru_p','iclddn_p','radlu_sfc','radclru_sfc')]))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    di={n:xp.asarray(v) for n,v in inputs.items()};dt={n:xp.asarray(v) if isinstance(v,np.ndarray) else v for n,v in tables.items()}
    dout={n:xp.zeros(v.shape,np.int32 if n=='iclddn_p' else np.float32) for n,v in outputs.items()}
    scratch=xp.empty((nc,140,4,nl),np.float32);transfer_march(di,dt,dout,scratch)
    total=bad=0
    for name,expected in outputs.items():
        got=dout[name].to_numpy();want=expected.astype(np.int32) if name=='iclddn_p' else expected
        diff=np.count_nonzero(got.view(np.uint32)!=want.view(np.uint32));total+=want.size;bad+=diff
        if diff:print(name,'differences',diff)
    print('march words',total,'differences',bad)
    if bad:raise SystemExit(1)
