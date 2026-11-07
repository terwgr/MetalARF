"""Compare native LW coefficient interpolation with selected source."""
from pathlib import Path
import json,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_lw_v842 import _solver_source,_STATIC_ARRAYS
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import setcoef,_static_words,_SETCOEF_FLOAT_OUTPUTS,_SETCOEF_INT_OUTPUTS
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=_solver_source();m=re.search(r'extern "C" __global__ void rlw_setcoef\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()+'''
#define NBNDLW 16
#define MXMOL 38
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
#define rlw_log(x) gfk_log(x)
'''
host=HostModule('mpas_v842_lw_setcoef',header+re.sub(r'\btz\b','tlevel',source[m.start():j])).get_function('rlw_setcoef')
nc,nl=2,8
inputs={'pavel':np.tile(np.array([900,700,500,300,100,30,5,1],np.float32),(nc,1)),
        'tavel':np.tile(np.array([285,270,255,235,220,215,225,240],np.float32),(nc,1)),
        'tz':np.tile(np.array([290,277,262,245,226,218,220,232,248],np.float32),(nc,1)),
        'tbound_v':np.array([288,268],np.float32),'semiss':np.full((nc,16),.97,np.float32),
        'coldry':np.full((nc,nl),2e23,np.float32),'wkl':np.full((nc,38,nl),2e18,np.float32),
        'wbroad':np.full((nc,nl),1e23,np.float32)}
inputs['pavel'][1,-1]=np.array([1],np.uint32).view(np.float32)[0]
inputs['wkl'][0,1,0]=0;inputs['wkl'][0,2,0]=np.float32(1e-20)
tables={n:_static_words(n,size,integer) for n,size,integer in _STATIC_ARRAYS if n in ('totplnk','totplk16','preflog','tref','chi')}
outputs={n:np.zeros((nc,nl),np.float32) for n in _SETCOEF_FLOAT_OUTPUTS}
outputs.update({n:np.zeros((nc,nl),np.int32) for n in _SETCOEF_INT_OUTPUTS})
outputs.update({'laytrop_v':np.zeros(nc,np.int32),'planklay':np.zeros((nc,nl,16),np.float32),
                'planklev':np.zeros((nc,nl+1,16),np.float32),'plankbnd':np.zeros((nc,16),np.float32)})
fragment=json.loads((ROOT/'woof/metal/signature_fragments/hexv842_mpas_rrtmg_lw_setcoef_f32.json').read_text())['kernels'][0]
resources=dict(inputs,**{n:tables['chi' if n=='chi_mls' else n] for n in ('totplnk','totplk16','preflog','tref','chi_mls')},**outputs)
host((1,),(128,),(*map(np.int32,(nc,nl,1)),*[resources[n] for n,_,_ in fragment['resources']]))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    di={n:xp.asarray(v) for n,v in inputs.items()};dt={n:xp.asarray(v) for n,v in tables.items()}
    dout={n:xp.zeros(v.shape,v.dtype) for n,v in outputs.items()}
    setcoef(di,dt,dout)
    total=bad=0
    for name,expected in outputs.items():
        got=dout[name].to_numpy();mismatch=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32));total+=expected.size;bad+=mismatch
        if mismatch:print(name,'differences',mismatch)
    print('setcoef words',total,'differences',bad)
    if bad:raise SystemExit(1)
