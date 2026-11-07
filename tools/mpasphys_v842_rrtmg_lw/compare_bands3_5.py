"""Compare native LW bands 3–5 with selected source under varied mixtures."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_rrtmg_lw.band_source import band_table_sizes
from woof.hex.mpasphys.rrtmg_lw_v842 import _solver_source
from woof.core.rrtmg_lw import GPU_BAND_TABS
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import band as run_band
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=_solver_source();macros=source[source.index('#define SL_PAVEL 0'):source.index('extern "C" __global__ void rlw_taugb1(')]
helpers=(ROOT/'woof/core/kernels/rrtmg_lw_taugb03_05.cu').read_text();helpers=helpers[helpers.index('#define CHI35'):helpers.index('// ---------------------------------------------------------------------\n// Band 3:')]
header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()+'''
#define NGPTLW 140
#define MAXXSEC 4
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
#define rlw_pow(x,y) gfk_pow(x,y)
__device__ float rlw_pow4(float x){float t=RLW_MU(x,x);return RLW_MU(t,t);}
'''
nc,nl=3,5;fs=np.ones((33,nc,nl),np.float32);isv=np.ones((6,nc,nl),np.int32)
fs[0]=np.array([900,400,150,30,5],np.float32);fs[1]=2e23;fs[2:10]=.3;fs[10:14]=.25;fs[14:26]=1;fs[26:33]=.1
fs[2,0,:]=.01;fs[2,1,:]=.5;fs[2,2,:]=9;fs[5,:,:]=2
fs[2,0,0]=np.float32(1e-40)
isv[0]=np.array([1,11,12,13,58],np.int32);isv[1]=[1,4,1,4,1];isv[2]=[4,1,4,1,4];isv[3]=[1,9,1,9,1];isv[4]=[1,2,1,2,1];isv[5]=[1,18,1,18,1]
laytrop=np.full(nc,3,np.int32);wx=np.full((nc,4,nl),.2,np.float32);chi=np.ones(413,np.float32)
rng=np.random.default_rng(842)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    dl,df,di,dw,dc=map(xp.asarray,(laytrop,fs,isv,wx,chi))
    for b in (3,4,5):
        m=re.search(r'extern "C" __global__ void rlw_taugb'+str(b)+r'\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
        while depth:
            if source[j]=='{':depth+=1
            elif source[j]=='}':depth-=1
            j+=1
        chosen=source[m.start():j];names=GPU_BAND_TABS[b]
        chosen=chosen.replace('const float* const* __restrict__ tabs,',','.join('const float* __restrict__ in_'+n for n in names)+',')
        for ix,n in enumerate(names):chosen=re.sub(r'const float\s*\*\s*'+n+r'\s*=\s*tabs\['+str(ix)+r'\];',f'const float* {n} = in_{n};',chosen)
        host=HostModule('mpas_v842_lw_band'+str(b),header+macros+helpers+chosen).get_function('rlw_taugb'+str(b))
        tabs={name:rng.uniform(.001,.5,size).astype(np.float32) for name,size in band_table_sizes(b).items()}
        taug=np.zeros((nc,nl,140),np.float32);fracs=np.zeros_like(taug)
        host(((nc*nl+127)//128,),(128,),(np.int32(nc),np.int32(nl),laytrop,fs,isv,wx,chi,np.float32(.999999),*[tabs[n] for n in names],taug,fracs))
        da=xp.zeros(taug.shape,np.float32);dr=xp.zeros(fracs.shape,np.float32)
        run_band(b,dl,df,di,dw,dc,{n:xp.asarray(v) for n,v in tabs.items()},da,dr,oneminus=np.float32(.999999))
        differences=0
        for name,got,expected in (('taug',da,taug),('fracs',dr,fracs)):
            actual=got.to_numpy();bad=np.argwhere(actual.view(np.uint32)!=expected.view(np.uint32));differences+=len(bad)
            if len(bad):print('band',b,name,'differences',len(bad),'first',bad[:3].tolist())
        print('band',b,'words',taug.size+fracs.size,'differences',differences)
        if differences:raise SystemExit(1)
