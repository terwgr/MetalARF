"""Compare native MYNN per-column driver surface controls to host source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_v842 import driver_surface
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

def function(name):
    m=re.search(r'__device__ __noinline__ float '+name+r'\s*\(',SOURCE);assert m
    i=SOURCE.index('{',m.end());j=i+1;depth=1
    while depth:
        if SOURCE[j]=='{':depth+=1
        elif SOURCE[j]=='}':depth-=1
        j+=1
    return SOURCE[m.start():j]

def main():
    m=re.search(r'__device__ __noinline__ void f_bl_mynn_run\s*\(.*?\)\s*\{',SOURCE,re.S);assert m;s=SOURCE[m.end():];a=s.index('// bl_mynn.F90:796');body=s[a:s.index('// bl_mynn.F90:823',a)]
    helper='''
struct MynnArena{};struct MynnFrame{MynnFrame(MynnArena&) {}};
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_pow(float a,float b){return mynn_ieee_result(a,b,gfk_pow(a,b));}
__device__ float mynn_log(float a){return mynn_ieee_result(a,a,gfk_log(a));}
__device__ float mynn_atan(float a){return mynn_ieee_result(a,a,glibc_atanf(a));}
template<int N> __device__ float mynn_ipow(float a){float t=mpas_mul(a,a);return N==2?t:mpas_mul(t,a);}
'''
    profiles=('qv1','rho1','ex1','dz1');columns=('ps1','ts1','ust1','hfx1','qfx1');outputs=('flqv','flqc','th_sfc','flq','flt','fltv','rmol1','zet','pmz','phh');consts=('cp','p1000mb','rcp','xlvcp','p608','karman','gravity')
    root=files('woof.core.kernels');src='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+root.joinpath('glibc_flt32.cuh').read_text()+root.joinpath('glibc_trig_flt32.cuh').read_text()+helper+'struct Column{MynnArena &arena;static constexpr float cphm_unst=16.f,cphh_unst=16.f;'+function('f_phim')+function('f_phih')+'};\n'
    macros=''.join(f'#define {n}(i) {n}_ptr[col*nl+(i)-1]\n' for n in profiles)
    decl=','.join('float* '+n+'_ptr' for n in profiles+columns+outputs);par=','.join('float '+n for n in consts)
    src+=macros+f'extern "C" __global__ void host_surf(int nc,int nl,{par},{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;int kts=1;float gtr=mpas_div(gravity,300.f);MynnArena a;Column obj{{a}};'+''.join(f'float {n}={n}_ptr[col];' for n in columns)+''.join('float '+n+';' for n in outputs)+'float sqcg,cpm,exnerg,phi_m;\n'+body+'\nphi_m=obj.f_phim(zet);pmz=mpas_sub(phi_m,zet);phh=obj.f_phih(zet);\n'+''.join(f'{n}_ptr[col]={n};' for n in outputs)+'}\n'
    host=HostModule('mpas_v842_mynn_driver_surface',src).get_function('host_surf');rng=np.random.default_rng(842);nc=257;nl=5
    f={n:rng.uniform(.01,1,(nc,nl)).astype(np.float32) for n in profiles};f['rho1']=rng.uniform(.6,1.2,(nc,nl)).astype(np.float32);f['ex1']=rng.uniform(.8,1,(nc,nl)).astype(np.float32);f['dz1']=rng.uniform(20,200,(nc,nl)).astype(np.float32)
    f['ps1']=rng.uniform(80000,105000,nc).astype(np.float32);f['ts1']=rng.uniform(250,310,nc).astype(np.float32);f['ust1']=rng.uniform(.05,.8,nc).astype(np.float32);f['hfx1']=rng.uniform(-200,200,nc).astype(np.float32);f['qfx1']=rng.uniform(-.001,.001,nc).astype(np.float32)
    for n in outputs:f[n]=np.empty(nc,np.float32)
    constants=dict(cp=1004.,p1000mb=100000.,rcp=287./1004.,xlvcp=2490.,p608=.61,karman=.4,gravity=9.81);names=profiles+columns+outputs
    expected={n:v.copy() for n,v in f.items()};host(((nc+127)//128,),(128,),(nc,nl,*[np.float32(constants[n]) for n in consts],*[expected[n] for n in names]))
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);actual={n:xp.asarray(v.copy()) for n,v in f.items()}
        driver_surface(actual,constants=constants)
        for n in names:
            got=actual[n].to_numpy();want=expected[n];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
            if bad.size:raise AssertionError(f'{n}: {bad.size} differences at {bad[0]} got {got.flat[bad[0]]} want {want.flat[bad[0]]}')
    print(f'MYNN driver surface controls match selected source for {nc} columns',flush=True)
if __name__=='__main__':main()
