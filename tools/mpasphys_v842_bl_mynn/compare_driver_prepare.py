"""Compare source-ordered cold/warm MYNN preparation with selected host code."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_bl_mynn.compare_pblh import host_module as pblh_host
from tools.mpasphys_v842_bl_mynn.compare_stability import host_module as stability_host
from tools.mpasphys_v842_bl_mynn.compare_initialize import host_module as initialize_host
from tools.mpasphys_v842_bl_mynn.compare_driver_surface import function
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from woof.hex.mpasphys.metal_bl_mynn_driver_v842 import prepare_columns
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

PRE_IN=('dz1','th1','sqv1','sqc1','sqi1','sqs1','ex1')
PRE_OUT=('qv1','qc1','qi1','qs1','sqw','thl','thetav','qke1','sh1','sm1','el_pbl1','tsq1','qsq1','cov1','cldfra_bl1','qc_bl1','qi_bl1')
SURF_PROFILES=('qv1','rho1','ex1','dz1');SURF_COLUMNS=('ps1','ts1','ust1','hfx1','qfx1')
SURF_OUTPUTS=('flqv','flqc','th_sfc','flq','flt','fltv','rmol1','zet','pmz','phh')
INIT_MAP={'dz':'dz1','u':'u1','v':'v1','thl':'thl','qw':'sqv1','theta':'th1','thetav':'thetav','cldfra_bl1d':'cldfra_bl1','edmf_w1':'edmf_w1','edmf_a1':'edmf_a1','sh':'sh1','sm':'sm1','el':'el_pbl1','qke':'qke1','tsq':'tsq1','qsq':'qsq1','cov':'cov1','zw':'zw','zi':'pblh1','rmo':'rmol_in','ust':'ust1','psig_bl':'psig_bl'}
CONSTS=('cp','p1000mb','rcp','xlvcp','p608','karman','gravity')

def host_helpers():
    m=re.search(r'__device__ __noinline__ void f_bl_mynn_run\s*\(.*?\)\s*\{',SOURCE,re.S);assert m;s=SOURCE[m.end():]
    def region(a,b):
        i=s.index('// bl_mynn.F90:'+str(a));return s[i:s.index('// bl_mynn.F90:'+str(b),i)]
    pre=region(634,664)+region(681,696)+'if(cold){\n'+region(710,722)+region(730,733)+'\n}'
    surf=region(796,823)
    helper='''
struct MynnArena{};struct MynnFrame{MynnFrame(MynnArena&) {}};
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_pow(float a,float b){return mynn_ieee_result(a,b,gfk_pow(a,b));}
__device__ float mynn_log(float a){return mynn_ieee_result(a,a,gfk_log(a));}
__device__ float mynn_atan(float a){return mynn_ieee_result(a,a,glibc_atanf(a));}
template<int N> __device__ float mynn_ipow(float a){float t=mpas_mul(a,a);return N==2?t:mpas_mul(t,a);}
'''
    root=files('woof.core.kernels');src='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+root.joinpath('glibc_flt32.cuh').read_text()+root.joinpath('glibc_trig_flt32.cuh').read_text()+helper+'struct Column{MynnArena &arena;static constexpr float cphm_unst=16.f,cphh_unst=16.f;'+function('f_phim')+function('f_phih')+'};\n'
    names=PRE_IN+PRE_OUT+('ust1','zw')
    macros=''.join(f'#define {n}(i) {n}_ptr[col*nl+(i)-1]\n' for n in PRE_IN+PRE_OUT)+'#define zw(i) zw_ptr[col*(nl+1)+(i)-1]\n'
    decl=','.join('float* '+n+'_ptr' for n in names)
    src+=macros+f'extern "C" __global__ void host_pre(int nc,int nl,int cold,{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;int k,kts=1,kte=nl;float ust1=ust1_ptr[col],xlvcp=2490.f,xlscp=2819.f;\n'+pre+'}\n'+''.join(f'#undef {n}\n' for n in PRE_IN+PRE_OUT+('zw',))
    macros=''.join(f'#define {n}(i) {n}_ptr[col*nl+(i)-1]\n' for n in SURF_PROFILES)
    decl=','.join('float* '+n+'_ptr' for n in SURF_PROFILES+SURF_COLUMNS+SURF_OUTPUTS);par=','.join('float '+n for n in CONSTS)
    src+=macros+f'extern "C" __global__ void host_surf(int nc,int nl,{par},{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;int kts=1;float gtr=mpas_div(gravity,300.f);MynnArena a;Column obj{{a}};'+''.join(f'float {n}={n}_ptr[col];' for n in SURF_COLUMNS)+''.join('float '+n+';' for n in SURF_OUTPUTS)+'float sqcg,cpm,exnerg,phi_m;\n'+surf+'\nphi_m=obj.f_phim(zet);pmz=mpas_sub(phi_m,zet);phh=obj.f_phih(zet);\n'+''.join(f'{n}_ptr[col]={n};' for n in SURF_OUTPUTS)+'}\n'
    mod=HostModule('mpas_v842_mynn_driver_prepare',src)
    return mod.get_function('host_pre'),mod.get_function('host_surf')


def main():
    pre,surf=host_helpers();pbl=pblh_host();scale=stability_host();init,initnames=initialize_host()
    rng=np.random.default_rng(842);nc,nl=7,17;shape=(nc,nl)
    profiles=set(PRE_IN+PRE_OUT+SURF_PROFILES+('u1','v1','edmf_w1','edmf_a1'))
    columns=set(SURF_COLUMNS+SURF_OUTPUTS+('xland1','dx1','pblh1','psig_bl','psig_shcu','rmol_in'))
    f={n:rng.uniform(.001,.01,shape).astype(np.float32) for n in profiles}
    f.update({n:rng.uniform(.01,.1,nc).astype(np.float32) for n in columns})
    f['dz1']=rng.uniform(55,100,shape).astype(np.float32);f['th1']=rng.uniform(289,295,shape).astype(np.float32)
    f['ex1']=rng.uniform(.9,1.,shape).astype(np.float32);f['rho1']=rng.uniform(.8,1.2,shape).astype(np.float32)
    f['u1']=rng.uniform(1,8,shape).astype(np.float32);f['v1']=rng.uniform(1,8,shape).astype(np.float32)
    f['ps1']=rng.uniform(80000,100000,nc).astype(np.float32);f['ts1']=rng.uniform(285,305,nc).astype(np.float32)
    f['ust1']=rng.uniform(.1,.5,nc).astype(np.float32);f['dx1']=rng.uniform(1000,5000,nc).astype(np.float32)
    f['xland1']=np.array([1,2,1,2,1,2,1],np.float32);f['zw']=np.zeros((nc,nl+1),np.float32);f['kpbl1']=np.zeros(nc,np.int32)
    constants=dict(cp=1004.,p1000mb=100000.,rcp=287./1004.,xlvcp=2490.,xlscp=2819.,p608=.61,karman=.4,gravity=9.81,ep1=.61)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for cold in (True,False):
            expected={n:v.copy() for n,v in f.items()};actual={n:xp.asarray(v.copy()) for n,v in f.items()}
            expected['edmf_w1'].fill(0);expected['edmf_a1'].fill(0)
            pre((1,),(128,),(nc,nl,int(cold),*[expected[n] for n in PRE_IN+PRE_OUT+('ust1','zw')]))
            def height_scale():
                zw=np.ascontiguousarray(expected['zw'][:,:nl]);pbl((1,),(128,),(nc,nl,expected['thetav'],expected['qke1'],zw,expected['dz1'],expected['xland1'],expected['pblh1'],expected['kpbl1']))
                z=np.zeros(nc,np.float32);phim=np.empty(nc,np.float32);phih=np.empty(nc,np.float32)
                scale((1,),(128,),(nc,z,expected['dx1'],expected['pblh1'],phim,phih,expected['psig_bl'],expected['psig_shcu']))
            if cold:
                height_scale();init((1,),(128,),(nc,nl,1,*[expected[INIT_MAP[n]] for n in initnames]))
            height_scale()
            surf((1,),(128,),(nc,nl,*[np.float32(constants[n]) for n in CONSTS],*[expected[n] for n in SURF_PROFILES+SURF_COLUMNS+SURF_OUTPUTS]))
            prepare_columns(actual,constants=constants,cold=cold)
            for n in expected:
                got=actual[n].to_numpy();want=expected[n];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'cold={cold} {n}: {bad.size} differences first={bad[0]} got={got.flat[bad[0]]} want={want.flat[bad[0]]}')
            print(f'MYNN source-ordered preparation cold={cold} matches selected host stages',flush=True)
if __name__=='__main__':main()
