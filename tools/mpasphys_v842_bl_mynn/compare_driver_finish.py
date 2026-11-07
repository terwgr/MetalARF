"""Compare source-ordered MYNN through tendencies with selected host stages."""
import numpy as np
import re
from importlib.resources import files
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_MODULE_CUDA as SOURCE
from tools.mpasphys_v842_bl_mynn.compare_tendencies import host_module as tendencies_host
from tools.mpasphys_v842_bl_mynn.exercise_driver_plume import fixture
from tools.mpasphys_v842_bl_mynn.compare_driver_prepare import host_helpers,PRE_IN,PRE_OUT,SURF_PROFILES,SURF_COLUMNS,SURF_OUTPUTS,INIT_MAP,CONSTS
from tools.mpasphys_v842_bl_mynn.compare_pblh import host_module as pblh_host
from tools.mpasphys_v842_bl_mynn.compare_stability import host_module as stability_host
from tools.mpasphys_v842_bl_mynn.compare_initialize import host_module as initialize_host
from tools.mpasphys_v842_bl_mynn.compare_condensation2 import host_module as condensation_host
from tools.mpasphys_v842_bl_mynn.compare_dmp import host_module as dmp_host
from tools.mpasphys_v842_bl_mynn.compare_turbulence25 import host_module as turbulence_host
from tools.mpasphys_v842_bl_mynn.compare_predict_columns import host_module as predict_host
from woof.hex.mpasphys import metal_bl_mynn_driver_v842 as driver
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def finish_hosts():
    m=re.search(r'__device__ __noinline__ void f_bl_mynn_run\s*\(.*?\)\s*\{',SOURCE,re.S);assert m
    s=SOURCE[m.end():];a=s.index('// bl_mynn.F90:973');heat=s[a:s.index('// bl_mynn.F90:981',a)]
    m=re.search(r'__device__ __noinline__ void f_retrieve_exchange_coeffs\s*\(.*?\)\s*\{',SOURCE,re.S);assert m
    s=SOURCE[m.end():];a=s.index('// bl_mynn_subroutines.F90:4331');b=s.index('// MPAS physics_mmm/bl_mynn_subroutines.F90:4344',a);exchange=s[a:b].strip()[:-1]
    helper="""
__device__ float mynn_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
__device__ float mynn_min(float a,float b){return a<b?a:b;}__device__ float mynn_max(float a,float b){return a>b?a:b;}
__device__ float mynn_pow(float a,float b){return mynn_ieee_result(a,b,gfk_pow(a,b));}
__device__ float mynn_exp(float a){return mynn_ieee_result(a,a,gfk_exp(a));}
"""
    src='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+files('woof.core.kernels').joinpath('glibc_flt32.cuh').read_text()+helper
    def kernel(name,names,body,extra):
        macros=''.join(f'#define {n}(i) {n}_ptr[col*nl+(i)-1]\n' for n in names)
        decl=','.join('float* '+n+'_ptr' for n in names)
        return macros+f'extern "C" __global__ void {name}(int nc,int nl,{extra}{decl}){{int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=nc)return;int kts=1,kte=nl,k;'+('float b1=24.f;' if name=='host_heat' else 'float dzk;')+body+'}\n'+''.join(f'#undef {n}\n' for n in names)
    src+=kernel('host_heat',('qke1','el_pbl1','p1','diss_heat'),heat,'float cp,')
    src+=kernel('host_exchange',('dfm','dfh','dz','k_m','k_h'),exchange,'')
    mod=HostModule('mpas_v842_mynn_driver_finish',src)
    return mod.get_function('host_heat'),mod.get_function('host_exchange')

def host_calls(nc,nl):
    pre,surf=host_helpers();pbl=pblh_host();scale=stability_host();init,initnames=initialize_host();cond,condnames,condconst=condensation_host()
    dmp,profiles,interfaces,columns,outcols,dmpconst,dmpflags=dmp_host();turb,turbnames=turbulence_host();predict,predictnames=predict_host()
    tend,tnames,tcolumns,tflags,tconst=tendencies_host();heat,exchange=finish_hosts()
    calls={}
    def launch(name,*args):name((1,),(128,),args)
    calls['driver_pre']=lambda d,**kw:launch(pre,nc,nl,int(kw['cold']),*[d[n] for n in PRE_IN+PRE_OUT+('ust1','zw')])
    calls['interface_prefix']=lambda src,dst:np.copyto(dst,src[:,:nl])
    calls['pbl_height']=lambda *a:launch(pbl,nc,nl,*a)
    calls['stability']=lambda zet,dx,pblh,out:launch(scale,nc,zet,dx,pblh,*[out[n] for n in ('phim','phih','psig_bl','psig_shcu')])
    calls['initialize']=lambda d,**kw:launch(init,nc,nl,int(kw['initialize_qke']),*[d[n] for n in initnames])
    calls['driver_surface']=lambda d,**kw:launch(surf,nc,nl,*[np.float32(kw['constants'][n]) for n in CONSTS],*[d[n] for n in SURF_PROFILES+SURF_COLUMNS+SURF_OUTPUTS])
    calls['condensation2']=lambda d,**kw:launch(cond,nc,nl,*[np.float32(kw['constants'][n]) for n in condconst],*[d[n] for n in condnames])
    def plume(d,*,constants,options):
        launch(dmp,nc,nl,*[np.float32(constants[n]) for n in dmpconst],*[int(options[n]) for n in dmpflags],*[d[n] for n in profiles+list(columns)+list(outcols[:3])+['kpbl','ktop']])
    calls['dmp']=plume
    calls['turbulence25']=lambda d,**kw:launch(turb,nc,nl,0,*[d[n] for n in turbnames])
    def prediction(d,*,closure,delt,karman,edmf_tke,tkebudget):
        launch(predict,nc,nl,np.float32(closure),np.float32(delt),np.float32(karman),int(edmf_tke),*[d[n] for n in ('ust','flt','flq','pmz','phh')],*[d[n] for n in predictnames])
    calls['predict_columns']=prediction
    calls['driver_heat']=lambda d,**kw:launch(heat,nc,nl,np.float32(kw['cp']),*[d[n] for n in ('qke1','el_pbl1','p1','diss_heat')])
    calls['driver_exchange']=lambda d:launch(exchange,nc,nl,*[d[n] for n in ('dfm','dfh','dz','k_m','k_h')])
    calls['tendencies']=lambda d,**kw:launch(tend,nc,nl,*[np.float32(kw['constants'][n]) for n in tconst],*[int(kw['flags'][n]) for n in tflags],*[d[n] for n in tuple(tnames)+tcolumns])
    return calls


def main():
    f,c=fixture();nc,nl=f['dz1'].shape
    calls=host_calls(nc,nl)
    flags=dict(flag_qc=1,flag_qi=1,flag_qnc=1,flag_qni=1,flag_qs=1,flag_qnwfa=1,flag_qnifa=1,flag_qnbca=0,flag_ozone=0,bl_mynn_cloudmix=1,bl_mynn_mixqt=0,bl_mynn_edmf_mom=0,bl_mynn_mixscalars=1)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for cold in (True,False):
            expected={n:v.copy() for n,v in f.items()};actual={n:xp.asarray(v.copy()) for n,v in f.items()}
            opts=dict(constants=c,cold=cold,momentum_opt=False,tke_opt=False,scalar_opt=True,delt=30.)
            gpu_s=driver.advance_plumes(actual,**opts)
            gpu_r=driver.finish_columns(actual,gpu_s,constants=c,flags=flags,delt=30.,dheatopt=True)
            previous={n:getattr(driver.ops,n) for n in calls};old_context,old_arrays=driver._context,driver._arrays
            try:
                for n,fn in calls.items():setattr(driver.ops,n,fn)
                driver._context=lambda x:None;driver._arrays=lambda x:np
                cpu_s=driver.advance_plumes(expected,**opts)
                cpu_r=driver.finish_columns(expected,cpu_s,constants=c,flags=flags,delt=30.,dheatopt=True)
            finally:
                for n,fn in previous.items():setattr(driver.ops,n,fn)
                driver._context,driver._arrays=old_context,old_arrays
            for name in expected:
                got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'cold={cold} {name}: {bad.size} differences at {bad[0]} got {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
            for name in cpu_s:
                got=gpu_s[name].to_numpy();want=cpu_s[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'cold={cold} scratch {name}: {bad.size} differences at {bad[0]} got {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
            for name in ('exch_m','exch_h','diss_heat'):
                got=gpu_r[name].to_numpy();want=cpu_r[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'cold={cold} {name}: {bad.size} differences at {bad[0]}')
            for name in cpu_r['tendencies']:
                got=gpu_r['tendencies'][name].to_numpy();want=cpu_r['tendencies'][name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'cold={cold} tendency {name}: {bad.size} differences at {bad[0]} got={got.flat[bad[0]]} want={want.flat[bad[0]]}')
            print(f'MYNN tendency/exchange source composition cold={cold} matches',flush=True)
if __name__=='__main__':main()
