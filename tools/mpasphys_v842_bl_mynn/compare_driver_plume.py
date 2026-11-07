"""Compare source-ordered MYNN plume/predictor composition with host source."""
import numpy as np
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


def main():
    f,c=fixture();nc,nl=f['dz1'].shape
    pre,surf=host_helpers();pbl=pblh_host();scale=stability_host();init,initnames=initialize_host();cond,condnames,condconst=condensation_host()
    dmp,profiles,interfaces,columns,outcols,dmpconst,dmpflags=dmp_host();turb,turbnames=turbulence_host();predict,predictnames=predict_host()
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
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for cold in (True,False):
            expected={n:v.copy() for n,v in f.items()};actual={n:xp.asarray(v.copy()) for n,v in f.items()}
            opts=dict(constants=c,cold=cold,momentum_opt=False,tke_opt=False,scalar_opt=True,delt=30.)
            gpu_s=driver.advance_plumes(actual,**opts)
            previous={n:getattr(driver.ops,n) for n in calls};old_context,old_arrays=driver._context,driver._arrays
            try:
                for n,fn in calls.items():setattr(driver.ops,n,fn)
                driver._context=lambda x:None;driver._arrays=lambda x:np
                cpu_s=driver.advance_plumes(expected,**opts)
            finally:
                for n,fn in previous.items():setattr(driver.ops,n,fn)
                driver._context,driver._arrays=old_context,old_arrays
            for name in expected:
                got=actual[name].to_numpy();want=expected[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'cold={cold} {name}: {bad.size} differences at {bad[0]} got {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
            for name in cpu_s:
                got=gpu_s[name].to_numpy();want=cpu_s[name];bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'cold={cold} scratch {name}: {bad.size} differences at {bad[0]} got {got.flat[bad[0]]} expected {want.flat[bad[0]]}')
            print(f'MYNN plume/predictor source composition cold={cold} matches',flush=True)
if __name__=='__main__':main()
