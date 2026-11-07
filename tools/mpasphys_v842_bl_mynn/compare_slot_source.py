"""Compare MYNN SlotCall pool outputs with selected compiled host stages."""
from types import SimpleNamespace
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_bl_mynn.exercise_slot import host_pools
from tools.mpasphys_v842_bl_mynn.compare_driver_finish import host_calls
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.native_powf import CUDA_NATIVE_POWF
from woof.hex.mpasphys.bl_mynn_v842 import ADAPTER_CUDA_SOURCE,MYNN_IEEE_ARITHMETIC_SUPPORT
from woof.hex.mpasphys import metal_bl_mynn_slot_v842 as slot,metal_bl_mynn_driver_v842 as driver
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    source=host_pools();source['constants']['xlv']=np.float32(2490*1004);source['constants']['xlf']=np.float32((2819-2490)*1004)
    nc,nl=source['nCellsSolve'],source['nVertLevels'];n=nc*nl
    a=ADAPTER_CUDA_SOURCE.index('extern "C" __global__ void mynn_pre(');b=ADAPTER_CUDA_SOURCE.index('// MPAS mpas_atmphys_driver_pbl.F:454-485',a)
    host=HostModule('mpas_v842_mynn_slot_adapter','#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+ADAPTER_CUDA_SOURCE[a:b])
    start=ADAPTER_CUDA_SOURCE.index('extern "C" __global__ void mynn_dx(')
    dx=HostModule('mpas_v842_mynn_slot_dx','#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+CUDA_NATIVE_POWF+MYNN_IEEE_ARITHMETIC_SUPPORT+ADAPTER_CUDA_SOURCE[start:]).get_function('mynn_dx')
    calls=host_calls(nc,nl)
    def run(kernel,count,*args):kernel(((count+127)//128,),(128,),(count,*args))
    def pre(mixing,specific,flags,*,exact_inverse):run(host.get_function('mynn_pre_rp1' if exact_inverse else 'mynn_pre'),n,*map(int,flags),*[mixing[k] for k in ('qv','qc','qi','qs')],*[specific[k] for k in ('sqv','sqc','sqi','sqs')])
    def post(mixing,tendencies,flags,delt,*,exact_inverse):run(host.get_function('mynn_post_rp1' if exact_inverse else 'mynn_post'),n,*map(int,flags),np.float32(delt),*[mixing[k] for k in ('qv','qc','qi','qs')],*[tendencies[k] for k in ('dqv','dqc','dqi','dqs')])
    calls.update(pre=pre,post=post,grid_spacing=lambda density,length,result:run(dx,nc,np.float32(length),density,result))
    class Context:
        def submit_bookkeeping(self,pairs):
            for src,dst in pairs:np.copyto(dst,src)
    fake=Context()
    def layout(name,count,ctx,arrays,params):
        assert name=='hexv842_sw_layout_f32';np.copyto(arrays['dst'],arrays['src'].T)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for cold,inverse,absent in ((True,False,False),(False,False,False),(True,True,False),(False,True,False),(False,False,True)):
            expected={k:({a:b.copy() for a,b in v.items()} if k in ('prepared','diag_physics','tend_physics','sfc_input','mesh') else dict(v) if isinstance(v,dict) else v) for k,v in source.items()}
            actual={k:({a:xp.asarray(b.copy()) for a,b in v.items()} if k in ('prepared','diag_physics','tend_physics','sfc_input','mesh') else dict(v) if isinstance(v,dict) else v) for k,v in source.items()}
            if absent:
                for pools in (expected,actual):
                    for name in ('f_qc','f_qi','f_qs','f_nc','f_ni','f_nifa','f_nwfa'):pools['scalar_state'][name]=0
            args=dict(slot='pbl',scheme='bl_mynn',options={},corrections=frozenset({'RP1'} if inverse else ()),itimestep=1 if cold else 2)
            slot.MetalMpasBlMynn(ctx).launch_slot(SimpleNamespace(pools=actual,**args))
            previous={name:getattr(driver.ops,name) for name in calls};old_dc,old_da=driver._context,driver._arrays;old_sc,old_sa,old_req,old_launch=slot._context,slot._arrays,slot._require,slot._launch
            try:
                for name,fn in calls.items():setattr(driver.ops,name,fn)
                driver._context=lambda _:None;driver._arrays=lambda _:np;slot._context=lambda _:fake;slot._arrays=lambda _:np;slot._require=lambda *args:None;slot._launch=layout
                slot.MetalMpasBlMynn(fake).launch_slot(SimpleNamespace(pools=expected,**args))
            finally:
                for name,fn in previous.items():setattr(driver.ops,name,fn)
                driver._context,driver._arrays=old_dc,old_da;slot._context,slot._arrays,slot._require,slot._launch=old_sc,old_sa,old_req,old_launch
            for pool in ('diag_physics','tend_physics','prepared'):
                for name,want in expected[pool].items():
                    got=actual[pool][name].to_numpy();bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                    if bad.size:raise AssertionError(f'cold={cold} {pool}.{name}: {bad.size} differences at {bad[0]} got={got.flat[bad[0]]} want={want.flat[bad[0]]}')
            print(f'MYNN SlotCall pool source-stage comparison cold={cold} RP1={inverse} absent={absent} matches',flush=True)
        before={(pool,k):v.to_numpy().copy() for pool in ('diag_physics','tend_physics','prepared') for k,v in actual[pool].items()}
        for options,corrections,scalar in (({'config_mynn_scaleaware':2},frozenset(),{}),({},frozenset(),{'f_nbca':1}),({},frozenset(),{'f_qoz':1}),({},frozenset(),{'f_nc':1})):
            saved=actual['scalar_state'].copy();actual['scalar_state'].update(scalar)
            try:slot.MetalMpasBlMynn(ctx).launch_slot(SimpleNamespace(pools=actual,slot='pbl',scheme='bl_mynn',options=options,corrections=corrections,itimestep=1))
            except ValueError:pass
            else:raise AssertionError('unsupported MYNN option accepted')
            actual['scalar_state'].update(saved)
            for (pool,name),want in before.items():
                if not np.array_equal(actual[pool][name].to_numpy().view(np.uint32),want.view(np.uint32)):raise AssertionError('refusal published '+pool+'.'+name)
if __name__=='__main__':main()
