"""Exercise native SW slot layouts, output publication and failure boundary."""
from types import SimpleNamespace
import numpy as np
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_tables, radiation_controls, run_shortwave
from woof.hex.mpasphys.metal_rrtmg_sw_slot_v842 import MetalMpasShortwave
from woof.hex.mpasphys.rrtmg_sw_v842 import SCALAR_OUTPUTS
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    nc,nz,nozn=5,6,2;rng=np.random.default_rng(842)
    fields=np.empty((12,nc,nz),np.float32);fields[0]=np.linspace(95000,20000,nz);fields[1]=np.linspace(280,230,nz);fields[2]=150;fields[3]=np.linspace(1,.7,nz)
    fields[4]=.003;fields[5]=.0001;fields[6]=.00003;fields[7]=.00002;fields[8]=rng.uniform(.1,.7,(nc,nz));fields[9]=.000012;fields[10]=.00003;fields[11]=.00005
    surf=np.zeros((7,nc),np.float32);surf[1]=[-180,-60,0,60,180];surf[2]=.2;surf[3]=280;surf[4]=[1,2,1,2,1];surf[6]=1
    pw=np.broadcast_to(np.linspace(100000,15000,nz+1,dtype=np.float32),(nc,nz+1)).copy();tw=np.broadcast_to(np.linspace(282,220,nz+1,dtype=np.float32),(nc,nz+1)).copy()
    pin=np.array([1,1100],np.float32);oz=np.full((nc,nozn),6e-6,np.float32);degrad=np.float32(np.pi/180)
    opts={'config_microp_re':True,'config_microp_scheme':'mp_thompson','config_radt_cld_overlap':'maximum_random','config_radt_cld_dcorrlen':'constant','config_o3climatology':True}
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);tables=initialize_tables(ctx);op=MetalMpasShortwave(ctx,tables)
        registry=MetalSlotRegistry();registry.register_metal('radiation_sw','rrtmg_sw',op.launch_slot)
        aliases=('pres_hyd_p','t_p','dz_p','pi_p','qvrad_p','qcrad_p','qirad_p','qsrad_p')
        prep={name:xp.asarray(fields[j].T.copy()) for j,name in enumerate(aliases)}
        prep['pres2_hyd_p']=xp.asarray(pw.T.copy());prep['t2_p']=xp.asarray(tw.T.copy())
        prep['qv_p']=xp.asarray(fields[4].T.copy());prep['nwfa_p']=xp.asarray(np.full((nz,nc),1e8,np.float32));prep['nifa_p']=xp.asarray(np.full((nz,nc),1e5,np.float32))
        diag={'cldfrac':xp.asarray(fields[8].T.copy()),'o3clim':xp.asarray(oz.T.copy()),'sfc_albedo':xp.asarray(surf[2])}
        diag.update({name:xp.asarray(fields[j].T.copy()) for j,name in enumerate(('re_cloud','re_ice','re_snow'),9)})
        diag.update({name:xp.zeros((nc,),np.float32) for name in SCALAR_OUTPUTS});diag['taod5502d']=xp.zeros((nc,),np.float32);diag['taod5503d']=xp.zeros((nz,nc),np.float32)
        surface={name:xp.asarray(surf[j]) for j,name in enumerate(('skintemp','xland','xice','snow'),3)}
        mesh={'latCell':xp.asarray(surf[0]*degrad),'lonCell':xp.asarray(surf[1]*degrad)}
        pools=dict(prepared=prep,diag_physics=diag,sfc_input=surface,mesh=mesh,atm_input={'pin':xp.asarray(pin)},nCellsSolve=nc,nVertLevels=nz,constants={'degrad':degrad,'dpd':np.float32(360/365)},scalar_state={'dt_radtsw':900,'has_reqc':1,'has_reqi':1,'has_reqs':1},calendar={'curr_julday':180,'julday':180,'gmt':12},tend_physics={'rthratensw':xp.zeros((nz,nc),np.float32)})
        call=SimpleNamespace(slot='radiation_sw',scheme='rrtmg_sw',pools=pools,options=opts,time_seconds=0,backend='metal')
        for aerosol in (False,True):
            opts['config_microp_scheme']='mp_thompson_aerosols' if aerosol else 'mp_thompson'
            before=op.restart_state();out=registry.run(call)
            for j,name in enumerate(SCALAR_OUTPUTS):
                if not np.array_equal(diag[name].to_numpy().view(np.uint32),out['scalars'][j].to_numpy().view(np.uint32)):raise AssertionError(name)
            if not np.array_equal(pools['tend_physics']['rthratensw'].to_numpy().view(np.uint32),out['tendency'].to_numpy().T.copy().view(np.uint32)):raise AssertionError('tendency layout')
            if aerosol:
                if not np.array_equal(diag['taod5503d'].to_numpy().view(np.uint32),out['taod5503d'].to_numpy().T.copy().view(np.uint32)):raise AssertionError('aerosol layout')
            saved={name:value.to_numpy().copy() for name,value in diag.items()};saved_t=pools['tend_physics']['rthratensw'].to_numpy().copy()
            op.restore_restart_state(before);registry.run(call)
            for name,value in saved.items():
                if not np.array_equal(diag[name].to_numpy().view(np.uint32),value.view(np.uint32)):raise AssertionError('replay '+name)
            if not np.array_equal(saved_t.view(np.uint32),pools['tend_physics']['rthratensw'].to_numpy().view(np.uint32)):raise AssertionError('replay tendency')
        # Reverse one sunlit pressure pair: source STOP must precede publication.
        bad=prep['pres_hyd_p'].to_numpy();bad[0,2]=20000;bad[1,2]=95000;prep['pres_hyd_p'][...]=xp.asarray(bad)
        previous={name:value.to_numpy().copy() for name,value in diag.items()};previous_t=pools['tend_physics']['rthratensw'].to_numpy().copy()
        try:registry.run(call)
        except ValueError as ex:
            if 'KISS requires bottom-to-top' not in str(ex):raise
        else:raise AssertionError('native pressure STOP not raised')
        for name,value in previous.items():
            if not np.array_equal(diag[name].to_numpy().view(np.uint32),value.view(np.uint32)):raise AssertionError('STOP published '+name)
        if not np.array_equal(previous_t.view(np.uint32),pools['tend_physics']['rthratensw'].to_numpy().view(np.uint32)):raise AssertionError('STOP published tendency')
    print('native SW registry dispatch, slot pools, aerosol, replay and pressure STOP pass',flush=True)

if __name__=='__main__':main()
