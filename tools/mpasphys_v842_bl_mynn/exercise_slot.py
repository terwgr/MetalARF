"""Exercise the bounded MYNN SlotCall against native-shaped Metal pools."""
from types import SimpleNamespace
import numpy as np
from tools.mpasphys_v842_bl_mynn.exercise_driver_plume import fixture
from woof.hex.mpasphys.metal_bl_mynn_slot_v842 import MetalMpasBlMynn
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def host_pools():
    f,c=fixture();nc,nl=f['dz1'].shape
    profile=lambda n:f[n].T.copy()
    prep={name:profile(src) for name,src in (('dz_p','dz1'),('u_p','u1'),('v_p','v1'),('w_p','w1'),('th_p','th1'),('t_p','tk1'),('pres_hyd_p','p1'),('pi_p','ex1'),('rho_p','rho1'),('qv_p','sqv1'),('qc_p','sqc1'),('qi_p','sqi1'),('qs_p','sqs1'),('nc_p','qnc1'),('ni_p','qni1'),('nifa_p','qnifa1'),('nwfa_p','qnwfa1'))};prep['psfc_p']=f['ps1'].copy()
    prep['w_p']=np.concatenate((prep['w_p'],np.full((1,nc),3.5,np.float32)),axis=0)
    for name in ('rublten_p','rvblten_p','rthblten_p','rqvblten_p','rqcblten_p','rqiblten_p','kzh_p','kzm_p'):prep[name]=np.zeros((nl,nc),np.float32)
    diag={name:profile(src) for name,src in (('el_pbl','el_pbl1'),('qke','qke1'),('cov','cov1'),('qsq','qsq1'),('tsq','tsq1'),('sh3d','sh1'),('sm3d','sm1'),('qc_bl','qc_bl1'),('qi_bl','qi_bl1'),('cldfrac_bl','cldfra_bl1'))}
    for name in ('kzh','kzm','kzq','dqke','qbuoy','qdiss','qshear','qwt','edmf_a','edmf_w','edmf_qt','edmf_thl','edmf_ent','edmf_qc','sub_thl','sub_qv','det_thl','det_qv'):diag[name]=np.full((nl,nc),.3,np.float32)
    for name,src in (('ust','ust1'),('hfx','hfx1'),('qfx','qfx1'),('rmol','rmol_in'),('wspd','wspd1')):diag[name]=f[src].copy()
    diag['hpbl']=np.zeros(nc,np.float32);diag['kpbl']=np.zeros(nc,np.int32)
    tend={n:np.zeros((nl,nc),np.float32) for n in ('rublten','rvblten','rthblten','rqvblten','rqcblten','rqiblten','rqsblten','rniblten','rncblten','rnifablten','rnwfablten')}
    scalar={n:1 for n in ('f_qc','f_qi','f_qs','f_nc','f_ni','f_nifa','f_nwfa')};scalar.update(f_nbca=0,f_qoz=0,dt_pbl=np.float32(30))
    raw={n:np.float32(c[n]) for n in ('cp','cpv','cice','cliq','ep_2','gravity','karman','r_d','r_v','xls','xlv')};raw.update(ep_1=np.float32(c['ep1']),p0=np.float32(c['p1000mb']),svpt0=np.float32(c['t0c']),xlf=np.float32(3.3e5))
    return {'nCellsSolve':nc,'nVertLevels':nl,'prepared':prep,'diag_physics':diag,'tend_physics':tend,'sfc_input':{'xland':f['xland1'].copy(),'skintemp':f['ts1'].copy()},'mesh':{'meshDensity':np.full(nc,1,np.float32)},'configs':{'config_len_disp':np.float32(3000),'config_do_restart':False,'config_do_DAcycling':False},'scalar_state':scalar,'constants':raw}


def main():
    host=host_pools()
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        p={k:({a:xp.asarray(b.copy()) for a,b in v.items()} if k in ('prepared','diag_physics','tend_physics','sfc_input','mesh') else v) for k,v in host.items()}
        call=SimpleNamespace(slot='pbl',scheme='bl_mynn',pools=p,options={},corrections=frozenset(),itimestep=1)
        MetalMpasBlMynn(ctx).launch_slot(call)
        for name in ('rqvblten','rthblten'):
            if not np.all(np.isfinite(p['tend_physics'][name].to_numpy())):raise AssertionError(name)
        print('MYNN bounded SlotCall completed using native-shaped pools',flush=True)
if __name__=='__main__':main()
