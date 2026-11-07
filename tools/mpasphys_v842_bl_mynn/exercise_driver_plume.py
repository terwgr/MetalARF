"""Exercise source-ordered MYNN preparation and plume/predictor on Metal."""
import numpy as np
from tools.mpasphys_v842_bl_mynn.compare_driver_prepare import PRE_IN,PRE_OUT,SURF_PROFILES,SURF_COLUMNS,SURF_OUTPUTS
from woof.hex.mpasphys.metal_bl_mynn_driver_v842 import advance_plumes
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def fixture():
    nc,nl=4,17;shape=(nc,nl);rng=np.random.default_rng(842)
    profiles=set(PRE_IN+PRE_OUT+SURF_PROFILES+('u1','v1','w1','tk1','p1','qnc1','qni1','qnwfa1','qnifa1','qnbca1','qozone1','edmf_w1','edmf_a1'))
    columns=set(SURF_COLUMNS+SURF_OUTPUTS+('xland1','dx1','pblh1','psig_bl','psig_shcu','rmol_in','wspd1','uoce1','voce1'))
    f={n:rng.uniform(.001,.01,shape).astype(np.float32) for n in profiles};f.update({n:rng.uniform(.01,.1,nc).astype(np.float32) for n in columns})
    f['dz1']=np.full(shape,100,np.float32);f['th1']=np.broadcast_to(np.linspace(300,295,nl,dtype=np.float32),shape).copy()
    f['tk1']=f['th1'].copy();f['p1']=np.broadcast_to(np.linspace(100000,70000,nl,dtype=np.float32),shape).copy();f['ex1']=np.full(shape,.96,np.float32);f['rho1']=np.full(shape,1.1,np.float32)
    f['u1']=rng.uniform(1,8,shape).astype(np.float32);f['v1']=rng.uniform(1,8,shape).astype(np.float32);f['w1']=np.zeros(shape,np.float32)
    f['ps1']=np.full(nc,100000,np.float32);f['ts1']=np.full(nc,305,np.float32);f['ust1']=np.full(nc,.4,np.float32);f['dx1']=np.full(nc,3000,np.float32)
    f['hfx1']=np.full(nc,100,np.float32);f['qfx1']=np.full(nc,1.e-4,np.float32);f['xland1']=np.array([1,2,1,2],np.float32);f['zw']=np.zeros((nc,nl+1),np.float32);f['kpbl1']=np.zeros(nc,np.int32)
    c=dict(cp=1004.,cpv=1850.,cice=2106.,cliq=4190.,ep_2=.622,ep_3=.378,r_d=287.,r_v=461.,xls=2.83e6,xlv=2.5e6,t0c=273.15,xlvcp=2490.,xlscp=2819.,ep1=.61,p1000mb=100000.,rcp=287./1004.,rvovrd=461./287.,p608=.61,karman=.4,gravity=9.81)
    return f,c


def main():
    f,c=fixture()
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for cold in (True,False):
            a={n:xp.asarray(v.copy()) for n,v in f.items()}
            s=advance_plumes(a,constants=c,cold=cold,momentum_opt=False,tke_opt=False,scalar_opt=True,delt=30.)
            for n in ('qke1','pblh1','qc_bl1'):
                if not np.all(np.isfinite(a[n].to_numpy())):raise AssertionError(n)
            print(f'MYNN source-ordered plume/predictor exercised cold={cold}',flush=True)
if __name__=='__main__':main()
