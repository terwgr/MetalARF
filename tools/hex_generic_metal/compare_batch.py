"""Compare the bounded generic wrapper to the existing Metal batch path."""
from datetime import datetime
import numpy as np
from woof.core.mpas_column_batch_metal import MetalMpasColumnBatchPhysics
from woof.core.mpas_column_physics_metal import MetalMpasColumnPhysics
from woof.core.mpas_column_metal_profiles import _PHYSICS
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

class IncidenceBatch(MetalMpasColumnBatchPhysics):
    def _physics_settings(self):
        return dict(super()._physics_settings(), icloud=2, gwd_opt=0)

def main():
    n, nz = 4, 8
    surface = {'landmask':np.ones(n,np.float32), 'xland':np.ones(n,np.float32),
               'ivgtyp':np.full(n,10,np.int32), 'isltyp':np.full(n,6,np.int32),
               'vegfra':np.full(n,50,np.float32), 'tsk':np.full(n,290,np.float32),
               'tmn':np.full(n,285,np.float32), 'xice':np.zeros(n,np.float32),
               'snow':np.zeros(n,np.float32), 'snow_depth':np.zeros(n,np.float32),
               'soil_temperature':np.full((4,n),285,np.float32),
               'soil_moisture':np.full((4,n),.3,np.float32), 'xice_threshold':.5}
    common=dict(n_levels=nz,n_columns=n,dt=1.,radiation_seconds=2.,surface_pbl_seconds=1.,
        start_time=datetime(2024,6,20,12),latitude_deg=np.full(n,35,np.float32),
        longitude_deg=np.full(n,-97,np.float32),terrain_height_m=np.zeros(n,np.float32),
        z_interface_nominal_m=np.arange(nz+1,dtype=np.float32)*250,p_top_pa=20000.,dx_m=12000.,physics_mode='exact')
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def synchronous(*args,**kw):
            result=submit(*args,**kw);result.wait();return result
        context._submit_kernel_batch=synchronous
        xp=MetalArrayModule(context)
        generic=MetalMpasColumnPhysics(physics=_PHYSICS,surface=surface,array_module=xp,**common)
        reference=IncidenceBatch(microphysics_scheme='wsm6',cumulus_scheme=None,array_module=xp,**surface,**common)
        level=(nz,n); iface=(nz+1,n)
        z=np.broadcast_to(np.arange(nz+1,dtype=np.float32)[:,None]*250,iface).copy()
        pi=np.float32(100000)*np.exp(-z/8000).astype(np.float32)
        p=np.sqrt(pi[:-1]*pi[1:]).astype(np.float32)
        inputs={'u':np.full(level,5,np.float32),'v':np.full(level,2,np.float32),
                'theta':np.full(level,290,np.float32),'pressure':p,'pressure_interface':pi,
                'z_interface':z,'w':np.zeros(iface,np.float32),'rho_dry':p/(287*280)}
        species={key:np.full(level,value,np.float32) for key,value in
            dict(qv=.006,qc=.0001,qr=.00002,qi=.00001,qs=.00001,qg=.00001).items()}
        def run(obj):
            a={k:xp.asarray(v) for k,v in inputs.items()}
            s={k:xp.asarray(v) for k,v in species.items()}
            snapshots=[]
            for step in range(2):
                result=obj.run_phase1(dt=1.,**a,**s)
                snapshots.extend(getattr(result,k).to_numpy() for k in ('du','dv','dtheta','dqv','dqc','dqr','dqi','dqs','dqg','h_diabatic'))
                out=obj.run_phase2(theta=a['theta'],pressure=a['pressure'],rho_dry=a['rho_dry'],z_interface=a['z_interface'],**s)
                snapshots.extend(v.to_numpy() for _,v in sorted(out.items()))
                snapshots.extend(v.to_numpy() for _,v in sorted(s.items()))
            state=obj.export_state();snapshots.extend(v for _,v in sorted(state['arrays'].items()))
            return snapshots,state
        left,ls=run(generic);right,rs=run(reference)
        assert len(left)==len(right)
        bad=sum(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)) for a,b in zip(left,right))
        words=sum(a.size for a in left)
        print('generic versus existing Metal batch',bad,'/',words,'words; counts',ls['scalars']['call_counts'])
        assert bad==0 and ls['scalars']==rs['scalars']
        checkpoint=generic.export_state();generic.restore_state(checkpoint)
        assert generic.export_state()['identity']==checkpoint['identity']

if __name__=='__main__': main()
