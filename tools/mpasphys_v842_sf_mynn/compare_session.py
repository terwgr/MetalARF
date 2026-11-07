"""Run sf_mynn on real hydrated MPAS Registry pools and restore their state."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from woof.hex.mpasphys.driver_v842 import resolve_namelist
from woof.hex.mpasphys.framework_backend_v842 import registry_source
from woof.hex.mpasphys.pool_v842 import RegistryLayout
from woof.hex.mpasphys.metal_session_v842 import MetalPhysicsSession
from woof.hex.mpasphys.metal_sf_mynn_v842 import MetalMpasSfMynn
from woof.hex.mpasphys.sf_mynn_v842 import FIELDS,CONSTANTS,OUTPUTS,_DIAG_INPUT,_DIAG_NAMES,_PROFILE,_SEA_HANDOFF
from tools.mpasphys_v842_sf_mynn.compare_columns import fixture
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

class Driver:
    def initialize(self,session): pass
    def restart_state(self): return {}
    def restore_restart_state(self,state):
        if state!={}: raise ValueError('unexpected driver state')

def main():
    s="&physics\nconfig_physics_suite='convection_permitting'\n/\n"
    d={'nCells':7,'nEdges':9,'nVertices':5,'nVertLevels':4,'nSoilLevels':4,'maxEdges':6,'maxEdges2':12,'vertexDegree':3,'nCellsSolve':6,'nEdgesSolve':8,'nVerticesSolve':5}
    layout=RegistryLayout(registry_source(),resolve_namelist(s),s,d)
    layout.dimensions.update(nCellsSolve=6,nEdgesSolve=8,nVerticesSolve=5)
    n=6; initial,c=fixture(n); get=lambda name:initial[FIELDS.index(name)].copy()
    with tempfile.TemporaryDirectory(prefix='mpas-sf-mynn-session-') as temp:
        root=Path(temp);records=[]
        for i,row in enumerate(layout.fields):
            r={k:row[k] for k in ('pool','name','kind','dimensions','shape')}
            if row['kind']=='character':r['text']=''
            elif row['pool']=='mesh' and row['name'] in ('indexToCellID','indexToEdgeID','indexToVertexID'):
                a=np.arange(1,np.prod(row['shape'])+1,dtype='<i4').reshape(row['shape'],order='F')
                entity={'indexToCellID':'nCells','indexToEdgeID':'nEdges','indexToVertexID':'nVertices'}[row['name']]
                owned=layout.dimensions[entity+'Solve'];count=layout.dimensions[entity]
                if count>owned:a[owned:count]=a[:count-owned]
                path=root/f'{i}.bin';data=a.tobytes(order='F');path.write_bytes(data)
                r.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            elif (row['pool'],row['name'])==('mesh','nominalMinDc'):
                a=np.full(row['shape'],3000,dtype='<f4',order='F')
                path=root/f'{i}.bin';data=a.tobytes(order='F');path.write_bytes(data)
                r.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            else:r['fill']=0
            records.append(r)
        (root/'manifest.json').write_text(json.dumps({'schema':'mpas-native-pool-payload-v1','fields':records}))
        with MetalContext(batch_kernel_calls=False) as context:
            original=context._submit_kernel_batch
            def sync(*a,**k):
                result=original(*a,**k);result.wait();return result
            context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
            session=MetalPhysicsSession(layout=layout,payload=root,context=context,dt=2,
                start='2024-02-28_00:00:00',stop='2024-02-28_00:01:00',driver_factory=lambda session:Driver())
            session.initialize();views=session.pools
            names=set(_DIAG_INPUT)|{'hpbl'}|{_DIAG_NAMES.get(name,name) for name in OUTPUTS if name not in ('wstar','qstar')}
            diag={name:views.field('diag_physics',name) for name in names}
            for name in _DIAG_INPUT:diag[name][...]=get(name)
            diag['hpbl'][...]=get('pblh')
            surface={name:views.field('sfc_input',name) for name in ('skintemp','xland','snowh','sst','xice')}
            for name,values in {'skintemp':get('tsk'),'xland':get('xland'),'snowh':get('snowh'),
                'sst':np.linspace(265,300,n,dtype=np.float32),'xice':np.array([0,.05,.15,.5,.8,1],np.float32)}.items():surface[name][...]=values
            density=views.field('mesh','meshDensity');density[...] = np.linspace(.7,1.4,n,dtype=np.float32)
            p0={name:get(field) for name,field in zip(_PROFILE,('u1d','v1d','t1d','qv1d','qv1d','p1d','rho1d','dz8w1d'))}
            profiles={name:xp.asarray(np.stack((value,value+np.float32(.01)))) for name,value in p0.items()}
            prepared={'psfc_p':xp.asarray(get('psfcpa')),**{name+'_sea':xp.zeros((n,),np.float32) for name in _SEA_HANDOFF}}
            pools={'mesh':{'meshDensity':density},'diag_physics':diag,'sfc_input':surface,'atm_input':profiles,
                'prepared':prepared,'constants':dict(zip(CONSTANTS,c)),'configs':{'config_len_disp':3000.,'config_do_restart':False},
                'scalar_state':{'xice_threshold':.1}}
            def run(step):
                op=MetalMpasSfMynn(context)
                op.launch_slot(SimpleNamespace(slot='surface_layer',scheme='sf_mynn',pools=pools,itimestep=step,
                    options={'config_frac_seaice':True},corrections=frozenset()))
            run(1); checkpoint=session.transaction.checkpoint();run(2)
            continuous={k:v.to_numpy() for k,v in diag.items()}
            sea={k:prepared[k+'_sea'].to_numpy() for k in _SEA_HANDOFF}
            session.transaction.restore(checkpoint);run(2)
            errors=[]
            for key,ref in continuous.items():
                if not np.array_equal(diag[key].to_numpy().view(np.uint32),ref.view(np.uint32)):errors.append(key)
            for key,ref in sea.items():
                if not np.array_equal(prepared[key+'_sea'].to_numpy().view(np.uint32),ref.view(np.uint32)):errors.append(key+'_sea')
            if errors:raise AssertionError('session continuation differs: '+str(errors))
            assert np.any(continuous['hfx']) and np.any(continuous['ust'])
            print('real Registry pool checkpoint continuation: fields',len(continuous),'sea',len(sea),'words',sum(v.size for v in continuous.values())+sum(v.size for v in sea.values()),'different',len(errors))
            session.close()
if __name__=='__main__':main()
