"""Small hydrated Registry fixture for composed native Metal callback checks."""
from contextlib import contextmanager
from pathlib import Path
import hashlib,json,tempfile
import numpy as np
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager,SlotRegistry
from woof.hex.mpasphys.framework_backend_v842 import registry_source
from woof.hex.mpasphys.framework_init_v842 import FrameworkInitialization
from woof.hex.mpasphys.pool_v842 import RegistryLayout
from woof.hex.mpasphys.metal_session_v842 import MetalPhysicsSession


@contextmanager
def session_fixture(context,config,namelist,*,dt=300,nz=6,slot_services=None):
    dims={'nCells':7,'nEdges':9,'nVertices':5,'nVertLevels':nz,'nSoilLevels':4,'maxEdges':6,'maxEdges2':12,'vertexDegree':3,'nCellsSolve':6,'nEdgesSolve':8,'nVerticesSolve':5}
    layout=RegistryLayout(registry_source(),config,namelist,dims)
    layout.dimensions.update(nCellsSolve=6,nEdgesSolve=8,nVerticesSolve=5)
    dry=resolve_namelist("&physics\nconfig_physics_suite='none'\n/\n")
    class Hooks:
        def invoke(self,operation,call):raise AssertionError(operation)
        def restart_state(self):return {}
        def restore_restart_state(self,state):assert not state
    def factory(session):return PhysicsDriver(dry,PhysicsManager(dry,dt),SlotRegistry(),Hooks())
    with tempfile.TemporaryDirectory(prefix='native-registry-') as temp:
        root=Path(temp);records=[]
        for i,row in enumerate(layout.fields):
            r={k:row[k] for k in ('pool','name','kind','dimensions','shape')}
            if row['kind']=='character':r['text']=''
            elif row['pool']=='mesh' and row['name'] in ('indexToCellID','indexToEdgeID','indexToVertexID'):
                a=np.arange(1,np.prod(row['shape'])+1,dtype='<i4').reshape(row['shape'],order='F');entity={'indexToCellID':'nCells','indexToEdgeID':'nEdges','indexToVertexID':'nVertices'}[row['name']];owned=dims[entity+'Solve'];count=dims[entity]
                if count>owned:a[owned:count]=a[:count-owned]
                data=a.tobytes(order='F');path=root/f'{i}.bin';path.write_bytes(data);r.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            elif (row['pool'],row['name']) == ('mesh','nominalMinDc'):
                a=np.full(row['shape'],12000,dtype='<f4',order='F');data=a.tobytes(order='F');path=root/f'{i}.bin';path.write_bytes(data);r.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            else:r['fill']=0
            records.append(r)
        (root/'manifest.json').write_text(json.dumps({'schema':'mpas-native-pool-payload-v1','fields':records}))
        session=MetalPhysicsSession(layout=layout,payload=root,context=context,dt=dt,
            start='2024-06-28_12:00:00',stop='2024-06-28_13:00:00',driver_factory=factory,config=config,slot_services=slot_services)
        framework=FrameworkInitialization(session);framework.physics_run_init(None)
        for species in ('qc','qr','qi','qs','qg','nc','ni','nifa','nwfa'):
            framework.scalar('f_'+species,session.native._dimension_values['state','index_'+species]>-1)
        session.native._initialized=True;session.initialize()
        yield session
        session.close()
