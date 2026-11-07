"""Exercise Noah on actual Registry views and a Metal transaction boundary."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import argparse,hashlib,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager,SlotRegistry,Backend
from woof.hex.mpasphys.framework_backend_v842 import registry_source
from woof.hex.mpasphys.framework_init_v842 import FrameworkInitialization
from woof.hex.mpasphys.pool_v842 import RegistryLayout
from woof.hex.mpasphys.metal_session_v842 import MetalPhysicsSession
from woof.hex.mpasphys.noah_v842 import _MPAS_IO_CUDA,_MPAS_INPUT_FIELDS,_MPAS_OUTPUT_FIELDS,_NATIVE_POOL_FIELDS,_NATIVE_PREPARED_FIELDS
from woof.hex.mpasphys.metal_noah_v842 import MetalMpasNoah
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
import re
p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);args=p.parse_args()
base=np.fromfile(args.fixture/'input.f32',np.float32);cat=np.fromfile(args.fixture/'input.i32',np.int32);tables=np.fromfile(args.fixture/'tables.bin',np.uint32)
s="&physics\nconfig_physics_suite='convection_permitting'\n/\n"
dims={'nCells':7,'nEdges':9,'nVertices':5,'nVertLevels':4,'nSoilLevels':4,'maxEdges':6,'maxEdges2':12,'vertexDegree':3,'nCellsSolve':6,'nEdgesSolve':8,'nVerticesSolve':5}
layout=RegistryLayout(registry_source(),resolve_namelist(s),s,dims)
layout.dimensions.update(nCellsSolve=6,nEdgesSolve=8,nVerticesSolve=5)
dry=resolve_namelist("&physics\nconfig_physics_suite='none'\n/\n")
class Hooks:
    def invoke(self,operation,call):
        if operation!='physics_get_tend':raise AssertionError(operation)
        return None
    def restart_state(self):return {}
    def restore_restart_state(self,state):pass
def factory(session):return PhysicsDriver(dry,PhysicsManager(dry,2),SlotRegistry(),Hooks())
with tempfile.TemporaryDirectory(prefix='noah-registry-') as temp:
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
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*a,**k):
            result=submit(*a,**k);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        state=xp.asarray(np.array([77,88,99,111],np.int32));table=xp.asarray(tables)
        slot=MetalMpasNoah(table,state)
        session=MetalPhysicsSession(layout=layout,payload=root,context=context,dt=2,start='2024-02-28_00:00:00',stop='2024-02-28_00:01:00',driver_factory=factory,config=resolve_namelist(s),slot_services={('land_surface','sf_noah'):slot})
        FrameworkInitialization(session).physics_run_init(None);session.native._initialized=True;session.initialize()
        fields={}
        for name,kind in _MPAS_INPUT_FIELDS.items():
            if name in _NATIVE_PREPARED_FIELDS:fields[name]=session.pools.prepared(_NATIVE_PREPARED_FIELDS[name])
            else:
                pool,field=_NATIVE_POOL_FIELDS[name];fields[name]=session.pools.field(pool,field)
        for name,value in fields.items():value.fill(0)
        body=_MPAS_IO_CUDA.split('void noah_from_mpas(',1)[1].split('\n}',1)[0]
        for idx,name,sub in re.findall(r'x\[(\d+)\]=(\w+)\[(.*?)\];',body):
            if name not in fields:continue
            value=fields[name];word=base[int(idx)]
            if value.ndim==2:value[int(sub.rsplit('+',1)[1])].fill(word)
            else:value.fill(word)
        fields['ivgtyp'].fill(int(cat[0]));fields['isltyp'].fill(int(cat[1]))
        fields['gsw'].fill(np.float32(base[14]*np.float32(1-base[21])));fields['rainncv'].fill(base[32])
        registry=MetalSlotRegistry();registry.register_metal('land_surface','sf_noah',slot.launch_slot)
        from woof.hex.mpasphys.noah_v842 import bind_native_pools
        session.native.put_text_field('sfc_input','mminlu','MODIFIED_IGBP_MODIS_NOAH')
        session.native.landuse_state={'isurban':int(cat[2])}
        session.slot_pools.noah_v842_resources={'tables':table,'landuse':'MODIFIED_IGBP_MODIS_NOAH'}
        session.native.put_variable('dt_pbl',np.array(base[33],np.float32))
        session.native.put_variable('xice_threshold',np.array(base[72],np.float32))
        session.native.put_field('sfc_input','isice',np.array(cat[3],np.int32))
        call=SimpleNamespace(slot='land_surface',scheme='sf_noah',backend='metal',pools=None,itimestep=int(cat[4]),options={'config_sfc_albedo':int(cat[5]),'config_microp_scheme':'on','config_convection_scheme':'off'})
        opts=bind_native_pools(session.slot_pools,call);call.pools=opts
        before=session.transaction.checkpoint();registry.run(call);after=session.transaction.checkpoint()
        session.transaction.restore(before);registry.run(call);replayed=session.transaction.checkpoint()
        for name in _MPAS_OUTPUT_FIELDS:
            pool,field=_NATIVE_POOL_FIELDS[name];key=f'{pool}.{field}.1'
            assert np.array_equal(after['pools'][key].view(np.uint32),replayed['pools'][key].view(np.uint32)),key
        assert np.array_equal(after['services']['land_surface/sf_noah']['module_locations'],replayed['services']['land_surface/sf_noah']['module_locations'])
        print('Registry fields',len(_MPAS_OUTPUT_FIELDS),'location state',replayed['services']['land_surface/sf_noah']['module_locations'].tolist(),'checkpoint replay passed; unrelated slots isolated')
        # Real driver scheduling with other framework operations isolated.
        from woof.hex.mpasphys.metal_transaction_v842 import MetalPhysicsTransaction
        active=resolve_namelist("&physics\nconfig_physics_suite='none'\nconfig_lsm_scheme='sf_noah'\nconfig_sfclayer_scheme='sf_mynn'\n/\n")
        active=replace(active,backends={**active.backends,'land_surface':'metal','surface_layer':'native-cpu'})
        active_registry=MetalSlotRegistry();session.register_slots(active_registry)
        active_registry.register('surface_layer','sf_mynn',Backend.NATIVE_CPU,lambda actual:None)
        active_registry.register('seaice','sf_noah',Backend.NATIVE_CPU,lambda actual:None)
        class IsolatedHooks:
            def invoke(self,operation,actual):return None
            def restart_state(self):return {}
            def restore_restart_state(self,value):pass
        def active_driver():return PhysicsDriver(active,PhysicsManager(active,2),active_registry,IsolatedHooks())
        actual=active_driver();actual.initialize(session)
        tx=MetalPhysicsTransaction(actual,session.pools,active_driver,stateful_services=session.transaction.stateful_services)
        initial=tx.checkpoint()
        def advance():
            tx.begin_step(itimestep=1,time_seconds=0,driver_pools=session)
            calls=[(c['slot'],c['backend']) for c in tx.driver.calls if c['slot']=='land_surface']
            assert calls==[('land_surface','metal')],calls
            tx.sources(driver_pools=session);tx.finish_step(driver_pools=session);tx.commit()
            return tx.checkpoint()
        one=advance();tx.restore(initial);two=advance()
        for name in _MPAS_OUTPUT_FIELDS:
            pool,field=_NATIVE_POOL_FIELDS[name];key=f'{pool}.{field}.1'
            assert np.array_equal(one['pools'][key].view(np.uint32),two['pools'][key].view(np.uint32)),key
        assert np.array_equal(one['services']['land_surface/sf_noah']['module_locations'],two['services']['land_surface/sf_noah']['module_locations'])
        print('actual PhysicsDriver dispatched Noah and reproduced its Registry/state checkpoint; unrelated hooks isolated')
        session.close()
