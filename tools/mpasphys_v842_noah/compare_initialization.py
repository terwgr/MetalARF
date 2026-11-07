"""Exercise Noah on actual Registry views and a Metal transaction boundary."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager,SlotRegistry
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
p=argparse.ArgumentParser();p.add_argument('resources',type=Path);p.add_argument('--restart',action='store_true');p.add_argument('--landuse');p.add_argument('--startup',action='store_true');p.add_argument('--resource-startup',action='store_true');args=p.parse_args()
s="&physics\nconfig_physics_suite='convection_permitting'\nconfig_o3climatology=.false.\n/\n"
if args.resource_startup:
    args.startup=True
    s=s.replace("config_physics_suite='convection_permitting'", "config_physics_suite='none'\nconfig_lsm_scheme='sf_noah'\nconfig_sfclayer_scheme='sf_mynn'")
if args.restart:s+="&restart\nconfig_do_restart=.true.\n/\n"
dims={'nCells':7,'nEdges':9,'nVertices':5,'nVertLevels':4,'nSoilLevels':4,'maxEdges':6,'maxEdges2':12,'vertexDegree':3,'nCellsSolve':6,'nEdgesSolve':8,'nVerticesSolve':5}
layout=RegistryLayout(registry_source(),resolve_namelist(s),s,dims)
layout.dimensions.update(nCellsSolve=6,nEdgesSolve=8,nVerticesSolve=5)
assert layout.configs['config_do_restart']==args.restart
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
        from woof.hex.mpasphys.metal_noah_v842 import load_native_resources
        packet=json.loads(args.resources.read_text())
        if args.landuse:packet['landuse']=args.landuse;packet.pop('isurban',None)
        resources=load_native_resources(packet,context)
        slot_services={('land_surface','sf_noah'):MetalMpasNoah(resources['tables'],xp.zeros((4,),np.int32))} if args.resource_startup else None
        session=MetalPhysicsSession(layout=layout,payload=root,context=context,dt=2,start='2024-02-28_00:00:00',stop='2024-02-28_00:01:00',driver_factory=factory,config=resolve_namelist(s),noah_resources=resources if args.startup else None,slot_services=slot_services)
        if not args.startup:
            FrameworkInitialization(session).physics_run_init(None)
            session.native._initialized=True;session.initialize()
        from woof.hex.mpasphys.metal_noah_v842 import load_native_resources
        from woof.hex.mpasphys.metal_noah_init_v842 import MetalNoahInitialization
        from woof.hex.mpasphys.noah_v842 import table_dtype
        from tools.thompson_real_column_parity.host_backend import HostModule
        from tools.mpasphys_v842_noah import compare_landuse as lu, compare_init as si
        import subprocess
        from woof.hex.mpasphys.metal_noah_v842 import packaged_decoder
        decoder = packet['rust_decoder'] if 'rust_decoder' in packet else packaged_decoder()
        decoded=subprocess.check_output((decoder['path'],packet['table_directory'],packet['landuse'],packet['soil_type']))
        tables=np.frombuffer(decoded,np.uint32,count=table_dtype().itemsize//4).copy()
        landtables=np.frombuffer(decoded,np.float32,offset=table_dtype().itemsize).copy()
        landhost=HostModule('noah_framework_landuse',lu.source).get_function('replay_landuse')
        soilhost=HostModule('noah_framework_soil',si.header+si.alias+si.source+si.entry).get_function('replay_init')
        f=FrameworkInitialization(session);n=dims['nCells'];ns=dims['nCellsSolve']
        session.native.put_text_field('sfc_input','mminlu',packet['landuse'])
        def put(pool,name,value):
            a=f.field(pool,name)
            if a.ndim and a.shape[-1]>=n:a[...,:n]=value
            else:a[...]=value
            f.write(pool,name,a)
        put('sfc_input','ivgtyp',np.arange(n)%9+1);put('sfc_input','isltyp',np.arange(n)%5+1)
        put('sfc_input','iswater',16);put('sfc_input','isice',24)
        for name in ('tslb','smois','sh2o'):
            a=f.field('sfc_input',name)
            a[...] = np.array([250,270,273.2,285],np.float32)[:,None] if name=='tslb' else np.float32(.25 if name=='smois' else .15)
            f.write('sfc_input',name,a)
        for name,value in [('snow',20),('snowh',.1),('snoalb',.7),('snowc',.3),('xice',0),('sfc_albbck',.2)]:put('sfc_input',name,value)
        put('mesh','rdzw',np.float32(.01));put('mesh','dzu',np.float32(100))
        put('mesh','latCell',np.linspace(-.7,.7,n,dtype=np.float32))
        for name in ('z0','znt','sfc_emibck','sfc_albedo','sfc_emiss','thc','mavail','ust'):
            if ('diag_physics',name) in session.native._fields:put('diag_physics',name,.1)
        # Prime cached solve views to prove initialization refreshes them.
        cached={}
        for pool,name in [('sfc_input',x) for x in ('tslb','smois','sh2o','snoalb','snow','snowh','sfc_albbck')]+[('diag_physics',x) for x in ('z0','znt','sfc_emibck','sfc_albedo','sfc_emiss','thc','mavail','ust')]:
            if (pool,name) in session.native._fields:cached[pool,name]=session.pools.field(pool,name)
        fields=(('mesh','latCell'),('sfc_input','snoalb'),('sfc_input','snowc'),('sfc_input','xice'),('sfc_input','sfc_albbck'),('diag_physics','z0'),('diag_physics','znt'),('diag_physics','sfc_emibck'),('diag_physics','sfc_albedo'),('diag_physics','sfc_emiss'),('diag_physics','thc'),('diag_physics','mavail'),('diag_physics','ust'))
        a=np.zeros((n,14),np.float32);ix=np.zeros((n,9),np.int32)
        for i,key in enumerate(fields):
            if key in session.native._fields:a[:,i]=f.field(*key)[:n]
        a[:,13]=np.float32(.02 if f.configs['config_frac_seaice'] else .5)
        ix[:,0]=f.field('sfc_input','ivgtyp')[:n]
        ix[:,1:]=[session.native.clock_state.current_time.day_of_year,16,24,int(f.configs['config_do_restart']),int(f.configs['config_sfc_albedo']),int(f.configs['config_frac_seaice']),int(fields[11] in session.native._fields),int(fields[12] in session.native._fields)]
        status=np.zeros(n,np.int32);landhost((1,),(128,),(a,ix,landtables,status,np.int32(n)));assert not status.any()
        init=MetalNoahInitialization(f,resources,context)
        if not args.startup:init.initialize_landuse()
        for i,key in enumerate(fields):
            if not args.startup and i>=4 and key in session.native._fields:
                actual=f.field(*key)[:n];assert np.array_equal(actual.view(np.uint32),a[:,i].view(np.uint32)),key
        names=('tslb','smois','sh2o','snoalb','snow','snowh')
        values=np.concatenate([f.field('sfc_input',name)[...,:n].reshape(-1,n).T for name in names],axis=1)
        ints=np.empty((n,4),np.int32);ints[:,0]=f.field('sfc_input','ivgtyp')[:n];ints[:,1]=f.field('sfc_input','isltyp')[:n]
        ints[:,2:]=int(f.configs['config_do_restart']),int(f.configs['config_sfc_snowalbedo'])
        status=np.zeros(n,np.int32);soilhost((1,),(128,),(values,ints,tables,status,np.int32(n)));assert not status.any()
        if args.startup:
            # Scheme tables are already explicitly loaded above; this check
            # isolates their callback while using all four common callbacks.
            def tables(call):session.native.scheme_state={'sf_noah':resources}
            def unexpected(call):raise AssertionError('initialization invoked unrelated physics: '+call.slot)
            hooks=session.bind_hooks(external={} if args.resource_startup else {'physics_tables_init':tables},save_fields=lambda:{},restore_fields=lambda state:None,after_call=lambda entry,call:None)
            config=resolve_namelist(s);registry=SlotRegistry()
            for slot,scheme in config.selectors.items():
                if config.active(slot):registry.register(slot,scheme,config.backends[slot],unexpected)
            for operation in hooks.missing_operations(config):hooks.external[operation]=unexpected
            driver=PhysicsDriver(config,PhysicsManager(config,2),registry,hooks)
            driver.initialize(session)
            assert [c['operation'] for c in driver.calls]==['physics_tables_init','physics_registry_init','physics_run_init','physics_init']
            expected_dzs=np.array([.1,.3,.6,1],np.float32)[:,None] if not args.restart else np.zeros((4,1),np.float32)
            assert np.all(f.field('sfc_input','dzs')[:,:n]==expected_dzs)
            if args.resource_startup:
                assert session.native.scheme_state=={'sf_noah':slot_services['land_surface','sf_noah']}
                assert not bool(session.native.get_variable('l_mp_tables')[0])
            assert int(session.native.get_variable('n_microp')[0])==1
            assert session.native.get_variable('dt_microp')[0]==np.float32(2)
            for i,key in enumerate(fields):
                if i>=4 and key in session.native._fields:
                    assert np.array_equal(f.field(*key)[:n].view(np.uint32),a[:,i].view(np.uint32)),key
            assert np.all(f.field('tend_physics','mp_top_level')==4)
            for (pool,name,level),(_,device,selection) in session.pools.buffers.items():
                expected=session.native.get_field(pool,name,level=level)[selection]
                assert np.array_equal(device.to_numpy().view(np.uint32),expected.view(np.uint32)),(pool,name,level)
        else:init.initialize_noah()
        offset=0
        for name in names:
            width=4 if name in ('tslb','smois','sh2o') else 1
            expected=values[:,offset:offset+width].T;actual=f.field('sfc_input',name)[...,:n].reshape(width,n)
            assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),name
            offset+=width
        for key,device in cached.items():
            host=f.field(*key)[...,:ns];assert np.array_equal(device.to_numpy().view(np.uint32),host.view(np.uint32)),key
        if not args.restart:
            baseline={name:f.field('sfc_input',name).copy() for name in names}
            invalid=f.field('sfc_input','isltyp');invalid[0]=0;f.write('sfc_input','isltyp',invalid)
            try:
                init.initialize_noah()
                raise AssertionError('expected native 140 refusal')
            except RuntimeError as error:
                assert '140' in str(error),error
            for name in names:
                assert np.array_equal(f.field('sfc_input',name).view(np.uint32),baseline[name].view(np.uint32)),name
        print('classification',packet['landuse'],'restart',args.restart,'startup callback' if args.startup else 'lower callbacks','all-cell initialization matched source for',n,'cells; refreshed',len(cached),'cached solve fields; halo included')
        if args.startup:
            driver.finalize(session)
            assert not session.native._initialized
            assert not session.native.scheme_state
        session.close()
