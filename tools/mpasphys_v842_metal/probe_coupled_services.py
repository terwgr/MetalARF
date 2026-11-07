"""Native Metal multi-service caller fixture, with explicit injected dynamics.

The PBL, surface layer, Noah, sea ice, Thompson, RP6 window, shared framework
and resident coupling are the actual service objects. Ice-free cells exercise
the sea-ice boundary without entering its ice-column solver. Dynamics is an
injected stationary candidate and winds are prescribed; this is not forecast parity.
"""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager,CORRECTIONS_V842
from woof.hex.mpasphys.framework_backend_v842 import registry_source
from woof.hex.mpasphys.pool_v842 import RegistryLayout
from woof.hex.mpasphys.metal_session_v842 import MetalPhysicsSession
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry, metal_slot_definitions
from woof.hex.mpasphys.metal_noah_v842 import MetalMpasNoah,load_native_resources
from woof.hex.mpasphys.metal_bl_mynn_slot_v842 import MetalMpasBlMynn
from woof.hex.mpasphys.metal_seaice_v842 import MetalMpasSeaice
from woof.hex.mpasphys.metal_thompson_column_v842 import MetalMpasThompson
from woof.hex.mpasphys.metal_thompson_tables_v842 import load_tables
from woof.hex.mpasphys.metal_native_coupling_v842 import MetalNativeCoupling
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

TEXT="""&physics
config_physics_suite='none'
config_sfclayer_scheme='sf_mynn'
config_lsm_scheme='sf_noah'
config_pbl_scheme='bl_mynn'
config_microp_scheme='mp_thompson'
config_pbl_interval='00:10:00'
config_frac_seaice=.true.
config_o3climatology=.false.
config_sst_update=.false.
config_sstdiurn_update=.false.
config_deepsoiltemp_update=.false.
/
"""

def same(a,b):
    if isinstance(b,np.ndarray):
        assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    elif isinstance(b,dict):
        assert a.keys()==b.keys()
        for k in b:same(a[k],b[k])
    elif isinstance(b,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

@contextmanager
def session_fixture(context,packet,tables,*,restart=False,start='2024-06-28_12:00:00',thompson_oracle=None,mynn_oracle=None,noah_oracle=None,correct_findings=('RP6',),fractional_ice=False):
    text=TEXT+('\n&restart\nconfig_do_restart=.true.\n/\n' if restart else '')
    # The sealed carrier retains RP6; resolve only source-backed service declarations.
    requested=frozenset(correct_findings)
    preliminary=resolve_namelist(TEXT)
    active={slot:'metal' for slot in preliminary.selectors if preliminary.active(slot)}
    xp=MetalArrayModule(context); noah=load_native_resources(packet,context)
    services={('land_surface','sf_noah'):MetalMpasNoah(noah['tables'],xp.zeros((4,),np.int32)),
              ('pbl','bl_mynn'):MetalMpasBlMynn(context),
              ('seaice','sf_noah'):MetalMpasSeaice(context,module_locations=[0,0],i_start=1,j_index=1),
              ('microphysics','mp_thompson'):MetalMpasThompson(tables)}
    definitions=metal_slot_definitions(services)
    def declared(slot,scheme,backend):
        return definitions.get((slot,scheme),(None,frozenset()))[1] if backend=='metal' else frozenset()
    config=resolve_namelist(TEXT,physics_mode='correct',correct_skip=set(CORRECTIONS_V842)-requested,
                           backends=active,declarations=declared,_allow_metal=True)
    if not requested <= config.corrections:
        raise ValueError('requested fixture corrections are not declared by actual services')
    dims={'nCells':6,'nEdges':8,'nVertices':5,'nVertLevels':17,'nSoilLevels':4,
          'maxEdges':6,'maxEdges2':12,'vertexDegree':3,'nCellsSolve':6,'nEdgesSolve':8,'nVerticesSolve':5}
    layout=RegistryLayout(registry_source(),config,text,dims)
    layout.dimensions.update(nCellsSolve=6,nEdgesSolve=8,nVerticesSolve=5)
    assert bool(layout.configs['config_do_restart'])==restart
    if thompson_oracle is not None:
        service=services['microphysics','mp_thompson'];launch=service.launch_slot
        service.launch_slot=lambda call:thompson_oracle.compare(call,launch)
    if mynn_oracle is not None:
        service=services['pbl','bl_mynn'];launch_mynn=service.launch_slot
        service.launch_slot=lambda call:mynn_oracle.compare(call,launch_mynn) if not mynn_oracle.calls else launch_mynn(call)
    if noah_oracle is not None:
        noah_service=services['land_surface','sf_noah'];launch_noah=noah_service.launch_slot
        noah_service.launch_slot=lambda call:noah_oracle.compare(call,launch_noah,noah_service) if not noah_oracle.calls else launch_noah(call)
    def factory(session):
        registry=MetalSlotRegistry();session.register_slots(registry)
        # Native hooks have empty bodies without MPAS_OPENACC / with cu off.
        empty=lambda call:None
        hooks=session.bind_hooks(external={'pre_microphysics':empty,'post_microphysics':empty,
                    'update_convection_step1':empty},save_fields=lambda:{},
                    restore_fields=lambda state:None,after_call=lambda entry,call:None)
        return PhysicsDriver(config,PhysicsManager(config,300),registry,hooks)
    with TemporaryDirectory(prefix='mpas-coupled-registry-') as temp:
        root=Path(temp);records=[]
        for i,row in enumerate(layout.fields):
            r={k:row[k] for k in ('pool','name','kind','dimensions','shape')}
            if row['kind']=='character':r['text']=''
            elif row['pool']=='mesh' and row['name'] in ('indexToCellID','indexToEdgeID','indexToVertexID'):
                a=np.arange(1,np.prod(row['shape'])+1,dtype='<i4').reshape(row['shape'],order='F')
                data=a.tobytes(order='F');path=root/f'{i}.bin';path.write_bytes(data)
                r.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            elif (row['pool'],row['name'])==('mesh','nominalMinDc'):
                a=np.full(row['shape'],3000,dtype='<f4',order='F')
                data=a.tobytes(order='F');path=root/f'{i}.bin';path.write_bytes(data)
                r.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            else:r['fill']=0
            records.append(r)
        (root/'manifest.json').write_text(json.dumps({'schema':'mpas-native-pool-payload-v1','fields':records}))
        session=MetalPhysicsSession(layout=layout,payload=root,context=context,dt=300,
            start=start,stop='2024-06-28_13:00:00',driver_factory=factory,
            config=config,noah_resources=noah,slot_services=services)
        session.native.put_text_field('sfc_input','mminlu',packet['landuse'])
        def put(pool,name,value,level=1):
            key=(pool,name)
            if key not in session.native._fields:raise KeyError(f'fixture requires native field {pool}.{name}')
            a=session.native.get_field(pool,name,level=level);a[...]=value
            session.native.put_field(pool,name,a,level=level,nlevels=session.native._fields[key][2])
        for name,value in [('ivgtyp',5),('isltyp',3),('iswater',16),('isice',24),('xland',1),
                           ('xice',0),('skintemp',300),('sst',295),('tmn',290),('snoalb',.5),
                           ('snowc',0),('snow',0),('snowh',0),('smois',.3),('sh2o',.25),
                           ('tslb',290),('greenfrac',.5),('albedo12m',.2),('vegfra',.5)]:put('sfc_input',name,value)
        for name,value in [('latCell',.3),('lonCell',.2),('meshDensity',1),('nominalMinDc',3000),
                           ('zz',1),('rdzw',.01),('dzu',100),('cellsOnEdge',1),('edgeNormalVectors',.5)]:put('mesh',name,value)
        z=session.native.get_field('mesh','zgrid');z[...] = np.arange(z.shape[0],dtype=np.float32)[:,None]*100
        put('mesh','zgrid',z)
        for level in (1,2):
            for name,value in [('rho_zz',1.15),('theta_m',290),('u',5),('w',.1)]:put('state',name,value,level)
            a=session.native.get_field('state','scalars',level=level)
            for species,value in [('qv',.006),('qc',.0001),('qr',.00008),('qi',.00001),('qs',.00001),('qg',.00001),('ni',900),('nr',2000)]:
                index=layout.indices['state','index_'+species]
                if index>0:a[index-1]=np.float32(value)
            put('state','scalars',a,level)
        for name,value in [('exner',.98),('exner_base',.97),('pressure_base',90000),('pressure_p',100),
                           ('rtheta_base',300),('rtheta_p',10),('rho_p',.1),('rho_edge',1.15),
                           ('uReconstructZonal',5),('uReconstructMeridional',3)]:put('diag',name,value)
        pressure=session.native.get_field('diag','pressure_base')
        pressure[...] = np.linspace(100000,50000,pressure.shape[0],dtype=np.float32)[:,None]
        put('diag','pressure_base',pressure)
        for name,value in [('hpbl',600),('el_pbl',10),('qke',.1),('cov',.001),('qsq',.001),('tsq',.001),
                           ('sh3d',.1),('sm3d',.1),('qc_bl',.0001),('qi_bl',.00002),('cldfrac_bl',.2),
                           ('ust',.4),('hfx',100),('qfx',.0001),('rmol',0),('wspd',5),('chs',.01),
                           ('chs2',.01),('cqs2',.01),('cpm',1004),('znt',.05),('qsfc',.005),
                           ('glw',300),('gsw',350),('sfc_emiss',.95),('mavail',.7),('nt_c',100e6)]:put('diag_physics',name,value)
        session.initialize()
        if fractional_ice:
            for name,value in [('xland',2),('xice',.5),('skintemp',263),('sst',280),('landmask',0)]:
                target=session.pools.field('sfc_input',name)
                target[...]=np.full(target.shape,value,dtype=target.dtype)
        if restart:session.resident_preparation.activate()
        yield session
        session.close()


def make_advance(session):
    p=session.pools;xp=p.arrays;dims=session.layout.dimensions;nc=dims['nCellsSolve'];nz=dims['nVertLevels']
    names=session.resident_preparation.names
    def wind(atmosphere):
        return SimpleNamespace(u_p=p.field('diag','uReconstructZonal'),v_p=p.field('diag','uReconstructMeridional'))
    adapter=MetalNativeCoupling(session,names,reconstruct_winds=wind)
    tx=session.transaction
    def advance(step):
        flags=tx.begin_step(itimestep=step,time_seconds=float(session.native.clock_state.elapsed),driver_pools=session.slot_pools)
        due=tx.driver.manager.pbl_due
        calls={entry['slot'] for entry in tx.driver.calls}
        assert ('pbl' in calls)==due and ('land_surface' in calls)==due
        held=tx.sources(driver_pools=session.slot_pools)
        state=SimpleNamespace(rho=p.field('state','rho_zz',level=2).copy(),scalars=p.field('state','scalars',level=2).copy())
        saved=SimpleNamespace(theta_m=p.field('state','theta_m',level=2).copy(),
            normal_velocity=p.field('state','u',level=2).copy(),vertical_velocity=p.field('state','w',level=2).copy(),
            density_perturbation=p.field('diag','rho_p').copy(),rho_theta_perturbation=p.field('diag','rtheta_p').copy(),
            pressure_perturbation=p.field('diag','pressure_p').copy(),exner=p.field('diag','exner').copy())
        vertical=SimpleNamespace(zz=p.field('mesh','zz'));reference=SimpleNamespace(pressure_base=p.field('diag','pressure_base'))
        outer=SimpleNamespace(rho=state.rho.copy(),scalars=state.scalars.copy())
        def synchronize(current,carried):adapter.synchronize(SimpleNamespace(state=current,saved=carried,vertical=vertical,reference=reference),level=2)
        session.resident_transport.bind(current=state,carried=saved,outer=outer,rate=None,held=held,
            scalar_advection=False,monotonic=False,positive_definite=False,convection='off',scalar_names=names,synchronize=synchronize)
        tx.finish_step(driver_pools=session.slot_pools)
        assert sum(c['slot']=='microphysics' for c in tx.driver.calls)==1
        tx.commit();session.native.advance_clock();return due
    return advance

def run(session, *, fractional_ice=False):
    tx=session.transaction;p=session.pools;advance=make_advance(session)
    assert advance(1)
    if fractional_ice:
        assert session.config.correct('RP3')
        for slot in ('surface_layer','land_surface','seaice'):
            assert sum(call['slot']==slot for call in tx.driver.calls)==1
        state=session.slot_services['seaice','sf_noah'].restart_state()['module_locations']
        assert np.array_equal(state,np.array([6,1],np.int32))
    middle=tx.checkpoint();assert not advance(2);assert advance(3);expected=tx.checkpoint()
    tx.restore(middle);assert not advance(2);assert advance(3);actual=tx.checkpoint();same(actual,expected)
    assert np.any(p.field('tend_physics','rthblten').to_numpy()!=0)
    print('actual surface/Noah/seaice/PBL/Thompson/window/held-source driver cadence and in-memory replay; synthetic '+('fractional-ice' if fractional_ice else 'ice-free')+' columns, injected stationary dynamics and prescribed winds',flush=True)

if __name__=='__main__':
    packet=json.loads(Path(sys.argv[1]).read_text())
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        context._submit_kernel_batch=sync
        tables=load_tables({'table_directory':sys.argv[2]},context)
        oracle=None
        if '--source' in sys.argv[3:]:
            from tools.mpasphys_v842_metal.oracle_coupled_thompson import ThompsonOracle
            oracle=ThompsonOracle(sys.argv[2],17)
        mynn=None
        if '--mynn-source' in sys.argv[3:]:
            from tools.mpasphys_v842_metal.oracle_coupled_mynn import MynnOracle
            mynn=MynnOracle()
        noah_source=None
        if '--noah-source' in sys.argv[3:]:
            from tools.mpasphys_v842_metal.oracle_coupled_noah import NoahOracle
            noah_source=NoahOracle()
        rp3='--rp3' in sys.argv[3:]
        with session_fixture(context,packet,tables,thompson_oracle=oracle,mynn_oracle=mynn,noah_oracle=noah_source,correct_findings=('RP3','RP6') if rp3 else ('RP6',),fractional_ice=rp3) as session:run(session,fractional_ice=rp3)
        if oracle is not None:
            assert oracle.calls==5 and oracle.words>0
            print('live coupled Thompson selected-source comparisons',oracle.calls,'words',oracle.words,flush=True)
        if mynn is not None:
            assert mynn.calls==1 and mynn.words>0
            print('live coupled MYNN whole selected-source comparison',mynn.calls,'words',mynn.words,flush=True)
        if noah_source is not None:
            assert noah_source.calls==1 and noah_source.words>0
            print('live coupled Noah selected-source comparison',noah_source.calls,'words',noah_source.words,flush=True)
