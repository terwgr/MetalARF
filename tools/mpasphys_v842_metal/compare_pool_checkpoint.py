"""Hydrate real Registry metadata, mutate Metal pools, snapshot and restore them."""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import os
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from woof.hex.mpasphys.driver_v842 import resolve_namelist, PhysicsDriver, PhysicsManager, SlotRegistry, NativeDriverHooks, SlotCall, StepFlags, Backend
from woof.hex.mpasphys.metal_transaction_v842 import MetalPhysicsTransaction
from woof.hex.mpasphys.metal_session_v842 import MetalPhysicsSession
from woof.hex.mpasphys.framework_backend_v842 import FrameworkPhysics, registry_source
from woof.hex.mpasphys.pool_v842 import NativePoolHydration, RegistryLayout
from woof.hex.mpasphys.metal_pools_v842 import MetalPoolViews
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    s="&physics\nconfig_physics_suite='convection_permitting'\n/\n"
    dimensions={'nCells':7,'nEdges':9,'nVertices':5,'nVertLevels':4,'nSoilLevels':4,
                'maxEdges':6,'maxEdges2':12,'vertexDegree':3,'nCellsSolve':6,
                'nEdgesSolve':8,'nVerticesSolve':5}
    layout=RegistryLayout(registry_source(),resolve_namelist(s),s,dimensions)
    layout.dimensions.update(nCellsSolve=6,nEdgesSolve=8,nVerticesSolve=5)
    rng=np.random.default_rng(421)
    with tempfile.TemporaryDirectory(prefix='mpas-v842-pool-') as temp:
        root=Path(temp); records=[]
        for i,row in enumerate(layout.fields):
            r={k:row[k] for k in ('pool','name','kind','dimensions','shape')}
            if row['kind']=='character': r['text']=''
            elif row['pool']=='mesh' and row['name'] in ('indexToCellID','indexToEdgeID','indexToVertexID'):
                a=np.arange(1,np.prod(row['shape'])+1,dtype='<i4').reshape(row['shape'],order='F')
                entity={'indexToCellID':'nCells','indexToEdgeID':'nEdges','indexToVertexID':'nVertices'}[row['name']]
                owned=layout.dimensions[entity+'Solve']; count=layout.dimensions[entity]
                if count>owned: a[owned:count]=a[:count-owned]
                path=root/f'{i}.bin'; data=a.tobytes(order='F'); path.write_bytes(data)
                r.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            elif (row['pool'],row['name']) in {('state','rho_zz'),('state','theta_m'),('state','scalars'),('mesh','nominalMinDc')}:
                a=rng.uniform(.2,2,row['shape']).astype('<f4',order='F')
                path=root/f'{i}.bin'; data=a.tobytes(order='F'); path.write_bytes(data)
                r.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            else: r['fill']=0
            records.append(r)
        (root/'manifest.json').write_text(json.dumps({'schema':'mpas-native-pool-payload-v1','fields':records}))
        native=FrameworkPhysics(layout=layout)
        hydration=NativePoolHydration(native,layout,root)
        hydration.populate()
        with MetalContext() as context:
            views=MetalPoolViews(native,layout,context)
            first=views.field('state','rho_zz',level=1)
            second=views.field('state','rho_zz',level=2)
            theta=views.field('state','theta_m',level=1)
            first[...] = np.arange(first.size,dtype=np.float32).reshape(first.shape) + np.float32(1)
            second[...] = np.arange(second.size,dtype=np.float32).reshape(second.shape) + np.float32(101)
            theta[...] = np.arange(theta.size,dtype=np.float32).reshape(theta.shape) + np.float32(201)
            before=views.snapshot()
            first.fill(-7); second.fill(-8); theta.fill(-9)
            views.restore(before)
            after=views.snapshot()
            if set(before)!=set(after): raise AssertionError('snapshot field set changed')
            for k in before:
                if not np.array_equal(before[k].view(np.uint32),after[k].view(np.uint32)):
                    raise AssertionError(f'pool restore differs: {k}')
            for d,k in ((first,'state.rho_zz.1'),(second,'state.rho_zz.2'),(theta,'state.theta_m.1')):
                if not np.array_equal(d.to_numpy(),before[k][...,:dimensions['nCellsSolve']]):
                    raise AssertionError(f'device restore differs: {k}, got {d.to_numpy().ravel()[:8]}, expected {before[k][...,:dimensions["nCellsSolve"]].ravel()[:8]}')
            dry=resolve_namelist("&physics\nconfig_physics_suite='none'\n/\n")
            class Hooks:
                def __init__(self): self.count=0
                def invoke(self, operation, call):
                    if operation != 'physics_get_tend': raise AssertionError(operation)
                    self.count += 1
                    first.fill(777)
                    return self.count
                def restart_state(self): return {'count':self.count}
                def restore_restart_state(self, state): self.count=state['count']
            def make_driver():
                return PhysicsDriver(dry,PhysicsManager(dry,2),SlotRegistry(),Hooks())
            driver=make_driver(); driver.initialize(views)
            transaction=MetalPhysicsTransaction(driver,views,make_driver)
            baseline=transaction.checkpoint()
            transaction.begin_step(itimestep=1,time_seconds=0,driver_pools=views)
            if transaction.sources(driver_pools=views)!=1: raise AssertionError('first held source')
            transaction.finish_step(driver_pools=views)
            transaction.rollback()
            rolled=transaction.checkpoint()
            if baseline['driver']!=rolled['driver']: raise AssertionError('driver rollback differs')
            for key in baseline['pools']:
                if not np.array_equal(baseline['pools'][key].view(np.uint32),rolled['pools'][key].view(np.uint32)):
                    raise AssertionError(f'transaction rollback differs: {key}')
            transaction.begin_step(itimestep=1,time_seconds=0,driver_pools=views)
            transaction.sources(driver_pools=views)
            transaction.finish_step(driver_pools=views)
            transaction.commit()
            if float(native.get_field('state','rho_zz')[0,0]) != 777:
                raise AssertionError('native observer read stale committed pool')
            committed=transaction.checkpoint()
            transaction.restore(baseline)
            transaction.restore(committed)
            if transaction.checkpoint()['driver']!=committed['driver']:
                raise AssertionError('committed driver restore differs')
            try: views.field('state','rho_zz',level=1,full=True)
            except ValueError: pass
            else: raise AssertionError('conflicting packed extents were admitted')
        with MetalContext() as context:
            xp=MetalArrayModule(context)
            dry=resolve_namelist("&physics\nconfig_physics_suite='none'\n/\n")
            class SessionHooks:
                def __init__(self, session): self.session,self.count=session,0
                def invoke(self, operation, call):
                    if operation != 'physics_get_tend': raise AssertionError(operation)
                    self.count += 1
                    self.session.pools.field('state','rho_zz',level=1).fill(123)
                    return self.count
                def restart_state(self):
                    return {'count':self.count, 'native_rho':float(self.session.native.get_field('state','rho_zz')[0,0])}
                def restore_restart_state(self, state): self.count=state['count']
            def factory(session):
                return PhysicsDriver(dry,PhysicsManager(dry,2),SlotRegistry(),SessionHooks(session))
            n_oz=layout.dimensions['nOznLevels']
            ozone_tables={'pin':np.linspace(1,1000,n_oz,dtype=np.float32),
                          'lat_ozone':np.linspace(-90,90,64,dtype=np.float32),
                          'ozmixin':rng.uniform(1e-6,1e-4,(12,64,n_oz)).astype(np.float32)}
            packet=os.environ.get('WOOF_MPAS_OZONE_PACKET')
            if packet:
                from woof.hex.mpasphys.framework_resources_v842 import decode_ozone_tables
                decoded=decode_ozone_tables(json.loads(Path(packet).read_text()),xp)
                ozone_tables={name:value.to_numpy() for name,value in decoded.items()}
            session=MetalPhysicsSession(layout=layout,payload=root,context=context,dt=2,
                start='2024-02-28_00:00:00',stop='2024-02-28_00:01:00',driver_factory=factory,config=resolve_namelist(s),ozone_tables=ozone_tables)
            from woof.hex.mpasphys.framework_init_v842 import FrameworkInitialization
            FrameworkInitialization(session).physics_run_init(None)
            session.native._initialized=True  # Isolated framework setup; not scheme initialization.
            session.initialize()
            from woof.hex.mpasphys.metal_framework_v842 import MetalFrameworkInitialization
            # Ozone is an explicit component callback; full framework
            # initialization still requires land-use and scheme callbacks.
            initialization=MetalFrameworkInitialization(session)
            if hasattr(initialization, 'physics_init'):
                raise AssertionError('partial ozone callback exposes full framework initialization')
            initialization.initialize_ozone(None)
            monthly_host=session.native.get_field('atm_input','ozmixm')
            if not np.any(monthly_host[:,:,:dimensions['nCells']]):
                raise AssertionError('Metal ozone initialization did not fill native halo cells')
            from tools.thompson_real_column_parity.host_backend import HostModule
            from woof.hex.mpasphys.radshared_v842 import CUDA_SOURCE
            source_start=CUDA_SOURCE.index('extern "C" __global__ void ozone_latitude(')
            source_end=CUDA_SOURCE.index('// MPAS mpas_atmphys_driver_radiation_lw.F',source_start)
            header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()
            ozone_host=HostModule('mpas_v842_ozone',header+'\n'+CUDA_SOURCE[source_start:source_end])
            all_cells=dimensions['nCells']; local_month=np.empty((all_cells,n_oz,12),np.float32)
            ozone_host.get_function('ozone_latitude')(((all_cells*n_oz*12+127)//128,),(128,),
                (all_cells,n_oz,np.ascontiguousarray(session.native.get_field('mesh','latCell')[:all_cells]),
                 ozone_tables['lat_ozone'],ozone_tables['ozmixin'],local_month))
            if not np.array_equal(monthly_host[:,:,:all_cells].view(np.uint32),local_month.transpose(2,1,0).copy().view(np.uint32)):
                raise AssertionError('native full-cell ozone initialization differs from source')
            prior=session.transaction.checkpoint()
            prepared_before=session.pools.prepared('psfc_p').to_numpy().copy()
            session.transaction.begin_step(itimestep=1,time_seconds=0,driver_pools=session)
            session.pools.prepared('psfc_p').fill(74567)
            session.ozone_service.initialized=False
            session.transaction.sources(driver_pools=session)
            session.transaction.finish_step(driver_pools=session)
            session.transaction.rollback()
            restored=session.transaction.checkpoint()
            if not session.ozone_service.initialized:
                raise AssertionError('ozone service state did not roll back')
            if not np.array_equal(session.pools.prepared('psfc_p').to_numpy(),prepared_before):
                raise AssertionError('prepared state rollback differs')
            if prior['driver'] != restored['driver']: raise AssertionError('session driver rollback differs')
            for key in prior['pools']:
                if not np.array_equal(prior['pools'][key].view(np.uint32),restored['pools'][key].view(np.uint32)):
                    raise AssertionError(f'session pool rollback differs: {key}')
            hooks=session.bind_hooks(external={},save_fields=lambda:{},restore_fields=lambda x:None,after_call=lambda e,c:None)
            options=dict(resolve_namelist(s).execution_options)
            options.update(config_sfc_albedo=True,config_bucket_radt=1.,config_bucket_rainc=1.)
            flags=StepFlags(greeness=True,rain_bucket=True,radiation_bucket=True)
            call=SlotCall('driver','driver',Backend.NATIVE_CPU,1,0.,2.,flags,session,options)
            hooks.invoke('physics_timetracker_state',call)
            hooks.invoke('o3climatology_from_MPAS',call)
            if not np.any(session.pools.field('diag_physics','o3clim').to_numpy()):
                raise AssertionError('native driver ozone callback did not publish ozone')
            green=session.pools.field('sfc_input','greenfrac')
            albedo=session.pools.field('sfc_input','albedo12m')
            land=session.pools.field('sfc_input','landmask')
            green[...] = np.arange(green.size,dtype=np.float32).reshape(green.shape)/np.float32(100)
            albedo[...] = np.full(albedo.shape,40,np.float32)
            land[...] = np.ones(land.shape,np.int32)
            hooks.invoke('physics_update_surface',replace(call,slot='physics_update_surface'))
            if not np.any(session.pools.field('sfc_input','vegfra').to_numpy()):
                raise AssertionError('driver surface callback did not update vegetation')
            call=SlotCall('update_convection_step2','driver',Backend.NATIVE_CPU,1,0.,2.,flags,session,options)
            session.pools.field('diag_physics','cuprec').fill(.75)
            hooks.invoke('update_convection_step2',call)
            if not np.all(session.pools.field('diag_physics','rainc').to_numpy()==.5):
                raise AssertionError('driver precipitation callback differs')
            from woof.hex.mpasphys.surface_update_v842 import RADIATION_FIELDS
            for name in RADIATION_FIELDS:
                session.pools.field('diag_physics',name).fill(.75)
            hooks.invoke('update_radiation_diagnostics',replace(call,slot='update_radiation_diagnostics'))
            for name in RADIATION_FIELDS:
                if not np.all(session.pools.field('diag_physics','ac'+name).to_numpy()==.5):
                    raise AssertionError(f'driver radiation callback differs: {name}')
            saved_service=session.transaction.checkpoint()
            clock_surface={k:v for k,v in saved_service['services'].items() if k != 'prepared'}
            if json.loads(json.dumps(clock_surface)) != clock_surface:
                raise AssertionError('service clock checkpoint is not JSON-roundtrip stable')
            if saved_service['services']['surface']['february']!=29:
                raise AssertionError('surface leap state missing from transaction checkpoint')
            session.native.advance_clock()
            session.native.put_text_field('state','xtime','2024-02-28_00:00:02',level=1,nlevels=2)
            session.surface_service.monthly.calendar.february=28
            session.surface_service.executions['physics_update_surface']=0
            session.transaction.restore(saved_service)
            state=session.surface_service.restart_state()
            if session.native.clock_state.elapsed != 0 or session.native.get_text_field('state','xtime',level=1) != saved_service['services']['clock']['text']['state.xtime.1']:
                raise AssertionError('framework clock or text restore differs')
            if state['february']!=29 or state['executions']['physics_update_surface']!=1 or state['executions']['update_radiation_diagnostics']!=1:
                raise AssertionError('surface callback state did not restore')
            session.pools.field('state','rho_zz',level=1).fill(456)
            current=session.transaction.checkpoint()
            if current['driver']['fields']['native_rho'] != 456:
                raise AssertionError('driver hook read stale native pool')
            from tools.mpasphys_v842_metal.compare_resident import exercise,exercise_surface,exercise_gwdo
            exercise(session)
            session.pools.field('mesh','meshDensity').fill(1)
            session.pools.field('sfc_input','xland').fill(1)
            session.layout.configs['config_len_disp']=12000.
            # Exercise real driver sequencing; only the unrelated radiation
            # physics kernel is a test callback, not a production substitute.
            active_text="&physics\nconfig_physics_suite='none'\nconfig_radt_sw_scheme='rrtmg_sw'\nconfig_radt_cld_scheme='cld_fraction'\nconfig_o3climatology=.true.\nconfig_bucket_update='00:00:02'\nconfig_bucket_radt=1.0\n/\n"
            active=resolve_namelist(active_text)
            active=replace(active,backends={**active.backends,'cloudiness':'metal'})
            def empty(call): return None
            externals={n:empty for n in ('physics_tables_init','physics_registry_init','physics_run_init','physics_init',
                'update_convection_step1',
                'atmphys_finalize')}
            active_hooks=session.bind_hooks(external=externals,save_fields=lambda:{},restore_fields=lambda x:None,after_call=lambda e,c:None)
            from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
            registry=MetalSlotRegistry()
            session.register_slots(registry)
            def radiation_callback(call):
                for name in RADIATION_FIELDS: session.pools.field('diag_physics',name).fill(.25)
                session.pools.field('tend_physics','rthratensw').fill(.0025)
            registry.register('radiation_sw','rrtmg_sw',Backend.NATIVE_CPU,radiation_callback)
            active_driver=PhysicsDriver(active,PhysicsManager(active,2),registry,active_hooks)
            active_driver.initialize(session)
            before_counts=session.surface_service.restart_state()['executions']
            active_driver.begin_step(itimestep=1,time_seconds=0, pools=session)
            if 'o3climatology_from_MPAS' not in [x['operation'] for x in active_driver.calls]:
                raise AssertionError('real driver did not call Metal CAM ozone service')
            nc=dimensions['nCellsSolve']
            local=np.ascontiguousarray(session.pools.field('atm_input','ozmixm').to_numpy().transpose(2,1,0))
            expected_ozone=np.empty((nc,n_oz),np.float32)
            ozone_host.get_function('ozone_time')(((nc*n_oz+127)//128,),(128,),
                (nc*n_oz,np.float32(session.arguments.calendar['julian']),local,expected_ozone))
            actual_ozone=session.pools.field('diag_physics','o3clim').to_numpy()
            if not np.array_equal(actual_ozone.view(np.uint32),expected_ozone.T.copy().view(np.uint32)):
                raise AssertionError(f'real driver CAM ozone differs from selected source: {actual_ozone.ravel()[:5]} vs {expected_ozone.T.ravel()[:5]}, julian {session.arguments.calendar["julian"]}')
            nz=dimensions['nVertLevels']
            pressure=session.pools.prepared('pres_p')[:nz]
            ozone_vmr=session.pools.field('diag_physics','o3vmr')
            session.ozone_service.vertical(pressure=pressure,out=ozone_vmr,pressure_in_pa=True)
            local_pressure=np.ascontiguousarray(pressure.to_numpy().T)
            local_ozone=np.ascontiguousarray(actual_ozone.T)
            expected_vmr=np.empty((nc,nz),np.float32); upper=np.zeros(nc,np.int32); low=np.zeros(1,np.int32)
            for k in range(nz-1,-1,-1):
                ozone_host.get_function('ozone_min_upper')((1,),(128,),(nc,n_oz,upper,low))
                ozone_host.get_function('ozone_vertical_level')((1,),(128,),
                    (nc,nz,n_oz,k,1,local_pressure,ozone_tables['pin'],local_ozone,low,upper,expected_vmr))
            if not np.array_equal(ozone_vmr.to_numpy().view(np.uint32),expected_vmr.T.copy().view(np.uint32)):
                raise AssertionError('native ozone vertical callback differs from selected source')
            after_counts=session.surface_service.restart_state()['executions']
            if after_counts['physics_update_surface']!=before_counts['physics_update_surface']+1 or after_counts['update_radiation_diagnostics']!=before_counts['update_radiation_diagnostics']+1:
                raise AssertionError('real driver did not invoke Metal surface services')
            if not np.any(session.pools.field('diag_physics','cldfrac').to_numpy()):
                raise AssertionError('real driver cloud slot produced no clouds')
            cloud_start=CUDA_SOURCE.index('extern "C" __global__ void cloud_from(')
            cloud_end=CUDA_SOURCE.index('// MPAS module_mp_thompson_cldfra3.F',cloud_start)
            cloud_host=HostModule('mpas_v842_cloud_fraction',header+'\n'+CUDA_SOURCE[cloud_start:cloud_end])
            nz=dimensions['nVertLevels']; nc=dimensions['nCellsSolve']
            prepared={name:np.ascontiguousarray(session.pools.prepared(name+'_p')[:nz].to_numpy().T)
                      for name in ('qv','qc','qi','qs','t','pres')}
            density=session.pools.field('mesh','meshDensity').to_numpy()
            land=session.pools.field('sfc_input','xland').to_numpy()
            outputs=[np.empty((nc,nz),np.float32) for _ in range(5)]
            dx=np.empty(nc,np.float32); xland=np.empty(nc,np.float32)
            cloud_host.get_function('cloud_from')(((nc+127)//128,),(128,),
                (nc,nz,np.float32(12000),density,land,*(prepared[n] for n in ('qv','qc','qi','qs')),dx,xland,*outputs))
            cloud_host.get_function('cloud_simple')(((nc*nz+127)//128,),(128,),
                (nc*nz,1,1,1,*outputs[:4],prepared['t'],prepared['pres'],outputs[4]))
            cloud_expected={name:value.T.copy() for name,value in zip(('qvrad','qcrad','qirad','qsrad','cldfrac'),outputs)}
            cloud_expected.update(dx=dx,xland=xland)
            for name,expected in cloud_expected.items():
                actual=session.resident_preparation.radshared[name].to_numpy()
                if not np.array_equal(actual.view(np.uint32),expected.view(np.uint32)):
                    raise AssertionError(f'real driver cloud carrier differs from source: {name}')
            carrier=session.resident_preparation.radshared['qcrad']
            before_cloud=carrier.to_numpy().copy()
            cloud_checkpoint=session.transaction.checkpoint()
            carrier.fill(-1)
            session.transaction.restore(cloud_checkpoint)
            if not np.array_equal(carrier.to_numpy().view(np.uint32),before_cloud.view(np.uint32)):
                raise AssertionError('resident radiation carrier did not restore')
            session.pools.field('state','rho_zz',level=2).fill(1.2)
            session.pools.field('diag','rho_edge').fill(1.1)
            held=active_driver.physics_get_tend(rk_step=1,dynamics_substep=1,pools=session)
            from tools.mpasphys_v842_metal.compare_live_tendencies import exercise as compare_held
            compare_held(session,held)
            if not np.any(held.rho_theta.to_numpy()):
                raise AssertionError('real driver produced no radiation source')
            if active_driver.physics_get_tend(rk_step=2,dynamics_substep=1,pools=session) is not held:
                raise AssertionError('real driver failed to hold sources across RK stages')
            from tools.mpasphys_v842_metal.compare_live_transport import exercise as compare_transport
            compare_transport(session,active_driver,held)
            exercise_surface(session)
            exercise_gwdo(session)
            # Isolate PBL rates while exercising the real resident tendency hook.
            nc=dimensions['nCellsSolve']; ne=dimensions['nEdgesSolve']; nz=dimensions['nVertLevels']
            session.pools.field('mesh','cellsOnEdge')[...] = np.vstack((np.arange(ne)%nc+1,(np.arange(ne)+1)%nc+1)).astype(np.int32)
            session.pools.field('mesh','east')[...] = np.tile(np.array([[1],[0],[0]],np.float32),(1,nc))
            session.pools.field('mesh','north')[...] = np.tile(np.array([[0],[1],[0]],np.float32),(1,nc))
            session.pools.field('mesh','edgeNormalVectors')[...] = np.tile(np.array([[.6],[.8],[0]],np.float32),(1,ne))
            for i,name in enumerate(('rublten','rvblten','rthblten','rqvblten','rqcblten','rqiblten','rqsblten','rniblten')):
                session.pools.field('tend_physics',name)[...] = np.full((nz,nc),np.float32((i+1)*.0001))
            options=dict(active.execution_options)
            options.update(config_pbl_scheme='bl_mynn',config_microp_scheme='mp_thompson',
                           config_convection_scheme='off',config_radt_lw_scheme='off',config_radt_sw_scheme='off')
            pbl_call=SlotCall('physics_get_tend','driver',Backend.NATIVE_CPU,1,0.,2.,StepFlags(),
                              {'pools':session,'rk_step':1,'dynamics_substep':1},options)
            valid_cells=session.pools.field('mesh','cellsOnEdge').to_numpy()
            invalid_cells=valid_cells.copy(); invalid_cells[0,0]=nc+1
            session.pools.field('mesh','cellsOnEdge')[...] = invalid_cells
            edge_before=session.pools.field('tend_physics','rublten_Edge').to_numpy().copy()
            try:
                active_hooks.invoke('physics_get_tend',pbl_call)
            except ValueError as error:
                if 'halo exchange' not in str(error): raise
            else:
                raise AssertionError('unsupported halo edge was admitted')
            if not np.array_equal(session.pools.field('tend_physics','rublten_Edge').to_numpy(),edge_before):
                raise AssertionError('halo-edge refusal mutated projected rates')
            session.pools.field('mesh','cellsOnEdge')[...] = valid_cells
            pbl_held=active_hooks.invoke('physics_get_tend',pbl_call)
            compare_held(session,pbl_held,pbl=True)
            from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
            from woof.hex.cuda_physics_v841 import _CUDA_SOURCE
            start=_CUDA_SOURCE.index('extern "C" __global__ void project_edges_v841_f32(')
            end=_CUDA_SOURCE.index('extern "C" __global__ void nonnegative_qv_copy_v841_f32(',start)
            projection=HostModule('mpas_v842_projection',CUDA_FTZ_HELPERS+_CUDA_SOURCE[start:end]).get_function('project_edges_v841_f32')
            g=session.resident_coupling.geometry; expected_edge=np.empty((nz,ne),np.float32); invalid=np.zeros(1,np.int32)
            projection(((ne+127)//128,),(128,),(nz,nc,ne,g.cells_on_edge.to_numpy(),g.east_cell.to_numpy(),
                g.north_cell.to_numpy(),g.edge_normal_vectors.to_numpy(),
                session.pools.field('tend_physics','rublten').to_numpy(),session.pools.field('tend_physics','rvblten').to_numpy(),
                np.ones((nz,ne),np.float32),expected_edge,invalid))
            if not np.array_equal(expected_edge.view(np.uint32),session.pools.field('tend_physics','rublten_Edge').to_numpy().view(np.uint32)):
                raise AssertionError('resident PBL edge projection differs from selected source')
            saved_geometry=session.transaction.checkpoint()
            session.pools.field('mesh','edgeNormalVectors').fill(0)
            session.transaction.restore(saved_geometry)
            if session.resident_coupling.geometry is not None:
                raise AssertionError('coupling retained stale geometry after pool restore')
            replay=active_hooks.invoke('physics_get_tend',pbl_call)
            compare_held(session,replay,pbl=True)
            if not np.array_equal(expected_edge.view(np.uint32),session.pools.field('tend_physics','rublten_Edge').to_numpy().view(np.uint32)):
                raise AssertionError('PBL projection changed after geometry restore')
            dummy_cells=valid_cells.copy(); dummy_cells[0,0]=0; dummy_cells[1,1]=dimensions['nCells']+1
            session.pools.field('mesh','cellsOnEdge')[...] = dummy_cells
            session.resident_coupling.restore_restart_state({})
            dummy_held=active_hooks.invoke('physics_get_tend',pbl_call)
            compare_held(session,dummy_held,pbl=True)
            g=session.resident_coupling.geometry
            du=np.pad(session.pools.field('tend_physics','rublten').to_numpy(),((0,0),(0,1)))
            dv=np.pad(session.pools.field('tend_physics','rvblten').to_numpy(),((0,0),(0,1)))
            dummy_edge=np.empty((nz,ne),np.float32)
            projection(((ne+127)//128,),(128,),(nz,nc+1,ne,g.cells_on_edge.to_numpy(),g.east_cell.to_numpy(),
                g.north_cell.to_numpy(),g.edge_normal_vectors.to_numpy(),du,dv,np.ones((nz,ne),np.float32),dummy_edge,invalid))
            if not np.array_equal(dummy_edge.view(np.uint32),session.pools.field('tend_physics','rublten_Edge').to_numpy().view(np.uint32)):
                raise AssertionError('native dummy-cell projection differs from selected source')
            # Actual driver ordering and registry dispatch, with unrelated glue
            # isolated; no claim is made for the remaining physics slots.
            from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
            slot_config=resolve_namelist("&physics\nconfig_physics_suite='none'\nconfig_sfclayer_scheme='sf_mynn'\nconfig_gwdo_scheme='bl_ysu_gwdo'\n/\n")
            slot_config=replace(slot_config,backends={**slot_config.backends,'surface_layer':'metal','gwdo':'metal'})
            metal_registry=MetalSlotRegistry()
            session.register_slots(metal_registry)
            resident_hooks=session.bind_hooks(external={},save_fields=lambda:{},restore_fields=lambda x:None,after_call=lambda e,c:None)
            class IsolatedGlue:
                def invoke(self, operation, call):
                    if operation == 'MPAS_to_physics' or operation.startswith(('allocate_','deallocate_')):
                        return resident_hooks.invoke(operation,call)
            slot_driver=PhysicsDriver(slot_config,PhysicsManager(slot_config,2),metal_registry,IsolatedGlue())
            slot_driver.initialize(session)
            for step in (1,2):
                slot_driver.begin_step(itimestep=step,time_seconds=(step-1)*2,pools=session)
                executed=[(x['slot'],x['backend']) for x in slot_driver.calls if x['slot'] in ('surface_layer','gwdo')]
                if executed != [('surface_layer','metal'),('gwdo','metal')]:
                    raise AssertionError(f'Metal driver dispatch differs: {executed}')
                slot_driver.physics_get_tend(rk_step=1,dynamics_substep=1,pools=session)
                slot_driver.finish_step(pools=session)
            print('real driver dispatched resident Metal SF and GWDO across two allocation lifetimes')
            session.close()
    print('real Registry hydration, nonzero multi-level Metal snapshot and restore match')

if __name__=='__main__': main()
