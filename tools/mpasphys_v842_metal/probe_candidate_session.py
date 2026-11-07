"""Bind one captured public dycore candidate to a hydrated native session.

Framework scheme callbacks remain isolated. Native wind reconstruction and core
carriers run on Metal; selected-source comparisons read the results afterward.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
from copy import copy
import numpy as np
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager,SlotRegistry,Backend
from woof.hex.mpasphys.framework_backend_v842 import registry_source
from woof.hex.mpasphys.framework_init_v842 import FrameworkInitialization
from woof.hex.mpasphys.metal_session_v842 import MetalPhysicsSession
from woof.hex.mpasphys.pool_v842 import RegistryLayout


def exercise(driver,current,carried,outer,raw,held,candidate_names,expected_base,expected_after,expected_rate,expected_q,*,wind_geometry):
    current=copy(current)
    current.scalars=current.scalars.copy()
    region=driver.regional_v841
    nc=region.n_cells_solve if region is not None else driver.ncells
    ne=region.n_edges_solve if region is not None else driver.nedges
    mesh=driver.atmosphere.mesh
    dims={'nCells':nc,'nEdges':ne,'nVertices':int(mesh.n_vertices),'nVertLevels':int(driver.nlev),
          'nSoilLevels':9,'maxEdges':int(mesh.max_edges),'maxEdges2':int(mesh.max_edges2),
          'vertexDegree':int(mesh.vertex_degree),'nCellsSolve':nc,'nEdgesSolve':ne,
          'nVerticesSolve':int(mesh.n_vertices)}
    text="&physics\nconfig_physics_suite='convection_permitting'\nconfig_microp_scheme='mp_thompson_aerosols'\n/\n"
    config=resolve_namelist(text); layout=RegistryLayout(registry_source(),config,text,dims)
    native_names=tuple(next(r['components'] for r in layout.fields if r['pool']=='state' and r['name']=='scalars'))
    mapping=[candidate_names.index(n) for n in native_names]
    dry=resolve_namelist("&physics\nconfig_physics_suite='none'\n/\n")
    class DryHooks:
        def invoke(self, operation, call): return None
        def restart_state(self): return {}
        def restore_restart_state(self, state): pass
    def factory(session): return PhysicsDriver(dry,PhysicsManager(dry,float(driver.config.config_dt)),SlotRegistry(),DryHooks())
    with TemporaryDirectory(prefix='mpas-candidate-pool-') as td:
        root=Path(td); records=[]
        for i,row in enumerate(layout.fields):
            record={k:row[k] for k in ('pool','name','kind','dimensions','shape')}
            if row['kind']=='character': record['text']=''
            elif row['pool']=='mesh' and row['name'] in ('indexToCellID','indexToEdgeID','indexToVertexID'):
                a=np.arange(1,np.prod(row['shape'])+1,dtype='<i4').reshape(row['shape'],order='F')
                path=root/f'{i}.bin'; data=a.tobytes(order='F'); path.write_bytes(data)
                record.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            elif (row['pool'],row['name'])==('mesh','nominalMinDc'):
                a=np.full(row['shape'],120000.,dtype='<f4',order='F')
                path=root/f'{i}.bin'; data=a.tobytes(order='F'); path.write_bytes(data)
                record.update(payload=path.name,sha256=hashlib.sha256(data).hexdigest())
            else: record['fill']=0
            records.append(record)
        (root/'manifest.json').write_text(json.dumps({'schema':'mpas-native-pool-payload-v1','fields':records}))
        session=MetalPhysicsSession(layout=layout,payload=root,context=current.rho.buffer.context,dt=float(driver.config.config_dt),
            start='2024-02-28_00:00:00',stop='2024-02-28_01:00:00',driver_factory=factory,config=config)
        FrameworkInitialization(session).physics_run_init(None)
        session.native._initialized=True  # Scheme initialization is outside this transport-only probe.
        session.initialize()
        p=session.pools
        active=resolve_namelist("&physics\nconfig_physics_suite='none'\nconfig_sfclayer_scheme='sf_mynn'\nconfig_convection_scheme='cu_grell_freitas'\n/\n")
        registry=SlotRegistry(); registry.register('surface_layer','sf_mynn',Backend.NATIVE_CPU,lambda call:None)
        registry.register('convection','cu_grell_freitas',Backend.NATIVE_CPU,lambda call:None)
        class Hooks(DryHooks):
            def invoke(self,operation,call):
                if operation=='post_transport_prepare': return session.resident_transport.prepare(call)
                return None
        native_driver=PhysicsDriver(active,PhysicsManager(active,session.dt),registry,Hooks())
        native_driver.initialize(session); native_driver.begin_step(itimestep=1,time_seconds=0,pools=session)
        from types import SimpleNamespace
        from woof.hex.mpasphys.metal_native_coupling_v842 import MetalNativeCoupling
        from woof.hex.mpasphys.metal_winds_v842 import MetalMpasWinds
        geometry=wind_geometry
        wind_solver=MetalMpasWinds(geometry)
        reconstructed=[]
        def reconstruct(atmosphere):
            result=wind_solver(atmosphere)
            reconstructed.append(result)
            return result
        adapter=MetalNativeCoupling(session,candidate_names,reconstruct_winds=reconstruct)
        def synchronize(state,saved):
            atmosphere=SimpleNamespace(state=state,saved=saved,vertical=driver.atmosphere.vertical,
                                       reference=driver.atmosphere.reference)
            adapter.synchronize(atmosphere,level=2)
        session.resident_transport.bind(current=current,carried=carried,outer=outer,rate=raw,held=held,
            scalar_advection=True,monotonic=True,positive_definite=False,convection='cu_grell_freitas',
            scalar_names=candidate_names,synchronize=synchronize)
        native_driver.finish_step(pools=session)
        winds=reconstructed[-1]
        # The native v8.4.2 compiler binds this shared kernel's trig calls to
        # its pinned GNU graph. Generic v8.4.1 raw CUDA uses another contract.
        from tools.thompson_real_column_parity.host_backend import HostModule
        from woof.hex.cuda_physics_prep_v841 import _CUDA_SOURCE
        from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
        from woof.hex.mpasphys.math_v842 import bind_common_carrier_math
        start=_CUDA_SOURCE.index('extern "C" __global__ void reconstruct_winds_v841_f32(')
        end=_CUDA_SOURCE.index('extern "C" __global__ void prep_noahmp_sounding_v841_f32(',start)
        source=bind_common_carrier_math(CUDA_FTZ_HELPERS+'\n'+_CUDA_SOURCE[start:end])
        wind_host=HostModule('mpas_v842_public_candidate_winds',source).get_function('reconstruct_winds_v841_f32')
        shape=(driver.nlev,geometry.n_cells)
        components=[np.empty(shape,np.float32) for _ in range(5)]
        invalid=np.zeros(1,np.int32)
        geo=[getattr(geometry,'_'+name).to_numpy() for name in
             ('edges_on_cell','n_edges_on_cell','coeffs_reconstruct','lat_cell','lon_cell')]
        wind_host(((geometry.n_cells+127)//128,),(128,),
            (np.int32(driver.nlev),np.int32(geometry.n_cells),np.int32(geometry.n_edges),
             np.int32(geometry.max_edges),carried.normal_velocity.to_numpy(),*geo,
             *[a.copy() for a in geo],*components,invalid))
        assert not invalid[0]
        if geometry.regional_garbage_cell is not None:
            garbage=geometry.regional_garbage_cell
            donor=garbage-1 if geometry.regional_garbage_donor_cell is None else int(geometry.regional_garbage_donor_cell)
            for value in components[-2:]:value[:,garbage]=value[:,donor]
        for value,want in zip((winds.u_p,winds.v_p),components[-2:]):
            assert value.to_numpy().tobytes()==want.tobytes(),'native pinned wind source differs'
        print('native public candidate wind source words',sum(a.size for a in components[-2:]),flush=True)
        if not np.array_equal(current.scalars.to_numpy().view(np.uint32),expected_after.view(np.uint32)):
            raise AssertionError('public candidate scalar handoff differs, including dummy column')
        expected_native=expected_after[mapping,:,:nc]
        for label,expected,actual in (
            ('baseline',expected_base[mapping,:,:nc],p.field('state','scalars',level=1)),
            ('current',expected_native,p.field('state','scalars',level=2)),
            ('theta rate',expected_rate[:,:nc],p.field('tend_physics','rthdynten')),
            ('water rate',expected_q[:,:nc],p.field('tend_physics','rqvdynten'))):
            if not np.array_equal(np.ascontiguousarray(expected).view(np.uint32),actual.to_numpy().view(np.uint32)):
                raise AssertionError(f'public candidate native {label} differs from selected source')
        # Downloads here are independent verification, after device publication.
        from tools.thompson_real_column_parity.host_backend import HostModule
        from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
        from woof.hex.cuda_physics_prep_v841 import RV_OVER_RD_F32
        from woof.hex.mpasphys.diagnostics_v842 import native_diagnostic_source
        host=HostModule('mpas_v842_native_candidate_diagnostics',CUDA_FTZ_HELPERS+native_diagnostic_source())
        fields=(('state','rho_zz',2,current.rho),('state','theta_m',2,carried.theta_m),
                ('state','u',2,carried.normal_velocity),('state','w',2,carried.vertical_velocity),
                ('diag','rho_p',1,carried.density_perturbation),('diag','rtheta_p',1,carried.rho_theta_perturbation),
                ('diag','pressure_p',1,carried.pressure_perturbation),('diag','exner',1,carried.exner),
                ('diag','uReconstructZonal',1,winds.u_p),('diag','uReconstructMeridional',1,winds.v_p))
        for pool,name,level,value in fields:
            expected=value.to_numpy()[...,:ne if name=='u' else nc]
            actual=p.field(pool,name,level=level).to_numpy()
            if not np.array_equal(np.ascontiguousarray(expected).view(np.uint32),actual.view(np.uint32)):
                raise AssertionError(f'device native carrier differs: {pool}.{name}')
        inputs=[carried.theta_m.to_numpy(),expected_after[candidate_names.index('qv')],
                current.rho.to_numpy(),driver.atmosphere.vertical.zz.to_numpy(),
                driver.atmosphere.reference.pressure_base.to_numpy(),carried.pressure_perturbation.to_numpy()]
        outputs=[np.empty_like(inputs[0]) for _ in range(3)];count=inputs[0].size
        host.get_function('mpas_output_diagnostics_v842')(((count+127)//128,),(128,),
            (np.int32(count),RV_OVER_RD_F32,*inputs,*outputs))
        for name,expected in zip(('theta','rho','pressure'),outputs):
            actual=p.field('diag',name).to_numpy()
            if not np.array_equal(np.ascontiguousarray(expected[:,:nc]).view(np.uint32),actual.view(np.uint32)):
                raise AssertionError(f'native source diagnostic differs: {name}')
        # Run the source cold-EOS operator on private copies of this actual
        # candidate; the forecast's carried arrays remain untouched.
        from woof.hex.mpasphys.metal_diagnostics_v842 import initialize_thermodynamics
        source_saved=SimpleNamespace(rho_theta_perturbation=carried.rho_theta_perturbation,
            exner=carried.exner.copy(),pressure_perturbation=carried.pressure_perturbation.copy())
        source_reference=SimpleNamespace(rho_theta_base=driver.atmosphere.reference.rho_theta_base,
            exner_base=driver.atmosphere.reference.exner_base.copy(),
            pressure_base=driver.atmosphere.reference.pressure_base.copy())
        eos=SimpleNamespace(saved=source_saved,reference=source_reference,vertical=driver.atmosphere.vertical)
        eos_inputs=[eos.vertical.zz.to_numpy(),source_saved.rho_theta_perturbation.to_numpy(),
                    source_reference.rho_theta_base.to_numpy()]
        eos_outputs=[source_saved.exner.to_numpy(),source_reference.exner_base.to_numpy(),
                     source_saved.pressure_perturbation.to_numpy(),source_reference.pressure_base.to_numpy()]
        count=eos_inputs[0].size;rgas=np.float32(287);cp=np.float32(1004.5)
        host.get_function('mpas_cold_thermodynamics_v842')(((count+127)//128,),(128,),
            (np.int32(count),rgas,np.float32(1e5),np.float32(rgas/np.float32(cp-rgas)),*eos_inputs,*eos_outputs))
        initialize_thermodynamics(eos)
        for name,value,expected in zip(('exner','exner_base','pressure_p','pressure_base'),
                (source_saved.exner,source_reference.exner_base,source_saved.pressure_perturbation,source_reference.pressure_base),eos_outputs):
            if not np.array_equal(value.to_numpy().view(np.uint32),expected.view(np.uint32)):
                raise AssertionError(f'public candidate cold EOS differs from source: {name}')
        # A failed wind callback must precede any destination pool writes.
        before=p.field('state','rho_zz',level=2).to_numpy()
        def refuse(atmosphere):raise ValueError('source wind unavailable')
        blocked=MetalNativeCoupling(session,candidate_names,reconstruct_winds=refuse)
        try:blocked.synchronize(SimpleNamespace(state=current,saved=carried,vertical=driver.atmosphere.vertical,
                                                 reference=driver.atmosphere.reference),level=2)
        except ValueError as error:
            if str(error)!='source wind unavailable':raise
        else:raise AssertionError('missing wind did not refuse coupling')
        if not np.array_equal(before.view(np.uint32),p.field('state','rho_zz',level=2).to_numpy().view(np.uint32)):
            raise AssertionError('missing wind mutated native carrier')
        adapter.synchronize(SimpleNamespace(state=outer,saved=carried,vertical=driver.atmosphere.vertical,
                                             reference=driver.atmosphere.reference),level=1)
        if not np.array_equal(np.ascontiguousarray(outer.rho.to_numpy()[:,:nc]).view(np.uint32),
                              p.field('state','rho_zz',level=1).to_numpy().view(np.uint32)):
            raise AssertionError('native time-level-one carrier differs')
        session.close()
    print('public candidate reached native finish with device-written core carriers and source diagnostics; pinned-source native winds')
