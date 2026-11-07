"""Exercise MPAS_to_physics on hydrated pools against host-selected kernels."""
import ctypes
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.driver_v842 import Backend,SlotCall,StepFlags
from woof.hex.mpasphys.framework_init_v842 import FrameworkInitialization
from woof.hex.mpasphys.resident_v842 import preparation_source


def exercise(session):
    p=session.pools; n=session.native; d=session.layout.dimensions
    nz,nc=d['nVertLevels'],d['nCellsSolve']; rng=np.random.default_rng(8427)
    def put(pool,name,value): p.field(pool,name)[...] = np.asarray(value,np.float32)
    shape=(nz,nc); levels=np.arange(nz,dtype=np.float32)[:,None]
    inputs={('state','rho_zz'):rng.uniform(.7,1.3,shape).astype(np.float32),
            ('state','theta_m'):rng.uniform(280,320,shape).astype(np.float32),
            ('diag','exner'):rng.uniform(.8,1.,shape).astype(np.float32),
            ('diag','pressure_base'):rng.uniform(70000,100000,shape).astype(np.float32),
            ('diag','pressure_p'):rng.uniform(-700,700,shape).astype(np.float32),
            ('mesh','zz'):rng.uniform(.8,1.2,shape).astype(np.float32),
            ('diag','uReconstructZonal'):rng.uniform(-5,15,shape).astype(np.float32),
            ('diag','uReconstructMeridional'):rng.uniform(-5,5,shape).astype(np.float32),
            ('mesh','latCell'):rng.uniform(-1,1,(nc,)).astype(np.float32),
            ('state','w'):rng.uniform(-1,1,(nz+1,nc)).astype(np.float32),
            ('mesh','zgrid'):np.broadcast_to(np.arange(nz+1,dtype=np.float32)[:,None]*150,(nz+1,nc)).copy()}
    names=session.resident_preparation.names
    inputs['state','scalars']=rng.uniform(0,.004,(len(names),nz,nc)).astype(np.float32)
    for key,value in inputs.items(): put(*key,value)
    FrameworkInitialization(session).physics_run_init(None)
    for name in ('nc','ni','nifa','nwfa'):
        n.put_variable('f_'+name,np.array(name in names,np.int32))
    n._initialized=True # The test isolates preparation; scheme init is not claimed.
    call=SlotCall('MPAS_to_physics','driver',Backend.NATIVE_CPU,1,0.,2.,StepFlags(),session,session.config.execution_options)
    hooks=session.bind_hooks(external={},save_fields=lambda:{},restore_fields=lambda x:None,after_call=lambda e,c:None)
    flag=hooks.invoke('MPAS_to_physics',call)
    if int(flag.to_numpy()[0]): raise AssertionError('preparation validation flag')
    src=HostModule('mpas_v842_preparation',preparation_source())
    mass_kernel=src.get_function('prep_mass_v841_f32'); interface_kernel=src.get_function('prep_interfaces_v841_f32')
    mass={name:np.empty(shape,np.float32) for name in ('zz_p','rho_dry','rho_p','th_p','t_p','pi_p','pres_p','zmid_p','dz_p')}
    qout=np.empty((len(names),nz,nc),np.float32); invalid=np.zeros(1,np.int32)
    ptrs=(ctypes.c_void_p*len(names))(*(qout[s].ctypes.data for s in range(len(names))))
    args=[nz,nc,len(names),np.float32(n.get_constant('r_v')/n.get_constant('r_d')),
          *(inputs[x] for x in (('state','rho_zz'),('state','theta_m'),('state','scalars'),('diag','exner'),('diag','pressure_base'),('diag','pressure_p'),('mesh','zgrid'),('mesh','zz'))),None,
          *(mass[x] for x in ('zz_p','rho_dry','rho_p','th_p','t_p','pi_p','pres_p','zmid_p','dz_p')),invalid]
    converted=[ctypes.c_void_p(x.ctypes.data) if isinstance(x,np.ndarray) else ctypes.cast(ptrs,ctypes.c_void_p) if x is None else x for x in args]
    mass_kernel._fn(1,1,1,128,1,1,*converted)
    for name,value in mass.items():
        if name=='rho_dry': actual=session.resident_preparation.rho_dry.to_numpy()
        else: actual=p.prepared(name)[:nz].to_numpy()
        if not np.array_equal(actual.view(np.uint32),value.view(np.uint32)): raise AssertionError(f'resident mass {name} differs')
    interfaces=('w_p','z_p','t2_p','pres2_p','pres2_hyd_p','pres2_hydd_p','weights_minus','weights_plus')
    mass_fields=('pres_hyd_p','pres_hydd_p','znu_p','znu_hyd_p')
    surfaces=('psfc_p','psfc_hyd_p','psfc_hydd_p','plrad')
    outputs={name:np.zeros((nz+1,nc),np.float32) for name in interfaces}
    outputs.update({name:np.zeros(shape,np.float32) for name in mass_fields})
    outputs.update({name:np.zeros(nc,np.float32) for name in surfaces})
    interface_kernel((1,),(128,),(nz,nc,np.float32(n.get_constant('gravity')),
        inputs['state','rho_zz'],inputs['mesh','zz'],qout[0],mass['pres_p'],mass['t_p'],mass['rho_p'],mass['dz_p'],
        inputs['mesh','zgrid'],inputs['state','w'],*(outputs[x] for x in ('w_p','z_p','t2_p','pres2_p','pres_hyd_p','pres2_hyd_p','pres_hydd_p','pres2_hydd_p','psfc_p','psfc_hyd_p','psfc_hydd_p','znu_p','znu_hyd_p','plrad')),
        invalid,inputs['diag','pressure_p'],inputs['diag','pressure_base'],outputs['weights_minus'],outputs['weights_plus']))
    for name in interfaces+mass_fields+surfaces:
        target={'weights_minus':'fzm_p','weights_plus':'fzp_p'}.get(name,name)
        actual=p.field('diag_physics','plrad').to_numpy() if name=='plrad' else p.prepared(target).to_numpy()
        expected=outputs[name]
        # Native prepared mass arrays include one interface-level allocation.
        if name in mass_fields: actual=actual[:nz]
        if name in ('weights_minus','weights_plus'):
            actual=actual[1:nz]; expected=expected[1:nz]
        if not np.array_equal(actual.view(np.uint32),expected.view(np.uint32)): raise AssertionError(f'resident interface {name} differs: {np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32))} words; max {np.max(np.abs(actual-expected))}; actual {actual.ravel()[:8]}, expected {expected.ravel()[:8]}')
    for target,source in (('u_p','uReconstructZonal'),('v_p','uReconstructMeridional')):
        if not np.array_equal(p.prepared(target)[:nz].to_numpy(),inputs['diag',source]): raise AssertionError(f'{target} differs')
    lat=inputs['mesh','latCell']/np.float32(n.get_constant('degrad'))
    if not np.array_equal(p.prepared('xlat_p').to_numpy().view(np.uint32),lat.view(np.uint32)):
        raise AssertionError('resident latitude differs')
    print('hydrated native MPAS_to_physics matches host-selected mass/interface and wind/latitude carriers')


def exercise_surface(session):
    from dataclasses import replace
    from woof.hex.physics_suites import load_table
    from woof.hex.mpasphys.metal_sf_mynn_v842 import MetalMpasSfMynn,CONSTANTS,OUTPUTS,_DIAG_INPUT,_DIAG_NAMES,_PROFILE,_SEA_HANDOFF
    from tools.mpasphys_v842_sf_mynn.compare_columns import fixture
    from woof.hex.mpasphys.sf_mynn_v842 import FIELDS
    p=session.pools; n=session.native; nz=session.layout.dimensions['nVertLevels']; nc=session.layout.dimensions['nCellsSolve']
    context=p.arrays.context; xp=p.arrays
    initial,_=fixture(nc); get=lambda name:initial[FIELDS.index(name)].copy()
    diag_names=set(_DIAG_INPUT)|{'hpbl'}|{_DIAG_NAMES.get(x,x) for x in OUTPUTS if x not in ('wstar','qstar')}
    diag={name:p.field('diag_physics',name) for name in diag_names}
    for name in _DIAG_INPUT: diag[name][...] = get(name)
    diag['hpbl'][...] = get('pblh')
    for name,value in {'skintemp':get('tsk'),'xland':get('xland'),'snowh':get('snowh'),
                       'sst':np.linspace(265,300,nc,dtype=np.float32),
                       'xice':np.linspace(0,1,nc,dtype=np.float32)}.items(): p.field('sfc_input',name)[...] = value
    p.field('mesh','meshDensity')[...] = np.linspace(.7,1.4,nc,dtype=np.float32)
    session.layout.configs['config_len_disp']=3000.
    session.layout.configs['config_do_restart']=False
    n.put_variable('xice_threshold',np.array(.1,np.float32))
    desc=load_table()['mpasphys_v842']['slots']['surface_layer']['cuda_entries']['sf_mynn']
    constants={name:n.get_constant(desc['constant_aliases'].get(name,name)) for name in CONSTANTS}
    profiles={name:p.prepared(source)[:nz] for name,source in desc['profiles'].items()}
    expected_diag={name:value.copy() for name,value in diag.items()}
    expected_prepared={'psfc_p':p.prepared('psfc_p'),**{name+'_sea':xp.zeros((nc,),np.float32) for name in _SEA_HANDOFF}}
    class Surface:
        def __getitem__(self,name): return p.field('sfc_input',name)
    pools={'mesh':{'meshDensity':p.field('mesh','meshDensity')},'diag_physics':expected_diag,
           'sfc_input':Surface(),'atm_input':profiles,'constants':constants,'configs':session.layout.configs,
           'scalar_state':{'xice_threshold':.1},'prepared':expected_prepared}
    call=SlotCall('surface_layer','sf_mynn',Backend.NATIVE_CPU,1,0.,2.,StepFlags(),pools,
                  {'config_frac_seaice':True})
    MetalMpasSfMynn(context).launch_slot(call)
    session.resident_preparation.surface_layer(replace(call,pools=session))
    for name in OUTPUTS:
        if name in ('wstar','qstar'): continue
        key=_DIAG_NAMES.get(name,name)
        if not np.array_equal(diag[key].to_numpy().view(np.uint32),expected_diag[key].to_numpy().view(np.uint32)):
            raise AssertionError(f'resident sf_mynn mapping differs: {key}')
    for name in _SEA_HANDOFF:
        if not np.array_equal(p.prepared(name+'_sea').to_numpy().view(np.uint32),expected_prepared[name+'_sea'].to_numpy().view(np.uint32)):
            raise AssertionError(f'resident sf_mynn sea carrier differs: {name}')
    print('resident prepared MYNN surface binding and nine sea carriers match direct slot')


def exercise_gwdo(session):
    from dataclasses import replace
    from woof.hex.mpasphys.metal_gwdo_v842 import run_gwdo
    p=session.pools; native=session.native
    nz=session.layout.dimensions['nVertLevels']; nc=session.layout.dimensions['nCellsSolve']
    z=np.arange(nz,dtype=np.float32)[:,None]
    broadcast=lambda x:np.broadcast_to(x,(nz,nc)).copy().astype(np.float32)
    for name,values in {'u_p':broadcast(8+z*.3),'v_p':broadcast(3+z*.2),'t_p':broadcast(290-z*4),
                        'qv_p':broadcast(.008-z*.0001),'pres_hydd_p':broadcast(95000-z*7000),
                        'pi_p':broadcast(.99-z*.03),'zmid_p':broadcast(150+z*350)}.items():
        p.prepared(name)[:nz][...] = values
    p.prepared('pres2_hydd_p')[...] = np.broadcast_to(98500-np.arange(nz+1,dtype=np.float32)[:,None]*7000,(nz+1,nc)).copy()
    p.field('mesh','meshDensity')[...] = np.linspace(.7,1.3,nc,dtype=np.float32)
    for name,value in {'var2d':300.,'con':.5,**{x:.2 for x in ('oa1','oa2','oa3','oa4')},
                       **{x:.5 for x in ('ol1','ol2','ol3','ol4')}}.items():
        p.field('sfc_input',name).fill(value)
    p.field('tend_physics','rublten').fill(.001); p.field('tend_physics','rvblten').fill(-.001)
    names=('dtaux3d','dtauy3d','rubldiff','rvbldiff','dusfcg','dvsfcg')
    diag={name:p.field('diag_physics',name) for name in names}
    expected_diag={name:value.copy() for name,value in diag.items()}
    expected_tend={name:p.field('tend_physics',name).copy() for name in ('rublten','rvblten')}
    class Pool:
        def __init__(self,name): self.name=name
        def __getitem__(self,name): return p.field(self.name,name)
    profiles={name:p.prepared(name) for name in ('u_p','v_p','t_p','qv_p','pres_hydd_p','pres2_hydd_p','pi_p','zmid_p')}
    constants={key:native.get_constant(source) for key,source in (('gravity','gravity'),('cp','cp'),('rd','r_d'),('fv','ep_1'),('pi','pii'))}
    run_gwdo(prepared=profiles,mesh=Pool('mesh'),sfc_input=Pool('sfc_input'),tend_physics=expected_tend,
             diag_physics=expected_diag,dt_pbl=float(native.get_variable('dt_pbl')[0]),
             config_len_disp=session.layout.configs['config_len_disp'],constants=constants)
    call=SlotCall('gwdo','bl_ysu_gwdo',Backend.NATIVE_CPU,1,0.,2.,StepFlags(),session,session.config.execution_options)
    session.resident_preparation.gwdo(call)
    for name,expected in expected_diag.items():
        if not np.array_equal(diag[name].to_numpy().view(np.uint32),expected.to_numpy().view(np.uint32)):
            raise AssertionError(f'resident GWDO diagnostic mapping differs: {name}')
    for name,expected in expected_tend.items():
        if not np.array_equal(p.field('tend_physics',name).to_numpy().view(np.uint32),expected.to_numpy().view(np.uint32)):
            raise AssertionError(f'resident GWDO tendency mapping differs: {name}')
    print('resident prepared GWDO binding matches direct source-proven slot')
