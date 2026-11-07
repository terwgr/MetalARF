"""Compare the native post-LSM callback and actual Registry publication with selected source."""
from types import SimpleNamespace
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_metal.session_fixture import session_fixture
from woof.hex.mpasphys.sfclay_mm5_v842 import CUDA_SOURCE, _shared_math_source, FIELDS, CONSTANTS, _MPAS_NAMES
from woof.hex.mpasphys.metal_surface_diagnostics_v842 import surface_diagnostics
from woof.hex.mpasphys.driver_v842 import resolve_namelist
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
start=CUDA_SOURCE.index('// MPAS mpas_atmphys_sfc_diagnostics.F:79-91')
end=CUDA_SOURCE.index('// MPAS physics_wrf/module_sf_sfcdiags.F:7-69 SFCDIAGS',start)
source=re.search(r'struct MM5Constants \{.*?\};',CUDA_SOURCE).group()+'\n'+CUDA_SOURCE[start:end]
host=HostModule('native_surface_diagnostics',_shared_math_source()+source).get_function('mm5_sfc_diagnostics')
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    n=513;rng=np.random.default_rng(842)
    base=rng.uniform(.1,.8,(len(FIELDS),n)).astype(np.float32)
    values={'psfcpa':rng.uniform(30000,105000,n),'tsk':rng.uniform(220,310,n),
            'xland':np.resize([1,1.5,2],n),'qsfc':rng.uniform(.001,.02,n),
            'hfx':rng.uniform(-30,60,n),'qfx':rng.uniform(-.00001,.00001,n),
            'chs2':np.resize([0,1e-5,1e-4,1e-3],n),'cqs2':np.resize([1e-3,1e-4,1e-5,0],n)}
    for name,value in values.items():base[FIELDS.index(name)]=value
    # Positive/negative subnormal fluxes and threshold-adjacent exchange.
    for name in ('hfx','qfx'):
        base[FIELDS.index(name),:8]=np.array([1,2,0x80000001,0x80000002,0x007fffff,0x807fffff,0,0x80000000],np.uint32).view(np.float32)
    ice=np.resize(np.array([0,.02,.5,1],np.float32),n)
    q2=rng.uniform(.002,.01,n).astype(np.float32);t2=rng.uniform(225,305,n).astype(np.float32)
    constants=np.array([1004.5,9.81,287.04/1004.5,287.04,2.5e6,.6112,17.67,29.65,273.15,.608,.622,.4,100000],np.float32)
    for scheme,lsm in [('sf_noah',1),('sf_noahmp',2),('off',0)]:
        expected=base.copy();actual=xp.asarray(base)
        host(((n+127)//128,),(128,),(expected,ice,q2,t2,constants,np.int32(n),np.int32(lsm),np.float32(.02)))
        surface_diagnostics(actual,xp.asarray(constants),lsm_scheme=scheme,xice=xp.asarray(ice),xice_threshold=.02,q2mxy=xp.asarray(q2),t2mxy=xp.asarray(t2))
        got=actual.to_numpy();bad=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32))
        print(scheme,'packed words',expected.size,'differences',bad);assert not bad
    for scheme,lsm in [('sf_noah',1),('sf_noahmp',2)]:
        text="&physics\nconfig_physics_suite='convection_permitting'\nconfig_lsm_scheme='"+scheme+"'\n/\n";config=resolve_namelist(text)
        with session_fixture(context,config,text) as session:
            provider=session.slot_pools;count=provider['nCellsSolve'];packed=base[:,:count].copy()
            provider.field('diag','surface_pressure')[...]=packed[FIELDS.index('psfcpa')]
            for name in ('skintemp','xland','xice'):
                src={'skintemp':'tsk'}.get(name,name)
                provider['sfc_input'][name][...]=ice[:count] if name=='xice' else packed[FIELDS.index(src)]
            for name in ('hfx','qfx','qsfc','chs2','cqs2'):
                provider['diag_physics'][_MPAS_NAMES.get(name,name)][...]=packed[FIELDS.index(name)]
            aliases={'g':'gravity','rovcp':'rcp','r':'R_d','ep1':'ep_1','ep2':'ep_2','p1000mb':'P0'}
            native_constants=np.array([provider.native.get_constant(aliases.get(name,name)) for name in CONSTANTS],np.float32)
            threshold=provider['scalar_state']['xice_threshold']
            expected=packed.copy();host((1,),(128,),(expected,ice[:count],q2[:count],t2[:count],native_constants,np.int32(count),np.int32(lsm),np.float32(threshold)))
            if lsm==2:
                provider.field('output_noahmp','q2mxy')[...]=q2[:count]
                provider.field('output_noahmp','t2mxy')[...]=t2[:count]
            hooks=session.bind_hooks(external={},save_fields=lambda:{},restore_fields=lambda state:None,after_call=lambda *args:None)
            call=SimpleNamespace(slot='atmphys_sfc_diagnostics',options={'config_lsm_scheme':scheme})
            before=session.transaction.checkpoint();hooks.invoke('atmphys_sfc_diagnostics',call)
            once=session.transaction.checkpoint();session.transaction.restore(before);hooks.invoke('atmphys_sfc_diagnostics',call)
            twice=session.transaction.checkpoint()
            for name in ('q2','t2','th2'):
                got=provider['diag_physics'][_MPAS_NAMES.get(name,name)].to_numpy()
                assert np.array_equal(got.view(np.uint32),expected[FIELDS.index(name)].view(np.uint32)),name
            for name,value in once['pools'].items():assert value.tobytes()==twice['pools'][name].tobytes(),name
            print(scheme,'actual Registry callback and transaction replay matched; other schemes/preparation isolated')
