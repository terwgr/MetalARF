import dataclasses, json, pathlib, sys
from types import SimpleNamespace
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from woof.metal.suites import configuration, initialize, execute
from tools.metal_combo_smoke import GateModule, GatesResolved
from tools.metal_admission_check import dycore_config_refusals
from woof.core.physics import _metal_admission_refusals
from woof.core.dycore import step
cfg = dataclasses.replace(configuration((0,5,0,0,0,'periodic'),sfclay_option=5),km_opt=2,tke_budget=1)
state = SimpleNamespace(_metal_setup_state=True,_array_module=GateModule(),
    p=SimpleNamespace(shape=(10,8,8)),mup=SimpleNamespace(shape=(8,8)),
    phb=SimpleNamespace(ndim=1),qv=object(),p_top=20000.)
def refusal(call):
    try: call()
    except (ValueError,NotImplementedError) as error:
        assert 'tke_budget' in str(error),str(error)
        return str(error)
    raise AssertionError('unsupported budget was admitted')
init = refusal(lambda:initialize(state,cfg,gates_only=True))
direct = refusal(lambda:step(state,cfg))
assert any('tke_budget' in r for r in dycore_config_refusals(cfg))
supported = dataclasses.replace(cfg,tke_budget=0)
assert not any('tke_budget' in r for r in dycore_config_refusals(supported))
assert not any('tke_budget' in r for r in _metal_admission_refusals(supported,radiation=None,cumulus=None,cam_ozone=None))
try: initialize(state,supported,gates_only=True)
except GatesResolved: pass
else: raise AssertionError('supported initialization did not reach allocation')
from woof.metal.runtime import MetalContext
original = MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
    result = original(self,*args,**kwargs)
    if result is not None: result.wait()
    return result
MetalContext._submit_kernel_batch=sync
outcome = execute((0,0,0,0,0,'periodic'),steps=1,turbulence='tke-1.5')
assert outcome[0]=='ok',outcome
print(json.dumps(dict(initialization=init,direct_step=direct,supported=outcome,inspector=True,synchronous=True)))
