import csv,json,pathlib,sys
from types import SimpleNamespace
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
    result=original(self,*args,**kwargs)
    if result is not None: result.wait()
    return result
MetalContext._submit_kernel_batch=sync
from woof.metal.array import MetalArrayModule
from woof.core.nssl2_radiation import update_effective_radii,launch_effective_radius_concentration
module=MetalArrayModule()
try:
    with (ROOT/'tests/data/nssl2/oracle/effective-radius.csv').open() as source:
        rows=list(csv.DictReader(source))
    assert len(rows)==48
    def host(name): return np.array([float(row[name]) for row in rows],np.float32).reshape(3,4,4)
    density=module.asarray(host('rho_kg_m3'))
    mapping={'qc':'qc','qndrop':'nc_per_kg','qi':'qi','qni':'ni_per_kg','qs':'qs','qns':'ns_per_kg'}
    state=SimpleNamespace(**{name:module.asarray(host(column)) for name,column in mapping.items()},
        **{name:module.empty((3,4,4),np.float32) for name in ['effc','effi','effs']})
    before={name:getattr(state,name).to_numpy().copy() for name in mapping}
    assert update_effective_radii(state,air_density=density)
    errors={};metres=[module.empty((3,4,4),np.float32) for _ in range(3)]
    launch_effective_radius_concentration(density,state.qc,module.asarray(host('nc_per_kg')*host('rho_kg_m3')),
        state.qi,module.asarray(host('ni_per_kg')*host('rho_kg_m3')),state.qs,module.asarray(host('ns_per_kg')*host('rho_kg_m3')),*metres)
    for name,column,concentration,lo,hi in zip(['effc','effi','effs'],
        ['re_cloud_m','re_ice_m','re_snow_m'],metres,[2.51,10.01,25.],[50.,125.,999.]):
        got=getattr(state,name).to_numpy();want=host(column)*np.float32(1e6)
        np.testing.assert_allclose(got,want,rtol=2.5e-6,atol=2e-5)
        np.testing.assert_allclose(concentration.to_numpy(),host(column),rtol=2e-6,atol=2e-12)
        np.testing.assert_array_equal(got.view(np.uint32),
            (concentration.to_numpy()*np.float32(1e6)).view(np.uint32))
        assert got.min()>=np.nextafter(np.float32(lo),np.float32(-np.inf)) and got.max()<=np.nextafter(np.float32(hi),np.float32(np.inf)),(name,got.min(),got.max())
        errors[name]=float(np.max(np.abs(got-want)))
    saved={name:getattr(state,name).to_numpy().copy() for name in ['effc','effi','effs']}
    assert not update_effective_radii(state,radiation_due=False)
    for name,want in {**before,**saved}.items():
        np.testing.assert_array_equal(getattr(state,name).to_numpy().view(np.uint32),want.view(np.uint32))
    print(json.dumps(dict(cases=len(rows),oracle='official WRF effective-radius fixture',
        max_error_micrometres=errors,registry_vs_concentration_bitwise=True,
        unchanged_forecast_inputs=True,not_due_noop=True,synchronous=True)))
finally: module.context.close()
