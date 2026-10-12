import json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
    result=original(self,*args,**kwargs)
    if result is not None: result.wait()
    return result
MetalContext._submit_kernel_batch=sync
from woof.metal.array import MetalArrayModule
from woof.metal.uwpbl_family import submit_uwpbl_columns,MASS_INPUTS,FULL_INPUTS,SURFACE_INPUTS
from tests.physics_reference.uwpbl_oracle import load,dims,step_arrays
from tests.physics_reference.uwpbl_ref.driver import camuwpbl_step
module=MetalArrayModule(); results=[]
try:
    fixture=load(ROOT/'tests/data/uwpbl/oracle/cases-g44')
    ncol,nk,nsteps=dims(fixture)
    for step in range(1,nsteps+1):
        a=step_arrays(fixture,step)
        ref=camuwpbl_step(a)
        def mass(name): return module.asarray(np.ascontiguousarray(a[name].T[:,None,:],np.float32))
        def surface(name): return module.asarray(np.ascontiguousarray(a[name][None,:],np.float32))
        inputs={name:mass(name) for name in MASS_INPUTS+FULL_INPUTS}
        inputs.update({name:surface(name) for name in SURFACE_INPUTS})
        carried={name:mass(name+'_in') for name in ['kvm3d','kvh3d']}
        carried.update({name:surface(name+'_in') for name in ['tauresx2d','tauresy2d']})
        got=submit_uwpbl_columns(module.context,inputs,carried,dt=float(a['dt']),itimestep=int(a['itimestep']))
        errors={}
        for name in ['kpbl2d','turbtype3d','kvm3d','kvh3d']:
            array=(carried[name] if name in carried else got[name]).to_numpy()
            array=array[0] if array.ndim==2 else array[:,0,:].T
            assert np.isfinite(array).all()
            if name in ['kpbl2d','turbtype3d']:
                np.testing.assert_array_equal(array,ref[name])
                np.testing.assert_array_equal(array,a[name])
            else:
                np.testing.assert_allclose(array,ref[name],rtol=0,atol=0.00013)
            errors[name]=float(np.max(np.abs(array-ref[name])))
        results.append(dict(step=step,columns=ncol,levels=nk,errors=errors))
    print(json.dumps(dict(reference='Darwin NumPy plus recorded WRF discrete diagnostics',steps=results,synchronous=True)))
finally: module.context.close()
