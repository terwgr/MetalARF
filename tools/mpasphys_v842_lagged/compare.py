"""Compare native lagged/recovery operations to the selected CUDA source."""
from types import SimpleNamespace as NS
import os
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_driver import _CUDA_SOURCE, CUDA_V841_PHYSICS_DRIVER_SOURCE
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.native_powf import CUDA_NATIVE_POWF
from woof.hex.mpasphys.metal_lagged_v842 import add_lagged, zero_speczone, recover_cells
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def section(name):
    source_text = _CUDA_SOURCE if name == 'recover_cells_f32' else CUDA_V841_PHYSICS_DRIVER_SOURCE
    start = source_text.index('extern "C" __global__ void '+name+'(')
    end = source_text.index('extern "C" __global__ void ', start + 20)
    return source_text[start:end]

source = CUDA_FTZ_HELPERS + CUDA_NATIVE_POWF + '''
#define C2(k,c,nc) ((k)*(nc)+(c))
#define DRIVER_ELEMENT_LOOP(total) for (int element=blockDim.x*blockIdx.x+threadIdx.x; element<(total); element+=gridDim.x*blockDim.x)
''' + ''.join(section(name) for name in ('recover_cells_f32', 'add_lagged_diabatic_theta_v842_f32', 'zero_speczone_lagged_diabatic_theta_v842_f32'))
module = HostModule('native_lagged_recovery', source)
add = module.get_function('add_lagged_diabatic_theta_v842_f32')
zero = module.get_function('zero_speczone_lagged_diabatic_theta_v842_f32')
recover = module.get_function('recover_cells_f32')
nz,nc=4,7
rng=np.random.default_rng(841)
a=lambda lo,hi:rng.uniform(lo,hi,(nz,nc)).astype(np.float32)
rho=a(.7,1.3);diab=a(-.005,.005);initial=a(-1,1);cells=np.array([0,3],np.int32)
expected=initial.copy();add((1,),(128,),(nz,nc,rho,diab,expected))
zeroed=diab.copy();zero((1,),(128,),(nz,nc,len(cells),cells,zeroed))
saved_rho=a(-.05,.05);saved_theta=a(-3,3);rho_pp=a(-.05,.05);theta_pp=a(-2,2)
rho_base=a(.8,1.2);theta_base=a(290,310);exner_base=a(.8,1.1);zz=a(.8,1.1);exner_saved=a(.8,1.1);pressure_saved=a(-1,1)
if os.environ.get("LAGGED_HOST_ONLY"):
    outputs=[np.empty_like(rho) for _ in range(7)]
    recover((1,), (128,), (nz,nc,3,1,np.float32(287),np.float32(1004.5),np.float32(100000),saved_rho,saved_theta,rho_pp,theta_pp,1,np.float32(60),zeroed,rho_base,theta_base,exner_base,zz,exner_saved,pressure_saved,*outputs))
    assert all(np.all(np.isfinite(x)) for x in outputs)
    print("selected host source executed", len(outputs)*rho.size, "recovery words", flush=True)
    raise SystemExit(0)

with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync; xp=MetalArrayModule(context);up=lambda x:xp.asarray(x.copy())
    ref=NS(rho_base=up(rho_base),rho_theta_base=up(theta_base),exner_base=up(exner_base))
    driver=NS(nlev=nz,ncells=nc,arrays=xp,atmosphere=NS(reference=ref,vertical=NS(zz=up(zz))),regional_v841=NS(masks=NS(n_spec_cells=len(cells),spec_cells=up(cells))))
    metal_tend=up(initial);metal_diab=up(diab);add_lagged(driver,up(rho),metal_diab,metal_tend);zero_speczone(driver,metal_diab)
    def equal(name,host,device):
        actual=device.to_numpy().view(np.uint32);want=host.view(np.uint32)
        bad=np.flatnonzero(actual!=want)
        if bad.size:raise AssertionError((name,int(bad[0]),hex(int(want.flat[bad[0]])),hex(int(actual.flat[bad[0]])),len(bad)))
        print(name,'0/',want.size,flush=True)
    equal('add',expected,metal_tend);equal('zero',zeroed,metal_diab)
    saved=NS(density_perturbation=up(saved_rho),rho_theta_perturbation=up(saved_theta),exner=up(exner_saved),pressure_perturbation=up(pressure_saved))
    acoustic=NS(rho_pp=up(rho_pp),rtheta_pp=up(theta_pp))
    for stage,has in ((1,True),(3,True),(3,False)):
        outputs=[np.empty_like(rho) for _ in range(7)]
        recover((1,),(128,),(nz,nc,stage,1,np.float32(287),np.float32(1004.5),np.float32(100000),saved_rho,saved_theta,rho_pp,theta_pp,int(has),np.float32(60),zeroed,rho_base,theta_base,exner_base,zz,exner_saved,pressure_saved,*outputs))
        got=recover_cells(driver,saved,acoustic,stage,metal_diab if has else None,60)
        for i,(want,value) in enumerate(zip(outputs,got)):equal(f'recovery {stage}/{has}/{i}',want,value)
    ref.rho_theta_base=up(np.full((nz,nc),-300.,np.float32))
    try:recover_cells(driver,saved,acoustic,3,metal_diab,60)
    except ValueError as error:
        if 'Exner argument' not in str(error):raise
    else:raise AssertionError('invalid Exner argument accepted')
    print('invalid Exner argument refused',flush=True)
