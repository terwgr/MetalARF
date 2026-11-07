"""Compare provisional strict Noah land column against selected host source."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/thompson_real_column_parity'))
import host_backend
from tests.physics_reference.noah_oracle import load_noah_oracle, FIXTURE_DT, FIXTURE_DZS
from woof.core import kernels
from woof.core.noah import _F2D, _F3D, load_tables, pack_params
from woof.metal.noah_family import _column_arrays
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED

assert ENABLED
source='#define __launch_bounds__(...)\n'+kernels.module_source('noah')
host=host_backend.HostModule('strict_noah_land',source)
k=host.get_function('noah_column')
params=pack_params(load_tables())
words=diff=finite_diff=nonfinite_diff=0
with MetalContext() as context:
    xp=MetalArrayModule(context)
    for fixture_name,glacial in [('noah-lsm.csv',False),('noah-lsm-frpcpn.csv',False),('noah-lsm.csv',True)]:
        fixture=load_noah_oracle(fixture_name)
        shape=fixture.ivgtyp.shape; ny,nx=shape
        ref={'ivgtyp':fixture.ivgtyp.copy(),'isltyp':fixture.isltyp.copy(),**{name:value.copy() for name,value in fixture.inputs.items()}}
        if glacial:ref['ivgtyp'][...]=15
        for name in ('noahres','reslin','chklowq'):ref[name]=np.zeros(shape,np.float32)
        ref['smcrel']=np.zeros((4,*shape),np.float32);ref['ebal']=np.zeros(shape,np.int32)
        dev={name:xp.asarray(value.copy()) for name,value in ref.items()}
        tables=[np.asarray(params.veg,dtype=np.float32).reshape(params.lucats,15),np.asarray(params.soil,dtype=np.float32).reshape(params.slcats,10),np.asarray(params.gen,dtype=np.float32),np.asarray(FIXTURE_DZS,dtype=np.float32)]
        arrays=_column_arrays(dev,tuple(xp.asarray(x) for x in tables))
        switches=fixture.switches
        scalars={'dt':FIXTURE_DT,'lucats':params.lucats,'slcats':params.slcats,'isurban':13,'isice':15,'xice_threshold':.5,'itimestep':2,'frpcpn':int(switches['frpcpn']),'usemonalb':int(switches['usemonalb']),'rdlai2d':int(switches['rdlai2d']),'opt_thcnd':switches['opt_thcnd'],'ny':ny,'nx':nx}
        args=[ref['ivgtyp'],ref['isltyp']]+[ref[name] for name in _F2D]+[ref[name] for name in _F3D]+[ref['ebal']]+tables
        args += [np.float32(scalars['dt'])]+[np.int32(scalars[x]) for x in ('lucats','slcats','isurban','isice')]+[np.float32(.5)]+[np.int32(scalars[x]) for x in ('itimestep','frpcpn','usemonalb','rdlai2d','opt_thcnd','ny','nx')]
        k(((ny*nx+63)//64,),(64,),args)
        context.submit_kernel('wrf_exact_noah_column',arrays,scalars,exact_policy=True,grid_size=ny*nx,threads_per_group=64)
        for name in (*_F2D,*_F3D,'ebal'):
            actual=dev[name].to_numpy(); expected=ref[name]; a=actual.view(np.uint32);b=expected.view(np.uint32)
            bad=np.flatnonzero(a.ravel()!=b.ravel());words+=a.size;diff+=len(bad)
            if actual.dtype==np.float32:
                finite=np.isfinite(actual.ravel()) & np.isfinite(expected.ravel())
                finite_diff+=int(np.count_nonzero(finite[bad]))
                nonfinite_diff+=int(np.count_nonzero(~finite[bad]))
            else:
                finite_diff+=len(bad)
            if len(bad):print(f'{fixture_name} glacial={glacial} {name}: {len(bad)} first={int(bad[0])} host=0x{int(b.ravel()[bad[0]]):08x} metal=0x{int(a.ravel()[bad[0]]):08x}')
        print(f'{fixture_name} glacial={glacial}: {ny*nx} columns')
print(f'words={words} different_words={diff} finite_differences={finite_diff} nonfinite_differences={nonfinite_diff} reference=selected-CUDA-source-compiled-on-host')
if diff:raise SystemExit(1)
