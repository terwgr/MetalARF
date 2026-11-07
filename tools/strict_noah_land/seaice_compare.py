"""Compare provisional strict sea-ice columns to the selected host source."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/thompson_real_column_parity'))
import host_backend
from tools.metal_noah_seaice_word_probe import columns
from woof.core import kernels
from woof.core.noah import SEAICE_F2D
from woof.metal.array import MetalArrayModule
from woof.metal.noah_family import submit_noah_seaice
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED
assert ENABLED
source='#define __launch_bounds__(...)\n'+kernels.module_source('noah')
refkernel=host_backend.HostModule('strict_noah_seaice',source).get_function('noah_seaice_column')
words=diff=active=0
with MetalContext() as context:
    xp=MetalArrayModule(context)
    for seed in (0,1):
        ny,nx=8,8; ref=columns(np.random.default_rng(seed),ny,nx)
        dev={name:xp.asarray(value.copy()) for name,value in ref.items()}
        caller={name:xp.asarray(value.copy()) for name,value in ref.items()}
        fatal=np.zeros(1,np.int32);devfatal=xp.asarray(fatal.copy())
        dt=np.float32(60.);frozen=np.int32(seed)
        args=[ref[n] for n in SEAICE_F2D]+[ref['tslb'],fatal,dt,np.float32(.5),frozen,np.float32(.65),np.float32(3.),np.float32(.001),np.float32(1.e10),np.int32(ny),np.int32(nx)]
        refkernel((1,),(64,),args)
        arrays={name:dev[name] for name in SEAICE_F2D};arrays.update(tslb=dev['tslb'],fatal=devfatal)
        scalars={'dt':float(dt),'xice_threshold':.5,'frpcpn':int(frozen),'albedo_default':.65,'sithick':3.,'snowdepth_min':.001,'snowdepth_max':1.e10,'ny':ny,'nx':nx}
        context.submit_kernel('wrf_exact_noah_seaice',arrays,scalars,exact_policy=True,grid_size=ny*nx,threads_per_group=64)
        submit_noah_seaice(caller,float(dt),frpcpn=bool(frozen),xice_threshold=.5,
                           albedo_default=.65,check_snow_density=False)
        for name in (*SEAICE_F2D,'tslb'):
            expected=ref[name].view(np.uint32).ravel();actual=dev[name].to_numpy().view(np.uint32).ravel();bad=np.flatnonzero(expected!=actual)
            words+=expected.size;diff+=len(bad)
            if len(bad):print(f'seed={seed} {name} differences={len(bad)} first={bad[0]} host=0x{int(expected[bad[0]]):08x} metal=0x{int(actual[bad[0]]):08x}')
            caller_words=caller[name].to_numpy().view(np.uint32).ravel()
            caller_bad=np.flatnonzero(expected!=caller_words);words+=expected.size;diff+=len(caller_bad)
            if len(caller_bad):print(f'seed={seed} caller {name} differences={len(caller_bad)}')
        words+=1;diff+=int(fatal[0]!=devfatal.to_numpy()[0]);active+=int(np.count_nonzero(ref['xice']>=.5))
print(f'words={words} different_words={diff} active_columns={active} reference=selected-CUDA-source-compiled-on-host')
if diff:raise SystemExit(1)
