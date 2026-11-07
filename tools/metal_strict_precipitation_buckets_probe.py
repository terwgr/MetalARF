"""Compare the two-window Metal caller with host-executed selected CUDA source."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from woof.wrf_exact import ENABLED, effective_options
from woof.core.kernels import module_source, module_options
from woof.core.output_diagnostics import PRECIP_HISTORY, boundary_due, initialize, update_precipitation
from woof.io.restart import SERIALIZED_SCRATCH_SLOTS, _scratch_manifest
from woof.io.wrf_output_schema import REGISTRY_VAR_META
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

if not ENABLED:
    raise SystemExit('GPUWM_WRF_EXACT=1 is required')
source = module_source('output_diagnostics')
shim = (ROOT / 'tools/thompson_real_column_parity/cuda_host_shim.h').read_text()
wrapper = '''\nextern "C" void run(const float*a,const float*b,const float*c,const float*d,const float*e,float*f,float*g,float*h,float*j,int n,int reset){blockDim.x=1;threadIdx.x=0;for(int k=0;k<n;++k){blockIdx.x=k;precipitation_buckets(a,b,c,d,e,f,g,h,j,n,reset);}}\n'''
class State:
    def __init__(self, xp, physics):
        self.xp, self.physics, self._scratch = xp, physics, {}
        self.elapsed_seconds = 0
    def scratch(self, shape, slot):
        if slot not in self._scratch:
            self._scratch[slot] = self.xp.zeros(shape, dtype=np.float32)
        return self._scratch[slot]
    def existing_scratch(self, slot):
        return self._scratch.get(slot)

with tempfile.TemporaryDirectory(prefix='strict-precip-source-') as tmp:
    tmp = Path(tmp)
    (tmp/'source.cpp').write_text(shim+'\n'+source+wrapper)
    subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(tmp/'source.cpp'),'-o',str(tmp/'source.dylib')],check=True)
    f = ctypes.CDLL(str(tmp/'source.dylib')).run
    fp = ctypes.POINTER(ctypes.c_float)
    f.argtypes = [fp]*9+[ctypes.c_int]*2
    ptr = lambda a: a.ctypes.data_as(fp)
    rng = np.random.default_rng(311)
    shape = (2, 64)
    inputs = [rng.uniform(-.01, .03, shape).astype(np.float32) for _ in range(4)]
    inputs.append(rng.uniform(260, 285, shape).astype(np.float32))
    for a in inputs[:4]:
        a.flat[:6] = [0., -0., np.nextafter(np.float32(0),np.float32(1)), -np.nextafter(np.float32(0),np.float32(1)), np.nan, np.inf]
    inputs[4].flat[:6] = [0., -0., 273.15, 273.14996, np.nan, np.inf]
    cfg = SimpleNamespace(ny=shape[0], nx=shape[1], prec_acc_dt=1., prec_acc_dt1=2., gsd_diagnostics=0, output_diagnostics=0, mp_physics=8)
    prefixes=('prec_acc_c','prec_acc_nc','snow_acc_nc','graup_acc_nc')
    names=[p+s for s in ('','1') for p in prefixes]
    assert set(names) <= SERIALIZED_SCRATCH_SLOTS
    for name in names:
        symbol, units, description = PRECIP_HISTORY[name]
        assert REGISTRY_VAR_META[symbol] == (description, units)
    with MetalContext() as context:
        context.batch_kernel_calls=False
        submit=context._submit_kernel_batch
        def sync(*args, **kwargs):
            result=submit(*args, **kwargs);result.wait();return result
        context._submit_kernel_batch=sync
        xp=MetalArrayModule(context)
        arrays=[xp.asarray(a) for a in inputs]
        physics=SimpleNamespace(cu_raincv=arrays[0], microphysics=SimpleNamespace(rainncv=arrays[1],snowncv=arrays[2],graupelncv=arrays[3]),fields={'t2':arrays[4]})
        state=State(xp,physics);initialize(state,cfg)
        refs={name:np.zeros(shape,np.float32) for name in names}
        differences=0; words=0
        for step, clock in enumerate((0.,30.,60.,90.,120.,150.)):
            state.elapsed_seconds=clock
            for suffix, minutes in (('',1.),('1',2.)):
                outputs=[refs[p+suffix] for p in prefixes]
                f(*map(ptr,inputs),*map(ptr,outputs),int(np.prod(shape)),int(boundary_due(clock,minutes*60)))
            update_precipitation(state,cfg)
            for name in names:
                got=state.existing_scratch(name).to_numpy()
                differences += int(np.count_nonzero(got.view(np.uint32)!=refs[name].view(np.uint32)))
                words += got.size
            if step == 2:
                manifest=_scratch_manifest(state)
                assert all('scratch/'+name in manifest for name in names)
                state2=State(xp,physics);initialize(state2,cfg)
                for name in names:
                    state2._scratch[name]=xp.asarray(state.existing_scratch(name).to_numpy())
                state=state2
        # Round-trip real DomainState checkpoint storage, then resume the caller
        # with the same synthetic producer inputs. No forecast physics is claimed.
        from woof.config import RunConfig
        from woof.core.grid import make_vertical_coord, make_base_state
        from woof.core.state import init_theta_perturbation
        from woof.io.restart import write_restart, restore_restart
        real_cfg=RunConfig(nx=shape[1],ny=shape[0],nz=5,dx=1000.,dy=1000.,ztop=5000.,dt=30.,run_seconds=180.,diff_opt=0,prec_acc_dt=1.,prec_acc_dt1=2.)
        coord=make_vertical_coord(real_cfg.nz,hybrid_opt=real_cfg.hybrid_opt,etac=real_cfg.etac)
        base=make_base_state(coord,lambda z:np.full_like(np.asarray(z,dtype=np.float64),300.),p_surf=real_cfg.p_surf,ztop=real_cfg.ztop)
        make_state=lambda: init_theta_perturbation(real_cfg,coord,base,lambda x,z:np.zeros((real_cfg.nz,real_cfg.ny,real_cfg.nx)),array_module=xp)
        actual=make_state()
        actual.physics=physics
        actual.elapsed_seconds=30.
        update_precipitation(actual,cfg)
        saved={name:actual.existing_scratch(name).to_numpy() for name in names}
        actual.physics=None
        path=tmp/'checkpoint.npz'
        write_restart(path,actual,real_cfg)
        resumed=make_state()
        restore_restart(path,resumed,real_cfg)
        assert resumed.elapsed_seconds == actual.elapsed_seconds
        for name in names:
            assert np.array_equal(saved[name].view(np.uint32),resumed.existing_scratch(name).to_numpy().view(np.uint32))
        actual.physics=physics;resumed.physics=physics
        actual.elapsed_seconds=resumed.elapsed_seconds=60.
        update_precipitation(actual,cfg);update_precipitation(resumed,cfg)
        for name in names:
            assert np.array_equal(actual.existing_scratch(name).to_numpy().view(np.uint32),resumed.existing_scratch(name).to_numpy().view(np.uint32))
        print(json.dumps({'source_sha256':hashlib.sha256(source.encode()).hexdigest(), 'effective_options':effective_options(module_options('output_diagnostics')), 'words':words,'differences':differences,'steps':6,'windows':2,'restored':'component scratch reconstruction','checkpoint_roundtrip':'real DomainState file, synthetic producers', 'checkpoint_manifest_slots':len(names),'history_symbols':len(names)}))
        if differences:
            raise SystemExit(1)
