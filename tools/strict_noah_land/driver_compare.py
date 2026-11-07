"""Compare the real Noah driver method against its selected host source calls."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/thompson_real_column_parity'))
import host_backend
from tests.physics_reference.noah_oracle import load_noah_oracle, FIXTURE_DT, FIXTURE_DZS
from woof.core import kernels, noah_forcing, noah_sfcdiags
from woof.core import constants as c
from woof.core.noah import _F2D, _F3D, SEAICE_F2D, load_tables, pack_params
from woof.core.physics import PhysicsDriver
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED
assert ENABLED
host = host_backend.HostModule('strict_noah_driver', '#define __launch_bounds__(...)\n' + kernels.module_source('noah'))
land = host.get_function('noah_column')
ice = host.get_function('noah_seaice_column')
forcing = host_backend.HostModule('strict_noah_forcing_driver', noah_forcing._SOURCE)
prologue, clear = (forcing.get_function(n) for n in ('noah_forcing', 'noah_rain_clear'))
diags = host_backend.HostModule('strict_noah_diags_driver', noah_sfcdiags._source()).get_function('noah_sfcdiags')
fixture = load_noah_oracle('noah-lsm.csv'); indices = [0, 42, 55, 67]
ref = {'ivgtyp': np.ascontiguousarray(fixture.ivgtyp[:, indices]),
       'isltyp': np.ascontiguousarray(fixture.isltyp[:, indices])}
ref.update({n: np.ascontiguousarray(v[..., indices]) for n, v in fixture.inputs.items()})
shape = (1, len(indices)); n = shape[1]
for name in ('noahres', 'reslin', 'chklowq'):
    ref[name] = np.zeros(shape, np.float32)
ref['smcrel'] = np.zeros((4, *shape), np.float32); ref['ebal'] = np.zeros(shape, np.int32)
# One prescribed sea-ice column, with a valid snow-depth pair.
ref['xice'][0, -1] = 1.; ref['snow'][0, -1] = 0.; ref['snowh'][0, -1] = 0.
for name in ('q2', 't2', 'th2'):
    ref[name] = np.zeros(shape, np.float32)
ice_initial = ref['tslb'][:, 0, -1].copy()
ref['br'] = ref['rib'].copy(); pending = np.full(shape, .0001, np.float32)
atmos = {'p_interface': np.stack([ref['sfcprs'] + 500, ref['sfcprs'] - 500]),
         'temperature': ref['sfctmp'][None].copy(), 'qv': ref['qv1'][None].copy(),
         'dz': ref['dz8w1'][None].copy()}
p = pack_params(load_tables())
tables = [np.asarray(p.veg, np.float32).reshape(p.lucats, 15), np.asarray(p.soil, np.float32).reshape(p.slcats, 10), np.asarray(p.gen, np.float32), np.asarray(FIXTURE_DZS, np.float32)]
words = diff = 0
with MetalContext() as context:
    xp = MetalArrayModule(context); dev = {name: xp.asarray(value.copy()) for name, value in ref.items()}
    atmosphere = {name: xp.asarray(value) for name, value in atmos.items()}
    driver = PhysicsDriver.__new__(PhysicsDriver)
    driver.state = SimpleNamespace(_metal_setup_state=True, p=atmosphere['temperature'])
    driver.fields = dev; driver.mp_physics = 0; driver.urban_coupler = None
    driver._pending_rainbl = xp.asarray(pending.copy()); driver.noah_params = p
    driver.bldt_seconds = driver.noah_seaice_dt = float(FIXTURE_DT)
    driver.noah_seaice_albedo_default = .65
    driver.noah_usemonalb = driver.noah_rdlai2d = False; driver.noah_opt_thcnd = 1
    cfg = SimpleNamespace(sf_surface_mosaic=0, sf_urban_physics=0)
    for option in ('mosaic', 'urban', 'ice'):
        bad_cfg = SimpleNamespace(sf_surface_mosaic=int(option == 'mosaic'), sf_urban_physics=int(option == 'urban'))
        if option == 'ice': driver.fields = dict(dev, ice_volume=dev['tsk'])
        before = dev['rainbl'].to_numpy().copy()
        try: driver._run_noah(atmosphere, bad_cfg, 1)
        except NotImplementedError: pass
        else: raise AssertionError(f'{option} was not refused')
        if not np.array_equal(before.view(np.uint32), dev['rainbl'].to_numpy().view(np.uint32)): raise AssertionError(f'{option} mutated rain')
        driver.fields = dev
    for step in range(2, 5):
        args = [atmos['p_interface'][0], atmos['p_interface'][1], atmos['temperature'][0], atmos['qv'][0], atmos['dz'][0], ref['br'], atmos['temperature'][0], np.int32(0), np.float32(273.15), pending] + [ref[x] for x in ('sfcprs','sfctmp','qv1','dz8w1','rib','sr','rainbl')] + [np.int64(n)]
        prologue((1,), (64,), args)
        args = [ref['ivgtyp'], ref['isltyp']] + [ref[x] for x in _F2D] + [ref[x] for x in _F3D] + [ref['ebal']] + tables
        args += [np.float32(FIXTURE_DT), np.int32(p.lucats), np.int32(p.slcats), np.int32(13), np.int32(15), np.float32(.5), np.int32(step), np.int32(0), np.int32(0), np.int32(0), np.int32(1), np.int32(*shape[:1]), np.int32(shape[1])]
        land((1,), (64,), args)
        fatal = np.zeros(1, np.int32)
        ice((1,), (64,), [ref[x] for x in SEAICE_F2D] + [ref['tslb'], fatal, np.float32(FIXTURE_DT), np.float32(.5), np.int32(0), np.float32(.65), np.float32(3.), np.float32(.001), np.float32(1.e10), np.int32(shape[0]), np.int32(shape[1])])
        if fatal[0]: raise AssertionError('source sea-ice snow-density fatal')
        clear((1,), (64,), [ref['rainbl'], pending, np.int64(n)])
        driver._run_noah(atmosphere, cfg, step)
        diags((1,), (64,), [ref[x] for x in ('psfc','tsk','cqs2','chs2','qsfc','qfx','hfx','qv1','q2','t2','th2')] + [np.float32(x) for x in (c.RD, c.CP, c.P0, c.RCP, 1.e-5)] + [np.int64(n)])
        driver._refresh_surface_diagnostics(atmosphere)
        for name in (*_F2D, *_F3D, 'ebal', 'q2', 't2', 'th2'):
            actual = dev[name].to_numpy(); expected = ref[name]
            if actual.dtype == np.float32 and (not np.isfinite(actual).all() or not np.isfinite(expected).all()): raise AssertionError(f'nonfinite {name}')
            bad = np.flatnonzero(actual.view(np.uint32).ravel() != expected.view(np.uint32).ravel())
            words += actual.size; diff += len(bad)
            if len(bad): print(f'step={step} field={name} differences={len(bad)}')
        if np.any(driver._pending_rainbl.to_numpy().view(np.uint32) != pending.view(np.uint32)): raise AssertionError('pending rain differs')
    if np.array_equal(ice_initial.view(np.uint32), ref['tslb'][:, 0, -1].view(np.uint32)):
        raise AssertionError('prescribed sea-ice state did not advance')
    print(f'words={words} different_words={diff} calls=3 prescribed_ice_columns=1 guard_checks=3 reference=selected-CUDA-source-compiled-on-host')
if diff: raise SystemExit(1)
