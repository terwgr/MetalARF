"""Compare the typed strict Noah urban handoff with selected host source."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/thompson_real_column_parity'))
import host_backend
from tests.physics_reference.noah_oracle import load_noah_oracle, FIXTURE_DT, FIXTURE_DZS
from woof.core import kernels
from woof.core.noah import _F2D, _F3D, load_tables, pack_params
from woof.metal.noah_family import submit_noah_column
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED
assert ENABLED
source = '#define __launch_bounds__(...)\n' + kernels.module_source('noah')
kernel = host_backend.HostModule('strict_noah_urban', source).get_function('noah_column_urban')
params = pack_params(load_tables())
fixture = load_noah_oracle('noah-lsm.csv')
ref = {'ivgtyp': fixture.ivgtyp[..., :5].copy(), 'isltyp': fixture.isltyp[..., :5].copy(),
       **{k: v[..., :5].copy() for k, v in fixture.inputs.items()}}
shape = ref['ivgtyp'].shape
ref['ivgtyp'][0] = [13, 13, 13, 13, 10]
for name in ('noahres', 'reslin', 'chklowq'):
    ref[name] = np.zeros(shape, np.float32)
ref['smcrel'] = np.zeros((4, *shape), np.float32)
ref['ebal'] = np.zeros(shape, np.int32)
mask = np.zeros(32, np.int32); mask[13] = 1
urban = {'frc_urb2d': np.array([[.2, .75, .99, 1, .7]], np.float32),
         'ts_urb2d': np.full(shape, 290, np.float32),
         'tsk_rural_bep': np.zeros(shape, np.float32),
         'rural_q1': np.full(shape, -1, np.float32),
         'rural_q2k': np.full(shape, -2, np.float32),
         'rural_zlvl': np.full(shape, -3, np.float32)}
tables = [np.asarray(params.veg, np.float32).reshape(params.lucats, 15),
          np.asarray(params.soil, np.float32).reshape(params.slcats, 10),
          np.asarray(params.gen, np.float32), np.asarray(FIXTURE_DZS, np.float32)]
with MetalContext() as context:
    xp = MetalArrayModule(context)
    dev = {key: xp.asarray(value.copy()) for key, value in ref.items()}
    ud = {key: xp.asarray(value.copy()) for key, value in urban.items()}
    ud.update(option=1, natural=7, category_mask=xp.asarray(mask))
    initial = {name: value.to_numpy().view(np.uint32).copy()
               for name, value in dev.items()}
    for natural in (0, params.lucats + 1):
        with patch.object(context, 'submit_kernel') as submit, \
             patch.object(context, 'submit_bookkeeping') as bookkeeping:
            try:
                submit_noah_column(dev, params, FIXTURE_DT, FIXTURE_DZS,
                                   urban=dict(ud, natural=natural))
            except ValueError as error:
                assert 'natural category' in str(error)
            else:
                raise AssertionError(f'invalid natural category {natural} accepted')
            assert submit.call_count == bookkeeping.call_count == 0
        for name, before in initial.items():
            assert np.array_equal(dev[name].to_numpy().view(np.uint32), before), name
    print('invalid natural categories refused before submission and output changes')
    for step in (1, 2):
        args = [ref['ivgtyp'], ref['isltyp']] + [ref[n] for n in _F2D] + [ref[n] for n in _F3D] + [ref['ebal']] + tables
        args += [np.float32(FIXTURE_DT), np.int32(params.lucats), np.int32(params.slcats), np.int32(13), np.int32(15), np.float32(.5), np.int32(step), np.int32(0), np.int32(0), np.int32(0), np.int32(1), np.int32(shape[0]), np.int32(shape[1])]
        args += [np.int32(1), mask, np.int32(mask.size), np.int32(7)] + [urban[n] for n in ('frc_urb2d', 'ts_urb2d', 'tsk_rural_bep', 'rural_q1', 'rural_q2k', 'rural_zlvl')]
        kernel((1,), (64,), args)
        submit_noah_column(dev, params, FIXTURE_DT, FIXTURE_DZS, itimestep=step, urban=ud)
        different = count = 0
        for name in (*_F2D, *_F3D, 'ebal', 'rural_q1', 'rural_q2k', 'rural_zlvl'):
            expected = urban[name] if name.startswith('rural_') else ref[name]
            actual = (ud[name] if name.startswith('rural_') else dev[name]).to_numpy()
            a, b = actual.view(np.uint32), expected.view(np.uint32)
            bad = np.flatnonzero(a.ravel() != b.ravel()); count += a.size; different += len(bad)
            if len(bad):
                i = bad[0]; print(name, 'differences', len(bad), 'first', i, hex(int(b.ravel()[i])), hex(int(a.ravel()[i])))
        print(f'step {step}: {different}/{count} word differences; reference=selected CUDA source compiled on host')
        if different: raise SystemExit(1)
