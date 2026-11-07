"""Compare actual advection launchers with the selected host-compiled CUDA source.

Run separately with each WOOF_WRF_EXACT_ADVECTION control. This is a host
source comparison, not an NVIDIA-device comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.advection_wrf_exact.generate import KERNELS, retained_source
from tools.smag_wrf_exact.compare import compare_words


def inputs(scale, ny=8):
    nz, nx = 8, 9
    rng = np.random.default_rng(184)
    shapes = {'scalar': (nz, ny, nx), 'u': (nz, ny, nx + 1),
              'v': (nz, ny + 1, nx), 'w': (nz + 1, ny, nx),
              'ru': (nz, ny, nx + 1), 'rv': (nz, ny + 1, nx),
              'rw': (nz + 1, ny, nx)}
    arrays = {key: (rng.standard_normal(shape).astype(np.float32) * scale)
              for key, shape in shapes.items()}
    # Keep velocities normal when testing subnormal transported operands.
    if scale < np.float32(1e-30):
        for key in ('ru', 'rv', 'rw'):
            arrays[key] = rng.uniform(-1, 1, shapes[key]).astype(np.float32)
    arrays.update(rdnw=np.linspace(-.3, -.6, nz, dtype=np.float32),
                  rdn=np.linspace(-.3, -.6, nz, dtype=np.float32),
                  fnm=np.full(nz, .43, np.float32),
                  fnp=np.full(nz, .57, np.float32))
    return arrays


def run(arrays, xp, metal, field, order, bx, by, spec, map_factor):
    from woof.core import advection
    a = {key: xp.asarray(value) for key, value in arrays.items()}
    target = a[field]
    tend = xp.asarray(np.full(target.shape, np.float32(2**-128), dtype=np.float32))
    coord = SimpleNamespace(**{key: a[key] for key in ('rdnw', 'rdn', 'fnm', 'fnp')})
    msf = xp.asarray(np.full(target.shape[1:], .875, np.float32)) if map_factor else None
    getattr(advection, 'launch_flux_div_' + field)(
        target, a['ru'], a['rv'], a['rw'], tend, coord, 3., 7.,
        open_x=bx, open_y=by, spec=spec, msf=msf, vorder=order)
    return tend.to_numpy() if metal else np.asarray(tend).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ny', type=int, choices=(1, 8), default=8)
    focus = ap.add_mutually_exclusive_group()
    focus.add_argument('--vertical-five-focused', action='store_true')
    focus.add_argument('--momentum-five-focused', action='store_true')
    args = ap.parse_args()
    focused = args.vertical_five_focused or args.momentum_five_focused
    from woof.wrf_exact import ENABLED, ADVECTION_ENABLED
    if not ENABLED:
        raise ValueError('GPUWM_WRF_EXACT must be enabled before import')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    source = '\n'.join(retained_source(ADVECTION_ENABLED))
    module = host_backend.HostModule('wrf_exact_advection', source)
    from woof.core import advection
    original = advection.get_kernel
    advection.get_kernel = lambda unit, name, *args: (module.get_function(name)
        if unit == 'advection' and name in KERNELS else original(unit, name, *args))
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    report = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'advection_control': bool(ADVECTION_ENABLED), 'cases': {}}
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for scale_name, scale in [('normal', np.float32(1)), ('subnormal', np.float32(2**-128))]:
            arrays = inputs(scale, args.ny)
            for field, order, boundary, mapped in product(
                    (('u', 'v') if args.momentum_five_focused else ('scalar', 'w') if args.vertical_five_focused else ('scalar', 'u', 'v', 'w')),
                    ((5,) if focused else (3, 5)),
                    (((0, 0, 0), (1, 1, 1)) if focused else
                     (((0, 0, 0),) if args.ny == 1 else
                      ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)))),
                    ((True,) if focused else (False, True))):
                bx, by, spec = boundary
                expected = run(arrays, np, False, field, order, bx, by, spec, mapped)
                actual = run(arrays, xp, True, field, order, bx, by, spec, mapped)
                name = f'{scale_name}_{field}_v{order}_x{bx}y{by}s{spec}_map{int(mapped)}'
                report['cases'][name] = compare_words(actual, expected)
                print(name, json.dumps(report['cases'][name]), flush=True)
    if focused and not any(x['subnormal_words'] for name,x in report['cases'].items() if name.startswith('subnormal_')):
        raise AssertionError('no subnormal vertical outputs retained')
    report['words'] = sum(x['words'] for x in report['cases'].values())
    report['different_words'] = sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('words', 'different_words')}))
    if report['different_words']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
