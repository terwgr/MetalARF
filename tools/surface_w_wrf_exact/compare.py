"""Compare the actual surface-w caller with selected host-compiled CUDA source."""
import argparse
import hashlib
from itertools import product
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.surface_w_wrf_exact.generate import retained_source
from tools.smag_wrf_exact.compare import compare_words


def run(arrays, xp, metal, mapped, bx, by):
    from woof.core.dycore import set_w_surface
    a = {key: xp.asarray(value) for key, value in arrays.items()}
    state = SimpleNamespace(**a, _metal_setup_state=metal,
                            cf1=np.float32(1.8286586), cf2=np.float32(-1.1352139),
                            cf3=np.float32(.3065553), has_msf=mapped)
    if metal:
        state._array_module = xp
    cfg = SimpleNamespace(dx=3., dy=7., open_x=bx, open_y=by,
                          specified=False, nested=False)
    set_w_surface(state, cfg)
    return state.w.to_numpy() if metal else np.asarray(state.w).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        raise ValueError('GPUWM_WRF_EXACT must be enabled before import')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    source = '\n'.join(retained_source())
    host_source = re.sub(r'\b(bx|by)\b', lambda m: 'boundary_' + m.group(), source)
    module = host_backend.HostModule('wrf_exact_surface_w', host_source)
    from woof.core import dycore
    original = dycore.get_kernel
    dycore.get_kernel = lambda unit, name, *args: (module.get_function(name)
        if unit == 'surface_w' and name == 'set_surface_w' else original(unit, name, *args))
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    rng = np.random.default_rng(773)
    nz, ny, nx = 4, 7, 8
    shapes = {'u': (nz, ny, nx + 1), 'v': (nz, ny + 1, nx),
              'ht': (ny, nx), 'msft': (ny, nx), 'w': (nz + 1, ny, nx)}
    report = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'cases': {}}
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for variant in ('normal', 'subnormal_terrain', 'subnormal_wind', 'flat'):
            arrays = {key: rng.uniform(-1, 1, shape).astype(np.float32)
                      for key, shape in shapes.items()}
            arrays['msft'] = rng.uniform(.8, 1.2, shapes['msft']).astype(np.float32)
            if variant == 'subnormal_terrain':
                arrays['ht'] *= np.float32(2**-128)
            elif variant == 'subnormal_wind':
                arrays['u'] *= np.float32(2**-128)
                arrays['v'] *= np.float32(2**-128)
            elif variant == 'flat':
                arrays['ht'].fill(0)
            for mapped, bx, by in product((False, True), repeat=3):
                expected = run(arrays, np, False, mapped, bx, by)
                actual = run(arrays, xp, True, mapped, bx, by)
                name = f'{variant}_map{int(mapped)}_x{int(bx)}y{int(by)}'
                report['cases'][name] = compare_words(actual, expected)
                print(name, json.dumps(report['cases'][name]), flush=True)
    report['words'] = sum(x['words'] for x in report['cases'].values())
    report['different_words'] = sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('words', 'different_words')}))
    if report['different_words']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
