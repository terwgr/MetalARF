"""Compare strict dry-tendency caller paths against their FP32 references."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.mp28_metal.parse import closure, split_top
from tools.smag_wrf_exact.compare import compare_words


def source():
    from woof.core.kernels import module_source
    code = '#define GPUWM_WRF_EXACT 1\n' + module_source('bandwidth_glue')
    expanded = subprocess.run(['clang++', '-E', '-P', '-std=c++17', '-x', 'c++', '-'],
                              input=code, text=True, capture_output=True, check=True).stdout
    selected, _ = closure(split_top(expanded), ('glue_capture_theta_forcing',))
    return '\n'.join(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        raise ValueError('GPUWM_WRF_EXACT must be enabled before import')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    retained = source()
    host = host_backend.HostModule('wrf_exact_theta_capture',
        'struct alignas(16) float4 { float x,y,z,w; };\n' + retained).get_function('glue_capture_theta_forcing')
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    from woof.metal.bandwidth_family import accumulate_scaled_array
    from woof.core import dycore
    report = {'source_sha256': hashlib.sha256(retained.encode()).hexdigest(), 'cases': {}}
    rng = np.random.default_rng(416)
    nz, ny, nx = 4, 5, 7
    shape, cols = (nz, ny, nx), (ny, nx)
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for subnormal in (False, True):
            rth = rng.uniform(-1, 1, shape).astype(np.float32)
            if subnormal: rth *= np.float32(2**-128)
            mub = rng.uniform(.75, 1.2, cols).astype(np.float32)
            mup = rng.uniform(-.1, .1, cols).astype(np.float32)
            c1 = np.array([.4, .6, .8, 1.], np.float32)
            c2 = np.array([.2, .3, .5, .7], np.float32)
            msf = rng.uniform(.8, 1.3, cols).astype(np.float32)
            for mapped in (False, True):
                a = {k: xp.asarray(v) for k, v in dict(rth_t=rth, mub2d=mub, mup=mup,
                                                        c1h=c1, c2h=c2, msft=msf).items()}
                dst = xp.asarray(np.zeros(shape, np.float32))
                state = SimpleNamespace(**a, rthften=dst, has_msf=mapped,
                                        _metal_setup_state=True, _array_module=xp)
                expected = np.zeros(shape, np.float32)
                n = expected.size
                host(((n + 511)//512,), (128,), (rth, mub, mup, c1, c2, msf, expected,
                                                 np.uint64(n), np.int32(ny*nx), np.int32(mapped)))
                state_type = dycore.DomainState
                dycore.DomainState = SimpleNamespace
                try:
                    dycore.capture_advective_theta_forcing(state, cfg=SimpleNamespace(cu_physics=0))
                finally:
                    dycore.DomainState = state_type
                key=f'capture_sub{int(subnormal)}_map{int(mapped)}'
                report['cases'][key] = compare_words(dst.to_numpy(), expected)
                dst = xp.asarray(np.zeros(shape, np.float32));state.rthften = dst
                dycore.capture_advective_theta_forcing(state, cfg=SimpleNamespace(cu_physics=3))
                expected = np.multiply(rth, msf[None], dtype=np.float32) if mapped else rth.copy()
                report['cases'][f'gf_sub{int(subnormal)}_map{int(mapped)}'] = compare_words(dst.to_numpy(), expected)
                # Exercise the real relaxation-fold caller and physics held-slot caller.
                init = rng.uniform(-1, 1, shape).astype(np.float32)
                if subnormal: init *= np.float32(2**-128)
                out = xp.asarray(init); state.scratch = lambda _shape, _slot: out
                spec = [(rth, out, None, None, None, 'smag_ru', 'x')]
                saved = (dycore._smag2d_specs, dycore.capture_folded_relaxation,
                         dycore._couple_dry_mixing_map_factor)
                dycore._smag2d_specs = lambda *a, **k: spec
                dycore.capture_folded_relaxation = lambda *a, **k: {'u': state.rth_t}
                dycore._couple_dry_mixing_map_factor = lambda *a, **k: None
                state.msfu = state.msft
                try:
                    dycore.fold_lateral_relaxation(state, SimpleNamespace())
                finally:
                    (dycore._smag2d_specs, dycore.capture_folded_relaxation,
                     dycore._couple_dry_mixing_map_factor) = saved
                addend = np.multiply(rth, msf[None], dtype=np.float32) if mapped else rth
                expected = np.add(init, addend, dtype=np.float32)
                report['cases'][f'relax_sub{int(subnormal)}_map{int(mapped)}'] = compare_words(out.to_numpy(), expected)
                out = xp.asarray(init);state.scratch = lambda _shape, _slot: out
                phys = SimpleNamespace(ru=state.rth_t, rv=None, rw=None, rtheta=None)
                dycore._fold_physics_wrf(state, spec, phys)
                report['cases'][f'fold_sub{int(subnormal)}_map{int(mapped)}'] = compare_words(out.to_numpy(), np.add(init, rth, dtype=np.float32))
                out = xp.asarray(init)
                accumulate_scaled_array(state.rth_t, out, np.float32(.75))
                expected = np.add(init, np.multiply(rth, np.float32(.75), dtype=np.float32), dtype=np.float32)
                report['cases'][f'scaled_sub{int(subnormal)}_map{int(mapped)}'] = compare_words(out.to_numpy(), expected)
    report['words'] = sum(x['words'] for x in report['cases'].values())
    report['different_words'] = sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if report['different_words']: raise SystemExit(1)


if __name__ == '__main__': main()
