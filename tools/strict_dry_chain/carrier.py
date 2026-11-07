"""Check the prepared runtime theta-carrier caller against binary32 expressions."""
import argparse
import ctypes
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.smag_wrf_exact.compare import compare_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_theta_perturbation
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    from woof.runtime import _hold_strict_theta_carrier
    from woof.verify.cases.igw import sounding

    cfg = RunConfig(nx=8, ny=7, nz=5, dx=1000., dy=1000., ztop=5000., dt=1., run_seconds=2.)
    source = '''extern "C" void reference(const float *b, const float *t, float *out, int n, int plane, int base3d) {
        for (int i=0; i<n; ++i) { volatile float total = b[base3d ? i : i/plane] + t[i];
            volatile float result = total - 300.0f; out[i] = result; }
    }'''
    flags = ['-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off', '-shared']
    cases = {}
    with tempfile.TemporaryDirectory() as directory:
        unit, lib = Path(directory)/'reference.cpp', Path(directory)/'reference.dylib'
        unit.write_text(source)
        subprocess.run(['clang++', *flags, str(unit), '-o', str(lib)], check=True)
        dll = ctypes.CDLL(str(lib)); ref = dll.reference
        ptr = ctypes.POINTER(ctypes.c_float)
        ref.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        def address(a): return a.ctypes.data_as(ptr)
        with MetalContext() as context:
            context.batch_kernel_calls = False
            submit = context._submit_kernel_batch
            def sync(*a, **kw):
                result = submit(*a, **kw); result.wait(); return result
            context._submit_kernel_batch = sync
            xp = MetalArrayModule(context)
            for base3d in (False, True):
                cfg = replace(cfg, terrain_opt=int(base3d))
                coord = make_vertical_coord(cfg.nz)
                base = make_base_state(coord, sounding, p_surf=cfg.p_surf, ztop=cfg.ztop,
                    terrain_z=np.zeros((cfg.ny, cfg.nx)) if base3d else None)
                state = init_theta_perturbation(cfg, coord, base,
                    lambda x,z: np.zeros((cfg.nz,cfg.ny,cfg.nx)), array_module=xp)
                original_base = state.thb.to_numpy()
                perturb = np.linspace(-1., 1., state.thp.size, dtype=np.float32).reshape(state.thp.shape)
                perturb.flat[:4] = np.array([1, 0x80000001, 0x007fffff, 0x807fffff], dtype=np.uint32).view(np.float32)
                state.thp[...] = perturb
                expected = np.empty_like(perturb)
                ref(address(original_base), address(perturb), address(expected), perturb.size,
                    cfg.ny*cfg.nx, int(base3d))
                prepared = SimpleNamespace(initial_result=SimpleNamespace(state=state))
                assert _hold_strict_theta_carrier(prepared) is prepared
                actual = state.thp.to_numpy()
                cases[f'base3d_{base3d}'] = compare_words(actual, expected)
                assert np.array_equal(state._setup_thb.view(np.uint32), original_base.view(np.uint32))
                assert np.all(state.thb.to_numpy() == np.float32(300.))
                _hold_strict_theta_carrier(prepared)
                cases[f'idempotent_{base3d}'] = compare_words(state.thp.to_numpy(), actual)
                assert np.array_equal(state._setup_thb.view(np.uint32), original_base.view(np.uint32))
    result = {'cases': cases, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'compiler_flags': flags,
              'different_words': sum(c['different_words'] for c in cases.values())}
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if result['different_words']: raise SystemExit(1)


if __name__ == '__main__':
    main()
