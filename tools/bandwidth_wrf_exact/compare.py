"""Compare actual Metal glue-add callers with the selected CUDA source on host."""
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
    assembled = '#define GPUWM_WRF_EXACT 1\n' + module_source('bandwidth_glue')
    expanded = subprocess.run(['clang++', '-E', '-P', '-std=c++17', '-x', 'c++', '-'],
                              input=assembled, text=True, capture_output=True, check=True).stdout
    selected, _ = closure(split_top(expanded), ('glue_add',))
    return '\n'.join(selected)


def words(n, offset):
    patterns = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                         0x3f800000, 0xbf800000, 0x3f000000, 0x33800000], np.uint32)
    return np.resize(np.roll(patterns, offset), n).view(np.float32).copy()


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
    host_source = 'struct alignas(16) float4 { float x,y,z,w; };\n' + retained
    host = host_backend.HostModule('wrf_exact_glue_add', host_source).get_function('glue_add')
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    from woof.metal.bandwidth_family import add_array, prepare_add_arrays
    report = {'source_sha256': hashlib.sha256(retained.encode()).hexdigest(), 'cases': {}}
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for size in (1, 3, 129, 513):
            for alias in (False, True):
                src, dst = words(size, 2), words(size, 6)
                if alias:
                    dst = src.copy()
                expected = dst.copy()
                host(((size + 511) // 512,), (128,),
                     (expected if alias else src, expected, np.uint64(size)))
                out = xp.asarray(dst)
                inp = out if alias else xp.asarray(src)
                if not add_array(inp, out):
                    raise AssertionError('actual single add was not admitted')
                name = f'single_{size}_alias{int(alias)}'
                report['cases'][name] = compare_words(out.to_numpy(), expected)
        for size in (3, 129, 513):
            rows = []
            for row in range(4):
                n = size + row
                a, b = words(n, row + 1), words(n, row + 5)
                expected = b.copy()
                host(((n + 511) // 512,), (128,), (a, expected, np.uint64(n)))
                rows.append((xp.asarray(a), xp.asarray(b), expected))
            state = SimpleNamespace()
            launch = prepare_add_arrays(state, tuple((a, b) for a, b, _ in rows))
            if launch is None:
                raise AssertionError('actual batched caller was not admitted')
            launch()
            for row, (_, dst, expected) in enumerate(rows):
                name = f'batch_{size}_row{row}'
                report['cases'][name] = compare_words(dst.to_numpy(), expected)
    report['words'] = sum(x['words'] for x in report['cases'].values())
    report['different_words'] = sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if report['different_words']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
