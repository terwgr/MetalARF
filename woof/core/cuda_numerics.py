"""CUDA numerical admission, cached per device and compiled physics unit.

Each contract executes its unit's own compiler path. Shared bitwise checking
covers gradual underflow, signed zero, seed generation and device libm.
"""
from __future__ import annotations

import numpy as np

_ADMITTED: set[tuple[int, str]] = set()


def require_cuda_numerics(unit: str, *, force: bool = False) -> None:
    import cupy as cp
    contract = _CONTRACTS[unit]
    device = int(cp.cuda.runtime.getDevice())
    key = (device, unit)
    if key in _ADMITTED and not force:
        return
    actual, expected = contract(device)
    if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
        raise RuntimeError(
            f"CUDA numerical admission failed for {unit}: uint32 got "
            f"{actual.view(np.uint32).tolist()} want {expected.view(np.uint32).tolist()}")
    _ADMITTED.add(key)


def _legacy_adapter(device):
    import cupy as cp
    from woof.core.rrtmg_legacy import _adapter_kernel, F
    x = np.array([1.0e-30, 1.0e-10, 1.0e-40, 1.0e-38, 4.0], np.float32)
    o = cp.zeros(3, dtype=cp.float32)
    _adapter_kernel("rla_probe")((1,), (32,), (cp.asarray(x), o))
    got = cp.asnumpy(o)
    want = np.array([x[0] * x[1], 1.0 if x[2] > F(0.0) else 0.0,
                     x[3] / x[4]], np.float32)
    return got, want


def _legacy_prep(device):
    import cupy as cp
    from woof.core.rrtmg_legacy_device import _gpu_module
    x = np.array([1.e-30, 1.e-10, 1.e-39, 2.0, 0.0, -0.0], np.float32)
    y = cp.empty(5, np.float32)
    _gpu_module(device).get_function("rp_probe")(
        (1,), (32,), (cp.asarray(x), y))
    expected = np.array([x[0]*x[1], 1.0, x[2]/x[3],
                         np.maximum(x[4], x[5]),
                         np.minimum(x[4], x[5])], np.float32)
    actual = cp.asnumpy(y)
    return actual, expected


def _mcica(device):
    import cupy as cp
    from woof.core.rrtmg_mcica import _mcica_gpu_kernel, _PMID_SCALE, _make_seeds, _kissvec
    probe_play = np.array([1013.2478, 991.7534, 962.4989, 927.0031],
                          dtype=np.float32)
    x = cp.asarray(np.concatenate(
        [np.array([1.0e-30, 1.0e-10], dtype=np.float32), probe_play]))
    out = cp.zeros(3, dtype=cp.float32)
    _mcica_gpu_kernel("rmcw_probe")((1,), (32,), (x, out))
    got = cp.asnumpy(out)
    pmid = (probe_play * _PMID_SCALE).astype(np.float32)[None, :]
    seeds = _make_seeds(pmid, 0)
    want = np.array([np.float32(1.0e-30) * np.float32(1.0e-10),
                     _kissvec(seeds)[0], _kissvec(seeds)[0]],
                    dtype=np.float32)
    return got, want


def _longwave(device):
    import cupy as cp
    from woof.core.rrtmg_lw import _gpu_kernel, expf, logf, powf
    x = cp.asarray(np.array(
        [1.0e-30, 1.0e-10, -100.0, 3.14159, 1.5, 0.65], dtype=np.float32))
    o = cp.zeros(4, dtype=cp.float32)
    _gpu_kernel("rlw_probe")((1,), (32,), (x, o, np.int32(4)))
    got = cp.asnumpy(o)
    want = np.array([
        np.float32(1.0e-30) * np.float32(1.0e-10),
        expf(np.float32(-100.0)),
        logf(np.float32(3.14159)),
        powf(np.float32(1.5), np.float32(0.65)),
    ], dtype=np.float32)
    return got, want


_CONTRACTS = {
    "legacy_adapter": _legacy_adapter,
    "legacy_prep": _legacy_prep,
    "mcica": _mcica,
    "longwave": _longwave,
}
