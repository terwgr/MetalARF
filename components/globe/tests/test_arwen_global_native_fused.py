"""The fused native column-water kernel returns the specification's bits.

Device only (skipped without CuPy or under GPUWM_NO_LOCAL_GPU): the
fused float64 column-water pass of ``NativeColumnBatch`` is driven on
random and adversarial float32 fields and compared, ``array_equal``,
against the numpy-specification chain evaluated with the same array
module (the chain the device path ran before it was fused).
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.globe.constants import GRAVITY_M_S2, WATER_SPECIES
from woof.globe.physics.native_batch import _column_water_kernel
from woof.local_gpu import no_local_gpu


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    if no_local_gpu():
        pytest.skip("GPUWM_NO_LOCAL_GPU is set: no device for the fused kernel")
    return cp


def _specification(xp, arrays):
    moist = 1.0 + xp.asarray(arrays["qv"], dtype=xp.float64)
    total = 0
    for name in WATER_SPECIES:
        total = total + xp.asarray(arrays[name], dtype=xp.float64) / moist
    dp = xp.asarray(arrays["dp"], dtype=xp.float64)
    return (total * dp / GRAVITY_M_S2).sum(axis=0)


def _fused(xp, arrays):
    out = xp.empty(arrays["dp"].shape, dtype=xp.float64)
    _column_water_kernel(xp)(
        arrays["qv"], arrays["qc"], arrays["qr"], arrays["qi"], arrays["qs"],
        arrays["qg"], arrays["dp"], float(GRAVITY_M_S2), out,
    )
    return out.sum(axis=0)


@pytest.mark.parametrize("family", ["random", "dry_and_saturated", "spikes"])
@pytest.mark.parametrize("seed", [1, 9])
def test_column_water_kernel_matches_the_specification_bits(cupy, family, seed):
    cp = cupy
    rng = np.random.default_rng(seed)
    shape = (12, 17, 23)
    if family == "random":
        host = {name: (rng.random(shape) * 2.0e-2).astype(np.float32) for name in WATER_SPECIES}
    elif family == "dry_and_saturated":
        host = {name: np.zeros(shape, np.float32) for name in WATER_SPECIES}
        host["qv"][:] = np.float32(3.0e-2)
        host["qv"][::2] = np.float32(1.0e-7)
        host["qc"][1::3] = np.float32(4.0e-3)
        host["qi"][2::3] = np.float32(1.0e-3)
    else:
        host = {name: np.zeros(shape, np.float32) for name in WATER_SPECIES}
        for name in host:
            for _ in range(20):
                k, j, i = rng.integers(shape[0]), rng.integers(shape[1]), rng.integers(shape[2])
                host[name][k, j, i] = np.float32(rng.random() * 5.0e-2)
    host["dp"] = (500.0 + 5000.0 * rng.random(shape)).astype(np.float32)
    arrays = {name: cp.asarray(value) for name, value in host.items()}
    spec = _specification(cp, arrays)
    fused = _fused(cp, arrays)
    assert spec.dtype == cp.float64 and fused.dtype == cp.float64
    assert bool(cp.array_equal(spec, fused)), (family, seed)
    assert float(cp.max(fused)) > 0.0
