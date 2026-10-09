"""The vertical coefficients do not depend on the CPU's vector math.

Checkpoints and receipts hash the hybrid A/B arrays through the
configuration identity.  numpy 2.5's AVX-512 power and exp loops round
differently in the last bit from the C library, so on an AVX-512 Linux
machine every shipped configuration hashed to a different identity than on
Windows or an older CPU, and the pinned identities in this suite failed
there.  The builders now take pow and exp one element at a time from the C
library, which is what every recorded identity was measured with.
"""

from __future__ import annotations

import math

import numpy as np

from woof.globe.vertical import HybridCoordinate, _libm_exp, _libm_pow


def test_the_power_helper_is_the_c_library_pow_element_by_element() -> None:
    x = np.linspace(0.0, 1.0, 97)
    got = _libm_pow(x, 1.7)
    assert got.shape == x.shape
    assert all(g == math.pow(float(v), 1.7) for g, v in zip(got, x))
    r = 1.0831
    steps = np.arange(1, 12, dtype=np.float64)
    got = _libm_pow(r, steps)
    assert got.shape == steps.shape
    assert all(g == math.pow(r, float(n)) for g, n in zip(got, steps))


def test_the_exp_helper_is_the_c_library_exp_element_by_element() -> None:
    z = -np.linspace(0.0, 6.9, 41)
    got = _libm_exp(z)
    assert got.shape == z.shape
    assert all(g == math.exp(float(v)) for g, v in zip(got, z))


def test_the_pressure_blend_weights_are_the_scalar_powers() -> None:
    coordinate = HybridCoordinate.pressure_blend(40, 100.0)
    eta = np.linspace(0.0, 1.0, 41, dtype=np.float64)
    expected = [math.pow(float(v), 1.7) for v in eta[:-1]]
    assert list(coordinate.b_half[:-1]) == expected
    assert coordinate.b_half[-1] == 1.0
