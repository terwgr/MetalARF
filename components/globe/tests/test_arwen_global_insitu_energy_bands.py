"""Calibration of the energy ledger's per-band, per-hemisphere kinetic
booking (woof.globe.insitu.energy, 2026-09-02).

Two synthetic families, each read in both directions:

* a single-degree rotational injection (and its removal) on a uniform
  column: the analytic kinetic energy of the injected degree is read in
  that degree's band and in no other band, the hemisphere split is exact
  for a single-parity field and the bands close on the unfiltered total;
* a broadband divergent injection over a varying surface pressure at a
  truncation that reaches every band (T213), read against the ledger's
  own per-level rows (an independent code path) and exactly zero in the
  untouched bands; the reference level's Parseval [rotational, divergent]
  entry against the synthesised level booking on every band;

and the two symmetries that must read zero everywhere: a rotation of the
whole state about the polar axis, and its mirror in the equator (which
swaps the hemispheres and keeps every band).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.constants import (  # noqa: E402
    GRAVITY_M_S2, GRID_TRACERS, KAPPA, REFERENCE_PRESSURE_PA,
)
from woof.globe.insitu.energy import (  # noqa: E402
    BAND_EDGES,
    HEMISPHERES,
    LEVEL_BAND_TARGET_PA,
    OperatorEnergyLedger,
    band_label,
    reference_level_index,
    spectral_bands,
)
from woof.globe.semi_implicit import BarotropicSemiImplicit  # noqa: E402
from woof.globe.state import MoistHybridState  # noqa: E402
from woof.globe.vertical import HybridCoordinate  # noqa: E402
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402

from test_arwen_global_vertical_modes import _quiet_model  # noqa: E402


def _rest_state(transform, vertical, ps, temperature_k: float = 250.0):
    """A resting isothermal atmosphere over the surface pressure field
    ``ps`` (grid array), zero water, zero grid tracers."""
    p_full = vertical.pressure(ps, transform.backend)["p_full"]
    theta = temperature_k / (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    zeros = np.zeros((vertical.nlev, *transform.spectral_shape), dtype=np.complex128)
    shape = (vertical.nlev, *transform.grid.shape)
    return MoistHybridState(
        vorticity=zeros.copy(), divergence=zeros.copy(), theta=transform.forward(theta),
        log_surface_pressure=transform.forward(np.log(ps)),
        qv=zeros.copy(),
        **{name: np.zeros(shape) for name in GRID_TRACERS},
    )


def _ledger(transform, vertical):
    model = _quiet_model(transform, vertical, BarotropicSemiImplicit(enabled=False))
    return OperatorEnergyLedger(model)


def _unpacked(ledger, state) -> dict:
    """The ledger's own unpack of one mark, read as a net against a zero
    opening mark: ``band_net["mark"]`` is the mark's absolute booking."""
    vector = np.asarray(ledger.measure(state), dtype=np.float64)
    zero = np.zeros_like(vector)
    return ledger.unpack(["start", "mark"], np.concatenate([zero, vector]))


def _column_bands(entry) -> np.ndarray:
    """``(nbands, 2)`` column kinetic energies [northern, southern]."""
    return np.asarray(entry["column_kinetic_j_m2"]["bands"], dtype=np.float64)


def _level_bands(entry) -> np.ndarray:
    return np.asarray(entry["level_kinetic_m2_s2"]["bands"], dtype=np.float64)


def _spectral_bands(entry) -> np.ndarray:
    """``(nbands, 2)`` reference-level [rotational, divergent] by Parseval."""
    return np.asarray(entry["level_spectral_m2_s2"]["bands"], dtype=np.float64)


def _kinetic_per_mass_by_degree(transform, coefficients) -> np.ndarray:
    """The analytic sphere-mean kinetic energy per unit mass of a
    vorticity (or divergence) field by degree: a^2 / (2 n (n+1)) sum_m
    w_m |c_nm|^2 / (4 pi) (the convention insitu.spectra states and its
    Parseval test holds)."""
    a = float(transform.grid.radius_m)
    n = np.arange(transform.truncation + 1, dtype=np.float64)
    factor = np.zeros(n.size)
    factor[1:] = a * a / (2.0 * n[1:] * (n[1:] + 1.0)) / (4.0 * math.pi)
    return transform.power_by_degree(coefficients) * factor


def _uniform_dp_over_g(transform, vertical, ps0: float) -> np.ndarray:
    ps = np.full(transform.grid.shape, ps0)
    dp = vertical.pressure(ps, transform.backend)["dp"]
    return np.asarray(dp)[:, 0, 0] / GRAVITY_M_S2


def test_bands_partition_the_degrees_and_clip_to_the_truncation():
    assert spectral_bands(255) == [(1, 20), (21, 60), (61, 120), (121, 200), (201, 255)]
    assert spectral_bands(63) == [(1, 20), (21, 60), (61, 63)]
    assert spectral_bands(21) == [(1, 20), (21, 21)]
    assert spectral_bands(200) == [(1, 20), (21, 60), (61, 120), (121, 200)]
    assert spectral_bands(5) == [(1, 5)]
    covered = []
    for lo, hi in spectral_bands(533):
        covered.extend(range(lo, hi + 1))
    assert covered == list(range(1, 534))
    assert band_label(201, 255) == "n201-255"
    assert BAND_EDGES[-1][1] is None
    with pytest.raises(ValueError):
        spectral_bands(0)


def test_reference_level_is_the_level_nearest_250_hpa_on_the_reference_column():
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    k = reference_level_index(vertical)
    a, b = vertical.a_half_pa, vertical.b_half
    p_half = a + b * 1.0e5
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    assert k == int(np.argmin(np.abs(p_full - LEVEL_BAND_TARGET_PA)))
    assert abs(p_full[k] - LEVEL_BAND_TARGET_PA) < 0.5 * np.max(np.diff(p_full))


def test_single_degree_rotational_injection_is_read_in_its_band_and_nowhere_else():
    """Family 1, injection: a vorticity field at degrees 30 and 31
    (orders 0..4: the two degrees share orders, so their cross terms are
    antisymmetric in the equator) on a resting uniform column reads its
    analytic column kinetic energy in band n021-060 over the sphere to
    1e-9 relative, exactly zero in every other band, an asymmetric
    hemisphere split that sums to the sphere value, the level booking at
    the reference level, and the bands close on the unfiltered total
    (uniform dp: the cross-band term vanishes by orthogonality)."""
    transform = SphericalHarmonicTransform.create(63, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(6, 100.0)
    ledger = _ledger(transform, vertical)
    assert ledger.band_labels == ["n001-020", "n021-060", "n061-063"]
    ps0 = 1.0e5
    rest = _rest_state(transform, vertical, np.full(transform.grid.shape, ps0))
    before = _unpacked(ledger, rest)
    assert np.all(_column_bands(before["band_net"]["mark"]) == 0.0)
    rng = np.random.default_rng(11)
    degrees = (30, 31)
    injected = rest.with_fields([f.copy() for f in rest.fields()])
    coefficients = np.zeros_like(injected.vorticity)
    for k in range(vertical.nlev):
        scale = 2.0e-5 * (1.0 + 0.3 * k)
        for n in degrees:
            coefficients[k, n, :5] = scale * (rng.standard_normal(5) + 1j * rng.standard_normal(5))
            coefficients[k, n, 0] = coefficients[k, n, 0].real
    injected.vorticity = transform.project(coefficients)
    after = _unpacked(ledger, injected)
    bands = _column_bands(after["band_net"]["mark"])
    per_mass = _kinetic_per_mass_by_degree(transform, injected.vorticity)[:, list(degrees)].sum(axis=1)
    dp_g = _uniform_dp_over_g(transform, vertical, ps0)
    expected_column = float(np.sum(per_mass * dp_g))
    assert expected_column > 1.0e3
    sphere = bands.sum(axis=1)
    assert sphere[1] == pytest.approx(expected_column, rel=1.0e-9)
    assert sphere[0] == 0.0 and sphere[2] == 0.0
    assert np.all(bands[[0, 2]] == 0.0)
    # Mixed parity: the hemispheres differ and still sum to the sphere.
    assert abs(bands[1, 0] - bands[1, 1]) > 1.0e-3 * expected_column
    # Closure: the unfiltered total equals the sum of the bands (uniform dp).
    total = np.asarray(after["band_net"]["mark"]["column_kinetic_j_m2"]["total"])
    assert total.sum() == pytest.approx(sphere.sum(), rel=1.0e-12)
    # The per-level rows (the other code path) agree with the total.
    net = np.asarray(after["net"]["mark"], dtype=np.float64)
    assert net[:-1, 0].sum() == pytest.approx(expected_column, rel=1.0e-9)
    # The level booking: kinetic energy per unit mass at the reference level.
    level = _level_bands(after["band_net"]["mark"])
    k250 = ledger.level_index
    assert level[1].sum() == pytest.approx(float(per_mass[k250]), rel=1.0e-9)
    assert np.all(level[[0, 2]] == 0.0)
    assert after["bands"]["level_index"] == k250
    assert after["bands"]["labels"] == ledger.band_labels
    # The Parseval entry (the other code path for the level): all of it
    # rotational, in the injected band, equal to the synthesised level
    # booking's sphere sum.
    spectral = _spectral_bands(after["band_net"]["mark"])
    assert spectral[1, 0] == pytest.approx(float(per_mass[k250]), rel=1.0e-9)
    assert spectral[1, 1] == 0.0
    assert np.all(spectral[[0, 2]] == 0.0)
    assert spectral[1].sum() == pytest.approx(level[1].sum(), rel=1.0e-9)
    total_spectral = np.asarray(after["band_net"]["mark"]["level_spectral_m2_s2"]["total"])
    assert total_spectral.sum() == pytest.approx(spectral.sum(), rel=1.0e-12)


def test_single_parity_injection_splits_the_hemispheres_exactly_in_half():
    """Degrees 30 and 32 at orders 0, 2, 4 only (n - m even for every
    component, so every product of two is symmetric in the equator)
    read the same energy in each hemisphere to roundoff."""
    transform = SphericalHarmonicTransform.create(63, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(6, 100.0)
    ledger = _ledger(transform, vertical)
    rest = _rest_state(transform, vertical, np.full(transform.grid.shape, 1.0e5))
    rng = np.random.default_rng(5)
    state = rest.with_fields([f.copy() for f in rest.fields()])
    coefficients = np.zeros_like(state.vorticity)
    for n in (30, 32):
        for m in (0, 2, 4):
            values = 3.0e-5 * (rng.standard_normal(vertical.nlev) + 1j * rng.standard_normal(vertical.nlev))
            coefficients[:, n, m] = values.real if m == 0 else values
    state.vorticity = transform.project(coefficients)
    bands = _column_bands(_unpacked(ledger, state)["band_net"]["mark"])
    assert bands[1, 0] > 1.0e3
    assert bands[1, 0] == pytest.approx(bands[1, 1], rel=1.0e-12)


def test_removal_from_one_band_reads_the_analytic_loss_there_and_zero_elsewhere():
    """Family 1, the other direction: an operator that halves the
    degree-30 coefficients removes three quarters of that band's energy
    and touches no other band; the ledger's net is the signed figure."""
    transform = SphericalHarmonicTransform.create(63, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(6, 100.0)
    ledger = _ledger(transform, vertical)
    rest = _rest_state(transform, vertical, np.full(transform.grid.shape, 1.0e5))
    rng = np.random.default_rng(23)
    state = rest.with_fields([f.copy() for f in rest.fields()])
    coefficients = np.zeros_like(state.vorticity)
    coefficients[:, 30, :7] = 2.0e-5 * (
        rng.standard_normal((vertical.nlev, 7)) + 1j * rng.standard_normal((vertical.nlev, 7))
    )
    coefficients[:, 30, 0] = coefficients[:, 30, 0].real
    # Energy in the other bands too, so "nowhere else" is a statement
    # about an operator acting inside a populated spectrum.
    coefficients[:, 8, :3] = 1.0e-5 * (rng.standard_normal((vertical.nlev, 3)) + 1j * rng.standard_normal((vertical.nlev, 3)))
    coefficients[:, 8, 0] = coefficients[:, 8, 0].real
    coefficients[:, 62, :9] = 1.0e-5 * (rng.standard_normal((vertical.nlev, 9)) + 1j * rng.standard_normal((vertical.nlev, 9)))
    coefficients[:, 62, 0] = coefficients[:, 62, 0].real
    state.vorticity = transform.project(coefficients)
    before = ledger.measure(state)
    halved = state.with_fields([f.copy() for f in state.fields()])
    halved.vorticity[:, 30, :] *= 0.5
    after = ledger.measure(halved)
    unpacked = ledger.unpack(["start", "halve"], np.concatenate([before, after]))
    bands_before = _column_bands(unpacked["band_start"])
    net = _column_bands(unpacked["band_net"]["halve"])
    assert bands_before[1].sum() > 1.0e3
    assert net[1].sum() == pytest.approx(-0.75 * bands_before[1].sum(), rel=1.0e-9)
    assert np.all(net[[0, 2]] == 0.0)
    assert bands_before[0].sum() > 0.0 and bands_before[2].sum() > 0.0


@pytest.fixture(scope="module")
def t213():
    transform = SphericalHarmonicTransform.create(213, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(3, 100.0)
    return transform, vertical


def _varying_ps(transform):
    lat, lon = transform.grid.mesh()
    return 1.0e5 * (1.0 + 0.05 * np.cos(2.0 * lat) * np.cos(lon) + 0.02 * np.sin(lat))


def test_broadband_divergent_injection_over_varying_pressure_reads_only_its_bands(t213):
    """Family 2 at a truncation that reaches every band: a broadband
    divergence in n121-200 over a varying surface pressure is read in that
    band alone, equal to the ledger's per-level kinetic rows (the other
    code path), and adding a rotational piece in n201-213 changes the
    n121-200 booking by exactly nothing.  The cross-band term of the
    unfiltered total is reported and small (the varying dp breaks the
    sphere orthogonality by the pressure's own amplitude, not more)."""
    transform, vertical = t213
    ledger = _ledger(transform, vertical)
    assert ledger.band_labels == ["n001-020", "n021-060", "n061-120", "n121-200", "n201-213"]
    rest = _rest_state(transform, vertical, _varying_ps(transform))
    rng = np.random.default_rng(3)
    only_divergent = rest.with_fields([f.copy() for f in rest.fields()])
    div = np.zeros_like(only_divergent.divergence)
    for n in range(121, 201):
        amplitude = 4.0e-6 / (n / 121.0) ** 1.5
        div[:, n, :n + 1] = amplitude * (
            rng.standard_normal((vertical.nlev, n + 1)) + 1j * rng.standard_normal((vertical.nlev, n + 1))
        )
    div[:, :, 0] = div[:, :, 0].real
    only_divergent.divergence = transform.project(div)
    a = _unpacked(ledger, only_divergent)
    bands_a = _column_bands(a["band_net"]["mark"])
    rows_a = np.asarray(a["net"]["mark"], dtype=np.float64)[:-1, 0].sum()
    assert bands_a[3].sum() > 10.0
    assert bands_a[3].sum() == pytest.approx(rows_a, rel=1.0e-12)
    assert np.all(bands_a[[0, 1, 2, 4]] == 0.0)
    both = only_divergent.with_fields([f.copy() for f in only_divergent.fields()])
    zeta = np.zeros_like(both.vorticity)
    for n in range(201, 214):
        zeta[:, n, :n + 1] = 2.0e-6 * (
            rng.standard_normal((vertical.nlev, n + 1)) + 1j * rng.standard_normal((vertical.nlev, n + 1))
        )
    zeta[:, :, 0] = zeta[:, :, 0].real
    both.vorticity = transform.project(zeta)
    only_rotational = rest.with_fields([f.copy() for f in rest.fields()])
    only_rotational.vorticity = both.vorticity
    b = _unpacked(ledger, both)
    r = _unpacked(ledger, only_rotational)
    bands_b = _column_bands(b["band_net"]["mark"])
    bands_r = _column_bands(r["band_net"]["mark"])
    assert np.all(bands_b[3] == bands_a[3])
    assert np.all(bands_b[4] == bands_r[4])
    assert np.all(bands_b[[0, 1, 2]] == 0.0)
    total_b = np.asarray(b["band_net"]["mark"]["column_kinetic_j_m2"]["total"]).sum()
    cross = total_b - bands_b.sum()
    assert abs(cross) < 0.05 * bands_b.sum(), cross
    # The level booking follows the same partition.
    level_b = _level_bands(b["band_net"]["mark"])
    assert np.all(level_b[[0, 1, 2]] == 0.0)
    assert np.all(level_b[3] == _level_bands(a["band_net"]["mark"])[3])
    # Parseval: the divergent injection is divergent in n121-200 only, the
    # rotational one rotational in n201-213 only, and each band's
    # rotational + divergent equals the synthesised level booking's
    # sphere sum (the cross term integrates to zero on the dealiased grid).
    spectral_b = _spectral_bands(b["band_net"]["mark"])
    assert np.all(spectral_b[[0, 1, 2]] == 0.0)
    assert spectral_b[3, 0] == 0.0 and spectral_b[3, 1] > 0.0
    assert spectral_b[4, 1] == 0.0 and spectral_b[4, 0] > 0.0
    np.testing.assert_allclose(spectral_b[3:].sum(axis=1), level_b[3:].sum(axis=1), rtol=1.0e-9)


def _broadband_state(transform, vertical, seed: int = 7):
    rest = _rest_state(transform, vertical, _varying_ps(transform))
    rng = np.random.default_rng(seed)
    state = rest.with_fields([f.copy() for f in rest.fields()])
    zeta = np.zeros_like(state.vorticity)
    div = np.zeros_like(state.divergence)
    for n in range(1, transform.truncation + 1):
        amplitude = 3.0e-5 / n ** 1.5
        zeta[:, n, :n + 1] = amplitude * (rng.standard_normal((vertical.nlev, n + 1)) + 1j * rng.standard_normal((vertical.nlev, n + 1)))
        div[:, n, :n + 1] = 0.3 * amplitude * (rng.standard_normal((vertical.nlev, n + 1)) + 1j * rng.standard_normal((vertical.nlev, n + 1)))
    zeta[:, :, 0] = zeta[:, :, 0].real
    div[:, :, 0] = div[:, :, 0].real
    state.vorticity = transform.project(zeta)
    state.divergence = transform.project(div)
    return state


def _rotate_about_the_pole(state, transform, angle_rad: float):
    m = np.arange(transform.truncation + 1)
    phase = np.exp(1j * m * angle_rad)[None, :]
    rotated = state.with_fields([transform.project(f * phase) for f in state.fields()])
    return rotated


def _mirror_in_the_equator(state, transform):
    n = np.arange(transform.truncation + 1)[:, None]
    m = np.arange(transform.truncation + 1)[None, :]
    parity = np.where((n - m) % 2 == 0, 1.0, -1.0)
    fields = list(state.fields())
    # Scalars mirror with the parity sign; a mirrored flow keeps its
    # divergence and reverses its vorticity (a pseudo-scalar).
    fields[0] = transform.project(-parity * fields[0])
    for index in (1, 2, 3, 4):
        fields[index] = transform.project(parity * fields[index])
    return state.with_fields(fields)


def test_a_rotation_about_the_pole_reads_zero_in_every_band_and_hemisphere(t213):
    """A pure rotation of the whole state about the polar axis (every
    coefficient turned by exp(i m dl)) moves no kinetic energy between
    bands or hemispheres: every net reads roundoff of the band's energy,
    and the unfiltered total too."""
    transform, vertical = t213
    ledger = _ledger(transform, vertical)
    state = _broadband_state(transform, vertical)
    rotated = _rotate_about_the_pole(state, transform, math.radians(37.0))
    before = ledger.measure(state)
    after = ledger.measure(rotated)
    unpacked = ledger.unpack(["start", "rotate"], np.concatenate([before, after]))
    start = _column_bands(unpacked["band_start"])
    net = _column_bands(unpacked["band_net"]["rotate"])
    assert np.all(start > 0.0)
    assert np.all(np.abs(net) < 1.0e-9 * start)
    level_start = _level_bands(unpacked["band_start"])
    level_net = _level_bands(unpacked["band_net"]["rotate"])
    assert np.all(np.abs(level_net) < 1.0e-9 * level_start)
    total_start = np.asarray(unpacked["band_start"]["column_kinetic_j_m2"]["total"])
    total_net = np.asarray(unpacked["band_net"]["rotate"]["column_kinetic_j_m2"]["total"])
    assert np.all(np.abs(total_net) < 1.0e-9 * total_start)
    spectral_start = _spectral_bands(unpacked["band_start"])
    spectral_net = _spectral_bands(unpacked["band_net"]["rotate"])
    assert np.all(spectral_start > 0.0)
    assert np.all(np.abs(spectral_net) < 1.0e-9 * spectral_start)
    # Both code paths agree on every band of the broadband state.
    np.testing.assert_allclose(spectral_start.sum(axis=1), level_start.sum(axis=1), rtol=1.0e-9)


def test_the_equatorial_mirror_keeps_every_band_and_swaps_the_hemispheres(t213):
    transform, vertical = t213
    ledger = _ledger(transform, vertical)
    state = _broadband_state(transform, vertical, seed=9)
    mirrored = _mirror_in_the_equator(state, transform)
    before = ledger.measure(state)
    after = ledger.measure(mirrored)
    unpacked = ledger.unpack(["start", "mirror"], np.concatenate([before, after]))
    start = _column_bands(unpacked["band_start"])
    net = _column_bands(unpacked["band_net"]["mirror"])
    sphere = start.sum(axis=1)
    assert np.all(sphere > 0.0)
    # The split is real before the mirror (a varying ps and random fields).
    assert np.any(np.abs(start[:, 0] - start[:, 1]) > 1.0e-3 * sphere)
    assert np.all(np.abs(net.sum(axis=1)) < 1.0e-9 * sphere)
    after_bands = start + net
    np.testing.assert_allclose(after_bands[:, 0], start[:, 1], rtol=1.0e-9)
    np.testing.assert_allclose(after_bands[:, 1], start[:, 0], rtol=1.0e-9)
    level_start = _level_bands(unpacked["band_start"])
    level_after = level_start + _level_bands(unpacked["band_net"]["mirror"])
    np.testing.assert_allclose(level_after[:, 0], level_start[:, 1], rtol=1.0e-9)
    assert list(HEMISPHERES) == ["northern", "southern"]
    spectral_start = _spectral_bands(unpacked["band_start"])
    spectral_net = _spectral_bands(unpacked["band_net"]["mirror"])
    assert np.all(np.abs(spectral_net) < 1.0e-9 * spectral_start)


def test_an_unchanged_state_reads_exactly_zero():
    transform = SphericalHarmonicTransform.create(42, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(4, 100.0)
    ledger = _ledger(transform, vertical)
    state = _broadband_state(transform, vertical, seed=2)
    before = ledger.measure(state)
    after = ledger.measure(state.with_fields([f.copy() for f in state.fields()]))
    unpacked = ledger.unpack(["start", "same"], np.concatenate([before, after]))
    assert np.all(_column_bands(unpacked["band_net"]["same"]) == 0.0)
    assert np.all(_level_bands(unpacked["band_net"]["same"]) == 0.0)
    assert np.all(_spectral_bands(unpacked["band_net"]["same"]) == 0.0)
    assert np.all(np.asarray(unpacked["net"]["same"]) == 0.0)
    by_degree = unpacked["band_net"]["same"]["level_spectral_by_degree_m2_s2"]
    assert np.all(np.asarray(by_degree["rotational"]) == 0.0)
    assert np.all(np.asarray(by_degree["divergent"]) == 0.0)
    assert ledger.width == ledger.base_width + 6 * (len(ledger.bands) + 1) + 2 * (transform.truncation + 1)


def test_the_per_degree_reading_is_the_band_reading_degree_by_degree():
    """The reference level's Parseval vector by total degree (2026-09-04):
    an injection at degrees 30 and 31 reads its energy at exactly those
    two degrees, zero at every other degree, each degree's value is the
    analytic one, the vector summed over a band's degrees is the band's
    Parseval entry, and its removal is booked degree by degree with the
    signed figure."""
    transform = SphericalHarmonicTransform.create(63, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(6, 100.0)
    ledger = _ledger(transform, vertical)
    rest = _rest_state(transform, vertical, np.full(transform.grid.shape, 1.0e5))
    rng = np.random.default_rng(31)
    state = rest.with_fields([f.copy() for f in rest.fields()])
    zeta = np.zeros_like(state.vorticity)
    div = np.zeros_like(state.divergence)
    for n in (30, 31):
        zeta[:, n, :6] = 2.0e-5 * (rng.standard_normal((vertical.nlev, 6)) + 1j * rng.standard_normal((vertical.nlev, 6)))
        zeta[:, n, 0] = zeta[:, n, 0].real
    div[:, 31, :4] = 1.0e-5 * (rng.standard_normal((vertical.nlev, 4)) + 1j * rng.standard_normal((vertical.nlev, 4)))
    div[:, 31, 0] = div[:, 31, 0].real
    state.vorticity = transform.project(zeta)
    state.divergence = transform.project(div)
    entry = _unpacked(ledger, state)["band_net"]["mark"]
    by_degree = entry["level_spectral_by_degree_m2_s2"]
    rot = np.asarray(by_degree["rotational"], dtype=np.float64)
    dvg = np.asarray(by_degree["divergent"], dtype=np.float64)
    assert rot.shape == dvg.shape == (transform.truncation + 1,)
    k = ledger.level_index
    expected_rot = _kinetic_per_mass_by_degree(transform, state.vorticity)[k]
    expected_div = _kinetic_per_mass_by_degree(transform, state.divergence)[k]
    assert expected_rot[30] > 0.0 and expected_rot[31] > 0.0 and expected_div[31] > 0.0
    np.testing.assert_allclose(rot[[30, 31]], expected_rot[[30, 31]], rtol=1.0e-9)
    np.testing.assert_allclose(dvg[31], expected_div[31], rtol=1.0e-9)
    other = np.ones(transform.truncation + 1, dtype=bool)
    other[[30, 31]] = False
    assert np.all(rot[other] == 0.0)
    other[30] = True
    assert np.all(dvg[other] == 0.0)
    # The band entry is the per-degree vector summed over the band.
    spectral = _spectral_bands(entry)
    for index, (lo, hi) in enumerate(ledger.bands):
        assert spectral[index, 0] == pytest.approx(rot[lo:hi + 1].sum(), rel=1.0e-12, abs=0.0)
        assert spectral[index, 1] == pytest.approx(dvg[lo:hi + 1].sum(), rel=1.0e-12, abs=0.0)
    # Removal, degree by degree: halving degree 30 books -0.75 of its
    # energy at degree 30 and exactly zero at degree 31.
    before = ledger.measure(state)
    halved = state.with_fields([f.copy() for f in state.fields()])
    halved.vorticity[:, 30, :] *= 0.5
    after = ledger.measure(halved)
    net = ledger.unpack(["start", "halve"], np.concatenate([before, after]))["band_net"]["halve"]
    net_rot = np.asarray(net["level_spectral_by_degree_m2_s2"]["rotational"], dtype=np.float64)
    assert net_rot[30] == pytest.approx(-0.75 * expected_rot[30], rel=1.0e-9)
    assert net_rot[31] == 0.0
    untouched = np.ones(transform.truncation + 1, dtype=bool)
    untouched[30] = False
    assert np.all(net_rot[untouched] == 0.0)
    assert np.all(np.asarray(net["level_spectral_by_degree_m2_s2"]["divergent"]) == 0.0)
