"""Third calibration family of the energy ledger's band, degree and
hemisphere booking (woof.globe.insitu.energy): a Rossby-Haurwitz
wave (Williamson et al. 1992, test 6).

The two families of test_arwen_global_insitu_energy_bands.py inject
COEFFICIENTS and read them back, so a wrong Parseval factor, a wrong
order weight or a wrong degree index that the injection and the reading
shared would cancel.  Here the wind is built from the analytic formulas
on the grid, analysed by the model's own vector transform, and the
ledger's readings are held against closed-form numbers that never pass
through a coefficient: the solid-body part reads (a omega)^2 / 3 per
unit mass at degree 1 (order 0, band n001-020); the wave part, a single
harmonic of degree R + 1 and order R (cos^R phi sin phi cos R lambda is
P_{R+1}^R up to a constant), reads the Gaussian quadrature of the
analytic wind minus the zonal part at degree R + 1 (band n021-060 for
R = 39); the flow is nondivergent and symmetric in the equator, so the
divergent entries read roundoff and each hemisphere holds exactly half.
Both directions: the injection, then an operator that halves the wave
amplitude (three quarters of the wave band leaves, nothing else moves)
and one that turns the wave in longitude through its analytic phase
(nothing moves anywhere).
"""
from __future__ import annotations


import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.constants import GRAVITY_M_S2  # noqa: E402
from woof.globe.vertical import HybridCoordinate  # noqa: E402
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402

from test_arwen_global_insitu_energy_bands import (  # noqa: E402
    _column_bands,
    _ledger,
    _level_bands,
    _rest_state,
    _spectral_bands,
    _uniform_dp_over_g,
    _unpacked,
)

RH_OMEGA_S = 7.848e-6
RH_R = 39


def _rossby_haurwitz_wind(transform, *, omega: float, k: float, r: int, phase: float = 0.0):
    lat, lon = transform.grid.mesh()
    a = float(transform.grid.radius_m)
    cos, sin = np.cos(lat), np.sin(lat)
    u = a * omega * cos + a * k * cos ** (r - 1) * (r * sin ** 2 - cos ** 2) * np.cos(r * lon + phase)
    v = -a * k * r * cos ** (r - 1) * sin * np.sin(r * lon + phase)
    return u, v


def _rossby_haurwitz_state(transform, vertical, ledger, *, omega: float, k: float, r: int, phase: float = 0.0):
    rest = _rest_state(transform, vertical, np.full(transform.grid.shape, 1.0e5))
    state = rest.with_fields([f.copy() for f in rest.fields()])
    u, v = _rossby_haurwitz_wind(transform, omega=omega, k=k, r=r, phase=phase)
    z, d = ledger.model.vector.vordiv_from_wind(u, v)
    zeta = np.zeros_like(state.vorticity)
    div = np.zeros_like(state.divergence)
    for level in range(vertical.nlev):
        zeta[level] = np.asarray(z)
        div[level] = np.asarray(d)
    state.vorticity = zeta
    state.divergence = div
    return state, u, v


def _sphere_mean(transform, field) -> float:
    weights = np.asarray(transform.grid.quadrature_weights, dtype=np.float64)
    return float(0.5 * np.sum(np.mean(field, axis=-1) * weights))


def test_a_rossby_haurwitz_wave_reads_its_zonal_and_wave_parts_in_their_bands_and_degrees():
    transform = SphericalHarmonicTransform.create(63, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(6, 100.0)
    ledger = _ledger(transform, vertical)
    a = float(transform.grid.radius_m)
    state, u, v = _rossby_haurwitz_state(transform, vertical, ledger, omega=RH_OMEGA_S, k=RH_OMEGA_S, r=RH_R)
    entry = _unpacked(ledger, state)["band_net"]["mark"]
    total_per_mass = _sphere_mean(transform, 0.5 * (u * u + v * v))
    zonal_per_mass = (a * RH_OMEGA_S) ** 2 / 3.0
    wave_per_mass = total_per_mass - zonal_per_mass
    assert zonal_per_mass > 800.0 and wave_per_mass > 10.0
    by_degree = entry["level_spectral_by_degree_m2_s2"]
    rot = np.asarray(by_degree["rotational"], dtype=np.float64)
    dvg = np.asarray(by_degree["divergent"], dtype=np.float64)
    assert rot[1] == pytest.approx(zonal_per_mass, rel=1.0e-9)
    assert rot[RH_R + 1] == pytest.approx(wave_per_mass, rel=1.0e-9)
    other = np.ones(rot.size, dtype=bool)
    other[[1, RH_R + 1]] = False
    assert np.all(np.abs(rot[other]) < 1.0e-9 * zonal_per_mass)
    assert np.all(np.abs(dvg) < 1.0e-9 * zonal_per_mass)
    spectral = _spectral_bands(entry)
    assert spectral[0, 0] == pytest.approx(zonal_per_mass, rel=1.0e-9)
    assert spectral[1, 0] == pytest.approx(wave_per_mass, rel=1.0e-9)
    assert abs(spectral[2, 0]) < 1.0e-9 * zonal_per_mass
    level = _level_bands(entry)
    np.testing.assert_allclose(level[0], [0.5 * zonal_per_mass] * 2, rtol=1.0e-9)
    np.testing.assert_allclose(level[1], [0.5 * wave_per_mass] * 2, rtol=1.0e-9)
    assert np.all(np.abs(level[2]) < 1.0e-9 * zonal_per_mass)
    # The column holds the wind on every level, so the booking is the
    # per-mass reading times the column's mass (ps - p_top) / g.
    column = _column_bands(entry)
    mass_over_g = float(np.sum(_uniform_dp_over_g(transform, vertical, 1.0e5)))
    assert mass_over_g == pytest.approx((1.0e5 - 100.0) / GRAVITY_M_S2, rel=1.0e-9)
    np.testing.assert_allclose(column[0], [0.5 * zonal_per_mass * mass_over_g] * 2, rtol=1.0e-9)
    np.testing.assert_allclose(column[1], [0.5 * wave_per_mass * mass_over_g] * 2, rtol=1.0e-9)
    assert np.all(np.abs(column[2]) < 1.0e-9 * zonal_per_mass * mass_over_g)
    total = np.asarray(entry["column_kinetic_j_m2"]["total"], dtype=np.float64)
    assert total.sum() == pytest.approx(total_per_mass * mass_over_g, rel=1.0e-9)


def test_halving_the_wave_amplitude_reads_three_quarters_of_its_band_and_leaves_the_zonal_band():
    transform = SphericalHarmonicTransform.create(63, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(6, 100.0)
    ledger = _ledger(transform, vertical)
    full, _, _ = _rossby_haurwitz_state(transform, vertical, ledger, omega=RH_OMEGA_S, k=RH_OMEGA_S, r=RH_R)
    half, _, _ = _rossby_haurwitz_state(transform, vertical, ledger, omega=RH_OMEGA_S, k=0.5 * RH_OMEGA_S, r=RH_R)
    turned, _, _ = _rossby_haurwitz_state(transform, vertical, ledger, omega=RH_OMEGA_S, k=RH_OMEGA_S, r=RH_R, phase=0.7)
    before = ledger.measure(full)
    unpacked = ledger.unpack(
        ["start", "halve", "turn"],
        np.concatenate([before, ledger.measure(half), ledger.measure(turned)]),
    )
    start = _column_bands(unpacked["band_start"])
    net = _column_bands(unpacked["band_net"]["halve"])
    assert start[1].sum() > 0.0
    assert net[1].sum() == pytest.approx(-0.75 * start[1].sum(), rel=1.0e-9)
    assert np.all(np.abs(net[0]) < 1.0e-9 * start[0])
    assert np.all(np.abs(net[2]) < 1.0e-9 * start[0])
    rot = np.asarray(unpacked["band_net"]["halve"]["level_spectral_by_degree_m2_s2"]["rotational"], dtype=np.float64)
    rot_start = np.asarray(unpacked["band_start"]["level_spectral_by_degree_m2_s2"]["rotational"], dtype=np.float64)
    assert rot[RH_R + 1] == pytest.approx(-0.75 * rot_start[RH_R + 1], rel=1.0e-9)
    assert abs(rot[1]) < 1.0e-9 * rot_start[1]
    # The turn is read as a net against the halved state: the wave band
    # regains its three quarters and nothing else moves, so the turned
    # state's own booking equals the full state's in every band and level.
    # (The empty truncation band holds roundoff of 1e-23 J/m2 on both
    # sides, so the tolerance is relative to the flow's energy.)
    after_turn = start + net + _column_bands(unpacked["band_net"]["turn"])
    np.testing.assert_allclose(after_turn, start, rtol=1.0e-9, atol=1.0e-9 * float(start.max()))
    level_start = _level_bands(unpacked["band_start"])
    level_turn = level_start + _level_bands(unpacked["band_net"]["halve"]) + _level_bands(unpacked["band_net"]["turn"])
    np.testing.assert_allclose(level_turn, level_start, rtol=1.0e-9, atol=1.0e-9 * float(level_start.max()))
