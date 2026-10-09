"""The spectral eddy viscosity of closure theory
(woof.globe.spectral.eddy_viscosity, 2026-09-08): the instrument reads a
planted spectrum both ways, the drain per degree is the analytic eddy
viscosity, the total drain on a Kolmogorov spectrum is the cascade rate
that spectrum implies (the property the EDQNM constants were derived for),
the shipped hyperdiffusion configs keep their hashes, and the closure door
is the record config with one table changed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.config import DEFAULT_DIFFUSION_CLOSURE, load_config  # noqa: E402
from woof.globe.configs_dir import config_root  # noqa: E402
from woof.globe.insitu.spectra import SpectralKineticEnergy  # noqa: E402
from woof.globe.runner import build_diffusion  # noqa: E402
from woof.globe.spectral.diffusion import ExponentialHyperdiffusion  # noqa: E402
from woof.globe.spectral.eddy_viscosity import (  # noqa: E402
    EDQNM_CUSP_AMPLITUDE,
    EDQNM_CUSP_DECAY,
    EDQNM_PLATEAU,
    KOLMOGOROV_CONSTANT,
    SpectralEddyViscosity,
    cascade_rate_from_spectrum,
)
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_M = 6.371e6
T = 31

#: Config identity hashes of the three shipped configs, taken at source
#: revision 6fcf1e932, before the closure existed (2026-09-08).  The
#: closure's fields join the identity only when it is selected, so these
#: must not move: a config that did not ask for the closure is the same run
#: it was before the closure was written.
PINNED_HASHES = {
    "configs/verify/arwen_global_gdas_t255_native_24h.toml": "93e5ee7500b8a57a",
    "configs/verify/arwen_global_gdas_t255_native_imex_24h.toml": "ae84e41ebfc767a2",
    "configs/verify/arwen_global_gdas_t383_native_24h.toml": "8d4a2799e40a8bc6",
}


@pytest.fixture(scope="module")
def transform():
    return SphericalHarmonicTransform.create(T, backend="numpy", precision="float64")


def wavenumbers(truncation: int, radius_m: float = EARTH_RADIUS_M) -> np.ndarray:
    """k_n = sqrt(n (n + 1)) / a; the tests that hold a transform pass its
    own radius (the tree's Earth radius is not 6.371e6 to the metre, and
    a 3e-5 mismatch in k is a 4e-5 mismatch in the reading)."""
    n = np.arange(truncation + 1, dtype=np.float64)
    return np.sqrt(n * (n + 1.0)) / float(radius_m)


def plant_spectrum(transform, ke_per_degree: np.ndarray, seed: int = 7):
    """Vorticity-like coefficients ``(nlev, T+1, T+1)`` whose kinetic energy
    per unit mass by degree (SpectralKineticEnergy convention) is
    ``ke_per_degree[level, n]`` exactly: equal power at every order of a
    degree, random phases, m = 0 real."""
    rng = np.random.default_rng(seed)
    nlev, t1 = ke_per_degree.shape
    radius = float(transform.grid.radius_m)
    coeff = np.zeros((nlev, t1, t1), dtype=np.complex128)
    for n in range(1, t1):
        # KE_n = a^2 / (2 n (n+1)) * sum_m w_m |c_nm|^2 / (4 pi), w_0 = 1,
        # w_{m>0} = 2, and equal |c| across the 2n+1 real degrees of freedom.
        power_sum = ke_per_degree[:, n] * (4.0 * math.pi) * 2.0 * n * (n + 1.0) / (radius * radius)
        amplitude = np.sqrt(power_sum / (2.0 * n + 1.0))
        for m in range(0, n + 1):
            if m == 0:
                coeff[:, n, 0] = amplitude * np.where(rng.random(nlev) < 0.5, -1.0, 1.0)
            else:
                phase = np.exp(1j * rng.uniform(0.0, 2.0 * math.pi, size=nlev))
                coeff[:, n, m] = amplitude * phase
    return transform.project(coeff)


def kolmogorov_ke(truncation: int, amplitudes, slope: float = -5.0 / 3.0,
                  radius_m: float = EARTH_RADIUS_M) -> np.ndarray:
    """Per-degree kinetic energy E_n = A k_n^slope dk_n for n >= 2 (m^2/s^2),
    one amplitude A per level (m^3/s^2 per unit wavenumber scale)."""
    k = wavenumbers(truncation, radius_m)
    dk = np.diff(k, prepend=0.0)
    ke = np.zeros((len(amplitudes), truncation + 1))
    for level, a in enumerate(amplitudes):
        ke[level, 2:] = a * k[2:] ** slope * dk[2:]
    return ke


# --------------------------------------------------------------- the reading
def test_the_planted_spectrum_reads_back_through_the_ledger_instrument(transform):
    ke = kolmogorov_ke(T, [2.0e-3, 5.0e-4, 1.0e-5], radius_m=transform.grid.radius_m)
    coeff = plant_spectrum(transform, ke)
    read = SpectralKineticEnergy(transform).by_degree(coeff)
    np.testing.assert_allclose(np.asarray(read), ke, rtol=1e-10, atol=1e-18)


def test_nu_infinity_is_the_plateau_times_root_energy_over_wavenumber(transform):
    closure = SpectralEddyViscosity()
    amplitudes = [2.0e-3, 5.0e-4, 1.0e-5]
    radius = float(transform.grid.radius_m)
    ke = kolmogorov_ke(T, amplitudes, radius_m=radius)
    k = wavenumbers(T, radius)
    k_c = k[T]
    # The planted per-degree energy is A k_n^(-5/3) dk_n with the degree
    # spacing dk_n of each degree; the reading compensates along the slope
    # and divides by the spacing AT the truncation, so the density it
    # returns is A k_c^(-5/3) times the mean of dk_n / dk_T over the five
    # tail degrees (1 + 4e-5 at T31, 1 + 6e-7 at T255: the spacing of the
    # sphere's degrees is nearly uniform); the expectation carries that
    # factor exactly.
    dk = np.diff(k, prepend=0.0)
    tail = np.arange(T + 1 - closure.tail_degrees, T + 1)
    spacing_ratio = np.mean(dk[tail]) / dk[T]
    expected = EDQNM_PLATEAU * np.sqrt(
        np.asarray(amplitudes) * k_c ** (-5.0 / 3.0) * spacing_ratio / k_c
    )
    got = closure.nu_infinity(ke, transform)
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    # And the uncompensated form agrees to the size of that spacing term.
    plain = EDQNM_PLATEAU * np.sqrt(np.asarray(amplitudes) * k_c ** (-5.0 / 3.0) / k_c)
    np.testing.assert_allclose(got, plain, rtol=1e-4)
    # The reading is a number of atmospheric size.  The observed upper
    # tropospheric spectrum (Nastrom and Gage 1985) is near 1.25e5 m^3/s^2
    # at a 500 km wavelength and falls as k^-5/3 below it, so E(k) = A
    # k^-5/3 with A about 8.5e-4 in SI (rad/m); at the T255 cutoff that is
    # E(k_c) about 1.8e4 m^3/s^2, a plateau viscosity of a few thousand
    # m^2/s, and an e-folding time at the truncation of about half a day,
    # some fifty times longer than the shipped 720 s, while at 250 km the
    # closure drains faster than the shipped drain (about 60 h against
    # 460 h).  These brackets hold that arithmetic.
    k255 = wavenumbers(255)
    nu255 = EDQNM_PLATEAU * math.sqrt(8.5e-4 * k255[255] ** (-5.0 / 3.0) / k255[255])
    assert 1.0e3 < nu255 < 1.0e4, nu255
    closure255 = SpectralEddyViscosity()
    nu_e_trunc = closure255.nu_plus(1.0) / EDQNM_PLATEAU * nu255
    tau_trunc_h = 1.0 / (nu_e_trunc * k255[255] ** 2) / 3600.0
    assert 5.0 < tau_trunc_h < 24.0, tau_trunc_h
    n250 = 160
    nu_e_250 = closure255.nu_plus(k255[n250] / k255[255]) / EDQNM_PLATEAU * nu255
    tau_250_h = 1.0 / (nu_e_250 * k255[n250] ** 2) / 3600.0
    assert 30.0 < tau_250_h < 120.0, tau_250_h


def test_the_null_reads_zero_and_moves_nothing(transform):
    closure = SpectralEddyViscosity()
    nlev = 3
    zero = np.zeros((nlev, T + 1, T + 1), dtype=np.complex128)
    ke = SpectralKineticEnergy(transform).by_degree(zero)
    nu = closure.nu_infinity(np.asarray(ke), transform)
    assert np.all(nu == 0.0)
    factor = np.asarray(closure.factors(transform, 300.0, nu))
    assert factor.shape == (nlev, T + 1)
    assert np.all(factor == 1.0)
    ke_planted = kolmogorov_ke(T, [1.0e-3, 1.0e-3, 1.0e-3], radius_m=transform.grid.radius_m)
    coeff = plant_spectrum(transform, ke_planted)
    same = closure.apply(coeff, transform, 300.0, nu)
    assert np.array_equal(np.asarray(same), np.asarray(coeff))


# --------------------------------------------------------------- the operator
def test_the_drain_per_degree_is_the_analytic_eddy_viscosity(transform):
    closure = SpectralEddyViscosity(preserve_degree=1)
    ke = kolmogorov_ke(T, [2.0e-3, 5.0e-4], radius_m=transform.grid.radius_m)
    coeff = plant_spectrum(transform, ke)
    instrument = SpectralKineticEnergy(transform)
    before = np.asarray(instrument.by_degree(coeff))
    nu = closure.nu_infinity(before, transform)
    dt = 300.0
    after = np.asarray(instrument.by_degree(closure.apply(coeff, transform, dt, nu)))
    k = wavenumbers(T, transform.grid.radius_m)
    nu_e = closure.eddy_viscosity_by_degree(nu, transform)
    expected_ratio = np.exp(-2.0 * dt * nu_e * (k * k) ** 1)
    expected_ratio[:, :2] = 1.0
    ratio = np.where(before > 0.0, after / np.where(before > 0.0, before, 1.0), 1.0)
    ratio[:, :2] = 1.0
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-9, atol=1e-12)
    # Something was drained: the last degree lost energy, degree 1 did not.
    assert np.all(after[:, T] < before[:, T])
    assert np.all(after[:, 1] == pytest.approx(before[:, 1], rel=1e-12))


def test_a_scalar_takes_the_eddy_diffusivity_at_the_eddy_prandtl_number(transform):
    closure = SpectralEddyViscosity(eddy_prandtl=0.6)
    ke = kolmogorov_ke(T, [1.0e-3], radius_m=transform.grid.radius_m)
    coeff = plant_spectrum(transform, ke)
    instrument = SpectralKineticEnergy(transform)
    nu = closure.nu_infinity(np.asarray(instrument.by_degree(coeff)), transform)
    dt = 300.0
    momentum = np.asarray(closure.factors(transform, dt, nu))
    scalar_coeff = closure.apply(coeff, transform, dt, nu, prandtl=0.6)
    scalar_factor = np.asarray(scalar_coeff)[0, :, 0].real / np.where(
        np.asarray(coeff)[0, :, 0].real == 0.0, 1.0, np.asarray(coeff)[0, :, 0].real
    )
    scalar_factor[np.asarray(coeff)[0, :, 0].real == 0.0] = 1.0
    np.testing.assert_allclose(scalar_factor[2:], momentum[0, 2:] ** (1.0 / 0.6), rtol=1e-9)


def test_the_cusp_has_the_edqnm_shape():
    closure = SpectralEddyViscosity()
    at_cutoff = closure.nu_plus(1.0) / EDQNM_PLATEAU
    at_half = closure.nu_plus(0.5) / EDQNM_PLATEAU
    assert at_cutoff == pytest.approx(1.0 + EDQNM_CUSP_AMPLITUDE / EDQNM_PLATEAU * math.exp(-EDQNM_CUSP_DECAY), rel=1e-12)
    assert 2.5 < at_cutoff < 2.8
    assert 1.05 < at_half < 1.12
    assert closure.nu_plus(0.0) == 0.0


def test_the_total_drain_on_a_kolmogorov_spectrum_is_the_cascade_rate():
    """The theory content: with E(k) = C_K eps^(2/3) k^(-5/3) through the
    cutoff, sum_n 2 nu_e(n) k_n^2 E_n over the resolved degrees equals eps
    to within the closure's own accuracy (the constants were fitted for a
    Kolmogorov constant of 1.4).  Read in the continuum and on the T255
    degrees of the sphere."""
    closure = SpectralEddyViscosity()
    # Continuum: drain / eps = 2 C_K^(3/2) int_0^1 nu_plus(x) x^(1/3) dx.
    x = np.linspace(1.0e-6, 1.0, 400001)
    integrand = closure.nu_plus(x) * x ** (1.0 / 3.0)
    ratio_continuum = 2.0 * KOLMOGOROV_CONSTANT ** 1.5 * np.trapezoid(integrand, x)
    assert 0.90 < ratio_continuum < 1.10, ratio_continuum
    # Sphere at T255: E_n = C_K eps^(2/3) k_n^(-5/3) dk_n, eps chosen so
    # the reading at the truncation is the planted one.
    truncation = 255
    k = wavenumbers(truncation)
    dk = np.diff(k, prepend=0.0)
    eps = 1.0e-4  # m^2/s^3, an upper-tropospheric mesoscale value
    ke = np.zeros((1, truncation + 1))
    ke[0, 2:] = KOLMOGOROV_CONSTANT * eps ** (2.0 / 3.0) * k[2:] ** (-5.0 / 3.0) * dk[2:]

    class _Geometry:
        class grid:
            radius_m = EARTH_RADIUS_M
        truncation = 255

    density = closure.energy_density_at_truncation(ke, _Geometry)
    assert cascade_rate_from_spectrum(float(density[0]), k[truncation]) == pytest.approx(eps, rel=1e-6)
    nu = closure.nu_infinity(ke, _Geometry)
    nu_e = closure.eddy_viscosity_by_degree(nu, _Geometry)
    drain = np.sum(2.0 * nu_e[0, 2:] * k[2:] ** 2 * ke[0, 2:])
    assert 0.85 < drain / eps < 1.15, drain / eps


# ------------------------------------------------------------- the config door
@pytest.mark.parametrize("relative,pinned", sorted(PINNED_HASHES.items()))
def test_the_shipped_configs_keep_their_hashes_and_the_hyperdiffusion(relative, pinned):
    # The experiments ship flat inside the package; the key keeps the
    # source tree's spelling so the pinned hashes read the same in both.
    cfg = load_config(config_root() / Path(relative).name)
    assert cfg.diffusion_closure == DEFAULT_DIFFUSION_CLOSURE
    assert cfg.config_hash.startswith(pinned)
    identity = cfg.config_identity
    for key in ("diffusion_closure", "closure_plateau", "closure_cusp_amplitude",
                "closure_cusp_decay", "closure_eddy_prandtl", "closure_tail_degrees"):
        assert key not in identity
    drain = build_diffusion(cfg)
    assert isinstance(drain, ExponentialHyperdiffusion)


def test_the_closure_door_is_the_record_config_with_one_table_changed():
    record = load_config(config_root() / "arwen_global_gdas_t255_native_24h.toml")
    door = load_config(config_root() / "arwen_global_gdas_t255_native_closure_24h.toml")
    assert door.diffusion_closure == "spectral_eddy_viscosity"
    assert door.config_hash != record.config_hash
    identity = door.config_identity
    assert identity["diffusion_closure"] == "spectral_eddy_viscosity"
    assert identity["closure_plateau"] == 0.267
    assert identity["closure_cusp_amplitude"] == 9.21
    assert identity["closure_cusp_decay"] == 3.03
    assert identity["closure_eddy_prandtl"] == 0.6
    assert identity["closure_tail_degrees"] == 5
    drain = build_diffusion(door)
    assert isinstance(drain, SpectralEddyViscosity)
    assert drain.describe()["kolmogorov_constant"] == KOLMOGOROV_CONSTANT
    # Everything else identical: the identities agree once the closure keys
    # and the name are removed.
    a = dict(record.config_identity)
    b = dict(identity)
    for key in ("diffusion_closure", "closure_plateau", "closure_cusp_amplitude",
                "closure_cusp_decay", "closure_eddy_prandtl", "closure_tail_degrees"):
        b.pop(key)
    a.pop("name", None)
    b.pop("name", None)
    assert a == b


def test_one_semi_lagrangian_step_under_the_closure_drains_only_the_resolved_tail():
    """The wiring: a T21 step with the closure equals the same step without
    a drain followed by a per-degree reduction (the drain sits between the
    dynamics and the fixers, which are off here), so every degree's kinetic
    energy is at most the drain-free step's, the last degree's is strictly
    below it, degrees 0 and 1 are untouched, and the receipt record carries
    one plateau viscosity per level read from a finite energy."""
    from test_arwen_global_semilag_step import _baroclinic, _model, _transform
    from woof.globe.vertical import HybridCoordinate

    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(8, 100.0)
    bundle = _baroclinic(transform, vertical)
    dt = 300.0
    free = _model(transform, vertical)
    closed = _model(transform, vertical)
    object.__setattr__(closed, "diffusion", SpectralEddyViscosity())
    free_state, _ = free.step(bundle, dt)
    closed_state, _ = closed.step(bundle, dt)
    instrument = SpectralKineticEnergy(transform)
    ke_free = np.asarray(instrument.by_degree(free_state.atmosphere.vorticity)) + np.asarray(
        instrument.by_degree(free_state.atmosphere.divergence))
    ke_closed = np.asarray(instrument.by_degree(closed_state.atmosphere.vorticity)) + np.asarray(
        instrument.by_degree(closed_state.atmosphere.divergence))
    assert np.all(np.isfinite(ke_closed))
    assert np.all(ke_closed <= ke_free * (1.0 + 1e-12))
    np.testing.assert_allclose(ke_closed[:, :2], ke_free[:, :2], rtol=1e-12, atol=1e-30)
    energetic = ke_free[:, transform.truncation] > 0.0
    assert energetic.any()
    assert np.all(ke_closed[energetic, transform.truncation] < ke_free[energetic, transform.truncation])
    record = closed.closure_record
    assert record["closure"] == "spectral_eddy_viscosity"
    nu = np.asarray(record["nu_infinity_m2_s"])
    assert nu.shape == (vertical.nlev,)
    assert np.all(np.isfinite(nu)) and np.all(nu >= 0.0) and nu.max() > 0.0
    assert record["bottom_level"] in (0, vertical.nlev - 1)
    # The lowest level is the one with the largest reference pressure.
    a = np.asarray(vertical.a_half_pa)
    b = np.asarray(vertical.b_half)
    p_half = a + b * 1.0e5
    assert record["bottom_level"] == int(np.argmax(np.sqrt(np.maximum(p_half[:-1], 1.0) * p_half[1:])))
    # The theta and vapor fields moved too (the scalar drain), and the
    # drain-free model has no record.
    assert not np.array_equal(np.asarray(closed_state.atmosphere.theta), np.asarray(free_state.atmosphere.theta))
    assert not hasattr(free, "closure_record")


def test_the_closure_refuses_a_hyperdiffusion_key_and_the_reverse(tmp_path):
    src = (config_root() / "arwen_global_gdas_t255_native_closure_24h.toml").read_text(encoding="utf-8")
    bad = src.replace('closure = "spectral_eddy_viscosity"\n', 'closure = "spectral_eddy_viscosity"\norder = 8\n')
    p = tmp_path / "bad.toml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="has no effect under closure"):
        load_config(p)
    src2 = (config_root() / "arwen_global_gdas_t255_native_24h.toml").read_text(encoding="utf-8")
    bad2 = src2.replace("preserve_degree = 1\n", "preserve_degree = 1\neddy_prandtl = 0.7\n", 1)
    p2 = tmp_path / "bad2.toml"
    p2.write_text(bad2, encoding="utf-8")
    with pytest.raises(ValueError, match="belongs to closure"):
        load_config(p2)
