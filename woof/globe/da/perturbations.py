"""The spectral perturbation family of the WOOF global ensemble.

Used twice: for the initial ensemble (every member is the base state plus
one draw) and for additive inflation (a fraction of the same draw added
after each analysis).  One draw is, per spectral field, a complex Gaussian
coefficient set with a red power spectrum (``n^slope`` in power per degree
up to ``max_degree``, nothing above), vertically correlated across levels
by a first-order autoregression (0.9 between adjacent levels), scaled so
the grid rms at each level equals the climatological amplitude of that
level (:func:`climatological_factor` times the option's amplitude):

* temperature: drawn as T', divided by the base state's Exner function on
  the grid and analysed back, so it enters theta as the T' it was drawn as;
* wind: drawn as vorticity so the wind is rotational (a divergent random
  part is gravity-wave energy the model would radiate; the time-lagged
  analysis differences carry the balanced divergent part);
* log surface pressure: one 2-D draw;
* vapor: a relative factor ``1 + f`` on the grid vapor (f clipped to
  +/- 0.5), so the perturbed vapor keeps its sign and its dry regions dry.

Under ``EnsembleOptions.perturbation_balance = "linear"`` (the default
since 2026-09-06) the mass fields are not drawn on their own: the
temperature and the log surface pressure are the ones the linear balance
equation gives the drawn rotational wind, ``laplacian(phi') = div(f grad
psi')`` with ``psi'`` the streamfunction of the drawn vorticity, the
temperature from the hydrostatic relation ``T' = -(1/R) d phi'/d ln p``
on the base state's own column and the log surface pressure from the
balanced geopotential at the lowest full level, ``phi'_bottom / (R
T_bottom)``; the vorticity draw is correlated in the vertical in ln p
on a basis of ``perturbation_vertical_modes`` smooth vertical modes
instead of level by level, so the thermal wind the balance implies is one
of atmospheric size.  A fraction ``perturbation_unbalanced_fraction`` of the independent
temperature and pressure draws is added on top (the part of a forecast
error the balance does not describe, and the only mass spread the
tropics get from the draw, where f is small and the balanced geopotential
with it).  Measured on the 2026-09-01 case (32 T127 members, three hours
of integration): the independent draws ("none") had turned 51 percent
divergent in kinetic energy, their surface-pressure spread had grown from
160 to 290 Pa in six hours against a station-measured background error of
75 to 125 Pa, and the analysis increment they produced carried a 500 hPa
height increment whose geostrophic wind was correlated 0.3 with the wind
increment in the extratropics: the mass and wind perturbations were
independent, the model radiated the difference as gravity waves, and the
covariance the filter used was theirs.

The normalisation is analytic in expectation, as :mod:`woof.da.perturb`
argues it should be: a real field with complex-orthonormal coefficients
has grid mean square ``(1 / 4 pi) sum_n (2n + 1) sigma_n^2`` when
``E|c_nm|^2 = sigma_n^2``, so ``sigma_n^2 = A n^slope`` is scaled by the
one constant that makes that sum one, and the realised rms of each draw
is recorded in the provenance instead of being forced.  For the wind the
same identity is applied to the kinetic energy of a vorticity field
(``a^2 / (8 pi n(n+1)) sum mult |zeta|^2``, Parseval against ``|v|^2/2``).
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from ..constants import DRY_AIR_GAS_CONSTANT, EARTH_ROTATION_RATE_S, KAPPA, REFERENCE_PRESSURE_PA, SPECTRAL_FIELDS
from ..state import ArwenGlobalState
from .options import EnsembleOptions

#: Vertical correlation between adjacent levels of one draw.
LEVEL_AUTOCORRELATION = 0.9

#: Climatological shape of the perturbation amplitude with pressure:
#: (pressure hPa, factor), interpolated linearly in ln p, held outside.
#: The factor multiplies the option's amplitude; the shapes are the
#: forecast-error profiles every global ensemble is built on (largest
#: wind error at the jet, temperature error tapering through the
#: stratosphere, vapor confined to where there is vapor).
CLIMATOLOGICAL_PROFILES = {
    "temperature": ((1000.0, 1.0), (300.0, 1.0), (100.0, 0.6), (30.0, 0.4), (1.0, 0.4)),
    "wind": ((1000.0, 0.8), (700.0, 0.9), (250.0, 1.2), (100.0, 1.0), (30.0, 0.8), (1.0, 0.8)),
    "vapor": ((1000.0, 1.0), (700.0, 1.0), (300.0, 1.5), (100.0, 0.5), (50.0, 0.0), (1.0, 0.0)),
}


def climatological_factor(kind: str, p_full_pa) -> np.ndarray:
    """The profile factor at each ``p_full_pa`` (Pa), ``(nlev,)``."""
    table = CLIMATOLOGICAL_PROFILES[kind]
    p_hpa = np.asarray([row[0] for row in table], dtype=np.float64)
    factor = np.asarray([row[1] for row in table], dtype=np.float64)
    order = np.argsort(p_hpa)
    ln_p = np.log(p_hpa[order] * 100.0)
    return np.interp(np.log(np.asarray(p_full_pa, dtype=np.float64)), ln_p, factor[order])


def _degree_weights(truncation: int, max_degree: int, slope: float) -> np.ndarray:
    """``sigma_n`` per degree, ``(T+1,)``, zero at n = 0 and above
    ``max_degree``, scaled so a field drawn with them has unit grid mean
    square in expectation."""
    n = np.arange(truncation + 1, dtype=np.float64)
    power = np.zeros(truncation + 1)
    keep = (n >= 1) & (n <= max_degree)
    power[keep] = n[keep] ** slope
    total = float(np.sum((2.0 * n + 1.0) * power)) / (4.0 * math.pi)
    if total <= 0.0:
        raise ValueError("the perturbation spectrum has no power")
    return np.sqrt(power / total)


def draw_coefficients(rng, truncation: int, nlev: int, sigma_n: np.ndarray,
                      level_basis=None) -> np.ndarray:
    """``(nlev, T+1, T+1)`` complex128 coefficients with ``E|c_nm|^2 =
    sigma_n^2`` on the triangle, real on m = 0, nothing above the
    triangle; in the vertical either autoregressive with
    :data:`LEVEL_AUTOCORRELATION` between adjacent levels (``level_basis``
    None, the independent family's rule) or a sum of vertical modes,
    ``c_k = sum_m level_basis[k, m] e_m`` with independent unit draws
    ``e_m`` (``level_basis`` ``(nlev, modes)`` with unit row norms, the
    balanced family's rule: a draw smooth enough in ln p to carry a
    thermal wind of atmospheric size)."""
    t = truncation + 1
    tri = np.tri(t, dtype=bool)
    out = np.zeros((nlev, t, t), dtype=np.complex128)

    def unit_draw():
        real = rng.standard_normal((t, t))
        imag = rng.standard_normal((t, t))
        unit = (real + 1j * imag) / math.sqrt(2.0)
        unit[:, 0] = real[:, 0]
        return np.where(tri, unit, 0.0)

    if level_basis is None:
        previous = None
        rho = LEVEL_AUTOCORRELATION
        for k in range(nlev):
            unit = unit_draw()
            if previous is not None:
                unit = rho * previous + math.sqrt(1.0 - rho * rho) * unit
            previous = unit
            out[k] = unit * sigma_n[:, None]
        return out
    basis = np.asarray(level_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != nlev:
        raise ValueError("level_basis must be (nlev, modes)")
    modes = [unit_draw() for _ in range(basis.shape[1])]
    for k in range(nlev):
        out[k] = sum(basis[k, m] * modes[m] for m in range(basis.shape[1])) * sigma_n[:, None]
    return out


def vertical_mode_basis(ln_p_full_ref: np.ndarray, modes: int) -> np.ndarray:
    """``(nlev, modes)`` the vertical basis of the balanced draw: cosines
    in the normalised ln p of the column (``cos(m pi s)`` with ``s`` from
    0 at the top to 1 at the surface; m = 0 the barotropic mode), with
    amplitudes ``(1 + m^2)^(-1/2)`` so the baroclinic modes fall away, the
    rows normalised so every level draws unit variance.  A draw on this
    basis has a bounded ln p derivative everywhere (the thermal wind of a
    balanced geopotential on it is the same size at a 5 hPa layer near the
    surface as at a 50 hPa one aloft), where an autoregression in level
    index or in ln p has a derivative that grows without bound as the
    layers thin: on the record's T127 column that made a 1.5 K
    temperature spread out of a 1.5 hPa pressure one."""
    lnp = np.asarray(ln_p_full_ref, dtype=np.float64).reshape(-1)
    n_modes = int(modes)
    if n_modes < 1:
        raise ValueError("vertical_mode_basis needs at least one mode")
    if lnp.size < 2 or lnp[-1] == lnp[0]:
        s_norm = np.zeros(lnp.size)
    else:
        s_norm = (lnp - lnp[0]) / (lnp[-1] - lnp[0])
    basis = np.zeros((lnp.size, n_modes))
    for m in range(n_modes):
        basis[:, m] = np.cos(m * math.pi * s_norm) / math.sqrt(1.0 + m * m)
    norms = np.sqrt(np.sum(basis ** 2, axis=1))
    return basis / norms[:, None]


#: The pressure the internal modes reach up to and vanish above: the
#: wind and temperature error of the free atmosphere lives in the
#: troposphere and the lower stratosphere, and on the column below this
#: pressure the first internal mode's wind peaks at 260 hPa with its
#: temperature extremes at 135 and 505 hPa.
INTERNAL_MODE_TOP_PA = 7000.0
#: The top of the low-level internal mode's column: ``sin^2(pi s)`` on the
#: column between this pressure and the surface peaks at 700 hPa with its
#: temperature extremes at 590 and 825 hPa, the shear of a low-level
#: baroclinic error (a boundary-layer or frontal temperature error at a
#: 1,000 km wavelength carries 3 m/s of 850 hPa wind over the lowest
#: 150 hPa and less than a degree of temperature).
INTERNAL_LOW_MODE_TOP_PA = 50000.0
#: The low-level mode's amplitude against the jet-level modes' unit: at
#: 1/sqrt(2) (the first jet-level mode's) it carried 1.8 K of balanced
#: temperature at 850 hPa on the T63 probe column for 1.4 m/s of wind
#: there; the temperature of a low-level shear at the internal part's
#: scale is what bounds it.
INTERNAL_LOW_MODE_AMPLITUDE = 0.5


def internal_mode_basis(ln_p_full_ref: np.ndarray, modes: int, *, top_pa: float = INTERNAL_MODE_TOP_PA,
                        low_top_pa: float = INTERNAL_LOW_MODE_TOP_PA) -> np.ndarray:
    """``(nlev, modes + 1)`` the vertical basis of the balanced draw's
    internal part: ``sin^2(m pi s)`` in the normalised ln p of the column
    between ``top_pa`` and the lowest full level (``s`` 0 at ``top_pa``, 1
    at the surface; zero above ``top_pa``), m = 1 the first internal mode,
    amplitudes ``(1 + m^2)^(-1/2)``, and as the last column the low-level
    mode ``sin^2(pi s_low)`` on the column between ``low_top_pa`` and the
    surface at :data:`INTERNAL_LOW_MODE_AMPLITUDE`; the matrix scaled so a
    unit draw on it has unit mean square over the column.  Every mode and
    its ln p derivative vanish at the surface and at its top: a draw on
    this basis carries wind and temperature aloft, no surface pressure
    and no surface temperature (the balanced temperature is the
    derivative of the geopotential in ln p, so the first jet-level mode is
    a warm core under an upper anomaly with a cold tropopause above it,
    and the low-level mode a warm layer at 825 hPa under a cold one at
    590), the part of a forecast error the external (cosine) modes cannot
    give without a surface signature of the same size.  A plain sine has
    its steepest gradient where it vanishes and put 1.5 K of balanced
    temperature into the lowest 100 hPa of the T63 probe column.  On the
    record's T127 column the cosine modes alone made 1.5 hPa of pressure
    spread carry 0.39 K of temperature and 1.6 m/s of wind, where the
    innovations put the background error at 0.7 hPa, 0.8 K and 3 m/s:
    the ratio is what the internal modes change.  ``modes`` 0 returns an
    empty basis (no internal part)."""
    lnp = np.asarray(ln_p_full_ref, dtype=np.float64).reshape(-1)
    n_modes = int(modes)
    if n_modes < 1:
        return np.zeros((lnp.size, 0))
    basis = np.zeros((lnp.size, n_modes + 1))
    bottom = float(lnp[-1])
    top = math.log(float(top_pa))
    low_top = math.log(float(low_top_pa))
    if bottom <= top or bottom <= low_top:
        raise ValueError("internal_mode_basis needs a column reaching below its top pressures")
    s = np.clip((lnp - top) / (bottom - top), 0.0, 1.0)
    for m in range(1, n_modes + 1):
        basis[:, m - 1] = np.sin(m * math.pi * s) ** 2 / math.sqrt(1.0 + m * m)
    s_low = np.clip((lnp - low_top) / (bottom - low_top), 0.0, 1.0)
    basis[:, n_modes] = np.sin(math.pi * s_low) ** 2 * INTERNAL_LOW_MODE_AMPLITUDE
    mean_square = float(np.mean(np.sum(basis ** 2, axis=1)))
    if mean_square <= 0.0:
        raise ValueError("internal_mode_basis has no level inside its column")
    return basis / math.sqrt(mean_square)


def latitude_envelope(sin_lat, tropical_fraction: float) -> np.ndarray:
    """``sqrt(f^2 + (1 - f^2) sin^2(lat))`` per latitude: the amplitude
    envelope the balanced wind draw is multiplied by on the grid, one at
    the poles and ``tropical_fraction`` at the equator (0.5: 0.66 at 30
    degrees, 0.79 at 45, 0.90 at 60).  The wind error of a global
    forecast is largest in the storm tracks and smallest in the tropics,
    and the record's motion-vector innovations (2.8 m/s rms, most of them
    between 30 S and 30 N) against its radiosonde innovations (3.3 m/s,
    most of them in the northern extratropics) read the same."""
    f = float(tropical_fraction)
    s = np.asarray(sin_lat, dtype=np.float64)
    return np.sqrt(f * f + (1.0 - f * f) * s * s)


def balanced_mass_from_vorticity(model, transform, base_atmosphere, vorticity_spectral):
    """The linearly balanced geopotential of a rotational wind perturbation
    and the temperature and log-surface-pressure perturbations it implies.

    ``laplacian(phi') = div(f grad psi') = f zeta' + beta d psi'/dy`` with
    ``psi'`` the streamfunction of the vorticity draw (``beta = df/dy``),
    solved in spectral space (the inverse Laplacian; the global mean of
    ``phi'`` is zero); then on the base state's own column ``T'_k = -(1/R)
    d phi'/d ln p`` by centred differences in ln p (one-sided at the top
    and bottom) and ``ln ps' = phi'_bottom / (R T_bottom)`` with ``T_bottom``
    the base temperature at the lowest full level, so the perturbed
    column's geopotential at the levels' pressures is the balanced one and
    its surface stays the orography.  In the tropics ``f`` is small and so
    is ``phi'``: no balance is imposed there, the balanced mass
    perturbation is what the wind supports.  Returns ``(temperature_grid,
    ln_ps_grid, phi_grid)`` on the transform's namespace."""
    backend = transform.backend
    xp = backend.xp
    zeta = xp.asarray(vorticity_spectral, dtype=backend.complex_dtype)
    psi = transform.inverse_laplacian(zeta)
    zeta_grid = transform.inverse(zeta)
    _d_east, d_north = transform.gradient(psi)
    sinlat = xp.asarray(transform.grid.sin_lat, dtype=backend.float_dtype)[:, None]
    coslat = xp.asarray(transform.grid.cos_lat, dtype=backend.float_dtype)[:, None]
    omega = float(getattr(model, "rotation_rate_s", EARTH_ROTATION_RATE_S))
    f = 2.0 * omega * sinlat
    beta = 2.0 * omega * coslat / float(transform.grid.radius_m)
    forcing = f[None] * zeta_grid + beta[None] * d_north
    phi = transform.inverse(transform.inverse_laplacian(transform.project(transform.forward(forcing))))
    g = model.grid_state(base_atmosphere, only=("p_full", "temperature"))
    ln_p = xp.log(g["p_full"])
    t_base = g["temperature"]
    nlev = int(phi.shape[0])
    temperature = xp.empty_like(phi)
    if nlev == 1:
        temperature[0] = 0.0
    else:
        temperature[0] = -(phi[0] - phi[1]) / (ln_p[0] - ln_p[1]) / DRY_AIR_GAS_CONSTANT
        temperature[-1] = -(phi[-2] - phi[-1]) / (ln_p[-2] - ln_p[-1]) / DRY_AIR_GAS_CONSTANT
        if nlev > 2:
            temperature[1:-1] = -(phi[:-2] - phi[2:]) / (ln_p[:-2] - ln_p[2:]) / DRY_AIR_GAS_CONSTANT
    ln_ps = phi[-1] / (DRY_AIR_GAS_CONSTANT * t_base[-1])
    model.release_syntheses()
    return temperature, ln_ps, phi


def _synoptic_vorticity_weights(truncation: int, max_degree: int, peak_degree: float, radius_m: float) -> np.ndarray:
    """``sigma_n`` per degree for a VORTICITY draw whose kinetic energy per
    degree is synoptic-peaked, ``KE_n proportional to (n / n0)^2 / (1 + (n /
    n0)^2)^(5/2)`` (rising as n^2 below the peak degree ``n0``, falling as
    n^-3 above it, the shape of a forecast-error wind spectrum), zero at
    n = 0 and above ``max_degree``, scaled so the drawn wind has unit rms
    in expectation.  The ``n^slope`` weights of :func:`_degree_weights`
    put 0.9 of a vorticity draw's kinetic energy at n = 1 (the wind's
    energy per degree goes as ``(2n + 1) sigma_n^2 / (n (n + 1))``), and a
    planetary-scale wind balances a planetary-scale mass field of ten
    hectopascals: the balanced family draws its wind on this shape."""
    n = np.arange(truncation + 1, dtype=np.float64)
    ke_scale = _kinetic_energy_scale(truncation, radius_m)
    ratio = n / float(peak_degree)
    ke_shape = np.zeros(truncation + 1)
    keep = (n >= 1) & (n <= max_degree)
    ke_shape[keep] = ratio[keep] ** 2 / (1.0 + ratio[keep] ** 2) ** 2.5
    # sigma_n^2 = KE_n / ((2n + 1) ke_scale_n); the wind rms of the whole
    # draw is sqrt(2 sum_n KE_n), normalised to one.
    sigma2 = np.zeros(truncation + 1)
    sigma2[keep] = ke_shape[keep] / ((2.0 * n[keep] + 1.0) * ke_scale[keep])
    total = 2.0 * float(np.sum((2.0 * n + 1.0) * sigma2 * ke_scale))
    if total <= 0.0:
        raise ValueError("the synoptic vorticity spectrum has no power")
    return np.sqrt(sigma2 / total)


def _kinetic_energy_scale(truncation: int, radius_m: float) -> np.ndarray:
    """``a^2 / (8 pi n(n+1))`` per degree; the multiplier that turns
    ``sum_m mult |zeta_nm|^2`` into the wind's kinetic energy (J/kg)."""
    n = np.arange(truncation + 1, dtype=np.float64)
    lam = n * (n + 1.0)
    lam[0] = np.inf
    return radius_m * radius_m / (8.0 * math.pi * lam)


def wind_rms_of_vorticity(coeff: np.ndarray, radius_m: float) -> np.ndarray:
    """Grid rms wind speed each level of a vorticity coefficient stack
    carries, ``(nlev,)`` (twice the kinetic energy, square-rooted)."""
    c = np.asarray(coeff)
    mult = np.where(np.arange(c.shape[-1]) == 0, 1.0, 2.0)
    power = np.sum(mult * (c.real ** 2 + c.imag ** 2), axis=-1)
    ke = np.sum(power * _kinetic_energy_scale(c.shape[-2] - 1, radius_m), axis=-1)
    return np.sqrt(2.0 * ke)


def grid_rms_of_scalar(coeff: np.ndarray) -> np.ndarray:
    """Grid rms each level of a scalar coefficient stack carries."""
    c = np.asarray(coeff)
    mult = np.where(np.arange(c.shape[-1]) == 0, 1.0, 2.0)
    return np.sqrt(np.sum(mult * (c.real ** 2 + c.imag ** 2), axis=(-2, -1)) / (4.0 * math.pi))


def member_rng(seed: int, member: int, purpose: str, cycle: int = 0):
    """The stream member ``member`` draws from for ``purpose`` at
    ``cycle``: reproducible on its own, independent of every other.  The
    purpose enters as a CRC-32, not Python's string hash, which is salted
    per process and would make every member a different member in every
    run (found when a test's truth changed between two processes)."""
    return np.random.default_rng([int(seed), int(member), int(cycle), zlib.crc32(purpose.encode("utf-8"))])


def draw_perturbation(model, transform, base_atmosphere, options: EnsembleOptions, rng, *,
                      amplitude_scale: float = 1.0) -> tuple[dict[str, object], dict[str, object]]:
    """One perturbation of the five spectral fields around
    ``base_atmosphere``: ``({field: spectral increment}, record)`` on the
    transform's namespace.  The record carries the per-level target
    amplitudes and the realised grid rms of every draw."""
    backend = transform.backend
    xp = backend.xp
    truncation = int(transform.truncation)
    nlev = int(model.nlev)
    max_degree = min(options.max_degree, truncation)
    sigma_n = _degree_weights(truncation, max_degree, float(options.perturbation_spectral_slope))
    radius = float(transform.grid.radius_m)
    scale = float(amplitude_scale)

    # The pressures the climatological profile is read at: the base
    # state's global-mean surface pressure through the hybrid coordinate.
    g = model.grid_state(base_atmosphere, only=("ps", "p_full"))
    ps_mean = float(transform.grid.global_mean(np.asarray(backend.to_numpy(g["ps"]))))
    a = np.asarray(model.vertical.a_half_pa)
    b = np.asarray(model.vertical.b_half)
    p_half = a + b * ps_mean
    p_full_ref = np.sqrt(p_half[:-1] * p_half[1:])
    p_full_grid = g["p_full"]

    balance = str(options.perturbation_balance)
    unbalanced = float(options.perturbation_unbalanced_fraction) if balance == "linear" else 1.0
    record: dict[str, object] = {
        "spectral_slope": float(options.perturbation_spectral_slope),
        "max_degree": int(max_degree),
        "level_autocorrelation": LEVEL_AUTOCORRELATION,
        "amplitude_scale": scale,
        "balance": balance,
        "unbalanced_fraction": unbalanced,
        "vertical_modes": int(options.perturbation_vertical_modes) if balance == "linear" else None,
        "reference_p_full_hpa": [float(v / 100.0) for v in p_full_ref],
        "fields": {},
    }
    increments: dict[str, object] = {}
    exner = (p_full_grid / REFERENCE_PRESSURE_PA) ** KAPPA

    # Temperature -> theta: the independent draw (the whole of it under
    # "none", the unbalanced residual under "linear").
    amp_t = unbalanced * scale * float(options.perturbation_temperature_k) * climatological_factor(
        "temperature", p_full_ref)
    c_t = draw_coefficients(rng, truncation, nlev, sigma_n) * amp_t[:, None, None]
    t_grid = transform.inverse(backend.asarray(c_t, dtype=backend.complex_dtype))
    record["fields"]["temperature_k"] = {
        "target_rms_per_level": [float(v) for v in amp_t],
        "realised_rms_per_level": [float(v) for v in grid_rms_of_scalar(c_t)],
        "route": "T' drawn, divided by the base Exner function on the grid, analysed into theta",
    }

    # Wind -> vorticity (rotational); under the linear balance the draw is
    # correlated in ln p, not in level index, and is two parts: the
    # external modes (cosines, with a surface-pressure signature) drawn
    # here and the internal modes (sines, without one) drawn below.
    amp_w = scale * float(options.perturbation_wind_m_s) * climatological_factor("wind", p_full_ref)
    level_basis = None
    ke_scale = _kinetic_energy_scale(truncation, radius)
    n = np.arange(truncation + 1, dtype=np.float64)
    if balance == "linear":
        level_basis = vertical_mode_basis(np.log(p_full_ref), int(options.perturbation_vertical_modes))
        peak = min(float(options.perturbation_peak_degree), float(max_degree))
        sigma_w = _synoptic_vorticity_weights(truncation, max_degree, peak, radius)
        record["wind_peak_degree"] = peak
    else:
        sigma_w = sigma_n
    c_z = draw_coefficients(rng, truncation, nlev, sigma_w, level_basis=level_basis)
    # A unit scalar draw has sum_n (2n+1) sigma_n^2 = 4 pi; as vorticity its
    # wind rms is sqrt(2 sum_n (2n+1) sigma_n^2 ke_scale_n); rescale so the
    # expected wind rms is one before the amplitude multiplies it (the
    # synoptic weights are already so scaled; the identity holds for both).
    expected = math.sqrt(2.0 * float(np.sum((2.0 * n + 1.0) * sigma_w ** 2 * ke_scale)))
    c_z = c_z / expected * amp_w[:, None, None]
    increments["vorticity"] = transform.project(backend.asarray(c_z, dtype=backend.complex_dtype))
    increments["divergence"] = transform.zeros(nlev)
    record["fields"]["wind_m_s"] = {
        "target_rms_per_level": [float(v) for v in amp_w],
        "realised_rms_per_level": [float(v) for v in wind_rms_of_vorticity(c_z, radius)],
        "route": "vorticity drawn (rotational wind), divergence untouched"
                 + (", on the vertical-mode basis, synoptic-peaked kinetic energy" if balance == "linear" else ""),
        "vertical_modes": None if level_basis is None else int(level_basis.shape[1]),
    }

    # Log surface pressure: the independent draw (whole, or the residual).
    amp_p = unbalanced * scale * float(options.perturbation_ln_surface_pressure)
    c_p = draw_coefficients(rng, truncation, 1, sigma_n)[0] * amp_p
    lnps_grid = transform.inverse(backend.asarray(c_p, dtype=backend.complex_dtype))
    record["fields"]["ln_surface_pressure"] = {
        "target_rms": amp_p,
        "realised_rms": float(grid_rms_of_scalar(c_p[None])[0]),
    }

    if balance == "linear":
        weights = np.asarray(transform.grid.quadrature_weights, dtype=np.float64)
        envelope = latitude_envelope(np.asarray(transform.grid.sin_lat),
                                     float(options.perturbation_tropical_wind_fraction))
        envelope_mean_square = float(np.sum(weights * envelope ** 2) / np.sum(weights))
        envelope_device = xp.asarray(envelope, dtype=backend.float_dtype)[None, :, None]

        def level_rms(field):
            host = np.asarray(backend.to_numpy(field), dtype=np.float64)
            return [float(math.sqrt(np.sum(weights[:, None] * host[k] ** 2) / (np.sum(weights) * host.shape[-1])))
                    for k in range(host.shape[0])]

        def enveloped(vorticity_spectral):
            grid = transform.inverse(vorticity_spectral) * envelope_device
            return transform.project(transform.forward(grid))

        def wind_mean_square(vorticity_spectral) -> float:
            host = np.asarray(backend.to_numpy(vorticity_spectral))
            return float(np.mean(wind_rms_of_vorticity(host, radius) ** 2))

        # The external part: the cosine-mode draw, enveloped in latitude,
        # balanced, and scaled so the balanced ln ps has the stated grid
        # rms (perturbation_ln_surface_pressure times the amplitude
        # scale).  One amplitude is free under a balance and for the
        # external modes it is the mass's: without this a planetary draw
        # at a small truncation balances kilopascals (a 2.5 m/s wind at
        # n = 1 carries 1,650 m2/s2 of geopotential) and the same wind at
        # the design's peak degree balances about 1.6 hPa, so the wind
        # option alone could not state the family at every truncation.
        zeta_ext = enveloped(increments["vorticity"])
        t_ext, lnps_ext, phi_ext = balanced_mass_from_vorticity(model, transform, base_atmosphere, zeta_ext)
        raw_rms = level_rms(lnps_ext[None])[0]
        target_rms = scale * float(options.perturbation_ln_surface_pressure)
        factor = 0.0 if raw_rms <= 0.0 else float(target_rms / raw_rms)
        zeta_ext = zeta_ext * factor
        t_ext = t_ext * factor
        lnps_ext = lnps_ext * factor
        phi_ext = phi_ext * factor
        external_mean_square = wind_mean_square(zeta_ext)

        # The internal part: the sine-mode draw (no surface signature),
        # enveloped and balanced the same way, scaled so the whole draw's
        # wind has the stated mean square over the column
        # (perturbation_wind_m_s through the climatological profile, at
        # the envelope's mean square); the second free amplitude is the
        # wind's, and the temperature the two parts carry between them is
        # the balance's own.  When the external part alone reaches the
        # stated wind the internal factor is zero and the record says so.
        internal_modes = int(options.perturbation_internal_modes)
        target_mean_square = float(np.mean(amp_w ** 2)) * envelope_mean_square
        internal_factor = 0.0
        internal_mean_square = 0.0
        internal_peak = min(float(options.perturbation_internal_peak_degree), float(max_degree))
        if internal_modes > 0:
            # The internal part's kinetic energy peaks at a shorter
            # wavelength than the external part's (degree 30, 1,300 km,
            # against 12): the balanced temperature of a given shear goes
            # as f L / (R d ln p), so the baroclinic part of a forecast
            # error can carry 3 m/s of shear on less than a degree only
            # at the scales fronts and jet streaks have.
            basis_int = internal_mode_basis(np.log(p_full_ref), internal_modes)
            sigma_int = _synoptic_vorticity_weights(truncation, max_degree, internal_peak, radius)
            expected_int = math.sqrt(2.0 * float(np.sum((2.0 * n + 1.0) * sigma_int ** 2 * ke_scale)))
            c_int = draw_coefficients(rng, truncation, nlev, sigma_int, level_basis=basis_int)
            c_int = c_int / expected_int * amp_w[:, None, None]
            zeta_int = enveloped(transform.project(backend.asarray(c_int, dtype=backend.complex_dtype)))
            internal_mean_square = wind_mean_square(zeta_int)
            deficit = max(target_mean_square - external_mean_square, 0.0)
            internal_factor = math.sqrt(deficit / internal_mean_square) if internal_mean_square > 0.0 else 0.0
            zeta_int = zeta_int * internal_factor
            t_int, lnps_int, phi_int = balanced_mass_from_vorticity(model, transform, base_atmosphere, zeta_int)
        else:
            zeta_int = transform.zeros(nlev)
            t_int = xp.zeros_like(t_ext)
            lnps_int = xp.zeros_like(lnps_ext)
            phi_int = xp.zeros_like(phi_ext)
        increments["vorticity"] = zeta_ext + zeta_int
        c_z = np.asarray(backend.to_numpy(increments["vorticity"]))
        t_bal = t_ext + t_int
        lnps_bal = lnps_ext + lnps_int
        phi_bal = phi_ext + phi_int
        # The base state's global-mean surface pressure is kept by
        # construction (the constant the mass rule would remove is
        # removed here and recorded), so every member starts on the
        # control's own mean and the members' conservation targets agree.
        base_lnps = transform.inverse(base_atmosphere.log_surface_pressure)
        ps_base = np.asarray(backend.to_numpy(xp.exp(base_lnps)), dtype=np.float64)
        ps_new = np.asarray(backend.to_numpy(xp.exp(base_lnps + lnps_bal)), dtype=np.float64)
        constant = math.log(transform.grid.global_mean(ps_base) / transform.grid.global_mean(ps_new))
        lnps_bal = lnps_bal + constant
        t_grid = t_grid + t_bal
        lnps_grid = lnps_grid + lnps_bal
        record["fields"]["wind_m_s"]["realised_rms_per_level"] = [float(v) for v in wind_rms_of_vorticity(c_z, radius)]
        record["fields"]["wind_m_s"]["balance_scale_factor"] = factor
        record["fields"]["wind_m_s"]["internal_scale_factor"] = internal_factor
        record["fields"]["wind_m_s"]["external_rms_per_level"] = [
            float(v) for v in wind_rms_of_vorticity(np.asarray(backend.to_numpy(zeta_ext)), radius)]
        record["fields"]["wind_m_s"]["internal_rms_per_level"] = [
            float(v) for v in wind_rms_of_vorticity(np.asarray(backend.to_numpy(zeta_int)), radius)]
        record["fields"]["balanced"] = {
            "route": (
                "laplacian(phi') = f zeta' + beta d psi'/dy from the drawn vorticity; T' = -(1/R) "
                "d phi'/d ln p on the base column; ln ps' = phi'_bottom / (R T_bottom); the external "
                "(cosine-mode) part scaled to the stated ln ps amplitude, the internal (sine-mode) part "
                "scaled so the whole wind has the stated mean square, both enveloped in latitude, the "
                f"base state's global-mean surface pressure kept; the independent temperature and pressure "
                f"draws at {unbalanced:g} of their amplitude added"
            ),
            "scale_factor_to_ln_ps_amplitude": factor,
            "internal_scale_factor_to_wind_amplitude": internal_factor,
            "internal_modes": internal_modes,
            "internal_mode_top_hpa": INTERNAL_MODE_TOP_PA / 100.0,
            "internal_low_mode_top_hpa": INTERNAL_LOW_MODE_TOP_PA / 100.0,
            "internal_peak_degree": internal_peak,
            "tropical_wind_fraction": float(options.perturbation_tropical_wind_fraction),
            "envelope_mean_square": envelope_mean_square,
            "wind_mean_square_target_m2_s2": target_mean_square,
            "wind_mean_square_external_m2_s2": external_mean_square,
            "wind_mean_square_internal_before_scaling_m2_s2": internal_mean_square,
            "ln_ps_rms_before_scaling": raw_rms,
            "mass_constant_ln_ps": constant,
            "temperature_rms_per_level": level_rms(t_bal),
            "temperature_rms_per_level_external": level_rms(t_ext),
            "temperature_rms_per_level_internal": level_rms(t_int),
            "geopotential_rms_per_level_m2_s2": level_rms(phi_bal),
            "ln_surface_pressure_rms": level_rms(lnps_bal[None])[0],
            "ln_surface_pressure_rms_internal": level_rms(lnps_int[None])[0],
        }
    increments["theta"] = transform.project(transform.forward(t_grid / exner))
    increments["log_surface_pressure"] = transform.project(transform.forward(lnps_grid))

    # Vapor, relative.
    amp_q = scale * float(options.perturbation_vapor_relative) * climatological_factor("vapor", p_full_ref)
    c_q = draw_coefficients(rng, truncation, nlev, sigma_n) * amp_q[:, None, None]
    f_grid = xp.clip(transform.inverse(backend.asarray(c_q, dtype=backend.complex_dtype)), -0.5, 0.5)
    q_grid = xp.maximum(transform.inverse(base_atmosphere.qv), 0.0)
    q_new = transform.project(transform.forward(q_grid * (1.0 + f_grid)))
    increments["qv"] = q_new - base_atmosphere.qv
    record["fields"]["vapor_relative"] = {
        "target_rms_per_level": [float(v) for v in amp_q],
        "realised_rms_per_level": [float(v) for v in grid_rms_of_scalar(c_q)],
        "route": "grid vapor times (1 + f), f clipped to +/- 0.5, analysed back",
    }
    return increments, record


def perturbed_state(model, transform, base: ArwenGlobalState, increments: dict[str, object]) -> ArwenGlobalState:
    """``base`` copied with the spectral increments added, the vapor
    repaired by the model's positivity repair and the state enforced."""
    state = base.copy()
    fields = list(state.atmosphere.fields())
    for index, name in enumerate(SPECTRAL_FIELDS):
        inc = increments.get(name)
        if inc is not None:
            fields[index] = fields[index] + inc
    state = ArwenGlobalState(state.atmosphere.with_fields(fields), state.surface, state.physics_state)
    state, _, _, _ = model._repair_positivity(state)
    model.enforce(state)
    return state


def lagged_differences(states, transform) -> list[dict[str, object]]:
    """Consecutive differences of the spectral fields of ``states`` (each
    an ``ArwenGlobalState`` or a ``MoistHybridState`` at the ensemble
    truncation), one dict per pair."""
    atmospheres = [getattr(s, "atmosphere", s) for s in states]
    out = []
    for earlier, later in zip(atmospheres[:-1], atmospheres[1:]):
        out.append({
            name: getattr(later, name) - getattr(earlier, name)
            for name in SPECTRAL_FIELDS
        })
    return out


def scale_difference(model, transform, base_atmosphere, difference: dict[str, object],
                     target_temperature_rms_k: float) -> tuple[dict[str, object], float]:
    """A lagged difference scaled so its temperature increment's grid rms
    (level mean) equals ``target_temperature_rms_k``; every field scaled
    by the same factor so the difference keeps its balance.  Returns
    ``(scaled, factor)``."""
    backend = transform.backend
    g = model.grid_state(base_atmosphere, only=("p_full",))
    exner = (g["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA
    t_inc = transform.inverse(difference["theta"]) * exner
    rms = float(np.sqrt(np.mean(np.asarray(backend.to_numpy(t_inc), dtype=np.float64) ** 2)))
    factor = 0.0 if rms <= 0.0 else float(target_temperature_rms_k) / rms
    return {name: value * factor for name, value in difference.items()}, factor


__all__ = [
    "CLIMATOLOGICAL_PROFILES",
    "LEVEL_AUTOCORRELATION",
    "balanced_mass_from_vorticity",
    "climatological_factor",
    "draw_coefficients",
    "draw_perturbation",
    "grid_rms_of_scalar",
    "lagged_differences",
    "member_rng",
    "perturbed_state",
    "scale_difference",
    "vertical_mode_basis",
    "wind_rms_of_vorticity",
]
