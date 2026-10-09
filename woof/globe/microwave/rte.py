"""Clear-sky microwave radiative transfer through a model column.

Plane-parallel, non-scattering, one path at the satellite zenith angle:

    L_toa = sum_l B(T_l) t_l (1 - exp(-dtau_l))
          + t_s [ e B(T_skin) + (1 - e) ( L_down + t_s B(T_cmb) ) ]

with ``t_l`` the transmittance from layer ``l`` to space, ``t_s`` the
total transmittance, ``L_down`` the downwelling radiance at the surface
from the same layers, ``e`` the quasi-polarized ocean emissivity and
``B`` the Planck function.  Brightness temperature is the Planck
inversion of ``L_toa`` at each quadrature frequency, then averaged over
the channel passband (sub-bands of equal weight, Gauss-Legendre points
across each sub-band's width, a flat response assumed).

Planck form: the integral runs in radiance with the exact Planck
function by default.  ``rayleigh_jeans=True`` runs the same integral
linear in temperature (B(T) := T), the limit named in the design; the
difference between the two on the ATMS channels is a fraction of a
kelvin and is one of the terms the calibration reports.

Layers: the column's levels are refined in ln p (``refinement`` sub-layers
per level pair) with temperature and vapour pressure linear in ln p; the
layer absorption is evaluated at the sub-layer's mean state; the
sub-layer thickness comes from the hypsometric equation with the virtual
temperature.  The column is extended isothermally above its top level to
``top_pa`` (1 Pa by default) and below its lowest level to ``BOTTOM_PA``
along the lowest layer's ln p slope (four times finer), each column's own
surface pressure trimming that slab; the sub-layer that straddles the
surface takes the mean of its own temperature and the 2 m temperature
when one is given.

Absorption input: P.676 takes the dry-air pressure and the vapour partial
pressure separately, so each sub-layer hands it ``p - e`` and ``e``.

Geometry: plane-parallel with sec(zenith) everywhere.  The spherical
correction to the local zenith angle is 0.7 degrees at 50 km for a 60
degree surface zenith and is not applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .absorption import absorption_coefficient, vapour_pressure_hpa
from .channels import CHANNELS, Channel
from .emissivity import COSMIC_BACKGROUND_K, fresnel_emissivity, quasi_polarized_emissivity

PLANCK_H = 6.62607015e-34
BOLTZMANN_K = 1.380649e-23
LIGHT_C = 299_792_458.0
GRAVITY_M_S2 = 9.80665
DRY_AIR_R = 287.05
EPSILON = 0.621981

#: The fixed bottom of the surface slab (Pa): below every surface pressure
#: ever recorded, so the layer set is the same for every column.
BOTTOM_PA = 110_000.0

#: Gauss-Legendre points and weights on [-1, 1] for the passband integral.
_GL_POINTS, _GL_WEIGHTS = np.polynomial.legendre.leggauss(5)


def planck_radiance(f_ghz, t_k):
    """Spectral radiance W m-2 sr-1 Hz-1 at frequency f (GHz), temperature T (K)."""
    nu = np.asarray(f_ghz, dtype=np.float64) * 1.0e9
    t = np.asarray(t_k, dtype=np.float64)
    x = PLANCK_H * nu / (BOLTZMANN_K * t)
    return 2.0 * PLANCK_H * nu ** 3 / LIGHT_C ** 2 / np.expm1(x)


def planck_temperature(f_ghz, radiance):
    """Inverse of :func:`planck_radiance`."""
    nu = np.asarray(f_ghz, dtype=np.float64) * 1.0e9
    l = np.asarray(radiance, dtype=np.float64)
    return PLANCK_H * nu / BOLTZMANN_K / np.log1p(2.0 * PLANCK_H * nu ** 3 / (LIGHT_C ** 2 * l))


def channel_quadrature(channel: Channel) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies (GHz) and normalised weights across the channel passband."""
    half = channel.bandwidth_mhz * 1.0e-3 / 2.0
    frequencies = []
    weights = []
    for centre in channel.passbands_ghz:
        frequencies.extend(centre + half * _GL_POINTS)
        weights.extend(_GL_WEIGHTS / 2.0 / len(channel.passbands_ghz))
    return np.asarray(frequencies), np.asarray(weights)


@dataclass(frozen=True)
class Column:
    """A batch of atmospheric columns on shared pressure levels.

    ``pressure_pa`` is 1-D (any monotonic order); ``temperature_k`` and
    ``specific_humidity`` are ``(nlev, ncol)``; the surface arrays are
    ``(ncol,)``.  ``air_temperature_2m_k`` may be None.
    """

    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    specific_humidity: np.ndarray
    surface_pressure_pa: np.ndarray
    skin_temperature_k: np.ndarray
    air_temperature_2m_k: np.ndarray | None = None

    def __post_init__(self) -> None:
        p = np.asarray(self.pressure_pa, dtype=np.float64)
        if p.ndim != 1 or p.size < 3:
            raise ValueError("pressure_pa must be 1-D with at least three levels")
        d = np.diff(p)
        if not (np.all(d > 0) or np.all(d < 0)):
            raise ValueError("pressure_pa must be strictly monotonic")
        t = np.asarray(self.temperature_k, dtype=np.float64)
        q = np.asarray(self.specific_humidity, dtype=np.float64)
        if t.shape != q.shape or t.ndim != 2 or t.shape[0] != p.size:
            raise ValueError("temperature_k and specific_humidity must be (nlev, ncol)")
        ncol = t.shape[1]
        for name in ("surface_pressure_pa", "skin_temperature_k"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (ncol,):
                raise ValueError(f"{name} must have shape ({ncol},)")

    @property
    def ncol(self) -> int:
        return int(np.asarray(self.temperature_k).shape[1])


@dataclass(frozen=True)
class Layers:
    """Refined sub-layers, top to bottom: mean pressure (hPa), mean
    temperature (K), vapour pressure (hPa), thickness (km), each
    ``(nlay, ncol)``, with a mask of layers above the surface."""

    p_hpa: np.ndarray
    t_k: np.ndarray
    e_hpa: np.ndarray
    dz_km: np.ndarray
    above_surface: np.ndarray
    # For weighting functions: the parent level index of each sub-layer.
    parent_level: np.ndarray
    #: ln p thickness of each sub-layer after trimming at the surface,
    #: ``(nlay, ncol)``; a weight per unit ln p is the weight over this.
    thickness_lnp: np.ndarray | None = None


def build_layers(column: Column, *, refinement: int = 4, top_pa: float = 1.0) -> Layers:
    p = np.asarray(column.pressure_pa, dtype=np.float64)
    t = np.asarray(column.temperature_k, dtype=np.float64)
    q = np.asarray(column.specific_humidity, dtype=np.float64)
    if p[0] > p[-1]:
        p, t, q = p[::-1], t[::-1], q[::-1]
    # Now ascending pressure: index 0 is the top.
    ncol = t.shape[1]
    ps = np.asarray(column.surface_pressure_pa, dtype=np.float64)

    # Isothermal extension above the top level.
    if p[0] > top_pa:
        p = np.concatenate([[top_pa], p])
        t = np.concatenate([t[:1], t], axis=0)
        q = np.concatenate([q[:1] * 0.0, q], axis=0)
    # Extension below the lowest level to the surface: over the ocean the
    # surface sits 10 to 20 hPa below the 1000 hPa analysis level, and
    # without this slab the column ended above the surface and the
    # surface-seeing channels missed its emission and absorption (1.4 K in
    # channel 4 at 1010 hPa).  The slab runs to a fixed BOTTOM_PA so the
    # layer set does not depend on which columns share a batch (the filter
    # evaluates subsets and needs the same kelvin), is refined
    # ``refinement * 4`` times so the trimmed sub-layer at each column's
    # surface is thin, and each column's own surface trims it below.  The
    # slab is appended whenever the level set stops above BOTTOM_PA, whether
    # or not this batch's surfaces reach into it, so the layer set is a
    # function of the level set alone and a column reads the same kelvin in
    # any company to the reduction order's last bit (1e-9 K is the bar the
    # tests hold).  Temperature follows the lowest layer's ln p slope
    # (bounded to 10 K), specific humidity is held.
    extended_bottom = False
    if p[-1] < BOTTOM_PA or ps.max() > p[-1]:
        p_bottom = max(BOTTOM_PA, ps.max() * (1.0 + 1.0e-6))
        slope = (t[-1] - t[-2]) / (np.log(p[-1]) - np.log(p[-2]))
        t_bottom = t[-1] + np.clip(slope * (np.log(p_bottom) - np.log(p[-1])), -10.0, 10.0)
        p = np.concatenate([p, [p_bottom]])
        t = np.concatenate([t, t_bottom[None, :]], axis=0)
        q = np.concatenate([q, q[-1:]], axis=0)
        extended_bottom = True
    parent = np.arange(p.size - 1)

    # Surface layer: between the lowest level above the surface and ps.
    # Levels below the surface are masked per column after refinement.
    ln_p = np.log(p)
    e = vapour_pressure_hpa(p[:, None], q)

    sub_lnp = []
    sub_t = []
    sub_e = []
    sub_parent = []
    sub_thickness = []
    for k in range(p.size - 1):
        n_sub = refinement * (4 if extended_bottom and k == p.size - 2 else 1)
        fractions = (np.arange(n_sub) + 0.5) / n_sub
        for frac in fractions:
            lnp_mid = ln_p[k] + frac * (ln_p[k + 1] - ln_p[k])
            sub_lnp.append(np.full(ncol, lnp_mid))
            sub_t.append(t[k] + frac * (t[k + 1] - t[k]))
            sub_e.append(e[k] + frac * (e[k + 1] - e[k]))
            sub_parent.append(parent[k])
            sub_thickness.append((ln_p[k + 1] - ln_p[k]) / n_sub)
    lnp_mid = np.asarray(sub_lnp)
    t_mid = np.asarray(sub_t)
    e_mid = np.maximum(np.asarray(sub_e), 0.0)
    thickness_lnp = np.asarray(sub_thickness)[:, None]
    p_mid = np.exp(lnp_mid)
    above = p_mid < ps[None, :]

    # The sub-layer straddling the surface is trimmed to the surface.
    lnp_lower = lnp_mid + thickness_lnp / 2.0
    lnp_upper = lnp_mid - thickness_lnp / 2.0
    ln_ps = np.log(ps)[None, :]
    straddles = (lnp_upper < ln_ps) & (lnp_lower >= ln_ps)
    trimmed_thickness = np.where(straddles, ln_ps - lnp_upper, thickness_lnp)
    above = above | straddles
    # Surface layer temperature blends toward the 2 m temperature where given.
    if column.air_temperature_2m_k is not None:
        t2 = np.asarray(column.air_temperature_2m_k, dtype=np.float64)[None, :]
        t_mid = np.where(straddles, 0.5 * (t_mid + t2), t_mid)

    # Hypsometric thickness in km, with virtual temperature.
    e_ratio = e_mid / (p_mid * 0.01)
    q_mid = EPSILON * e_ratio / (1.0 - (1.0 - EPSILON) * e_ratio)
    tv = t_mid * (1.0 + (1.0 / EPSILON - 1.0) * q_mid)
    dz_km = DRY_AIR_R * tv / GRAVITY_M_S2 * trimmed_thickness / 1000.0
    return Layers(
        p_hpa=p_mid * 0.01,
        t_k=t_mid,
        e_hpa=e_mid,
        dz_km=np.where(above, dz_km, 0.0),
        above_surface=above,
        parent_level=np.asarray(sub_parent),
        thickness_lnp=np.where(above, np.broadcast_to(trimmed_thickness, p_mid.shape), 0.0),
    )


def _integrate(frequencies, layers: Layers, skin_k, emissivity, sec_zenith, *,
               rayleigh_jeans: bool, absorption_layers: Layers | None = None,
               return_down: bool = False):
    """Top-of-atmosphere radiance ``(nfreq, ncol)`` and, for diagnostics,
    the per-layer weights ``(nfreq, nlay, ncol)`` and surface term.

    ``absorption_layers`` evaluates the optical depths on another layer
    set (the calibration's frozen absorption: emission from the planted
    column, opacity from the reference), so a planted temperature reads
    back through the weights alone."""
    opacity = absorption_layers if absorption_layers is not None else layers
    # P.676 takes the DRY-air pressure and the vapour partial pressure as two
    # arguments; the layer carries the total pressure, so the dry part is the
    # total less the vapour.  Handing it the total overstated the oxygen line
    # strength and width by e/p (up to 3 percent in a moist boundary layer)
    # and read 0.7 K warm in channels 3 and 4, 0.35 K in channel 5, 0.4 K in
    # channel 16 on the record day's columns (1.1 to 1.3 K at the wettest).
    alpha = absorption_coefficient(
        frequencies, opacity.p_hpa - opacity.e_hpa, opacity.t_k, opacity.e_hpa
    )
    dtau = alpha * opacity.dz_km[None, :, :] * sec_zenith[None, None, :]
    # Transmittance from the top of each layer to space (cumulative from
    # the top), and total.
    tau_above = np.concatenate(
        [np.zeros_like(dtau[:, :1, :]), np.cumsum(dtau, axis=1)[:, :-1, :]], axis=1
    )
    t_above = np.exp(-tau_above)
    absorbed = 1.0 - np.exp(-dtau)
    t_total = np.exp(-np.sum(dtau, axis=1))

    f = np.asarray(frequencies)[:, None, None]
    if rayleigh_jeans:
        b_layer = np.broadcast_to(layers.t_k[None, :, :], dtau.shape)
        b_skin = np.broadcast_to(skin_k[None, :], t_total.shape)
        b_cmb = COSMIC_BACKGROUND_K
    else:
        b_layer = planck_radiance(f, layers.t_k[None, :, :])
        b_skin = planck_radiance(np.asarray(frequencies)[:, None], skin_k[None, :])
        b_cmb = planck_radiance(np.asarray(frequencies)[:, None], COSMIC_BACKGROUND_K)

    up_weights = t_above * absorbed
    upwelling = np.sum(b_layer * up_weights, axis=1)
    # Downwelling at the surface: transmittance from each layer down to
    # the surface is total / (transmittance above the layer's bottom).
    tau_below = np.sum(dtau, axis=1, keepdims=True) - tau_above - dtau
    t_below = np.exp(-tau_below)
    down_weights = t_below * absorbed
    downwelling = np.sum(b_layer * down_weights, axis=1) + t_total * b_cmb

    e = emissivity[None, :]
    surface = t_total * (e * b_skin + (1.0 - e) * downwelling)
    radiance = upwelling + surface
    if return_down:
        return radiance, up_weights, t_total, down_weights
    return radiance, up_weights, t_total


def brightness_temperature(
    column: Column,
    channels: tuple[int, ...] | list[int],
    zenith_deg,
    scan_angle_deg,
    *,
    emissivity=None,
    refinement: int = 4,
    top_pa: float = 1.0,
    rayleigh_jeans: bool = False,
    absorption_reference: Column | None = None,
) -> np.ndarray:
    """Channel brightness temperatures ``(nchan, ncol)`` for the batch.

    ``emissivity`` overrides the ocean model with a fixed value or a
    ``(ncol,)`` array (the calibration's black surface, for instance).
    ``absorption_reference`` freezes the opacity on another column of the
    same levels (the calibration's linear reading)."""
    layers = build_layers(column, refinement=refinement, top_pa=top_pa)
    absorption_layers = (
        None if absorption_reference is None
        else build_layers(absorption_reference, refinement=refinement, top_pa=top_pa)
    )
    zenith = np.broadcast_to(np.asarray(zenith_deg, dtype=np.float64), (column.ncol,))
    scan = np.broadcast_to(np.asarray(scan_angle_deg, dtype=np.float64), (column.ncol,))
    sec = 1.0 / np.cos(np.deg2rad(zenith))
    skin = np.asarray(column.skin_temperature_k, dtype=np.float64)
    out = np.empty((len(channels), column.ncol))
    for row, number in enumerate(channels):
        channel = CHANNELS[number - 1]
        frequencies, weights = channel_quadrature(channel)
        if emissivity is None:
            e_v, e_h = fresnel_emissivity(channel.centre_ghz, skin, zenith)
            e = quasi_polarized_emissivity(e_v, e_h, scan, channel.polarization)
        else:
            e = np.broadcast_to(np.asarray(emissivity, dtype=np.float64), (column.ncol,))
        radiance, _, _ = _integrate(
            frequencies, layers, skin, e, sec, rayleigh_jeans=rayleigh_jeans,
            absorption_layers=absorption_layers,
        )
        if rayleigh_jeans:
            tb = radiance
        else:
            tb = planck_temperature(frequencies[:, None], radiance)
        out[row] = np.sum(weights[:, None] * tb, axis=0)
    return out


def temperature_jacobian(
    column: Column,
    channels: tuple[int, ...] | list[int],
    zenith_deg,
    scan_angle_deg,
    *,
    emissivity=None,
    refinement: int = 4,
    top_pa: float = 1.0,
) -> tuple[Layers, np.ndarray, np.ndarray]:
    """The brightness temperature's derivative to each refined layer's
    temperature, ``(layers, jacobian (nchan, nlay, ncol), tb (nchan, ncol))``:
    the layer's upwelling weight plus the part of its downwelling the
    surface reflects (``t_total (1 - e) down_weight``), each through the
    Planck derivative ratio ``B'(T_layer) / B'(T_B)`` (one in the
    Rayleigh-Jeans limit the 50 to 58 GHz band sits near).  With the
    absorption held (the frozen-opacity form the calibration reads a
    planted layer back through), ``T_B(x + dx) = T_B(x) + sum_l J_l dT_l``
    is the tangent-linear model the ensemble members are evaluated with
    about their mean column (:mod:`woof.globe.microwave.entry`);
    the residual to the full transfer is the absorption's temperature
    term, measured per window on a subsample and carried in the receipt."""
    layers = build_layers(column, refinement=refinement, top_pa=top_pa)
    zenith = np.broadcast_to(np.asarray(zenith_deg, dtype=np.float64), (column.ncol,))
    scan = np.broadcast_to(np.asarray(scan_angle_deg, dtype=np.float64), (column.ncol,))
    sec = 1.0 / np.cos(np.deg2rad(zenith))
    skin = np.asarray(column.skin_temperature_k, dtype=np.float64)
    jac = np.empty((len(channels), layers.p_hpa.shape[0], column.ncol))
    tb_out = np.empty((len(channels), column.ncol))
    for row, number in enumerate(channels):
        channel = CHANNELS[number - 1]
        frequencies, weights = channel_quadrature(channel)
        if emissivity is None:
            e_v, e_h = fresnel_emissivity(channel.centre_ghz, skin, zenith)
            e = quasi_polarized_emissivity(e_v, e_h, scan, channel.polarization)
        else:
            e = np.broadcast_to(np.asarray(emissivity, dtype=np.float64), (column.ncol,))
        radiance, up_weights, t_total, down_weights = _integrate(
            frequencies, layers, skin, e, sec, rayleigh_jeans=False, return_down=True,
        )
        tb = planck_temperature(frequencies[:, None], radiance)                      # (nfreq, ncol)
        f = np.asarray(frequencies)[:, None, None]
        # dB/dT at the layer temperature over dB/dT at the brightness
        # temperature: the Planck derivative ratio of the linearisation.
        db_layer = _planck_derivative(f, layers.t_k[None, :, :])
        db_tb = _planck_derivative(np.asarray(frequencies)[:, None], tb)
        total_weight = up_weights + (t_total * (1.0 - e))[:, None, :] * down_weights
        per_freq = total_weight * db_layer / db_tb[:, None, :]
        jac[row] = np.sum(weights[:, None, None] * per_freq, axis=0)
        tb_out[row] = np.sum(weights[:, None] * tb, axis=0)
    return layers, jac, tb_out


def _planck_derivative(f_ghz, t_k):
    """dB/dT of the Planck radiance at ``f_ghz`` and ``t_k`` (any units
    consistent with :func:`planck_radiance`; only ratios are read)."""
    x = PLANCK_H * np.asarray(f_ghz, dtype=np.float64) * 1.0e9 / (BOLTZMANN_K * np.asarray(t_k, dtype=np.float64))
    ex = np.exp(x)
    return ex * x / (t_k * (ex - 1.0) ** 2)


def surface_transmittance(column: Column, frequencies, zenith_deg, *, refinement: int = 4,
                          top_pa: float = 1.0) -> np.ndarray:
    """Total transmittance surface to space per frequency, ``(nfreq, ncol)``."""
    layers = build_layers(column, refinement=refinement, top_pa=top_pa)
    zenith = np.broadcast_to(np.asarray(zenith_deg, dtype=np.float64), (column.ncol,))
    sec = 1.0 / np.cos(np.deg2rad(zenith))
    skin = np.asarray(column.skin_temperature_k, dtype=np.float64)
    _, _, t_total = _integrate(
        np.asarray(frequencies, dtype=np.float64), layers, skin, np.ones(column.ncol), sec,
        rayleigh_jeans=True,
    )
    return t_total


def weighting_function(
    column: Column,
    number: int,
    zenith_deg,
    *,
    refinement: int = 4,
    top_pa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rayleigh-Jeans temperature weights of channel ``number`` per
    refined layer: ``(layer_pressure_hpa (nlay, ncol), weights (nlay, ncol),
    surface_transmittance (ncol,))``.  The weights sum with the surface
    term to one; their peak names the layer a channel sounds."""
    layers = build_layers(column, refinement=refinement, top_pa=top_pa)
    zenith = np.broadcast_to(np.asarray(zenith_deg, dtype=np.float64), (column.ncol,))
    sec = 1.0 / np.cos(np.deg2rad(zenith))
    channel = CHANNELS[number - 1]
    frequencies, weights = channel_quadrature(channel)
    skin = np.asarray(column.skin_temperature_k, dtype=np.float64)
    _, up_weights, t_total = _integrate(
        frequencies, layers, skin, np.ones(column.ncol), sec, rayleigh_jeans=True
    )
    band_weights = np.sum(weights[:, None, None] * up_weights, axis=0)
    band_total = np.sum(weights[:, None] * t_total, axis=0)
    return layers.p_hpa, band_weights, band_total
