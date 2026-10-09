"""Ocean surface emissivity for the microwave operator: specular Fresnel
reflection from sea water with the Meissner and Wentz (2004) double-Debye
dielectric constant.

Model, stated plainly:

* Dielectric constant of sea water, Meissner and Wentz, IEEE TGRS 42(9),
  2004 ("The complex dielectric constant of pure and sea water from
  microwave satellite observations"), the double-Debye fit with salinity
  corrections, salinity fixed at 35 psu.  Valid 1 to 90 GHz by the fit;
  the 50 to 58 GHz sounding channels sit inside it.
* Reflection from a flat (specular) surface, Fresnel coefficients for
  vertical and horizontal polarization at the local zenith angle;
  emissivity is one minus reflectivity (Kirchhoff).
* No wind roughness and no foam.  Both raise the emissivity by a few
  hundredths at 50 GHz for a 10 m/s wind, so channels 3 and 4, whose
  surface transmittance is 0.3 to 0.6, read cold by a few tenths of a
  kelvin per m/s of wind in this model; channels 5 to 15 do not see the
  surface at the tenth-of-a-kelvin level.  The wind dependence of the
  channel 3 and 4 residual is one of the diagnostics the scoring
  records, so the size of this omission is measured on the day.
* Quasi-polarization mixing with the scan angle is applied by the
  radiative transfer, not here.
"""

from __future__ import annotations

import numpy as np

SALINITY_PSU = 35.0
COSMIC_BACKGROUND_K = 2.725


def sea_water_permittivity(f_ghz, sst_k, salinity_psu: float = SALINITY_PSU) -> np.ndarray:
    """Complex relative permittivity (eps' - i eps'') of sea water,
    Meissner and Wentz (2004) double-Debye with salinity corrections.

    ``f_ghz`` and ``sst_k`` broadcast."""
    f = np.asarray(f_ghz, dtype=np.float64)
    t = np.asarray(sst_k, dtype=np.float64) - 273.15
    s = float(salinity_psu)

    # Pure water (Table III of the paper).
    eps_s0 = (3.70886e4 - 8.2168e1 * t) / (4.21854e2 + t)
    eps_10 = 5.7230 + 2.2379e-2 * t - 7.1237e-4 * t ** 2
    nu_10 = (45.0 + t) / (5.0478 - 7.0315e-2 * t + 6.0059e-4 * t ** 2)
    eps_inf0 = 3.6143 + 2.8841e-2 * t
    nu_20 = (45.0 + t) / (1.3652e-1 + 1.4825e-3 * t + 2.4166e-4 * t ** 2)

    # Conductivity of sea water (Stogryn form as fitted in the paper).
    sigma_35 = (
        2.903602 + 8.607e-2 * t + 4.738817e-4 * t ** 2
        - 2.991e-6 * t ** 3 + 4.3047e-9 * t ** 4
    )
    r_15 = s * (37.5109 + 5.45216 * s + 1.4409e-2 * s ** 2) / (1004.75 + 182.283 * s + s ** 2)
    alpha_0 = (6.9431 + 3.2841 * s - 9.9486e-2 * s ** 2) / (84.85 + 69.024 * s + s ** 2)
    alpha_1 = 49.843 - 0.2276 * s + 0.198e-2 * s ** 2
    r_tr15 = 1.0 + (t - 15.0) * alpha_0 / (alpha_1 + t)
    sigma = sigma_35 * r_15 * r_tr15  # S/m

    # Salinity corrections of the Debye parameters.
    eps_s = eps_s0 * np.exp(-3.56417e-3 * s + 4.74868e-6 * s ** 2 + 1.15574e-5 * t * s)
    nu_1 = nu_10 * (1.0 + s * (2.39357e-3 - 3.13530e-5 * t + 2.52477e-7 * t ** 2))
    eps_1 = eps_10 * np.exp(-6.28908e-3 * s + 1.76032e-4 * s ** 2 - 9.22144e-5 * t * s)
    nu_2 = nu_20 * (1.0 + s * (-1.99723e-2 + 1.81176e-4 * t))
    eps_inf = eps_inf0 * (1.0 + s * (-2.04265e-3 + 1.57883e-4 * t))

    j = 1j
    eps = (
        (eps_s - eps_1) / (1.0 + j * f / nu_1)
        + (eps_1 - eps_inf) / (1.0 + j * f / nu_2)
        + eps_inf
        - j * sigma / (2.0 * np.pi * 8.8541878128e-12 * f * 1.0e9)
    )
    return eps


def fresnel_emissivity(f_ghz, sst_k, zenith_deg, salinity_psu: float = SALINITY_PSU):
    """``(e_v, e_h)`` specular emissivities at the local zenith angle.

    All three inputs broadcast."""
    eps = sea_water_permittivity(f_ghz, sst_k, salinity_psu)
    theta = np.deg2rad(np.asarray(zenith_deg, dtype=np.float64))
    cos_t = np.cos(theta)
    sin2 = np.sin(theta) ** 2
    root = np.sqrt(eps - sin2)
    r_v = (eps * cos_t - root) / (eps * cos_t + root)
    r_h = (cos_t - root) / (cos_t + root)
    e_v = 1.0 - np.abs(r_v) ** 2
    e_h = 1.0 - np.abs(r_h) ** 2
    return e_v, e_h


def quasi_polarized_emissivity(e_v, e_h, scan_angle_deg, polarization: str):
    """Mix the V and H emissivities the way the rotating ATMS reflector
    does: QV reads V at nadir and H at 90 degrees of scan, QH the
    reverse."""
    a = np.deg2rad(np.asarray(scan_angle_deg, dtype=np.float64))
    c2 = np.cos(a) ** 2
    s2 = np.sin(a) ** 2
    if polarization == "QV":
        return e_v * c2 + e_h * s2
    if polarization == "QH":
        return e_h * c2 + e_v * s2
    raise ValueError(f"polarization must be QV or QH, not {polarization!r}")
