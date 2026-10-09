"""A spectral eddy viscosity derived from closure theory: the drain at the
truncation is set by the resolved flow's own energy there, not tuned.

The shipped hyperdiffusion (``ExponentialHyperdiffusion``) removes energy
at a fixed e-folding time at the truncation, a number chosen on a ladder
of arms.  This module replaces that number with the eddy viscosity the
two-point closure theory of turbulence (EDQNM; Kraichnan 1976, Chollet and
Lesieur 1981, Lesieur and Metais 1996; on the sphere Frederiksen and
Davies 1997) assigns to a truncation inside a forward energy cascade:

    nu(k | k_c) = nu_plus(k / k_c) * sqrt(E(k_c) / k_c)
    nu_plus(x)  = plateau + cusp_amplitude * exp(-cusp_decay / x)

with k_c the cutoff wavenumber, E(k_c) the kinetic-energy spectrum
density at the cutoff (m^3/s^2 per unit wavenumber) read from the model's
own coefficients every step, and the constants 0.267, 9.21 and 3.03 the
EDQNM values for a Kolmogorov constant of 1.4.  The plateau is the
eddy viscosity the unresolved eddies exert on the large scales; the cusp
near the cutoff is the local (triad) transfer across it.  With E(k_c) on
a k^-5/3 range the total drain below k_c equals the cascade rate the
spectrum implies (the Kolmogorov relation E(k) = C_K eps^(2/3) k^(-5/3)
solved for eps), which is the property the constants were derived for
and the property ``tests/test_arwen_global_spectral_eddy_viscosity.py``
holds.  Scalars (potential temperature, vapor, log surface pressure) take
the eddy diffusivity nu / Pr_t with the EDQNM eddy Prandtl number 0.6
(Lesieur and Rogallo 1989).

On the sphere the wavenumber of total degree n is k_n = sqrt(n (n + 1)) / a
and the spectrum density is the per-degree energy over the degree spacing,
E(k_n) = E_n / (k_n - k_{n-1}).  E(k_c) is read as the mean over the last
``tail_degrees`` degrees of E_n compensated to the truncation along the
inertial slope, so one noisy last degree does not set the drain.

The operator is applied as an exact exponential factor per degree and
per level (every level reads its own E(k_c)), unconditionally stable, the
same way the hyperdiffusion is applied; degrees up to ``preserve_degree``
are left untouched.  The theory constants are exposed so a sweep can
measure the sensitivity to them; at their defaults nothing here is tuned.
Not built: the stochastic backscatter term of the same closure (energy
returned from below the cutoff), which EDQNM also predicts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .transform import SphericalHarmonicTransform

#: EDQNM eddy-viscosity plateau for a Kolmogorov constant of 1.4
#: (Chollet and Lesieur 1981).
EDQNM_PLATEAU = 0.267
#: Cusp coefficient and decay of the same closure: nu_plus(x) = 0.267 +
#: 9.21 exp(-3.03 / x), x = k / k_c.
EDQNM_CUSP_AMPLITUDE = 9.21
EDQNM_CUSP_DECAY = 3.03
#: Eddy Prandtl number of the closure for a passive scalar (Lesieur and
#: Rogallo 1989).
EDQNM_EDDY_PRANDTL = 0.6
#: The Kolmogorov constant the constants above were derived under.
KOLMOGOROV_CONSTANT = 1.4
#: The inertial slope the tail average is compensated along.
INERTIAL_SLOPE = -5.0 / 3.0


@dataclass(frozen=True)
class SpectralEddyViscosity:
    plateau: float = EDQNM_PLATEAU
    cusp_amplitude: float = EDQNM_CUSP_AMPLITUDE
    cusp_decay: float = EDQNM_CUSP_DECAY
    eddy_prandtl: float = EDQNM_EDDY_PRANDTL
    tail_degrees: int = 5
    inertial_slope: float = INERTIAL_SLOPE
    preserve_degree: int = 1

    def __post_init__(self) -> None:
        for name in ("plateau", "cusp_decay", "eddy_prandtl"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.cusp_amplitude) or self.cusp_amplitude < 0.0:
            raise ValueError("cusp_amplitude must be finite and nonnegative")
        if self.tail_degrees < 1:
            raise ValueError("tail_degrees must be >= 1")
        if not math.isfinite(self.inertial_slope) or self.inertial_slope >= 0.0:
            raise ValueError("inertial_slope must be finite and negative")
        if self.preserve_degree < 0:
            raise ValueError("preserve_degree must be >= 0")

    # ----------------------------------------------------------------- geometry
    @staticmethod
    def wavenumbers(transform: SphericalHarmonicTransform) -> np.ndarray:
        """k_n = sqrt(n (n + 1)) / a for n = 0..T (rad/m); k_0 = 0."""
        n = np.arange(transform.truncation + 1, dtype=np.float64)
        return np.sqrt(n * (n + 1.0)) / float(transform.grid.radius_m)

    def nu_plus(self, x) -> np.ndarray:
        """The dimensionless eddy viscosity nu_plus(k / k_c); zero at k = 0."""
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x)
        positive = x > 0.0
        out[positive] = self.plateau + self.cusp_amplitude * np.exp(-self.cusp_decay / x[positive])
        return out

    # ------------------------------------------------------------ the reading
    def energy_density_at_truncation(self, ke_by_degree: np.ndarray,
                                     transform: SphericalHarmonicTransform) -> np.ndarray:
        """E(k_c) in m^3/s^2 per unit wavenumber from per-degree kinetic
        energy ``ke_by_degree[..., n]`` (m^2/s^2): the last ``tail_degrees``
        degrees compensated to the truncation along the inertial slope,
        averaged, divided by the degree spacing at the truncation."""
        ke = np.asarray(ke_by_degree, dtype=np.float64)
        truncation = int(transform.truncation)
        if ke.shape[-1] != truncation + 1:
            raise ValueError(
                f"ke_by_degree has {ke.shape[-1]} degrees, the transform {truncation + 1}"
            )
        tail = min(int(self.tail_degrees), truncation)
        k = self.wavenumbers(transform)
        k_c = k[truncation]
        degrees = np.arange(truncation + 1 - tail, truncation + 1)
        # E_n ~ k_n^slope on the inertial range, so E_N is estimated from
        # E_n by (k_c / k_n)^slope.
        compensation = (k_c / k[degrees]) ** self.inertial_slope
        e_trunc = np.mean(ke[..., degrees] * compensation, axis=-1)
        spacing = k[truncation] - k[truncation - 1]
        return np.maximum(e_trunc, 0.0) / spacing

    def nu_infinity(self, ke_by_degree: np.ndarray,
                    transform: SphericalHarmonicTransform) -> np.ndarray:
        """The plateau eddy viscosity, plateau * sqrt(E(k_c) / k_c), per
        leading index of ``ke_by_degree`` (m^2/s)."""
        k_c = self.wavenumbers(transform)[int(transform.truncation)]
        density = self.energy_density_at_truncation(ke_by_degree, transform)
        return self.plateau * np.sqrt(density / k_c)

    # ------------------------------------------------------------ the operator
    def eddy_viscosity_by_degree(self, nu_infinity, transform: SphericalHarmonicTransform) -> np.ndarray:
        """nu_e(n) = (nu_plus(k_n / k_c) / plateau) * nu_infinity, shape
        ``nu_infinity.shape + (T + 1,)`` (m^2/s)."""
        k = self.wavenumbers(transform)
        x = k / k[int(transform.truncation)]
        shape = self.nu_plus(x) / self.plateau
        nu_inf = np.asarray(nu_infinity, dtype=np.float64)
        return nu_inf[..., None] * shape[None, :] if nu_inf.ndim else nu_inf * shape

    def factors(self, transform: SphericalHarmonicTransform, dt_s: float, nu_infinity):
        """Per-degree multiplier on a velocity-like coefficient over one
        step, exp(-dt nu_e(n) k_n^2), shape ``nu_infinity.shape + (T + 1,)``
        on the transform's device in its float dtype; degrees up to
        ``preserve_degree`` read one.  Kinetic energy decays at twice the
        exponent, as under the hyperdiffusion."""
        xp = transform.backend.xp
        k = self.wavenumbers(transform)
        nu_e = self.eddy_viscosity_by_degree(nu_infinity, transform)
        exponent = -float(dt_s) * nu_e * (k * k)
        factor = np.exp(exponent)
        factor[..., : self.preserve_degree + 1] = 1.0
        return xp.asarray(factor, dtype=transform.backend.float_dtype)

    def apply(self, coeff, transform: SphericalHarmonicTransform, dt_s: float, nu_infinity, *,
              prandtl: float = 1.0):
        """Apply the drain to ``coeff[..., n, m]``; ``nu_infinity`` is one
        value per leading index of ``coeff`` (per level) or a scalar for a
        two-dimensional field.  ``prandtl`` > 0 divides the viscosity
        (a scalar field takes nu / Pr_t)."""
        if not math.isfinite(prandtl) or prandtl <= 0.0:
            raise ValueError("prandtl must be finite and positive")
        factor = self.factors(transform, dt_s, nu_infinity)
        if prandtl != 1.0:
            factor = factor ** (1.0 / float(prandtl))
        return transform.project(coeff * factor[..., None])

    def describe(self) -> dict[str, object]:
        return {
            "closure": "spectral_eddy_viscosity",
            "plateau": float(self.plateau),
            "cusp_amplitude": float(self.cusp_amplitude),
            "cusp_decay": float(self.cusp_decay),
            "eddy_prandtl": float(self.eddy_prandtl),
            "tail_degrees": int(self.tail_degrees),
            "inertial_slope": float(self.inertial_slope),
            "preserve_degree": int(self.preserve_degree),
            "kolmogorov_constant": KOLMOGOROV_CONSTANT,
        }


def cascade_rate_from_spectrum(energy_density_at_cutoff: float, k_c: float,
                               kolmogorov_constant: float = KOLMOGOROV_CONSTANT) -> float:
    """eps from E(k_c) = C_K eps^(2/3) k_c^(-5/3): the cascade rate a
    Kolmogorov spectrum through the cutoff implies (m^2/s^3)."""
    return (float(energy_density_at_cutoff) * k_c ** (5.0 / 3.0) / kolmogorov_constant) ** 1.5
