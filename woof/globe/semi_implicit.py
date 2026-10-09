"""Semi-implicit treatment of the hydrostatic gravity-wave modes.

Two schemes share one seam (:meth:`linear_tendencies` subtracted from the
explicit right-hand side; ``pre_apply`` / ``post_apply`` integrating
exactly that linear operator by off-centred Crank-Nicolson around the
explicit step, through ``MoistHybridModel.integrate_dynamics``):

* :class:`VerticalModeSemiImplicit` (default) treats EVERY vertical mode
  implicitly through the linear gravity-wave operator of the
  hybrid-coordinate hydrostatic equations about an isothermal reference
  state, derived below in the model's own discrete forms.
* :class:`BarotropicSemiImplicit` treats the external (barotropic) mode
  only through a single reference speed.  It stays selectable for
  identity-locked archives and A/B runs and is bit-identical to its era.

Why the external-only scheme is no longer the default (audit 2026-09-01,
DN-1): it leaves every internal mode explicit, so the rest-state step
ceiling at T533 was 104.7 s regardless of the reference speed and the
advective CFL gate did not bound the step (dt=60 s linearly unstable at
142.7 m/s while the gate tripped at 149.3).  Measured consequence at
250 hPa: more than half the mesoscale kinetic energy beyond ~330 km was
divergent (0.7 at 100-200 km against 0.35-0.45 in GFS/IFS) - explicit,
undamped internal gravity waves.  On the 40-level surface_stretched
default stack the external-only residual is linearly unstable at rest
even at the shipped dt=30 s (growth 7.1e-6 per step at n=257 with the
shipped hyperdiffusion; strict ceiling 1.3 s at T533, 32 s at T21).

The split is subtract-then-integrate (commit 854c97d5f): the explicit
integrator never sees the linear operator, and the corrector never
re-integrates terms the explicit step already carried - applying
Crank-Nicolson on top of a right-hand side that still contained the wave
double-stepped the external mode, and the first real-data T63 run rode
that amplification to 1256 hPa by hour 9.

WHERE the corrector sits matters at a balanced state.  With F(y_b) = 0
the explicit part integrates -L y_b, a tendency as large as the terrain's
own pressure-gradient force, and one Crank-Nicolson step after it does
not undo that beyond first order: at rest over a 2 km mountain (T42,
20-level pressure_blend, the auditor's t6 state, rhs residual 4e-7
m/s^2) the corrector-after-the-step order reaches 2.7 m/s and 3.2 K
within one hour at dt = 60 s under the external proxy (14 m/s, 16.6 K at
dt = 300) and 1.6 m/s, 1.2 K under this operator.  The vertical-mode
scheme therefore applies the SQUARE ROOT of the Crank-Nicolson map on
either side of the explicit step (H R3 H with H^2 = CN, a symmetric
split): same linearized stability as the corrector-after order (the two
are similar matrices; measured ceilings identical to the rung), and the
same hour measures 8e-3 m/s and 0.25 K at dt = 60 s (0.18 m/s, 1.5 K at
dt = 300).  The theta drift that remains scales with dt and is the
first-order split's residual against the reference (DN-2): the per-step
commutator of the coded Jacobian with the reference operator over
terrain.  The default [time] integrator no longer runs this split: the
IMEX Runge-Kutta pair of imex.py advances ``rhs`` (still ``F - L``) and
``L`` with shared abscissae, so every balanced state is an exact fixed
point (the same hour measures 4.2e-4 m/s and 5.5e-4 K at both steps, the
explicit residual's own floor); its solves go through
:meth:`VerticalModeSemiImplicit.solve_shifted`.  The pair was chosen by
measurement on the linearized coded step (ARS(2,3,2) measured unstable
at rest on this operator, 7e-5 per step at T533 dt = 60; the
per-substep IMEX Euler form is exact at rest and unstable under
advection, 2.3e-3 per step at U = 50 m/s; imex.py records the family
and the member).  Under ``integrator = "ssprk3"`` or ``"rk4"`` the
symmetric split below runs bit-identically to its era, and the external
proxy keeps its corrector-after order.

DERIVATION of the vertical-mode operator.  Prognostic variables per
spherical harmonic of total degree n (k^2 = n(n+1)/a^2): divergence D_k,
potential temperature theta_k, ln ps.  Linearized about rest, an
isothermal T_ref, a uniform ps_ref and no rotation, in exactly the discrete
forms the model integrates (dynamics.rhs, vertical.py):

  continuity      ln ps_t = -w . D,          w_k = dp_k / ps_ref
  interface omega omega_j = -sum_{k<j} (dB_k ps_t + dp_k D_k)
  thermodynamic   theta_t = -Gamma D, the flux-form theta tendency of
                  rhs() at the reference profile theta_ref = T_ref /
                  exner: horizontal flux dp theta_ref D, vertical flux
                  omega_j f_j with f_j the interface value of theta_ref
                  reconstructed the way _vertical_scalar_flux_divergence
                  does (van Leer limited, bounded) averaged over the two
                  upstream directions - the central-difference
                  linearization of the coded upwind switch - and the
                  -theta dp_t term.

  The averaged face value is a DOCUMENTED DIVERGENCE from the coded flux.
  The switch is C0: at each interface the code takes the reconstruction
  from the upstream side, chosen by the sign of omega_j, so the operator
  subtracts the mean of two branches neither of which the code integrates
  at any one interface.  Each one-sided branch differs from the average by
  up to 245 K per unit divergence at level 0 on the 20-level pressure_blend
  stack (99% of max|C|) and 63 K on the 40-level surface_stretched default
  (34% of max|C| = 185 K), in the theta rows of the top two layers only
  (the next two levels are 6.8 / 2.2 K and 1.9 / 1.7 K).  Near rest the
  explicit residual is therefore piecewise-linear in the sign of omega,
  with residual speeds sqrt|eig(-B dC)| of 72.6 / 32.4 m/s (20-level) and
  20.7 / 7.7 m/s (40-level) for the two leading residual modes, which the
  explicit step carries inside its advective budget; both one-sided
  branches still give real positive mode spectra (354.9 / 232.3 and 357.6
  / 252.1 m/s for the leading pair on the 20-level stack against the
  averaged 356.2 / 242.6).
  divergence      D_t = k^2 (phi' + R Tv_k factor_k ln ps'), the
                  Laplacian of the Bernoulli geopotential plus the
                  divergence of the pressure-gradient force with
                  factor_k = d ln p_k / d ln ps exactly on the grid
                  (_pressure_gradient_factor), where phi' = G Tv' +
                  R T_ref (1 - factor) ln ps' and G is the model's own
                  hydrostatic integration (Tv linear in ln p over the half
                  layer, DN-3) applied to unit temperature perturbations,
                  and Tv' = exner theta' + kappa T_ref factor ln ps'
                  because theta, not T, is prognostic.

Collecting x = (theta', ln ps'): D_t = k^2 B x, x_t = C D, and the
vertical-structure matrix M = -B C (nlev x nlev) has real positive
eigenvalues c_j^2 (equivalent depths c_j^2 / g).  Checked against the
central-difference Jacobian of the coded rhs at an isothermal rest state:
B to 3.3e-10 and C to 2.5e-15 relative on the 20-level pressure_blend
stack, 2.1e-9 / 4.0e-15 on the 40-level surface_stretched default
(tests/test_arwen_global_vertical_modes.py).  Mode speeds at
T_ref = 320 K: 20-level pressure_blend 356.2 / 242.6 / 149.9 / 97.3 /
68.5 ... m/s (equivalent depths 12935 / 6004 / 2292 / 965 m); 40-level
surface_stretched 354.9 / 227.4 / 151.5 / 109.7 / 84.8 ... m/s (12843 /
5271 / 2340 / 1226 m); the audit's linearization of the 286/220 K column gave
325.4 / 191.5 / 120.7 / 81.2, i.e. the reference operator over-covers
every mode of that column, which is the stability requirement of the
split (a reference slower than the atmosphere's own mode leaves a
residual explicit wave; audit DN-2's 2x2 analysis and the module's
former 450 m/s rule).

The Crank-Nicolson step of L = [[0, k^2 B], [C, 0]] with implicit weight
alpha (0.5 neutral, IFS-style off-centring is alpha = (1 + eps) / 2) is
solved per total degree through the Helmholtz reduction

  (I + alpha^2 dt^2 k^2 M) D^{n+1} = D* + dt k^2 B x*
                                     - alpha (1 - alpha) dt^2 k^2 M D*
  x^{n+1} = x* + dt C [(1 - alpha) D* + alpha D^{n+1}]

with (I + s M)^{-1} = V diag(1 / (1 + s c_j^2)) V^{-1} in the modal
basis - one nlev x nlev matrix per degree, cached per (dt, alpha) - so
every mode receives exactly the scalar off-centred Crank-Nicolson factor
g = (1 + i(1-alpha) c k dt) / (1 - i alpha c k dt): |g| = 1 at alpha =
0.5, < 1 above.  That is :meth:`VerticalModeSemiImplicit.apply`.  The
half map the model runs, H = sqrt(CN), uses that L^2 is block diagonal
(k^2 B C on divergence, k^2 C B on the thermodynamic pair, both with the
modes' -omega_j^2 on the diagonal in their modal bases), so any real
function of L is P + Q L with P, Q functions of L^2: per mode p_j +
i q_j omega_j = sqrt(g_j) (principal root; the null mode of B and degree
zero give p = 1, q = dt/2).  H^2 = CN checked to 2e-15
(:meth:`VerticalModeSemiImplicit.half_apply`).

Reference temperature (fixed by measurement).  The instrument is the
linearized rest ceiling of the coded step (H R3 H with H = sqrt of the
off-centred Crank-Nicolson map, alpha = 0.5, the whole-rhs Jacobian
rescaled by k^2) over four columns with surface temperature 286 / 300 /
320 / 340 K linear in ln p to 220 K; the rule is to maximize the MINIMUM
over those columns, which requires the reference to over-cover every
column's external mode: a colder reference collapses once a column's
external mode outruns it, a warmer one enlarges the explicit residual of
the first-order split (DN-2).

  20-level pressure_blend, T533, no diffusion: T_ref = 300 K gives 572 /
  564 / 591 / 273 s, 320 K gives 449 / 501 / 498 / 523 s, 350 K gives
  292 / 366 / 426 / 426 s (with the shipped hyperdiffusion, min over
  degrees: 445 / 602 / 606 / 639 s at 320 K).
  40-level surface_stretched (the default stack), T533, no diffusion:
  280 K gives 1818 / 366 / 234 / 186 s, 300 K 831 / 1136 / 397 / 242 s,
  320 K 400 / 669 / 1006 / 437 s, 340 K 280 / 376 / 717 / 912 s, 350 K
  246 / 315 / 522 / 758 s; with the shipped hyperdiffusion (order 4,
  14400 s at truncation, min over degrees) the minima are 186 / 242 /
  395 / 276 / 243 s.  T255 scales by k at rest: 320 K gives 835 / 1397 /
  2100 / 912 s (868 s minimum with diffusion) against 504 s at 300 K.

320 K maximizes the minimum on both stacks (400 s on the default against
242 s at 300 K, where the 340 K column's external mode at 350.4 m/s
outruns the 300 K operator's 343.6; the 320 K operator's 354.9 covers all
four).  On the 286 K column alone the ordering reverses (831 / 400 / 246 s
at 300 / 320 / 350 K on the default stack, 572 / 449 / 292 on the
20-level): that is the single-column reading, not the rule, and a
reference tuned to it would leave every warmer column with an explicit
residual wave.  The external-only scheme measures 100.8 s on every column
of the 20-level ladder and 1.3 s on the default stack.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np

from .spectral.backend import device_cache_key
from .constants import DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2, KAPPA, REFERENCE_PRESSURE_PA
from .state import MoistHybridState
from .vertical import HybridCoordinate


def _metal_operator_resource_cache(transform, context):
    """Metal copies of immutable semi-implicit tables, scoped to a transform."""
    cache = getattr(transform, "_semi_implicit_metal_resources", None)
    if cache is None or cache["context"] is not context:
        cache = {"context": context, "solve": {}, "half_apply": {}}
        transform._semi_implicit_metal_resources = cache
    return cache


class LinearTendencies(NamedTuple):
    """The weighted linear pair/triple the explicit right-hand side must
    not integrate.  ``divergence`` broadcasts against the (nlev, n, m)
    divergence stack (2-D level-independent for the barotropic proxy, 3-D
    for the vertical-mode operator); ``theta`` is None when the scheme
    does not touch the thermodynamic variable."""

    divergence: object
    theta: object | None
    log_surface_pressure: object


@dataclass(frozen=True)
class BarotropicSemiImplicit:
    """External-mode-only proxy: one reference speed, one weighted
    oscillator dD/dt = w c^2 k^2 p, dp/dt = -w Dbar.

    Stability of the residual explicit system requires the reference speed
    to be at least the model's own effective barotropic coupling speed,
    which the hybrid coordinate pushes ABOVE the textbook sqrt(R * Tbar):
    measured on a 286-K-surface atmosphere by perturbing one lnps mode,
    the per-level coupling implies 297..369 m/s and the delta-B-weighted
    barotropic mode 331 m/s, scaling like the square root of column
    temperature.  The 450 m/s default covers every state the enforce()
    bounds admit with >=1.2 margin (over-implicit only distorts fast-mode
    phase, which nothing here reads); a config that lowers it below the
    atmosphere it runs owns the residual explicit wave it reintroduces.
    Every internal mode stays explicit (see the module docstring).
    """

    enabled: bool = True
    external_wave_speed_m_s: float = 450.0
    divergence_weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.external_wave_speed_m_s) or self.external_wave_speed_m_s <= 0.0:
            raise ValueError("external_wave_speed_m_s must be finite and positive")
        if not math.isfinite(self.divergence_weight) or not 0.0 <= self.divergence_weight <= 1.0:
            raise ValueError("divergence_weight must lie in [0,1]")

    @property
    def active(self) -> bool:
        return self.enabled and self.divergence_weight > 0.0

    @property
    def scheme(self) -> str:
        return "external"

    def explicit_wave_speed_m_s(self, vertical) -> float:
        """The gravity-wave speed the explicit budget still carries: the
        reference speed times the fraction the split leaves explicit."""
        if self.active:
            return (1.0 - self.divergence_weight) * self.external_wave_speed_m_s
        return self.external_wave_speed_m_s

    def describe(self, vertical) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "enabled": bool(self.enabled),
            "external_wave_speed_m_s": float(self.external_wave_speed_m_s),
            "weight": float(self.divergence_weight),
        }

    def _mean_weights(self, transform, vertical):
        return transform.backend.asarray(
            vertical.delta_b / np.sum(vertical.delta_b),
            dtype=transform.backend.float_dtype,
        )

    def _wave_operator(self, transform):
        xp = transform.backend.xp
        n = xp.arange(transform.truncation + 1, dtype=transform.backend.float_dtype)
        return n * (n + 1.0) / (transform.grid.radius_m ** 2)

    def linear_tendencies(self, state: MoistHybridState, transform, vertical):
        """Weighted linear pair the explicit RHS must not integrate.

        The divergence field is level-independent (the proxy acts on the
        barotropic mode only; subtracting the same spectral field from
        every level leaves each baroclinic difference untouched).  Returns
        ``None`` when inactive.
        """
        if not self.active:
            return None
        xp = transform.backend.xp
        weights = self._mean_weights(transform, vertical)
        barotropic = xp.sum(weights[:, None, None] * state.divergence, axis=0)
        k2 = self._wave_operator(transform)
        w = self.divergence_weight
        c2 = self.external_wave_speed_m_s ** 2
        divergence_t = w * c2 * k2[:, None] * state.log_surface_pressure
        logps_t = -w * barotropic
        divergence_t = transform.project(divergence_t[None])[0]
        logps_t = transform.project(logps_t)
        return LinearTendencies(divergence_t, None, logps_t)

    def pre_apply(self, state: MoistHybridState, transform, vertical, dt_s: float):
        """The proxy is a Lie split: nothing happens before the explicit
        step, and the state passes through untouched (bit-identity)."""
        return state, {"semi_implicit_max_divergence_increment_s1": 0.0}

    def post_apply(self, state: MoistHybridState, transform, vertical, dt_s: float):
        return self.apply(state, transform, vertical, dt_s)

    def solve_shifted(self, state: MoistHybridState, transform, vertical, tau_s: float):
        """``(I - tau w L) y = r`` for the weighted pair: the implicit stage
        solve of the IMEX integrator (imex.py).  Level by level ``D = r_D +
        tau w c^2 k^2 p`` and, through the barotropic mean, ``p = (r_p - tau
        w rbar_D) / (1 + tau^2 w^2 c^2 k^2)``.  Inactive: identity."""
        if not self.active:
            return state
        xp = transform.backend.xp
        weights = self._mean_weights(transform, vertical)
        barotropic = xp.sum(weights[:, None, None] * state.divergence, axis=0)
        p = state.log_surface_pressure
        k2 = self._wave_operator(transform)
        tw = float(tau_s) * self.divergence_weight
        c2k2 = (self.external_wave_speed_m_s ** 2) * k2[:, None]
        p_new = (p - tw * barotropic) / (1.0 + tw * tw * c2k2)
        correction = tw * c2k2 * p_new
        p_new[0, 0] = p[0, 0]
        correction[0, 0] = 0.0
        divergence = transform.project(state.divergence + correction[None])
        logps = transform.project(p_new)
        return state.with_fields(
            (
                state.vorticity,
                divergence,
                state.theta,
                logps,
                state.qv,
                state.qc,
                state.qr,
                state.qi,
                state.qs,
                state.qg,
            )
        )

    def apply(self, state: MoistHybridState, transform, vertical, dt_s: float):
        """Crank-Nicolson integration of the subtracted linear pair.

        Exactly the (I - h/2 M)^-1 (I + h/2 M) step of the weighted
        oscillator dD/dt = w c^2 k^2 p, dp/dt = -w D: neutral amplitude
        (|amplification| = 1) at every dt, which is what makes the wave's
        implicit treatment unconditionally stable.
        """
        if not self.active:
            return state, {"semi_implicit_max_divergence_increment_s1": 0.0}
        xp = transform.backend.xp
        weights = self._mean_weights(transform, vertical)
        barotropic = xp.sum(weights[:, None, None] * state.divergence, axis=0)
        p = state.log_surface_pressure
        k2 = self._wave_operator(transform)
        half = 0.5 * float(dt_s) * self.divergence_weight
        a = half * (self.external_wave_speed_m_s ** 2) * k2[:, None]
        b = half
        rhs_d = barotropic + a * p
        rhs_p = p - b * barotropic
        denominator = 1.0 + a * b
        d_new = (rhs_d + a * rhs_p) / denominator
        p_new = rhs_p - b * d_new
        d_new[0, 0] = barotropic[0, 0]
        p_new[0, 0] = p[0, 0]
        correction = d_new - barotropic
        divergence = transform.project(state.divergence + correction[None])
        logps = transform.project(p_new)
        maximum = float(
            np.max(np.abs(transform.backend.to_numpy(correction)))
        )
        return state.with_fields(
            (
                state.vorticity,
                divergence,
                state.theta,
                logps,
                state.qv,
                state.qc,
                state.qr,
                state.qi,
                state.qs,
                state.qg,
            )
        ), {"semi_implicit_max_divergence_increment_s1": maximum}


class VerticalStructureOperator:
    """The linear gravity-wave operator of one hybrid stack about an
    isothermal reference: matrices ``b_matrix`` (nlev x nlev+1, D_t = k^2
    B x), ``c_matrix`` (nlev+1 x nlev, x_t = C D), ``vertical_structure``
    M = -B C, its modal decomposition, and the per-degree Helmholtz
    inverses.  Built once in float64 numpy (the specification); device
    copies are cast to the backend's float dtype."""

    def __init__(
        self,
        vertical: HybridCoordinate,
        reference_temperature_k: float,
        reference_surface_pressure_pa: float,
        *,
        gas_constant: float = DRY_AIR_GAS_CONSTANT,
    ) -> None:
        self.vertical = vertical
        self.reference_temperature_k = float(reference_temperature_k)
        self.reference_surface_pressure_pa = float(reference_surface_pressure_pa)
        self.gas_constant = float(gas_constant)
        self._build()
        self._device: dict[tuple, tuple] = {}
        self._helmholtz: dict[tuple, object] = {}
        self._half_maps: dict[tuple, tuple] = {}

    # ----------------------------------------------------------------- build
    def _build(self) -> None:
        vertical = self.vertical
        nlev = vertical.nlev
        t_ref = self.reference_temperature_k
        ps_ref = self.reference_surface_pressure_pa
        R = self.gas_constant
        a_half = vertical.a_half_pa
        b_half = vertical.b_half
        p_half = a_half + b_half * ps_ref
        p_full = np.sqrt(p_half[:-1] * p_half[1:])
        dp = np.diff(p_half)
        delta_b = np.diff(b_half)
        exner = (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
        theta_ref = t_ref / exner
        # d ln p_k / d ln ps, exactly as _pressure_gradient_factor.
        factor = 0.5 * (b_half[:-1] / p_half[:-1] + b_half[1:] / p_half[1:]) * ps_ref

        # G: geopotential response to a unit temperature at one level, from
        # the model's own hydrostatic integration (linear in Tv at fixed
        # pressure, so the unit responses ARE the operator).
        geopotential = np.zeros((nlev, nlev))
        columns = p_half[:, None, None]
        for j in range(nlev):
            unit = np.zeros((nlev, 1, 1))
            unit[j] = 1.0
            geopotential[:, j] = vertical.hydrostatic_geopotential(
                unit, np.zeros((1, 1)), columns, gas_constant=R
            )[:, 0, 0]

        # Reference interface values of theta_ref: the model's limited
        # reconstruction from above and from below, averaged (the
        # central-difference linearization of the upwind switch at rest).
        # A documented divergence from the coded C0 switch: each one-sided
        # branch differs from this average by up to 245 K per unit D at
        # level 0 (20-level pressure_blend, 99% of max|C|; 63 K, 34%, on
        # the 40-level default), top two layers only, leaving a
        # piecewise-linear explicit residual with speeds 72.6 / 32.4 m/s
        # (20-level) and 20.7 / 7.7 m/s (40-level); module docstring.
        gradient = np.zeros(nlev)
        above_gradient = (theta_ref[1:-1] - theta_ref[:-2]) / (p_full[1:-1] - p_full[:-2])
        below_gradient = (theta_ref[2:] - theta_ref[1:-1]) / (p_full[2:] - p_full[1:-1])
        product = above_gradient * below_gradient
        monotone = product > 0.0
        gradient[1:-1] = np.where(
            monotone,
            2.0 * product / np.where(monotone, above_gradient + below_gradient, 1.0),
            0.0,
        )
        above = theta_ref[:-1]
        below = theta_ref[1:]
        face_above = above + gradient[:-1] * (p_half[1:nlev] - p_full[:-1])
        face_below = below + gradient[1:] * (p_half[1:nlev] - p_full[1:])
        lower = np.minimum(above, below)
        upper = np.maximum(above, below)
        face_above = np.minimum(np.maximum(face_above, lower), upper)
        face_below = np.minimum(np.maximum(face_below, lower), upper)
        face = np.zeros(nlev + 1)
        face[1:nlev] = 0.5 * (face_above + face_below)

        # Gamma: theta_t = -Gamma D, one unit divergence at a time through
        # continuity, the interface omega, and the flux-form tendency.
        gamma = np.zeros((nlev, nlev))
        for j in range(nlev):
            divergence = np.zeros(nlev)
            divergence[j] = 1.0
            mass_flux_divergence = dp * divergence
            ps_t = -np.sum(mass_flux_divergence)
            dp_t = delta_b * ps_t
            omega = np.zeros(nlev + 1)
            for k in range(nlev):
                omega[k + 1] = omega[k] - dp_t[k] - mass_flux_divergence[k]
            omega[-1] = 0.0
            flux = omega * face
            vertical_flux = flux[1:] - flux[:-1]
            horizontal = dp * theta_ref * divergence
            theta_t = (-horizontal - vertical_flux - theta_ref * dp_t) / dp
            gamma[:, j] = -theta_t

        weights = dp / ps_ref
        b_matrix = np.concatenate(
            [
                geopotential * exner[None, :],
                (KAPPA * t_ref * (geopotential @ factor) + R * t_ref)[:, None],
            ],
            axis=1,
        )
        c_matrix = np.concatenate([-gamma, -weights[None, :]], axis=0)
        structure = -b_matrix @ c_matrix
        eigenvalues, eigenvectors = np.linalg.eig(structure)
        if np.max(np.abs(eigenvalues.imag)) > 1.0e-9 * np.max(np.abs(eigenvalues.real)):
            raise ValueError(
                "vertical-structure matrix has complex eigenvalues: the "
                "reference state does not define real gravity-wave modes"
            )
        if np.min(eigenvalues.real) <= 0.0:
            raise ValueError(
                "vertical-structure matrix has a nonpositive eigenvalue: a "
                "mode of the reference state would grow instead of oscillate"
            )
        order = np.argsort(-eigenvalues.real)
        # The (theta, ln ps) block C B shares the nonzero spectrum of B C
        # (eigenvectors C v_j) and adds the one-dimensional null space of
        # B - the combination with no pressure-gradient force - at zero.
        # Its decomposition carries the symmetric half map on that block.
        x_values, x_vectors = np.linalg.eig(c_matrix @ b_matrix)
        if np.max(np.abs(x_values.imag)) > 1.0e-9 * np.max(np.abs(x_values.real)):
            raise ValueError(
                "thermodynamic block of the vertical-structure operator has "
                "complex eigenvalues: the reference state does not define "
                "real gravity-wave modes"
            )
        x_values = x_values.real
        x_values[np.abs(x_values) <= 1.0e-9 * np.max(np.abs(x_values))] = 0.0
        if np.max(x_values) > 0.0:
            raise ValueError(
                "thermodynamic block of the vertical-structure operator has a "
                "positive eigenvalue: a mode of the reference state would grow "
                "instead of oscillate"
            )
        self.x_eigenvalues = x_values
        self.x_eigenvectors = x_vectors.real
        self.p_full = p_full
        self.p_half = p_half
        self.theta_ref = theta_ref
        self.exner = exner
        self.pressure_gradient_factor = factor
        self.geopotential_matrix = geopotential
        self.gamma = gamma
        self.mass_weights = weights
        self.b_matrix = b_matrix
        self.c_matrix = c_matrix
        self.vertical_structure = structure
        self.eigenvalues = eigenvalues.real[order]
        self.eigenvectors = eigenvectors.real[:, order]
        self.phase_speeds_m_s = np.sqrt(self.eigenvalues)
        self.equivalent_depths_m = self.eigenvalues / GRAVITY_M_S2

    # ------------------------------------------------------------ describe
    @property
    def nlev(self) -> int:
        return int(self.vertical.nlev)

    @property
    def fastest_mode_m_s(self) -> float:
        return float(self.phase_speeds_m_s[0])

    def describe(self) -> dict[str, object]:
        return {
            "reference_temperature_k": self.reference_temperature_k,
            "reference_surface_pressure_pa": self.reference_surface_pressure_pa,
            "nlev": self.nlev,
            "phase_speeds_m_s": [float(v) for v in self.phase_speeds_m_s],
            "equivalent_depths_m": [float(v) for v in self.equivalent_depths_m],
        }

    # ------------------------------------------------------------- devices
    def device_matrices(self, backend):
        key = (*device_cache_key(backend.xp), str(backend.float_dtype))
        cached = self._device.get(key)
        if cached is None:
            cached = (
                backend.asarray(self.b_matrix, dtype=backend.float_dtype),
                backend.asarray(self.c_matrix, dtype=backend.float_dtype),
                backend.asarray(self.vertical_structure, dtype=backend.float_dtype),
            )
            self._device[key] = cached
        return cached

    def helmholtz_inverse(self, backend, k2: np.ndarray, dt_s: float, alpha: float):
        """``(I + alpha^2 dt^2 k_n^2 M)^{-1}`` per total degree, shape
        (n_degrees, nlev, nlev), through the modal decomposition
        V diag(1/(1 + s c_j^2)) V^{-1} in float64 and cast to the backend."""
        k2 = np.asarray(k2, dtype=np.float64)
        key = (
            *device_cache_key(backend.xp), str(backend.float_dtype),
            float(dt_s), float(alpha), k2.tobytes(),
        )
        cached = self._helmholtz.get(key)
        if cached is None:
            scale = (float(alpha) * float(dt_s)) ** 2 * k2
            modal = 1.0 / (1.0 + scale[:, None] * self.eigenvalues[None, :])
            inverse = np.einsum(
                "ij,nj,jk->nik",
                self.eigenvectors,
                modal,
                np.linalg.inv(self.eigenvectors),
            )
            # Degree zero has k = 0: the inverse is the identity exactly,
            # so the global means pass through untouched.
            inverse[k2 == 0.0] = np.eye(self.nlev)
            cached = backend.asarray(inverse, dtype=backend.float_dtype)
            self._helmholtz[key] = cached
        return cached

    @staticmethod
    def _half_map_scalars(frequencies: np.ndarray, dt_s: float, alpha: float):
        """``(p, q)`` with ``sqrt(g) = p + i q omega`` for the off-centred
        Crank-Nicolson factor ``g = (1 + i (1-a) omega dt) / (1 - i a omega
        dt)`` of every mode frequency (principal square root); ``omega = 0``
        (degree zero and the null mode) gives ``p = 1, q = dt / 2``, the
        limit of the formula."""
        y = frequencies * float(dt_s)
        g = (1.0 + 1j * (1.0 - alpha) * y) / (1.0 - 1j * alpha * y)
        root = np.sqrt(g)
        p = root.real
        q = np.where(
            frequencies > 0.0,
            root.imag / np.where(frequencies > 0.0, frequencies, 1.0),
            0.5 * float(dt_s),
        )
        return p, q

    def symmetric_half_maps(self, backend, k2: np.ndarray, dt_s: float, alpha: float):
        """The square root ``H = P + Q L`` of the off-centred Crank-Nicolson
        map, one (P, Q) pair per block and per total degree.

        ``L^2`` is block diagonal (``k^2 B C`` on divergence, ``k^2 C B`` on
        the thermodynamic pair) with the modes' ``-omega_j^2`` on the
        diagonal in the modal bases, so any real function of ``L`` is ``P +
        Q L`` with ``P, Q`` functions of ``L^2``: per mode ``p_j + i q_j
        omega_j = sqrt(g_j)``.  Returns ``(P_D, Q_D, P_x, Q_x)`` of shapes
        (n_degrees, nlev, nlev) twice and (n_degrees, nlev+1, nlev+1)
        twice, float64-built and cast to the backend."""
        k2 = np.asarray(k2, dtype=np.float64)
        key = (
            *device_cache_key(backend.xp), str(backend.float_dtype),
            float(dt_s), float(alpha), k2.tobytes(),
        )
        cached = self._half_maps.get(key)
        if cached is None:
            k = np.sqrt(k2)
            speeds_d = np.sqrt(self.eigenvalues)
            speeds_x = np.sqrt(-self.x_eigenvalues)
            p_d, q_d = self._half_map_scalars(k[:, None] * speeds_d[None, :], dt_s, alpha)
            p_x, q_x = self._half_map_scalars(k[:, None] * speeds_x[None, :], dt_s, alpha)
            v = self.eigenvectors
            v_inverse = np.linalg.inv(v)
            w = self.x_eigenvectors
            w_inverse = np.linalg.inv(w)
            maps = (
                np.einsum("ij,nj,jk->nik", v, p_d, v_inverse),
                np.einsum("ij,nj,jk->nik", v, q_d, v_inverse),
                np.einsum("ij,nj,jk->nik", w, p_x, w_inverse),
                np.einsum("ij,nj,jk->nik", w, q_x, w_inverse),
            )
            cached = tuple(backend.asarray(m, dtype=backend.float_dtype) for m in maps)
            self._half_maps[key] = cached
        return cached


#: One operator per (A, B, T_ref, ps_ref).  NO device id: the operator
#: itself holds only NumPy arrays, and the three caches of DEVICE arrays it
#: owns (:attr:`~VerticalStructureOperator._device`, ``_helmholtz``,
#: ``_half_maps``) each carry
#: :func:`~woof.globe.spectral.backend.device_cache_key` in their own
#: keys, so one shared operator serves two cards and hands each its own
#: matrices.  Keying this dictionary as well would rebuild the NumPy
#: eigen-decomposition per card for nothing; keying it INSTEAD of the three
#: would be the multi-card defect.  Gate CARD-1 covers all three.
_OPERATORS: dict[tuple, VerticalStructureOperator] = {}


def vertical_structure_operator(
    vertical: HybridCoordinate,
    reference_temperature_k: float,
    reference_surface_pressure_pa: float,
) -> VerticalStructureOperator:
    """The operator of a stack, built once per (A, B, T_ref, ps_ref) and
    per device."""
    key = (
        vertical.a_half_pa.tobytes(),
        vertical.b_half.tobytes(),
        float(reference_temperature_k),
        float(reference_surface_pressure_pa),
    )
    operator = _OPERATORS.get(key)
    if operator is None:
        operator = VerticalStructureOperator(
            vertical, reference_temperature_k, reference_surface_pressure_pa
        )
        _OPERATORS[key] = operator
    return operator


@dataclass(frozen=True)
class VerticalModeSemiImplicit:
    """Every vertical gravity-wave mode implicit (module docstring).

    ``off_centring_weight`` is the implicit weight alpha on the new time
    level: 0.5 is neutral Crank-Nicolson, 1.0 backward Euler; a value
    below 0.5 amplifies every implicit mode (|(1 + (1-a) z) / (1 - a z)|
    > 1 on the imaginary axis) and is refused.  ``divergence_weight`` is
    the fraction of the operator treated implicitly; below 1 the rest
    rides the explicit budget and counts against the CFL gate.

    alpha ships at 0.5; the measured trade of 0.55 is recorded here and
    the default is the owner's decision.  0.55 raises the linearized rest
    ceilings (minimum over the 286 / 300 / 320 / 340 K ladder columns at
    T_ref = 320 K) and removes the Doppler growth past the CFL gate
    (dynamics.cfl): 20-level pressure_blend at T533 from 449 to 1126 s
    without diffusion and 445 to 1834 s with the shipped hyperdiffusion
    (T21, the 286 K column: 16770 to 69360 s); 40-level surface_stretched
    default at T533 from 400 to 514 s and 395 to 550 s, at T255 from 835
    to 1074 s and 868 to 1283 s (on the default stack the warmest column
    binds the 0.55 ceiling, so the gain there is 1.3-1.5x, not the 20-level
    stack's 2.5-4x).  It damps the slow manifold: rotational kinetic energy
    at T42 20-level, dt = 600 s, retains 0.8319 of its initial value after
    24 h against 0.8462 at 0.5 (-1.7% per 24 h); on the T21 20-level moist
    24 h case the kinetic-energy ratio falls from 0.9475 to 0.9234 (2.4%
    more loss) and the relative total-water and axial-angular-momentum
    drifts grow from -3.2e-6 / -3.3e-6 to -2.2e-4 / -2.2e-4 (70x).
    """

    enabled: bool = True
    reference_temperature_k: float = 320.0
    reference_surface_pressure_pa: float = 1.0e5
    off_centring_weight: float = 0.5
    divergence_weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.reference_temperature_k) or not (
            150.0 <= self.reference_temperature_k <= 400.0
        ):
            raise ValueError(
                "reference_temperature_k must lie in 150..400 K: the "
                "isothermal reference sets every implicit mode speed "
                "(sqrt(R T / (1 - kappa)) for the external mode)"
            )
        if not math.isfinite(self.reference_surface_pressure_pa) or not (
            30_000.0 <= self.reference_surface_pressure_pa <= 120_000.0
        ):
            raise ValueError(
                "reference_surface_pressure_pa must lie in the 30000..120000 "
                "Pa research bounds the hybrid stack is evaluated at"
            )
        if not math.isfinite(self.off_centring_weight) or not (
            0.5 <= self.off_centring_weight <= 1.0
        ):
            raise ValueError(
                "off_centring_weight must lie in [0.5, 1]: below 0.5 the "
                "implicit step amplifies every gravity-wave mode it treats "
                "(|(1 + (1-a) z)/(1 - a z)| > 1 on the imaginary axis)"
            )
        if not math.isfinite(self.divergence_weight) or not 0.0 <= self.divergence_weight <= 1.0:
            raise ValueError("divergence_weight must lie in [0,1]")

    @property
    def active(self) -> bool:
        return self.enabled and self.divergence_weight > 0.0

    @property
    def scheme(self) -> str:
        return "vertical_modes"

    def operator(self, vertical) -> VerticalStructureOperator:
        return vertical_structure_operator(
            vertical, self.reference_temperature_k, self.reference_surface_pressure_pa
        )

    def explicit_wave_speed_m_s(self, vertical) -> float:
        """The fastest mode speed times the fraction left explicit."""
        fastest = self.operator(vertical).fastest_mode_m_s
        if self.active:
            return (1.0 - self.divergence_weight) * fastest
        return fastest

    def describe(self, vertical) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "enabled": bool(self.enabled),
            "off_centring_weight": float(self.off_centring_weight),
            "weight": float(self.divergence_weight),
            **self.operator(vertical).describe(),
        }

    def _wave_operator(self, transform) -> np.ndarray:
        n = np.arange(transform.truncation + 1, dtype=np.float64)
        return n * (n + 1.0) / (transform.grid.radius_m ** 2)

    def linear_tendencies(self, state: MoistHybridState, transform, vertical):
        """``w L y``: divergence, theta and ln ps tendencies of the linear
        operator at the state, all three subtracted by rhs()."""
        if not self.active:
            return None
        backend = transform.backend
        xp = backend.xp
        b_matrix, c_matrix, _structure = self.operator(vertical).device_matrices(backend)
        k2 = backend.asarray(self._wave_operator(transform), dtype=backend.float_dtype)
        w = backend.float_dtype(self.divergence_weight)
        from woof.metal.array import MetalArray
        metal_context = getattr(transform, "metal_context", None)
        if (isinstance(state.theta, MetalArray) and metal_context is not None
                and xp is np and np.dtype(backend.float_dtype) == np.dtype(np.float64)):
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs

            nlevel, ndegree, mcount = map(int, state.theta.shape)
            if (tuple(state.divergence.shape) != (nlevel, ndegree, mcount)
                    or tuple(state.log_surface_pressure.shape) != (ndegree, mcount)):
                raise ValueError("resident semi-implicit state shapes do not agree")
            nmodes = ndegree * mcount
            metal = MetalArrayModule(metal_context)

            def complex_resource(value):
                if (not isinstance(value, MetalArray)
                        or value.buffer.context is not metal_context
                        or value.dtype != np.dtype(np.complex128)):
                    raise TypeError(
                        "resident semi-implicit inputs must be complex128 on their Metal context"
                    )
                return value.view((2 * value.size,), dtype=np.float64)

            def real_resource(value):
                return metal.asarray(np.ascontiguousarray(
                    value, dtype=np.float64
                ).reshape(-1))

            divergence_t = metal.empty(
                (nlevel, ndegree, mcount), dtype=np.complex128
            )
            x_t = metal.empty(
                (nlevel + 1, ndegree, mcount), dtype=np.complex128
            )
            resources = {
                "divergence": complex_resource(state.divergence),
                "theta": complex_resource(state.theta),
                "logps": complex_resource(state.log_surface_pressure),
                "b_matrix": real_resource(b_matrix),
                "c_matrix": real_resource(c_matrix),
                "k2": real_resource(k2),
                "weight": real_resource(np.asarray([w], dtype=np.float64)),
                "divergence_tendency": divergence_t.view(
                    (2 * divergence_t.size,), dtype=np.float64
                ),
                "state_tendency": x_t.view(
                    (2 * x_t.size,), dtype=np.float64
                ),
            }
            metal_context.submit_kernel(
                "global_semilag_linear_tendencies_fragment", resources,
                {"nlevel": nlevel, "ndegree": ndegree, "mcount": mcount},
                exact_policy=False, grid_size=nmodes,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_LINEAR_TENDENCIES_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            plane_bytes = ndegree * mcount * np.dtype(np.complex128).itemsize
            theta_t = x_t.view(state.theta.shape, dtype=np.complex128)
            logps_t = x_t.view(
                state.log_surface_pressure.shape, dtype=np.complex128,
                byte_offset=nlevel * plane_bytes,
            )
            return LinearTendencies(
                transform.project(divergence_t),
                transform.project(theta_t),
                transform.project(logps_t),
            )
        x = xp.concatenate([state.theta, state.log_surface_pressure[None]], axis=0)
        divergence_t = w * k2[None, :, None] * xp.einsum("ij,jnm->inm", b_matrix, x)
        x_t = w * xp.einsum("ij,jnm->inm", c_matrix, state.divergence)
        return LinearTendencies(
            transform.project(divergence_t),
            transform.project(x_t[:-1]),
            transform.project(x_t[-1]),
        )

    def _finish(self, state, transform, d_new, x_new, divergence, x, *, metric=True,
                metric_value=None):
        """Pin degree zero, project, and rebuild the state with the
        divergence-increment metric (``metric=False`` skips the metric
        for a caller that discards it: the IMEX stage solve, whose
        increment the integrator books itself)."""
        backend = transform.backend
        from woof.metal.array import MetalArray
        if isinstance(d_new, MetalArray):
            if not isinstance(x_new, MetalArray):
                raise TypeError("resident semi-implicit output pair is incomplete")
            if metric and metric_value is None:
                raise ValueError("resident semi-implicit metric must be reduced on Metal")
            nlevel = int(state.theta.shape[0])
            plane_bytes = int(np.prod(state.theta.shape[1:])) * np.dtype(np.complex128).itemsize
            theta_new = x_new[:nlevel]
            logps_new = x_new.view(
                state.log_surface_pressure.shape, dtype=np.complex128,
                byte_offset=nlevel * plane_bytes,
            )
            maximum = float(metric_value) if metric else 0.0
            return state.with_fields((
                state.vorticity, d_new, theta_new, logps_new,
                state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
            )), {"semi_implicit_max_divergence_increment_s1": maximum}
        # Degree zero carries the global means: k = 0 leaves the divergence
        # untouched and the D(0,0) coefficient is zero for any wind, so the
        # means pass through; pinned explicitly so that cannot drift.
        d_new[:, 0, 0] = divergence[:, 0, 0]
        x_new[:, 0, 0] = x[:, 0, 0]
        maximum = 0.0
        if metric:
            correction = d_new - divergence
            maximum = float(np.max(np.abs(backend.to_numpy(correction))))
        return state.with_fields(
            (
                state.vorticity,
                transform.project(d_new),
                transform.project(x_new[:-1]),
                transform.project(x_new[-1]),
                state.qv,
                state.qc,
                state.qr,
                state.qi,
                state.qs,
                state.qg,
            )
        ), {"semi_implicit_max_divergence_increment_s1": maximum}

    def half_apply(self, state: MoistHybridState, transform, vertical, dt_s: float):
        """The square root ``H`` of the off-centred Crank-Nicolson map of
        the subtracted operator over ``dt``: ``H y = P y + Q L y`` per
        block (VerticalStructureOperator.symmetric_half_maps).  Applied
        before AND after the explicit step it composes to exactly the
        Crank-Nicolson step of :meth:`apply` (``H^2 = CN``, checked to
        2e-15) with the split made symmetric, which is what keeps a
        balanced state balanced (module docstring)."""
        if not self.active:
            return state, {"semi_implicit_max_divergence_increment_s1": 0.0}
        backend = transform.backend
        xp = backend.xp
        operator = self.operator(vertical)
        b_matrix, c_matrix, _structure = operator.device_matrices(backend)
        k2_host = self._wave_operator(transform)
        k2 = backend.asarray(k2_host, dtype=backend.float_dtype)
        p_d, q_d, p_x, q_x = operator.symmetric_half_maps(
            backend, k2_host, float(dt_s) * self.divergence_weight, self.off_centring_weight
        )
        divergence = state.divergence
        metal_context = getattr(transform, "metal_context", None)
        from woof.metal.array import MetalArray
        if (metal_context is not None and xp is np
                and np.dtype(backend.float_dtype) == np.dtype(np.float64)):
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs

            nlevel, ndegree, mcount = map(int, divergence.shape)
            if (tuple(state.theta.shape) != (nlevel, ndegree, mcount)
                    or tuple(state.log_surface_pressure.shape) != (ndegree, mcount)):
                raise ValueError("resident semi-implicit state shapes do not agree")
            nmodes = ndegree * mcount
            metal = MetalArrayModule(metal_context)

            def complex_resource(value):
                if isinstance(value, MetalArray):
                    if (value.buffer.context is not metal_context
                            or value.dtype != np.dtype(np.complex128)):
                        raise TypeError(
                            "resident semi-implicit inputs must be complex128 on their Metal context"
                        )
                    resident = value
                else:
                    resident = metal.asarray(np.ascontiguousarray(
                        value, dtype=np.complex128
                    ))
                return resident.view((2 * resident.size,), dtype=np.float64)

            x_new = metal.empty(
                ((nlevel + 1) * nmodes * 2,), dtype=np.float64
            )
            divergence_new = metal.empty(
                (nlevel * nmodes * 2,), dtype=np.float64
            )
            partial_max = metal.empty((nmodes,), dtype=np.float64)
            resource_cache = _metal_operator_resource_cache(
                transform, metal_context
            )["half_apply"]
            resource_key = (
                operator, backend.name, np.dtype(backend.float_dtype).str,
                float(dt_s), float(self.divergence_weight),
                float(self.off_centring_weight), k2_host.tobytes(),
            )
            operator_resources = resource_cache.get(resource_key)
            if operator_resources is None:
                def real_resource(value):
                    return metal.asarray(np.ascontiguousarray(
                        value, dtype=np.float64
                    ).reshape(-1))

                operator_resources = {
                    "b_matrix": real_resource(b_matrix),
                    "c_matrix": real_resource(c_matrix),
                    "p_d": real_resource(p_d),
                    "q_d": real_resource(q_d),
                    "p_x": real_resource(p_x),
                    "q_x": real_resource(q_x),
                    "k2": real_resource(k2),
                }
                resource_cache[resource_key] = operator_resources
            resources = {
                "divergence": complex_resource(divergence),
                "theta": complex_resource(state.theta),
                "logps": complex_resource(state.log_surface_pressure),
                **operator_resources,
                "divergence_new": divergence_new,
                "x_new": x_new,
                "partial_max": partial_max,
            }
            metal_context.submit_kernel(
                "global_semilag_half_apply_fragment", resources,
                {"nlevel": nlevel, "ndegree": ndegree, "mcount": mcount},
                exact_policy=False, grid_size=nmodes,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_HALF_APPLY_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            maximum = metal.empty((1,), dtype=np.float64)
            metal_context.submit_kernel(
                "global_semilag_max_finalize_fragment",
                {"partial_max": partial_max, "maximum": maximum},
                {"partial_count": nmodes}, exact_policy=False,
                grid_size=1,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_MAX_FINALIZE_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            d_new = divergence_new.view(
                (nlevel, ndegree, mcount), dtype=np.complex128
            )
            x_new_complex = x_new.view(
                (nlevel + 1, ndegree, mcount), dtype=np.complex128
            )
            return self._finish(
                state, transform, d_new, x_new_complex, divergence, None,
                metric_value=float(maximum.to_numpy()[0]),
            )
        x = xp.concatenate([state.theta, state.log_surface_pressure[None]], axis=0)
        # L y: divergence rows k^2 B x, thermodynamic rows C D.
        l_divergence = k2[None, :, None] * xp.einsum("ij,jnm->inm", b_matrix, x)
        l_x = xp.einsum("ij,jnm->inm", c_matrix, divergence)
        d_new = (
            xp.einsum("nij,jnm->inm", p_d, divergence)
            + xp.einsum("nij,jnm->inm", q_d, l_divergence)
        )
        x_new = (
            xp.einsum("nij,jnm->inm", p_x, x)
            + xp.einsum("nij,jnm->inm", q_x, l_x)
        )
        return self._finish(state, transform, d_new, x_new, divergence, x)

    def pre_apply(self, state: MoistHybridState, transform, vertical, dt_s: float):
        """The half map before the explicit step (symmetric split)."""
        return self.half_apply(state, transform, vertical, dt_s)

    def post_apply(self, state: MoistHybridState, transform, vertical, dt_s: float):
        """The half map after the explicit step (symmetric split)."""
        return self.half_apply(state, transform, vertical, dt_s)

    def apply(self, state: MoistHybridState, transform, vertical, dt_s: float):
        """The full off-centred Crank-Nicolson step of the subtracted
        operator over ``dt`` through the per-degree Helmholtz solve
        (module docstring); ``half_apply`` twice is this map, and this is
        the form the A/B against the external proxy compares."""
        if not self.active:
            return state, {"semi_implicit_max_divergence_increment_s1": 0.0}
        backend = transform.backend
        xp = backend.xp
        operator = self.operator(vertical)
        b_matrix, c_matrix, structure = operator.device_matrices(backend)
        k2_host = self._wave_operator(transform)
        k2 = backend.asarray(k2_host, dtype=backend.float_dtype)
        dt = backend.float_dtype(float(dt_s) * self.divergence_weight)
        alpha = backend.float_dtype(self.off_centring_weight)
        one_minus_alpha = backend.float_dtype(1.0 - self.off_centring_weight)
        inverse = operator.helmholtz_inverse(
            backend, k2_host, float(dt_s) * self.divergence_weight, self.off_centring_weight
        )
        divergence = state.divergence
        x = xp.concatenate([state.theta, state.log_surface_pressure[None]], axis=0)
        k2_col = k2[None, :, None]
        forcing = (
            divergence
            + dt * k2_col * xp.einsum("ij,jnm->inm", b_matrix, x)
            - (alpha * one_minus_alpha * dt * dt) * k2_col
            * xp.einsum("ij,jnm->inm", structure, divergence)
        )
        d_new = xp.einsum("nij,jnm->inm", inverse, forcing)
        x_new = x + dt * xp.einsum(
            "ij,jnm->inm", c_matrix, one_minus_alpha * divergence + alpha * d_new
        )
        return self._finish(state, transform, d_new, x_new, divergence, x)

    def solve_shifted(self, state: MoistHybridState, transform, vertical, tau_s: float,
                      *, keep_resident: bool = False):
        """``(I - tau w L) y = r``: the implicit stage solve of the IMEX
        integrator (imex.py).  Eliminating the thermodynamic rows,

          (I + tau^2 w^2 k^2 M) D = r_D + tau w k^2 B r_x
          x = r_x + tau w C D

        one Helmholtz inverse per total degree (``helmholtz_inverse`` at
        ``alpha = 1``, cached per ``tau``).  Degree zero passes through
        untouched (``_finish``).  Inactive: identity."""
        if not self.active:
            return state
        backend = transform.backend
        xp = backend.xp
        operator = self.operator(vertical)
        b_matrix, c_matrix, _structure = operator.device_matrices(backend)
        k2_host = self._wave_operator(transform)
        k2 = backend.asarray(k2_host, dtype=backend.float_dtype)
        tw_host = float(tau_s) * self.divergence_weight
        tw = backend.float_dtype(tw_host)
        inverse = operator.helmholtz_inverse(backend, k2_host, tw_host, 1.0)
        divergence = state.divergence
        metal_context = getattr(transform, "metal_context", None)
        from woof.metal.array import MetalArray
        resident_state = isinstance(state.theta, MetalArray)
        if resident_state:
            x = None
        else:
            x = xp.concatenate([state.theta, state.log_surface_pressure[None]], axis=0)
        if (metal_context is not None and xp is np
                and np.dtype(backend.float_dtype) == np.dtype(np.float64)
                and state.divergence.ndim == 3 and state.divergence.shape[1] <= 128):
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs

            nlevel, ndegree, mcount = map(int, divergence.shape)
            nmodes = ndegree * mcount
            metal = MetalArrayModule(metal_context)

            def complex_resource(value):
                if isinstance(value, MetalArray):
                    if value.buffer.context is not metal_context:
                        raise ValueError(
                            "Global semi-implicit input belongs to another Metal context"
                        )
                    if value.dtype != np.dtype(np.complex128):
                        raise TypeError(
                            "Global semi-implicit resident inputs must be complex128"
                        )
                    resident = metal.asarray(value)
                else:
                    resident = metal.asarray(np.ascontiguousarray(
                        value, dtype=np.complex128))
                return resident.view(
                    (2 * resident.size,), dtype=np.float64
                )

            resource_cache = _metal_operator_resource_cache(
                transform, metal_context
            )["solve"]
            resource_key = (
                operator, backend.name, np.dtype(backend.float_dtype).str,
                float(tw_host), k2_host.tobytes(),
            )
            operator_resources = resource_cache.get(resource_key)
            if operator_resources is None:
                inverse_host = np.ascontiguousarray(inverse, dtype=np.float64)
                b_host = np.ascontiguousarray(b_matrix, dtype=np.float64)
                c_host = np.ascontiguousarray(c_matrix, dtype=np.float64)
                k2_flat = np.ascontiguousarray(k2, dtype=np.float64)
                operator_resources = {
                    "inverse": metal.asarray(inverse_host.reshape(-1)),
                    "b_matrix": metal.asarray(b_host.reshape(-1)),
                    "c_matrix": metal.asarray(c_host.reshape(-1)),
                    "k2": metal.asarray(k2_flat.reshape(-1)),
                    "tau_weight": metal.asarray(np.asarray(
                        [tw_host], dtype=np.float64
                    )),
                }
                resource_cache[resource_key] = operator_resources

            resources = {
                "divergence": complex_resource(divergence),
                "theta": complex_resource(state.theta),
                "logps": complex_resource(state.log_surface_pressure),
                **operator_resources,
                "divergence_new": metal.empty((2 * divergence.size,), dtype=np.float64),
                "x_new": metal.empty((
                    2 * (nlevel + 1) * ndegree * mcount,
                ), dtype=np.float64),
            }
            metal_context.submit_kernel(
                "global_semilag_solve_shifted_fragment", resources,
                {"nlevel": nlevel, "ndegree": ndegree, "mcount": mcount},
                exact_policy=False, grid_size=2 * nmodes,
                threads_per_group=kernel_specs.GLOBAL_SEMILAG_SOLVE_SHIFTED_FRAGMENT_THREADS_PER_GROUP,
            )
            if keep_resident:
                from .state import to_metal_atmosphere

                ndegree, mcount = map(int, divergence.shape[-2:])
                d_new = resources["divergence_new"].view(
                    divergence.shape, dtype=np.complex128
                )
                x_resident = resources["x_new"].view(
                    (nlevel + 1, ndegree, mcount), dtype=np.complex128
                )
                plane_bytes = ndegree * mcount * np.dtype(np.complex128).itemsize
                theta_new = x_resident.view(
                    state.theta.shape, dtype=np.complex128
                )
                logps_new = x_resident.view(
                    state.log_surface_pressure.shape, dtype=np.complex128,
                    byte_offset=nlevel * plane_bytes,
                )
                solved = state.with_fields((
                    state.vorticity, d_new, theta_new, logps_new, state.qv,
                    *[getattr(state, name) for name in (
                        "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng"
                    )],
                ))
                solved = to_metal_atmosphere(solved, metal_context)
                # The tracer pressure synthesis explicitly handles the
                # resident log(ps) input at its transform boundary; keep the
                # solved state itself resident through that consumer.
                return solved

            d_parts = resources["divergence_new"].to_numpy().reshape(-1, 2)
            x_parts = resources["x_new"].to_numpy().reshape(-1, 2)
            d_new = (d_parts[:, 0] + 1j * d_parts[:, 1]).reshape(divergence.shape)
            x_shape = (nlevel + 1, ndegree, mcount)
            x_new = (x_parts[:, 0] + 1j * x_parts[:, 1]).reshape(x_shape)
            solved, _metrics = self._finish(
                state, transform, d_new, x_new, divergence, x, metric=False
            )
            return solved
        if x is None:
            theta_host = (state.theta.to_numpy()
                          if isinstance(state.theta, MetalArray) else state.theta)
            logps_host = (state.log_surface_pressure.to_numpy()
                          if isinstance(state.log_surface_pressure, MetalArray)
                          else state.log_surface_pressure)
            x = xp.concatenate([
                theta_host, logps_host[None]
            ], axis=0)
        forcing = divergence + tw * k2[None, :, None] * xp.einsum("ij,jnm->inm", b_matrix, x)
        d_new = xp.einsum("nij,jnm->inm", inverse, forcing)
        x_new = x + tw * xp.einsum("ij,jnm->inm", c_matrix, d_new)
        # No metric: the increment of the stage solve was copied to the
        # host (a whole divergence spectrum, 21 MB at T255, twice per
        # step) and discarded (profile 2026-09-04).
        solved, _metrics = self._finish(
            state, transform, d_new, x_new, divergence, x, metric=False
        )
        return solved


SEMI_IMPLICIT_SCHEMES = ("vertical_modes", "external")


def build_semi_implicit(
    scheme: str,
    *,
    enabled: bool,
    weight: float,
    external_wave_speed_m_s: float,
    reference_temperature_k: float,
    reference_surface_pressure_pa: float,
    off_centring_weight: float,
):
    """The scheme a config selects, by name."""
    if scheme == "vertical_modes":
        return VerticalModeSemiImplicit(
            enabled=enabled,
            reference_temperature_k=reference_temperature_k,
            reference_surface_pressure_pa=reference_surface_pressure_pa,
            off_centring_weight=off_centring_weight,
            divergence_weight=weight,
        )
    if scheme == "external":
        return BarotropicSemiImplicit(
            enabled=enabled,
            external_wave_speed_m_s=external_wave_speed_m_s,
            divergence_weight=weight,
        )
    raise ValueError(
        f"unknown semi-implicit scheme {scheme!r}; known: {', '.join(SEMI_IMPLICIT_SCHEMES)}"
    )


__all__ = [
    "BarotropicSemiImplicit",
    "LinearTendencies",
    "SEMI_IMPLICIT_SCHEMES",
    "VerticalModeSemiImplicit",
    "VerticalStructureOperator",
    "build_semi_implicit",
    "vertical_structure_operator",
]
