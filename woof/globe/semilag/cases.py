"""Kernel gate cases: prescribed flows with an exact answer.

These drive :mod:`woof.globe.semilag.trajectory` and
:mod:`woof.globe.semilag.interpolate` and nothing else.  No model, no
physics, no spectral transform: the flow is analytic, the exact solution at
the end of the integration is the field the integration started from, and the
error norms are therefore self-contained rather than read off somebody's
table.

KERN-1, solid-body rotation.  The whole point of the pi/2 rotation angle is
that the flow runs the tracer straight over both poles, so a meridional
weight set built as if the Gauss-Legendre nodes were equispaced, or a polar
reflection that lands in the wrong hemisphere, cannot hide.  The solid-body
trajectory also has a closed form, which turns the departure-point solver
into something with an exact answer to be graded against rather than a
converged fixed point to be trusted.

KERN-2, the deformational flows of Nair and Lauritzen 2010 in the form
standardised by Lauritzen and others in 2012.  Solid-body rotation moves a
field without deforming it and so tests almost nothing about a stencil; these
flows shear the field into filaments finer than the grid by half the period
and then return it exactly to where it started, which is what separates a
scheme that interpolates from a scheme that also dissipates.

The wind fields are the published ones, coefficient for coefficient, because
a gate that calls itself a published case and is not one cannot be compared
with anything: the non-divergent field carries 10 R/T, and the DIVERGENT
field carries 5 R/T on its zonal component and 5/2 R/T on its meridional one
together with the same 2 pi R / T constant background rotation the
non-divergent field carries, at half the deformation amplitude of the
non-divergent case.  It was coded here at the non-divergent amplitude with
no background, which is twice the deformation of the published case, and
that is what the 0.88 convergence slope was measuring.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import math
from typing import Any

import numpy as np

from woof.globe.spectral.grid import GaussianGrid

from .interpolate import Stencil, gather_batch
from .tables import SphericalGridTables
from .tracers import DEFICIT_FIXERS, fix_mass
from .trajectory import CartesianWind, cartesian_wind, departure_points

#: The period every case of record runs on, seconds.  Twelve days is the
#: revolution time of the solid-body flow and the full deformation-and-return
#: period of the deformational flows.
CASE_PERIOD_S = 12.0 * 86400.0

DEFORMATIONAL_CASES = (
    "nondivergent_one_cell",
    "nondivergent_two_cell",
    "divergent",
    "nondivergent_two_cell_background",
)

INITIAL_DISTRIBUTIONS = ("cosine_bells", "gaussian_hills", "slotted_cylinders")


def _mesh(grid: GaussianGrid):
    lam = np.asarray(grid.lon_rad, dtype=np.float64)[None, :]
    phi = np.asarray(grid.lat_rad, dtype=np.float64)[:, None]
    return np.broadcast_to(lam, grid.shape), np.broadcast_to(phi, grid.shape)


def area_weights(grid: GaussianGrid) -> np.ndarray:
    """Area of every cell, in units of the sphere's own area.

    Gauss-Legendre quadrature weights sum to 2 over sin(lat) and the
    longitudes are uniform, so ``w_j / (2 * nlon)`` is the fraction of the
    sphere one cell covers and the whole table sums to exactly 1.
    """
    w = np.asarray(grid.quadrature_weights, dtype=np.float64)
    cell = w[:, None] / (2.0 * grid.nlon)
    return np.broadcast_to(cell, grid.shape).copy()


# ---------------------------------------------------------------------------
# KERN-1: solid-body rotation


def solid_body_axis(alpha: float) -> np.ndarray:
    """The rotation axis of the solid-body flow at tilt ``alpha``."""
    return np.array([-math.sin(alpha), 0.0, math.cos(alpha)], dtype=np.float64)


def solid_body_wind(lam, phi, *, u0: float, alpha: float):
    """The classical solid-body wind, tilted ``alpha`` from the polar axis."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    u = u0 * (np.cos(phi) * ca + np.sin(phi) * np.cos(lam) * sa)
    v = -u0 * np.sin(lam) * sa
    return u, v


def solid_body_departure(lam, phi, *, u0: float, alpha: float, dt: float,
                         radius: float):
    """The exact departure point of the solid-body flow, in closed form.

    A rigid rotation carries the arrival point back along a great circle
    about the flow's own axis, so Rodrigues' formula gives the departure
    point with no iteration and no truncation error at all.  This is the
    reference the fixed-point solver is graded against.
    """
    axis = solid_body_axis(alpha)
    theta = -(u0 / radius) * dt
    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi) + 0.0 * x
    r = np.stack([x, y, z], axis=0)
    k = axis.reshape(3, *([1] * (r.ndim - 1)))
    kdotr = k[0] * r[0] + k[1] * r[1] + k[2] * r[2]
    kcross = np.stack([
        k[1] * r[2] - k[2] * r[1],
        k[2] * r[0] - k[0] * r[2],
        k[0] * r[1] - k[1] * r[0],
    ], axis=0)
    rot = (r * math.cos(theta) + kcross * math.sin(theta)
           + k * kdotr * (1.0 - math.cos(theta)))
    norm = np.sqrt(rot[0] ** 2 + rot[1] ** 2 + rot[2] ** 2)
    rot = rot / norm
    return np.arctan2(rot[1], rot[0]), np.arcsin(np.clip(rot[2], -1.0, 1.0))


# ---------------------------------------------------------------------------
# KERN-2: deformational flows


def deformational_wind(case: str, lam, phi, t: float, *, radius: float,
                       period: float = CASE_PERIOD_S):
    """One of the four prescribed deformational flows, at time ``t``."""
    if case not in DEFORMATIONAL_CASES:
        raise ValueError(
            f"unknown deformational case {case!r}; the four of record are "
            f"{DEFORMATIONAL_CASES}"
        )
    k = 10.0 * radius / period
    taper = math.cos(math.pi * t / period)
    background = 2.0 * math.pi * radius / period
    #: the divergent field's own amplitude, half the non-divergent one's
    kd = 5.0 * radius / period
    if case == "nondivergent_one_cell":
        u = k * np.sin(0.5 * lam) ** 2 * np.sin(2.0 * phi) * taper
        v = 0.5 * k * np.sin(lam) * np.cos(phi) * taper
    elif case == "nondivergent_two_cell":
        u = k * np.sin(lam) ** 2 * np.sin(2.0 * phi) * taper
        v = k * np.sin(2.0 * lam) * np.cos(phi) * taper
    elif case == "divergent":
        # The published divergent field, coefficient for coefficient: half
        # the non-divergent deformation amplitude, the same constant
        # background rotation, and the same rotating longitude the
        # background implies.
        shifted = lam - background * t / radius
        u = (-kd * np.sin(0.5 * shifted) ** 2 * np.sin(2.0 * phi)
             * np.cos(phi) ** 2 * taper + background * np.cos(phi))
        v = 0.5 * kd * np.sin(shifted) * np.cos(phi) ** 3 * taper
    else:
        shifted = lam - background * t / radius
        u = (k * np.sin(shifted) ** 2 * np.sin(2.0 * phi) * taper
             + background * np.cos(phi))
        v = k * np.sin(2.0 * shifted) * np.cos(phi) * taper
    return u, v


#: Centres of the two-blob initial distributions, radians.
BLOB_CENTRES = ((5.0 * math.pi / 6.0, 0.0), (7.0 * math.pi / 6.0, 0.0))


def _great_circle(lam, phi, lam0, phi0):
    dot = (math.sin(phi0) * np.sin(phi)
           + math.cos(phi0) * np.cos(phi) * np.cos(lam - lam0))
    return np.arccos(np.clip(dot, -1.0, 1.0))


def cosine_bells(lam, phi, *, radius_rad: float = 0.5, background: float = 0.1,
                 amplitude: float = 0.9):
    out = np.full(np.shape(lam), background, dtype=np.float64)
    for lam0, phi0 in BLOB_CENTRES:
        r = _great_circle(lam, phi, lam0, phi0)
        inside = r < radius_rad
        h = 0.5 * (1.0 + np.cos(math.pi * r / radius_rad))
        out = np.where(inside, background + amplitude * h, out)
    return out


def gaussian_hills(lam, phi, *, width: float = 5.0, amplitude: float = 0.95):
    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi) + 0.0 * x
    out = np.zeros(np.shape(lam), dtype=np.float64)
    for lam0, phi0 in BLOB_CENTRES:
        x0 = math.cos(phi0) * math.cos(lam0)
        y0 = math.cos(phi0) * math.sin(lam0)
        z0 = math.sin(phi0)
        d2 = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
        out = out + amplitude * np.exp(-width * d2)
    return out


def slotted_cylinders(lam, phi, *, radius_rad: float = 0.5,
                      background: float = 0.1, amplitude: float = 1.0):
    out = np.full(np.shape(lam), background, dtype=np.float64)
    for lam0, phi0 in BLOB_CENTRES:
        r = _great_circle(lam, phi, lam0, phi0)
        inside = r <= radius_rad
        dlam = np.abs(((lam - lam0 + math.pi) % (2.0 * math.pi)) - math.pi)
        bridge = (dlam < radius_rad / 6.0) & (
            (phi - phi0) > -5.0 * radius_rad / 12.0)
        out = np.where(inside & ~bridge, amplitude, out)
    return out


DISTRIBUTIONS = {
    "cosine_bells": cosine_bells,
    "gaussian_hills": gaussian_hills,
    "slotted_cylinders": slotted_cylinders,
}


# ---------------------------------------------------------------------------
# norms


@dataclass(frozen=True)
class TransportNorms:
    l1: float
    l2: float
    linf: float
    mass_relative_change: float
    minimum: float
    maximum: float
    initial_minimum: float
    initial_maximum: float

    @property
    def undershoot(self) -> float:
        return self.minimum - self.initial_minimum

    @property
    def overshoot(self) -> float:
        return self.maximum - self.initial_maximum

    def as_dict(self) -> dict:
        return {
            "l1": self.l1,
            "l2": self.l2,
            "linf": self.linf,
            "mass_relative_change": self.mass_relative_change,
            "undershoot": self.undershoot,
            "overshoot": self.overshoot,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def transport_norms(final: np.ndarray, exact: np.ndarray,
                    weights: np.ndarray) -> TransportNorms:
    """The normalized error norms of Williamson and others, 1994."""
    err = final - exact
    w = weights
    denom1 = float(np.sum(w * np.abs(exact)))
    denom2 = float(np.sqrt(np.sum(w * exact ** 2)))
    m_final = float(np.sum(w * final))
    m_exact = float(np.sum(w * exact))
    return TransportNorms(
        l1=float(np.sum(w * np.abs(err))) / denom1,
        l2=float(np.sqrt(np.sum(w * err ** 2))) / denom2,
        linf=float(np.max(np.abs(err))) / float(np.max(np.abs(exact))),
        mass_relative_change=(m_final - m_exact) / m_exact,
        minimum=float(np.min(final)),
        maximum=float(np.max(final)),
        initial_minimum=float(np.min(exact)),
        initial_maximum=float(np.max(exact)),
    )


def mixing_diagnostics(chi: np.ndarray, xi: np.ndarray, weights: np.ndarray,
                       *, chi_min: float = 0.1, chi_max: float = 1.0,
                       curve=(-0.8, 0.0, 0.9)) -> dict:
    """Real mixing, range-preserving unmixing and overshooting.

    The pair of tracers starts on the concave curve ``xi = a*chi^2 + b*chi +
    c``.  A transport scheme moves points off that curve in three
    distinguishable ways and the three numbers separate them: a point that
    falls between the curve and the straight chord joining its endpoints has
    been really mixed; a point on the far side of the chord has been unmixed
    while staying inside the initial range; a point outside the initial range
    of either tracer has overshot.

    The definition used here, stated so it is not confused with another:
    each number is the area-weighted mean over the WHOLE sphere of the
    Euclidean distance in the (chi, xi) plane from the point to the boundary
    of its class, with the area weights normalized to one.  Points of the
    other two classes contribute zero to it.
    """
    a, b, c = curve

    def f(x):
        return a * x * x + b * x + c

    xi_lo, xi_hi = f(chi_min), f(chi_max)
    # the chord through the two endpoints of the curve
    slope = (xi_hi - xi_lo) / (chi_max - chi_min)

    def chord(x):
        return xi_lo + slope * (x - chi_min)

    in_range = (chi >= chi_min) & (chi <= chi_max)
    lo = min(xi_lo, xi_hi, f(-b / (2.0 * a)) if a != 0 else xi_lo)
    hi = max(xi_lo, xi_hi, f(-b / (2.0 * a)) if a != 0 else xi_hi)
    in_range = in_range & (xi >= min(lo, hi)) & (xi <= max(lo, hi))

    # signed distance to the curve, approximated by the vertical offset
    # scaled to a perpendicular distance through the local slope
    dfdx = 2.0 * a * chi + b
    scale = np.sqrt(1.0 + dfdx ** 2)
    above_curve = xi - f(chi)
    below_chord = chord(chi) - xi

    real = in_range & (above_curve <= 0.0) & (below_chord <= 0.0)
    unmix = in_range & (below_chord > 0.0)
    over = ~in_range

    d_real = np.abs(above_curve) / scale
    d_unmix = np.abs(below_chord) / math.sqrt(1.0 + slope * slope)
    d_over = np.maximum(
        np.maximum(chi_min - chi, chi - chi_max),
        np.maximum(min(lo, hi) - xi, xi - max(lo, hi)),
    )
    d_over = np.maximum(d_over, 0.0)

    total = float(np.sum(weights))
    return {
        "l_r": float(np.sum(weights * np.where(real, d_real, 0.0))) / total,
        "l_u": float(np.sum(weights * np.where(unmix, d_unmix, 0.0))) / total,
        "l_o": float(np.sum(weights * np.where(over, d_over, 0.0))) / total,
        "definition": (
            "area-weighted mean over the whole sphere of the distance in the "
            "(chi, xi) plane to the boundary of the point's class"
        ),
    }


# ---------------------------------------------------------------------------
# drivers


@dataclass
class CaseRun:
    name: str
    truncation: int
    steps: int
    dt_s: float
    monotone: bool
    iterations: int
    norms: TransportNorms
    trajectory_error_max_m: float = float("nan")
    extra: dict = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {
            "case": self.name,
            "truncation": self.truncation,
            "steps": self.steps,
            "dt_s": self.dt_s,
            "monotone": self.monotone,
            "trajectory_iterations": self.iterations,
            "trajectory_error_max_m": self.trajectory_error_max_m,
        }
        out.update(self.norms.as_dict())
        out.update(self.extra)
        return out


def _levels(field2d: np.ndarray, nlev: int) -> np.ndarray:
    return np.repeat(field2d[None, :, :], nlev, axis=0)


def run_solid_body(
    truncation: int,
    *,
    xp,
    dtype,
    alpha: float,
    steps: int = 256,
    distribution: str = "cosine_bells",
    monotone: bool = True,
    iterations: int = 3,
    nlev: int = 4,
    period: float = CASE_PERIOD_S,
) -> CaseRun:
    """KERN-1: one full revolution, exact solution equal to the start."""
    grid = GaussianGrid.create(int(truncation))
    tables = SphericalGridTables.create(grid, xp=xp, dtype=dtype)
    lam2, phi2 = _mesh(grid)
    weights = area_weights(grid)
    dt = period / float(steps)
    u0 = 2.0 * math.pi * grid.radius_m / period

    u2, v2 = solid_body_wind(lam2, phi2, u0=u0, alpha=alpha)
    u = xp.asarray(_levels(u2, nlev), dtype=dtype)
    v = xp.asarray(_levels(v2, nlev), dtype=dtype)
    vx, vy, vz = cartesian_wind(u, v, tables, xp=xp, dtype=dtype)
    rate = xp.zeros_like(vx)
    wind = CartesianWind(vx=xp.ascontiguousarray(vx),
                         vy=xp.ascontiguousarray(vy),
                         vz=xp.ascontiguousarray(vz), level_rate=rate)

    stencil, _ = departure_points(wind, tables, dt, iterations=iterations)
    exact_lam, exact_phi = solid_body_departure(
        lam2, phi2, u0=u0, alpha=alpha, dt=dt, radius=grid.radius_m)
    got_lam = np.asarray(_to_host(xp, stencil.xi))[0] * tables.dlam
    got_phi = np.asarray(_to_host(xp, stencil.phi))[0]
    dot = (np.sin(exact_phi) * np.sin(got_phi)
           + np.cos(exact_phi) * np.cos(got_phi)
           * np.cos(exact_lam - got_lam))
    traj_err = float(grid.radius_m
                     * np.max(np.arccos(np.clip(dot, -1.0, 1.0))))

    start = DISTRIBUTIONS[distribution](lam2, phi2)
    field = xp.asarray(_levels(start, nlev), dtype=dtype)
    for _ in range(int(steps)):
        field = gather_batch([field], stencil, monotone=monotone)[0]
    final = np.asarray(_to_host(xp, field))[0].astype(np.float64)

    return CaseRun(
        name=f"solid_body/alpha={alpha:.6f}/{distribution}",
        truncation=int(truncation), steps=int(steps), dt_s=dt,
        monotone=monotone, iterations=int(iterations),
        norms=transport_norms(final, start, weights),
        trajectory_error_max_m=traj_err,
    )


def run_deformational(
    truncation: int,
    *,
    xp,
    dtype,
    case: str,
    steps: int = 256,
    distribution: str = "gaussian_hills",
    monotone: bool = True,
    iterations: int = 3,
    nlev: int = 4,
    period: float = CASE_PERIOD_S,
    with_mixing: bool = False,
) -> CaseRun:
    """KERN-2: deform to filaments by ``T/2`` and return exactly by ``T``."""
    grid = GaussianGrid.create(int(truncation))
    tables = SphericalGridTables.create(grid, xp=xp, dtype=dtype)
    lam2, phi2 = _mesh(grid)
    weights = area_weights(grid)
    dt = period / float(steps)

    start = DISTRIBUTIONS[distribution](lam2, phi2)
    field = xp.asarray(_levels(start, nlev), dtype=dtype)
    partner = None
    if with_mixing:
        partner = xp.asarray(
            _levels(0.9 - 0.8 * start ** 2, nlev), dtype=dtype)

    half = None
    for index in range(int(steps)):
        t_mid = (index + 0.5) * dt
        u2, v2 = deformational_wind(case, lam2, phi2, t_mid,
                                    radius=grid.radius_m, period=period)
        u = xp.asarray(_levels(u2, nlev), dtype=dtype)
        v = xp.asarray(_levels(v2, nlev), dtype=dtype)
        vx, vy, vz = cartesian_wind(u, v, tables, xp=xp, dtype=dtype)
        wind = CartesianWind(xp.ascontiguousarray(vx),
                             xp.ascontiguousarray(vy),
                             xp.ascontiguousarray(vz),
                             xp.zeros_like(vx))
        stencil, _ = departure_points(wind, tables, dt, iterations=iterations)
        batch = [field] if partner is None else [field, partner]
        out = gather_batch(batch, stencil, monotone=monotone)
        field = out[0]
        if partner is not None:
            partner = out[1]
        if index + 1 == int(steps) // 2:
            half = np.asarray(_to_host(xp, field))[0].astype(np.float64)

    final = np.asarray(_to_host(xp, field))[0].astype(np.float64)
    extra: dict = {}
    if half is not None:
        extra["half_period_min"] = float(np.min(half))
        extra["half_period_max"] = float(np.max(half))
    if partner is not None:
        chi = final
        xi = np.asarray(_to_host(xp, partner))[0].astype(np.float64)
        extra.update({f"mixing_{k}": v
                      for k, v in mixing_diagnostics(chi, xi, weights).items()})

    return CaseRun(
        name=f"deformational/{case}/{distribution}",
        truncation=int(truncation), steps=int(steps), dt_s=dt,
        monotone=monotone, iterations=int(iterations),
        norms=transport_norms(final, start, weights),
        extra=extra,
    )


def _to_host(xp, array):
    if getattr(xp, "__name__", "") == "cupy":
        return xp.asnumpy(array)
    return np.asarray(array)


# ---------------------------------------------------------------------------
# KERN-2 against the shipped transport
#
# The semi-Lagrangian tracer path and the van Leer flux form are two answers
# to the same equation and the case below runs them through ONE driver: one
# grid, one initial field, one step count, one sample of the wind, and a
# branch of four lines where the transport happens.  Two drivers would be two
# things that can drift apart, and the whole value of this instrument is that
# the only difference between its arms is the scheme.
#
# The flux form is run on the NON-DIVERGENT cases only.  It transports MASS
# against a pseudo-density it carries itself, and a constant density is
# consistent with continuity exactly when the flow does not diverge; under
# case "divergent" the density would have to evolve and a comparison of two
# schemes would silently become a comparison of two density assumptions.
# The semi-Lagrangian arm runs every case, and the divergent one is reported
# alone rather than paired.


#: Layer thickness the two-scheme comparison runs on, Pa.  Any constant does,
#: because both arms are linear in it: the flux form carries mass = q*dp and
#: divides it back out, and the gather never sees it at all.
COMPARISON_DP_PA = 1000.0


def _flux_form_transport(transform, nlev: int, *, courant_limit: float = 0.25):
    """The shipped van Leer transport on a pure-sigma column.

    The comparison runs with no vertical motion at all, so the coordinate
    only has to exist and be a legal one; a uniform sigma ladder is the
    coordinate with the fewest choices in it, and every choice a hybrid
    ladder would add would be a difference between the arms that is not
    the transport scheme.
    """
    from ..transport import GridTracerTransport
    from ..vertical import HybridCoordinate

    return GridTracerTransport(
        transform=transform,
        vertical=HybridCoordinate(
            a_half_pa=np.zeros(int(nlev) + 1, dtype=np.float64),
            b_half=np.linspace(0.0, 1.0, int(nlev) + 1),
        ),
        courant_limit=float(courant_limit),
    )


def run_deformational_transport(
    transform,
    *,
    case: str,
    scheme: str,
    steps: int = 256,
    distribution: str = "gaussian_hills",
    monotone: bool = True,
    fixer: str = "bermejo_conde_additive",
    iterations: int = 3,
    nlev: int = 4,
    period: float = CASE_PERIOD_S,
) -> CaseRun:
    """KERN-2 through the tracer path of record, either scheme.

    ``scheme`` is ``"semi_lagrangian"`` (the stencil, the quasi-monotone
    limiter and the mass fixer, exactly as ``semilag.step`` runs them) or
    ``"flux_form"`` (``transport.GridTracerTransport``, exactly as
    ``dynamics.step`` runs it).  The exact solution at ``t = T`` is the
    initial field, so the norms are the scheme's own error over a full
    deformation and return.
    """
    if scheme not in ("semi_lagrangian", "flux_form"):
        raise ValueError(
            "the deformational comparison runs 'semi_lagrangian' or "
            f"'flux_form', got {scheme!r}"
        )
    if scheme == "flux_form" and case == "divergent":
        raise ValueError(
            "the flux-form arm is not run on the divergent case: it carries "
            "mass against a pseudo-density, and the constant density this "
            "comparison starts both arms from is consistent with continuity "
            "only where the flow does not diverge, so the arms would differ "
            "by a density assumption and not by a transport scheme"
        )
    backend = transform.backend
    xp = backend.xp
    dtype = backend.float_dtype
    grid = transform.grid
    tables = SphericalGridTables.create(grid, xp=xp, dtype=dtype)
    lam2, phi2 = _mesh(grid)
    weights = area_weights(grid)
    dt = float(period) / float(steps)

    start = DISTRIBUTIONS[distribution](lam2, phi2)
    field = xp.asarray(_levels(start, nlev), dtype=dtype)
    dp = xp.full((nlev, *grid.shape), dtype(COMPARISON_DP_PA), dtype=dtype)
    omega_half = xp.zeros((nlev + 1, *grid.shape), dtype=dtype)
    flux = _flux_form_transport(transform, nlev) if scheme == "flux_form" else None
    density = dp
    fixer_max = 0.0
    clip_share = 0.0
    substeps_max = 0
    half = None

    for index in range(int(steps)):
        t_mid = (index + 0.5) * dt
        u2, v2 = deformational_wind(case, lam2, phi2, t_mid,
                                    radius=grid.radius_m, period=period)
        u = xp.asarray(_levels(u2, nlev), dtype=dtype)
        v = xp.asarray(_levels(v2, nlev), dtype=dtype)
        if scheme == "flux_form":
            advanced, metrics = flux.advance(
                {"q": field}, dp, dp * u, dp * v, omega_half, dt, step=index,
            )
            field = advanced["q"]
            # The mixing ratio is re-referenced to the SAME density every
            # step, which is exactly what dynamics.step does: the layer
            # thickness a tracer sits in comes from the continuity, and
            # the transport's own pseudo-density is compared against it
            # rather than carried.  The two differ by the flow's discrete
            # divergence, which is zero for these cases analytically and
            # not on the grid, so the flux arm's mass moves by that
            # mismatch.  It is reported, and the alternative was measured
            # and refused: carrying the pseudo-density forward on a
            # prescribed wind compounds the same mismatch until the
            # density crosses zero and the positive-definite scheme fails
            # its own floor gate at step 24 of this case.
            density = metrics["pseudo_density"]
            substeps_max = max(
                substeps_max,
                *(int(metrics.get(f"substeps_{d}", 1)) for d in "xyz"),
            )
        else:
            vx, vy, vz = cartesian_wind(u, v, tables, xp=xp, dtype=dtype)
            wind = CartesianWind(xp.ascontiguousarray(vx),
                                 xp.ascontiguousarray(vy),
                                 xp.ascontiguousarray(vz),
                                 xp.zeros_like(vx))
            stencil, _ = departure_points(wind, tables, dt,
                                          iterations=iterations)
            want = bool(monotone and fixer in DEFICIT_FIXERS)
            rows = gather_batch([field], stencil, monotone=monotone,
                                deficit=want)
            if want:
                advanced, deficits = rows[0][0], {"q": rows[0][1]}
            else:
                advanced, deficits = rows[0], None
            fixed, marks = fix_mass(
                {"q": advanced}, {"q": field}, dp, dp, transform,
                scheme=fixer, deficits=deficits,
            )
            field = fixed["q"]
            fixer_max = max(
                fixer_max,
                float(marks["semilag_tracer_mass_fixer_relative__q"]),
            )
            clip_share = max(
                clip_share, float(marks["semilag_tracer_clip_share__q"])
            )
        if index + 1 == int(steps) // 2:
            half = np.asarray(_to_host(xp, field))[0].astype(np.float64)

    final = np.asarray(_to_host(xp, field))[0].astype(np.float64)
    rho = np.asarray(_to_host(xp, density))[0].astype(np.float64)
    mass_end = float(np.sum(weights * final * COMPARISON_DP_PA))
    mass_start = float(np.sum(weights * start * COMPARISON_DP_PA))
    extra: dict = {
        "scheme": scheme, "distribution": distribution,
        "tracer_mass_relative_change": (mass_end - mass_start) / mass_start,
        "pseudo_density_relative_drift": float(
            np.max(np.abs(rho - COMPARISON_DP_PA)) / COMPARISON_DP_PA
        ),
    }
    if scheme == "semi_lagrangian":
        extra["tracer_fixer"] = fixer
        extra["mass_fixer_max_step_relative"] = fixer_max
        extra["clip_share_max"] = clip_share
    else:
        extra["substeps_max"] = substeps_max
    if half is not None:
        extra["half_period_min"] = float(np.min(half))
        extra["half_period_max"] = float(np.max(half))
    return CaseRun(
        name=f"deformational/{case}/{distribution}/{scheme}",
        truncation=int(grid.truncation), steps=int(steps), dt_s=dt,
        monotone=monotone, iterations=int(iterations),
        norms=transport_norms(final, start, weights),
        extra=extra,
    )
