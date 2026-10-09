"""Point observation operators of the global DA streams that the v1 door
does not carry, with their calibration: GNSS radio-occultation
refractivity at a tangent height (as a column function and as a
member-batched operator the ensemble filter calls), and the
wind-at-assigned-pressure read an atmospheric motion vector needs (the
sounding operator, restated so its calibration is on record here beside
the refractivity's).

Every column operator here acts on ONE sampled column: profiles of
temperature (K), specific humidity (kg/kg) and pressure (Pa) on the
model's full levels, top to bottom, plus the surface pressure and the
surface geopotential for the height integration.  The column is what the
door's ``_ModelSpace`` and the ensemble filter's ``MemberOperators``
synthesize at a row's position; :class:`RefractivityOperator` does the
same synthesis for every member and hands the ensemble filter the
``(members, rows)`` array its ``PointObs.operator`` contract asks for, so
a refractivity batch joins the analysis as a foreign variable with its
own operator and O-A is judged on the analysed members like any other.

Refractivity (Smith and Weintraub 1953, the two-term form every RO
operator of record uses below 60 km)::

    N = 77.6 p / T + 3.73e5 e / T^2        p, e in hPa, T in K

with the vapour pressure ``e = q p / (eps + (1 - eps) q)``, eps 0.622.
The model's N is evaluated on its full levels, the levels are given
heights by the hypsometric integration from the surface (virtual
temperature, the model's gravity), and ``ln N`` is interpolated linearly
in height to the tangent height; a target outside the column's span is
refused (NaN), never extrapolated, because the row's error is a percent
of N and an extrapolated N above the top is a guess dressed as a number.
The retrieval's own refractivity is a local quantity at the tangent
point; the bending-angle operator the design prefers as the numerical
reference (an established implementation, the ray integral through the
model's refractivity) is not built here and is named in the module doc.

Acceptance contract of the refractivity row: measurement
``ro_refractivity_tangent_point``, time the occultation's reference
instant, location the tangent point at each level (the retrieval's own),
vertical coordinate HEIGHT above mean sea level in ``elevation_m`` (never
the dry pressure in ``level_pa``, carried for the column gates only),
representativeness one ray through a 52 km column, bias treatment none
below the superrefraction height (rows there are dropped by the door),
error 1 percent of N below 10 km and 2 percent above with no vertical
correlation stated (the door writes one row per 200 m, so adjacent rows
are correlated and the filter's thinning keeps one per cell and layer).

Calibration (``python -m woof.globe.obs_operators calibrate``;
``tests/test_arwen_global_obs_operators.py`` holds the bars) runs two
synthetic families both directions: a dry ISA column and a moist tropical
column with an exponential vapour profile, each on the model's 40 levels
against the same construction on 400 levels as the truth.  Direction one:
refractivity rows drawn from the truth at heights between the levels read
back within a few hundredths of a percent, except in the one layer that
straddles a lapse-rate kink (the ISA tropopause, the 195 K cap), where the
linear temperature between two levels cannot follow the corner and the
error reaches 0.42 (dry) and 0.54 (moist) percent; both numbers are
inside the 1 percent observation error and both are recorded.  Direction
two: the column warmed 1 K at fixed pressure is read at FIXED HEIGHT, where
the hydrostatic lift of the levels makes the response pass through zero
near 8 km; the operator's response matches the 400-level truth's to a
median 0.0005 to 0.001 N/T (maximum 0.10 at the kink layers), and at the
lowest target it matches the analytic -N_dry/T - 2 N_wet/T; an unchanged
column moves nothing (exact zero).  The member operator is calibrated on
the same families through a planted sampler (the synthesis replaced by
the column itself): the member read-back equals the column read-back to
rounding, a member missing a field is refused by name, and two members
that differ by the planted 1 K read the truth's response.
The wind operator's families: a profile linear in ln p (read back exactly
at every intermediate pressure) and a jet profile (bounded by the stated
0.5 m/s at 25 hPa spacing in the jet band).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from .constants import (
    DRY_AIR_GAS_CONSTANT as MODEL_DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2 as MODEL_GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
)

REFRACTIVITY_DRY_K_HPA = 77.6
REFRACTIVITY_WET_K2_HPA = 3.73e5
EPSILON = 0.622
GRAVITY_M_S2 = 9.80616
DRY_AIR_GAS_CONSTANT = 287.04

CALIBRATION_BARS = {
    # Read-back of rows drawn from the column at mid-layer heights: the
    # largest error sits in the one layer that straddles a lapse-rate kink
    # (the ISA tropopause, the moist family's 195 K cap), where a linear
    # temperature between two levels cannot follow the corner; elsewhere the
    # error is a few hundredths of a percent.  Both are inside the 1 percent
    # observation error and both are recorded.
    "refractivity_read_back_max_fraction": 0.006,
    "refractivity_read_back_median_fraction": 0.001,
    # The 1 K response at fixed height against the 400-level column's own
    # response, in units of N/T at the target (the scale of the response
    # itself); the kink layers again set the maximum.
    "refractivity_response_max_n_over_t": 0.15,
    "refractivity_response_median_n_over_t": 0.005,
    # At the lowest target the hydrostatic lift is small and the analytic
    # derivative -N/T (dry) -2 N_wet/T (wet) must be met within this
    # fraction: the sign and magnitude check of the response.
    "refractivity_surface_response_fraction": 0.10,
    "wind_linear_read_back_m_s": 1.0e-9,
    "wind_jet_read_back_m_s": 0.5,
}

#: The spectral fields a member must carry for the refractivity operator;
#: a member without one is refused by the field's name.
REFRACTIVITY_REQUIRED_FIELDS = ("log_surface_pressure", "theta", "qv")


def vapour_pressure_pa(q, p_pa):
    """Vapour pressure from specific humidity at pressure ``p_pa``."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p_pa, dtype=np.float64)
    return q * p / (EPSILON + (1.0 - EPSILON) * q)


def refractivity_n(p_pa, t_k, q):
    """Local refractivity in N-units from pressure (Pa), temperature (K)
    and specific humidity (kg/kg); the two-term Smith and Weintraub form."""
    p_hpa = np.asarray(p_pa, dtype=np.float64) / 100.0
    t = np.asarray(t_k, dtype=np.float64)
    e_hpa = vapour_pressure_pa(q, p_pa) / 100.0
    return REFRACTIVITY_DRY_K_HPA * p_hpa / t + REFRACTIVITY_WET_K2_HPA * e_hpa / (t * t)


def column_heights_m(t_k, q, p_pa, ps_pa, phi_s, *, gravity=GRAVITY_M_S2,
                     gas_constant=DRY_AIR_GAS_CONSTANT):
    """Heights (m above mean sea level) of a column's full levels by the
    hypsometric integration from the surface, ``t_k``, ``q`` and ``p_pa``
    ordered top to bottom; ``phi_s`` the surface geopotential (m2/s2)."""
    t = np.asarray(t_k, dtype=np.float64)
    qv = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    p = np.asarray(p_pa, dtype=np.float64)
    tv = t * (1.0 + 0.61 * qv)
    z = np.empty_like(p)
    z[-1] = phi_s / gravity + gas_constant * tv[-1] / gravity * math.log(ps_pa / p[-1])
    for k in range(p.size - 2, -1, -1):
        tv_layer = 0.5 * (tv[k] + tv[k + 1])
        z[k] = z[k + 1] + gas_constant * tv_layer / gravity * math.log(p[k + 1] / p[k])
    return z


def refractivity_at_heights(t_k, q, p_pa, ps_pa, phi_s, target_z_m, *,
                            gravity=GRAVITY_M_S2, gas_constant=DRY_AIR_GAS_CONSTANT):
    """Model refractivity at tangent heights.

    The full levels are given heights by :func:`column_heights_m`; inside
    the layer that holds a target, ``ln p`` is linear in height (the same
    layer-mean virtual temperature the hypsometric integration assumed, so
    the pressure at the target is the column's own), temperature and
    vapour are linear in ``ln p`` between the two levels, and N is
    evaluated from those three at the target.  Interpolating N itself (or
    ln N) in height would carry the layer's curvature into the value; the
    dry term ``77.6 p/T`` is captured exactly in ``p`` this way and only
    the temperature's within-layer shape remains.  Targets above the top
    full level or below the lowest are NaN (refused, not extrapolated).
    Returns ``(values, heights_of_levels)``."""
    t = np.asarray(t_k, dtype=np.float64)
    qv = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    p = np.asarray(p_pa, dtype=np.float64)
    z = column_heights_m(t, qv, p, ps_pa, phi_s, gravity=gravity, gas_constant=gas_constant)
    targets = np.asarray(target_z_m, dtype=np.float64)
    # Levels are top to bottom: z decreasing, p increasing.  Ascending
    # copies for the search.
    order = np.argsort(z)
    z_asc = z[order]
    ln_p_asc = np.log(p[order])
    t_asc = t[order]
    q_asc = qv[order]
    values = np.full(targets.shape, np.nan)
    inside = (targets >= z_asc[0]) & (targets <= z_asc[-1])
    if inside.any():
        zt = targets[inside]
        upper = np.clip(np.searchsorted(z_asc, zt, side="right"), 1, z_asc.size - 1)
        lower = upper - 1
        span = z_asc[upper] - z_asc[lower]
        frac = np.where(span > 0.0, (zt - z_asc[lower]) / np.where(span > 0.0, span, 1.0), 0.0)
        ln_p_t = ln_p_asc[lower] + frac * (ln_p_asc[upper] - ln_p_asc[lower])
        t_t = t_asc[lower] + frac * (t_asc[upper] - t_asc[lower])
        q_t = q_asc[lower] + frac * (q_asc[upper] - q_asc[lower])
        values[inside] = refractivity_n(np.exp(ln_p_t), t_t, q_t)
    return values, z


def refractivity_at_heights_columns(t_k, q, p_pa, ps_pa, phi_s, target_z_m, *,
                                    gravity=GRAVITY_M_S2, gas_constant=DRY_AIR_GAS_CONSTANT):
    """:func:`refractivity_at_heights` over MANY columns at once: ``t_k``,
    ``q`` and ``p_pa`` are ``(nlev, n)`` (top to bottom), ``ps_pa``,
    ``phi_s`` and ``target_z_m`` ``(n,)``, one target per column (a row's
    tangent height at the row's own tangent point).  The same arithmetic
    as the column function, vectorised over the columns: the heights by
    the hypsometric integration, ``ln p`` linear in height inside the
    layer that holds the target, temperature and vapour linear in ``ln p``
    between the two levels, the target outside the column's span NaN.
    The member-batched operator of the ensemble filter runs this on every
    row of a window in one call; ``tests/test_arwen_global_obs_operators.py``
    holds it to the column function to rounding."""
    t = np.asarray(t_k, dtype=np.float64)
    qv = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    p = np.asarray(p_pa, dtype=np.float64)
    ps = np.asarray(ps_pa, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_s, dtype=np.float64).reshape(-1)
    targets = np.asarray(target_z_m, dtype=np.float64).reshape(-1)
    nlev, n = p.shape
    tv = t * (1.0 + 0.61 * qv)
    z = np.empty_like(p)
    z[-1] = phi / gravity + gas_constant * tv[-1] / gravity * np.log(ps / p[-1])
    for k in range(nlev - 2, -1, -1):
        tv_layer = 0.5 * (tv[k] + tv[k + 1])
        z[k] = z[k + 1] + gas_constant * tv_layer / gravity * np.log(p[k + 1] / p[k])
    ln_p = np.log(p)
    # Levels are top to bottom: z decreasing with k.  The layer holding a
    # target: the first level (from the top) at or below it is ``lower``.
    inside = (targets >= z[-1]) & (targets <= z[0])
    below = (z <= targets[None, :])                      # (nlev, n) levels at or under the target
    lower = np.argmax(below, axis=0)                     # first True from the top
    lower = np.clip(lower, 1, nlev - 1)
    upper = lower - 1
    cols = np.arange(n)
    z_lo, z_up = z[lower, cols], z[upper, cols]
    span = z_up - z_lo
    frac = np.where(span > 0.0, (targets - z_lo) / np.where(span > 0.0, span, 1.0), 0.0)
    ln_p_t = ln_p[lower, cols] + frac * (ln_p[upper, cols] - ln_p[lower, cols])
    t_t = t[lower, cols] + frac * (t[upper, cols] - t[lower, cols])
    q_t = qv[lower, cols] + frac * (qv[upper, cols] - qv[lower, cols])
    values = np.full(n, np.nan)
    if inside.any():
        values[inside] = refractivity_n(np.exp(ln_p_t[inside]), t_t[inside], q_t[inside])
    return values, z


def interp_ln_pressure(profile, p_pa, target_p_pa):
    """Linear-in-ln p interpolation of one profile (top to bottom) to target
    pressures; targets outside the span are NaN.  The read an atmospheric
    motion vector and a sounding level take from the model column."""
    p = np.asarray(p_pa, dtype=np.float64)
    prof = np.asarray(profile, dtype=np.float64)
    targets = np.asarray(target_p_pa, dtype=np.float64)
    ln_p = np.log(p)
    order = np.argsort(ln_p)
    values = np.interp(np.log(targets), ln_p[order], prof[order])
    outside = (targets < p.min()) | (targets > p.max())
    return np.where(outside, np.nan, values)


# --------------------------------------------------------- member operator

def _default_sampler(transform, coeff, lat, lon):
    from woof.globe.spectral.sampling import sample_scalar

    return sample_scalar(transform, coeff, lat, lon)


def _default_host(transform, coeff):
    return np.asarray(transform.backend.to_numpy(coeff)).astype(np.complex128, copy=False)


class RefractivityOperator:
    """H(x_k) of refractivity rows on every member: the ensemble filter's
    ``PointObs.operator(members, batch) -> (R, n)``.

    ``transform`` and ``vertical`` are the ensemble's; ``terrain_spectral``
    the surface geopotential's coefficients (numpy complex).  For each
    member the surface pressure, potential temperature and vapour columns
    are synthesised at the batch's distinct points (``sampler``, by default
    the tree's ``sample_scalar``; ``host`` moves a member's coefficients to
    the host), the full-level pressures follow the hybrid coordinate, the
    temperature the Exner function, and :func:`refractivity_at_heights`
    reads ln N at each row's ``elevation_m`` (the tangent height) with the
    model's own gravity and gas constant.  A member without one of
    :data:`REFRACTIVITY_REQUIRED_FIELDS` is refused by that field's name;
    a batch of another variable is refused by name too.
    """

    def __init__(self, transform, vertical, terrain_spectral, *, sampler=None, host=None):
        self.transform = transform
        self.vertical = vertical
        self.terrain = np.asarray(terrain_spectral)
        self.a = np.asarray(vertical.a_half_pa, dtype=np.float64)
        self.b = np.asarray(vertical.b_half, dtype=np.float64)
        self._sampler = sampler or _default_sampler
        self._host = host or _default_host

    @classmethod
    def for_model(cls, model, transform, **kwargs) -> "RefractivityOperator":
        terrain = _default_host(transform, transform.forward(model.surface_geopotential))
        return cls(transform, model.vertical, terrain, **kwargs)

    def _member_fields(self, member, index: int):
        atmosphere = getattr(member, "atmosphere", member)
        missing = [name for name in REFRACTIVITY_REQUIRED_FIELDS
                   if not hasattr(atmosphere, name) or getattr(atmosphere, name) is None]
        if missing:
            raise ValueError(
                f"refractivity operator: member {index} lacks the spectral field(s) "
                f"{missing}; the operator needs {list(REFRACTIVITY_REQUIRED_FIELDS)} and "
                "refuses rather than read a column of nothing"
            )
        return tuple(self._host(self.transform, getattr(atmosphere, name))
                     for name in REFRACTIVITY_REQUIRED_FIELDS)

    def __call__(self, members, batch) -> np.ndarray:
        variable = getattr(batch, "variable", "refractivity_n")
        if variable != "refractivity_n":
            raise ValueError(
                f"refractivity operator asked to evaluate {variable!r}; it evaluates "
                "refractivity_n rows only"
            )
        lat = np.asarray(batch.latitude_deg, dtype=np.float64).reshape(-1)
        lon = np.asarray(batch.longitude_deg, dtype=np.float64).reshape(-1)
        target_z = np.asarray(batch.elevation_m, dtype=np.float64).reshape(-1)
        n = lat.size
        members = list(members)
        out = np.full((len(members), n), np.nan)
        if n == 0:
            return out
        pairs = np.stack([lat, lon], axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        ulat, ulon = unique[:, 0].copy(), unique[:, 1].copy()
        at = np.asarray(inverse).ravel()
        phi_s = np.asarray(self._sampler(self.transform, self.terrain, ulat, ulon), dtype=np.float64)
        nlev = self.a.size - 1
        for k, member in enumerate(members):
            lnps, theta, qv = self._member_fields(member, k)
            stack = np.concatenate([theta, qv, lnps[None]], axis=0)
            sampled = np.asarray(self._sampler(self.transform, stack, ulat, ulon), dtype=np.float64)
            ps = np.exp(sampled[-1])                                    # (U,)
            p_half = self.a[:, None] + self.b[:, None] * ps[None, :]      # (nlev+1, U)
            p_full = np.sqrt(p_half[:-1] * p_half[1:])                    # (nlev, U)
            t_prof = sampled[:nlev] * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
            q_prof = np.maximum(sampled[nlev:2 * nlev], 0.0)
            for u in range(ulat.size):
                rows_here = np.nonzero(at == u)[0]
                got, _ = refractivity_at_heights(
                    t_prof[:, u], q_prof[:, u], p_full[:, u], ps[u], phi_s[u], target_z[rows_here],
                    gravity=MODEL_GRAVITY_M_S2, gas_constant=MODEL_DRY_AIR_GAS_CONSTANT,
                )
                out[k, rows_here] = got
        return out


# ------------------------------------------------------------- calibration

def _isa_column(n_levels: int = 40, p_top_pa: float = 1000.0):
    """A dry ISA column on ``n_levels`` full levels equally spaced in ln p."""
    ps = 101_325.0
    p = np.exp(np.linspace(math.log(p_top_pa), math.log(ps * 0.995), n_levels))
    t = np.where(p >= 22_632.06,
                 288.15 * (p / 101_325.0) ** (1.0 / 5.255877),
                 216.65)
    q = np.zeros_like(p)
    return t, q, p, ps, 0.0


def _isa_height_m(p_pa):
    p = np.asarray(p_pa, dtype=np.float64)
    trop = 288.15 / 0.0065 * (1.0 - (p / 101_325.0) ** (1.0 / 5.255877))
    strat = 11_000.0 - 6341.62 * np.log(p / 22_632.06)
    return np.where(p >= 22_632.06, trop, strat)


def _tropical_column(n_levels: int = 40):
    """A moist column: 300 K surface, 6.5 K/km to a 100 hPa tropopause at
    195 K, vapour 18 g/kg at the surface falling with a 2.5 km scale."""
    ps = 100_800.0
    p = np.exp(np.linspace(math.log(1000.0), math.log(ps * 0.995), n_levels))
    z_guess = _isa_height_m(p)
    t = np.maximum(300.0 - 0.0065 * z_guess, 195.0)
    q = 0.018 * np.exp(-z_guess / 2500.0)
    return t, q, p, ps, 0.0


class _PlantedMember:
    """A member whose spectral fields ARE the planted column: the sampler
    below returns them at every point, so the member operator's arithmetic
    is measured against the column functions with no synthesis in the
    way.  ``theta`` is the column's potential temperature on the levels,
    ``qv`` its vapour, ``log_surface_pressure`` ln ps."""

    def __init__(self, t_k, q, p_pa, ps_pa):
        class _Atmosphere:
            pass

        atmosphere = _Atmosphere()
        atmosphere.theta = np.asarray(t_k, dtype=np.float64) * (
            REFERENCE_PRESSURE_PA / np.asarray(p_pa, dtype=np.float64)) ** KAPPA
        atmosphere.qv = np.asarray(q, dtype=np.float64)
        atmosphere.log_surface_pressure = np.array(math.log(ps_pa))
        self.atmosphere = atmosphere


class _PlantedVertical:
    """A hybrid coordinate whose full levels reproduce the planted column's
    pressures at its surface pressure: pure pressure levels (b = 0) whose
    half levels are the geometric mid-points, so ``sqrt(p_half p_half)``
    returns the column's own levels exactly."""

    def __init__(self, p_full_pa):
        p = np.asarray(p_full_pa, dtype=np.float64)
        ratio = p[1] / p[0]
        half = np.empty(p.size + 1)
        half[0] = p[0] / math.sqrt(ratio)
        for k in range(p.size):
            half[k + 1] = p[k] * p[k] / half[k]
        self.a_half_pa = half
        self.b_half = np.zeros_like(half)


class _PlantedTransform:
    class backend:  # noqa: N801
        @staticmethod
        def to_numpy(x):
            return np.asarray(x)


class _Batch:
    def __init__(self, lat, lon, z, variable="refractivity_n"):
        self.latitude_deg = np.asarray(lat, dtype=np.float64)
        self.longitude_deg = np.asarray(lon, dtype=np.float64)
        self.elevation_m = np.asarray(z, dtype=np.float64)
        self.variable = variable


def planted_member_operator(t_k, q, p_pa, ps_pa, phi_s=0.0):
    """A :class:`RefractivityOperator` bound to the planted column (the
    calibration's and the tests' way in)."""
    vertical = _PlantedVertical(p_pa)
    terrain = np.array([phi_s])
    return RefractivityOperator(
        _PlantedTransform(), vertical, terrain,
        sampler=lambda tr, c, lat, lon: _planted_sampler_stack(c, lat),
        host=lambda tr, c: np.asarray(c, dtype=np.float64),
    )


def _planted_sampler_stack(coeff, lat):
    coeff = np.asarray(coeff, dtype=np.float64)
    n = np.asarray(lat).size
    if coeff.ndim == 1 and coeff.size == 1:
        return np.full(n, float(coeff[0]))
    return np.repeat(coeff.reshape(coeff.shape[0], -1)[:, :1], n, axis=1)


def calibrate() -> dict:
    """The two-family, two-direction calibration of the column operators
    and the member operator."""
    report: dict = {"bars": CALIBRATION_BARS, "families": {}, "member_operator": {}}
    for name, column in (("isa_dry", _isa_column()), ("tropical_moist", _tropical_column())):
        t, q, p, ps, phi_s = column
        z = column_heights_m(t, q, p, ps, phi_s)
        # Direction one: rows drawn from a finer version of the same column
        # (the analytic ISA / the same construction on 400 levels) at
        # heights between the coarse levels.
        if name == "isa_dry":
            t_f, q_f, p_f, ps_f, _ = _isa_column(400)
        else:
            t_f, q_f, p_f, ps_f, _ = _tropical_column(400)
        z_f = column_heights_m(t_f, q_f, p_f, ps_f, 0.0)
        n_f = refractivity_n(p_f, t_f, q_f)
        targets = 0.5 * (z[1:] + z[:-1])
        targets = targets[(targets > z_f.min()) & (targets < z_f.max())]
        truth = np.exp(np.interp(targets, z_f[::-1], np.log(n_f[::-1])))
        got, _ = refractivity_at_heights(t, q, p, ps, phi_s, targets)
        read_back_errors = np.abs(got / truth - 1.0)
        read_back = float(np.nanmax(read_back_errors))
        read_back_median = float(np.nanmedian(read_back_errors))
        # Direction two: warm the column 1 K at fixed pressure and read the
        # response at FIXED HEIGHT.  Warming lifts the pressure levels
        # hydrostatically, so at a fixed height the pressure rises too and
        # the response is not -N/T except near the surface: it passes
        # through zero near 8 km and changes sign above.  The reference is
        # the 400-level column's own response (the synthetic family's
        # truth), judged in units of N/T at the target; the analytic
        # derivative -N_dry/T - 2 N_wet/T is met at the lowest target,
        # where the lift is small, as the sign-and-magnitude check.
        got_warm, _ = refractivity_at_heights(t + 1.0, q, p, ps, phi_s, targets)
        z_fw = column_heights_m(t_f + 1.0, q_f, p_f, ps_f, 0.0)
        n_fw = refractivity_n(p_f, t_f + 1.0, q_f)
        truth_warm = np.exp(np.interp(targets, z_fw[::-1], np.log(n_fw[::-1])))
        expected = truth_warm - truth
        t_at = np.interp(targets, z_f[::-1], t_f[::-1])
        p_at = np.exp(np.interp(targets, z_f[::-1], np.log(p_f[::-1])))
        q_at = np.interp(targets, z_f[::-1], q_f[::-1])
        n_dry = REFRACTIVITY_DRY_K_HPA * (p_at / 100.0) / t_at
        n_wet = REFRACTIVITY_WET_K2_HPA * (vapour_pressure_pa(q_at, p_at) / 100.0) / (t_at * t_at)
        scale = (n_dry + n_wet) / t_at
        response = got_warm - got
        response_errors = np.abs(response - expected) / scale
        response_error = float(np.nanmax(response_errors))
        response_median = float(np.nanmedian(response_errors))
        lowest = int(np.argmin(targets))
        analytic_surface = -n_dry[lowest] / t_at[lowest] - 2.0 * n_wet[lowest] / t_at[lowest]
        surface_response_error = float(abs(response[lowest] / analytic_surface - 1.0))
        # An unchanged column moves nothing: exact zero by construction.
        got_same, _ = refractivity_at_heights(t, q, p, ps, phi_s, targets)
        unchanged = float(np.nanmax(np.abs(got_same - got)))
        # The vectorised column form (what the ensemble filter's operators
        # run) against the column function, target by target.
        columns, _ = refractivity_at_heights_columns(
            np.repeat(t[:, None], targets.size, axis=1), np.repeat(q[:, None], targets.size, axis=1),
            np.repeat(p[:, None], targets.size, axis=1), np.full(targets.size, ps),
            np.full(targets.size, phi_s), targets)
        vectorised_minus_column = float(np.nanmax(np.abs(columns - got)))
        # Above the top and below the surface: refused.
        refused, _ = refractivity_at_heights(t, q, p, ps, phi_s, np.array([z.max() + 5000.0, z.min() - 500.0]))
        report["families"][name] = {
            "levels": int(p.size),
            "targets": int(targets.size),
            "read_back_max_fraction": read_back,
            "read_back_median_fraction": read_back_median,
            "read_back_worst_target_m": float(targets[int(np.nanargmax(read_back_errors))]),
            "response_vs_fine_max_n_over_t": response_error,
            "response_vs_fine_median_n_over_t": response_median,
            "response_surface_vs_analytic_fraction": surface_response_error,
            "unchanged_column_moves": unchanged,
            "vectorised_minus_column_max": vectorised_minus_column,
            "outside_span_refused": bool(np.all(np.isnan(refused))),
            "n_surface": float(refractivity_n(p[-1], t[-1], q[-1])),
            "pass": (read_back <= CALIBRATION_BARS["refractivity_read_back_max_fraction"]
                     and read_back_median <= CALIBRATION_BARS["refractivity_read_back_median_fraction"]
                     and response_error <= CALIBRATION_BARS["refractivity_response_max_n_over_t"]
                     and response_median <= CALIBRATION_BARS["refractivity_response_median_n_over_t"]
                     and surface_response_error <= CALIBRATION_BARS["refractivity_surface_response_fraction"]
                     and unchanged == 0.0 and vectorised_minus_column <= 1.0e-9
                     and bool(np.all(np.isnan(refused)))),
        }
        # The member operator on the same column through the planted
        # sampler: two members, the column and the column warmed 1 K.
        operator = planted_member_operator(t, q, p, ps, phi_s)
        members = [_PlantedMember(t, q, p, ps), _PlantedMember(t + 1.0, q, p, ps)]
        batch = _Batch(np.full(targets.size, 10.0), np.full(targets.size, 20.0), targets)
        member_values = operator(members, batch)
        # The member arithmetic uses the model's gravity and gas constant;
        # compare against the column function with the same constants.
        column_values, _ = refractivity_at_heights(
            t, q, p, ps, phi_s, targets,
            gravity=MODEL_GRAVITY_M_S2, gas_constant=MODEL_DRY_AIR_GAS_CONSTANT)
        member_read_back = float(np.nanmax(np.abs(member_values[0] - column_values)))
        member_response = member_values[1] - member_values[0]
        member_response_error = float(np.nanmax(np.abs(member_response - expected) / scale))
        report["member_operator"][name] = {
            "members": 2,
            "rows": int(targets.size),
            "member_minus_column_max": member_read_back,
            "response_vs_fine_max_n_over_t": member_response_error,
            "pass": (member_read_back <= 1.0e-9
                     and member_response_error <= CALIBRATION_BARS["refractivity_response_max_n_over_t"]),
        }
    # Wind at an assigned pressure.
    p = np.exp(np.linspace(math.log(1000.0), math.log(100_000.0), 40))
    linear = 3.0 + 4.0 * np.log(p / 1.0e5)
    targets = np.exp(0.5 * (np.log(p[1:]) + np.log(p[:-1])))
    got = interp_ln_pressure(linear, p, targets)
    linear_error = float(np.max(np.abs(got - (3.0 + 4.0 * np.log(targets / 1.0e5)))))
    # A jet: 60 m/s Gaussian in ln p around 250 hPa, width 0.25 in ln p.
    jet = lambda pp: 60.0 * np.exp(-((np.log(pp / 25_000.0)) / 0.25) ** 2)  # noqa: E731
    p_jet = np.exp(np.linspace(math.log(10_000.0), math.log(60_000.0), 40))  # about 25 hPa apart in the jet band
    got_jet = interp_ln_pressure(jet(p_jet), p_jet, targets[(targets > 10_000.0) & (targets < 60_000.0)])
    truth_jet = jet(targets[(targets > 10_000.0) & (targets < 60_000.0)])
    jet_error = float(np.nanmax(np.abs(got_jet - truth_jet)))
    refused = interp_ln_pressure(linear, p, np.array([500.0, 101_000.0]))
    report["wind_at_pressure"] = {
        "linear_in_ln_p_read_back_m_s": linear_error,
        "jet_read_back_max_m_s": jet_error,
        "outside_span_refused": bool(np.all(np.isnan(refused))),
        "pass": (linear_error <= CALIBRATION_BARS["wind_linear_read_back_m_s"]
                 and jet_error <= CALIBRATION_BARS["wind_jet_read_back_m_s"]
                 and bool(np.all(np.isnan(refused)))),
    }
    report["pass"] = (
        all(f["pass"] for f in report["families"].values())
        and all(f["pass"] for f in report["member_operator"].values())
        and report["wind_at_pressure"]["pass"]
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m woof.globe.obs_operators")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate", help="run both families both directions and print the report")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        report = calibrate()
        print(json.dumps(report, indent=1))
        return 0 if report["pass"] else 1
    return 2


__all__ = [
    "CALIBRATION_BARS",
    "REFRACTIVITY_REQUIRED_FIELDS",
    "RefractivityOperator",
    "calibrate",
    "column_heights_m",
    "interp_ln_pressure",
    "planted_member_operator",
    "refractivity_at_heights",
    "refractivity_at_heights_columns",
    "refractivity_n",
    "vapour_pressure_pa",
]


if __name__ == "__main__":
    sys.exit(main())
