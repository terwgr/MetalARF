"""WOOF global's own convection closure: ``arwen-massflux-v1``.

A scale-aware bulk mass-flux scheme with a deep and a shallow branch,
written for the native suite's own grid (the T255-T533 Gaussian grids,
52-25 km at the equator), its own vertical split (40 hybrid levels, first
full level 22.9 m above ground) and its own step (50-200 s under the
vertical-mode semi-implicit dycore).  It is selected with
``cumulus = "own"`` in the native adapter options and runs in the cumulus
slot between YSU and Morrison exactly where Grell-Freitas runs.

The column model
----------------
*Source layer.*  Every full level whose height lies below the held PBL
depth (clipped to ``source_depth_min_m``..``source_depth_max_m``; level 0
always) feeds the updraft, mass-weighted, so the plume leaves the source
top with the layer's mean dry static energy, water and momentum and unit
normalised mass flux.

*Trigger.*  The plume rises with an explicit vertical-momentum equation,
d(w^2)/dz = 2 a B - 2 b eps w^2 (Simpson and Wiggert), from an initial
w^2 = ``initial_w2_m2_s2`` + 4 w*^2, where w* is the convective velocity
scale of the surface buoyancy flux the runtime hands the slot (hfx, qfx,
PBL depth).  The buoyancy carries a surface-flux temperature excess
min(``trigger_max_k``, ``trigger_scale`` theta*) in the unsaturated
sub-cloud part of the ascent, diluted by entrainment.  A column whose
plume cannot cross its CIN (w^2 reaches zero before or in cloud) does not
convect; the dilute CAPE the plume accumulates is the instability the
closure is bounded by.  The trigger reads the column as it stands after
this call's radiation and PBL forcing, so the instability those rates
generate is seen on the same call.

*Branches.*  A plume computed with the deep entrainment and precipitation
conversion whose cloud depth reaches ``deep_minimum_depth_m`` is deep.
Otherwise a second plume with the shallow entrainment, no precipitation
conversion and a depth cap gives the shallow branch (cloud depth at least
``shallow_minimum_depth_m``).  A column runs at most one branch.

*Scale awareness (the function of dx).*  The updraft area fraction of a
grid cell is sigma(dx) = min(1, (dx_c / dx)^2) with dx_c =
``dx_convective_m`` (4 km, the spacing at which deep convection is
resolved).  The deep base mass flux is multiplied by beta(dx) =
(1 - sigma)^2 (the Arakawa-Wu vanishing-area factor) and the deep
entrainment rate is eps_deep(dx) = eps_deep0 / max(1 - sigma, 0.05), so
the deep branch dilutes faster and carries less as the grid refines and is
shut (beta = 0) at dx <= dx_c.  At 52 km sigma is 0.006 and beta 0.988;
at 25 km 0.026 and 0.949; at 10 km 0.16 and 0.71; at 5 km 0.64 and
0.13.  The shallow branch is not scaled: it is sub-grid at every one of
these spacings.  dx is the adapter's ``dx_m`` (the truncation's equatorial
spacing, applied everywhere because the spectral truncation is isotropic).

*Closure.*  Deep: the mean of a boundary-layer quasi-equilibrium mass flux
M_blqe = F_h / (h_src - h_above) (F_h the moist-static-energy forcing of
the source layer from the PBL and radiation rates the runtime hands the
slot, or the surface fluxes when no rates are handed; the denominator the
source-layer excess over the air just above it, floored at
``blqe_h_gap_floor_j_kg``) and a moisture-convergence mass flux
M_mconv = MC / P_unit (MC the column vertical moisture convergence
-int omega dq/dp dp/g below cloud top, P_unit the precipitation the
normalised plume produces per unit base mass flux), times beta(dx),
bounded by the CAPE quasi-equilibrium flux CAPE / (tau dCAPE/dM) with
tau = ``cape_timescale_s`` and dCAPE/dM the environmental warming the
unit plume produces through the cloud layer.  Shallow: M = 0.03 rho w*
(Grant's boundary-layer scaling).  Both branches are bounded so that no
level loses more than ``transfer_bound`` of its own mass to entrainment
plus subsidence in one step, which is what keeps every species positive.

*Entrainment and detrainment.*  Fractional entrainment per metre is the
branch's rate; turbulent detrainment balances it (unit mass flux) up to
the first level of negative buoyancy in cloud, above which the mass flux
falls linearly to zero at the cloud top (organised detrainment).  Cloud
water and ice detrained there enter Morrison's qc and qi species; the
liquid/ice split of updraft condensate is linear in temperature between
273.15 K and 253.15 K, freezing releasing the latent heat of fusion in
the updraft.  Saturation is over liquid water throughout (a stated
simplification).

*Precipitation.*  Updraft condensate converts at
1 - exp(-``deep_conversion_per_m`` dz) per layer; the precipitation falls
through the column, evaporates (sublimates) into sub-saturated air below
cloud base, bounded by the layer's saturation deficit, and its frozen part
melts where the environment is above freezing, bounded to cool the layer
halfway to 0 C per step; whatever ice remains melts in the lowest layer
so the surface precipitation is liquid.  There is no separate downdraft:
the evaporative cooling below cloud base is the scheme's downdraft
effect.

*Convective momentum transport.*  u and v are transported by the same
entraining plume with the Gregory pressure-gradient term
``momentum_pgf_coefficient`` (0.7) making the updraft momentum follow the
environmental shear; the grid-mean tendency is the flux-form
-d[M (u_up - u)]/dz, which conserves column momentum exactly.  The kinetic
energy the transport removes is not returned as heat.

Ledgers (exact by construction, tested to 1e-6 relative in float64)
-------------------------------------------------------------------
Per column, in the scheme's own metric (dry mixing ratio times dp/g):

* water: -sum_k dp_k/g (dqv + dqc + dqi)_k dt = RAINC;
* enthalpy: sum_k dp_k/g (cp dT + L_v dqv - L_f dqi)_k = 0, i.e. the
  sensible heating equals the latent heat of the vapour condensed (the
  precipitation plus the detrained condensate) net of the evaporation and
  melting of the precipitation;
* momentum: sum_k dp_k/g du_k = 0 and the same for v.

The runtime credits the surface reservoir from the batch's own measured
column-water change in the model's specific-humidity metric, which differs
from RAINC by O(q); RAINC feeds the accumulators and the land bucket.

Not carried: the dycore's advective tendencies (the exchange hands none),
precipitation species (qr, qs, qg) and number concentrations are not
transported, no refreezing of liquid precipitation.

Grade of record: the scheme lost, so it stays selectable and not default
-----------------------------------------------------------------------
T255, 40 levels, 24 h from GDAS 2026-09-01 00Z with real statics, f024
against GFS, the 12-24 h 250 hPa spectrum, matched-dt arms on this tree
(own / Grell-Freitas):

* dt 100 s: CONUS land 2 m bias +1.35 / +0.66 K, rmse 2.79 / 2.43 K; NH
  mid-latitude land -0.38 / -0.65 K, rmse 2.72 / 2.63 K; global MSLP rmse
  2.38 / 2.38 hPa; 10 m wind bias +0.30 / +0.29 m/s, rmse 1.77 / 1.93 m/s;
  grid-limit energy 0.50x / 0.68x observed, NH 250 km 0.83x / 1.00x,
  global 250 km 0.61x / 0.83x; divergent share of the 100-200 km band
  0.46 / 0.55 (NH 0.42 / 0.47).
* dt 50 s: CONUS +1.50 / +0.72 K, rmse 2.91 / 2.47 K; NH mid-latitude
  -0.33 / -0.64 K, rmse 2.78 / 2.68 K; global MSLP rmse 2.17 / 2.18 hPa;
  10 m wind +0.29 / +0.27 m/s, rmse 1.77 / 1.96 m/s; grid-limit 0.51x /
  0.70x, NH 250 km 0.81x / 1.00x, global 0.60x / 0.84x; divergent share
  0.48 / 0.58 (NH 0.44 / 0.49).
* Rain: global mean 24 h convective rain 2.11 mm (dt 100) and 2.08 mm
  (dt 50) against GF 2.28 / 2.24 mm, the GFS 1.77 mm convective and
  2.95 mm total; over land 1.15 mm against GF 1.77 and GFS 0.96.  The
  rain lands where CAPE is: rain-weighted mean CAPE 3.41x the area mean
  (GF 2.94x, the GFS convective rain itself 2.60x), 85 percent of the
  convective rain on the 25 percent of columns with CAPE >= 200 J/kg
  (GF 80, GFS 71), Spearman rain against CAPE 0.64 (GF 0.71, GFS 0.75),
  Spearman against the GFS convective rain 0.67 (GF 0.77).

The rule (not worse on CONUS and NH mid-latitude 2 m bias and rmse,
global MSLP rmse and 10 m wind, and a lower NH 250 km ratio and
divergent share) fails on the CONUS and NH mid-latitude 2 m rmse at
both steps: the scheme lowers the divergent share by 0.07-0.10 and
takes the NH 250 km energy from the 1.00x observed GF holds down to
0.81-0.83x, which is damping below the observed level rather than a
cleaner cascade, and it warms CONUS land a further 0.7-0.8 K at f024
with 0.3-0.4 K more rmse.  The CONUS warming with the small land rain (1.15
against GF's 1.77 mm) is the lead the next revision should follow: too
little convective rain and cloud over summer land, and the closure
damping the grid scales that carry the observed energy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import numpy as np

from ..constants import (
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    EPSILON,
    GRAVITY_M_S2,
    LATENT_HEAT_FUSION,
    LATENT_HEAT_VAPORIZATION,
)

SCHEME_NAME = "arwen-massflux-v1"
FREEZING_K = 273.15
#: Updraft condensate is all liquid at FREEZING_K and all ice at this
#: temperature, linearly between.
ALL_ICE_K = 253.15
#: Columns packed per pass; the same reasoning as woof.globe.core.gf
#: GF_COLUMN_CHUNK (a whole T533 batch's working set is several GB and the
#: scheme never reads across columns).
MASSFLUX_COLUMN_CHUNK = 131_072


@dataclass(frozen=True)
class MassFluxParameters:
    """The scheme's constants; every one is named where it is used."""

    dx_convective_m: float = 4_000.0
    deep_entrainment_per_m: float = 1.0e-4
    shallow_entrainment_per_m: float = 2.0e-3
    deep_conversion_per_m: float = 2.0e-3
    deep_minimum_depth_m: float = 3_000.0
    shallow_minimum_depth_m: float = 300.0
    shallow_maximum_depth_m: float = 3_000.0
    source_depth_min_m: float = 100.0
    source_depth_max_m: float = 1_500.0
    initial_w2_m2_s2: float = 1.0
    trigger_scale: float = 2.0
    trigger_max_k: float = 1.0
    buoyancy_coefficient: float = 2.0 / 3.0     # a: virtual-mass reduced
    drag_coefficient: float = 1.0              # b: entrainment drag on w^2
    cape_timescale_s: float = 3_600.0
    cape_minimum_j_kg: float = 10.0
    cape_response_floor: float = 1.0           # J/kg per unit mass flux
    blqe_h_gap_floor_j_kg: float = 2_000.0
    precipitation_unit_floor: float = 1.0e-3   # kg/kg per unit mass flux
    shallow_mass_flux_coefficient: float = 0.03
    momentum_pgf_coefficient: float = 0.7
    evaporation_coefficient: float = 0.5       # per 100 hPa at zero RH
    transfer_bound: float = 0.5
    top_levels_excluded: int = 3

    @property
    def identity(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class MassFluxResult:
    """The CumulusResult field names (woof.globe.core.physics) plus momentum.

    ``rthcuten`` theta K/s, ``rq*cuten`` dry mixing ratio kg/kg/s,
    ``rucuten``/``rvcuten`` m/s^2, ``rainc`` the call's RAINCV in kg/m2;
    ``diagnostics`` grid-mean readings of the call.
    """

    rthcuten: object
    rqvcuten: object
    rqccuten: object
    rqicuten: object
    rucuten: object
    rvcuten: object
    rainc: object
    diagnostics: dict = field(default_factory=dict)


def scale_factors(dx_m: float, params: MassFluxParameters) -> tuple[float, float, float]:
    """(sigma, beta, eps_deep) for this grid spacing; see the module docstring."""
    dx = float(dx_m)
    if not dx > 0.0:
        raise ValueError(f"dx_m must be positive, got {dx_m!r}")
    sigma = min(1.0, (params.dx_convective_m / dx) ** 2)
    beta = (1.0 - sigma) ** 2
    eps_deep = params.deep_entrainment_per_m / max(1.0 - sigma, 0.05)
    return sigma, beta, eps_deep


def saturation_mixing_ratio(xp, temperature, pressure):
    """Bolton saturation mixing ratio over liquid water (kg/kg)."""
    # Clamped below so a masked-out parcel's placeholder state cannot
    # overflow the exponential; 100 K is far below any live column.
    t = xp.maximum(temperature, 100.0)
    es = 611.2 * xp.exp(17.67 * (t - FREEZING_K) / (t - 29.65))
    es = xp.minimum(es, 0.5 * pressure)
    return EPSILON * es / (pressure - es)


def _saturation_slope(xp, temperature, qs):
    return qs * (17.67 * 243.5) / (temperature - 29.65) ** 2


def _ice_fraction(xp, temperature):
    return xp.clip(
        (FREEZING_K - temperature) / (FREEZING_K - ALL_ICE_K), 0.0, 1.0
    )


def _virtual(temperature, r):
    return temperature * (1.0 + r / EPSILON) / (1.0 + r)


def _gather(xp, level_array, index):
    """level_array[index[j], j] for every column j."""
    return xp.take_along_axis(level_array, index[None, :], axis=0)[0]


def _condense(xp, s_mix, q_mix, ll_mix, li_mix, gz, p):
    """Saturation adjustment of a mixed updraft parcel.

    Returns (s, q, ll, li, l_total) after condensation/evaporation and
    freezing/melting, with ``s`` the budget value s_mix + L_v (l - l_mix)
    + L_f (li - li_mix) so the ledger is algebraic, and the temperature
    consistent with that ``s``.
    """
    cp = DRY_AIR_CP
    lv = LATENT_HEAT_VAPORIZATION
    lf = LATENT_HEAT_FUSION
    qt = q_mix + ll_mix + li_mix
    l_mix = ll_mix + li_mix
    t0 = (s_mix - gz) / cp
    t = t0
    for _ in range(4):
        qs = saturation_mixing_ratio(xp, t, p)
        l_new = xp.maximum(qt - qs, 0.0)
        fi = _ice_fraction(xp, t)
        residual = cp * (t - t0) - lv * (l_new - l_mix) - lf * (fi * l_new - li_mix)
        slope = cp + xp.where(
            l_new > 0.0, (lv + lf * fi) * _saturation_slope(xp, t, qs), 0.0
        )
        t = t - residual / slope
    qs = saturation_mixing_ratio(xp, t, p)
    l_new = xp.maximum(qt - qs, 0.0)
    fi = _ice_fraction(xp, t)
    li = fi * l_new
    ll = l_new - li
    q = qt - l_new
    s = s_mix + lv * (l_new - l_mix) + lf * (li - li_mix)
    return s, q, ll, li, l_new


def _ascend(xp, env, params, *, entrainment, conversion, max_depth, w0sq,
            trigger_k, src, dp_src):
    """Pass 1: the normalised plume's intensive profile and its geometry.

    The updraft's intensive state does not depend on detrainment (detrained
    air leaves with the updraft's own properties), so this pass runs with
    entrainment only and pass 2 (``_mass_flux_profile``) lays the
    detrainment profile onto it in closed form.
    """
    g = GRAVITY_M_S2
    cp = DRY_AIR_CP
    a_coef = params.buoyancy_coefficient
    b_coef = params.drag_coefficient
    c_pgf = params.momentum_pgf_coefficient
    t, q, qc, qi, u, v = (env[name] for name in ("t", "q", "qc", "qi", "u", "v"))
    s, p, dp, dz, z, z_half = (env[name] for name in ("s", "p", "dp", "dz", "z", "z_half"))
    nz, n = t.shape
    dtype = t.dtype
    zero = xp.zeros(n, dtype)
    eta, s_u, q_u, ll_u, li_u, u_u, v_u, w2 = (zero.copy() for _ in range(8))
    alive = xp.ones(n, dtype=bool)
    k_base = xp.full(n, -1, dtype=xp.int32)
    k_d = xp.full(n, -1, dtype=xp.int32)
    k_top = xp.full(n, -1, dtype=xp.int32)
    cape = zero.copy()
    z_base = zero.copy()
    rec = {
        name: xp.zeros((nz, n), dtype)
        for name in ("s_u", "q_u", "ll_c", "li_c", "u_u", "v_u", "pf", "b")
    }
    tiny = dtype.type(1.0e-30)
    top_limit = max(nz - int(params.top_levels_excluded), 2)
    for k in range(nz):
        src_k = src[k]
        pre_alive = alive & ~src_k & (eta > 0.0)
        e_k = xp.where(
            src_k, dp[k] / dp_src,
            xp.where(pre_alive, entrainment * dz[k] * eta, 0.0),
        )
        eta_mix = eta + e_k
        frac = e_k / xp.maximum(eta_mix, tiny)
        s_mix = s_u + frac * (s[k] - s_u)
        q_mix = q_u + frac * (q[k] - q_u)
        ll_mix = ll_u + frac * (qc[k] - ll_u)
        li_mix = li_u + frac * (qi[k] - li_u)
        u_mix = u_u + frac * (u[k] - u_u)
        v_mix = v_u + frac * (v[k] - v_u)
        if k < nz - 1:
            u_mix = xp.where(pre_alive, u_mix + c_pgf * (u[k + 1] - u[k]), u_mix)
            v_mix = xp.where(pre_alive, v_mix + c_pgf * (v[k + 1] - v[k]), v_mix)
        s_c, q_c, ll_c, li_c, l_c = _condense(
            xp, s_mix, q_mix, ll_mix, li_mix, g * z[k], p[k]
        )
        pf = xp.where(src_k, 0.0, 1.0 - xp.exp(-conversion * dz[k]))
        loading = (ll_c + li_c) * (1.0 - pf)
        t_u = (s_c - g * z[k]) / cp
        tv_u = _virtual(t_u, q_c)
        tv_e = _virtual(t[k], q[k])
        b = g * ((tv_u - tv_e) / tv_e - loading + (qc[k] + qi[k]))
        b = b + xp.where(
            l_c > 0.0, 0.0, g * trigger_k / (tv_e * xp.maximum(eta_mix, tiny))
        )
        b = xp.where(src_k, 0.0, b)
        w2_new = xp.where(
            src_k, w0sq,
            w2 * (1.0 - 2.0 * b_coef * entrainment * dz[k]) + 2.0 * a_coef * b * dz[k],
        )
        keep = src_k | pre_alive
        for name, value in (
            ("s_u", s_c), ("q_u", q_c), ("ll_c", ll_c), ("li_c", li_c),
            ("u_u", u_mix), ("v_u", v_mix), ("pf", pf), ("b", b),
        ):
            rec[name][k] = xp.where(keep, value, 0.0)
        cloud = pre_alive & (l_c > 0.0)
        new_base = cloud & (k_base < 0)
        k_base = xp.where(new_base, k, k_base)
        z_base = xp.where(new_base, z_half[k], z_base)
        k_top = xp.where(pre_alive, k, k_top)
        k_d = xp.where(cloud & (b < 0.0) & (k_d < 0) & (k > k_base), k, k_d)
        cape = cape + xp.where(cloud, xp.maximum(b, 0.0) * dz[k], 0.0)
        depth_ok = (k_base < 0) | ((z_half[k + 1] - z_base) <= max_depth)
        alive = alive & (src_k | ((w2_new > 0.0) & (k < top_limit) & depth_ok))
        eta = xp.where(alive, eta_mix, 0.0)
        s_u = xp.where(alive, s_c, s_u)
        q_u = xp.where(alive, q_c, q_u)
        ll_u = xp.where(alive, ll_c * (1.0 - pf), ll_u)
        li_u = xp.where(alive, li_c * (1.0 - pf), li_u)
        u_u = xp.where(alive, u_mix, u_u)
        v_u = xp.where(alive, v_mix, v_u)
        w2 = w2_new
    valid = (k_base >= 0) & (k_top >= k_base)
    k_base = xp.where(valid, k_base, 0)
    k_top = xp.where(valid, k_top, 0)
    k_d = xp.where(k_d < 0, k_top, k_d)
    depth = xp.where(
        valid, _gather(xp, z_half, k_top + 1) - _gather(xp, z_half, k_base), 0.0
    )
    return {
        **rec, "k_base": k_base, "k_d": k_d, "k_top": k_top,
        "valid": valid, "depth": depth, "cape": cape,
    }


def _mass_flux_profile(xp, env, plume, *, entrainment, src, dp_src, k_src_top, select):
    """Pass 2: normalised mass flux eta at half levels, E and D per layer.

    Unit mass flux from the source top to the first level of negative
    in-cloud buoyancy, then linear to zero at the cloud top; columns not in
    ``select`` carry zeros.
    """
    dp, dz, z_half = env["dp"], env["dz"], env["z_half"]
    nz, n = dp.shape
    dtype = dp.dtype
    kk = xp.arange(nz, dtype=xp.int32)[:, None]
    k_d, k_top = plume["k_d"], plume["k_top"]
    z_top = _gather(xp, z_half, k_top + 1)
    z_d = _gather(xp, z_half, k_d)
    lin = xp.clip(
        (z_top[None] - z_half[1:]) / xp.maximum(z_top - z_d, dtype.type(1.0e-6))[None],
        0.0, 1.0,
    )
    eta_src = xp.cumsum(xp.where(src, dp / dp_src[None], 0.0), axis=0)
    eta_out = xp.where(
        src, eta_src,
        xp.where(
            kk < k_d[None], 1.0,
            xp.where(kk <= k_top[None], lin, 0.0),
        ),
    )
    eta_out = xp.where(select[None], eta_out, 0.0)
    eta_half = xp.concatenate([xp.zeros((1, n), dtype), eta_out], axis=0)
    eta_in = eta_half[:-1]
    in_cloud = select[None] & (kk > k_src_top[None]) & (kk <= k_top[None])
    e = xp.where(
        src & select[None], dp / dp_src[None],
        xp.where(in_cloud, entrainment * dz * eta_in, 0.0),
    )
    d = xp.maximum(eta_in + e - eta_out, 0.0)
    return eta_half, e, d, in_cloud


def _unit_tendencies(xp, env, plume, eta_half, e, d, in_cloud, params):
    """Per-unit-base-mass-flux tendencies (E/D form) and precipitation."""
    g = GRAVITY_M_S2
    dp = env["dp"]
    eta_in, eta_out = eta_half[:-1], eta_half[1:]
    eta_mix = eta_in + e
    pf = plume["pf"]
    detrained = {
        "s": plume["s_u"], "q": plume["q_u"],
        "qc": plume["ll_c"] * (1.0 - pf), "qi": plume["li_c"] * (1.0 - pf),
        "u": plume["u_u"], "v": plume["v_u"],
    }
    c_pgf = params.momentum_pgf_coefficient
    tendencies = {}
    for name in ("s", "q", "qc", "qi", "u", "v"):
        psi = env[name]
        above = xp.concatenate([psi[1:], xp.zeros_like(psi[:1])], axis=0)
        budget = -e * psi + d * detrained[name] + eta_out * above - eta_in * psi
        if name in ("u", "v"):
            shear = above - psi
            budget = budget - xp.where(in_cloud, c_pgf * eta_mix * shear, 0.0)
        tendencies[name] = g / dp * budget
    precip_liquid = eta_mix * plume["ll_c"] * pf
    precip_ice = eta_mix * plume["li_c"] * pf
    return tendencies, precip_liquid, precip_ice


def _fall(xp, env, mass_flux, precip_liquid, precip_ice, k_base, dt, params):
    """Precipitation fall with bounded evaporation and melting.

    Returns (ds, dq) additive tendencies (J/kg/s, kg/kg/s) and the surface
    rain (kg/m2/s), sweeping top-down so the flux at each layer is what the
    layers above left.
    """
    g = GRAVITY_M_S2
    cp = DRY_AIR_CP
    lv = LATENT_HEAT_VAPORIZATION
    lf = LATENT_HEAT_FUSION
    ls = lv + lf
    t, q, p, dp = env["t"], env["q"], env["p"], env["dp"]
    nz, n = t.shape
    dtype = t.dtype
    ds = xp.zeros((nz, n), dtype)
    dq = xp.zeros((nz, n), dtype)
    flux_l = xp.zeros(n, dtype)
    flux_i = xp.zeros(n, dtype)
    k_e = params.evaporation_coefficient / 1.0e4
    for k in range(nz - 1, -1, -1):
        flux_l = flux_l + mass_flux * precip_liquid[k]
        flux_i = flux_i + mass_flux * precip_ice[k]
        below_base = k < k_base
        qs = saturation_mixing_ratio(xp, t[k], p[k])
        deficit = xp.maximum(qs - q[k], 0.0)
        rh_gap = deficit / xp.maximum(qs, dtype.type(1.0e-12))
        fraction = xp.where(below_base, xp.minimum(k_e * rh_gap * dp[k], 1.0), 0.0)
        evap_l = fraction * flux_l
        evap_i = fraction * flux_i
        evap = evap_l + evap_i
        cap = params.transfer_bound * deficit * dp[k] / (g * dt)
        scale = xp.where(evap > cap, cap / xp.maximum(evap, dtype.type(1.0e-30)), 1.0)
        evap_l = evap_l * scale
        evap_i = evap_i * scale
        flux_l = flux_l - evap_l
        flux_i = flux_i - evap_i
        melt_cap = xp.maximum(t[k] - FREEZING_K, 0.0) * (
            params.transfer_bound * cp * dp[k] / (g * lf * dt)
        )
        melt = xp.minimum(flux_i, melt_cap)
        if k == 0:
            melt = flux_i
        flux_i = flux_i - melt
        flux_l = flux_l + melt
        dq[k] = (evap_l + evap_i) * g / dp[k]
        ds[k] = -(lv * evap_l + ls * evap_i + lf * melt) * g / dp[k]
    return ds, dq, flux_l


def massflux_columns(xp, columns, *, dt_s, dx_m, forcing_s=None, forcing_q=None,
                     params=None):
    """The scheme on packed columns; every array is (nz, ncol) bottom-up.

    ``columns``: p, dp (Pa), dz (m), t (K), qv, qc, qi (dry mixing ratio),
    u, v (m/s), omega_half ((nz+1, ncol), Pa/s), hfx (W/m2), qfx
    (kg/m2/s), pblh (m).  ``forcing_s``/``forcing_q`` are the (nz, ncol)
    dry-static-energy (J/kg/s) and vapour (kg/kg/s) rates the rest of the
    physics applied this call, read by the quasi-equilibrium closure;
    None falls back to the surface fluxes.  Returns a dict of tendencies
    (dtheta is not formed here: ``dt`` is K/s), ``rain`` (kg/m2 over
    ``dt_s``), ``mass_flux`` (kg/m2/s), branch masks and CAPE.
    """
    params = MassFluxParameters() if params is None else params
    g = GRAVITY_M_S2
    cp = DRY_AIR_CP
    lv = LATENT_HEAT_VAPORIZATION
    dt = float(dt_s)
    p, dp, dz = columns["p"], columns["dp"], columns["dz"]
    t, q, qc, qi = columns["t"], columns["qv"], columns["qc"], columns["qi"]
    u, v = columns["u"], columns["v"]
    dtype = t.dtype
    nz, n = t.shape
    z_half = xp.concatenate(
        [xp.zeros((1, n), dtype), xp.cumsum(dz, axis=0)], axis=0
    )
    z = z_half[:-1] + 0.5 * dz
    s = cp * t + g * z
    env = {
        "t": t, "q": q, "qc": qc, "qi": qi, "u": u, "v": v, "s": s,
        "p": p, "dp": dp, "dz": dz, "z": z, "z_half": z_half,
    }
    sigma, beta, eps_deep = scale_factors(dx_m, params)

    # Source layer and its surface-flux scales.
    z_src_top = xp.clip(
        columns["pblh"], params.source_depth_min_m, params.source_depth_max_m
    )
    kk = xp.arange(nz, dtype=xp.int32)[:, None]
    # Level 0 always feeds the plume; the source never reaches the excluded
    # top levels (a short test column keeps at least its lowest level).
    src = (z <= z_src_top[None]) | (kk == 0)
    src = src & (kk < max(nz - params.top_levels_excluded - 1, 1))
    k_src_top = xp.sum(src, axis=0).astype(xp.int32) - 1
    dp_src = xp.sum(xp.where(src, dp, 0.0), axis=0)
    rho0 = p[0] / (DRY_AIR_GAS_CONSTANT * _virtual(t[0], q[0]))
    hfx, qfx = columns["hfx"], columns["qfx"]
    buoyancy_flux = hfx + 0.61 * cp * t[0] * qfx
    w_star = xp.cbrt(
        xp.maximum(buoyancy_flux, 0.0) * g * z_src_top / (rho0 * cp * t[0])
    )
    theta_star = xp.where(
        w_star > 0.0,
        xp.maximum(buoyancy_flux, 0.0) / (rho0 * cp * xp.maximum(w_star, dtype.type(1.0e-6))),
        0.0,
    )
    trigger_k = xp.minimum(params.trigger_scale * theta_star, params.trigger_max_k)
    w0sq = params.initial_w2_m2_s2 + 4.0 * w_star * w_star

    deep = _ascend(
        xp, env, params, entrainment=eps_deep,
        conversion=params.deep_conversion_per_m, max_depth=float("inf"),
        w0sq=w0sq, trigger_k=trigger_k, src=src, dp_src=dp_src,
    )
    shallow = _ascend(
        xp, env, params, entrainment=params.shallow_entrainment_per_m,
        conversion=0.0, max_depth=params.shallow_maximum_depth_m,
        w0sq=w0sq, trigger_k=trigger_k, src=src, dp_src=dp_src,
    )
    deep_ok = (
        deep["valid"] & (deep["depth"] >= params.deep_minimum_depth_m)
        & (deep["cape"] >= params.cape_minimum_j_kg) & (beta > 0.0)
    )
    shallow_ok = (
        ~deep_ok & shallow["valid"]
        & (shallow["depth"] >= params.shallow_minimum_depth_m)
        & (buoyancy_flux > 0.0)
    )
    plume = {
        name: xp.where(
            deep_ok[None] if deep[name].ndim == 2 else deep_ok,
            deep[name], shallow[name],
        )
        for name in deep
    }
    select = deep_ok | shallow_ok
    # Typed constants: a where() over two Python floats would promote the
    # column to float64 and double the device working set.
    entrainment = xp.where(
        deep_ok, dtype.type(eps_deep), dtype.type(params.shallow_entrainment_per_m)
    )
    eta_half, e, d, in_cloud = _mass_flux_profile(
        xp, env, plume, entrainment=entrainment[None], src=src, dp_src=dp_src,
        k_src_top=k_src_top, select=select,
    )
    unit, precip_l, precip_i = _unit_tendencies(
        xp, env, plume, eta_half, e, d, in_cloud, params
    )

    # Closures.
    k_base, k_top = plume["k_base"], plume["k_top"]
    cloud_layer = select[None] & (kk >= k_base[None]) & (kk <= k_top[None])
    response = xp.sum(
        xp.where(cloud_layer, g * dz / (cp * t) * unit["s"], 0.0), axis=0
    )
    m_cape = plume["cape"] / (
        params.cape_timescale_s * xp.maximum(response, params.cape_response_floor)
    )
    h_src = xp.sum(xp.where(src, (s + lv * q) * dp, 0.0), axis=0) / dp_src
    h_above = _gather(xp, s + lv * q, xp.minimum(k_src_top + 1, nz - 1))
    if forcing_s is None or forcing_q is None:
        f_h = hfx + lv * qfx
    else:
        f_h = xp.sum(xp.where(src, (forcing_s + lv * forcing_q) * dp / g, 0.0), axis=0)
    m_blqe = xp.maximum(f_h, 0.0) / xp.maximum(h_src - h_above, params.blqe_h_gap_floor_j_kg)
    omega = columns["omega_half"]
    q_above = xp.concatenate([q[1:], q[-1:]], axis=0)
    convergence = xp.where(
        kk < k_top[None], -omega[1:] * (q - q_above) / g, 0.0
    ).sum(axis=0)
    p_unit = xp.sum(precip_l + precip_i, axis=0)
    m_mconv = xp.maximum(convergence, 0.0) / xp.maximum(p_unit, params.precipitation_unit_floor)
    m_deep = 0.5 * (m_blqe + m_mconv) * beta
    m_deep = xp.minimum(m_deep, m_cape)
    m_shallow = params.shallow_mass_flux_coefficient * rho0 * w_star
    # Positivity: no layer gives up more than transfer_bound of itself.
    loss = xp.where(
        (kk <= k_top[None]) & select[None],
        (e + eta_half[:-1]) * g * dt / dp, 0.0,
    ).max(axis=0)
    m_pos = params.transfer_bound / xp.maximum(loss, dtype.type(1.0e-30))
    mass_flux = xp.where(deep_ok, m_deep, xp.where(shallow_ok, m_shallow, 0.0))
    mass_flux = xp.maximum(xp.minimum(mass_flux, m_pos), 0.0)
    deep_ok = deep_ok & (mass_flux > 0.0)
    shallow_ok = shallow_ok & (mass_flux > 0.0)

    ds_fall, dq_fall, rain_rate = _fall(
        xp, env, mass_flux, precip_l, precip_i, k_base, dt, params
    )
    out = {name: mass_flux[None] * unit[name] for name in unit}
    out["s"] = out["s"] + ds_fall
    out["q"] = out["q"] + dq_fall
    return {
        "dt": out["s"] / cp, "dqv": out["q"], "dqc": out["qc"], "dqi": out["qi"],
        "du": out["u"], "dv": out["v"],
        "rain": rain_rate * dt, "mass_flux": mass_flux,
        "deep": deep_ok, "shallow": shallow_ok, "cape": plume["cape"],
        "cloud_base": xp.where(select, k_base, -1),
        "cloud_top": xp.where(select, k_top, -1),
        "scale": {"sigma": sigma, "beta": beta, "eps_deep": eps_deep},
    }


def _array_module(value):
    if isinstance(value, np.ndarray):
        return np
    import cupy

    return cupy


class ArwenMassFluxV1:
    """The ``cumulus = "own"`` callable; the GrellFreitas seam's interface.

    ``bind_driver`` receives the runtime's forcing lanes (the PBL and
    radiation rates, WRF-named as woof.globe.core.gf reads them);
    ``__call__(atmosphere=, fields=, state=, cfg=)`` returns a
    :class:`MassFluxResult`.  xp-agnostic: numpy arrays run on the host,
    cupy arrays on the device, column-chunked the way the GF seam is.
    """

    name = SCHEME_NAME

    def __init__(self, *, column_chunk: int | None = None,
                 parameters: MassFluxParameters | None = None):
        chunk = MASSFLUX_COLUMN_CHUNK if column_chunk is None else int(column_chunk)
        if chunk < 1:
            raise ValueError(f"column_chunk must be positive, got {chunk}")
        self.column_chunk = chunk
        self.parameters = MassFluxParameters() if parameters is None else parameters
        self._driver = None

    def bind_driver(self, driver) -> None:
        self._driver = driver

    @property
    def identity(self) -> dict[str, object]:
        return {"scheme": SCHEME_NAME, "parameters": self.parameters.identity}

    def _lane(self, name, shape):
        lane = None if self._driver is None else getattr(self._driver, name, None)
        if lane is None:
            return None
        if tuple(lane.shape) != tuple(shape):
            raise ValueError(f"driver {name} must be {tuple(shape)}, got {tuple(lane.shape)}")
        return lane

    def __call__(self, *, atmosphere, fields, state, cfg):
        temperature = atmosphere["temperature"]
        xp = _array_module(temperature)
        nz, ny, nx = temperature.shape
        ncol = ny * nx
        shape = (nz, ny, nx)
        dtype = temperature.dtype
        omega = getattr(state, "omega_half", None)
        if omega is None:
            w = state.w
            rho = atmosphere["rho"]
            rho_half = xp.concatenate([rho[:1], 0.5 * (rho[:-1] + rho[1:]), rho[-1:]], axis=0)
            omega = -rho_half * GRAVITY_M_S2 * w
        pblh = fields.get("pblh")
        if pblh is None:
            dz_full = atmosphere["dz"]
            height = xp.cumsum(dz_full, axis=0) - 0.5 * dz_full
            kpbl = xp.asarray(fields["kpbl"], dtype=xp.int32)
            pblh = xp.take_along_axis(
                height, xp.maximum(kpbl - 1, 0)[None], axis=0
            )[0]
        exner = atmosphere["exner"]
        rthblten = self._lane("gf_rthblten", shape)
        rqvblten = self._lane("gf_rqvblten", shape)
        rthraten = None
        if self._driver is not None:
            lw = getattr(self._driver, "rthratenlw", None)
            sw = getattr(self._driver, "rthratensw", None)
            if lw is not None and sw is not None:
                rthraten = lw + sw
        forcing_s = None
        forcing_q = None
        if rthblten is not None and rqvblten is not None:
            heating = rthblten if rthraten is None else rthblten + rthraten
            forcing_s = DRY_AIR_CP * exner * heating
            forcing_q = rqvblten
        dt = float(getattr(cfg, "clock_dt", getattr(cfg, "dt", None)))
        dx = float(cfg.dx)

        def level(a):
            return a.reshape(a.shape[0], ncol)

        def plane(a):
            return a.reshape(ncol)

        packed = {
            "p": level(atmosphere["pressure"]), "dp": level(atmosphere["p_interface"][:-1] - atmosphere["p_interface"][1:]),
            "dz": level(atmosphere["dz"]), "t": level(temperature),
            "qv": level(atmosphere["qv"]), "qc": level(atmosphere["qc"]),
            "qi": level(atmosphere["qi"]), "u": level(atmosphere["u"]),
            "v": level(atmosphere["v"]), "omega_half": level(omega),
            "hfx": plane(fields["hfx"]), "qfx": plane(fields["qfx"]),
            "pblh": plane(pblh),
        }
        packed_s = None if forcing_s is None else level(forcing_s)
        packed_q = None if forcing_q is None else level(forcing_q)
        names = ("dt", "dqv", "dqc", "dqi", "du", "dv")
        out = {name: xp.zeros((nz, ncol), dtype) for name in names}
        rain = xp.zeros(ncol, dtype)
        mass_flux = xp.zeros(ncol, dtype)
        deep = xp.zeros(ncol, dtype=bool)
        shallow = xp.zeros(ncol, dtype=bool)
        cape = xp.zeros(ncol, dtype)
        scale = None
        for lo in range(0, ncol, self.column_chunk):
            hi = min(lo + self.column_chunk, ncol)
            chunk = {
                name: xp.ascontiguousarray(value[..., lo:hi])
                for name, value in packed.items()
            }
            result = massflux_columns(
                xp, chunk, dt_s=dt, dx_m=dx,
                forcing_s=None if packed_s is None else xp.ascontiguousarray(packed_s[:, lo:hi]),
                forcing_q=None if packed_q is None else xp.ascontiguousarray(packed_q[:, lo:hi]),
                params=self.parameters,
            )
            for name in names:
                out[name][:, lo:hi] = result[name]
            rain[lo:hi] = result["rain"]
            mass_flux[lo:hi] = result["mass_flux"]
            deep[lo:hi] = result["deep"]
            shallow[lo:hi] = result["shallow"]
            cape[lo:hi] = result["cape"]
            scale = result["scale"]
            del result, chunk
            if getattr(xp, "get_default_memory_pool", None) is not None:
                # The level loop leaves the pool fragmented by hundreds of
                # short-lived temporaries; hand the blocks back between
                # chunks so the pool does not hold several times the live
                # working set across the run.  The blocks go back to the
                # pool the RUN spends, not to the default pool by name: a
                # run under another allocator released nothing here while
                # its temporaries stayed held.
                from woof.globe.device_memory import installed_pool

                release = getattr(installed_pool(xp), "free_all_blocks", None)
                if release is not None:
                    release()

        def grid(a):
            return xp.ascontiguousarray(a.reshape(shape).astype(dtype, copy=False))

        exner_flat = level(exner)
        rthcuten = grid(out["dt"] / exner_flat)
        diagnostics = {
            "deep_column_fraction": float(deep.astype(dtype).mean()),
            "shallow_column_fraction": float(shallow.astype(dtype).mean()),
            "mean_base_mass_flux_kg_m2_s": float(mass_flux.mean()),
            "mean_convective_rain_kg_m2": float(rain.mean()),
            "mean_dilute_cape_j_kg": float(cape.mean()),
            **{f"scale_{k}": float(v) for k, v in (scale or {}).items()},
        }
        return MassFluxResult(
            rthcuten=rthcuten, rqvcuten=grid(out["dqv"]),
            rqccuten=grid(out["dqc"]), rqicuten=grid(out["dqi"]),
            rucuten=grid(out["du"]), rvcuten=grid(out["dv"]),
            rainc=xp.ascontiguousarray(rain.reshape(ny, nx)),
            diagnostics=diagnostics,
        )


__all__ = [
    "SCHEME_NAME", "ArwenMassFluxV1", "MassFluxParameters", "MassFluxResult",
    "massflux_columns", "saturation_mixing_ratio", "scale_factors",
]
