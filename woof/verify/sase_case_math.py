"""Host calculations used by the public SASE verification cases."""

from __future__ import annotations
import math
from woof.core.sase_limits import (E_MIN,
                                    MAX_COLUMN_LEVELS)
import numpy as np
from woof.core.sase_math import (
    BL89_MIX_EXP,
    BLACKADAR_LAMBDA,
    CKS_BLEND_EXP,
    CNU_MAX,
    CP_AIR,
    C_E,
    C_ED,
    C_ES,
    C_K,
    C_KS,
    C_KV,
    C_MOM_BG,
    C_S,
    EP2_RV,
    F_CAP_KNEE,
    F_CAP_WIDTH,
    F_MAX,
    F_MIN,
    G_ACCEL,
    KARMAN,
    LS_COEF,
    N2_SCREEN,
    NU_BLEND_EPS,
    P0_REF,
    PR_LES,
    PR_RANS,
    RD_AIR,
    RIB_CRIT,
    RIB_WSPD2_FLOOR,
    RV_AIR,
    SFC_WSPD_FLOOR,
    SMAG_KM_CAP,
    SVP1,
    SVP2,
    SVP3,
    SVPT0,
    SensorState,
    VENT_DEPTH_CAP,
    VENT_ENT_COEF,
    VENT_MASK,
    VENT_MB_COEF,
    VENT_MIN_RUN_CELLS,
    VENT_QT_STEP_CAP,
    VENT_SAT_ADJUST_ITERS,
    VENT_SIGW_SHARE,
    VENT_THETA_STEP_CAP,
    WSensorState,
    XLV,
    _FILTER_WEIGHTS,
    _blackadar_length,
    _solve_tail,
    _w_bound_tail,
    partition_cap,
    prandtl_blend,
)


def bulk_richardson_zi(u, v, theta, z):
    """Per-column bulk-Richardson BL height (S3-6f, YSU convention).

    Rib(k) = (theta_k - theta_1)*(G_ACCEL*z_k/theta_1)
             / max(u_k^2 + v_k^2, RIB_WSPD2_FLOOR)
    against the level-1 thermal, first upward crossing of RIB_CRIT with
    linear interpolation in Rib between the bracketing layer centers --
    the audited ``npref.np_ysu_column`` ``diagnose`` crossing collapsed
    to the single registered critical value (documented simplification:
    module docstring, S3-6f section; dry theta stands in for theta_v).
    ``z`` are layer-center heights, ``(nz,) + trail`` or full shape (the
    :func:`_column_geometry` convention).  Floors at the first INTERIOR
    layer center z[1] (stable-BL fallback); a column with no crossing
    returns the top layer center (documented permissive fallback);
    nz = 1 returns z[0].  Returns the ``(ny, nx)`` FP64 height field.
    """
    u64, v64, th = (np.asarray(a, dtype=np.float64) for a in (u, v, theta))
    shape = th.shape
    z64 = np.broadcast_to(np.asarray(z, dtype=np.float64), shape)
    nz = shape[0]
    if nz == 1:
        return z64[0].copy()
    thermal = th[0]
    spd2 = np.maximum(u64 ** 2 + v64 ** 2, RIB_WSPD2_FLOOR)
    rib = (th - thermal) * (G_ACCEL * z64 / thermal) / spd2
    rib[0] = 0.0                               # own-level Rib, exact
    crossed = rib >= RIB_CRIT
    crossed[0] = False
    has = crossed.any(axis=0)
    kc = np.argmax(crossed, axis=0)            # first True (>= 1)
    kc = np.where(has, kc, nz - 1)
    kx = kc[None]
    rib_up = np.take_along_axis(rib, kx, axis=0)[0]
    rib_dn = np.take_along_axis(rib, kx - 1, axis=0)[0]
    z_up = np.take_along_axis(z64, kx, axis=0)[0]
    z_dn = np.take_along_axis(z64, kx - 1, axis=0)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (RIB_CRIT - rib_dn) / (rib_up - rib_dn)
    zi = np.where(has, z_dn + frac * (z_up - z_dn), z64[-1])
    return np.maximum(zi, z64[1])


def _ddz(f, dz, periodic: bool = False, dz_col=None):
    if dz_col is not None:
        return _ddz_var(f, dz_col, periodic)
    if periodic:
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dz)
    if f.shape[0] == 1:
        # Degenerate single-level column (S3-6b nz=1 decay fixture): a
        # one-cell clamped column has no vertical variation; derivative
        # is identically zero.  Pure extension -- nz >= 2 is untouched
        # (the nz = 1 clamped path previously raised IndexError).
        return np.zeros_like(f)
    out = np.empty_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dz)
    out[0] = (f[1] - f[0]) / dz
    out[-1] = (f[-1] - f[-2]) / dz
    return out


def dissipation_length(e, delta, n2=None, lb=None, f=1.0):
    """Stability-limited dissipation length with the regime blend
    (GEOMETRIC since S3-9; module docstring, S3-9 section).

        l = min(delta**f * lb**(1-f), LS_COEF*sqrt(e)/N)    [N^2 > 0]

    ``lb`` is the RANS-limb dissipation composition (the live path
    passes min(l_B, l_eps_BL89) from :func:`bl89_rans_lengths`) and
    ``f`` the dynamic partition weight, blending the two regimes the
    length must serve IN LOG SPACE: at f = 1 (resolved, LES-like
    fields) the blend is exactly ``delta`` -- delta**1.0 * lb**0.0 =
    delta*1.0 is FP-exact -- so the v0 form min(delta, l_s) is
    recovered bitwise; at f = 0 (degenerate resolved field, the
    coarse-mesh RANS regime, where ``dynamic_solve`` falls back to
    (0, 0)) the blend is exactly ``lb`` (delta**0.0 * lb**1.0 =
    1.0*lb, FP-exact) so l = min(l_B, l_s) = l_v, the vertical
    mixing-length scale.  Both endpoints are bitwise IDENTICAL to the
    retired linear form f*delta + (1-f)*lb; between them the
    geometric form stays within (delta/lb)**f of lb where the linear
    form floored l_d at f*delta -- the F-Y1 defect (S3-9 section).
    EXPONENTIATION DOMAIN: delta and lb must be strictly positive --
    every live caller guarantees it (delta is a filter width; lb from
    bl89_rans_lengths is min of the positive Blackadar length at
    layer centers z >= thick_0/2 > 0 and the E_MIN-floored BL89
    lengths), documented rather than asserted per module convention.
    The stability limit enters ONCE through the outer min: l_v's own
    stability limit is the same l_s, so blending the NEUTRAL-limb lb
    avoids counting it twice.  ``lb=None`` (default) preserves the v0
    behavior unchanged bitwise; unstable/neutral cells (n2 <= 0 or
    ``n2 is None``) never engage the l_s branch.
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    if lb is None:
        l = np.full_like(e64, float(delta))
    else:
        # S3-9 GEOMETRIC blend (was the linear f*delta + (1-f)*lb):
        # log-space interpolation keeps l_d within (delta/lb)**f of
        # the RANS bound.  Endpoints FP-exact in IEEE-754: x**1.0 == x,
        # x**0.0 == 1.0, 1.0*x == x, so f = 0 gives lb bitwise and
        # f = 1 gives delta bitwise -- identical to the linear form's
        # endpoint arithmetic (module docstring, S3-9 section).
        blend = (float(delta) ** f
                 * np.asarray(lb, dtype=np.float64) ** (1.0 - f))
        l = blend * np.ones_like(e64)
    if n2 is not None:
        n2 = np.asarray(n2, dtype=np.float64)
        stable = n2 > 0.0
        ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
        l = np.where(stable, np.minimum(l, ls), l)
    return l


def _z_centers(dz_col, shape):
    """Layer-center heights implied by layer thicknesses ``dz_col``.

    Convention (stage-3 model-column mode): ``dz_col`` holds layer
    thicknesses, shape ``(nz,)`` (a shared column) or ``(nz, ny, nx)``
    (per-column); interface positions are the cumulative thickness sums
    up from z = 0, and level ``k``'s value sits at its layer center,
    ``z_k = sum(dz_col[:k]) + 0.5*dz_col[k]``.
    """
    t = np.asarray(dz_col, dtype=np.float64)
    if t.ndim == 1:
        if t.shape[0] != shape[0]:
            raise ValueError(
                f"dz_col has {t.shape[0]} levels, field has {shape[0]}")
        t = t[:, None, None]
    elif t.shape != tuple(shape):
        raise ValueError(
            f"dz_col shape {t.shape} must be (nz,) or match field "
            f"shape {tuple(shape)}")
    return np.cumsum(t, axis=0) - 0.5 * t


def brunt_vaisala_n2(theta, dz, dz_col=None):
    """Moist-static N^2 = (g/theta) * d(theta)/dz, clamped-z stencil.

    Stage-3 Task 6 authority extension (new function; no existing default
    changes): the stability-length input the model driver computes from
    its own theta profile, on the SAME clamped (variable-``dz_col``)
    vertical stencil every SASE operator uses -- the discrete form is
    therefore pinned here, not re-derived at the call site.
    """
    theta64 = np.asarray(theta, dtype=np.float64)
    return (G_ACCEL / theta64) * _ddz(theta64, dz, dz_col=dz_col)


def _ddz_var(f, dz_col, periodic: bool = False):
    """Clamped-z vertical derivative on variable layer thicknesses.

    Interior rows use the three-point Lagrange stencil on the nonuniform
    center-to-center spacings h_m = z[k]-z[k-1], h_p = z[k+1]-z[k]:

        f'(z_k) = -(h_p/(h_m*(h_p+h_m))) * f[k-1]
                  + ((h_p-h_m)/(h_p*h_m)) * f[k]
                  + (h_m/(h_p*(h_p+h_m))) * f[k+1]

    which is exact for quadratics in z (the two-sided form
    (f[k+1]-f[k-1])/(z[k+1]-z[k-1]) is only linear-exact on a stretched
    grid) and reduces to the centered (f[k+1]-f[k-1])/(2h) when
    h_p == h_m == h.  Edge rows are one-sided two-point over the edge
    center spacing (linear-exact), matching the uniform clamped path.

    ``periodic`` is accepted only to reject it: the roll-based periodic
    vertical is the conservation theorem's domain (the uniform periodic
    box) and must not be silently combined with variable spacing.
    """
    if periodic:
        raise ValueError(
            "dz_col is incompatible with a periodic vertical: the "
            "conservation ledger's periodic path is defined on the "
            "uniform box only")
    f = np.asarray(f, dtype=np.float64)
    if f.shape[0] == 1:
        # Degenerate single-level column: no vertical variation (see the
        # matching nz = 1 branch in ``_ddz``).
        return np.zeros_like(f)
    z = _z_centers(dz_col, f.shape)
    h_m = z[1:-1] - z[:-2]
    h_p = z[2:] - z[1:-1]
    out = np.empty_like(f)
    out[1:-1] = (-(h_p / (h_m * (h_p + h_m))) * f[:-2]
                 + ((h_p - h_m) / (h_p * h_m)) * f[1:-1]
                 + (h_m / (h_p * (h_p + h_m))) * f[2:])
    out[0] = (f[1] - f[0]) / (z[1] - z[0])
    out[-1] = (f[-1] - f[-2]) / (z[-1] - z[-2])
    return out


def vertical_mixing_length(z, e, n2=None, z0=0.0):
    """S3-6b vertical mixing length l_v = min(l_B, LS_COEF*sqrt(e)/N).

    The N^2 > 0 branch applies the Deardorff stability limit (the same
    l_s the dissipation length uses); neutral/unstable cells keep the
    Blackadar length.  Broadcasts ``z`` (per-level or full-shape)
    against ``e``; returns the full-shape FP64 length.

    S3-6h role: this frozen v0 form is now the LES LIMB (and one term
    of the RANS composition) of the split step's blended vertical
    length -- the live channel is l_v = f*THIS + (1-f)*
    min(THIS, l_mix_BL89) (:func:`bl89_rans_lengths`; module
    docstring, S3-6h section).  The function itself is unchanged
    bitwise.
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    l = _blackadar_length(z, z0) * np.ones_like(e64)
    if n2 is not None:
        n2 = np.asarray(n2, dtype=np.float64)
        stable = n2 > 0.0
        ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
        l = np.where(stable, np.minimum(l, ls), l)
    return l


def _bl89_first_crossing(c1, c2, rem, h):
    """Smallest s in [0, h] with c2*s^2 + c1*s == rem, else NaN.

    One segment of the BL89 displacement integral (all FP64, vectorized
    over columns).  DERIVATION: within a segment of extent h whose
    piecewise-linear theta_v runs from theta_a (at the parcel-side end)
    to theta_b, the integrand against a parcel of potential temperature
    theta_p is beta*(theta(s) - theta_p) = (c1 + 2*c2*s) with

        c1 = beta*(theta_a - theta_p),
        c2 = beta*(theta_b - theta_a)/(2*h),

    so the spent energy over a partial traversal s is EXACTLY

        I(s) = integral_0^s (c1 + 2*c2*sigma) d sigma = c1*s + c2*s^2

    -- quadratic in s, which is why the discrete integral is exact for
    every piecewise-linear theta_v profile (the quadratic-exactness
    class pinned by the analytic fixtures).  The parcel stops at the
    FIRST s where I(s) equals the remaining budget rem (>= 0): the
    smallest root of c2*s^2 + c1*s - rem = 0 inside [0, h].  Root
    finding is the numerically stable q-form (q = -(c1 +
    sign(c1)*sqrt(disc))/2, roots q/c2 and -rem/q), which degrades
    gracefully to the linear root rem/c1 as c2 -> 0; the exact c2 == 0
    branch is handled explicitly.  No root in [0, h] (including a
    negative discriminant) means the parcel traverses the whole
    segment: the caller subtracts I(h) from rem and continues.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2, rem64, h64 = (np.broadcast_to(np.asarray(a, dtype=np.float64),
                                      c1.shape)
                      for a in (c2, rem, h))
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = c1 * c1 + 4.0 * c2 * rem64
        sq = np.sqrt(np.maximum(disc, 0.0))
        q = -0.5 * (c1 + np.where(c1 >= 0.0, sq, -sq))
        root_a = np.where(c2 != 0.0, q / np.where(c2 != 0.0, c2, 1.0),
                          np.nan)
        root_b = np.where(q != 0.0, -rem64 / np.where(q != 0.0, q, 1.0),
                          # q == 0 <=> c1 == 0 and disc == 0 <=> the
                          # double root sits at s = 0 (rem == 0 with a
                          # locally flat or stabilizing integrand).
                          0.0)
        lin = np.where(c1 > 0.0, rem64 / np.where(c1 > 0.0, c1, 1.0),
                       np.nan)
    quad = c2 != 0.0
    cands = (np.where(quad, root_a, np.nan),
             np.where(quad, root_b, lin))
    tol = 1.0e-12 * h64
    best = np.full(c1.shape, np.nan)
    for cand in cands:
        ok = (np.isfinite(cand) & (disc >= 0.0)
              & (cand >= -tol) & (cand <= h64 + tol))
        cand = np.clip(cand, 0.0, h64)
        best = np.where(ok & (~np.isfinite(best) | (cand < best)),
                        cand, best)
    return best


def bl89_displacement_lengths(theta, e, z, thick):
    """S3-6h BL89 parcel displacement lengths ``(l_up, l_down)``.

    Per column and level k (Bougeault-Lacarrere 1989, MWR 117,
    1872-1890): ``l_up(k)`` is the FIRST upward displacement l at which

        integral_{z_k}^{z_k + l} beta * (theta_v(z') - theta_v(z_k)) dz'
            == e(k),        beta = G_ACCEL/theta_v(z_k)

    (the parcel's kinetic-energy budget exhausted against the
    integrated buoyancy resistance; the accumulate-until-exceed
    Meso-NH convention), bounded by the column top interface
    (l_up <= htop - z_k); ``l_down(k)`` is the mirror image downward,
    integrand beta*(theta_v(z_k) - theta_v(z')), bounded by the surface
    (l_down <= z_k).  Unstable stretches contribute NEGATIVELY (the
    parcel re-accelerates), which is the non-locality this amendment
    exists for: a parcel below an inversion is stopped by the
    inversion's INTEGRATED buoyancy regardless of the local N^2 --
    information the local l_s = LS_COEF*sqrt(e)/N cannot carry.

    DISCRETE INTEGRAL (exact for piecewise-linear theta_v): theta_v is
    taken linear between adjacent layer centers, held constant from the
    lowest center to the surface (z' in [0, z_0]) and from the highest
    center to the top interface -- the sub-half-layer profile is
    unresolved, so constant extension is the accurate discrete choice.
    Each full segment then contributes its exact quadrature
    (c1 + c2*h)*h and the fractional last segment solves the quadratic
    I(s) = rem exactly (:func:`_bl89_first_crossing` has the
    derivation), so the only approximation in the whole construction
    is the piecewise-linear profile representation itself.
    theta_v = dry theta (L1 dry core; registered v0 simplification --
    moist theta_v is SASE-M scope).  ``e`` is floored at E_MIN (module
    idiom), which keeps both lengths strictly positive.  ``z``/``thick``
    follow the :func:`_column_geometry` conventions ((nz,) + trail or
    full shape; scalar thick for uniform columns).  Returns two FP64
    full-shape arrays.
    """
    th = np.asarray(theta, dtype=np.float64)
    shape = th.shape
    nz = shape[0]
    e64 = np.broadcast_to(
        np.maximum(np.asarray(e, dtype=np.float64), E_MIN), shape)
    z64 = np.broadcast_to(np.asarray(z, dtype=np.float64), shape)
    t64 = np.broadcast_to(np.asarray(thick, dtype=np.float64)
                          * np.ones((1,) * th.ndim), shape)
    htop = z64[-1] + 0.5 * t64[-1]
    trail = shape[1:]
    l_up = np.empty(shape)
    l_down = np.empty(shape)
    for k in range(nz):
        beta = G_ACCEL / th[k]
        for down in (False, True):
            rem = e64[k].copy()
            acc = np.zeros(trail)
            active = np.ones(trail, dtype=bool)
            out_k = np.zeros(trail)
            prev_z, prev_t = z64[k], th[k]
            nodes = range(k - 1, -2, -1) if down else range(k + 1, nz + 1)
            for j in nodes:
                if down:
                    node_z = z64[j] if j >= 0 else np.zeros(trail)
                    node_t = th[j] if j >= 0 else th[0]
                    h = prev_z - node_z
                    c1 = beta * (th[k] - prev_t)
                    c2d = beta * (prev_t - node_t)
                else:
                    node_z = z64[j] if j < nz else htop
                    node_t = th[j] if j < nz else th[nz - 1]
                    h = node_z - prev_z
                    c1 = beta * (prev_t - th[k])
                    c2d = beta * (node_t - prev_t)
                pos = h > 0.0
                c2 = np.where(pos, c2d, 0.0) / np.where(pos, 2.0 * h, 1.0)
                s = _bl89_first_crossing(c1, c2, rem, h)
                hit = active & np.isfinite(s)
                out_k = np.where(hit, acc + s, out_k)
                seg = np.where(pos, (c1 + c2 * h) * h, 0.0)
                rem = np.where(active & ~hit,
                               np.maximum(rem - seg, 0.0), rem)
                acc = acc + np.where(active, h, 0.0)
                active = active & ~hit
                prev_z, prev_t = node_z, node_t
                if not active.any():
                    break
            bound = z64[k] if down else htop - z64[k]
            out_k = np.where(active, bound, out_k)
            if down:
                l_down[k] = out_k
            else:
                l_up[k] = out_k
    return l_up, l_down


def bl89_combine(l_up, l_down):
    """S3-6h combination forms ``(l_mix, l_eps)`` from the displacement
    pair (constants block at BL89_MIX_EXP has the choice rationale and
    citations):

        l_mix = (0.5*(l_up^-p + l_down^-p))^(-1/p),   p = BL89_MIX_EXP
        l_eps = min(l_up, l_down)

    Both inputs are strictly positive (bl89_displacement_lengths
    guarantees it), and min <= the negative power mean <= max, so
    l_eps <= l_mix always: dissipation is never weaker than mixing.
    """
    up64 = np.asarray(l_up, dtype=np.float64)
    dn64 = np.asarray(l_down, dtype=np.float64)
    p = BL89_MIX_EXP
    l_mix = (0.5 * (up64 ** (-p) + dn64 ** (-p))) ** (-1.0 / p)
    return l_mix, np.minimum(up64, dn64)


def bl89_moist_excursion_lengths(n2_eff, e, z, thick):
    """SASE-M1b moist parcel-excursion lengths ``(l_up, l_down)``
    (module docstring, SASE-M1b section; SASE-M spec section 3b).

    The BL89 displacement construction evaluated against the M1 MOIST
    buoyancy: per column and level k, ``l_up(k)`` is the FIRST upward
    displacement l at which

        integral_{z_k}^{z_k + l} R(z') dz' == e(k),
        R(z') = integral_{z_k}^{z'} N^2_eff(s) ds

    with ``n2_eff`` the :func:`moist_n2` effective-stability FIELD
    (DK82 saturated N^2_m in saturated cells, condensate loading
    included; the dry field elsewhere); ``l_down(k)`` is the mirror
    image downward (R(z') = integral_{z'}^{z_k} N^2_eff ds), bounded
    by the column top interface and the surface exactly as the dry
    lengths are.  DERIVATION OF THE FAMILY IDENTITY: in the dry
    machinery the integrand is beta*(theta_v(z') - theta_v(z_k)) with
    beta = g/theta_v(z_k), which IS R(z') for the parcel-frozen-beta
    stratification measure N^2_beta = beta*d(theta_v)/dz -- so this
    function is the SAME accumulate-until-exceed excursion integral
    with the M1 buoyancy substituted as the stratification measure,
    exactly the spec's "excursion integrand is the M1 buoyancy".
    Moist-UNSTABLE stretches (N^2_m < 0) contribute negatively (the
    saturated parcel re-accelerates), so a moist-unstable deck spends
    nothing until the moist-stable lid: l_up = distance-to-lid plus a
    finite penetration -- the geometry the dry-theta lengths cannot
    see (they read the deck's DRY stability and stop parcels inside
    it, hiding the lid, while the free Blackadar l_B fallback caps the
    master length at the wrong scale entirely; the G-M3 defect).

    DISCRETE FORM (exact for piecewise-constant N^2_eff): the M1
    buoyancy exists as a FIELD at layer centers (s^-2), not as a
    potential like theta, so within the segment between adjacent
    centers the stratification measure is held CONSTANT at the
    arithmetic face mean 0.5*(n2_eff[j] + n2_eff[j+1]) -- the module's
    face convention (:func:`_face_average`) -- making R(z') piecewise
    LINEAR and continuous (R accumulated across segments as
    c1_next = c1 + N^2_seg*h) and the outer integral exactly quadratic
    per segment: :func:`_bl89_first_crossing` solves the fractional
    last segment with c1 = R at the segment start and c2 = N^2_seg/2,
    identically the dry quadrature class.  End segments (surface to
    the lowest center, highest center to the top interface) carry
    slope 0 -- the constant-extension convention of the dry lengths.
    LEVEL BRACKETING (pinned by the deck-under-lid fixture): n2_eff
    rides the centered clamped stencil, so a lid at interface z_L is
    felt by the segment BELOW the lid-adjacent center -- excursions
    terminate within one cell of the geometric distance-to-lid,
    (z_L - z_k) - dz <= l_up <= (z_invtop - z_k) on the fixture.
    ``e`` is floored at E_MIN (module idiom); with rem > 0 and the
    strictly positive geometric bounds (z_k >= thick_0/2), both
    lengths are strictly positive.  Zero new constants: E_MIN, the
    face mean, and the crossing solver are all pre-existing machinery.
    Shapes follow :func:`bl89_displacement_lengths` (``n2_eff`` full
    (nz, ...) layout; ``e``/``z``/``thick`` broadcast against it).
    """
    n2 = np.asarray(n2_eff, dtype=np.float64)
    shape = n2.shape
    nz = shape[0]
    e64 = np.broadcast_to(
        np.maximum(np.asarray(e, dtype=np.float64), E_MIN), shape)
    z64 = np.broadcast_to(np.asarray(z, dtype=np.float64), shape)
    t64 = np.broadcast_to(np.asarray(thick, dtype=np.float64)
                          * np.ones((1,) * n2.ndim), shape)
    htop = z64[-1] + 0.5 * t64[-1]
    trail = shape[1:]
    l_up = np.empty(shape)
    l_down = np.empty(shape)
    for k in range(nz):
        for down in (False, True):
            rem = e64[k].copy()
            acc = np.zeros(trail)
            run = np.zeros(trail)              # R at the segment start
            active = np.ones(trail, dtype=bool)
            out_k = np.zeros(trail)
            prev_z = z64[k]
            nodes = range(k - 1, -2, -1) if down else range(k + 1, nz + 1)
            for j in nodes:
                if down:
                    node_z = z64[j] if j >= 0 else np.zeros(trail)
                    h = prev_z - node_z
                    slope = (0.5 * (n2[j] + n2[j + 1]) if j >= 0
                             else np.zeros(trail))
                else:
                    node_z = z64[j] if j < nz else htop
                    h = node_z - prev_z
                    slope = (0.5 * (n2[j - 1] + n2[j]) if j < nz
                             else np.zeros(trail))
                c1 = run
                pos = h > 0.0
                c2 = np.where(pos, 0.5 * slope, 0.0)
                s = _bl89_first_crossing(c1, c2, rem, h)
                hit = active & np.isfinite(s)
                out_k = np.where(hit, acc + s, out_k)
                seg = np.where(pos, (c1 + c2 * h) * h, 0.0)
                rem = np.where(active & ~hit,
                               np.maximum(rem - seg, 0.0), rem)
                acc = acc + np.where(active, h, 0.0)
                run = run + np.where(pos, slope * h, 0.0)
                active = active & ~hit
                prev_z = node_z
                if not active.any():
                    break
            bound = z64[k] if down else htop - z64[k]
            out_k = np.where(active, bound, out_k)
            if down:
                l_down[k] = out_k
            else:
                l_up[k] = out_k
    return l_up, l_down


def bl89_rans_lengths(theta, e, z, thick, n2=None, z0=0.0, n2_dry=None):
    """S3-6h RANS-limb vertical lengths ``(l_mix_rans, l_eps_rans)``.

    The composed lengths the split step's RANS limb rides (module
    docstring, S3-6h section):

        l_mix_rans = min(l_B, l_s, l_mix_BL89)   [the mixing length]
        l_eps_rans = min(l_B,      l_eps_BL89)   [the dissipation limb]

    SASE-M1b seam (module docstring, SASE-M1b section; spec section
    3b): with ``n2_dry`` given (the DRY N^2 field, so ``n2`` must be
    the M1 effective field), the M1-substituted cells (n2 != n2_dry --
    the seam's own mask, exactly the cells where the moist closure
    claimed authority) have BOTH composed lengths ADDITIONALLY bounded
    by the moist master-length limb

        l_m = min(l_up_m, l_down_m)     [MOIST_MASTER_LENGTH =
                                         "bl89-n2eff-excursion-min-v1"]

    from :func:`bl89_moist_excursion_lengths` -- the min member is the
    family's own dissipation convention (BL89's l_eps) and the unique
    conservative choice for a master-length CEILING: the amendment
    exists because K at the free fallback length OVER-mixes (the G-M3
    deck-clearing defect), so the bound on the K-side l_mix_rans must
    never be slacker than the tightest excursion direction (a lid
    60 m above must bound the master length at the 60 m class no
    matter how far the parcel could fall).  Unsubstituted cells keep
    their bits VERBATIM (np.where copies the FALSE branch), an empty
    mask skips the machinery entirely (the unsaturated-gate contract,
    poison-pinned by the control/lake fixtures), and ``n2_dry=None``
    (every pre-M1b caller) is the S3-6h composition unchanged bitwise.
    The S3-6h monotonicity statement survives verbatim: no RANS-limb
    length is ever LONGER than its pre-amendment value.

    * the l_B min is the registered kappa-z-class floor/match
      (BL89_KZ_MATCH): near the surface l_B -> kappa*(z + z0) enforces
      the neutral log-layer constant exactly (in a neutral column the
      BL89 lengths are pure geometry, l_down = z >= l_B and
      l_up = htop - z, so the min leaves l_B bitwise everywhere except
      within ~l_B of the model top), and aloft it retains the audited
      Blackadar asymptote;
    * the l_s min on the MIXING length is the registered retention
      decision (BL89_LS_DECISION, adjudicated at the uniform-
      stratification fixture) -- applied here through the frozen
      :func:`vertical_mixing_length` so the composition is literally
      "the v0 length, further bounded by BL89";
    * the dissipation limb takes NO l_s here because the l_d blend's
      OUTER min applies l_s once (:func:`dissipation_length`, the
      S3-6b double-counting rule).

    The whole amendment is therefore monotone: no RANS-limb length is
    ever LONGER than its pre-S3-6h value, so every suppression the v0
    formulation achieved is preserved and BL89 can only add more.
    Monkeypatching THIS function to return
    ``(vertical_mixing_length(z, e, n2, z0), broadcast l_B)``
    reproduces the pre-S3-6h formulation bitwise at f = 0 (the RED-leg
    idiom of the jet-decoupling fixture, mirroring S3-6g's
    PR_RANS-pinning argument).
    """
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    lb = _blackadar_length(z, z0) * np.ones_like(e64)
    l_les = vertical_mixing_length(z, e, n2, z0)      # min(l_B, l_s)
    l_up, l_dn = bl89_displacement_lengths(theta, e, z, thick)
    l_mix_bl, l_eps_bl = bl89_combine(l_up, l_dn)
    l_mix_rans = np.minimum(l_les, l_mix_bl)
    l_eps_rans = np.minimum(lb, l_eps_bl)
    if n2_dry is not None and n2 is not None:
        # SASE-M1b moist master-length limb (docstring above; module
        # docstring, SASE-M1b section): min-bound BOTH compositions by
        # the moist excursion length in the M1-substituted cells only.
        subst = (np.asarray(n2, dtype=np.float64)
                 != np.asarray(n2_dry, dtype=np.float64))
        if bool(subst.any()):
            l_up_m, l_dn_m = bl89_moist_excursion_lengths(n2, e, z,
                                                          thick)
            l_m = np.minimum(l_up_m, l_dn_m)
            l_mix_rans = np.where(subst, np.minimum(l_mix_rans, l_m),
                                  l_mix_rans)
            l_eps_rans = np.where(subst, np.minimum(l_eps_rans, l_m),
                                  l_eps_rans)
    return l_mix_rans, l_eps_rans


def stable_limit_coefficient(l_rans, e, n2=None):
    """S3-6i RANS-limb diffusivity coefficient C_r (module docstring,
    S3-6i section): the min-aware blend that decouples the stable-limit
    diffusivity from the neutral-wall calibration C_KV.

        rho = min(l_rans/l_s, 1),  l_s = LS_COEF*sqrt(e)/N  [N^2 > 0]
        C_r = C_KV + (C_KS/LS_COEF - C_KV)*rho**CKS_BLEND_EXP

    with rho = 0 where N^2 <= 0 or ``n2`` is None, so K_v =
    C_r*l_rans*sqrt(e) recovers

    * the wall/neutral calibration C_KV FP-EXACTLY where the
      stability length is slack (neutral/unstable cells: C_KV +
      dC*0.0 == C_KV bitwise -- the log-layer hard constraint), and
    * the decoupled stable limit C_KS*e/N exactly where l_s binds
      (there l_rans == l_s bitwise, since l_s is a term of the
      l_mix_rans min, so rho == 1.0).

    Between the regimes the coefficient interpolates smoothly on how
    close the operating length sits to the local stability limit --
    a bound-by geometry/inversion parcel (l_B or l_BL89 binding well
    under l_s) keeps most of the wall calibration; a parcel at the
    stability limit mixes at the stable calibration.  Continuity
    across the neutral<->stable transition (C^0 in the input n2 --
    linear deficit onset -- and C^1 in N = sqrt(n2); S3-9c narrowed
    claim) and the co-location of the remaining (pre-existing)
    min-switch kinks are derived in the
    module docstring; C_r is bounded in [C_KS/LS_COEF, C_KV] (the
    rho clip makes that hold for ANY l_rans, though every caller
    passes l_rans <= l_s by construction).  ``e`` is floored at E_MIN
    (module idiom); broadcasts like :func:`vertical_mixing_length`.
    """
    l64 = np.asarray(l_rans, dtype=np.float64)
    if n2 is None:
        return np.full(l64.shape, C_KV)
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    n2 = np.asarray(n2, dtype=np.float64)
    stable = n2 > 0.0
    ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
    rho = np.where(stable, np.minimum(l64 / ls, 1.0), 0.0)
    return C_KV + (C_KS / LS_COEF - C_KV) * rho ** CKS_BLEND_EXP


def stable_dissipation_coefficient(l_d, e, n2=None, f=1.0):
    """S3-6k stable-limb DISSIPATION coefficient C_eps (module docstring,
    S3-6k section): the structural mirror of
    :func:`stable_limit_coefficient` on the dissipation half of the pair.

        l_s    = LS_COEF*sqrt(e)/N                        [N^2 > 0]
        rho    = min(l_d/l_s, 1)
        w      = rho**CKS_BLEND_EXP
        C_rans = (1 - w)*C_E + w*C_ES
        C_eps  = f*C_E + (1 - f)*C_rans                   [N^2 > 0]
        C_eps  = C_E                                      [otherwise]

    so eps = C_eps*e^{3/2}/l_d recovers

    * the neutral-wall calibration C_E BITWISE in neutral/unstable cells
      and where ``n2`` is None -- by SELECTION, the same
      ``if (has_ces && ls_v > 0.0f)`` gate the device carries, not by
      arithmetic cancellation: at N^2 <= 0 the blend would reduce to
      f*C_E + (1-f)*C_E, which is NOT C_E bitwise at general f (3661 of
      a uniform 10001-point f grid land 1 ulp off, measured this
      session; the recorded production f = 4.1188928660938e-05 is one
      of them).  So the log-layer engine, the C_KV = C_E**(1/3)
      identity and the M1-substituted cells are untouched, and the FP64
      authority and its FP32 mirror agree there bitwise;
    * the Deardorff stable asymptote C_ES BITWISE where l_s binds l_d --
      :func:`dissipation_length` ends with the same ``np.minimum(l, ls)``
      this function divides by, so l_d == l_s bitwise there and rho is
      exactly 1.0.  Measured this session with
      ``np.float64(x).tobytes().hex()``: the two-product form returns
      0.19 (``52b81e85eb51c83f``) while the affine
      C_E + (C_ES - C_E)*w returns 0.19000000000000006
      (``54b81e85eb51c83f``).  That is why the form is two-product;
    * C_E BITWISE at f = 1 for ANY stratification (1.0*C_E +
      0.0*C_rans), the S3-6i LES-limb property carried onto the
      dissipation half, which is what keeps every LES fixture inert with
      the switch ON as well as OFF.

    ``l_d`` is the DISSIPATION length -- the length the coefficient
    multiplies -- never ``l_mix_rans``: a coefficient must pair with its
    own length.  ``e`` must be the SAME field :func:`dissipation_length`
    consumed (it is floored at E_MIN here exactly as there, so a caller
    that passes the post-source e* can drive rho above 1 and would be
    silently clipped rather than caught).  Broadcasts like
    :func:`stable_limit_coefficient`.
    """
    l64 = np.asarray(l_d, dtype=np.float64)
    if n2 is None:
        return np.full(l64.shape, C_E)
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    n2 = np.asarray(n2, dtype=np.float64)
    stable = n2 > 0.0
    ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
    rho = np.where(stable, np.minimum(l64 / ls, 1.0), 0.0)
    w = rho ** CKS_BLEND_EXP
    c_rans = (1.0 - w) * C_E + w * C_ES
    # The neutral/unstable return is the SELECTED literal C_E, never the
    # blend evaluated at a C_rans that happens to equal it: f*C_E +
    # (1-f)*C_E is NOT C_E bitwise at general f (measured this session
    # over a uniform 10001-point f grid, 3661 values return
    # 0.9299999999999999 or 0.9300000000000002 instead of 0.93
    # c3f5285c8fc2ed3f -- including the recorded production
    # f = 4.1188928660938e-05).  The device gate is
    # ``if (has_ces && ls_v > 0.0f)`` (sase.cu, S3-6k block), which
    # leaves the multiplicand the literal (double)c_e wherever no
    # stability length exists; this np.where IS that gate, so the
    # authority and its mirror agree bitwise on the cells the amendment
    # is documented not to touch.
    return np.where(stable, f * C_E + (1.0 - f) * c_rans, C_E)


def neutral_dissipation_length(z, delta, f=1.0, z0=0.0):
    """S3-12 STATE-INDEPENDENT reference length of the additive channel.

        l_ref = delta**f * l_B(z + z0)**(1 - f)

    the same geometric regime blend :func:`dissipation_length` runs,
    evaluated on the NEUTRAL member of each limb (LD_ADDITIVE_CHANNEL):
    ``delta`` at f = 1 -- Deardorff's own filter width, FP-exact -- and
    the Blackadar length at f = 0.  Carries NO ``e`` and no ``n2``,
    which is the property the whole amendment rests on: divided into
    ``C_ED*e**1.5`` it yields a dissipation channel that is genuinely
    e^{3/2}, where the e-linear channel's own RANS input
    (min(l_B, l_eps_BL89)) is not, because the BL89 displacement lengths
    solve a parcel-energy integral and scale as sqrt(e).

    Endpoints are FP-exact by the same IEEE-754 identities
    :func:`dissipation_length` documents (x**1.0 == x, x**0.0 == 1.0).
    Broadcasts over ``z`` like :func:`_blackadar_length`.
    """
    lb = _blackadar_length(np.asarray(z, dtype=np.float64), z0)
    return float(delta) ** f * lb ** (1.0 - f)


def additive_dissipation_coefficient(l_d, l_ref, e, n2=None, f=1.0,
                                     c_base=C_E):
    """S3-12 effective dissipation coefficient with Deardorff's SECOND
    channel added (module docstring, S3-12 section).

        l_s   = LS_COEF*sqrt(max(e, E_MIN))/N              [N^2 > 0]
        rho   = min(l_d/l_s, 1)
        w     = rho**CKS_BLEND_EXP
        C_eps = c_base + (1 - f)*w*C_ED*(l_d/l_ref)        [N^2 > 0]
        C_eps = c_base                                     [otherwise]

    so that eps = C_eps*e^{3/2}/l_d is EXACTLY

        eps = c_base*e^{3/2}/l_d  +  (1-f)*w*C_ED*e^{3/2}/l_ref,

    i.e. the HEAD channel untouched plus a second channel divided by a
    length that carries no ``e``.  Returning the effective COEFFICIENT
    rather than the rate keeps the split step's analytic decay substep
    exact: the sum is still of the form -K*e^{3/2}, so
    e = e*/(1 + K*sqrt(e*)*dt/2)^2 holds with no new integrator.

    Four properties, each pinned, each essential:

    * NOWHERE WEAKER.  Every factor of the added term is non-negative
      and l_d/l_ref is finite, so C_eps >= c_base POINTWISE and the
      dissipation rate is nowhere below HEAD's.  This is the property
      LD_STABILITY_LIMIT_REJECTED lacked -- removing a min can only
      lengthen l_d, and this term cannot shorten it.
    * NEUTRAL/UNSTABLE BITWISE, BY SELECTION.  At n2 <= 0 or
      ``n2 is None`` the SELECTED ``c_base`` is returned, exactly the
      S3-6k gate (``if (has_ced && ls_v > 0.0f)`` on the device), not an
      arithmetic cancellation: c_base + 0.0*x is c_base bitwise only
      because the multiply is selected away, and the np.where IS the
      selection.  The neutral log layer, the C_KV**3 = C_E identity and
      every M1-substituted cell are therefore untouched.
    * LES LIMB BITWISE.  The (1 - f) factor makes the change RANS-only:
      at f = 1 the added term is exactly 0.0 for ANY stratification, so
      every LES fixture is inert with the switch on.  This is not only
      the S3-6i/S3-6k pattern -- it is required, because at f = 1
      c_base = C_E = 0.93 already EXCEEDS Deardorff's own neutral total
      c_eps,1 + c_eps,2 = 0.70, so the LES limb is not the limb the
      published coefficient is missing from.
    * STABILITY-GATED.  w = rho**CKS_BLEND_EXP is 1.0 exactly where l_s
      binds l_d (:func:`dissipation_length` ends with the same
      ``np.minimum(l, ls)``, so l_d == l_s bitwise and rho == 1.0) and
      falls off quadratically as the stability limit goes slack.  A
      neutral RANS column has l_d == l_ref and would otherwise take the
      full C_E + C_ED = 1.44, a 55% dissipation increase that moves the
      log-law slope by ~10% -- outside the log-layer fixture's 2% band.
      The gate is what keeps the wall calibration.

    ``l_d`` is the DISSIPATION length -- the length the coefficient
    multiplies; ``l_ref`` the state-independent reference of
    :func:`neutral_dissipation_length`; ``e`` the SAME field
    :func:`dissipation_length` consumed.  ``c_base`` is the coefficient
    this channel is added TO (C_E on the default path; the S3-6k
    per-cell blend when that switch is also on -- the two amendments
    compose, and the composition is what the config ID records).
    """
    l64 = np.asarray(l_d, dtype=np.float64)
    lref = np.asarray(l_ref, dtype=np.float64)
    base = np.broadcast_to(np.asarray(c_base, dtype=np.float64),
                           np.broadcast(l64, lref).shape)
    if n2 is None:
        return np.array(base)
    e64 = np.maximum(np.asarray(e, dtype=np.float64), E_MIN)
    n2 = np.asarray(n2, dtype=np.float64)
    stable = n2 > 0.0
    ls = LS_COEF * np.sqrt(e64) / np.sqrt(np.where(stable, n2, 1.0))
    rho = np.where(stable, np.minimum(l64 / ls, 1.0), 0.0)
    w = rho ** CKS_BLEND_EXP
    added = (1.0 - f) * w * C_ED * (l64 / lref)
    return np.where(stable, base + added, base)


def _face_average(field):
    """Interior-face value by arithmetic mean of the neighbor centers.

    The face-diffusivity convention of the implicit channel (documented
    choice; harmonic averaging is the alternative for strongly layered
    K, not needed at the smooth K_v profiles of this closure).
    """
    return 0.5 * (field[:-1] + field[1:])


def _column_geometry(shape, dz=None, dz_col=None):
    """(z centers, face spacings h, layer thicknesses) for a column set.

    Exactly one of ``dz`` (uniform spacing; returns scalar h/thickness)
    and ``dz_col`` (layer thicknesses, shape (nz,) or matching
    ``shape``; the ``_z_centers`` cumsum-half-layer convention) must be
    given.  Shapes broadcast against fields of shape ``shape`` (any
    rank with z leading -- (nz,) columns and (nz, ny, nx) fields).
    """
    nz = shape[0]
    trail = (1,) * (len(shape) - 1)
    if (dz is None) == (dz_col is None):
        raise ValueError("exactly one of dz and dz_col must be given")
    if dz_col is None:
        z = ((np.arange(nz, dtype=np.float64) + 0.5) * float(dz))
        return z.reshape((nz,) + trail), float(dz), float(dz)
    t = np.asarray(dz_col, dtype=np.float64)
    if t.ndim == 1:
        if t.shape[0] != nz:
            raise ValueError(
                f"dz_col has {t.shape[0]} levels, field has {nz}")
        t = t.reshape((nz,) + trail)
    elif t.shape != tuple(shape):
        raise ValueError(
            f"dz_col shape {t.shape} must be (nz,) or match field "
            f"shape {tuple(shape)}")
    z = np.cumsum(t, axis=0) - 0.5 * t
    return z, z[1:] - z[:-1], t


def implicit_vertical_diffusion(phi, k_face, dt, dz=None, dz_col=None,
                                drag_bottom=None):
    """Backward-Euler vertical diffusion: one Thomas solve per column.

    Solves (I - dt*D_v) phi_new = phi with D_v the flux-form clamped
    operator: (D_v phi)_k = (F_{k+1/2} - F_{k-1/2})/thick_k, face flux
    F = K_f*(phi_{k+1} - phi_k)/h at the nz-1 interior faces and ZERO
    flux through both ends.  ``k_face`` holds the interior face
    diffusivities (leading dimension nz-1, broadcastable).  Properties
    (each pinned by test):

    * unconditionally stable -- the matrix is an irreducibly diagonally
      dominant M-matrix for any dt, K_f >= 0, so the solve exists and
      backward Euler damps every mode by 1/(1 + dt*lambda_m) in (0, 1];
    * max principle -- the inverse of an M-matrix with unit row sums of
      the diagonal/off-diagonal split is nonnegative with row sums 1,
      so min(phi) <= phi_new <= max(phi) elementwise;
    * conservation -- summing thick_k*(phi_new - phi)_k/dt telescopes
      the face fluxes to the (zero) end fluxes, so sum(thick*phi) is
      exact up to solver roundoff;
    * uniform-dz columns: cell-centered cosines cos(pi*m*(k+1/2)/nz)
      are the exact eigenvectors (DCT-II diagonalization of the
      zero-flux operator) with lambda_m = (4K/dz^2)*sin^2(pi*m/(2nz)),
      giving the closed-form fixture.

    S3-6j ``drag_bottom`` (module docstring, S3-6j section): a
    non-negative linearized surface-stress conductance c [m/s]
    (scalar or broadcastable to the trailing shape) replaces the
    bottom end's zero flux with the IMPLICIT flux
    F_{-1/2} = c*phi_new_0 -- the coefficient dt*c/thick_0 folds into
    the bottom diagonal (the YSU ``diag[0] = 1 + fric`` pattern,
    npref.py:6495-6497).  The augmented matrix adds a positive
    diagonal term, so it stays a strictly diagonally dominant
    M-matrix (nonnegative inverse, row sums <= 1): unconditional
    stability survives, the max principle weakens to the one-sided
    bound |phi_new| <= max|phi| (drag pulls the bottom cell toward
    rest, never past it), and conservation becomes flux-consistent --
    sum(thick*(phi_new - phi)) = -dt*c*phi_new_0 exactly (pinned).
    ``drag_bottom=None`` (and exactly-zero c) is bitwise the zero-flux
    solve.

    The Thomas sweep runs FP64, vectorized over the trailing
    dimensions; nz = 1 returns the input unchanged (no faces; a
    ``drag_bottom`` is deliberately NOT applied on this faceless
    branch -- documented, mirrored by the device kernel).
    """
    phi64 = np.asarray(phi, dtype=np.float64)
    nz = phi64.shape[0]
    if nz == 1:
        return phi64.copy()
    kf = np.asarray(k_face, dtype=np.float64)
    if kf.shape[0] != nz - 1:
        raise ValueError(
            f"k_face leading dimension {kf.shape[0]} must be nz-1 = "
            f"{nz - 1}")
    _, h, thick = _column_geometry(phi64.shape, dz=dz, dz_col=dz_col)
    tb = np.broadcast_to(np.asarray(thick, dtype=np.float64), phi64.shape)
    r = dt * kf / h                            # face conductances * dt
    sub = np.zeros_like(phi64)
    sup = np.zeros_like(phi64)
    sub[1:] = -(r / tb[1:])                    # couples k to k-1
    sup[:-1] = -(r / tb[:-1])                  # couples k to k+1
    diag = 1.0 - sub - sup
    if drag_bottom is not None:
        # S3-6j implicit surface stress: dt*c/thick_0 into the bottom
        # diagonal (docstring above; c = 0.0 is an FP-exact no-op).
        c = np.asarray(drag_bottom, dtype=np.float64)
        if np.any(c < 0.0):
            raise ValueError("drag_bottom must be non-negative")
        diag[0] = diag[0] + dt * c / tb[0]
    # Thomas forward elimination / back substitution, columns in bulk.
    cprime = np.empty_like(phi64)
    dprime = np.empty_like(phi64)
    cprime[0] = sup[0] / diag[0]
    dprime[0] = phi64[0] / diag[0]
    for k in range(1, nz):
        m = diag[k] - sub[k] * cprime[k - 1]
        cprime[k] = sup[k] / m
        dprime[k] = (phi64[k] - sub[k] * dprime[k - 1]) / m
    out = np.empty_like(phi64)
    out[-1] = dprime[-1]
    for k in range(nz - 2, -1, -1):
        out[k] = dprime[k] - cprime[k] * out[k + 1]
    return out


def _vertical_production(k_face, comps, h, thick):
    """Cell-centered vertical shear production from implicit-solved fields.

    Per interior face, eps_f = sum_i K_f*(delta phi_i)^2/h is the KE
    drain rate integrated over the face's unit column; splitting each
    face's eps half to each neighbor cell and dividing by the cell
    thickness gives P_k = 0.5*(eps_{k-1/2} + eps_{k+1/2})/thick_k with
    sum_k P_k = sum_f K_f*(delta phi)^2/(h*thick) -- on uniform columns
    exactly -sum phi.(D_v phi), the pairing the ledger theorem uses
    (module docstring, identity (ii)).  Pointwise non-negative.
    """
    shape = comps[0].shape
    p = np.zeros(shape)
    if shape[0] < 2:
        return p
    eps = np.zeros((shape[0] - 1,) + shape[1:])
    for comp in comps:
        d = comp[1:] - comp[:-1]
        eps = eps + k_face * d * d / h
    tb = np.broadcast_to(np.asarray(thick, dtype=np.float64), shape)
    p[:-1] += 0.5 * eps / tb[:-1]
    p[1:] += 0.5 * eps / tb[1:]
    return p


SFC_SIMILARITY_BETA_M = 4.8


SFC_SIMILARITY_BETA_H = 7.8


def _mo_surface_stable(spd1, theta1, theta_s, z1, z0, z0h=None,
                       iters=20):
    """Monin-Obukhov surface layer on the stable Businger-Dyer limb.

    Returns ``(u_star, theta_star, obukhov_L)`` for a prescribed
    surface potential temperature ``theta_s`` under the level-1 wind
    and temperature, iterating

        u*     = kappa*spd1/(ln(z1/z0)  - psi_m(z1/L) + psi_m(z0/L))
        theta* = kappa*(theta1 - theta_s)
                 /(ln(z1/z0h) - psi_h(z1/L) + psi_h(z0h/L))
        L      = u*^2*theta1/(kappa*G_ACCEL*theta*)

    with psi_m = -BETA_M*zeta and psi_h = -BETA_H*zeta (the linear
    stable limb; SFC_SIMILARITY_BETA_M/_H carry the provenance).  The
    neutral values seed the iteration and an unstable or vanishing
    theta* returns the neutral pair, so the routine is total.  Stable
    similarity is a contraction here for the depths this engine runs;
    ``iters`` is fixed rather than tolerance-tested so the result is a
    deterministic function of its inputs.
    """
    z0h = z0 if z0h is None else z0h
    lnm, lnh = math.log(z1 / z0), math.log(z1 / z0h)
    spd1 = max(float(spd1), SFC_WSPD_FLOOR)
    dth = float(theta1) - float(theta_s)
    ust = KARMAN * spd1 / lnm
    tst = KARMAN * dth / lnh
    if dth <= 0.0:
        return ust, 0.0, math.inf
    for _ in range(iters):
        if tst <= 0.0 or ust <= 0.0:
            return KARMAN * spd1 / lnm, 0.0, math.inf
        lmo = ust * ust * float(theta1) / (KARMAN * G_ACCEL * tst)
        dm = lnm + SFC_SIMILARITY_BETA_M * (z1 - z0) / lmo
        dh = lnh + SFC_SIMILARITY_BETA_H * (z1 - z0h) / lmo
        if dm <= 0.0 or dh <= 0.0:                # decoupled: no solution
            return 0.0, 0.0, math.inf
        ust = KARMAN * spd1 / dm
        tst = KARMAN * dth / dh
    lmo = (ust * ust * float(theta1) / (KARMAN * G_ACCEL * tst)
           if tst > 0.0 else math.inf)
    return ust, tst, lmo


def column_stable_cooling(geo_u, geo_v, fcor, nz, dz, theta_init,
                          dt, steps, theta_surface, cooling_rate,
                          z0=0.1, e_init=None, stable_dissipation=False,
                          additive_dissipation=False):
    """Stably stratified column under a PRESCRIBED SURFACE COOLING RATE:
    the closure's single-column benchmark engine.

    The regime every other column engine here lacks.  ``column_ekman_
    balance`` runs the drag+PGF balance NEUTRALLY (uniform theta, no
    heat channel), so it cannot score a stable boundary layer at all;
    the jet and lake fixtures run REAL frames, which pin what this
    closure does on one case rather than what it does against a
    published reference.  This engine is the third thing: a fully
    specified idealized stable column whose answer is published, so the
    closure's stable limb can be SCORED and not merely compared with
    itself.  The forcing is deliberately spelled as arguments -- the
    benchmark's own numbers live with the benchmark
    (:mod:`woof.verify.cases.gabls1`), never here.

    THE COLUMN.  An f-plane with a prescribed constant geostrophic wind
    (``geo_u``, ``geo_v``), the same simultaneous explicit forcing
    ``column_ekman_balance`` uses,

        du += dt*fcor*(v - geo_v),   dv += -dt*fcor*(u - geo_u),

    over a prognostic potential temperature whose lower boundary is a
    prescribed surface value falling linearly at ``cooling_rate``
    [K s^-1].  Surface momentum and heat both come from the SAME
    Monin-Obukhov solve (:func:`_mo_surface_stable`) on the stable
    Businger-Dyer limb, so the surface layer is the benchmark's, not
    this closure's: the momentum row enters the implicit solve as the
    drag conductance c = u*^2/max(|V1|, SFC_WSPD_FLOOR) (the live S3-6j
    bottom boundary condition) and the heat row as the explicit
    bottom-cell deposit dt*u**theta*/thick_0 (SFC_SCALAR_FLUX, the
    live convention).

    THE INTERIOR IS THE LIVE RANS LIMB, evaluated through the same
    authority functions the split step calls at f = 0 -- there is no
    second formulation here:

        (l_mix_rans, l_eps_rans) = bl89_rans_lengths(...)
        K_v  = stable_limit_coefficient(l_mix_rans, e, N^2)
               * l_mix_rans * sqrt(e)
        K_h  = K_v/prandtl_blend(0.0)
        l_d  = dissipation_length(e, delta=1, n2, lb=l_eps_rans, f=0)
        eps  : the S3-6d ANALYTIC decay substep at the coefficient the
               two switches select, exactly as sase_split_step forms it

    with e advanced by shear production on the solved faces, buoyancy
    destruction -K_h*N^2, the analytic decay, and the implicit 2*K_v
    e-transport -- the split step's own order of operations, minus the
    horizontal channel a single column does not have.  ``delta`` is
    irrelevant at f = 0 (the geometric blend returns l_eps_rans
    FP-exactly) and the additive channel's reference length is
    l_B(z + z0) there, likewise independent of it.

    Returns the final profiles plus the surface series (u*, theta*, the
    surface potential temperature and the kinematic heat flux per
    step), which is what a benchmark scores.  The boundary-layer depth
    is NOT computed here: its definition belongs to the benchmark
    (there are several, and they disagree), so the caller gets the
    stress profile and applies its own.
    """
    nz = int(nz)
    dz = float(dz)
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    theta = np.array(theta_init, dtype=np.float64)
    e = (np.full(nz, E_MIN) if e_init is None
         else np.maximum(np.asarray(e_init, dtype=np.float64), E_MIN))
    u = np.full(nz, float(geo_u))
    v = np.full(nz, float(geo_v))
    pr_t = prandtl_blend(0.0)
    series = {"ust": np.empty(steps), "tstar": np.empty(steps),
              "theta_s": np.empty(steps), "hfx_kin": np.empty(steps)}
    th_s = float(theta_surface)
    for n in range(steps):
        spd1 = float(np.hypot(u[0], v[0]))
        ust, tst, _ = _mo_surface_stable(spd1, theta[0], th_s, z[0], z0)
        series["ust"][n] = ust
        series["tstar"][n] = tst
        series["theta_s"][n] = th_s
        series["hfx_kin"][n] = -ust * tst        # w'theta' at the surface
        n2 = brunt_vaisala_n2(theta, dz)
        e64 = np.maximum(e, E_MIN)
        l_mix_r, l_eps_r = bl89_rans_lengths(theta, e64, z, dz, n2, z0)
        kv = (stable_limit_coefficient(l_mix_r, e64, n2) * l_mix_r
              * np.sqrt(e64))
        k_face = _face_average(kv)
        # momentum: PGF/Coriolis at the pre-update state, then the
        # implicit solve with the live drag row
        du = dt * fcor * (v - float(geo_v))
        dv = -dt * fcor * (u - float(geo_u))
        u, v = u + du, v + dv
        c_drag = ust * ust / max(spd1, SFC_WSPD_FLOOR)
        u = implicit_vertical_diffusion(u, k_face, dt, dz=dz,
                                        drag_bottom=c_drag)
        v = implicit_vertical_diffusion(v, k_face, dt, dz=dz,
                                        drag_bottom=c_drag)
        # heat: explicit surface deposit then the implicit K_h channel
        theta[0] += dt * (-ust * tst) / dz
        theta = implicit_vertical_diffusion(theta, k_face / pr_t, dt,
                                            dz=dz)
        # subgrid energy: the split step's own sequence at f = 0
        p_v = _vertical_production(k_face, (u, v), dz, dz)
        buoy = -(kv / pr_t) * n2
        ld = dissipation_length(e64, 1.0, n2, lb=l_eps_r, f=0.0)
        e_star = e + dt * (p_v + buoy)
        c_eps = (stable_dissipation_coefficient(ld, e64, n2, f=0.0)
                 if stable_dissipation else C_E)
        if additive_dissipation:
            l_ref = neutral_dissipation_length(z, 1.0, f=0.0, z0=z0)
            c_eps = additive_dissipation_coefficient(
                ld, l_ref, e64, n2, f=0.0, c_base=c_eps)
        b = c_eps * np.sqrt(np.maximum(e_star, E_MIN)) / (2.0 * ld)
        e = np.maximum(e_star / (1.0 + b * dt) ** 2, E_MIN)
        e = np.maximum(implicit_vertical_diffusion(
            e, 2.0 * k_face, dt, dz=dz), E_MIN)
        th_s -= dt * float(cooling_rate)
    e64 = np.maximum(e, E_MIN)
    n2 = brunt_vaisala_n2(theta, dz)
    l_mix_r, _ = bl89_rans_lengths(theta, e64, z, dz, n2, z0)
    kv = (stable_limit_coefficient(l_mix_r, e64, n2) * l_mix_r
          * np.sqrt(e64))
    k_face = _face_average(kv)
    return {"z": z, "u": u, "v": v, "theta": theta, "e": e, "kv": kv,
            "z_face": 0.5 * (z[1:] + z[:-1]),
            "uw": -k_face * (u[1:] - u[:-1]) / dz,
            "vw": -k_face * (v[1:] - v[:-1]) / dz,
            "wt": -(k_face / pr_t) * (theta[1:] - theta[:-1]) / dz,
            **{k: val for k, val in series.items()}}
