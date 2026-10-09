"""Shared SASE closure constants and host mathematics for production backends."""

from __future__ import annotations
from dataclasses import dataclass
from woof.core.sase_limits import E_MIN, MAX_COLUMN_LEVELS
import numpy as np


C_E = 0.93


C_K = 0.10


PR_LES = 1.0 / 3.0


PR_RANS = 0.85


C_MOM_BG = C_K / PR_LES


CNU_MAX = 0.5


F_MIN, F_MAX = 0.0, 1.0


LS_COEF = 0.76


G_ACCEL = 9.81


KARMAN = 0.4


BLACKADAR_LAMBDA = 150.0


C_S = 0.25


SMAG_KM_CAP = 10.0


NU_BLEND_EPS = 1.0e-30


C_KV = C_E ** (1.0 / 3.0)


F_CAP_KNEE = 1.0


F_CAP_WIDTH = 2.0


RIB_CRIT = 0.25


RIB_WSPD2_FLOOR = 1.0


N2_SCREEN = 1.0e-4


BL89_MIX_EXP = 2.0 / 3.0


C_KS = 0.25


CKS_BLEND_EXP = 2.0


C_ES = 0.19


C_ED = 0.51


SFC_WSPD_FLOOR = 0.1


CP_AIR = 1004.5


RD_AIR = 287.0


RV_AIR = 461.6


EP2_RV = RD_AIR / RV_AIR


XLV = 2.5e6


SVP1 = 0.6112


SVP2 = 17.67


SVP3 = 29.65


SVPT0 = 273.15


P0_REF = 1.0e5


VENT_MASK = "bulk-theta-es-v1"


VENT_MB_COEF = 0.03


VENT_ENT_COEF = 0.4


VENT_SIGW_SHARE = 2.0 / 3.0


VENT_DEPTH_CAP = 4000.0


VENT_THETA_STEP_CAP = 0.14


VENT_QT_STEP_CAP = VENT_THETA_STEP_CAP * CP_AIR / XLV


VENT_MIN_RUN_CELLS = 2


VENT_SAT_ADJUST_ITERS = 12


_FILTER_WEIGHTS = {
    2: np.array([0.25, 0.5, 0.25], dtype=np.float64),
    4: np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=np.float64),
}


@dataclass(frozen=True)
class SensorState:
    alpha: float
    slope: float
    e_res: float


def partition_cap(delta_h: float, zi: float) -> float:
    """S3-6f prescribed partition cap f_cap(delta_h/z_i), scalar.

    FP-exact 1.0 for rho = delta_h/z_i <= F_CAP_KNEE (exp(-0.0) == 1.0:
    the gray zone and LES regime are untouched bitwise), the C^1
    Gaussian ramp exp(-((rho - F_CAP_KNEE)/F_CAP_WIDTH)^2) beyond it --
    monotone decreasing, 0.78 at rho = 2, 0.37 at rho = 3, 1.6e-9 at
    rho = 10 (form rationale: module docstring; Honnert/Shin-Hong-class
    prescribed Delta/z_i bound, credited).  ``zi`` must be positive
    (the z_i diagnosis floors at the first interior layer center).
    """
    rho = float(delta_h) / float(zi)
    x = max(rho - F_CAP_KNEE, 0.0) / F_CAP_WIDTH
    return float(np.exp(-(x * x)))


@dataclass(frozen=True)
class WSensorState:
    f_w: float          #: upper bound on f (1.0 = sensor abstains)
    alpha_w: float      #: w-sensed subgrid fraction (degenerate 1.0)
    e_res_w: float      #: screened resolved-w energy 0.5*D2_w(2)
    coverage: float     #: screen-passing cell fraction


def _w_bound_tail(d2w2: float, count: int, ncell: int,
                  e_mean: float) -> WSensorState:
    """Shared scalar tail of the w-sensor (host + device launcher).

    One copy of the E_r_w = 0.5*D2_w(2), alpha_w clip, f_w = 1 -
    alpha_w arithmetic and the silent-screen abstention (count == 0 ->
    f_w = 1.0, no constraint), so the device path cannot drift from the
    authority (the ``_solve_tail`` idiom).
    """
    if count == 0:
        return WSensorState(f_w=1.0, alpha_w=1.0, e_res_w=0.0,
                            coverage=0.0)
    e_res_w = 0.5 * float(d2w2)
    alpha_w = float(np.clip(e_mean / (e_mean + e_res_w), 0.0, 1.0))
    return WSensorState(f_w=1.0 - alpha_w, alpha_w=alpha_w,
                        e_res_w=e_res_w,
                        coverage=float(count) / float(ncell))


def _solve_tail(gram, proj):
    """Shared 2x2 dynamic-solve tail: cond gate, solve, clip/recovery.

    Extracted (S3-3 carry-forward) so the device launcher
    ``woof.core.sase.launch_dynamic_solve`` and :func:`dynamic_solve`
    share ONE copy of the tail arithmetic: the ``np.linalg.cond`` gate at
    1e12, the 2x2 ``np.linalg.solve``, and the clip/recovery order for
    (c_nu, f) from the raw weights a = f*c_nu, b = (1-f)*C_MOM_BG
    (S3-6g: the momentum-basis weight is the fixed momentum-channel
    constant, bit-identical to the former C_K/PR_T spelling -- decision
    table, module docstring).  Pure refactor: operations are
    line-identical to both former copies, so every golden (host and
    device) is unchanged bitwise.
    """
    if np.linalg.cond(gram) > 1.0e12:
        # Degenerate resolved field (alpha -> 1): fall back to the
        # equilibrium momentum weight, no eddy contribution.
        return 0.0, 0.0
    eddy_w, mom_w = np.linalg.solve(gram, proj)
    f = float(np.clip(1.0 - mom_w / C_MOM_BG, F_MIN, F_MAX))
    c_nu = float(np.clip(eddy_w / max(f, 1.0e-12), 0.0, CNU_MAX))
    return c_nu, f


def prandtl_blend(f: float) -> float:
    """S3-6g regime-consistent turbulent Prandtl number Pr_t(f).

        Pr_t(f) = f*PR_LES + (1 - f)*PR_RANS

    -- the registered blend Pr_t = PR_RANS + f*(PR_LES - PR_RANS)
    (controller ledger 2026-07-20, S3-6g registration) rewritten in the
    two-product form every other regime blend of this module uses
    (:func:`dissipation_length`'s l_d, :func:`governed_stress`'s nu),
    so BOTH endpoints are FP-exact: f = 1 gives 1.0*PR_LES +
    0.0*PR_RANS = PR_LES bitwise (the LES regime and every f = 1
    reduction pin are untouched), f = 0 gives PR_RANS bitwise (the
    RANS fixed points are exact).  Affine, hence monotone, between.
    ``f`` is the step's USED partition weight (post-S3-6f caps): the
    same regime judgment that blends l_d and the governed stress.
    Reads the module globals, so instrumented runs can pin the regime
    -- monkeypatching PR_RANS = PR_LES reproduces the pre-S3-6g
    fixed-Pr formulation (bitwise at f = 0, the RANS-column fixtures'
    branch), which is how the inversion-persistence fixture produces
    its RED leg.
    """
    return f * PR_LES + (1.0 - f) * PR_RANS


def _blackadar_length(z, z0=0.0):
    """Neutral Blackadar (1962) length l_B = k*(z+z0)/(1+k*(z+z0)/lambda).

    Rises as k*(z+z0) off the surface, saturates at BLACKADAR_LAMBDA
    aloft.  ``z`` is height above the surface at layer centers; ``z0``
    (surface roughness) is an argument rather than a registered
    constant because it is model surface data, not closure physics --
    layer-center evaluation (z >= dz1/2 >> z0) makes it subdominant and
    the default 0.0 is exact for the reference fixtures.
    """
    kz = KARMAN * (np.asarray(z, dtype=np.float64) + z0)
    return kz / (1.0 + kz / BLACKADAR_LAMBDA)
