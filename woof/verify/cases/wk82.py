"""Weisman-Klemp (1982) quarter-circle supercell benchmark.

Qualification preserves the WOOF v1.0.3 instantaneous mover measurements
and bounds. Persistent signed-object tracking is an additional diagnostic.

Sounding (Weisman & Klemp 1982 MWR, equations (1)-(2)):

    theta(z) = theta_0 + (theta_tr - theta_0) * (z/z_tr)^(5/4)   z <= z_tr
             = theta_tr * exp(g * (z - z_tr) / (cp * T_tr))      z >  z_tr
    RH(z)    = 1 - 0.75 * (z/z_tr)^(5/4)                         z <= z_tr
             = 0.25                                              z >  z_tr
    qv(z)    = min(RH(z) * qvs(p, T), 14 g/kg)

with theta_0 = 300 K, theta_tr = 343 K, z_tr = 12 km, T_tr = 213 K.

The WRF v4.6.1 quarter_ss initializer reads the tabulated
``test/em_quarter_ss/input_sounding`` in ``get_sounding``; the analytic
formulas above follow WK82 directly.

RH -> qv closure: the saturation mixing ratio uses the model's OWN Teten
formula -- the constants of ``phys/module_mp_kessler.F`` already in
``woof.core.constants`` (SVP1/SVP2/SVP3/SVPT0, EP2 = Rd/Rv) -- so that
"relative humidity" means the same thing to the initial state and to the
Kessler scheme that later acts on it.  Since qvs depends on the pressure
and the pressure (hydrostatically) on qv, the profile is closed by a
fixed-point iteration in the style of the local ``get_sounding``'s own
hydrostatic loops (a fixed 10-iteration sweep, ``do it=1,10``): integrate
the Exner function with the exact virtual potential temperature
``theta_v = theta * (1 + (Rv/Rd) qv) / (1 + qv)`` (the ratio comes from
the constants -- never a hardcoded 1.61/0.61) on a 10 m fine grid that
includes every requested height, then refresh qv = min(RH*qvs, 14 g/kg).
Float64 throughout (setup-time code).
"""

from __future__ import annotations

import numpy as np

OUTPUT_REQUIREMENTS = ("matplotlib", "netCDF4")

from woof.core import constants as c

#: WK82 sounding parameters (Weisman & Klemp 1982, eqs. (1)-(2)).
THETA_0 = 300.0     # surface potential temperature (K)
THETA_TR = 343.0    # tropopause potential temperature (K)
Z_TR = 12000.0      # tropopause height (m)
T_TR = 213.0        # tropopause temperature (K)
QV0 = 0.014         # boundary-layer mixing-ratio cap (kg/kg): 14 g/kg

#: Fine-grid spacing (m) and fixed-point sweep count of the hydrostatic
#: qv closure (the sweep count mirrors get_sounding's ``do it=1,10``).
_DZ_FINE = 10.0
_N_ITER = 10


def wk82_theta(z) -> np.ndarray:
    """WK82 eq. (1): potential temperature (K) at height ``z`` (m)."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(z <= Z_TR,
                    THETA_0 + (THETA_TR - THETA_0) * (z / Z_TR) ** 1.25,
                    THETA_TR * np.exp(c.G * (z - Z_TR) / (c.CP * T_TR)))


def wk82_relhum(z) -> np.ndarray:
    """WK82 eq. (2): relative humidity (0-1) at height ``z`` (m)."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(z <= Z_TR, 1.0 - 0.75 * (z / Z_TR) ** 1.25, 0.25)


def saturation_mixing_ratio(p, T) -> np.ndarray:
    """Saturation mixing ratio (kg/kg) over water at pressure ``p`` (Pa)
    and temperature ``T`` (K), exactly as the local Kessler scheme
    (phys/module_mp_kessler.F) diagnoses it: Teten
    ``es = 1000*svp1*exp(svp2*(T - svpt0)/(T - svp3))`` (Pa),
    ``qvs = ep2*es/(p - es)``."""
    es = 1000.0 * c.SVP1 * np.exp(c.SVP2 * (np.asarray(T, np.float64)
                                            - c.SVPT0)
                                  / (np.asarray(T, np.float64) - c.SVP3))
    return c.EP2 * es / (np.asarray(p, np.float64) - es)


def wk82_sounding(z, p_surf: float = 1.0e5, full: bool = False):
    """WK82 analytic sounding at heights ``z`` (m): ``(theta, qv)``.

    ``theta`` (K) and ``qv`` (kg/kg) are float64 arrays shaped like ``z``.
    ``full=True`` appends the closure's hydrostatic ``(p, T)`` (Pa, K) at
    the same heights -- the pressure the RH -> qv conversion used (the
    module docstring describes the fixed-point construction).  ``p_surf``
    defaults to the WK82/em_quarter_ss 1000 hPa surface.
    """
    z_in = np.asarray(z, dtype=np.float64)
    zq = np.atleast_1d(z_in).ravel()
    if zq.size == 0 or (zq < 0.0).any():
        raise ValueError("wk82_sounding needs nonempty heights >= 0, got "
                         f"range [{zq.min() if zq.size else np.nan}, "
                         f"{zq.max() if zq.size else np.nan}]")

    # Integration grid: 10 m spacing from the surface past the highest
    # requested level, with every requested height inserted exactly so no
    # output value is interpolated.
    zmax = float(zq.max())
    zg = np.union1d(np.arange(0.0, zmax + _DZ_FINE, _DZ_FINE), zq)
    th = wk82_theta(zg)
    rh = wk82_relhum(zg)
    dz = np.diff(zg)

    qv = np.zeros_like(zg)
    pi = np.empty_like(zg)
    for _ in range(_N_ITER):
        thv = th * (1.0 + c.RVOVRD * qv) / (1.0 + qv)
        thv_mid = 0.5 * (thv[:-1] + thv[1:])
        pi[0] = (p_surf / c.P0) ** c.RCP
        pi[1:] = pi[0] - np.cumsum(c.G * dz / (c.CP * thv_mid))
        p = c.P0 * pi ** (1.0 / c.RCP)
        T = th * pi
        qv = np.minimum(rh * saturation_mixing_ratio(p, T), QV0)

    idx = np.searchsorted(zg, zq)                  # exact: zq is in zg
    pick = lambda a: a[idx].reshape(z_in.shape)
    if full:
        return pick(th), pick(qv), pick(p), pick(T)
    return pick(th), pick(qv)


# ---------------------------------------------------------------------------
# Task 11: the full 3-D quarter-circle supercell benchmark.
# ---------------------------------------------------------------------------

#: WK82 quarter_ss thermal (module_initialize_ideal.F ideal_pert CASE
#: (quarter_ss), lines 1088-1135): amplitude (K), horizontal radius (m),
#: center height / half-depth (m).  The Fortran uses 1500 m (the plan
#: text's "1.4 km" reflects the WK82 paper; the plan defers the init to
#: the bundle's initializer, which is transcribed here verbatim).
BUBBLE_DELT = 3.0
BUBBLE_RADIUS = 10000.0
BUBBLE_ZC = 1500.0

#: Quarter-circle hodograph: the (z [m], u, v [m/s]) wind columns of the
#: local WRF v4.6.1 ``test/em_quarter_ss/input_sounding``, verbatim, up to
#: 7 km; every shipped row above 7 km holds (27.15, 3.5) exactly, so the
#: 30 km sentinel row reproduces the file's constant-above-7-km profile
#: under linear interpolation.  The constant storm-motion frame translation
#: that keeps the right mover centered is already baked into these columns
#: (surface wind (-8.10, -4.94), not zero) -- used as-is, nothing further
#: subtracted (plan Task 11; no umove/vmove defaults exist in this tree).
HODOGRAPH_ZUV = np.array([
    (10.0, -8.0998, -4.9419), (35.0, -8.0976, -4.7968),
    (100.0, -8.0802, -4.4199), (200.0, -8.0208, -3.8426),
    (300.0, -7.9222, -3.2706), (400.0, -7.7848, -2.7067),
    (500.0, -7.6092, -2.1535), (600.0, -7.3963, -1.6136),
    (700.0, -7.1470, -1.0894), (800.0, -6.8626, -0.5835),
    (900.0, -6.5442, -0.0982), (1000.0, -6.1935, 0.3642),
    (1100.0, -5.8121, 0.8017), (1200.0, -5.4017, 1.2121),
    (1300.0, -4.9642, 1.5935), (1400.0, -4.5018, 1.9442),
    (1500.0, -4.0165, 2.2626), (1600.0, -3.5106, 2.5470),
    (1700.0, -2.9864, 2.7963), (1800.0, -2.4465, 3.0092),
    (1900.0, -1.8933, 3.1848), (2000.0, -1.3294, 3.3222),
    (2100.0, -0.7574, 3.4208), (2200.0, -0.1801, 3.4802),
    (2300.0, 0.4000, 3.5000), (2400.0, 0.9691, 3.5000),
    (2500.0, 1.5383, 3.5000), (2600.0, 2.1074, 3.5000),
    (2700.0, 2.6766, 3.5000), (2800.0, 3.2457, 3.5000),
    (2900.0, 3.8149, 3.5000), (3000.0, 4.3840, 3.5000),
    (3100.0, 4.9532, 3.5000), (3200.0, 5.5223, 3.5000),
    (3300.0, 6.0915, 3.5000), (3400.0, 6.6606, 3.5000),
    (3500.0, 7.2298, 3.5000), (3600.0, 7.7989, 3.5000),
    (3700.0, 8.3681, 3.5000), (3800.0, 8.9372, 3.5000),
    (3900.0, 9.5064, 3.5000), (4000.0, 10.0755, 3.5000),
    (4100.0, 10.6447, 3.5000), (4200.0, 11.2138, 3.5000),
    (4300.0, 11.7830, 3.5000), (4400.0, 12.3521, 3.5000),
    (4500.0, 12.9213, 3.5000), (4600.0, 13.4904, 3.5000),
    (4700.0, 14.0596, 3.5000), (4800.0, 14.6287, 3.5000),
    (4900.0, 15.1979, 3.5000), (5000.0, 15.7670, 3.5000),
    (5100.0, 16.3362, 3.5000), (5200.0, 16.9053, 3.5000),
    (5300.0, 17.4745, 3.5000), (5400.0, 18.0436, 3.5000),
    (5500.0, 18.6128, 3.5000), (5600.0, 19.1819, 3.5000),
    (5700.0, 19.7511, 3.5000), (5800.0, 20.3202, 3.5000),
    (5900.0, 20.8894, 3.5000), (6000.0, 21.4585, 3.5000),
    (6100.0, 22.0277, 3.5000), (6200.0, 22.5968, 3.5000),
    (6300.0, 23.1660, 3.5000), (6400.0, 23.7351, 3.5000),
    (6500.0, 24.3043, 3.5000), (6600.0, 24.8734, 3.5000),
    (6700.0, 25.4426, 3.5000), (6800.0, 26.0117, 3.5000),
    (6900.0, 26.5809, 3.5000), (7000.0, 27.1500, 3.5000),
    (30000.0, 27.1500, 3.5000),
])

#: Analysis times (s) for the split geometry and plan-view PNGs.
ANALYSIS_TIMES = (3600.0, 5400.0, 7200.0)

#: Updraft-core threshold (m/s) and analysis height (m) for the split
#: metrics (plan: "w > 15 m/s clusters at z ~ 5 km").
CORE_W = 15.0
CORE_Z = 5000.0

#: UH-proxy integration bounds (m): plan "int w*zeta dz, 2-5 km".
UH_ZBOT, UH_ZTOP = 2000.0, 5000.0

#: Dry-mass drift sampling cadence (steps) for the ``mass_drift_max``
#: metric (Phase 2 final review T11-1: the original 100-step cadence could
#: miss short-lived boundary-exchange spikes between samples; the FP64
#: device-sum reduction is cheap next to a model step, so every 10 steps
#: costs nothing measurable).
MASS_SAMPLE_EVERY = 10
TRACKING_DIAGNOSTIC_INTERVAL_S = 60.0


def mass_sample_steps(n_total: int) -> range:
    """1-based step indices at which :func:`run` samples the dry-mass
    drift for ``mass_drift_max``: every :data:`MASS_SAMPLE_EVERY` steps.
    (``run()`` additionally folds the final post-run drift into the
    metric, so an ``n_total`` off the cadence still gates its last step.)
    """
    return range(MASS_SAMPLE_EVERY, n_total + 1, MASS_SAMPLE_EVERY)

#: Canonical WOOF v1.0.3 gates, with strict bounds (None is unbounded).
#: Core counts and movers use the instantaneous thresholded 5-km snapshots.
#: Separation growth is measured over the early and total analysis intervals;
#: right-mover displacement projects onto the source 0--6 km shear normal.
#: Persistent signed identities are additional ``tracked_*`` diagnostics and
#: do not replace these snapshot measurements or change qualification.
GATES = {
    "n_cores_3600": (1, None),
    "n_cores_5400": (1, None),
    "n_cores_7200": (1, None),
    "sep_3600": (0.0, None),
    "sep_growth_3600_5400": (0.0, None),
    "sep_growth": (0.0, None),
    "right_of_shear_m": (0.0, None),
    "w_max": (25.0, 70.0),
    "w_sustained_fraction": (0.95, None),
    "qr_max": (5.0e-3, None),
    "mass_drift_max": (None, 2.0e-3),
    "mass_drift": (None, 3.0e-4),
    "mass_closure_residual_max": (None, 1.0e-5),
}


def hodograph_uv(coord, ztop: float):
    """u/v at the model half levels, consumed exactly as the local WRF
    initializer consumes ``get_sounding``'s wind columns
    (module_initialize_ideal.F:1536-1551, flat terrain): linear
    interpolation against the FULL (moist) sounding pressure at
    ``p_level = c3h*(p_surf - p_top) + c4h + p_top``, with ``p_surf``/
    ``p_top`` the sounding pressures at z = 0 and z = ztop (WRF
    ``grid%p_top = interp_0(p_in, zk, ztop)``).  Below the file's lowest
    wind row (10 m) the interpolation clamps, as WRF's ``interp_0`` does.
    Returns float64 ``(u, v)``, each ``(nz,)``.
    """
    zs = HODOGRAPH_ZUV[:, 0]
    _, _, p_all, _ = wk82_sounding(np.concatenate([zs, [0.0, ztop]]),
                                   full=True)
    p_snd, p_surf, p_top = p_all[:-2], p_all[-2], p_all[-1]
    p_half = coord.c3h * (p_surf - p_top) + coord.c4h + p_top
    # np.interp needs ascending x; pressure decreases with height.
    u = np.interp(p_half, p_snd[::-1], HODOGRAPH_ZUV[::-1, 1])
    v = np.interp(p_half, p_snd[::-1], HODOGRAPH_ZUV[::-1, 2])
    return u, v


def default_config():
    """Plan Task 11 setup: 168 x 168 x 20 km at dx = dy = 1 km, nz = 60
    (~333 m), dt = 6 s, open x/y lateral boundaries, implicit-w Rayleigh
    top over the top 5 km, the production dissipation package (km_opt = 4
    Smagorinsky + monotonic 6th-order filter), Kessler warm rain, f = 0
    (the model carries no Coriolis); WRF's external-mode divergence
    damping at the em_quarter_ss namelist value (emdiv = 0.01) -- the
    stock stabilizer of the open-boundary external mode.  wrfout frames
    every 1800 s (t = 0/1800/3600/5400/7200: the analysis times; the
    ~60 MB 3-D frames make the WRF-default 5-min output pointlessly heavy
    for a benchmark)."""
    from woof.config import RunConfig

    return RunConfig(nx=168, ny=168, nz=60, dx=1000.0, dy=1000.0,
                     ztop=20000.0, dt=6.0, run_seconds=7200.0,
                     moist=True, mp_physics=1, moist_adv_opt=1,
                     km_opt=4, c_s=0.25, diff_6th_opt=2,
                     diff_6th_factor=0.12,
                     damp_opt=3, zdamp=5000.0, dampcoef=0.2,
                     open_x=True, open_y=True, emdiv=0.01,
                     output_interval_s=1800.0, case="wk82")


def _thp_func(cfg):
    """The quarter_ss thermal shared by CUDA and Metal initialization."""
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    y = (np.arange(ny) + 0.5) * cfg.dy - 0.5 * ny * cfg.dy

    def thp_func(x, z):
        xrad = x[None, None, :] / BUBBLE_RADIUS
        yrad = y[None, :, None] / BUBBLE_RADIUS
        zrad = (z[:, None, None] - BUBBLE_ZC) / BUBBLE_ZC
        rad = np.sqrt(xrad ** 2 + yrad ** 2 + zrad ** 2)
        thp = np.where(rad <= 1.0,
                       BUBBLE_DELT * np.cos(0.5 * np.pi * rad) ** 2, 0.0)
        return np.broadcast_to(thp, (nz, ny, nx))

    return thp_func


def build(cfg, coord, base, *, array_module=None):
    """Balanced moist WK82 state + quarter-circle hodograph + 3 K thermal.

    Transcribed from the local quarter_ss initializer: the WK82 qv profile
    rides the dry base state through ``init_moist_balanced`` (the WRF
    moist rebalance), the thermal is ``delt*cos(pi*RAD/2)^2`` inside the
    unit ellipsoid RAD centered at (domain center, 1.5 km) with 10 km
    horizontal and 1.5 km vertical radii (the geopotential is re-integrated
    through the bubble's alpha by the same builder), and the winds are the
    shipped hodograph columns broadcast to the staggered u/v fields (flat
    terrain: one column everywhere, as in the Fortran's z_at_u == 0 path).
    """
    from woof.core.moist import init_moist_balanced
    from woof.core.state import DTYPE, _state_array_module

    s = init_moist_balanced(cfg, coord, base,
                            lambda z: wk82_sounding(z)[1], _thp_func(cfg),
                            array_module=array_module)
    u, v = hodograph_uv(coord, cfg.ztop)
    xp = _state_array_module(s)
    if getattr(s, "_metal_setup_state", False):
        # Metal requires explicit complete uploads; expand this setup-only
        # profile on the host before transferring the staggered winds.
        s.u[...] = xp.asarray(np.broadcast_to(
            np.asarray(u, dtype=DTYPE)[:, None, None], s.u.shape), dtype=DTYPE)
        s.v[...] = xp.asarray(np.broadcast_to(
            np.asarray(v, dtype=DTYPE)[:, None, None], s.v.shape), dtype=DTYPE)
    else:
        s.u[...] = xp.asarray(u, dtype=DTYPE)[:, None, None]
        s.v[...] = xp.asarray(v, dtype=DTYPE)[:, None, None]
    return s


def _clusters(mask, min_cells=3):
    """4-connected clusters of True cells of a 2-D mask: list of index
    arrays ``(rows, cols)``, ignoring clusters below ``min_cells``."""
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    for j0, i0 in zip(*np.nonzero(mask)):
        if seen[j0, i0]:
            continue
        stack = [(j0, i0)]
        seen[j0, i0] = True
        cells = []
        while stack:
            j, i = stack.pop()
            cells.append((j, i))
            for jj, ii in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                if (0 <= jj < mask.shape[0] and 0 <= ii < mask.shape[1]
                        and mask[jj, ii] and not seen[jj, ii]):
                    seen[jj, ii] = True
                    stack.append((jj, ii))
        if len(cells) >= min_cells:
            out.append(np.asarray(cells).T)
    return out


def sustained_updraft_fraction(history, threshold=25.0, start=3600.0,
                               end=7200.0) -> float:
    """Fraction of finite ``w_max`` samples above threshold in a time span."""
    history = np.asarray(history, dtype=np.float64)
    selected = history[(history[:, 0] >= start) & (history[:, 0] <= end), 1]
    if selected.size == 0 or not np.isfinite(selected).all():
        return float("nan")
    return float(np.mean(selected > threshold))


def separation_growth_intervals(separation) -> tuple[float, float]:
    """Meridional separation changes across 3600-5400 and 5400-7200 s."""
    return (float(separation[5400.0] - separation[3600.0]),
            float(separation[7200.0] - separation[5400.0]))


def right_of_shear_distance(start_xy, end_xy, shear_uv) -> float:
    """Signed displacement (m) along the right normal of ``shear_uv``."""
    du, dv = np.asarray(shear_uv, dtype=np.float64)
    magnitude = float(np.hypot(du, dv))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return float("nan")
    dx, dy = np.asarray(end_xy, dtype=np.float64) - np.asarray(
        start_xy, dtype=np.float64)
    return float((dx * dv - dy * du) / magnitude)


def _movers(w2d, dx, dy, shear_uv, threshold=CORE_W):
    """Split diagnostics from a plan-view w slice (ny, nx).

    Returns the number of cores plus x/y centroids of the right and left
    members among the two strongest.  Right/left are classified by each
    centroid's projection on the right normal of ``shear_uv`` rather than
    by a hard-coded south/north proxy.
    """
    clusters = _clusters(w2d > threshold)
    if not clusters:
        return 0, np.nan, np.nan, np.nan, np.nan
    stats = []
    for rows, cols in clusters:
        wgt = w2d[rows, cols]
        stats.append((float(wgt.max()),
                      float((cols * wgt).sum() / wgt.sum()) * dx,
                      float((rows * wgt).sum() / wgt.sum()) * dy))
    stats.sort(reverse=True)               # strongest cores first
    if len(stats) < 2:
        return len(stats), np.nan, np.nan, np.nan, np.nan
    du, dv = np.asarray(shear_uv, dtype=np.float64)
    right_normal = np.array([dv, -du], dtype=np.float64)
    pair = sorted(stats[:2], key=lambda item:
                  item[1] * right_normal[0] + item[2] * right_normal[1],
                  reverse=True)
    _, xr, yr = pair[0]
    _, xl, yl = pair[1]
    return len(stats), xr, yr, xl, yl


def _uh_proxy(state, cfg, z_half):
    """UH proxy max(int w*zeta dz) over 2-5 km, float64 (ny, nx) max.

    zeta = dv/dx - du/dy with the staggered winds averaged to mass points
    and centered differences (open boundaries: one-sided edges via
    np.gradient); w averaged to half levels; the eta-layer depths come
    from the base-state half-level heights.
    """
    import cupy as cp

    sel = (z_half >= UH_ZBOT) & (z_half <= UH_ZTOP)
    dz = np.gradient(z_half)[sel]
    u = cp.asnumpy(state.u[sel]).astype(np.float64)
    v = cp.asnumpy(state.v[sel]).astype(np.float64)
    wf = cp.asnumpy(state.w).astype(np.float64)
    wh = 0.5 * (wf[:-1] + wf[1:])[sel]
    um = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    vm = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    zeta = (np.gradient(vm, cfg.dx, axis=2)
            - np.gradient(um, cfg.dy, axis=1))
    uh = np.sum(wh * zeta * dz[:, None, None], axis=0)
    return float(uh.max())


def _uh_proxy_arrays(u, v, w, cfg, z_half):
    """UH proxy on explicit host fields, retaining the original formula."""
    sel = (z_half >= UH_ZBOT) & (z_half <= UH_ZTOP)
    dz = np.gradient(z_half)[sel]
    u = np.asarray(u, dtype=np.float64)[sel]
    v = np.asarray(v, dtype=np.float64)[sel]
    wf = np.asarray(w, dtype=np.float64)
    wh = 0.5 * (wf[:-1] + wf[1:])[sel]
    um = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    vm = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    zeta = (np.gradient(vm, cfg.dx, axis=2)
            - np.gradient(um, cfg.dy, axis=1))
    uh = np.sum(wh * zeta * dz[:, None, None], axis=0)
    return float(uh.max())


def _wk82_diagnostic_levels(z_half):
    """Return source-selected 2--5 km, flow 0--6 km, and nearest zeta5 levels."""
    z_half = np.asarray(z_half, dtype=np.float64)
    uh = np.flatnonzero((z_half >= UH_ZBOT) & (z_half <= UH_ZTOP))
    flow = np.flatnonzero((z_half >= 0.0) & (z_half <= 6000.0))
    if uh.size == 0 or flow.size == 0:
        raise RuntimeError("WK82 tracking requires source 2-5 km and 0-6 km levels")
    zeta5 = int(np.argmin(np.abs(z_half - CORE_Z)))
    if zeta5 not in uh:
        raise RuntimeError("nearest mass level to 5 km is outside source UH levels")
    return uh, flow, zeta5


def _wk82_tracking_planes(u_window, v_window, w_window, *, k_start,
                          w_start, kw5, uh_levels, flow_levels, zeta5_level,
                          cfg, z_half):
    """Compute source-exact signed diagnostic planes from host array slices.

    The callers copy only the vertical range needed for the 2--5 km UH and
    0--6 km steering flow. Promotion happens before horizontal C-grid
    averaging and differentiation, as in ``_uh_proxy_arrays``.
    """
    u = np.asarray(u_window, dtype=np.float64)
    v = np.asarray(v_window, dtype=np.float64)
    w = np.asarray(w_window, dtype=np.float64)
    um = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    vm = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    uh_local = uh_levels - k_start
    flow_local = flow_levels - k_start
    zeta_local = zeta5_level - k_start
    zeta_uh = (np.gradient(vm[uh_local], cfg.dx, axis=2)
               - np.gradient(um[uh_local], cfg.dy, axis=1))
    wuh_local = uh_levels - w_start
    wbar_uh = 0.5 * (w[wuh_local] + w[wuh_local + 1])
    dz = np.gradient(np.asarray(z_half, dtype=np.float64))[uh_levels]
    uh2d = np.sum(wbar_uh * zeta_uh * dz[:, None, None], axis=0)
    zeta5 = (np.gradient(vm[zeta_local], cfg.dx, axis=1)
             - np.gradient(um[zeta_local], cfg.dy, axis=0))
    z_flow = np.asarray(z_half, dtype=np.float64)[flow_levels]
    depth = float(z_flow[-1] - z_flow[0])
    flow_dz = np.diff(z_flow)[:, None, None]
    flow_u = np.sum(0.5 * (um[flow_local[:-1]] + um[flow_local[1:]])
                    * flow_dz, axis=0) / depth
    flow_v = np.sum(0.5 * (vm[flow_local[:-1]] + vm[flow_local[1:]])
                    * flow_dz, axis=0) / depth
    w5 = w[kw5 - w_start]
    return {"w5": w5, "zeta5": zeta5, "uh2d": uh2d,
            "flow_u": flow_u, "flow_v": flow_v}


def _wk82_tracking_window(uh_levels, flow_levels, kw5):
    """Contiguous vertical slices sufficient for all 60-second diagnostics."""
    k_start = int(min(uh_levels[0], flow_levels[0]))
    k_stop = int(max(uh_levels[-1], flow_levels[-1])) + 1
    w_start = int(min(uh_levels[0], kw5))
    w_stop = int(max(uh_levels[-1] + 1, kw5)) + 1
    return k_start, k_stop, w_start, w_stop


class _WK82DiagnosticArchive:
    """Write each compact frame to a closed NPZ for live CPU replay."""
    def __init__(self, path, shape, *, dx, dy, shear_uv, uh_levels,
                 flow_levels, zeta5_level, kw5, z_half):
        from pathlib import Path

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._shape = tuple(shape)
        self._dx, self._dy = float(dx), float(dy)
        self._shear_uv = tuple(float(v) for v in shear_uv)
        self._uh_levels = tuple(int(v) for v in uh_levels)
        self._flow_levels = tuple(int(v) for v in flow_levels)
        self._zeta5_level, self._kw5 = int(zeta5_level), int(kw5)
        self._z_half = tuple(float(v) for v in np.asarray(z_half, dtype=np.float64))
        self._write_metadata()

    def write(self, time_s, planes):
        import os

        t = int(round(time_s))
        arrays = {name: np.asarray(
            plane, dtype=np.float32 if name == "w5" else np.float64)
                  for name, plane in planes.items()}
        final = self.path / f"frame_t{t:06d}.npz"
        temporary = self.path / f".frame_t{t:06d}.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, time_s=np.asarray(t, dtype=np.int32),
                                **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final)

    def _write_metadata(self):
        import json

        metadata = {
            "diagnostic_version": "bounded_persistent_tracks_60s_v4",
            "cadence_s": 60,
            "shape": self._shape,
            "dx_m": self._dx,
            "dy_m": self._dy,
            "shear_uv_m_s": self._shear_uv,
            "source_uh_levels": self._uh_levels,
            "flow_levels_0_6km": self._flow_levels,
            "zeta5_mass_level": self._zeta5_level,
            "w5_full_level": self._kw5,
            "z_half_m": self._z_half,
            "planes": {
                "w5": "vertical velocity at original source ~5 km full level (FP32)",
                "zeta5": "signed d(v)/dx-d(u)/dy at nearest mass level to 5 km (s-1)",
                "uh2d": "source 2-5 km integral w*zeta*dz (m2 s-2)",
                "flow_u": "mass-grid 0-6 km layer-mean u (m s-1)",
                "flow_v": "mass-grid 0-6 km layer-mean v (m s-1)",
            },
            "derived_plane_precision": "FP64 after source FP32 fields were promoted before averaging/differentiation",
        }
        (self.path / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n")

    def close(self):
        pass


class _WK82TraceStream:
    """Flush one JSONL status record after every tracker update."""
    def __init__(self, path):
        self._file = open(path, "w", encoding="utf-8")

    def write(self, result):
        import json

        self._file.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def _snapshot_metrics(metrics, snaps, shear_uv, cfg):
    """Compute the canonical WOOF v1.0.3 instantaneous mover measurements."""
    for t, snap in snaps.items():
        n_cores, x_right, y_right, x_left, y_left = _movers(
            snap["w5"], cfg.dx, cfg.dy, shear_uv)
        tag = str(int(t))
        metrics["n_cores_" + tag] = n_cores
        metrics["x_right_" + tag] = x_right
        metrics["y_right_" + tag] = y_right
        metrics["x_left_" + tag] = x_left
        metrics["y_left_" + tag] = y_left
        metrics["sep_" + tag] = y_left - y_right
    metrics["right_dy"] = metrics["y_right_7200"] - metrics["y_right_3600"]
    metrics["left_dy"] = metrics["y_left_7200"] - metrics["y_left_3600"]
    growth_1, growth_2 = separation_growth_intervals({
        t: metrics["sep_" + str(int(t))] for t in ANALYSIS_TIMES})
    metrics["sep_growth_3600_5400"] = growth_1
    metrics["sep_growth_5400_7200"] = growth_2
    metrics["sep_growth"] = metrics["sep_7200"] - metrics["sep_3600"]
    metrics["right_rel_dy"] = metrics["right_dy"] - metrics["left_dy"]
    metrics["right_of_shear_m"] = right_of_shear_distance(
        (metrics["x_right_3600"], metrics["y_right_3600"]),
        (metrics["x_right_7200"], metrics["y_right_7200"]), shear_uv)
    return metrics


def _persistent_tracking_metrics(destination, tracker, tracker_at_analysis,
                                 shear_uv, outdir):
    """Report conservative persistent-identity diagnostics without gating."""
    metrics = {}
    metrics["diagnostic_version"] = "bounded_persistent_tracks_60s_v4"
    metrics["tracking_status"] = "valid"
    metrics["tracking_confidence"] = {}
    metrics["tracking_trace"] = tracker.trace
    invalid = []
    analysis_paths = {}
    for t in ANALYSIS_TIMES:
        tag = str(int(t))
        result = tracker_at_analysis.get(float(t))
        if result is None or result.get("status") not in ("seeded", "matched", "bounded"):
            invalid.append({"time_s": t,
                            "status": "missing" if result is None else result.get("status")})
            for name in ("x_right", "y_right", "x_left", "y_left", "sep", "sep_upper"):
                metrics[f"{name}_{tag}"] = float("nan")
            metrics["tracking_confidence"][tag] = "unavailable"
            continue
        paths = result.get("hypotheses") or [result]
        analysis_paths[t] = paths
        if len(paths) > 1:
            metrics["tracking_status"] = "bounded"
        for side in ("right", "left"):
            for axis in ("x", "y"):
                values = [float(p["members"][side][f"{axis}_m"]) for p in paths]
                lo, hi = min(values), max(values)
                metrics[f"{axis}_{side}_{tag}"] = lo if lo == hi else float("nan")
                metrics[f"{axis}_{side}_min_{tag}"] = lo
                metrics[f"{axis}_{side}_max_{tag}"] = hi
        separations = [float(p["separation_m"]) for p in paths]
        metrics[f"sep_{tag}"] = min(separations)
        metrics[f"sep_upper_{tag}"] = max(separations)
        metrics["tracking_confidence"][tag] = {
            "status": result["status"],
            "hypothesis_count": len(paths),
            "members": [p["members"] for p in paths],
        }
    if invalid:
        metrics["tracking_status"] = "failed_closed"
        metrics["tracking_invalid_analysis_times"] = invalid
    # Report the least favorable admissible realization of each diagnostic.
    # Subtracting endpoint interval bounds is conservative even when paths
    # share ancestry; it never selects a favorable child after a split.
    for start, end in ((3600, 5400), (5400, 7200), (3600, 7200)):
        suffix = "" if (start, end) == (3600, 7200) else f"_{start}_{end}"
        metrics[f"sep_growth{suffix}"] = (metrics[f"sep_{end}"]
                                           - metrics[f"sep_upper_{start}"])
        metrics[f"sep_growth_upper{suffix}"] = (metrics[f"sep_upper_{end}"]
                                                 - metrics[f"sep_{start}"])
    for side in ("right", "left"):
        metrics[f"{side}_dy"] = (metrics[f"y_{side}_7200"]
                                   - metrics[f"y_{side}_3600"])
    metrics["right_rel_dy"] = metrics["right_dy"] - metrics["left_dy"]
    motion = []
    for start in analysis_paths.get(3600.0, []):
        for end in analysis_paths.get(7200.0, []):
            a, b = start["members"]["right"], end["members"]["right"]
            motion.append(right_of_shear_distance(
                (a["x_m"], a["y_m"]), (b["x_m"], b["y_m"]), shear_uv))
    metrics["right_of_shear_m"] = min(motion) if motion else float("nan")
    if outdir is not None:
        import json
        from pathlib import Path

        Path(outdir, "wk82_tracker_trace.json").write_text(
            json.dumps(tracker.trace, indent=2, sort_keys=True, allow_nan=False) + "\n")
    destination.update({f"tracked_{key.removeprefix('tracking_')}": value
                        for key, value in metrics.items()})
    return destination


def _tracking_metrics(metrics, tracker, tracker_at_analysis, snaps,
                      shear_uv, cfg, outdir):
    """Preserve canonical snapshot qualification and add tracked diagnostics."""
    _snapshot_metrics(metrics, snaps, shear_uv, cfg)
    return _persistent_tracking_metrics(metrics, tracker, tracker_at_analysis,
                                        shear_uv, outdir)


def _plot_planviews(snaps, key, x, y, path, label, cmap, vmin, vmax):
    """One row of plan views (t = 3600/5400/7200 s) of ``snaps[t][key]``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(snaps), figsize=(15.0, 4.8),
                             sharey=True)
    xk, yk = x / 1000.0, y / 1000.0
    for ax, (t, snap) in zip(np.atleast_1d(axes), sorted(snaps.items())):
        pm = ax.pcolormesh(xk, yk, snap[key], cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"t = {int(t)} s")
        ax.set_xlabel("x (km)")
        ax.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel("y (km)")
    fig.colorbar(pm, ax=list(np.atleast_1d(axes)), label=label,
                 shrink=0.85)
    fig.suptitle(f"wk82: {label}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_hodograph(coord, ztop, z_half, path):
    """Quarter-circle hodograph: model-level winds + the shipped columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u, v = hodograph_uv(coord, ztop)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot(HODOGRAPH_ZUV[:-1, 1], HODOGRAPH_ZUV[:-1, 2], "-", color="0.7",
            label="input_sounding columns")
    sel = z_half <= 12000.0
    sc = ax.scatter(u[sel], v[sel], c=z_half[sel] / 1000.0, cmap="viridis",
                    s=18, zorder=3, label="model half levels")
    for zz in (1000.0, 3000.0, 6000.0):
        k = int(np.argmin(np.abs(z_half - zz)))
        ax.annotate(f"{zz / 1000:.0f} km", (u[k], v[k]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0.0, color="0.85", lw=0.8)
    ax.axvline(0.0, color="0.85", lw=0.8)
    ax.set_xlabel("u (m/s)")
    ax.set_ylabel("v (m/s)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=ax, label="z (km)")
    ax.set_title("wk82 quarter-circle hodograph (storm-motion frame)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_cuda(outdir=None) -> dict:
    """Build the WK82 supercell case, integrate 2 h, return the metrics.

    Metrics (plan Task 11): ``w_max_hist`` (per-step (t, max w) history),
    ``w_max`` (peak after 1800 s spin-up), ``w_sustained_fraction`` (the
    fraction of t = [3600, 7200] samples above 25 m/s),
    ``qr_max`` (kg/kg, peak over the run), split diagnostics at
    t = 3600/5400/7200 s (canonical instantaneous ``n_cores_T`` and
    strongest-two right/left centroids, with persistent opposite-sign
    identities reported separately under ``tracked_*``), the
    ``uh_proxy`` max(int w*zeta dz, 2-5 km) at the final time, ``nan``,
    and relative dry-mass drift ``mass_drift`` (FP64) plus the maximum
    relative closure residual against an independently integrated lateral
    boundary flux.  Separation is gated at 3600 s, across the early
    3600->5400 s interval, and in total from 3600->7200 s; the later
    interval remains informational.  Right-mover displacement is projected
    against the 0-6 km mean-shear vector.  With
    an ``outdir``, writes the three PNGs (wk82_w_z5km/wk82_qr_z1km plan
    views at the analysis times, and the hodograph), ``wrfout_wk82.nc``
    (a frame every ``cfg.output_interval_s``), compressed diagnostic planes
    every 60 s, and a flushed JSONL tracking trace.
    """
    import cupy as cp
    from pathlib import Path

    from woof.core.dycore import stability_report, step
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.config import validate_run_config

    # This idealized case has no PBL, so km_opt=4 selects the ported WRF
    # vertical_diffusion_2 stresses and surface-flux branch in addition to
    # horizontal Smagorinsky diffusion.
    cfg = validate_run_config(default_config())
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = build(cfg, coord, base)

    z_half = state.height_half()
    kw5 = int(np.argmin(np.abs(base.phb / c.G - CORE_Z)))  # w level ~ 5 km
    kq1 = int(np.argmin(np.abs(z_half - 1000.0)))          # half level ~ 1 km
    from woof.verify.wk82_tracking import WK82StormTracker
    uh_levels, flow_levels, zeta5_level = _wk82_diagnostic_levels(z_half)
    u_start, u_stop, w_start, w_stop = _wk82_tracking_window(
        uh_levels, flow_levels, kw5)
    u_hod, v_hod = hodograph_uv(coord, cfg.ztop)
    k0 = int(np.argmin(np.abs(z_half)))
    k6 = int(np.argmin(np.abs(z_half - 6000.0)))
    shear_uv = (u_hod[k6] - u_hod[k0], v_hod[k6] - v_hod[k0])
    tracker = WK82StormTracker(cfg.dx, cfg.dy, shear_uv)

    def dry_mass():
        return float(cp.sum((state.mub2d + state.mup).astype(cp.float64)))

    writer = None
    diagnostic_archive = None
    trace_stream = None
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        from woof.io.wrfout import WrfoutWriter, state_frame, wrf_time_str
        writer = WrfoutWriter(outdir / "wrfout_wk82.nc", nx=cfg.nx,
                              ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                              title="woof WK82 quarter-circle supercell")
        rainnc = state.scratch((cfg.ny, cfg.nx), "mp_rainnc")
        diagnostic_archive = _WK82DiagnosticArchive(
            outdir / "wk82_diagnostics_60s", (cfg.ny, cfg.nx), dx=cfg.dx,
            dy=cfg.dy, shear_uv=shear_uv, uh_levels=uh_levels,
            flow_levels=flow_levels, zeta5_level=zeta5_level, kw5=kw5,
            z_half=z_half)
        trace_stream = _WK82TraceStream(outdir / "wk82_tracker_trace.jsonl")

    def write_frame(t_s: float) -> None:
        writer.write_frame(wrf_time_str(t_s), state_frame(state)
                           | {"RAINNC": cp.asnumpy(rainnc)})

    mass0 = dry_mass()
    mass_drift_max = 0.0
    boundary_flux_integral = 0.0
    mass_closure_residual_max = 0.0
    n_total = int(round(cfg.run_seconds / cfg.dt))
    n_frame = max(int(round(cfg.output_interval_s / cfg.dt)), 1)
    snap_steps = {int(round(t / cfg.dt)): t for t in ANALYSIS_TIMES}
    tracking_every = max(int(round(TRACKING_DIAGNOSTIC_INTERVAL_S / cfg.dt)), 1)
    mass_steps = set(mass_sample_steps(n_total))       # T11-1: 10-step cadence

    w_hist = []
    qr_max = 0.0
    snaps = {}
    tracker_at_analysis = {}

    def accumulate_boundary_flux(increment: float) -> None:
        nonlocal boundary_flux_integral
        boundary_flux_integral += float(increment)

    try:
        if writer is not None:
            write_frame(0.0)
        for n in range(1, n_total + 1):
            step(state, cfg, mass_flux_observer=accumulate_boundary_flux)
            t = n * cfg.dt
            w_hist.append((t, float(state.w.max())))
            qr_max = max(qr_max, float(state.qr.max()))
            if n in mass_steps:
                mass_now = dry_mass()
                mass_drift_max = max(
                    mass_drift_max, abs(mass_now - mass0) / mass0)
                mass_closure_residual_max = max(
                    mass_closure_residual_max,
                    abs((mass_now - mass0) - boundary_flux_integral) / mass0)
            if n % tracking_every == 0:
                u_host = cp.asnumpy(state.u[u_start:u_stop]).astype(np.float64)
                v_host = cp.asnumpy(state.v[u_start:u_stop]).astype(np.float64)
                w_host = cp.asnumpy(state.w[w_start:w_stop]).astype(np.float64)
                planes = _wk82_tracking_planes(
                    u_host, v_host, w_host, k_start=u_start, w_start=w_start,
                    kw5=kw5, uh_levels=uh_levels, flow_levels=flow_levels,
                    zeta5_level=zeta5_level, cfg=cfg, z_half=z_half)
                if diagnostic_archive is not None:
                    diagnostic_archive.write(t, planes)
                tracker_result = tracker.update(
                    t, planes["w5"], planes["zeta5"], planes["uh2d"],
                    planes["flow_u"], planes["flow_v"])
                if trace_stream is not None:
                    trace_stream.write(tracker_result)
                if n in snap_steps:
                    snaps[snap_steps[n]] = {
                        "w5": planes["w5"].astype(np.float64),
                        "qr1": 1000.0 * cp.asnumpy(state.qr[kq1]).astype(np.float64),
                    }
                    tracker_at_analysis[snap_steps[n]] = tracker_result
            if writer is not None and n % n_frame == 0:
                write_frame(t)
    finally:
        if diagnostic_archive is not None:
            diagnostic_archive.close()
        if trace_stream is not None:
            trace_stream.close()
        if writer is not None:
            writer.close()

    report = stability_report(state, cfg)
    finite = all(bool(cp.isfinite(q).all()) for q in
                 (state.qv, state.qc, state.qr))
    mass1 = dry_mass()
    mass_drift_max = max(mass_drift_max, abs(mass1 - mass0) / mass0)
    mass_closure_residual_max = max(
        mass_closure_residual_max,
        abs((mass1 - mass0) - boundary_flux_integral) / mass0)

    th = np.asarray(w_hist, dtype=np.float64)
    after = th[th[:, 0] > 1800.0]
    sustained_fraction = sustained_updraft_fraction(th)

    metrics = {
        "nan": bool(report["nan"] or not finite),
        "w_max": float(after[:, 1].max()),
        "w_max_sustained": float(th[(th[:, 0] >= 3600.0), 1].min()),
        "w_sustained_fraction": sustained_fraction,
        "w_max_hist": [tuple(row) for row in th],
        "qr_max": qr_max,
        "uh_proxy": _uh_proxy(state, cfg, z_half),
        "mass_drift": abs(mass1 - mass0) / mass0,
        "mass_drift_max": mass_drift_max,
        "mass_closure_residual_max": mass_closure_residual_max,
        "boundary_flux_exchange": abs(boundary_flux_integral) / mass0,
    }
    metrics = _tracking_metrics(metrics, tracker, tracker_at_analysis, snaps,
                                shear_uv, cfg, outdir)
    if outdir is not None:
        x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
        y = (np.arange(cfg.ny) + 0.5) * cfg.dy - 0.5 * cfg.ny * cfg.dy
        wmax = max(np.abs(s["w5"]).max() for s in snaps.values()) or 1.0
        _plot_planviews(snaps, "w5", x, y, outdir / "wk82_w_z5km.png",
                        "w at z = 5 km (m/s)", "RdBu_r", -wmax, wmax)
        qmax = max(s["qr1"].max() for s in snaps.values()) or 0.1
        _plot_planviews(snaps, "qr1", x, y, outdir / "wk82_qr_z1km.png",
                        "qr at z = 1 km (g/kg)", "Greens", 0.0, qmax)
        _plot_hodograph(coord, cfg.ztop, z_half, outdir / "wk82_hodograph.png")
    return metrics


def _dry_mass_metal(state) -> float:
    """FP32 column addition followed by the existing FP64 dry-mass sum."""
    from woof.verify.metal_adapter import host_state

    fields = host_state(state, ("mub2d", "mup"))
    mass = (np.asarray(fields.mub2d, dtype=np.float32)
            + np.asarray(fields.mup, dtype=np.float32))
    return float(np.sum(mass.astype(np.float64), dtype=np.float64))


def _run_metal(outdir=None) -> dict:
    """Run WK82 via the native Metal DomainState and acoustic-flux observer."""
    from pathlib import Path

    from woof.config import validate_run_config
    from woof.core.dycore import stability_report, step
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.verify.metal_adapter import CaseBackend, as_host

    cfg = validate_run_config(default_config())
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    out_path = Path(outdir) if outdir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    with CaseBackend("metal") as backend:
        state = build(cfg, coord, base, array_module=backend.arrays)
        z_half = as_host(state.height_half())
        kw5 = int(np.argmin(np.abs(base.phb / c.G - CORE_Z)))
        kq1 = int(np.argmin(np.abs(z_half - 1000.0)))
        from woof.verify.wk82_tracking import WK82StormTracker
        uh_levels, flow_levels, zeta5_level = _wk82_diagnostic_levels(z_half)
        u_start, u_stop, w_start, w_stop = _wk82_tracking_window(
            uh_levels, flow_levels, kw5)
        u_hod, v_hod = hodograph_uv(coord, cfg.ztop)
        k0 = int(np.argmin(np.abs(z_half)))
        k6 = int(np.argmin(np.abs(z_half - 6000.0)))
        shear_uv = (u_hod[k6] - u_hod[k0], v_hod[k6] - v_hod[k0])
        tracker = WK82StormTracker(cfg.dx, cfg.dy, shear_uv)
        writer = None
        diagnostic_archive = None
        trace_stream = None
        rainnc = None
        if out_path is not None:
            from woof.io.wrfout import WrfoutWriter, state_frame, wrf_time_str

            rainnc = state.scratch((cfg.ny, cfg.nx), "mp_rainnc")
            writer = WrfoutWriter(out_path / "wrfout_wk82.nc", nx=cfg.nx,
                                  ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                                  title="woof WK82 quarter-circle supercell")
            diagnostic_archive = _WK82DiagnosticArchive(
                out_path / "wk82_diagnostics_60s", (cfg.ny, cfg.nx),
                dx=cfg.dx, dy=cfg.dy, shear_uv=shear_uv,
                uh_levels=uh_levels, flow_levels=flow_levels,
                zeta5_level=zeta5_level, kw5=kw5, z_half=z_half)
            trace_stream = _WK82TraceStream(out_path / "wk82_tracker_trace.jsonl")

        def write_frame(t_s: float) -> None:
            writer.write_frame(wrf_time_str(t_s), state_frame(state)
                               | {"RAINNC": as_host(rainnc)})

        mass0 = _dry_mass_metal(state)
        mass_drift_max = 0.0
        boundary_flux_integral = 0.0
        mass_closure_residual_max = 0.0
        n_total = int(round(cfg.run_seconds / cfg.dt))
        n_frame = max(int(round(cfg.output_interval_s / cfg.dt)), 1)
        snap_steps = {int(round(t / cfg.dt)): t for t in ANALYSIS_TIMES}
        tracking_every = max(int(round(TRACKING_DIAGNOSTIC_INTERVAL_S / cfg.dt)), 1)
        mass_steps = set(mass_sample_steps(n_total))
        w_hist = []
        qr_max = 0.0
        snaps = {}
        tracker_at_analysis = {}

        def accumulate_boundary_flux(increment: float) -> None:
            nonlocal boundary_flux_integral
            boundary_flux_integral += float(increment)

        try:
            if writer is not None:
                write_frame(0.0)
            for n in range(1, n_total + 1):
                step(state, cfg, mass_flux_observer=accumulate_boundary_flux)
                t = n * cfg.dt
                w = as_host(state.w)
                qr = as_host(state.qr)
                w_hist.append((t, float(w.max())))
                qr_max = max(qr_max, float(qr.max()))
                if n in mass_steps:
                    mass_now = _dry_mass_metal(state)
                    mass_drift_max = max(
                        mass_drift_max, abs(mass_now - mass0) / mass0)
                    mass_closure_residual_max = max(
                        mass_closure_residual_max,
                        abs((mass_now - mass0) - boundary_flux_integral) / mass0)
                if n % tracking_every == 0:
                    u_host = as_host(state.u[u_start:u_stop]).astype(np.float64)
                    v_host = as_host(state.v[u_start:u_stop]).astype(np.float64)
                    w_host = as_host(state.w[w_start:w_stop]).astype(np.float64)
                    planes = _wk82_tracking_planes(
                        u_host, v_host, w_host, k_start=u_start, w_start=w_start,
                        kw5=kw5, uh_levels=uh_levels, flow_levels=flow_levels,
                        zeta5_level=zeta5_level, cfg=cfg, z_half=z_half)
                    if diagnostic_archive is not None:
                        diagnostic_archive.write(t, planes)
                    tracker_result = tracker.update(
                        t, planes["w5"], planes["zeta5"], planes["uh2d"],
                        planes["flow_u"], planes["flow_v"])
                    if trace_stream is not None:
                        trace_stream.write(tracker_result)
                    if n in snap_steps:
                        snaps[snap_steps[n]] = {
                            "w5": planes["w5"].astype(np.float64),
                            "qr1": 1000.0 * qr[kq1].astype(np.float64),
                        }
                        tracker_at_analysis[snap_steps[n]] = tracker_result
                if writer is not None and n % n_frame == 0:
                    write_frame(t)
        finally:
            if diagnostic_archive is not None:
                diagnostic_archive.close()
            if trace_stream is not None:
                trace_stream.close()
            if writer is not None:
                writer.close()

        report = stability_report(state, cfg)
        finite = all(np.isfinite(as_host(q)).all()
                     for q in (state.qv, state.qc, state.qr))
        mass1 = _dry_mass_metal(state)
        mass_drift_max = max(mass_drift_max, abs(mass1 - mass0) / mass0)
        mass_closure_residual_max = max(
            mass_closure_residual_max,
            abs((mass1 - mass0) - boundary_flux_integral) / mass0)

        history = np.asarray(w_hist, dtype=np.float64)
        after = history[history[:, 0] > 1800.0]
        metrics = {
            "nan": bool(report["nan"] or not finite),
            "w_max": float(after[:, 1].max()),
            "w_max_sustained": float(history[history[:, 0] >= 3600.0, 1].min()),
            "w_sustained_fraction": sustained_updraft_fraction(history),
            "w_max_hist": [tuple(row) for row in history],
            "qr_max": qr_max,
            "uh_proxy": _uh_proxy_arrays(as_host(state.u), as_host(state.v),
                                          as_host(state.w), cfg, z_half),
            "mass_drift": abs(mass1 - mass0) / mass0,
            "mass_drift_max": mass_drift_max,
            "mass_closure_residual_max": mass_closure_residual_max,
            "boundary_flux_exchange": abs(boundary_flux_integral) / mass0,
        }
        metrics = _tracking_metrics(metrics, tracker, tracker_at_analysis,
                                    snaps, shear_uv, cfg, out_path)
        if out_path is not None:
            x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
            y = (np.arange(cfg.ny) + 0.5) * cfg.dy - 0.5 * cfg.ny * cfg.dy
            wmax = max(np.abs(s["w5"]).max() for s in snaps.values()) or 1.0
            _plot_planviews(snaps, "w5", x, y,
                            out_path / "wk82_w_z5km.png",
                            "w at z = 5 km (m/s)", "RdBu_r", -wmax, wmax)
            qmax = max(s["qr1"].max() for s in snaps.values()) or 0.1
            _plot_planviews(snaps, "qr1", x, y,
                            out_path / "wk82_qr_z1km.png",
                            "qr at z = 1 km (g/kg)", "Greens", 0.0, qmax)
            _plot_hodograph(coord, cfg.ztop, z_half,
                            out_path / "wk82_hodograph.png")
    return metrics


def run(outdir=None, *, backend: str = "cuda") -> dict:
    """Run WK82 with CUDA by default or the native Metal backend."""
    if backend == "cuda":
        return _run_cuda(outdir)
    if backend == "metal":
        return _run_metal(outdir)
    raise ValueError(f"unknown wk82 backend: {backend!r}")
