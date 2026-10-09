"""The vertical-correlation instrument the per-class localisation cutoffs are derived from.

A Gaspari-Cohn cutoff in ln p is a statement about how far a report's
information should reach up and down a column, and the only thing that
can say how far it does reach is the ensemble the filter runs on: the
correlation, across the members, between what the report sees and the
analysed field at every level of the same column.  This module reads that
correlation off a member set (:func:`member_column_fields` from the
member checkpoints on the host, :func:`vertical_correlation_profiles`
across them) and fits the Gaspari-Cohn support to it
(:func:`fit_gaspari_cohn_cutoff`), one cutoff per report class
(:data:`REPORT_CLASSES`), by region and globally.  The door leg
``woof global da localisation`` runs it on an ensemble store and writes
the derivation; the values in :class:`~woof.globe.da.options.FilterOptions`
are the derivation of record.

The signal a sample correlation carries is read as the noise-corrected
rms correlation ``sqrt(max(mean(corr^2) - 1/(N-1), 0))`` over the region's
columns (the sampling floor of ``corr^2`` for ``N`` members is ``1/(N-1)``;
a mean of ``corr`` itself averages a sign-changing structure to nothing),
normalised by its value at zero separation, and the cutoff is the ``2c``
whose ``GC(d / c)`` fits that normalised profile in least squares over the
separations the columns span.  A class that reaches several target fields
(a surface-pressure report reaches temperature and wind at every level)
takes the mean of its targets' signal profiles.

Why the derivation exists: the record of 2026-09-06 (32 T127 members,
independent temperature, pressure and wind draws) carried pressure-to-column
correlations of 0.1 to 0.3 with no structure a physical balance would give
them, and a surface-pressure report with no vertical localisation moved the
250 hPa wind through them; the balanced family of the same date gives the
correlations a shape, and the shape is what the cutoffs are fitted to.

Why a mass source reaches the wind through the STREAMFUNCTION: the
correlation at one column between a mass field and a wind component is
zero under any balance (the wind is the derivative of the mass field, so
the two are in quadrature), and the first reading of this instrument on a
balanced 16-member ensemble fitted a 6.0 ln p cutoff to a pressure-to-wind
profile that was flat noise at 0.13 to 0.18 from the surface to the model
top; read through the vorticity it was flat at 0.16 to 0.22, because the
point vorticity is weighted to the small scales of the spectrum and the
geopotential to the large ones (laplacian(phi') = f zeta'), so the two
correlate weakly at one point under any spectrum of scales.  The
streamfunction is the wind field weighted as the mass field is (phi' = f
psi' under geostrophy), so the reach a mass report has into the wind of a
column is read as its reach into the column's streamfunction, and a wind
report's reach into the temperature the same way from its streamfunction;
a wind report's reach into the wind is its own component.  Every class
therefore names the (source, target) field pairs it is read from.

The second instrument here is the radiance's: a report that senses a
LAYER rather than a level (an ATMS or ABI channel) carries a vertical
localisation PROFILE instead of a cutoff, the channel's weighting
function convolved with the Gaspari-Cohn kernel of a vertical length the
ensemble measured at that window (:func:`profile_from_weighting_function`,
:func:`vertical_correlation_length`, :func:`cutoff_from_half_width`),
tabulated on :data:`~woof.globe.da.observations.LOCALISATION_AXIS_LNP`
and read by :func:`woof.globe.da.letkf_point.profile_weight` at
every level of every column.  Placing a radiance at its weighting
function's centroid with a wide cutoff let a stratospheric channel update
the boundary layer through the kernel's tail (Campbell, Bishop and Hodyss
2010 name the failure); the profile is the model-space form.  Its length
bounds are the radiance stream's own (:data:`RADIANCE_CUTOFF_BOUNDS_LNP`),
narrower than the fit bounds of the point classes above.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from woof.da.letkf import gaspari_cohn

from ..constants import EARTH_RADIUS_M, KAPPA, REFERENCE_PRESSURE_PA
from .observations import LOCALISATION_AXIS_LNP

#: The regions the profiles are read in: (south, north) in degrees.
REGIONS = {
    "global": (-90.0, 90.0),
    "nh_extratropics": (20.0, 90.0),
    "tropics": (-20.0, 20.0),
    "sh_extratropics": (-90.0, -20.0),
}

#: The report classes and what each reaches: ``level`` is where a report
#: of the class samples (``"surface"`` the lowest full level, pressures in
#: hPa levels aloft, averaged), ``pairs`` the (source field, target field)
#: correlations the class's profile is the mean of (``lnps`` the log
#: surface pressure, ``psi`` the streamfunction, the wind's own component
#: for wind-to-wind, the streamfunction for every mass-to-wind or
#: wind-to-mass pair), and ``option`` the :class:`FilterOptions` field the
#: fitted cutoff is.
REPORT_CLASSES = {
    "surface_pressure": {"level": "surface", "pairs": (("lnps", "t"), ("lnps", "psi")),
                         "option": "pressure_vertical_cutoff_lnp"},
    "surface_temperature": {"level": "surface", "pairs": (("t", "t"), ("t", "psi"), ("t", "q")),
                            "option": "surface_vertical_cutoff_lnp"},
    "surface_wind": {"level": "surface", "pairs": (("u", "u"), ("v", "v"), ("psi", "t")),
                     "option": "surface_wind_vertical_cutoff_lnp"},
    "aloft_temperature": {"level": (850.0, 500.0, 250.0), "pairs": (("t", "t"), ("t", "psi")),
                          "option": "vertical_cutoff_lnp"},
    "aloft_wind": {"level": (850.0, 500.0, 250.0), "pairs": (("u", "u"), ("v", "v"), ("psi", "t")),
                   "option": "aloft_wind_vertical_cutoff_lnp"},
    "aloft_humidity": {"level": (850.0, 700.0, 500.0), "pairs": (("q", "q"), ("q", "t")),
                       "option": "aloft_humidity_vertical_cutoff_lnp"},
}

#: The separations (ln p) the fit is taken over, and the cutoffs it may
#: return: a support below the first model layer or above the whole
#: column is not a localisation.
FIT_SEPARATION_MAX_LNP = 3.5
CUTOFF_BOUNDS_LNP = (0.05, 6.0)
#: The reference of the normalised profile is the largest signal within
#: this separation of the source (the report's own layer), so one noisy
#: level does not set the scale.
REFERENCE_SEPARATION_LNP = 0.25
#: A class whose correlation with every level sits inside the sampling
#: floor carries this support: the report reaches its own layer and no
#: more, and the fit says so by name (``no_signal_above_floor``).
NO_SIGNAL_CUTOFF_LNP = 0.25


def member_column_fields(paths, transform, vertical, *, precision=np.float32) -> dict[str, np.ndarray]:
    """The grid fields the instrument reads, from member checkpoints on the
    host: ``t (R, nlev, nlat, nlon)`` temperature, ``u``, ``v``, ``psi``
    (the streamfunction of the vorticity), ``q`` (vapor), ``lnps (R, nlat,
    nlon)``, ``lnpf (R, nlev, nlat, nlon)`` the full-level ln p.  ``transform`` is a numpy
    transform at the members' truncation, ``vertical`` the run's hybrid
    coordinate."""
    from woof.globe.spectral.vector import VorticityDivergenceOperator

    from ..checkpoint import read_checkpoint

    vector = VorticityDivergenceOperator(transform)
    out: dict[str, list] = {"t": [], "u": [], "v": [], "psi": [], "q": [], "lnps": [], "lnpf": []}
    for path in paths:
        _metadata, arrays = read_checkpoint(path)
        theta = np.asarray(transform.inverse(arrays["atmosphere__theta"].astype(np.complex128)), dtype=np.float64)
        qv = np.asarray(transform.inverse(arrays["atmosphere__qv"].astype(np.complex128)), dtype=np.float64)
        lnps = np.asarray(transform.inverse(arrays["atmosphere__log_surface_pressure"].astype(np.complex128)), dtype=np.float64)
        pressure = vertical.pressure(np.exp(lnps), transform.backend)
        p_full = np.asarray(pressure["p_full"], dtype=np.float64)
        u, v = vector.wind_from_vordiv(arrays["atmosphere__vorticity"].astype(np.complex128),
                                       arrays["atmosphere__divergence"].astype(np.complex128))
        psi = np.asarray(transform.inverse(transform.inverse_laplacian(
            arrays["atmosphere__vorticity"].astype(np.complex128))), dtype=np.float64)
        out["t"].append((theta * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA).astype(precision))
        out["u"].append(np.asarray(u, dtype=precision))
        out["v"].append(np.asarray(v, dtype=precision))
        out["psi"].append(psi.astype(precision))
        out["q"].append(qv.astype(precision))
        out["lnps"].append(lnps.astype(precision))
        out["lnpf"].append(np.log(p_full).astype(precision))
    return {name: np.stack(values) for name, values in out.items()}


def _correlation(a, b):
    """Member-axis correlation of ``a (R, ...)`` with ``b (R, ...)``."""
    pa = a - a.mean(axis=0, keepdims=True)
    pb = b - b.mean(axis=0, keepdims=True)
    num = (pa * pb).sum(axis=0)
    den = np.sqrt((pa ** 2).sum(axis=0) * (pb ** 2).sum(axis=0))
    return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)


def _nearest_level(mean_lnp_column: np.ndarray, pressure_hpa: float) -> int:
    return int(np.argmin(np.abs(mean_lnp_column - math.log(float(pressure_hpa) * 100.0))))


def vertical_correlation_profiles(fields: dict[str, np.ndarray], latitude_deg, quadrature_weights,
                                  *, classes=REPORT_CLASSES, regions=REGIONS) -> dict[str, object]:
    """Per report class and region, the noise-corrected signal correlation
    between the class's source field at its level and each target field
    at every level, the mean over the class's (source, target) pairs, with
    the ln p separation from the source: ``{class: {region: {"separation":
    [...], "signal": [...], "corr_mean": [...], "corr2_mean": [...]}}}``
    plus the members, the noise floor and the mean column.  ``fields``
    carries ``t``, ``u``, ``v``, ``psi``, ``q``, ``lnps`` and ``lnpf``
    (:func:`member_column_fields`)."""
    members = int(fields["t"].shape[0])
    if members < 3:
        raise ValueError("the correlation instrument needs at least 3 members")
    nlev = int(fields["t"].shape[1])
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    w = np.asarray(quadrature_weights, dtype=np.float64).reshape(-1)
    nlon = int(fields["t"].shape[-1])
    W = np.repeat(w[:, None], nlon, axis=1)
    LAT = np.repeat(lat[:, None], nlon, axis=1)
    lnpf_mean = fields["lnpf"].mean(axis=0)
    lnps_mean = fields["lnps"].mean(axis=0)
    floor = 1.0 / (members - 1)
    mean_column = np.array([float(np.sum(W * lnpf_mean[k]) / np.sum(W)) for k in range(nlev)])

    def region_mean(field2d, region):
        south, north = regions[region]
        mask = (LAT >= south) & (LAT <= north) & np.isfinite(field2d)
        return float(np.sum(W[mask] * field2d[mask]) / np.sum(W[mask]))

    def source_of(name, level):
        """The source field of one pair at the class's level and the ln p it sits at."""
        if name == "lnps":
            return fields["lnps"], lnps_mean
        k = nlev - 1 if level == "surface" else int(level)
        return fields[name][:, k], lnpf_mean[k]

    def levels_of(where):
        return ["surface"] if where == "surface" else [_nearest_level(mean_column, p) for p in where]

    out: dict[str, object] = {
        "members": members, "noise_floor_corr2": floor, "nlev": nlev,
        "mean_ln_p_full_by_level": [float(v) for v in mean_column],
        "mean_p_full_hpa_by_level": [float(math.exp(v) / 100.0) for v in mean_column],
        "classes": {},
    }
    for name, spec in classes.items():
        per_region: dict[str, object] = {}
        for region in regions:
            corr_acc = np.zeros(nlev)
            corr2_acc = np.zeros(nlev)
            sep_acc = np.zeros(nlev)
            count = 0
            for level in levels_of(spec["level"]):
                for source_name, target_name in spec["pairs"]:
                    source, source_lnp = source_of(source_name, level)
                    arr = fields[target_name]
                    for k in range(nlev):
                        c = _correlation(source, arr[:, k])
                        corr_acc[k] += region_mean(c, region)
                        corr2_acc[k] += region_mean(c ** 2, region)
                        sep_acc[k] += region_mean(np.abs(lnpf_mean[k] - source_lnp), region)
                    count += 1
            corr_mean = corr_acc / count
            corr2_mean = corr2_acc / count
            separation = sep_acc / count
            signal = np.sqrt(np.maximum(corr2_mean - floor, 0.0))
            per_region[region] = {
                "separation": [float(v) for v in separation],
                "corr_mean": [float(v) for v in corr_mean],
                "corr2_mean": [float(v) for v in corr2_mean],
                "signal": [float(v) for v in signal],
                "pairs": count,
                "pair_fields": [list(pair) for pair in spec["pairs"]],
            }
        out["classes"][name] = per_region
    return out


def fit_gaspari_cohn_cutoff(separation, signal, *, max_separation: float = FIT_SEPARATION_MAX_LNP,
                            bounds=CUTOFF_BOUNDS_LNP) -> dict[str, float]:
    """The Gaspari-Cohn support ``2c`` (the cutoff the filter takes) whose
    ``GC(d / c)`` fits ``signal(d) / signal(0)`` in least squares over
    ``d <= max_separation``, by a dense scan of ``c`` inside ``bounds``
    (the profile is a few dozen points; a scan has no local minima to
    fall into).  Returns the cutoff, the fit's rms residual, the
    normalised profile's reference value and ``cutoff_at_bound``: ``None``
    when the scan's best lies inside the bounds, ``"upper"`` or
    ``"lower"`` when it is the bound itself, so a receipt says when a
    cutoff is the scan's limit and not a value the profile chose (the
    2026-09-06 derivation's surface-pressure class returned the 6.0 upper
    bound: a balanced ensemble's pressure signal is column-deep and the
    profile asked for more than the whole column)."""
    d = np.asarray(separation, dtype=np.float64).reshape(-1)
    s = np.asarray(signal, dtype=np.float64).reshape(-1)
    keep = np.isfinite(d) & np.isfinite(s) & (d <= float(max_separation))
    d = d[keep]
    s = s[keep]
    if d.size < 3:
        raise ValueError("fit_gaspari_cohn_cutoff needs at least three separations inside the fit range")
    near = d <= (float(d.min()) + REFERENCE_SEPARATION_LNP)
    reference = float(s[near].max()) if near.any() else float(s[np.argmin(d)])
    if reference <= 0.0:
        return {"cutoff_lnp": float(NO_SIGNAL_CUTOFF_LNP), "fit_rms_residual": None, "signal_at_source": 0.0,
                "points": int(d.size), "no_signal_above_floor": True, "cutoff_at_bound": None}
    target = s / reference
    lo, hi = float(bounds[0]), float(bounds[1])
    cutoffs = np.exp(np.linspace(math.log(lo), math.log(hi), 600))
    best = None
    for index, cutoff in enumerate(cutoffs):
        weights = gaspari_cohn(d / cutoff, 1.0)
        residual = float(math.sqrt(np.mean((np.asarray(weights, dtype=np.float64) - target) ** 2)))
        if best is None or residual < best[1]:
            best = (float(cutoff), residual, index)
    at_bound = "upper" if best[2] == cutoffs.size - 1 else "lower" if best[2] == 0 else None
    return {"cutoff_lnp": best[0], "fit_rms_residual": best[1], "signal_at_source": reference,
            "points": int(d.size), "no_signal_above_floor": False, "cutoff_at_bound": at_bound}


def derive_cutoffs(profiles: dict[str, object], *, region: str = "global") -> dict[str, object]:
    """The fitted cutoff of every class in ``profiles`` (from
    :func:`vertical_correlation_profiles`) for ``region``, keyed by the
    :class:`FilterOptions` field it is, with the fit's residual."""
    out: dict[str, object] = {}
    for name, per_region in profiles["classes"].items():
        row = per_region[region]
        fit = fit_gaspari_cohn_cutoff(row["separation"], row["signal"])
        out[REPORT_CLASSES[name]["option"]] = {"class": name, "region": region, **fit}
    return out


def derive_from_store(config, store: str | Path, *, step: int | None = None, regions=REGIONS) -> dict[str, object]:
    """The derivation on an ensemble store: the manifest's member files (or
    the members at ``step``), the numpy float64 transform at their
    truncation, the config's vertical coordinate; the profiles by class
    and region and the fitted cutoffs for every region."""
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    from .ensemble import ENSEMBLE_MANIFEST_NAME

    store = Path(store)
    manifest_path = store / ENSEMBLE_MANIFEST_NAME if store.is_dir() else store
    store = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if step is None:
        paths = [store / entry["file"] for entry in manifest["members"]]
    else:
        paths = sorted(store.glob(f"arwen_global_member*_step{int(step):08d}.npz"))
    if not paths:
        raise ValueError(f"no member checkpoints under {store}" + ("" if step is None else f" at step {step}"))
    with np.load(paths[0], allow_pickle=False) as archive:
        nlev, tp1, _ = archive["atmosphere__theta"].shape
        nlat, nlon = archive["surface__land_fraction"].shape
    transform = SphericalHarmonicTransform.create(
        int(tp1 - 1), nlat=int(nlat), nlon=int(nlon), dealias_factor=float(config.dealias_factor),
        radius_m=float(EARTH_RADIUS_M), backend="numpy", precision="float64")
    fields = member_column_fields([str(p) for p in paths], transform, config.vertical)
    profiles = vertical_correlation_profiles(fields, transform.grid.latitude_deg, transform.grid.quadrature_weights,
                                             regions=regions)
    profiles["members_read"] = [str(p) for p in paths]
    profiles["truncation"] = int(tp1 - 1)
    profiles["cutoffs"] = {region: derive_cutoffs(profiles, region=region) for region in regions}
    profiles["rule"] = (
        "one Gaspari-Cohn cutoff (2c, in ln p) per report class, fitted in least squares to the "
        "noise-corrected rms correlation between what the class's report sees and the analysed "
        "fields at every level of the same column (the streamfunction standing for the wind in every "
        "mass-to-wind and wind-to-mass pair), across the members, normalised at the source; "
        "the global fit is the value FilterOptions carries, the regional fits are recorded beside it"
    )
    return profiles

# ---------------------------------------------------------------------------
# The radiance profile: a layer report's model-space localisation
# ---------------------------------------------------------------------------

#: Gaspari-Cohn (cutoff ``2c`` at the zero) reads one half at about 0.57 of
#: the half support ``c``, so a measured half width ``h`` (where the
#: ensemble correlation falls to one half) maps to a zero at ``2c = 2h /
#: 0.57 = 3.5 h``.
HALF_WIDTH_TO_CUTOFF = 3.5

#: The bounds a radiance channel's measured cutoff is held inside (ln p),
#: the stream's own and not the point classes' fit bounds: below 0.6 a channel
#: whose ensemble correlations collapse to a level would localise to
#: nothing between the analysis levels; above 3.0 (a factor 20 in pressure)
#: the profile would reach every level and the model-space form would say
#: nothing.
RADIANCE_CUTOFF_BOUNDS_LNP = (0.6, 3.0)


def cutoff_from_half_width(half_width_lnp: float, *, bounds=RADIANCE_CUTOFF_BOUNDS_LNP) -> float:
    """The Gaspari-Cohn zero (ln p) from a measured correlation half width."""
    if not math.isfinite(half_width_lnp) or half_width_lnp <= 0.0:
        return float(bounds[1])
    return float(min(max(HALF_WIDTH_TO_CUTOFF * float(half_width_lnp), bounds[0]), bounds[1]))


def point_profile(lnp: float, cutoff_lnp: float, axis=LOCALISATION_AXIS_LNP) -> np.ndarray:
    """``(M,)`` the Gaspari-Cohn rule of a point row, on the axis (the
    check that a profile row and a point row agree)."""
    return gaspari_cohn(np.abs(np.asarray(axis, dtype=np.float64) - float(lnp)) / float(cutoff_lnp), 1.0)


def profile_from_weighting_function(layer_lnp, weights, cutoff_lnp: float, axis=LOCALISATION_AXIS_LNP) -> np.ndarray:
    """``(M,)`` the localisation profile of a channel: its weighting
    function (``weights`` per layer at ``layer_lnp``, any positive
    measure) convolved with Gaspari-Cohn of ``cutoff_lnp`` and normalised
    to one at its peak."""
    z = np.asarray(layer_lnp, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if z.shape != w.shape:
        raise ValueError("layer_lnp and weights must have one entry per layer")
    finite = np.isfinite(z) & np.isfinite(w) & (w > 0.0)
    if not finite.any():
        raise ValueError("the weighting function carries no positive weight on a finite layer")
    z = z[finite]
    w = w[finite]
    ax = np.asarray(axis, dtype=np.float64)
    kernel = gaspari_cohn(np.abs(ax[:, None] - z[None, :]) / float(cutoff_lnp), 1.0)   # (M, L)
    prof = kernel @ w
    peak = float(prof.max())
    if peak <= 0.0:
        raise ValueError("the convolved profile is zero everywhere; the cutoff or the weights are degenerate")
    return np.clip(prof / peak, 0.0, 1.0)


def profile_centroid_lnp(layer_lnp, weights) -> float:
    """The weight-weighted mean ln p of a weighting function (the row's
    ``ln_pressure`` for thinning and the receipt)."""
    z = np.asarray(layer_lnp, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(z) & np.isfinite(w) & (w > 0.0)
    return float(np.sum(w[finite] * z[finite]) / np.sum(w[finite]))


def vertical_correlation_length(theta_fields, ln_p_full, target_lnp, *, latitude_deg=None,
                                column_mask=None, ring_stride: int = 4, lon_stride: int = 4,
                                threshold: float = 0.5) -> dict[str, float]:
    """The ensemble's vertical correlation half width about ``target_lnp``.

    ``theta_fields`` is ``(R, nlev, nlat, nlon)`` (numpy or cupy, the
    members' potential temperature on the ensemble grid), ``ln_p_full``
    ``(nlev, nlat, nlon)`` the ensemble-mean ln p of the levels.  On a
    subsample of columns (every ``ring_stride`` ring and ``lon_stride``
    longitude, restricted to ``column_mask`` ``(nlat, nlon)`` when given,
    area-weighted by cos latitude when ``latitude_deg`` is given) the
    correlation over members between the level nearest ``target_lnp`` and
    every level is taken, averaged over the columns, and the ln p distance
    at which it falls below ``threshold`` is read upward and downward.
    Returns the half widths (``up``, ``down``, their mean), the target
    level's ln p and the columns sampled; a side the correlation never
    crosses reads the distance to the last level on that side."""
    xp = np
    arr = theta_fields
    if hasattr(arr, "get"):
        import cupy  # noqa: F401 - the device namespace of the fields

        xp = arr.__class__.__module__.split(".")[0]
        xp = __import__(xp)
    sub = arr[:, :, ::int(ring_stride), ::int(lon_stride)]
    lnp = ln_p_full[:, ::int(ring_stride), ::int(lon_stride)]
    lnp_host = np.asarray(lnp.get() if hasattr(lnp, "get") else lnp, dtype=np.float64)
    r, nlev, nj, ni = sub.shape
    weight = np.ones((nj, ni))
    if latitude_deg is not None:
        lat = np.asarray(latitude_deg, dtype=np.float64)[::int(ring_stride)]
        weight = weight * np.cos(np.deg2rad(lat))[:, None]
    if column_mask is not None:
        mask = np.asarray(column_mask, dtype=bool)[::int(ring_stride), ::int(lon_stride)]
        weight = np.where(mask, weight, 0.0)
    if not np.any(weight > 0.0):
        weight = np.ones((nj, ni))
    # The target level per column: the nearest in ln p.
    target = xp.asarray(np.abs(lnp_host - float(target_lnp)).argmin(axis=0))          # (nj, ni)
    pert = sub - sub.mean(axis=0, keepdims=True)
    sd = xp.sqrt((pert ** 2).sum(axis=0) / max(r - 1, 1))                            # (nlev, nj, ni)
    at_target = xp.take_along_axis(pert, target[None, None, :, :], axis=1)[:, 0]     # (R, nj, ni)
    sd_target = xp.take_along_axis(sd, target[None, :, :], axis=0)[0]                  # (nj, ni)
    cov = (pert * at_target[:, None]).sum(axis=0) / max(r - 1, 1)                     # (nlev, nj, ni)
    denominator = sd * sd_target[None]
    corr = xp.where(denominator > 0.0, cov / xp.where(denominator > 0.0, denominator, 1.0), 0.0)
    corr_host = np.asarray(corr.get() if hasattr(corr, "get") else corr, dtype=np.float64)
    target_host = np.asarray(target.get() if hasattr(target, "get") else target)
    # Correlation as a function of ln p distance from the target, averaged
    # over columns on a common distance axis (the levels' ln p differ per
    # column by the surface pressure; the distances are what is averaged).
    dz = lnp_host - np.take_along_axis(lnp_host, target_host[None], axis=0)           # (nlev, nj, ni)
    bins = np.arange(-6.0, 6.0 + 1e-9, 0.1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    total = np.zeros(centres.size)
    count = np.zeros(centres.size)
    idx = np.clip(np.digitize(dz.reshape(nlev, -1), bins) - 1, 0, centres.size - 1)
    w_flat = np.broadcast_to(weight.reshape(1, -1), (nlev, nj * ni)).reshape(-1)
    np.add.at(total, idx.reshape(-1), (corr_host.reshape(nlev, -1) * weight.reshape(1, -1)).reshape(-1))
    np.add.at(count, idx.reshape(-1), w_flat)
    mean_corr = np.where(count > 0.0, total / np.where(count > 0.0, count, 1.0), np.nan)

    def crossing(sign: int) -> float:
        # Walk away from the target until the mean correlation falls below
        # the threshold; the distance to the last populated bin when it never does.
        order = np.argsort(sign * centres)
        last = 0.0
        for k in order:
            d = sign * centres[k]
            if d < 0.0 or not np.isfinite(mean_corr[k]):
                continue
            if mean_corr[k] < threshold:
                return float(d)
            last = float(d)
        return float(last) if last > 0.0 else float("nan")

    up = crossing(-1)      # toward lower pressure (negative dz)
    down = crossing(+1)    # toward higher pressure
    values = [v for v in (up, down) if np.isfinite(v)]
    half = float(np.mean(values)) if values else float("nan")
    return {
        "target_lnp": float(target_lnp),
        "target_pressure_pa": float(math.exp(float(target_lnp))),
        "half_width_up_lnp": up,
        "half_width_down_lnp": down,
        "half_width_lnp": half,
        "threshold": float(threshold),
        "columns_sampled": int(np.sum(weight > 0.0)),
        "members": int(r),
        "cutoff_lnp": cutoff_from_half_width(half),
    }


__all__ = [
    "CUTOFF_BOUNDS_LNP",
    "cutoff_from_half_width",
    "derive_cutoffs",
    "derive_from_store",
    "fit_gaspari_cohn_cutoff",
    "FIT_SEPARATION_MAX_LNP",
    "HALF_WIDTH_TO_CUTOFF",
    "member_column_fields",
    "NO_SIGNAL_CUTOFF_LNP",
    "point_profile",
    "profile_centroid_lnp",
    "profile_from_weighting_function",
    "RADIANCE_CUTOFF_BOUNDS_LNP",
    "REFERENCE_SEPARATION_LNP",
    "REGIONS",
    "REPORT_CLASSES",
    "vertical_correlation_length",
    "vertical_correlation_profiles",
]
