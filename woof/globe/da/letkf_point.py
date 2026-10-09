"""The point-observation LETKF on the sphere: :mod:`woof.da.letkf`'s
arithmetic under a ragged neighbour list.

What is the regional module's, by import: the Gaspari-Cohn function
(:func:`woof.da.letkf.gaspari_cohn`, cutoff = the ``2c`` where the weight
is exactly zero), the eigensolver selection (this project's batched
Jacobi kernel on the device, LAPACK on the host), the allocation-failure
recognition, the RTPS and RTPP relaxations, Hunt's prior inflation and the
closed-form inactive point (a gridpoint with no report inside its cutoff
takes ``(s - 1) x'`` with ``s = (1 - alpha) sqrt(rho) + alpha``, exactly
zero at rho = 1).  The transform itself is Hunt, Kostelich and Szunyogh
(2007) steps 4 to 9 in the regional module's form: one ``eigh`` of
``(R-1)I/rho + C Yb`` gives ``Pa~`` and the symmetric ``Wa`` at once.

What differs, and why: the regional filter's observations are gridded on
the model grid and every gridpoint gathers a fixed index stencil.  On the
Gaussian grid that stencil fails twice: longitude is periodic and the
regional stencil has no wrap, and the zonal spacing falls below a
kilometre at the pole-most ring, so a 1,200 km radius sized on the minimum
spacing spans the whole ring.  Here every observation is a point (latitude,
longitude, ln p) and every analysis COLUMN gathers its own neighbours: for
a chunk of latitude rings the candidates are the reports inside the rings'
latitude band widened by the largest cutoff, the geodesic distance to
every column of the chunk is taken in float64, the Gaspari-Cohn weight
selects the reports with a nonzero weight, and the chunk's columns are
padded to the widest local count (at most ``max_local_obs``; a column
with more keeps the largest weights and the count dropped is recorded).
Padding slots carry weight exactly zero and an index clamped to a valid
report, so they drop out of the solve without a branch, as the regional
module's masked slots do.

The solve is batched over LEVELS as well as columns: a chunk's columns are
taken in sub-chunks and every plane of a sub-chunk (the 3-D levels and the
surface the 2-D fields sit at) enters ONE eigendecomposition batch, so
the device factors thousands of ``R x R`` matrices per launch instead of
one plane's worth.  A plane of a column with no report inside its lens is
solved too (its matrix is ``(R-1)I/rho`` exactly, a fixed point of the
Jacobi kernel) and its increment is then taken from the closed form
above, bit for bit what the inactive-point rule gives.  Measured
2026-09-06 on the case's real hour (75,000 reports, 32 T127 members, the
RTX 5090): the per-plane batches of 768 matrices took 30 s of a 557 s
cycle; the level batches take it in a fraction of that (the receipt
carries the wall).  The increments are formed in place over the
perturbation stack, so the solve holds one field stack beside the prior
instead of two.

The vertical metric is ln p.  A report sits at its ``ln_pressure``; an
analysis gridpoint of a 3-D field at level k of a column sits at the
column's ``ln p_full[k]``; the 2-D field (ln ps) sits at the column's
``ln ps``.  The vertical weight is Gaspari-Cohn of the ln p separation
over the report's own vertical cutoff (per report, so a surface-pressure
report can carry no vertical localisation while a 2 m temperature report
carries a shallow one), and the two weights multiply.  Because the
horizontal weight and the report gather are per column and the vertical
weight is per level, a column chunk gathers its reports once and the
level loop re-weights them: the gather is paid ``nlev`` times less than a
per-gridpoint formulation would pay it.

A report that senses a LAYER rather than a level (a radiance) carries a
vertical localisation PROFILE instead of a cutoff
(:attr:`~woof.globe.da.observations.PointObs.localisation_profile`,
tabulated on :data:`~woof.globe.da.observations.LOCALISATION_AXIS_LNP`):
its weight at an analysis level is the profile read by linear
interpolation at the level's ln p (the channel's weighting function
convolved with the Gaspari-Cohn kernel, the model-space placement of a
radiance in a level-by-level local solve), so a channel that sounds 300 to
100 hPa updates those levels and not the surface, whatever single
pressure names its centroid.  Point rows keep the Gaspari-Cohn rule on
their own ln p bit for bit: the two forms are selected per report slot.
The hybrid covariance (``hybrid_beta`` below one, with ``static_prior``
and ``FlatObs.static_sim``): the control's weights are solved in an
AUGMENTED perturbation space, the N members' perturbations scaled by
``sqrt(beta (M-1)/(N-1))`` beside K static draws scaled by ``sqrt((1 -
beta)(M-1)/K)`` (``M = N + K``), so the covariance the localised gain is
built from is exactly ``beta L o P_ens + (1 - beta) L o P_static`` (one
positive-semidefinite matrix; Kretschmer, Hunt and Ott 2015) and the
control increment is ``X_aug A^-1 Y_aug^T R^-1 d_H`` with ``A = (M-1)I/rho
+ Y_aug^T R^-1 Y_aug`` solved per gridpoint as a batched linear system
(no square root is needed for the control).  The members' own transform
stays the pure ensemble one: under recentring by increment their mean
increment is replaced by the control's, so the hybrid reaches every
member through the control while the perturbation update keeps the
dynamic covariance the members carry (the split every operational
hybrid makes between its EnKF members and its hybrid control).

Fail-closed, as the regional module: every degenerate input raises
:class:`woof.da.letkf.LetkfError`; no path returns NaN or the prior
disguised as an analysis.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from woof.da.letkf import (
    LetkfError,
    _eigendecompose,
    _get_xp,
    _is_device_memory_error,
    _release_device_scratch,
    _resolve_eigensolver,
    _sync_namespace,
    gaspari_cohn,
)

from ..constants import EARTH_RADIUS_M
from .observations import LOCALISATION_AXIS_LNP

__all__ = [
    "ColumnGeometry",
    "PointAnalysis",
    "PointLetkfConfig",
    "PointLetkfDiagnostics",
    "FlatObs",
    "analyze_points",
    "analyze_points_with_control",
    "flatten_batches",
    "profile_weight",
    "hybrid_single_observation_increment",
    "single_observation_increment",
]


def _shape_of(array) -> tuple[int, ...]:
    """The array's shape without a host conversion (numpy, cupy or a list)."""
    shape = getattr(array, "shape", None)
    if shape is None:
        shape = np.asarray(array).shape
    return tuple(int(v) for v in shape)


@dataclass(frozen=True)
class ColumnGeometry:
    """Where the analysis gridpoints are.

    latitude_deg, longitude_deg
        The Gaussian grid's ``(nlat,)`` and ``(nlon,)`` nodes, degrees.
    ln_p_full
        ``(nlev, nlat, nlon)`` ln of the full-level pressure the 3-D fields
        sit at (the ensemble mean's).
    ln_ps
        ``(nlat, nlon)`` ln of the surface pressure the 2-D field sits at.
    radius_m
        The sphere the geodesic is measured on.
    """

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    ln_p_full: object
    ln_ps: object
    radius_m: float = EARTH_RADIUS_M

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitude_deg, dtype=np.float64).reshape(-1)
        lon = np.asarray(self.longitude_deg, dtype=np.float64).reshape(-1)
        if lat.size < 2 or lon.size < 4:
            raise LetkfError("ColumnGeometry needs at least 2 latitudes and 4 longitudes")
        if np.any(np.abs(lat) > 90.0) or not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise LetkfError("ColumnGeometry latitude/longitude are not finite degrees")
        object.__setattr__(self, "latitude_deg", lat)
        object.__setattr__(self, "longitude_deg", lon)
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise LetkfError("ColumnGeometry.radius_m must be finite and positive")
        # Shapes are read from the arrays themselves: ``ln_p_full`` and
        # ``ln_ps`` live on the analysis namespace (a device array refuses
        # an implicit host conversion, measured on the RTX 5090; a getattr
        # default is evaluated eagerly, so the host fallback is a branch).
        shape3 = _shape_of(self.ln_p_full)
        shape2 = _shape_of(self.ln_ps)
        if len(shape3) != 3 or shape3[1:] != (lat.size, lon.size):
            raise LetkfError(
                f"ColumnGeometry.ln_p_full must be (nlev, {lat.size}, {lon.size}), got {shape3}"
            )
        if shape2 != (lat.size, lon.size):
            raise LetkfError(
                f"ColumnGeometry.ln_ps must be ({lat.size}, {lon.size}), got {shape2}"
            )

    @property
    def nlat(self) -> int:
        return int(self.latitude_deg.size)

    @property
    def nlon(self) -> int:
        return int(self.longitude_deg.size)

    @property
    def nlev(self) -> int:
        return int(_shape_of(self.ln_p_full)[0])


#: The horizontal weight is quantised to this many steps between 0 and 1
#: when the reports of a column are ranked for the local cap, so that the
#: ranking is one integer sort (weight, then report index) on every array
#: module: 2^30, a resolution of 1e-9 in a weight that is itself a
#: fifth-order polynomial of a rounded distance.
_WEIGHT_QUANTUM = float(1 << 30)


@dataclass(frozen=True)
class PointLetkfConfig:
    """The transform's settings; see :class:`woof.da.letkf.LetkfConfig`
    for the meaning of each (the names are the same)."""

    rtps_alpha: float
    prior_inflation: float = 1.0
    relaxation: str = "rtps"
    max_local_obs: int = 400
    #: Latitude rings per column chunk (``None``: sized from the budget).
    chunk_rings: int | None = None
    #: Longitude segments a chunk's rings are cut into (``None``: one; a
    #: chunk the device refuses at one ring and one column per level batch
    #: is cut into two, four, ... segments down to a single column before
    #: the analysis is refused, so a card that cannot hold one ring's
    #: gather beside the ensemble still produces the same analysis in
    #: smaller pieces).
    chunk_segments: int | None = None
    #: Device scratch the analysis may hold: half for a chunk's gathered
    #: reports, half for one level batch's matrices (both priced
    #: pessimistically by two, so the real footprint is about half).
    memory_budget_mib: float = 2048.0
    solve_dtype: str = "float64"
    eigensolver: str = "auto"

    def __post_init__(self) -> None:
        a = float(self.rtps_alpha)
        if not math.isfinite(a) or not 0.0 <= a <= 1.0:
            raise LetkfError(f"rtps_alpha must lie in [0, 1], got {a!r}")
        rho = float(self.prior_inflation)
        if not math.isfinite(rho) or rho <= 0.0:
            raise LetkfError(f"prior_inflation must be finite and positive, got {rho!r}")
        if self.relaxation not in ("rtps", "rtpp"):
            raise LetkfError(f"relaxation must be rtps or rtpp, got {self.relaxation!r}")
        if isinstance(self.max_local_obs, bool) or int(self.max_local_obs) < 1:
            raise LetkfError("max_local_obs must be a positive whole number")
        if self.chunk_rings is not None and int(self.chunk_rings) < 1:
            raise LetkfError("chunk_rings must be >= 1 or None")
        if self.chunk_segments is not None and int(self.chunk_segments) < 1:
            raise LetkfError("chunk_segments must be >= 1 or None")
        if not math.isfinite(self.memory_budget_mib) or self.memory_budget_mib <= 0.0:
            raise LetkfError("memory_budget_mib must be finite and positive")
        if self.solve_dtype not in ("float32", "float64"):
            raise LetkfError("solve_dtype must be float32 or float64")
        if self.eigensolver not in ("auto", "jacobi", "library"):
            raise LetkfError("eigensolver must be auto, jacobi or library")


@dataclass
class PointLetkfDiagnostics:
    members: int = 0
    grid_shape: tuple[int, int, int] = (0, 0, 0)
    observations: int = 0
    prior_inflation: float = 1.0
    rtps_alpha: float = 0.0
    relaxation: str = "rtps"
    eigensolver: str = ""
    max_jacobi_sweeps: int = 0
    #: Columns with at least one report inside the horizontal cutoff.
    active_columns: int = 0
    total_columns: int = 0
    #: Gridpoints (3-D levels plus the 2-D surface) that entered a solve.
    active_points: int = 0
    total_points: int = 0
    #: Widest local report count any column saw, and the padded width of
    #: the widest chunk.
    max_local_obs: int = 0
    max_padded_slots: int = 0
    #: Reports a column could not take because it already held
    #: ``max_local_obs`` closer ones, summed over columns.
    dropped_by_cap: int = 0
    chunks: int = 0
    chunk_rings: int = 0
    #: Longitude segments each ring chunk was cut into at the end (1: whole rings).
    chunk_segments: int = 1
    chunk_oom_shrinks: int = 0
    #: Level batches solved (one eigendecomposition launch each) and the
    #: widest, in (level, column) pairs.
    level_batches: int = 0
    max_batch_points: int = 0
    #: Where the solve ran (``device`` or ``host``) and its whole wall.
    path: str = ""
    wall_seconds: float = 0.0
    setup_seconds: float = 0.0
    gather_seconds: float = 0.0
    solve_seconds: float = 0.0
    finish_seconds: float = 0.0
    mean_increment_rms: dict = field(default_factory=dict)
    #: Grid rms of the control increment per field (amendment A), when formed.
    control_increment_rms: dict = field(default_factory=dict)
    #: Fields whose members agreed to rounding and were left out of the
    #: solve with a zero increment (named, never silently).
    zero_spread_fields: list = field(default_factory=list)
    prior_spread: dict = field(default_factory=dict)
    posterior_spread: dict = field(default_factory=dict)
    #: The hybrid: the ensemble weight and the static draws in the
    #: augmented control solve (0 draws and beta 1 when not hybrid), the
    #: seconds the augmented solves took, and the solver used.
    hybrid_beta: float = 1.0
    static_samples: int = 0
    hybrid_solve_seconds: float = 0.0
    hybrid_solver: str = ""


@dataclass
class FlatObs:
    """Every batch's reports in one set of arrays, on the analysis
    namespace.  ``hcut_m`` and ``vcut`` are per report; a report with no
    vertical localisation carries ``vcut = inf``."""

    lat_rad: object
    lon_rad: object
    lnp: object
    value: object
    err2: object
    sim: object
    hcut_m: object
    vcut: object
    count: int
    #: ``(n,)`` the member mean of ``sim``, computed ONCE here over the
    #: concatenated ``(R, n)`` array and gathered by every chunk: the
    #: innovation is ``value - simbar`` on exactly these bits, so a report
    #: whose value IS this mean has an innovation of exactly zero and moves
    #: the ensemble mean by exactly zero (the agreeing-observation family).
    simbar: object = None
    #: ``(n,)`` the high-resolution innovation ``y - H(x_H^b)`` formed from
    #: the control (deterministic) background, or None.  When present the
    #: solve also returns the control increment ``X_L Pa~ C d_H`` (amendment
    #: A: the control gets its own analysis through the ensemble
    #: covariance; the ensemble-mean innovation never touches it).
    control_innovation: object = None
    #: ``(n, M)`` the vertical localisation profile of every report on
    #: :data:`~woof.globe.da.observations.LOCALISATION_AXIS_LNP`
    #: (zeros for a report without one) and ``(n,)`` bool, which reports
    #: carry one; None when no report does (the level loop then never
    #: touches the table).
    vprof: object = None
    has_profile: object = None
    #: ``(K, n)`` the observation-space perturbations of the K static
    #: draws (``H(x_mean + x_s) - H(x_mean)``), or None: the hybrid's
    #: static block of ``Y_aug``.
    static_sim: object = None


@dataclass
class PointAnalysis:
    """What :func:`analyze_points_with_control` returns: the per-member
    increments ``{field: (R, ...)}`` and, when the observations carried a
    control innovation, the control increment ``{field: (...)}`` on the
    same grid (one member's shape, no member axis)."""

    increments: dict
    control_increment: dict | None = None


def flatten_batches(batches, xp, *, horizontal_cutoff_m, vertical_cutoff_for,
                    solve_dtype, control: bool = False, static: bool = False) -> FlatObs:
    """One :class:`FlatObs` from :class:`~woof.globe.da.observations.PointObs`
    batches.  ``horizontal_cutoff_m`` is the default horizontal cutoff;
    ``vertical_cutoff_for(batch, surface_mask)`` returns the per-row
    vertical cutoff array (``inf`` for none).  Rows are validated: finite
    positions, filled ``ln_pressure``, filled ``simulated``.  With ``control`` True every
    batch must carry ``control_simulated`` (``(1, n)`` H of the control
    background) and the flat set carries ``control_innovation``.  With
    ``static`` True every batch must carry ``static_simulated`` (``(K, n)``,
    one K for all) and the flat set carries ``static_sim``."""
    lat, lon, lnp, val, err2, sims, hcut, vcut, ctrl, stat = [], [], [], [], [], [], [], [], [], []
    profiles, has_profile = [], []
    members = None
    draws = None
    for b in batches:
        if b.count == 0:
            continue
        prof = getattr(b, "localisation_profile", None)
        if prof is not None:
            prof = np.asarray(prof, dtype=np.float64)
            if prof.shape != (b.count, LOCALISATION_AXIS_LNP.size):
                raise LetkfError(
                    f"batch {b.stream!r}/{b.variable!r} carries a localisation profile of shape "
                    f"{prof.shape}, not ({b.count}, {LOCALISATION_AXIS_LNP.size})"
                )
            profiles.append(prof)
            has_profile.append(np.ones(b.count, dtype=bool))
        else:
            profiles.append(np.zeros((b.count, LOCALISATION_AXIS_LNP.size)))
            has_profile.append(np.zeros(b.count, dtype=bool))
        if static:
            if getattr(b, "static_simulated", None) is None:
                raise LetkfError(
                    f"batch {b.stream!r}/{b.variable!r} has no static_simulated H(x_mean + x_s) - H(x_mean); "
                    "the hybrid analysis needs the static draws' observation-space perturbations on every row"
                )
            s = np.asarray(b.static_simulated, dtype=np.float64)
            if s.ndim != 2 or s.shape[1] != b.count or not np.all(np.isfinite(s)):
                raise LetkfError(
                    f"batch {b.stream!r}/{b.variable!r} static_simulated is not (K, rows) and finite"
                )
            if draws is None:
                draws = int(s.shape[0])
            elif int(s.shape[0]) != draws:
                raise LetkfError(
                    f"batch {b.stream!r}/{b.variable!r} carries {s.shape[0]} static draws where the first "
                    f"batch carries {draws}"
                )
            stat.append(s)
        if control:
            if getattr(b, "control_simulated", None) is None:
                raise LetkfError(
                    f"batch {b.stream!r}/{b.variable!r} has no control_simulated H(x_H^b); "
                    "the control analysis needs the high-resolution innovation of every row"
                )
            c = np.asarray(b.control_simulated, dtype=np.float64).reshape(-1)
            if c.size != b.count or not np.all(np.isfinite(c)):
                raise LetkfError(
                    f"batch {b.stream!r}/{b.variable!r} control_simulated is not one finite value per row"
                )
            ctrl.append(np.asarray(b.value, dtype=np.float64) - c)
        if b.simulated is None:
            raise LetkfError(
                f"batch {b.stream!r}/{b.variable!r} has no simulated H(x_k); "
                "run the operators before the filter"
            )
        sim = np.asarray(b.simulated, dtype=np.float64)
        if members is None:
            members = sim.shape[0]
        elif sim.shape[0] != members:
            raise LetkfError(
                f"batch {b.stream!r}/{b.variable!r} simulates {sim.shape[0]} members "
                f"where the first batch simulates {members}"
            )
        if not np.all(np.isfinite(sim)):
            raise LetkfError(
                f"batch {b.stream!r}/{b.variable!r} has non-finite H(x_k); the "
                "forward operator failed on a member and the filter will not average over that"
            )
        if not np.all(np.isfinite(b.ln_pressure)):
            raise LetkfError(
                f"batch {b.stream!r}/{b.variable!r} has rows whose ln_pressure is "
                "not filled; a surface row takes the model's ln ps at the station "
                "from the operator"
            )
        lat.append(np.deg2rad(b.latitude_deg))
        lon.append(np.deg2rad(b.longitude_deg))
        lnp.append(b.ln_pressure)
        val.append(b.value)
        err2.append(b.error.astype(np.float64) ** 2)
        sims.append(sim)
        h = float(horizontal_cutoff_m if b.horizontal_cutoff_km is None
                  else b.horizontal_cutoff_km * 1000.0)
        if not math.isfinite(h) or h <= 0.0:
            raise LetkfError("horizontal cutoff must be finite and positive")
        hcut.append(np.full(b.count, h))
        v = np.asarray(vertical_cutoff_for(b, b.surface), dtype=np.float64)
        if v.shape != (b.count,):
            raise LetkfError("vertical_cutoff_for must return one cutoff per row")
        if np.any(v <= 0.0):
            raise LetkfError("vertical cutoffs must be positive (inf for none)")
        vcut.append(v)
    if members is None:
        return FlatObs(*(xp.zeros(0, dtype=np.float64) for _ in range(5)),
                       xp.zeros((0, 0), dtype=np.float64),
                       xp.zeros(0), xp.zeros(0), 0, xp.zeros(0))
    e2 = np.concatenate(err2).astype(np.dtype(solve_dtype))
    if not np.all(e2 > 0.0):
        raise LetkfError(
            f"an observation error's SQUARE underflows to zero in the {solve_dtype} "
            "solve; use float64 or rescale the observation units"
        )
    sim_all = np.concatenate(sims, axis=1)
    any_profile = np.concatenate(has_profile)
    return FlatObs(
        lat_rad=xp.asarray(np.concatenate(lat)),
        lon_rad=xp.asarray(np.concatenate(lon)),
        lnp=xp.asarray(np.concatenate(lnp)),
        value=xp.asarray(np.concatenate(val)),
        err2=xp.asarray(e2),
        sim=xp.asarray(sim_all),
        hcut_m=xp.asarray(np.concatenate(hcut)),
        vcut=xp.asarray(np.concatenate(vcut)),
        count=int(sum(v.size for v in val)),
        simbar=xp.asarray(sim_all.mean(axis=0)),
        control_innovation=xp.asarray(np.concatenate(ctrl)) if control else None,
        vprof=xp.asarray(np.concatenate(profiles, axis=0)) if any_profile.any() else None,
        has_profile=xp.asarray(any_profile) if any_profile.any() else None,
        static_sim=xp.asarray(np.concatenate(stat, axis=1)) if static else None,
    )


def profile_weight(vprof, lnp, xp):
    """``(G, P)`` the tabulated localisation weight of the ``(G, P, M)``
    profiles at the ``(G, P)`` ln p values: linear interpolation on
    :data:`~woof.globe.da.observations.LOCALISATION_AXIS_LNP`,
    the end values held outside the axis."""
    axis0 = float(LOCALISATION_AXIS_LNP[0])
    step = float(LOCALISATION_AXIS_LNP[1] - LOCALISATION_AXIS_LNP[0])
    m = int(LOCALISATION_AXIS_LNP.size)
    f = xp.clip((lnp - axis0) / step, 0.0, float(m - 1))
    i0 = xp.minimum(xp.floor(f).astype(int), m - 2)
    frac = f - i0
    lo = xp.take_along_axis(vprof, i0[..., None], axis=-1)[..., 0]
    hi = xp.take_along_axis(vprof, (i0 + 1)[..., None], axis=-1)[..., 0]
    return lo * (1.0 - frac) + hi * frac


def _geodesic(lat1, lon1, lat2, lon2, radius_m, xp):
    """Haversine, float64, broadcasting."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (xp.sin(dlat * 0.5) ** 2
         + xp.cos(lat1) * xp.cos(lat2) * xp.sin(dlon * 0.5) ** 2)
    return (2.0 * radius_m) * xp.arcsin(xp.sqrt(xp.minimum(a, 1.0)))


def _column_bytes(members: int, slots: int, solve_itemsize: int, candidates: int) -> int:
    """Device bytes one column of a gathered chunk holds: the candidate
    distance, weight and ordering rows over ``candidates`` reports, and the
    gathered ``(P, R)`` perturbations with the per-slot weight, index,
    error, innovation, ln p and vertical-cutoff arrays.  Pessimistic by a
    factor two for the allocator, as the regional module prices its chunk."""
    r = int(members)
    p = int(slots)
    ds = int(solve_itemsize)
    per_column = 3 * int(candidates) * 8 + r * p * ds + 8 * p * 8
    return 2 * per_column


def _pair_bytes(members: int, slots: int, solve_itemsize: int) -> int:
    """Device bytes one (level, column) pair costs in a level batch: the
    ``(R, P)`` C matrix, the ``R x R`` stacks (A, its symmetrised copy, the
    eigenvectors, Pa and Wa), the per-slot weights.  Pessimistic by two."""
    r = int(members)
    p = int(slots)
    ds = int(solve_itemsize)
    return 2 * (r * p * ds + 5 * r * r * ds + 3 * p * ds)


def single_observation_increment(prior_field, simulated, value, error, weight,
                                 *, prior_inflation: float = 1.0):
    """The analytic ensemble-mean increment ONE report produces at every
    gridpoint, for the calibration family.

    With a single report the localised Kalman gain at a gridpoint is
    ``K = w P_xy / (w P_yy + sigma^2)`` (R-localisation: the report's
    error variance is divided by the weight ``w``), where ``P_xy`` is the
    sample covariance between the gridpoint's prior and ``H(x)`` over the
    members and ``P_yy`` the sample variance of ``H(x)``, both with Hunt's
    ``rho`` multiplying them.  The mean increment is ``K (y - mean H(x))``.
    ``prior_field`` is ``(R, ...)``, ``simulated`` ``(R,)``, ``weight``
    broadcastable to the field's trailing shape."""
    xb = np.asarray(prior_field, dtype=np.float64)
    r = xb.shape[0]
    x_pert = xb - xb.mean(axis=0, keepdims=True)
    s = np.asarray(simulated, dtype=np.float64)
    s_pert = s - s.mean()
    rho = float(prior_inflation)
    p_xy = rho * np.tensordot(s_pert, x_pert, axes=(0, 0)) / (r - 1)
    p_yy = rho * float(np.sum(s_pert ** 2)) / (r - 1)
    w = np.asarray(weight, dtype=np.float64)
    d = float(value) - float(s.mean())
    gain = np.where(w > 0.0, w * p_xy / (w * p_yy + float(error) ** 2), 0.0)
    return gain * d


def hybrid_single_observation_increment(prior_field, static_field, simulated, static_simulated,
                                        value, error, weight, beta, *, prior_inflation: float = 1.0):
    """The analytic HYBRID control increment ONE report produces at every
    gridpoint: the localised gain of ``beta P_ens + (1 - beta) P_static``
    with ``P_ens`` the members' sample covariance (``1/(R-1)``, Hunt's
    rho on it) and ``P_static`` the K static draws' (``1/K``, the draws
    being zero-mean by construction and not centred).  ``prior_field``
    ``(R, ...)``, ``static_field`` ``(K, ...)``, ``simulated`` ``(R,)``,
    ``static_simulated`` ``(K,)``; the innovation is the control's
    ``value - H(x_H^b)`` handed in as ``value`` minus zero here (pass the
    innovation as ``value`` with ``simulated`` centred, or the raw pair)."""
    xb = np.asarray(prior_field, dtype=np.float64)
    r = xb.shape[0]
    x_pert = xb - xb.mean(axis=0, keepdims=True)
    s = np.asarray(simulated, dtype=np.float64)
    s_pert = s - s.mean()
    rho = float(prior_inflation)
    p_xy = rho * np.tensordot(s_pert, x_pert, axes=(0, 0)) / (r - 1)
    p_yy = rho * float(np.sum(s_pert ** 2)) / (r - 1)
    xs = np.asarray(static_field, dtype=np.float64)
    k = xs.shape[0]
    ss = np.asarray(static_simulated, dtype=np.float64)
    q_xy = rho * np.tensordot(ss, xs, axes=(0, 0)) / k
    q_yy = rho * float(np.sum(ss ** 2)) / k
    b = float(beta)
    h_xy = b * p_xy + (1.0 - b) * q_xy
    h_yy = b * p_yy + (1.0 - b) * q_yy
    w = np.asarray(weight, dtype=np.float64)
    d = float(value) - float(s.mean())
    gain = np.where(w > 0.0, w * h_xy / (w * h_yy + float(error) ** 2), 0.0)
    return gain * d


def _batched_solve(xp, amat, rhs):
    """``A^-1 rhs`` for a batch of symmetric positive-definite matrices
    ``(G, M, M)`` and right-hand sides ``(G, M)``: the backend's batched
    linear solve (LU on the device through cuBLAS, LAPACK on the host).
    Returns ``(solution, solver_name)``."""
    try:
        out = xp.linalg.solve(amat, rhs[..., None])[..., 0]
        return out, "batched-solve"
    except Exception as exc:  # noqa: BLE001 - the library's failure is the message
        if _is_device_memory_error(exc):
            raise
        raise LetkfError(
            "the batched linear solve of the augmented hybrid matrix (M-1)I/rho + "
            f"Y_aug^T R^-1 Y_aug failed: {type(exc).__name__}: {exc}.  No control analysis was "
            "produced; do not treat the prior as one."
        ) from exc


def analyze_points(prior, obs: FlatObs, geometry: ColumnGeometry,
                   config: PointLetkfConfig,
                   diagnostics: PointLetkfDiagnostics | None = None):
    """One LETKF analysis of point observations.  Returns per-member
    increments to ADD to the prior, ``{field: (R, ...)}``; the control
    increment, when ``obs.control_innovation`` is set, is reached through
    :func:`analyze_points_with_control`."""
    return analyze_points_with_control(prior, obs, geometry, config, diagnostics).increments


def analyze_points_with_control(prior, obs: FlatObs, geometry: ColumnGeometry,
                                config: PointLetkfConfig,
                                diagnostics: PointLetkfDiagnostics | None = None, *,
                                static_prior=None, hybrid_beta: float = 1.0) -> PointAnalysis:
    """One LETKF analysis of point observations: the per-member increments
    and, when ``obs.control_innovation`` is set, the control increment
    ``X_L w_H`` with ``w_H = Pa~ C d_H`` at every gridpoint (the same
    localised weights, the high-resolution innovation in place of the
    ensemble-mean one; rho and the relaxation act on the perturbations
    only, so the control increment is the localised gain applied to d_H).

    prior
        ``{name: array}``, each ``(R, nlev, nlat, nlon)`` (a 3-D field) or
        ``(R, nlat, nlon)`` (a 2-D field at the surface); numpy or cupy,
        the analysis runs in whichever namespace they arrive in.
    obs
        :class:`FlatObs` on the same namespace.
    geometry
        :class:`ColumnGeometry`.
    static_prior, hybrid_beta
        The hybrid (the module doc): ``static_prior`` ``{name: (K, ...)}``
        the K static draws' grid perturbations on the same fields and
        shapes, ``obs.static_sim`` their observation-space perturbations,
        ``hybrid_beta`` the ensemble weight in (0, 1].  Below one the
        control increment is solved in the augmented space; at one the
        static draws are not consulted.  A beta below one without draws is
        refused: an analysis that ran on the ensemble alone while its
        receipt claimed a hybrid is the breakage the refusal prevents.
    """
    if diagnostics is None:
        diagnostics = PointLetkfDiagnostics()
    t_enter = time.perf_counter()
    names = list(prior)
    if not names:
        raise LetkfError("analyze_points: the prior has no fields")
    xp = _get_xp(*[prior[n] for n in names])
    work_dtype = np.dtype(xp.asarray(prior[names[0]]).dtype)
    if work_dtype.kind != "f":
        work_dtype = np.dtype(np.float64)
    solve_dtype = np.dtype(config.solve_dtype)
    nlat, nlon, nlev = geometry.nlat, geometry.nlon, geometry.nlev
    beta = float(hybrid_beta)
    if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise LetkfError(f"hybrid_beta must lie in (0, 1], got {beta!r}")
    hybrid = beta < 1.0
    if hybrid and (static_prior is None or obs.static_sim is None):
        raise LetkfError(
            f"hybrid_beta {beta} asks for the static covariance's share of the gain but no static "
            "draws were handed in (static_prior and FlatObs.static_sim); the solve refuses rather than "
            "analyse on the ensemble alone under a receipt that claims a hybrid"
        )

    # ---- validate the prior ------------------------------------------
    members = None
    pri: dict[str, object] = {}
    three_d: dict[str, bool] = {}
    for n in names:
        arr = xp.asarray(prior[n], dtype=work_dtype)
        if arr.ndim == 4:
            if arr.shape[1:] != (nlev, nlat, nlon):
                raise LetkfError(
                    f"prior field {n!r} has shape {arr.shape}, expected (R, {nlev}, {nlat}, {nlon})"
                )
            three_d[n] = True
        elif arr.ndim == 3:
            if arr.shape[1:] != (nlat, nlon):
                raise LetkfError(
                    f"prior field {n!r} has shape {arr.shape}, expected (R, {nlat}, {nlon})"
                )
            three_d[n] = False
        else:
            raise LetkfError(f"prior field {n!r} must be 3-D or 4-D with a leading member axis")
        if members is None:
            members = int(arr.shape[0])
        elif int(arr.shape[0]) != members:
            raise LetkfError("prior fields disagree on the member count")
        if not bool(xp.all(xp.isfinite(arr))):
            raise LetkfError(f"prior field {n!r} contains non-finite values")
        pri[n] = arr
    if members < 2:
        raise LetkfError(f"LETKF needs at least 2 members, got {members}")
    if obs.count and int(obs.sim.shape[0]) != members:
        raise LetkfError(
            f"observations simulate {int(obs.sim.shape[0])} members, the prior has {members}"
        )
    # ---- the static draws of the hybrid ------------------------------
    xs: dict[str, object] = {}
    draws = 0
    if hybrid:
        for n in names:
            if n not in static_prior:
                raise LetkfError(f"static_prior lacks the field {n!r} the prior analyses")
            arr = xp.asarray(static_prior[n], dtype=work_dtype)
            if arr.shape[1:] != pri[n].shape[1:]:
                raise LetkfError(
                    f"static_prior field {n!r} has shape {arr.shape}, expected (K, {pri[n].shape[1:]})"
                )
            if draws == 0:
                draws = int(arr.shape[0])
            elif int(arr.shape[0]) != draws:
                raise LetkfError("static_prior fields disagree on the draw count")
            if not bool(xp.all(xp.isfinite(arr))):
                raise LetkfError(f"static_prior field {n!r} contains non-finite values")
            xs[n] = arr
        if draws < 1:
            raise LetkfError("the hybrid needs at least one static draw")
        if obs.count and tuple(_shape_of(obs.static_sim)) != (draws, int(obs.count)):
            raise LetkfError(
                f"FlatObs.static_sim has shape {_shape_of(obs.static_sim)}, expected ({draws}, {int(obs.count)})"
            )
    diagnostics.hybrid_beta = beta
    diagnostics.static_samples = draws

    diagnostics.members = members
    diagnostics.grid_shape = (nlev, nlat, nlon)
    diagnostics.observations = int(obs.count)
    diagnostics.total_columns = nlat * nlon
    # Gridpoints the level loop can visit: every 3-D level of every
    # column plus the surface the 2-D fields sit at.
    diagnostics.total_points = (nlev + 1) * nlat * nlon
    diagnostics.prior_inflation = float(config.prior_inflation)
    diagnostics.rtps_alpha = float(config.rtps_alpha)
    diagnostics.relaxation = str(config.relaxation)
    eigensolver = _resolve_eigensolver(xp, members, solve_dtype, config)
    diagnostics.eigensolver = eigensolver

    # ---- prior perturbations, spread, inactive-point transform ------
    xb = {}
    spreadless = []
    for n in names:
        xb[n] = pri[n] - pri[n].mean(axis=0, keepdims=True)
        sigma = xp.sqrt((xb[n] ** 2).sum(axis=0) / (members - 1))
        scale = float(xp.abs(pri[n]).max())
        widest = float(sigma.max())
        if widest <= 1e-12 * max(scale, 1e-300):
            spreadless.append(n)
        diagnostics.prior_spread[n] = float(sigma.mean())
    if spreadless and len(spreadless) == len(names):
        raise LetkfError(
            f"prior fields {spreadless} have no usable ensemble spread anywhere: the "
            "members are identical to rounding, so there is no background "
            "covariance and no analysis to compute (an ensemble-generation "
            "failure, not something the filter papers over with zero)"
        )
    # A field whose members agree to rounding (the velocity potential of a
    # rotational draw before its first step) has no covariance with any
    # report and its increment is exactly zero by the arithmetic; it is
    # left out of the solve and named, and the refusal above stays for an
    # ensemble with no spread in any field.
    diagnostics.zero_spread_fields = list(spreadless)
    names = [n for n in names if n not in spreadless]
    rho = float(config.prior_inflation)
    inactive_scale = 1.0 if rho == 1.0 else (
        (1.0 - float(config.rtps_alpha)) * math.sqrt(rho) + float(config.rtps_alpha))
    step = work_dtype.type(inactive_scale - 1.0)
    all_names = list(pri)
    want_control = obs.control_innovation is not None
    control = {n: xp.zeros(pri[n].shape[1:], dtype=work_dtype) for n in all_names} if want_control else None
    if want_control and obs.count and int(xp.asarray(obs.control_innovation).size) != int(obs.count):
        raise LetkfError("control_innovation must carry one value per report")
    diagnostics.path = "host" if xp is np else "device"
    if obs.count == 0:
        increments = {n: (xp.zeros_like(pri[n]) if inactive_scale == 1.0 else xb[n] * step) for n in all_names}
        _finish(xp, all_names, pri, increments, members, diagnostics)
        _sync_namespace(xp)
        diagnostics.wall_seconds = time.perf_counter() - t_enter
        return PointAnalysis(increments, control)

    # ---- geometry on the namespace -----------------------------------
    lat_rad = xp.asarray(np.deg2rad(geometry.latitude_deg))
    lon_rad = xp.asarray(np.deg2rad(geometry.longitude_deg))
    ln_p_full = xp.asarray(geometry.ln_p_full, dtype=np.float64)
    ln_ps = xp.asarray(geometry.ln_ps, dtype=np.float64)
    radius = float(geometry.radius_m)
    obs_lat_np = np.asarray(_to_host(xp, obs.lat_rad))
    obs_lat_sorted = np.sort(obs_lat_np)
    hcut_max_m = float(_to_host(xp, obs.hcut_m).max())
    band_rad = hcut_max_m / radius
    ident = xp.eye(members, dtype=solve_dtype)
    scale_i = solve_dtype.type(members - 1) / solve_dtype.type(rho)
    alpha = solve_dtype.type(config.rtps_alpha)
    cap = int(config.max_local_obs)
    planes = nlev + 1                     # the 3-D levels and the surface plane
    fields3 = [n for n in names if three_d[n]]
    fields2 = [n for n in names if not three_d[n]]
    ncol = nlat * nlon
    # The augmented space of the hybrid: M = N + K columns, the members'
    # block scaled so its share of the covariance is beta and the static
    # block so its share is 1 - beta (the module doc).  The control's
    # weights alone are solved there; the members keep the ensemble
    # transform.
    aug = members + draws if hybrid else members
    if hybrid:
        ident_aug = xp.eye(aug, dtype=solve_dtype)
        scale_aug = solve_dtype.type(aug - 1) / solve_dtype.type(rho)
        c_ens = solve_dtype.type(math.sqrt(beta * (aug - 1) / (members - 1)))
        c_sta = solve_dtype.type(math.sqrt((1.0 - beta) * (aug - 1) / draws))
    hybrid_seconds = 0.0
    hybrid_solver = ""
    # The increments are formed in place over the perturbation stack: a
    # (level, column) pair is solved exactly once, so once its increment is
    # written its perturbation is never read again, and an unsolved pair
    # keeps ``x' (s - 1)`` from the closed form.  ``done`` is the ledger of
    # the columns already written: a chunk retried after the device refused
    # an allocation skips them, because their stack now holds increments and
    # a second solve would read those as perturbations.
    xb_flat3 = {n: xb[n].reshape(members, nlev, ncol) for n in fields3}
    xb_flat2 = {n: xb[n].reshape(members, ncol) for n in fields2}
    xs_flat3 = {n: xs[n].reshape(draws, nlev, ncol) for n in fields3} if hybrid else {}
    xs_flat2 = {n: xs[n].reshape(draws, ncol) for n in fields2} if hybrid else {}
    control3 = {n: control[n].reshape(nlev, ncol) for n in fields3} if want_control else {}
    control2 = {n: control[n].reshape(ncol) for n in fields2} if want_control else {}
    done = np.zeros(ncol, dtype=bool)

    def _leave_unsolved(csel) -> None:
        """The closed form for every pair of the columns ``csel`` (a slice
        of the flat column index, or an index array) no solve visited (no
        candidate report in the band, or none inside any lens): exactly
        zero at rho = 1, else ``(s - 1) x'``."""
        for n in fields3:
            block = xb_flat3[n][:, :, csel]
            xb_flat3[n][:, :, csel] = 0 if inactive_scale == 1.0 else block * step
        for n in fields2:
            block = xb_flat2[n][:, csel]
            xb_flat2[n][:, csel] = 0 if inactive_scale == 1.0 else block * step

    budget = int(config.memory_budget_mib * (1 << 20))
    gather_budget = max(1, budget // 2)
    solve_budget = max(1, budget // 2)
    lat_deg = np.asarray(geometry.latitude_deg, dtype=np.float64)

    def _candidates(j0: int, j1: int) -> int:
        band = np.deg2rad(lat_deg[j0:j1])
        lo = float(band.min()) - band_rad
        hi = float(band.max()) + band_rad
        return int(np.searchsorted(obs_lat_sorted, hi, side="right") - np.searchsorted(obs_lat_sorted, lo, side="left"))

    def _rings_from(j0: int, ceiling: int) -> int:
        """Rings from ``j0`` whose gather fits the budget (at least one);
        the hybrid's static block is gathered beside the members', so the
        augmented count prices the column."""
        rings = 1
        while j0 + rings < nlat and rings < ceiling:
            g = (rings + 1) * nlon
            if g * _column_bytes(aug, cap, solve_dtype.itemsize, _candidates(j0, j0 + rings + 1)) > gather_budget:
                break
            rings += 1
        return rings

    fixed_rings = None if config.chunk_rings is None else max(1, min(nlat, int(config.chunk_rings)))
    pair_cost = _pair_bytes(aug, cap, solve_dtype.itemsize)
    columns_per_batch = max(1, solve_budget // (pair_cost * planes))

    _sync_namespace(xp)
    diagnostics.setup_seconds = time.perf_counter() - t_enter
    gather_seconds = 0.0
    solve_seconds = 0.0
    active_columns = 0
    active_points = 0
    max_local = 0
    max_padded = 0
    dropped = 0
    max_sweeps = 0
    chunks = 0
    level_batches = 0
    max_batch = 0

    def _solve_columns(c0: int, c1: int, gathered: dict):
        """Every plane of the chunk's columns ``[c0, c1)`` (chunk-local
        indices) in one eigendecomposition batch.  Returns the count of
        active (level, column) pairs and the sweeps."""
        nonlocal level_batches, max_batch, hybrid_seconds, hybrid_solver
        gs = c1 - c0
        cols = gathered["cols_flat"][c0:c1]                        # (Gs,) flat column indices
        cols_host = gathered["cols_host"][c0:c1]
        if gathered["whole_rings"]:
            c_lo = int(cols_host[0])
            csel = slice(c_lo, c_lo + gs)                          # a contiguous view of the stack
        else:
            csel = cols                                            # a segment: gather and scatter
        wh_sel = gathered["wh_sel"][c0:c1]                         # (Gs, P)
        yb_sub = gathered["yb_all"][c0:c1]                         # (Gs, P, R)
        dvec = gathered["dvec_all"][c0:c1]                         # (Gs, P)
        err2 = gathered["err2_all"][c0:c1]
        lnp_obs = gathered["lnp_obs"][c0:c1]
        vcut_obs = gathered["vcut_obs"][c0:c1]
        dctl = gathered["dctl_all"][c0:c1] if want_control else None
        p = int(wh_sel.shape[1])
        # The planes' ln p per column: the 3-D levels, then the surface.
        lnp_plane = xp.empty((gs, planes), dtype=np.float64)
        lnp_plane[:, :nlev] = xp.swapaxes(ln_p_full.reshape(nlev, ncol)[:, cols], 0, 1)
        lnp_plane[:, nlev] = ln_ps.reshape(ncol)[cols]
        dz = xp.abs(lnp_plane[:, :, None] - lnp_obs[:, None, :])  # (Gs, planes, P)
        wv = gaspari_cohn(dz / vcut_obs[:, None, :], 1.0)
        del dz
        if gathered["prof_sel"] is not None:
            # A report that carries a localisation profile (a radiance
            # row: its channel's weighting function convolved with the
            # measured vertical kernel) is weighted by the tabulated
            # profile at every plane's ln p; the point rows keep the
            # Gaspari-Cohn rule above bit for bit.
            prof_sel = gathered["prof_sel"][c0:c1]                 # (Gs, P, M)
            prof_mask = gathered["prof_mask"][c0:c1]               # (Gs, P)
            tabulated = profile_weight(prof_sel[:, None, :, :],
                                       xp.broadcast_to(lnp_plane[:, :, None], (gs, planes, p)), xp)
            wv = xp.where(prof_mask[:, None, :], tabulated, wv)
            del tabulated, prof_sel, prof_mask
        w = wh_sel[:, None, :] * wv.astype(solve_dtype)            # (Gs, planes, P)
        del wv
        good = w > 0
        active = good.any(axis=2)                                  # (Gs, planes)
        winv = xp.where(good, w / err2[:, None, :], 0)
        del w
        if not bool(xp.all(xp.isfinite(winv))):
            raise LetkfError(
                "localised inverse observation error weight/sigma^2 overflowed "
                f"the {config.solve_dtype} solve; use float64 or rescale the units"
            )
        # Only the pairs with a report inside their lens are factored: a
        # pair without one has the matrix (R-1)I/rho exactly and takes the
        # closed form below, so its factorisation would be discarded.  The
        # inactive pairs are 15 to 25 percent of the whole on the case's
        # real hours, and the eigendecomposition is the solve's cost.
        active_flat = xp.nonzero(active.reshape(-1))[0]
        n_active = int(active_flat.size)
        if n_active == 0:
            del winv
            level_batches += 1
            max_batch = max(max_batch, gs * planes)
            _leave_unsolved(csel)
            done[cols_host] = True
            return 0, 0
        wctl_aug = None
        if hybrid and dctl is not None:
            # The augmented solve: [c_e Yb, c_s Ys] (Gs, P, M), one linear
            # system per active pair for the control's weights, formed and
            # released before the members' matrices so the two never stand
            # on the card together.
            _sync_namespace(xp)
            t_hyb = time.perf_counter()
            ys_sub = gathered["ys_all"][c0:c1]                                     # (Gs, P, K)
            yb_aug = xp.concatenate([yb_sub * c_ens, ys_sub * c_sta], axis=2)     # (Gs, P, M)
            cflat_aug = (xp.swapaxes(yb_aug, 1, 2)[:, None, :, :] * winv[:, :, None, :]).reshape(gs, planes * aug, p)
            amat_aug = (cflat_aug @ yb_aug).reshape(gs * planes, aug, aug)[active_flat] + scale_aug * ident_aug[None]
            amat_aug = (amat_aug + xp.swapaxes(amat_aug, 1, 2)) * solve_dtype.type(0.5)
            rhs = (cflat_aug @ dctl[:, :, None]).reshape(gs * planes, aug)[active_flat]    # (Ma, M)
            del yb_aug, cflat_aug
            wctl_aug, hybrid_solver = _batched_solve(xp, amat_aug, rhs)
            if not bool(xp.all(xp.isfinite(wctl_aug))):
                raise LetkfError(
                    "the augmented hybrid solve handed back non-finite control weights; "
                    "do not apply this analysis"
                )
            del amat_aug, rhs
            _sync_namespace(xp)
            hybrid_seconds += time.perf_counter() - t_hyb
        # C = Yb^T diag(w / sigma^2) for every plane of every column: the
        # padding slots' weights are exactly zero, so their columns of C
        # are exact zeros and the reports there drop out of every product.
        cmat = xp.swapaxes(yb_sub, 1, 2)[:, None, :, :] * winv[:, :, None, :]   # (Gs, planes, R, P)
        del winv
        cflat = cmat.reshape(gs, planes * members, p)
        amat = (cflat @ yb_sub).reshape(gs * planes, members, members) + scale_i * ident[None]
        amat = (amat + xp.swapaxes(amat, 1, 2)) * solve_dtype.type(0.5)
        evals, evecs, sweeps = _eigendecompose(xp, amat[active_flat], eigensolver)
        del amat
        if not bool(xp.all(evals > 0)):
            raise LetkfError(
                "the localised LETKF matrix (R-1)I/rho + C Yb came back with a "
                f"non-positive eigenvalue ({float(evals.min())!r}); suspect "
                "non-finite H(x_k) or an error small enough to overflow 1/err^2"
            )
        inv = 1.0 / evals
        pa = (evecs * inv[:, None, :]) @ xp.swapaxes(evecs, 1, 2)          # (Ma, R, R)
        rt = xp.sqrt(inv * solve_dtype.type(members - 1))
        wa = (evecs * rt[:, None, :]) @ xp.swapaxes(evecs, 1, 2)
        del evecs, evals, inv, rt
        wbar = (cflat @ dvec[:, :, None]).reshape(gs * planes, members)[active_flat]   # (Ma, R)
        wbar = xp.einsum("grs,gs->gr", pa, wbar)
        wctl = None
        if dctl is not None and wctl_aug is None:
            wctl = (cflat @ dctl[:, :, None]).reshape(gs * planes, members)[active_flat]
            wctl = xp.einsum("grs,gs->gr", pa, wctl)
        del cmat, cflat, pa
        level_batches += 1
        max_batch = max(max_batch, gs * planes)
        # (Gs, planes) pairs are laid out plane-fastest; the 3-D fields take
        # the first nlev planes of every column, the 2-D fields the last.
        # ``position`` maps a pair to its row in the active set; an inactive
        # pair reads row 0 (a finite matrix) and its result is discarded
        # by the closed-form selection below.
        position = xp.zeros(gs * planes, dtype=xp.int64)
        position[active_flat] = xp.arange(n_active, dtype=xp.int64)
        pairs3 = (gs, nlev)
        idx3 = position[(xp.arange(gs)[:, None] * planes + xp.arange(nlev)[None, :]).reshape(-1)]
        idx2 = position[xp.arange(gs) * planes + nlev]
        active3 = active[:, :nlev]                                          # (Gs, nlev)
        active2 = active[:, nlev]                                           # (Gs,)

        def _transform(xbg, sel, xsg=None):
            """The increment of one field at the selected pairs: ``xbg``
            ``(R, M')`` in the solve dtype, ``sel`` the pair indices,
            ``xsg`` ``(K, M')`` the static draws of the hybrid or None."""
            dbar = xp.einsum("mg,gm->g", xbg, wbar[sel])
            xa = xp.einsum("mg,gmk->kg", xbg, wa[sel])
            if config.rtps_alpha > 0.0:
                if config.relaxation == "rtpp":
                    xa = xa * (1 - alpha) + xbg * alpha
                else:
                    sb = xp.sqrt((xbg ** 2).sum(axis=0) / (members - 1))
                    sa = xp.sqrt((xa ** 2).sum(axis=0) / (members - 1))
                    relax = xp.where(sa > 0, alpha * (sb - sa) / xp.where(sa > 0, sa, 1) + 1, 1)
                    xa = xa * relax[None, :]
            inc = (dbar[None, :] + xa - xbg).astype(work_dtype)
            ctl = None
            if wctl is not None:
                ctl = xp.einsum("mg,gm->g", xbg, wctl[sel]).astype(work_dtype)
            elif wctl_aug is not None:
                xaug = xp.concatenate([xbg * c_ens, xsg * c_sta], axis=0)         # (M, M')
                ctl = xp.einsum("mg,gm->g", xaug, wctl_aug[sel]).astype(work_dtype)
            return inc, ctl

        if fields3:
            for n in fields3:
                view = xb_flat3[n][:, :, csel]                               # (R, nlev, Gs)
                xbw = xp.swapaxes(view, 1, 2)                                # (R, Gs, nlev)
                xbg = xbw.reshape(members, gs * nlev).astype(solve_dtype)
                xsg = None
                if wctl_aug is not None:
                    xsg = xp.swapaxes(xs_flat3[n][:, :, csel], 1, 2).reshape(draws, gs * nlev).astype(solve_dtype)
                inc, ctl = _transform(xbg, idx3, xsg)
                inc = inc.reshape(members, *pairs3)                          # (R, Gs, nlev)
                keep = xp.zeros_like(xbw) if inactive_scale == 1.0 else xbw * step
                xb_flat3[n][:, :, csel] = xp.swapaxes(xp.where(active3[None], inc, keep), 1, 2)
                if ctl is not None:
                    control3[n][:, csel] = xp.swapaxes(
                        xp.where(active3, ctl.reshape(*pairs3), work_dtype.type(0)), 0, 1)
        if fields2:
            for n in fields2:
                view = xb_flat2[n][:, csel]                                  # (R, Gs)
                xbg = view.astype(solve_dtype)
                xsg = None
                if wctl_aug is not None:
                    xsg = xs_flat2[n][:, csel].astype(solve_dtype)
                inc, ctl = _transform(xbg, idx2, xsg)
                keep = xp.zeros_like(view) if inactive_scale == 1.0 else view * step
                xb_flat2[n][:, csel] = xp.where(active2[None], inc, keep)
                if ctl is not None:
                    control2[n][csel] = xp.where(active2, ctl, work_dtype.type(0))
        done[cols_host] = True
        return n_active, int(sweeps)

    def _chunk(j0: int, j1: int, columns: int, i0: int = 0, i1: int | None = None):
        """Gather the rings [j0, j1) at the longitudes [i0, i1) (the whole
        rings when ``i1`` is None) and solve their columns ``columns`` at a
        time.  Every column is solved on its own reports, so a longitude
        cut changes the footprint and nothing else."""
        nonlocal gather_seconds, solve_seconds, active_columns, active_points
        nonlocal max_local, max_padded, dropped, max_sweeps, chunks
        t_gather = time.perf_counter()
        rings = j1 - j0
        i1 = nlon if i1 is None else int(i1)
        whole = i0 == 0 and i1 == nlon
        ni = i1 - i0
        cols_host = (np.arange(j0, j1)[:, None] * nlon + np.arange(i0, i1)[None, :]).reshape(-1)
        pending = np.nonzero(~done[cols_host])[0]
        if pending.size == 0:
            return None
        lat0 = float(np.deg2rad(lat_deg[j0:j1]).min()) - band_rad
        lat1 = float(np.deg2rad(lat_deg[j0:j1]).max()) + band_rad
        cand_np = np.nonzero((obs_lat_np >= lat0) & (obs_lat_np <= lat1))[0]
        csel_all = slice(j0 * nlon, j1 * nlon) if whole else xp.asarray(cols_host)
        if cand_np.size == 0:
            _leave_unsolved(csel_all)
            done[cols_host] = True
            return None
        cand = xp.asarray(cand_np)
        g = rings * ni
        clat = xp.repeat(lat_rad[j0:j1], ni)
        clon = xp.tile(lon_rad[i0:i1], rings)
        cols_flat = xp.asarray(cols_host)
        dist = _geodesic(clat[:, None], clon[:, None],
                         obs.lat_rad[cand][None, :], obs.lon_rad[cand][None, :],
                         radius, xp)                               # (G, C)
        wh = gaspari_cohn(dist / obs.hcut_m[cand][None, :], 1.0)   # (G, C)
        del dist
        nvalid = (wh > 0).sum(axis=1)
        local_max = int(nvalid.max()) if int(nvalid.size) else 0
        if local_max == 0:
            _leave_unsolved(csel_all)
            done[cols_host] = True
            return None
        p = min(local_max, cap)
        if local_max > cap:
            dropped += int(xp.maximum(nvalid - cap, 0).sum())
        max_local = max(max_local, local_max)
        max_padded = max(max_padded, p)
        # The p reports of a column with the largest horizontal weights, ties
        # broken by report index (the lower first).  The weights tie whenever
        # reports share a position (every level of a sounding, the rows of
        # one station): 91 percent of the case's reports do, so an argsort of
        # the weights alone leaves the choice at the cap to the sort
        # implementation, and numpy and cupy chose differently (the device
        # and host increments then differed by up to 0.4 of a field's
        # maximum, 2026-09-07).  The key is the weight quantised to 2^-30
        # (weights closer than that are a tie too) over the reversed index,
        # so one descending integer sort gives one order everywhere.
        # ceil, so a weight above zero never ranks with the zero weights of
        # the reports outside the lens (a floor put a 1e-12 weight there and
        # the pair's activity then depended on the chunk's slot count).
        key = ((xp.ceil(wh * _WEIGHT_QUANTUM).astype(xp.int64) << 32)
               + (xp.int64(0xFFFFFFFF) - xp.arange(int(wh.shape[1]), dtype=xp.int64))[None, :])
        if p < int(wh.shape[1]):
            order = xp.argsort(-key, axis=1)[:, :p]
        else:
            order = xp.argsort(-key, axis=1)
        del key
        wh_sel = xp.take_along_axis(wh, order, axis=1)             # (G, P)
        gidx = cand[order]                                         # (G, P)
        del wh, order
        active_columns += int((nvalid > 0)[xp.asarray(pending)].sum())
        flat_idx = gidx.reshape(-1)
        s = obs.sim[:, flat_idx].reshape(members, g, p).astype(solve_dtype)
        sbar = obs.simbar[flat_idx].reshape(g, p).astype(solve_dtype)
        gathered = {
            "cols_flat": cols_flat,
            "cols_host": cols_host,
            "whole_rings": whole,
            "wh_sel": wh_sel.astype(solve_dtype),
            "yb_all": xp.ascontiguousarray(xp.moveaxis(s - sbar[None], 0, 2)),   # (G, P, R)
            "dvec_all": obs.value[flat_idx].reshape(g, p).astype(solve_dtype) - sbar,
            "dctl_all": (obs.control_innovation[flat_idx].reshape(g, p).astype(solve_dtype)
                         if want_control else None),
            "err2_all": obs.err2[flat_idx].reshape(g, p).astype(solve_dtype),
            "lnp_obs": obs.lnp[flat_idx].reshape(g, p),
            "vcut_obs": obs.vcut[flat_idx].reshape(g, p),
            "prof_sel": None,
            "prof_mask": None,
            "ys_all": None,
        }
        del s, sbar
        if obs.vprof is not None:
            prof_mask = obs.has_profile[flat_idx].reshape(g, p)
            if bool(prof_mask.any()):
                gathered["prof_mask"] = prof_mask
                gathered["prof_sel"] = obs.vprof[flat_idx].reshape(g, p, -1)
        if hybrid and want_control:
            ss = obs.static_sim[:, flat_idx].reshape(draws, g, p).astype(solve_dtype)
            gathered["ys_all"] = xp.ascontiguousarray(xp.moveaxis(ss, 0, 2))     # (G, P, K)
            del ss
        del gidx, flat_idx
        _sync_namespace(xp)
        t_solve = time.perf_counter()
        gather_seconds += t_solve - t_gather
        # Column batches over the pending columns only: a solved column's
        # stack holds its increment and is never read again, so a batch is
        # the longest run of unsolved columns from c0, at most ``columns``.
        c0 = int(pending[0])
        while c0 < g:
            if done[cols_host[c0]]:
                c0 += 1
                continue
            c1 = c0 + 1
            while c1 < g and c1 < c0 + columns and not done[cols_host[c1]]:
                c1 += 1
            try:
                ng, sweeps = _solve_columns(c0, c1, gathered)
            except Exception as exc:  # noqa: BLE001 - only allocation failures are retried
                if not _is_device_memory_error(exc) or columns <= 1:
                    raise
                columns = max(1, columns // 2)
                diagnostics.chunk_oom_shrinks += 1
                _release_device_scratch(xp)
                continue
            active_points += ng
            max_sweeps = max(max_sweeps, sweeps)
            c0 = c1
        _sync_namespace(xp)
        solve_seconds += time.perf_counter() - t_solve
        chunks += 1
        return columns

    j0 = 0
    rings_cap = fixed_rings if fixed_rings is not None else nlat
    columns = columns_per_batch
    segments = 1 if config.chunk_segments is None else min(nlon, max(1, int(config.chunk_segments)))
    first_rings = None
    while j0 < nlat:
        rings = rings_cap if fixed_rings is not None else _rings_from(j0, rings_cap)
        j1 = min(nlat, j0 + rings)
        try:
            if segments == 1:
                shrunk = _chunk(j0, j1, columns)
            else:
                width = -(-nlon // segments)
                shrunk = None
                for i0 in range(0, nlon, width):
                    got = _chunk(j0, j1, columns, i0, min(nlon, i0 + width))
                    if got is not None:
                        columns = got
                        shrunk = got
        except Exception as exc:  # noqa: BLE001 - only allocation failures are retried
            if not _is_device_memory_error(exc):
                raise
            if columns > 1:
                columns = max(1, columns // 2)
            elif rings > 1:
                rings_cap = max(1, rings // 2)
            elif segments < nlon:
                # One ring did not fit at one column per level batch: take
                # the ring in longitude pieces (two, four, ... down to
                # single columns).  The columns are independent, so the
                # analysis is the same to rounding.
                segments = min(nlon, segments * 2)
            else:
                raise LetkfError(
                    "the device refused an allocation with the chunk already at "
                    "one column of one latitude ring and one column per level batch; "
                    "free memory on the card, lower max_local_obs or shrink the "
                    "localisation radius"
                ) from exc
            diagnostics.chunk_oom_shrinks += 1
            _release_device_scratch(xp)
            continue
        if first_rings is None:
            first_rings = rings
        if shrunk is not None:
            columns = shrunk
        j0 = j1

    diagnostics.gather_seconds = gather_seconds
    diagnostics.solve_seconds = solve_seconds
    diagnostics.active_columns = active_columns
    diagnostics.active_points = active_points
    diagnostics.max_local_obs = max_local
    diagnostics.max_padded_slots = max_padded
    diagnostics.dropped_by_cap = dropped
    diagnostics.chunks = chunks
    diagnostics.chunk_rings = int(first_rings or 0)
    diagnostics.chunk_segments = segments
    diagnostics.max_jacobi_sweeps = max_sweeps
    diagnostics.level_batches = level_batches
    diagnostics.max_batch_points = max_batch
    diagnostics.hybrid_solve_seconds = hybrid_seconds
    diagnostics.hybrid_solver = hybrid_solver
    increments = {n: xb[n] for n in names}
    for n in all_names:
        if n not in increments:
            increments[n] = xp.zeros_like(pri[n]) if inactive_scale == 1.0 else xb[n] * step
    t_finish = time.perf_counter()
    _finish(xp, all_names, pri, increments, members, diagnostics)
    if control is not None:
        for n in all_names:
            if not bool(xp.all(xp.isfinite(control[n]))):
                raise LetkfError(
                    f"the control increment for {n!r} is non-finite; do not apply this analysis"
                )
            diagnostics.control_increment_rms[n] = float(xp.sqrt((control[n] ** 2).mean()))
    _sync_namespace(xp)
    diagnostics.finish_seconds = time.perf_counter() - t_finish
    diagnostics.wall_seconds = time.perf_counter() - t_enter
    return PointAnalysis(increments, control)


def _to_host(xp, arr):
    if xp is np:
        return np.asarray(arr)
    return xp.asnumpy(arr)


def _finish(xp, names, pri, increments, members, diagnostics):
    for n in names:
        inc = increments[n]
        if not bool(xp.all(xp.isfinite(inc))):
            raise LetkfError(
                f"analysis increment for {n!r} is non-finite; every specific guard "
                "passed, so this is a numerical failure in the transform: do not "
                "apply this analysis"
            )
        diagnostics.mean_increment_rms[n] = float(xp.sqrt((inc.mean(axis=0) ** 2).mean()))
        post = pri[n] + inc
        pm = post.mean(axis=0, keepdims=True)
        diagnostics.posterior_spread[n] = float(
            xp.sqrt(((post - pm) ** 2).sum(axis=0) / (members - 1)).mean())
