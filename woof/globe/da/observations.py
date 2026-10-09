"""The observation contract of the WOOF global ensemble filter.

One :class:`PointObs` is one stream's reports of one neutral variable at
arbitrary points on the sphere and in ln p, with the forward operator
already evaluated on every member (``simulated``, ``(R, n)``).  The filter
never asks what the operator was: a station's 2 m temperature, a
sounding's 500 hPa wind, a satellite motion vector, a refractivity profile
point, a clear-sky brightness temperature and an analysis pseudo-report
all arrive as the same rows, and a stream joins the system by producing
this structure, not by a code path inside the filter (the arbitrary
acceptance test).

Who fills ``simulated``: for the neutral point vocabulary
(:data:`woof.globe.obs_table.VARIABLE_TABLE`) the filter's own
member-batched operators (:mod:`woof.globe.da.operators`) do it
from :class:`~woof.globe.obs_table.ObsRow` lists through
:func:`batches_from_rows`.  A stream with its own operator (a radiance, a
refractivity) evaluates it per member and hands the filled batch over.

``ln_pressure`` is the report's position in the vertical localisation
metric.  An aloft report carries ``ln(level_pa)``; a surface report is
placed at the model's ln ps at the station (the operator fills it), so a
2 m report sits at the column's surface and the vertical weight to level
k is ``GC(|ln p_k - ln ps| / cutoff)``.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

#: The ln p axis a row's vertical localisation PROFILE is tabulated on
#: (:attr:`PointObs.localisation_profile`): 0.05 in ln p from 10 Pa to
#: 120 kPa, 188 nodes.  A radiance row (a weighting function, not a level)
#: carries its localisation weight as a function of the analysis level's
#: ln p on this axis, and the filter reads it by linear interpolation at
#: every level of every column; a point row carries none and takes the
#: Gaspari-Cohn rule on its own ln p.
LOCALISATION_AXIS_LNP = np.arange(math.log(10.0), math.log(120_000.0) + 1.0e-9, 0.05)


@dataclass
class PointObs:
    """One stream, one variable, ``n`` reports.

    stream
        The stream's name (``iem-asos-csv``, ``igra2-levels-csv``,
        ``goes-abi-dmw``, ...), as the receipt groups O-B and O-A.
    variable
        The neutral variable (``temperature_k``, ``wind_u_m_s``, ...) or a
        stream's own (``brightness_temperature_k``, ``refractivity_n``).
    latitude_deg, longitude_deg
        ``(n,)`` degrees; longitude in [-180, 360).
    ln_pressure
        ``(n,)`` the report's ln p (Pa) in the localisation metric; NaN
        for a surface row until the operator fills it with the model's
        ln ps at the station.
    surface
        ``(n,)`` bool: a surface row (2 m, 10 m, station pressure).
    value, error
        ``(n,)`` the report and its error standard deviation, the
        variable's units.
    simulated
        ``(R, n)`` H(x_k) for every background member, or None until the
        operator runs.
    control_simulated
        ``(1, n)`` H(x_H^b) of the control (deterministic, high-resolution)
        background, or None until its operator runs; the control's
        innovation ``y - H(x_H^b)`` is formed from it (amendment A).
    identity
        ``(n,)`` the rows' identity hashes (``ObsRow.identity_hash``) for
        the assimilation chain; empty strings when the stream has none.
    valid_time
        The reports' valid instants (UTC), or None.
    horizontal_cutoff_km, vertical_cutoff_lnp
        Optional per-batch localisation overrides; ``None`` takes the
        filter's ``FilterOptions`` rule for the variable.
    localisation_profile
        ``(n, M)`` or None: the row's vertical localisation weight as a
        function of the analysis level's ln p, tabulated on
        :data:`LOCALISATION_AXIS_LNP` (``M`` nodes), for a report that
        senses a LAYER rather than a level (a radiance: the channel's
        weighting function convolved with the Gaspari-Cohn kernel of the
        stream's vertical length, the model-space placement).  The filter
        reads it by linear interpolation at every level of every column
        and ignores ``ln_pressure`` and ``vertical_cutoff_lnp`` for the
        vertical weight of such a row (``ln_pressure`` still places the
        row for thinning).  None on every point stream.
    """

    stream: str
    variable: str
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    ln_pressure: np.ndarray
    surface: np.ndarray
    value: np.ndarray
    error: np.ndarray
    simulated: np.ndarray | None = None
    control_simulated: np.ndarray | None = None
    #: ``(K, n)`` the observation-space perturbations of the K static
    #: draws of the hybrid covariance, ``H(x_mean + x_s) - H(x_mean)``,
    #: or None until the hybrid analysis fills them (the analysis fills
    #: them on the assimilated rows after quality control).
    static_simulated: np.ndarray | None = None
    #: ``(n,)`` station elevation above sea level (m) for a surface row
    #: (the pressure and 2 m reductions need it); zeros when the stream
    #: has none, ignored aloft.
    elevation_m: np.ndarray | None = None
    identity: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    valid_time: list[dt.datetime] | None = None
    horizontal_cutoff_km: float | None = None
    vertical_cutoff_lnp: float | None = None
    localisation_profile: np.ndarray | None = None
    #: The stream's own forward operator, ``operator(members, batch) ->
    #: (R, batch.count)`` over a list of member states and the batch (or a
    #: subset of it) whose rows it evaluates, for a variable the package
    #: cannot evaluate itself (a radiance, a refractivity).  The analysis
    #: calls it on the analysed members for O-A; a batch of a foreign
    #: variable without one is judged on O-B alone and reported INCOMPLETE.
    operator: object = None
    #: Bookkeeping the quality control fills: why rows left, how many.
    rejections: dict[str, int] = field(default_factory=dict)
    #: ``(n,)`` what each row measures (the table's ``measurement`` column,
    #: :data:`woof.globe.obs_table.MEASUREMENT_TABLE`); empty
    #: strings when the stream states none.  The operators read it where a
    #: measurement needs its own error treatment (a motion vector's height
    #: assignment).
    measurement: np.ndarray | None = None
    #: ``(n,)`` the member-mean wind change over one height-assignment
    #: sigma at the row's assigned pressure (m/s), filled by the operators
    #: for motion-vector rows and NaN elsewhere: the situation-dependent
    #: part of a motion vector's error the door folds into ``error``.
    assignment_shear: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = int(np.asarray(self.value).size)
        for name in ("latitude_deg", "longitude_deg", "ln_pressure", "value", "error"):
            arr = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if arr.size != n:
                raise ValueError(
                    f"PointObs({self.stream!r}, {self.variable!r}).{name} has "
                    f"{arr.size} rows where value has {n}"
                )
            object.__setattr__(self, name, arr)
        surface = np.asarray(self.surface, dtype=bool).reshape(-1)
        if surface.size != n:
            raise ValueError("PointObs.surface must have one flag per row")
        self.surface = surface
        if self.elevation_m is None:
            self.elevation_m = np.zeros(n, dtype=np.float64)
        else:
            elev = np.asarray(self.elevation_m, dtype=np.float64).reshape(-1)
            if elev.size != n:
                raise ValueError("PointObs.elevation_m must have one value per row")
            self.elevation_m = elev
        if self.identity is None or np.asarray(self.identity).size == 0:
            self.identity = np.array([""] * n, dtype=object)
        else:
            ident = np.asarray(self.identity, dtype=object).reshape(-1)
            if ident.size != n:
                raise ValueError("PointObs.identity must have one hash per row")
            self.identity = ident
        if self.measurement is None or np.asarray(self.measurement).size == 0:
            self.measurement = np.array([""] * n, dtype=object)
        else:
            meas = np.asarray(self.measurement, dtype=object).reshape(-1)
            if meas.size != n:
                raise ValueError("PointObs.measurement must have one label per row")
            self.measurement = meas
        if self.assignment_shear is None:
            self.assignment_shear = np.full(n, np.nan)
        else:
            shear = np.asarray(self.assignment_shear, dtype=np.float64).reshape(-1)
            if shear.size != n:
                raise ValueError("PointObs.assignment_shear must have one value per row")
            self.assignment_shear = shear
        if self.simulated is not None:
            sim = np.asarray(self.simulated)
            if sim.ndim != 2 or sim.shape[1] != n:
                raise ValueError(
                    f"PointObs.simulated must be (members, {n}), got {sim.shape}"
                )
        if self.control_simulated is not None:
            ctl = np.asarray(self.control_simulated, dtype=np.float64).reshape(1, -1)
            if ctl.shape[1] != n:
                raise ValueError(
                    f"PointObs.control_simulated must be (1, {n}), got {np.asarray(self.control_simulated).shape}"
                )
            self.control_simulated = ctl
        if self.localisation_profile is not None:
            prof = np.asarray(self.localisation_profile, dtype=np.float64)
            if prof.ndim != 2 or prof.shape != (n, LOCALISATION_AXIS_LNP.size):
                raise ValueError(
                    f"PointObs.localisation_profile must be ({n}, {LOCALISATION_AXIS_LNP.size}) on "
                    f"LOCALISATION_AXIS_LNP, got {prof.shape}"
                )
            if n and (not np.all(np.isfinite(prof)) or np.any(prof < 0.0) or np.any(prof > 1.0 + 1e-12)
                      or not np.all(prof.max(axis=1) > 0.0)):
                raise ValueError(
                    "PointObs.localisation_profile must be finite weights in [0, 1] with a positive "
                    "maximum on every row: a row that localises nowhere would be assimilated nowhere "
                    "while its receipt counted it"
                )
            self.localisation_profile = prof
        if self.static_simulated is not None:
            sta = np.asarray(self.static_simulated, dtype=np.float64)
            if sta.ndim != 2 or sta.shape[1] != n:
                raise ValueError(
                    f"PointObs.static_simulated must be (static draws, {n}), got {sta.shape}"
                )
            self.static_simulated = sta
        if n:
            if not np.all(np.isfinite(self.value)):
                raise ValueError("PointObs.value has non-finite rows; drop them before the filter")
            if not np.all(np.isfinite(self.error)) or not np.all(self.error > 0.0):
                raise ValueError(
                    "PointObs.error must be finite and positive on every row: it is "
                    "a standard deviation, and zero means a report the filter must "
                    "match exactly, which it cannot represent"
                )
            if np.any(np.abs(self.latitude_deg) > 90.0):
                raise ValueError("PointObs.latitude_deg outside [-90, 90]")

    @property
    def count(self) -> int:
        return int(self.value.size)

    @property
    def members(self) -> int | None:
        return None if self.simulated is None else int(np.asarray(self.simulated).shape[0])

    def subset(self, keep) -> "PointObs":
        """A copy holding the rows ``keep`` (bool mask or index array)."""
        keep = np.asarray(keep)
        if keep.dtype == bool:
            keep = np.nonzero(keep)[0]
        return PointObs(
            stream=self.stream, variable=self.variable,
            latitude_deg=self.latitude_deg[keep],
            longitude_deg=self.longitude_deg[keep],
            ln_pressure=self.ln_pressure[keep],
            surface=self.surface[keep],
            value=self.value[keep], error=self.error[keep],
            simulated=None if self.simulated is None else np.asarray(self.simulated)[:, keep],
            control_simulated=None if self.control_simulated is None else self.control_simulated[:, keep],
            static_simulated=None if self.static_simulated is None else self.static_simulated[:, keep],
            elevation_m=self.elevation_m[keep],
            identity=self.identity[keep],
            valid_time=None if self.valid_time is None else [self.valid_time[int(k)] for k in keep],
            horizontal_cutoff_km=self.horizontal_cutoff_km,
            vertical_cutoff_lnp=self.vertical_cutoff_lnp,
            localisation_profile=None if self.localisation_profile is None else self.localisation_profile[keep],
            operator=self.operator,
            rejections=dict(self.rejections),
            measurement=self.measurement[keep],
            assignment_shear=self.assignment_shear[keep],
        )

    def level_pa(self) -> np.ndarray:
        """``(n,)`` the report pressure the operators take: NaN for a
        surface row, ``exp(ln_pressure)`` aloft."""
        return np.where(self.surface, np.nan, np.exp(self.ln_pressure))

    def innovation(self) -> np.ndarray:
        """``y - mean_k H(x_k)``, ``(n,)``."""
        if self.simulated is None:
            raise ValueError("PointObs.simulated is not filled; run the operators first")
        return self.value - np.asarray(self.simulated, dtype=np.float64).mean(axis=0)

    def control_innovation(self) -> np.ndarray:
        """``y - H(x_H^b)``, ``(n,)``, the control's own innovation."""
        if self.control_simulated is None:
            raise ValueError("PointObs.control_simulated is not filled; run the control operator first")
        return self.value - self.control_simulated[0]

    def spread(self) -> np.ndarray:
        """Background ensemble spread in observation space, ``(n,)``."""
        if self.simulated is None:
            raise ValueError("PointObs.simulated is not filled; run the operators first")
        sim = np.asarray(self.simulated, dtype=np.float64)
        r = sim.shape[0]
        return np.sqrt(((sim - sim.mean(axis=0)) ** 2).sum(axis=0) / max(r - 1, 1))


def concatenate(batches: list[PointObs]) -> PointObs:
    """One batch from several of the same stream and variable."""
    if not batches:
        raise ValueError("nothing to concatenate")
    first = batches[0]
    for b in batches[1:]:
        if (b.stream, b.variable) != (first.stream, first.variable):
            raise ValueError("concatenate needs batches of one stream and variable")
    sims = [b.simulated for b in batches]
    ctls = [b.control_simulated for b in batches]
    stas = [b.static_simulated for b in batches]
    return PointObs(
        stream=first.stream, variable=first.variable,
        latitude_deg=np.concatenate([b.latitude_deg for b in batches]),
        longitude_deg=np.concatenate([b.longitude_deg for b in batches]),
        ln_pressure=np.concatenate([b.ln_pressure for b in batches]),
        surface=np.concatenate([b.surface for b in batches]),
        value=np.concatenate([b.value for b in batches]),
        error=np.concatenate([b.error for b in batches]),
        simulated=None if any(s is None for s in sims) else np.concatenate(
            [np.asarray(s) for s in sims], axis=1),
        control_simulated=None if any(c is None for c in ctls) else np.concatenate(ctls, axis=1),
        static_simulated=None if any(s is None for s in stas) else np.concatenate(stas, axis=1),
        elevation_m=np.concatenate([b.elevation_m for b in batches]),
        identity=np.concatenate([b.identity for b in batches]),
        valid_time=None if any(b.valid_time is None for b in batches) else [
            t for b in batches for t in b.valid_time],
        horizontal_cutoff_km=first.horizontal_cutoff_km,
        vertical_cutoff_lnp=first.vertical_cutoff_lnp,
        localisation_profile=(
            None if any(b.localisation_profile is None for b in batches)
            else np.concatenate([b.localisation_profile for b in batches], axis=0)),
        operator=first.operator,
        measurement=np.concatenate([b.measurement for b in batches]),
        assignment_shear=np.concatenate([b.assignment_shear for b in batches]),
    )


__all__ = ["LOCALISATION_AXIS_LNP", "PointObs", "concatenate"]
