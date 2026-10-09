"""Observations at their own times (amendment B).

An hourly cycle is a window, not an instant: a report at 12:08 is compared
with the state at 12:08, not with the state at 13:00.  The
:class:`ObservationWindow` holds the batches of one analysis window and,
as the integration passes through the window, evaluates each member's
observation-space equivalent ``y_ij = H_j(x_i(t_j))`` for the reports whose
time falls in the bin the step lands on; what is retained is the
observation-space trajectory (the ``(R, n)`` array of the batch, filled
column by column), never a full state per step.

Release 1 uses validated short time bins: the window ``(start, end]`` is
cut into ``bin_s``-wide bins, a report joins the bin its time falls in
(reports before the window join the first bin, reports after it the last),
and each bin is observed at its END, the step the bin closes on (every
report in the bin has happened by then, so the comparison is causal, and
a report is never more than one bin from its state).  A bin of the
window's whole length reproduces the instantaneous comparison at the
analysis time; the OSSE measures the sensitivity to the bin width
(``family bins``).  Rows a window never saw (an integration that stopped
early) are observed at the end by :meth:`finish`, and the record says how
many rows were observed at their own time and how many at the end.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

from .observations import PointObs
from .operators import evaluate_batches

__all__ = ["ObservationWindow", "batches_unevaluated"]


def batches_unevaluated(rows, operators) -> list[PointObs]:
    """The :class:`PointObs` batches of an :class:`~woof.globe.obs_table.ObsRow`
    list, one per (source, variable), with ``simulated`` None, the aloft
    rows' ``ln_pressure`` filled and the surface rows' NaN until a window
    or the analysis operators fill it; each batch's ``operator`` is the
    operators' batch operator."""
    if not rows:
        return []
    lat = np.array([r.latitude_deg for r in rows])
    lon = np.array([r.longitude_deg for r in rows])
    elev = np.array([r.elevation_m for r in rows])
    level = np.array([np.nan if r.level_pa is None else r.level_pa for r in rows])
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((row.source, row.variable), []).append(index)
    batches = []
    for (source, variable), indices in groups.items():
        idx = np.asarray(indices)
        batches.append(PointObs(
            stream=source, variable=variable,
            latitude_deg=lat[idx], longitude_deg=lon[idx],
            ln_pressure=np.where(np.isnan(level[idx]), np.nan, np.log(np.where(np.isnan(level[idx]), 1.0, level[idx]))),
            surface=np.isnan(level[idx]),
            value=np.array([rows[i].value for i in idx]),
            error=np.array([rows[i].error for i in idx]),
            simulated=None,
            elevation_m=elev[idx],
            identity=np.array([rows[i].identity_hash() for i in idx], dtype=object),
            valid_time=[rows[i].valid_time for i in idx],
            operator=operators.batch_operator,
            measurement=np.array([rows[i].measurement for i in idx], dtype=object),
        ))
    return batches


@dataclass
class ObservationWindow:
    """The reports of one analysis window and their observation-space
    trajectories.

    batches
        The :class:`PointObs` batches (every row with a ``valid_time``).
    start_s, end_s
        The window in model seconds, ``(start, end]``; the analysis is at
        ``end_s``.
    epoch
        The UTC instant of model time zero.
    bin_s
        The bin width in seconds; the window must be a whole number of
        bins.
    dt_s
        The integration step the bins snap to.
    """

    batches: list[PointObs]
    start_s: float
    end_s: float
    epoch: dt.datetime
    bin_s: float
    dt_s: float
    #: Filled at construction: per batch the bin index of every row.
    _bins: list[np.ndarray] = field(default_factory=list, repr=False)
    _observed: list[np.ndarray] = field(default_factory=list, repr=False)
    _control_observed: list[np.ndarray] = field(default_factory=list, repr=False)
    _at_own_time: dict[str, int] = field(default_factory=dict, repr=False)
    _at_end: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        span = float(self.end_s) - float(self.start_s)
        if span <= 0.0 or self.bin_s <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("ObservationWindow needs end_s > start_s and positive bin_s and dt_s")
        n = span / float(self.bin_s)
        if abs(n - round(n)) > 1.0e-6:
            raise ValueError(f"the window ({span:g} s) is not a whole number of {self.bin_s:g} s bins")
        self.nbins = int(round(n))
        steps = span / float(self.dt_s)
        if abs(steps - round(steps)) > 1.0e-6:
            raise ValueError(f"the window ({span:g} s) is not a whole number of {self.dt_s:g} s steps")
        epoch = self.epoch if self.epoch.tzinfo else self.epoch.replace(tzinfo=dt.timezone.utc)
        self.epoch = epoch.astimezone(dt.timezone.utc)
        for batch in self.batches:
            if batch.valid_time is None:
                raise ValueError(
                    f"batch {batch.stream!r}/{batch.variable!r} carries no valid times; a window "
                    "compares each report with the state at its own time"
                )
            t = np.array([(v.astimezone(dt.timezone.utc) - self.epoch).total_seconds() for v in batch.valid_time])
            bins = np.floor((t - float(self.start_s)) / float(self.bin_s)).astype(int)
            # A report exactly at the window's end belongs to the last bin.
            bins = np.clip(bins, 0, self.nbins - 1)
            self._bins.append(bins)
            self._observed.append(np.zeros(batch.count, dtype=bool))
            self._control_observed.append(np.zeros(batch.count, dtype=bool))
        self._at_own_time = {"members": 0, "control": 0}
        self._at_end = {"members": 0, "control": 0}

    # -- bin geometry -------------------------------------------------------

    def bin_time_s(self, index: int) -> float:
        """The model time bin ``index`` is observed at: the step time
        nearest the bin's end (the window's end for the last bin)."""
        close = float(self.start_s) + (index + 1) * float(self.bin_s)
        steps = round((close - float(self.start_s)) / float(self.dt_s))
        t = float(self.start_s) + steps * float(self.dt_s)
        return min(max(t, float(self.start_s) + float(self.dt_s)), float(self.end_s))

    def due(self, time_s: float) -> list[int]:
        """The bins observed at ``time_s`` (within half a step)."""
        half = 0.5 * float(self.dt_s)
        return [i for i in range(self.nbins) if abs(self.bin_time_s(i) - float(time_s)) < half]

    def rows_in_bins(self, batch_index: int, bins: list[int]) -> np.ndarray:
        chosen = np.isin(self._bins[batch_index], np.asarray(bins, dtype=int))
        return np.nonzero(chosen)[0]

    # -- observing ----------------------------------------------------------

    def observe(self, states, time_s: float, operators, *, control: bool = False) -> int:
        """Evaluate ``operators`` on ``states`` for every row whose bin is
        due at ``time_s``: into ``simulated`` (``(R, n)``, the members) or,
        with ``control``, into ``control_simulated`` (``(1, n)``).  Surface
        rows take their ``ln_pressure`` from the members' evaluation.
        Returns the rows observed."""
        bins = self.due(time_s)
        if not bins:
            return 0
        return self._observe_bins(states, bins, operators, control=control, at_end=False)

    def _observe_bins(self, states, bins, operators, *, control: bool, at_end: bool) -> int:
        total = 0
        flags = self._control_observed if control else self._observed
        picks = []
        for b, batch in enumerate(self.batches):
            rows = self.rows_in_bins(b, bins)
            rows = rows[~flags[b][rows]]
            picks.append(rows)
        if any(r.size for r in picks):
            # One evaluation over every batch's due rows (the wind synthesis
            # is the cost and is paid once per bin, not once per batch).
            evaluate_batches(operators, states, self.batches, rows=picks,
                             target="control_simulated" if control else "simulated")
        for b, rows in enumerate(picks):
            if rows.size == 0:
                continue
            flags[b][rows] = True
            total += int(rows.size)
        key = "control" if control else "members"
        if at_end:
            self._at_end[key] += total
        else:
            self._at_own_time[key] += total
        return total

    def finish(self, states, operators, *, control: bool = False) -> int:
        """Observe every row not yet observed on ``states`` (the end of the
        window), and record how many needed it."""
        pending = [b for b in range(len(self.batches))
                   if not (self._control_observed if control else self._observed)[b].all()]
        if not pending:
            return 0
        return self._observe_bins(states, list(range(self.nbins)), operators, control=control, at_end=True)

    def complete(self, *, control: bool = False) -> bool:
        flags = self._control_observed if control else self._observed
        return all(f.all() for f in flags)

    def drop_unobserved(self) -> list[PointObs]:
        """The batches with the rows no state ever observed removed (a
        window that finished without ``finish``): what the analysis takes."""
        out = []
        for b, batch in enumerate(self.batches):
            keep = self._observed[b]
            if batch.control_simulated is not None:
                keep = keep & self._control_observed[b]
            out.append(batch.subset(keep) if not keep.all() else batch)
        return out

    def record(self) -> dict[str, object]:
        """The receipt row: bin width, bins, rows per bin, how many rows were
        observed at their own time and how many at the window's end."""
        per_bin = np.zeros(self.nbins, dtype=int)
        for bins in self._bins:
            per_bin += np.bincount(bins, minlength=self.nbins)[: self.nbins]
        return {
            "rule": (
                "each report is compared with the state at the step its bin closes on; "
                "the window (start, end] is cut into bin_s-wide bins, reports before the window "
                "join the first bin and after it the last"
            ),
            "start_s": float(self.start_s), "end_s": float(self.end_s), "bin_s": float(self.bin_s),
            "dt_s": float(self.dt_s), "bins": int(self.nbins),
            "bin_observe_times_s": [self.bin_time_s(i) for i in range(self.nbins)],
            "rows_per_bin": [int(v) for v in per_bin],
            "rows_observed_at_own_time": dict(self._at_own_time),
            "rows_observed_at_end": dict(self._at_end),
            "complete": {"members": self.complete(), "control": self.complete(control=True)},
        }
