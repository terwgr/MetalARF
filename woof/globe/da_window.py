"""The door's side of observations at their own times (amendment B).

The window itself is the ensemble package's
(:class:`woof.globe.da.window.ObservationWindow`: the batches of one
analysis window, observed column by column as the integration passes the
bins, the control through ``control=True``).  What the door adds:

* :func:`rows_in_window`, the reports whose valid time lies in the
  trailing window ``(t0, t1]`` the cycle fetches for (the real-time cycle
  cannot see reports after ``t1``; a delayed replay could, and the
  package's window would take them into its last bin);
* :func:`binning_sensitivity`, the measured difference between the
  analysis-instant equivalent and the binned equivalent of the same rows
  (what an analysis-instant comparison would have mistaken for an
  innovation), taken on the control once per cycle;
* the LINEARISED observation-space analysis equivalent of a binned row
  (amendment G) and its label: the binned background equivalent plus the
  increment in observation space at the analysis instant,
  ``H(x^b(t_j)) + [H(x^a(t1)) - H(x^b(t1))]``.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .obs_table import ObsRow

LINEARISED_LABEL = (
    "linearised observation-space analysis equivalent: the time-binned "
    "background equivalent plus the increment in observation space at the "
    "analysis instant (H(x^a(t1)) - H(x^b(t1))); the report is not re-compared "
    "with a re-integrated analysis trajectory"
)


def rows_in_window(rows: list[ObsRow], start_utc: dt.datetime, window_start_s: float, window_end_s: float) -> list[ObsRow]:
    """The rows whose valid time lies in ``(t0, t1]`` of model time."""
    out = []
    for row in rows:
        when = row.valid_time if row.valid_time.tzinfo else row.valid_time.replace(tzinfo=dt.timezone.utc)
        t = (when - start_utc).total_seconds()
        if window_start_s < t <= window_end_s + 1.0e-6:
            out.append(row)
    return out


def binning_sensitivity(instant_values, binned_values) -> dict[str, float] | None:
    """rms and largest absolute difference between the analysis-instant
    equivalent and the binned equivalent of the same rows."""
    if instant_values is None or binned_values is None:
        return None
    a = np.asarray(instant_values, dtype=np.float64).reshape(-1)
    b = np.asarray(binned_values, dtype=np.float64).reshape(-1)
    if a.size == 0 or a.shape != b.shape:
        return None
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return None
    diff = a[finite] - b[finite]
    return {"rows": int(finite.sum()), "rms": float(np.sqrt(np.mean(diff ** 2))), "max_abs": float(np.max(np.abs(diff)))}


def linearised_analysis_equivalent(binned_background, instant_background, instant_analysis) -> np.ndarray:
    """``H(x^b(t_j)) + [H(x^a(t1)) - H(x^b(t1))]`` (:data:`LINEARISED_LABEL`)."""
    return (np.asarray(binned_background, dtype=np.float64)
            + np.asarray(instant_analysis, dtype=np.float64) - np.asarray(instant_background, dtype=np.float64))


__all__ = [
    "LINEARISED_LABEL",
    "binning_sensitivity",
    "linearised_analysis_equivalent",
    "rows_in_window",
]
