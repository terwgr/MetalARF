"""The door's control options and the comparison record of amendment A.

Design amendment A (2026-09-06): the T255 control is NOT updated by the
ensemble-mean increment.  For every observation the control's innovation
``d_H = y - H(x_H^b)`` is formed on the control's own background and the
T127 ensemble supplies the covariance alone; the ensemble package forms
that analysis (:func:`woof.globe.da.analysis.analyze_ensemble`
with a :class:`~woof.globe.da.analysis.ControlBackground`: one
local solve returns the members' increments and the control's, the control
increment tapered by degree (amendment C, ``FilterOptions.
transfer_taper_start_degree`` / ``transfer_taper_end_degree``), embedded in
the control's triangle and added).  The door's part is here:

* :class:`ControlOptions`, the door's spelling of how the control is
  analysed and how the ensemble follows it: which increment the control
  takes (its own, the default; or the ensemble-mean transfer as the
  COMPARISON experiment, by name), how far the ensemble is recentred on
  it, the taper degrees, how the increment is inserted (direct, or the
  incremental analysis update of the cycle door), and the hybrid
  covariance (the ensemble weight beta, the static covariance table and
  the draws per analysis; :mod:`woof.globe.da.static_covariance`).
  :meth:`ControlOptions.filter_options` lays the shared settings over the
  package's ``FilterOptions`` so the two never disagree.
* :func:`mean_increment_comparison`, the record that lays the
  ensemble-mean increment (never applied) beside the control's own.
* :func:`degree_power`, power per total degree of a spectral field (the
  anchor's departure spectrum reads it).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from .constants import KAPPA, REFERENCE_PRESSURE_PA
from .da.options import DEFAULT_HYBRID_BETA

INCREMENT_SOURCES = ("control", "ensemble-mean")
INCREMENT_APPLICATIONS = ("direct", "iau")
RECENTRE_MODES = ("increment", "state")


@dataclass(frozen=True)
class ControlOptions:
    """How the control member is analysed and how the ensemble follows it.

    increment_source
        ``control`` (the default, amendment A): the control's own increment
        from its own innovations through the ensemble covariance.
        ``ensemble-mean``: the comparison experiment, the ensemble-mean
        increment tapered, embedded and applied (the old path), allowed by
        name for the twin's comparison arm and reported as such.
    recentre_fraction
        1.0 recentres the ensemble mean on the control analysis restricted
        to the ensemble truncation (full recentring); a fraction in (0, 1)
        moves it that far (partial recentring, an experiment); 0 leaves the
        ensemble mean where its own analysis put it.
    recentre_mode
        ``increment`` (the default, the package's decision 18 by
        measurement): the ensemble-mean increment is replaced by the
        control's increment (anchor included) truncated to the ensemble
        triangle, so the members keep their own terrain-consistent
        background; ``state``: the control analysis truncated to the
        ensemble truncation replaces the ensemble mean (on a control whose
        orography is finer than the ensemble's the truncated state carries
        the finer terrain's surface pressure onto the coarser grid).
    taper_full_degree, taper_zero_degree
        The spectral taper of the control increment: weight one at and
        below ``taper_full_degree``, zero at and above ``taper_zero_degree``,
        raised-cosine between.  ``None`` takes the package's defaults (0.6
        and 1.0 times the ensemble truncation).
    increment_application
        ``iau`` (the default since 2026-09-06) re-integrates the window
        from its start with the increment added in equal parts at every
        step (the cycle door's incremental analysis update; the members
        take theirs over the next window through the package); ``direct``
        inserts the control increment at the analysis instant.
    hybrid_beta, static_covariance, static_samples
        The hybrid covariance ``beta B_ens + (1 - beta) B_static`` (one
        positive-semidefinite representation inside the localised solve):
        the ensemble weight in (0, 1], the static covariance table
        (``packaged``, the shipped lagged-forecast estimate, or a path
        written by ``woof global da static-covariance``; ``None`` with
        beta 1 only) and the static draws per analysis.  ``None`` for the
        table and the draws keeps the package's own values.
    """

    increment_source: str = "control"
    recentre_fraction: float = 1.0
    recentre_mode: str = "increment"
    taper_full_degree: int | None = None
    taper_zero_degree: int | None = None
    increment_application: str = "iau"
    hybrid_beta: float = DEFAULT_HYBRID_BETA
    static_covariance: str | None = "packaged"
    static_samples: int | None = None

    def __post_init__(self) -> None:
        if self.increment_source not in INCREMENT_SOURCES:
            raise ValueError(f"increment_source must be one of {INCREMENT_SOURCES}")
        f = float(self.recentre_fraction)
        if not math.isfinite(f) or not 0.0 <= f <= 1.0:
            raise ValueError("recentre_fraction must lie in [0, 1]")
        if self.recentre_mode not in RECENTRE_MODES:
            raise ValueError(f"recentre_mode must be one of {RECENTRE_MODES}")
        for name in ("taper_full_degree", "taper_zero_degree"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or int(value) < 0):
                raise ValueError(f"{name} must be a nonnegative whole number or None")
        if (self.taper_full_degree is not None and self.taper_zero_degree is not None
                and int(self.taper_zero_degree) <= int(self.taper_full_degree)):
            raise ValueError("taper_zero_degree must exceed taper_full_degree")
        if self.increment_application not in INCREMENT_APPLICATIONS:
            raise ValueError(f"increment_application must be one of {INCREMENT_APPLICATIONS}")
        beta = float(self.hybrid_beta)
        if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
            raise ValueError(f"hybrid_beta must lie in (0, 1], got {beta!r}")
        if beta < 1.0 and self.static_covariance is None:
            raise ValueError(
                f"hybrid_beta {beta} asks for a static covariance's share of the gain and "
                "static_covariance names no table (packaged or a path)"
            )
        if self.static_samples is not None and (isinstance(self.static_samples, bool) or int(self.static_samples) < 1):
            raise ValueError("static_samples must be a positive whole number or None")

    def filter_options(self, base):
        """The package's ``FilterOptions`` with this record's shared
        settings laid over it: the taper degrees when named, the recentring
        fraction, the hybrid (beta, the table, the draws when named); the
        members take their increment over the next window
        (``increment_application``, ``iau`` by default since 2026-09-06), the
        control's insertion is the cycle door's."""
        kwargs = {"recentering_fraction": float(self.recentre_fraction),
                  "recentering_mode": str(self.recentre_mode),
                  "hybrid_beta": float(self.hybrid_beta),
                  "static_covariance": self.static_covariance}
        if self.taper_full_degree is not None:
            kwargs["transfer_taper_start_degree"] = int(self.taper_full_degree)
        if self.taper_zero_degree is not None:
            kwargs["transfer_taper_end_degree"] = int(self.taper_zero_degree)
        if self.static_samples is not None:
            kwargs["static_samples"] = int(self.static_samples)
        return dataclasses.replace(base, **kwargs)

    def taper_degrees_for(self, ensemble_truncation: int) -> tuple[int, int]:
        """``(start, end)`` of the control increment's taper at
        ``ensemble_truncation``: this record's degrees where named, the
        package's defaults otherwise."""
        from .da.options import FilterOptions

        return self.filter_options(FilterOptions()).taper_degrees(int(ensemble_truncation))

    def identity(self) -> dict[str, object]:
        return {
            "increment_source": self.increment_source,
            "recentre_fraction": float(self.recentre_fraction),
            "recentre_mode": self.recentre_mode,
            "taper_full_degree": self.taper_full_degree,
            "taper_zero_degree": self.taper_zero_degree,
            "increment_application": self.increment_application,
            "hybrid_beta": float(self.hybrid_beta),
            "static_covariance": None if self.static_covariance is None else str(self.static_covariance),
            "static_samples": None if self.static_samples is None else int(self.static_samples),
        }


def degree_power(coeff) -> np.ndarray:
    """Power per total degree of a scalar coefficient stack ``(..., n,
    m)``, summed over the leading axes and over m (with the factor two
    for m > 0), divided by 4 pi so the sum over degrees is the grid mean
    square of the field (per level, summed over levels)."""
    c = np.asarray(coeff)
    mult = np.where(np.arange(c.shape[-1]) == 0, 1.0, 2.0)
    power = mult * (c.real ** 2 + c.imag ** 2)
    axes = tuple(range(c.ndim - 2)) + (c.ndim - 1,)
    return np.sum(power, axis=axes) / (4.0 * math.pi)


def _summarise_increment(increment_spectral: dict, model, transform, atmosphere) -> dict[str, float]:
    """rms of a spectral increment in physical units on the grid of
    ``transform`` (temperature K through the local Exner function of
    ``atmosphere``, wind m/s, ln ps, qv)."""
    backend = transform.backend
    to_numpy = backend.to_numpy
    xp = backend.xp
    out: dict[str, float] = {}

    def dev(coeff):
        return xp.asarray(coeff, dtype=backend.complex_dtype)

    theta = increment_spectral.get("theta")
    if theta is not None:
        g = model.grid_state(atmosphere, only=("p_full",))
        exner = np.asarray(to_numpy((g["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA), dtype=np.float64)
        model.release_syntheses()
        grid_theta = np.asarray(to_numpy(transform.inverse(dev(theta))), dtype=np.float64)
        out["temperature_k_rms"] = float(np.sqrt(np.mean((grid_theta * exner) ** 2)))
    zeta = increment_spectral.get("vorticity")
    div = increment_spectral.get("divergence")
    if zeta is not None and div is not None:
        u, v = model.vector.wind_from_vordiv(dev(zeta), dev(div))
        out["wind_m_s_rms"] = float(np.sqrt(np.mean(
            np.asarray(to_numpy(u), dtype=np.float64) ** 2 + np.asarray(to_numpy(v), dtype=np.float64) ** 2)))
    for key, name in (("ln_surface_pressure_rms", "log_surface_pressure"), ("qv_kg_kg_rms", "qv")):
        coeff = increment_spectral.get(name)
        if coeff is not None:
            grid = np.asarray(to_numpy(transform.inverse(dev(coeff))), dtype=np.float64)
            out[key] = float(np.sqrt(np.mean(grid ** 2)))
    return out


def mean_increment_comparison(
    ensemble_mean_increment: dict, control_increment_ens: dict, model, transform, atmosphere,
) -> dict[str, object]:
    """The comparison record of amendment A: the ensemble-mean increment
    the old path applied against the control's own, both at the ensemble
    truncation, rms per field and the rms of their difference."""
    xp = transform.backend.xp
    mean = {k: xp.asarray(v) for k, v in ensemble_mean_increment.items() if v is not None}
    control = {k: xp.asarray(v) for k, v in control_increment_ens.items() if v is not None}
    common = sorted(set(mean) & set(control))
    difference = {name: control[name] - mean[name] for name in common}
    return {
        "note": (
            "the ensemble-mean increment is NOT applied to the control (amendment A); "
            "it is recorded here beside the control's own increment for the twin's "
            "comparison arm"
        ),
        "ensemble_mean_increment_rms": _summarise_increment(mean, model, transform, atmosphere),
        "control_increment_rms": _summarise_increment(control, model, transform, atmosphere),
        "difference_rms": _summarise_increment(difference, model, transform, atmosphere),
    }


__all__ = [
    "INCREMENT_APPLICATIONS",
    "INCREMENT_SOURCES",
    "ControlOptions",
    "degree_power",
    "mean_increment_comparison",
]
