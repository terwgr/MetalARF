"""The DA scorecard: O-B and O-A as distributions per stream, variable,
region and cycle, with the four assessments of amendment G.

Every analysis the door forms carries one card, built from the same
departures the analysis computed (observation minus background before
the update, observation minus analysis after it), grouped three ways at
once: by the STREAM the report came from (the ``source`` of its obs-table
row), by the neutral VARIABLE and by REGION (:data:`REGIONS`,
latitude-longitude boxes that overlap deliberately: ``global`` holds every
row, ``conus`` a subset of ``nh_extratropics``).  The cycle door stacks
one card per cycle into the receipt and prints the per-cycle table.

THE RULE (amendment G, 2026-09-06, replacing the earlier "INCOMPLETE when
O-A is not below O-B"): an analysis need not move closer to every
stream.  A background of 0 with reports +1 and -3 at equal weights
analyses to -2/3 and moves AWAY from the +1 report; a system that fits
every report is overfitting.  So O-B and O-A are kept as distributions
(bias, rms, quantiles, the fraction of rows the analysis moved closer),
Desroziers-style consistency diagnostics ride beside them with their
assumptions stated, and FOUR assessments are separated in the card:

engineering validity
    Did ingest, quality control, operators, analysis and restart run:
    every stream that offered rows reached the operators and has an O-A.
    The ONLY hard gate: a stream whose rows never reached the operators
    or whose O-A was not evaluated reads INCOMPLETE and the card is
    incomplete.
statistical consistency
    Are residuals, assigned errors, spread and correlations mutually
    plausible: per stream and variable the Desroziers estimate of the
    observation error ``sigma_o^2 ~ E[d_oa d_ob]`` against the assigned
    error (assumes a linear analysis with the right gain and independent,
    unbiased errors; a ratio far from one says the assigned error or the
    background error is wrong), the innovation variance against
    ``sigma_o^2 + spread_H^2`` when the ensemble spread in observation
    space is known, and whether O-A rms fell below O-B rms.  Readings,
    never a gate: observation errors are not retuned until these
    diagnostics look right.
physical consistency
    Budgets, imbalance, moisture and precipitation shocks after the
    update: filled by the filter and the cycle door (mass offset, vapor
    repair, the surface-pressure tendency of the first step after the
    analysis against the last step before it).
predictive value
    Forecast skill against withheld or independent verification: the
    withheld rows' O-B and O-A here (the cross-validation reading), the
    observation scorecard on the forecast elsewhere.

A linearised observation-space O-A is labelled as such wherever the
departures carry the label.

What this scorecard is not: the observation scorecard
(:mod:`woof.globe.obs_scorecard`) reads a FORECAST against the
stations and soundings and is the number of record for a forecast's
skill.  This one reads the analysis against the reports it was offered.

Calibration (:func:`calibrate`, held by
``tests/test_arwen_global_da_scorecard.py``): planted departures in both
directions, ten families: halved departures read as moved closer with
every row improved; doubled departures read as moved away and stay
engineering-complete; an unmoved stream reads moved: false; the
conflicting-report family (+1 and -3 against 0) reads moved closer with
half the rows worsened; the Desroziers estimate recovers a planted
observation error of 1 and of 2 within ten percent; rows planted inside
one region's box land in it and in every box containing it and no
other; withheld rows are read separately; an empty card is incomplete; a
stream that never reached the operators is incomplete beside one that
did; the per-cycle merge counts what each cycle said.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

SCHEMA = "gpuwm.arwen-global-da-scorecard/v2"

RULE = (
    "four assessments, separated: engineering validity (every stream that "
    "offered rows reached the operators and has an O-A; the only hard gate, "
    "INCOMPLETE otherwise), statistical consistency (Desroziers sigma_o against "
    "the assigned error, innovation variance against sigma_o^2 + spread_H^2, "
    "O-A rms against O-B rms; readings, never a gate), physical consistency "
    "(budgets and imbalance after the update, filled by the filter and the "
    "cycle), predictive value (the withheld rows here, the observation "
    "scorecard on the forecast).  An analysis need not move closer to every "
    "stream: a background of 0 with reports +1 and -3 at equal weights "
    "analyses to -2/3 and moves away from the +1 report"
)

DESROZIERS_ASSUMPTIONS = (
    "E[d_oa d_ob^T] = R holds for a linear analysis with the optimal gain and "
    "independent, unbiased observation and background errors; the scalar "
    "estimate here is sqrt(mean(d_oa * d_ob)) over the rows of one stream and "
    "variable, undefined (None) when that mean is not positive, and it reads "
    "the assigned error and the background error together, so a ratio away "
    "from one names a misfit without saying whose"
)

#: Ratios inside this band read as plausible; outside, implausible.
PLAUSIBLE_RATIO_BAND = (0.5, 2.0)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass(frozen=True)
class Region:
    """A latitude-longitude box; longitudes in -180..180, inclusive edges."""

    name: str
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float
    description: str

    def contains(self, latitude_deg, longitude_deg) -> np.ndarray:
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        lon = ((lon + 180.0) % 360.0) - 180.0
        inside = (lat >= self.latitude_min) & (lat <= self.latitude_max)
        if self.longitude_min <= self.longitude_max:
            inside &= (lon >= self.longitude_min) & (lon <= self.longitude_max)
        else:
            inside &= (lon >= self.longitude_min) | (lon <= self.longitude_max)
        return inside

    def identity(self) -> dict[str, object]:
        return {
            "latitude_min": self.latitude_min, "latitude_max": self.latitude_max,
            "longitude_min": self.longitude_min, "longitude_max": self.longitude_max,
            "description": self.description,
        }


#: The regions every card carries, in table order.  ``global`` is the
#: region the assessments are read in; the rest are the reader's breakdown.
REGIONS: tuple[Region, ...] = (
    Region("global", -90.0, 90.0, -180.0, 180.0, "every row"),
    Region("nh_extratropics", 20.0, 90.0, -180.0, 180.0, "latitude 20N and north"),
    Region("tropics", -20.0, 20.0, -180.0, 180.0, "20S to 20N"),
    Region("sh_extratropics", -90.0, -20.0, -180.0, 180.0, "latitude 20S and south"),
    Region("conus", 24.0, 50.0, -125.0, -66.0, "24N to 50N, 125W to 66W"),
)
VERDICT_REGION = "global"


@dataclass
class Departures:
    """Per-row departures of one analysis: what the card is built from.

    Every array has one entry per row the operators evaluated.  ``level_pa``
    is NaN for a surface row.  ``withheld`` marks the rows the analysis
    never saw (the door's cross-validation fraction).  ``error`` is the
    observation error (standard deviation) THE FILTER WEIGHTED THE ROW BY:
    the calibrated error when an observation-error calibration is in force
    (:mod:`woof.globe.da.observation_errors`), the door's assigned
    error otherwise; ``door_error`` is the error the door assigned the row
    whatever the filter used (NaN when the caller did not say), so the
    Desroziers reading is taken against the error the analysis actually
    ran with and the receipt still shows the door's figure beside it.
    ``spread_h`` is the background ensemble spread in observation space
    (NaN when the filter has no ensemble), ``time_offset_s`` the report
    time minus the time of the state it was compared with (0 for an
    analysis-instant comparison).  ``o_minus_a_label`` says what O-A is
    (an evaluation on the analysed state, or the linearised equivalent of
    amendment G).
    """

    source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    variable: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    latitude_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    longitude_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    level_pa: np.ndarray = field(default_factory=lambda: np.zeros(0))
    o_minus_b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    o_minus_a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    withheld: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    error: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spread_h: np.ndarray = field(default_factory=lambda: np.zeros(0))
    time_offset_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    o_minus_a_label: str = "O-A evaluated on the analysed state at the analysis instant"
    door_error: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        n = self.o_minus_b.size
        if self.error.size != n:
            self.error = np.full(n, math.nan)
        if self.door_error.size != n:
            self.door_error = np.full(n, math.nan)
        if self.spread_h.size != n:
            self.spread_h = np.full(n, math.nan)
        if self.time_offset_s.size != n:
            self.time_offset_s = np.zeros(n)

    @property
    def count(self) -> int:
        return int(self.o_minus_b.size)

    @classmethod
    def from_rows(
        cls, rows: Sequence, hx_background, hx_analysis, *, withheld: bool,
        spread_h=None, time_offset_s=None, o_minus_a_label: str | None = None,
        error=None,
    ) -> "Departures":
        """Departures of ``rows`` (obs-table ``ObsRow`` objects) against the
        background and analysis operator values; rows whose operator read
        NaN (a surface-pressure row aloft) are left out.  ``error`` is the
        per-row observation error the filter weighted by when it is not the
        rows' own (a calibration laid over the doors' assigned errors); the
        rows' own errors are carried as ``door_error`` either way."""
        rows = list(rows)
        if not rows:
            return cls()
        hb = np.asarray(hx_background, dtype=np.float64)
        ha = np.asarray(hx_analysis, dtype=np.float64)
        if hb.shape != (len(rows),) or ha.shape != (len(rows),):
            raise ValueError(
                f"departures need one operator value per row: {len(rows)} rows, "
                f"H(b) {hb.shape}, H(a) {ha.shape}"
            )
        value = np.array([row.value for row in rows], dtype=np.float64)
        door = np.array([row.error for row in rows], dtype=np.float64)
        weighted = door if error is None else np.asarray(error, dtype=np.float64).reshape(-1)
        if weighted.shape != (len(rows),):
            raise ValueError(
                f"departures need one weighted error per row: {len(rows)} rows, error {weighted.shape}"
            )
        keep = np.isfinite(hb) & np.isfinite(ha)
        n = int(keep.sum())
        spread = (np.full(len(rows), math.nan) if spread_h is None
                  else np.asarray(spread_h, dtype=np.float64).reshape(-1))
        offset = (np.zeros(len(rows)) if time_offset_s is None
                  else np.asarray(time_offset_s, dtype=np.float64).reshape(-1))
        out = cls(
            source=np.array([row.source for row in rows], dtype=object)[keep],
            variable=np.array([row.variable for row in rows], dtype=object)[keep],
            latitude_deg=np.array([row.latitude_deg for row in rows], dtype=np.float64)[keep],
            longitude_deg=np.array([row.longitude_deg for row in rows], dtype=np.float64)[keep],
            level_pa=np.array(
                [math.nan if row.level_pa is None else row.level_pa for row in rows],
                dtype=np.float64,
            )[keep],
            o_minus_b=(value - hb)[keep],
            o_minus_a=(value - ha)[keep],
            withheld=np.full(n, bool(withheld)),
            error=weighted[keep],
            spread_h=spread[keep],
            time_offset_s=offset[keep],
            door_error=door[keep],
        )
        if o_minus_a_label:
            out.o_minus_a_label = o_minus_a_label
        return out

    @classmethod
    def concatenate(cls, parts: Iterable["Departures"]) -> "Departures":
        parts = [part for part in parts if part.count]
        if not parts:
            return cls()
        labels = sorted({p.o_minus_a_label for p in parts})
        return cls(
            source=np.concatenate([p.source for p in parts]),
            variable=np.concatenate([p.variable for p in parts]),
            latitude_deg=np.concatenate([p.latitude_deg for p in parts]),
            longitude_deg=np.concatenate([p.longitude_deg for p in parts]),
            level_pa=np.concatenate([p.level_pa for p in parts]),
            o_minus_b=np.concatenate([p.o_minus_b for p in parts]),
            o_minus_a=np.concatenate([p.o_minus_a for p in parts]),
            withheld=np.concatenate([p.withheld for p in parts]),
            error=np.concatenate([p.error for p in parts]),
            spread_h=np.concatenate([p.spread_h for p in parts]),
            time_offset_s=np.concatenate([p.time_offset_s for p in parts]),
            o_minus_a_label=labels[0] if len(labels) == 1 else "; ".join(labels),
            door_error=np.concatenate([p.door_error for p in parts]),
        )


def _stats(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "bias": None, "rms": None, "quantiles": None}
    return {
        "n": int(values.size),
        "bias": float(np.mean(values)),
        "rms": float(math.sqrt(np.mean(values * values))),
        "quantiles": {f"p{int(round(q * 100)):02d}": float(np.quantile(values, q)) for q in QUANTILES},
    }


def _consistency(omb: np.ndarray, oma: np.ndarray, error: np.ndarray, spread: np.ndarray,
                 door_error: np.ndarray | None = None) -> dict[str, object]:
    """The Desroziers reading and the innovation-variance reading of one
    cell, with the plausibility band applied where a ratio exists.
    ``error`` is the observation error the filter weighted the rows by,
    so the ratio judges the error the analysis ran with; ``door_error``
    (the doors' assigned errors) is recorded beside it as
    ``door_sigma_o`` and ``calibrated`` says whether the two differ.  The
    filter of the 2026-09-06 balance package lays a Desroziers table over
    the doors' errors, and a reading taken against the doors' figure judged
    an error the analysis never used (the motion vectors read 0.66 of an
    assigned 4.0 m/s where the filter had weighted them at 2.8)."""
    n = omb.size
    if n == 0:
        return {
            "desroziers_sigma_o": None, "assigned_sigma_o": None, "door_sigma_o": None, "calibrated": False,
            "desroziers_ratio": None, "innovation_variance_ratio": None, "plausible": None, "reading": "no rows",
        }
    cross = float(np.mean(omb * oma))
    sigma_est = math.sqrt(cross) if cross > 0.0 else None
    finite_error = error[np.isfinite(error)]
    assigned = float(math.sqrt(np.mean(finite_error ** 2))) if finite_error.size else None
    door = None
    if door_error is not None:
        finite_door = np.asarray(door_error, dtype=np.float64)
        finite_door = finite_door[np.isfinite(finite_door)]
        door = float(math.sqrt(np.mean(finite_door ** 2))) if finite_door.size else None
    calibrated = bool(door is not None and assigned is not None and not math.isclose(door, assigned, rel_tol=1.0e-9, abs_tol=0.0))
    against = (f"the calibrated error {assigned:.4g} (the doors assigned {door:.4g})" if calibrated and assigned is not None
               else f"assigned {assigned:.4g}" if assigned is not None else "no assigned error")
    ratio = (sigma_est / assigned) if (sigma_est is not None and assigned) else None
    finite_spread = np.isfinite(spread) & np.isfinite(error)
    innovation_ratio = None
    if finite_spread.any():
        expected = float(np.mean(error[finite_spread] ** 2 + spread[finite_spread] ** 2))
        if expected > 0.0:
            innovation_ratio = float(np.mean(omb[finite_spread] ** 2)) / expected
    ratios = [r for r in (ratio, innovation_ratio) if r is not None]
    low, high = PLAUSIBLE_RATIO_BAND
    plausible = None if not ratios else all(low <= r <= high for r in ratios)
    if sigma_est is None:
        reading = "mean(d_oa d_ob) is not positive: the analysis overshot the reports on average (or n is small); no error estimate"
    elif plausible is None:
        reading = "no assigned error to compare"
    elif plausible:
        reading = f"Desroziers sigma_o {sigma_est:.4g} against {against} (ratio {ratio:.3g}) inside the {low}..{high} band"
    else:
        reading = f"Desroziers sigma_o {sigma_est:.4g} against {against} (ratio {ratio:.3g}) outside the {low}..{high} band"
        if innovation_ratio is not None and not (low <= innovation_ratio <= high):
            reading += f"; innovation variance is {innovation_ratio:.3g} of sigma_o^2 + spread_H^2"
    return {
        "desroziers_sigma_o": sigma_est, "assigned_sigma_o": assigned, "door_sigma_o": door,
        "calibrated": calibrated, "desroziers_ratio": ratio,
        "innovation_variance_ratio": innovation_ratio, "plausible": plausible,
        "reading": reading, "assumptions": DESROZIERS_ASSUMPTIONS,
    }


def _cell(dep: Departures, mask: np.ndarray) -> dict[str, object]:
    held = mask & dep.withheld
    used = mask & ~dep.withheld
    omb = dep.o_minus_b[mask]
    oma = dep.o_minus_a[mask]
    closer = np.abs(oma) < np.abs(omb)
    return {
        "n": int(mask.sum()),
        "o_minus_b": _stats(omb),
        "o_minus_a": _stats(oma),
        "moved_closer_fraction": float(np.mean(closer)) if omb.size else None,
        "assimilated": {
            "n": int(used.sum()),
            "o_minus_b": _stats(dep.o_minus_b[used]),
            "o_minus_a": _stats(dep.o_minus_a[used]),
        },
        "withheld": {
            "n": int(held.sum()),
            "o_minus_b": _stats(dep.o_minus_b[held]),
            "o_minus_a": _stats(dep.o_minus_a[held]),
        },
        "consistency": _consistency(omb, oma, dep.error[mask], dep.spread_h[mask], dep.door_error[mask]),
        "time_offset_max_abs_s": float(np.max(np.abs(dep.time_offset_s[mask]))) if omb.size else 0.0,
    }


def _assess(cell: dict[str, object]) -> dict[str, object]:
    """The per-cell assessments: engineering (rows reached the operators
    and have an O-A) and statistical consistency (readings)."""
    if cell["n"] == 0:
        return {
            "engineering": {"verdict": "incomplete", "reason": "no row of this stream and variable reached the operators"},
            "statistical_consistency": {"o_a_rms_below_o_b_rms": None, "moved": None,
                                        "moved_closer_fraction": None, "reading": "no rows"},
        }
    omb = cell["o_minus_b"]["rms"]
    oma = cell["o_minus_a"]["rms"]
    below = oma < omb
    moved = oma != omb
    consistency = cell["consistency"]
    closer = cell.get("moved_closer_fraction")
    closer_text = "" if closer is None else f"; {closer:.0%} of rows moved closer"
    if not moved:
        reading = f"O-A rms equals O-B rms ({omb:.6g}) on {cell['n']} rows: the analysis did not move against this stream"
    elif below:
        reading = f"O-A rms {oma:.6g} below O-B rms {omb:.6g} on {cell['n']} rows{closer_text}"
    else:
        reading = f"O-A rms {oma:.6g} above O-B rms {omb:.6g} on {cell['n']} rows{closer_text} (a reading, not a failure)"
    return {
        "engineering": {"verdict": "pass", "reason": f"{cell['n']} rows reached the operators with an O-A"},
        "statistical_consistency": {
            "o_a_rms_below_o_b_rms": bool(below), "moved": bool(moved),
            "moved_closer_fraction": cell["moved_closer_fraction"],
            "desroziers_ratio": consistency["desroziers_ratio"],
            "innovation_variance_ratio": consistency["innovation_variance_ratio"],
            "plausible": consistency["plausible"],
            "reading": reading + "; " + consistency["reading"],
        },
    }


def _card_assessments(streams: dict[str, object], dep: Departures | None) -> dict[str, object]:
    engineering_failures = []
    below, not_below, unmoved = [], [], []
    plausible, implausible, unread = 0, 0, 0
    for source, stream in streams.items():
        for variable, row in stream["variables"].items():
            label = f"{source}/{variable}"
            a = row["assessments"]
            if a["engineering"]["verdict"] != "pass":
                engineering_failures.append(label)
                continue
            s = a["statistical_consistency"]
            if not s["moved"]:
                unmoved.append(label)
            elif s["o_a_rms_below_o_b_rms"]:
                below.append(label)
            else:
                not_below.append(label)
            if s["plausible"] is None:
                unread += 1
            elif s["plausible"]:
                plausible += 1
            else:
                implausible += 1
    withheld_reading = None
    if dep is not None and dep.count and dep.withheld.any():
        omb = dep.o_minus_b[dep.withheld]
        oma = dep.o_minus_a[dep.withheld]
        withheld_reading = {
            "rows": int(omb.size),
            "o_minus_b_rms": float(np.sqrt(np.mean(omb ** 2))),
            "o_minus_a_rms": float(np.sqrt(np.mean(oma ** 2))),
        }
    return {
        "engineering": {
            "verdict": "pass" if streams and not engineering_failures else "incomplete",
            "failures": engineering_failures if streams else ["(no stream reached the operators)"],
            "reads": "did ingest, quality control, operators, analysis and O-A run for every stream that offered rows; the only hard gate",
        },
        "statistical_consistency": {
            "o_a_below_o_b": below, "o_a_not_below_o_b": not_below, "unmoved": unmoved,
            "plausible_cells": plausible, "implausible_cells": implausible, "unread_cells": unread,
            "reads": "Desroziers sigma_o against the assigned error, innovation variance against sigma_o^2 + spread_H^2, O-A rms against O-B rms; readings, never a gate",
        },
        "physical_consistency": {
            "reads": "budgets, imbalance, moisture and precipitation shocks after the update; filled by the filter and the cycle door",
        },
        "predictive_value": {
            "withheld": withheld_reading,
            "reads": "the withheld rows' O-B and O-A here; the observation scorecard on the forecast is the number of record",
        },
    }


def scorecard(
    dep: Departures, *, regions: Sequence[Region] = REGIONS,
    label: str | None = None,
) -> dict[str, object]:
    """One card: per stream, per variable, per region; the assessments read
    in :data:`VERDICT_REGION`.  ``label`` names the analysis (its time)."""
    names = [region.name for region in regions]
    if VERDICT_REGION not in names:
        raise ValueError(f"the regions must include {VERDICT_REGION!r}, the verdict region")
    if len(set(names)) != len(names):
        raise ValueError("region names must be distinct")
    region_masks = {
        region.name: region.contains(dep.latitude_deg, dep.longitude_deg)
        for region in regions
    }
    streams: dict[str, object] = {}
    for source in sorted(set(dep.source.tolist())):
        in_source = dep.source == source
        variables: dict[str, object] = {}
        for variable in sorted(set(dep.variable[in_source].tolist())):
            in_variable = in_source & (dep.variable == variable)
            per_region = {
                name: _cell(dep, in_variable & mask)
                for name, mask in region_masks.items()
            }
            variables[variable] = {
                "regions": per_region,
                "assessments": _assess(per_region[VERDICT_REGION]),
            }
        streams[source] = _stream_summary(variables, int(in_source.sum()))
    assessments = _card_assessments(streams, dep)
    return {
        "schema": SCHEMA,
        "label": label,
        "regions": {region.name: region.identity() for region in regions},
        "verdict_region": VERDICT_REGION,
        "rule": RULE,
        "o_minus_a_label": dep.o_minus_a_label,
        "streams": streams,
        "rows": dep.count,
        "assessments": assessments,
        "verdict": "complete" if assessments["engineering"]["verdict"] == "pass" else "incomplete",
        "incomplete_streams": assessments["engineering"]["failures"],
    }


def _stream_summary(variables: dict[str, object], rows: int) -> dict[str, object]:
    incomplete = [v for v, row in variables.items() if row["assessments"]["engineering"]["verdict"] != "pass"]
    not_below = [v for v, row in variables.items()
                 if row["assessments"]["engineering"]["verdict"] == "pass"
                 and not row["assessments"]["statistical_consistency"]["o_a_rms_below_o_b_rms"]]
    return {
        "variables": variables,
        "verdict": "pass" if not incomplete else "incomplete",
        "incomplete_variables": incomplete,
        "o_a_not_below_o_b_variables": not_below,
        "rows": rows,
    }


def _pool(*parts: dict) -> dict[str, object]:
    """Pool ``{n, bias, rms}`` cells (or the ensemble receipt's
    ``{count, o_minus_x: {mean, rms}}`` shape) into one."""
    n_total = 0
    sum_values = 0.0
    sum_squares = 0.0
    for part in parts:
        if not part:
            continue
        n = int(part.get("n", part.get("count", 0)) or 0)
        if n == 0:
            continue
        bias = part.get("bias", part.get("mean"))
        rms = part.get("rms")
        if bias is None or rms is None:
            continue
        n_total += n
        sum_values += float(bias) * n
        sum_squares += float(rms) ** 2 * n
    if n_total == 0:
        return {"n": 0, "bias": None, "rms": None, "quantiles": None}
    return {
        "n": n_total, "bias": sum_values / n_total,
        "rms": math.sqrt(sum_squares / n_total), "quantiles": None,
    }


def scorecard_from_stream_table(
    streams: dict, *, label: str | None = None,
) -> dict[str, object]:
    """The card from the ensemble filter's ``streams[stream][variable]``
    receipt (:mod:`woof.globe.da.analysis`): per region the
    assimilated and withheld rows each carry ``count`` and O-B / O-A
    statistics; the card pools them and applies this module's
    assessments (the Desroziers reading needs rows and is unread here)."""
    card_streams: dict[str, object] = {}
    for source in sorted(streams):
        variables: dict[str, object] = {}
        for variable in sorted(streams[source]):
            entry = streams[source][variable]
            regions_in = entry.get("regions", {}) or {}
            per_region: dict[str, object] = {}
            for region in REGIONS:
                row = regions_in.get(region.name) or {}
                used = row.get("assimilated") or {}
                held = row.get("withheld") or {}

                def stats(part, key):
                    value = part.get(key)
                    if not isinstance(value, dict):
                        return {"n": 0, "bias": None, "rms": None, "quantiles": None}
                    return {"n": int(part.get("count", 0)), "bias": value.get("mean", value.get("bias")),
                            "rms": value.get("rms"), "quantiles": None}

                used_b, used_a = stats(used, "o_minus_b"), stats(used, "o_minus_a")
                held_b, held_a = stats(held, "o_minus_b"), stats(held, "o_minus_a")
                pooled_b, pooled_a = _pool(used_b, held_b), _pool(used_a, held_a)
                per_region[region.name] = {
                    "n": int(used.get("count", 0)) + int(held.get("count", 0)),
                    "o_minus_b": pooled_b,
                    "o_minus_a": pooled_a,
                    "moved_closer_fraction": None,
                    "assimilated": {"n": int(used.get("count", 0)), "o_minus_b": used_b, "o_minus_a": used_a},
                    "withheld": {"n": int(held.get("count", 0)), "o_minus_b": held_b, "o_minus_a": held_a},
                    "consistency": {
                        "desroziers_sigma_o": None, "assigned_sigma_o": None, "door_sigma_o": None,
                        "calibrated": False, "desroziers_ratio": None,
                        "innovation_variance_ratio": None, "plausible": None,
                        "reading": "pooled statistics only; the Desroziers reading needs rows",
                    },
                    "time_offset_max_abs_s": 0.0,
                }
            cell = per_region[VERDICT_REGION]
            assessments = _assess(cell) if cell["o_minus_a"]["rms"] is not None else _assess({**cell, "n": 0})
            if cell["n"] and cell["o_minus_a"]["rms"] is None:
                assessments["engineering"] = {
                    "verdict": "incomplete",
                    "reason": "no operator re-evaluated this stream on the analysis",
                }
            variables[variable] = {
                "regions": per_region, "assessments": assessments,
                "filter_verdict": entry.get("verdict"),
            }
        card_streams[source] = _stream_summary(variables, int(sum(
            variables[v]["regions"][VERDICT_REGION]["n"] for v in variables)))
    assessments = _card_assessments(card_streams, None)
    return {
        "schema": SCHEMA,
        "label": label,
        "regions": {region.name: region.identity() for region in REGIONS},
        "verdict_region": VERDICT_REGION,
        "rule": RULE,
        "o_minus_a_label": "the ensemble filter's own receipt on the members",
        "streams": card_streams,
        "rows": int(sum(s["rows"] for s in card_streams.values())),
        "assessments": assessments,
        "verdict": "complete" if assessments["engineering"]["verdict"] == "pass" else "incomplete",
        "incomplete_streams": assessments["engineering"]["failures"],
    }


def merge_cycles(cards: Sequence[tuple[str, dict]]) -> dict[str, object]:
    """The per-cycle table of a run: for every stream and variable, how
    many cycles were engineering-complete, in how many O-A rms fell below
    O-B rms and which cycles it did not, plus the global-region O-B and
    O-A rms and the Desroziers ratio of every cycle in order."""
    summary: dict[str, dict] = {}
    for cycle_label, card in cards:
        for source, stream in card["streams"].items():
            for variable, row in stream["variables"].items():
                entry = summary.setdefault(source, {}).setdefault(variable, {
                    "cycles": 0, "complete": 0, "o_a_below_o_b": 0,
                    "incomplete_cycles": [], "o_a_not_below_o_b_cycles": [],
                    "o_minus_b_rms": [], "o_minus_a_rms": [], "desroziers_ratio": [], "rows": [],
                })
                cell = row["regions"][VERDICT_REGION]
                a = row["assessments"]
                entry["cycles"] += 1
                if a["engineering"]["verdict"] == "pass":
                    entry["complete"] += 1
                    if a["statistical_consistency"]["o_a_rms_below_o_b_rms"]:
                        entry["o_a_below_o_b"] += 1
                    else:
                        entry["o_a_not_below_o_b_cycles"].append(cycle_label)
                else:
                    entry["incomplete_cycles"].append(cycle_label)
                entry["o_minus_b_rms"].append(cell["o_minus_b"]["rms"])
                entry["o_minus_a_rms"].append(cell["o_minus_a"]["rms"])
                entry["desroziers_ratio"].append(cell["consistency"]["desroziers_ratio"])
                entry["rows"].append(cell["n"])
    return {
        "schema": SCHEMA,
        "cycles": [label for label, _ in cards],
        "streams": summary,
        "complete_cycles": sum(1 for _, card in cards if card["verdict"] == "complete"),
        "incomplete_cycles": [label for label, card in cards if card["verdict"] != "complete"],
        "rule": RULE,
    }


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}"


def render_table(card: dict[str, object], *, region: str = VERDICT_REGION) -> str:
    """A text table of one card in ``region``: stream, variable, n, O-B
    bias / rms, O-A bias / rms, the fraction of rows moved closer, the
    Desroziers ratio, whether O-A rms fell below O-B rms, and the
    engineering verdict."""
    assessments = card.get("assessments") or {}
    stat = assessments.get("statistical_consistency") or {}
    lines = [
        f"DA scorecard ({card.get('label') or 'analysis'}, region {region}): "
        f"engineering {card['verdict'].upper()}"
        + (f", incomplete: {', '.join(card['incomplete_streams'])}"
           if card["verdict"] != "complete" else "")
        + (f"; O-A above O-B on {', '.join(stat['o_a_not_below_o_b'])} (a reading, not a failure)"
           if stat.get("o_a_not_below_o_b") else ""),
        f"{'stream':<20} {'variable':<20} {'n':>6} {'O-B bias':>10} {'O-B rms':>10} "
        f"{'O-A bias':>10} {'O-A rms':>10} {'closer':>7} {'Desroz':>7} {'O-A<O-B':>8} engineering",
    ]
    for source, stream in card["streams"].items():
        for variable, row in stream["variables"].items():
            cell = row["regions"].get(region)
            if cell is None:
                continue
            a = row["assessments"]
            s = a["statistical_consistency"]
            closer = cell.get("moved_closer_fraction")
            below = s.get("o_a_rms_below_o_b_rms")
            lines.append(
                f"{source:<20} {variable:<20} {cell['n']:>6} "
                f"{_fmt(cell['o_minus_b']['bias']):>10} {_fmt(cell['o_minus_b']['rms']):>10} "
                f"{_fmt(cell['o_minus_a']['bias']):>10} {_fmt(cell['o_minus_a']['rms']):>10} "
                f"{('-' if closer is None else f'{closer:.2f}'):>7} "
                f"{_fmt(cell['consistency']['desroziers_ratio']):>7} "
                f"{('-' if below is None else ('yes' if below else 'no')):>8} "
                f"{a['engineering']['verdict']}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibration: planted departures, both directions
# ---------------------------------------------------------------------------

def _planted(source, variable, lat, lon, omb, oma, withheld=False, error=None) -> Departures:
    n = len(lat)
    return Departures(
        source=np.array([source] * n, dtype=object),
        variable=np.array([variable] * n, dtype=object),
        latitude_deg=np.asarray(lat, dtype=np.float64),
        longitude_deg=np.asarray(lon, dtype=np.float64),
        level_pa=np.full(n, math.nan),
        o_minus_b=np.asarray(omb, dtype=np.float64),
        o_minus_a=np.asarray(oma, dtype=np.float64),
        withheld=np.full(n, bool(withheld)),
        error=np.full(n, math.nan) if error is None else np.full(n, float(error)),
    )


def _cell_of(card, source, variable):
    return card["streams"][source]["variables"][variable]["regions"]["global"]


def _stat_of(card, source, variable):
    return card["streams"][source]["variables"][variable]["assessments"]["statistical_consistency"]


def calibrate() -> dict[str, object]:
    """Every family below is planted and read back; the returned readings
    carry the expected and the measured values and the test holds them.
    A family that fails is a defect of the instrument, not of any analysis."""
    rng = np.random.default_rng(7)
    readings: dict[str, object] = {}
    n = 40
    lat = rng.uniform(30.0, 45.0, n)
    lon = rng.uniform(-110.0, -80.0, n)
    omb = rng.normal(0.0, 2.0, n)

    # Family 1: the analysis halves every departure: moved closer, every
    # row improved, engineering complete.
    card = scorecard(_planted("s", "temperature_k", lat, lon, omb, 0.5 * omb))
    cell = _cell_of(card, "s", "temperature_k")
    readings["halved_departures"] = {
        "expected": {"verdict": "complete", "below": True, "closer": 1.0},
        "verdict": card["verdict"],
        "below": _stat_of(card, "s", "temperature_k")["o_a_rms_below_o_b_rms"],
        "closer": cell["moved_closer_fraction"],
        "rms_ratio": cell["o_minus_a"]["rms"] / cell["o_minus_b"]["rms"],
    }
    # Family 2: the analysis doubles every departure: moved away, still
    # engineering complete (the rule of amendment G).
    card = scorecard(_planted("s", "temperature_k", lat, lon, omb, 2.0 * omb))
    readings["doubled_departures"] = {
        "expected": {"verdict": "complete", "below": False, "closer": 0.0},
        "verdict": card["verdict"],
        "below": _stat_of(card, "s", "temperature_k")["o_a_rms_below_o_b_rms"],
        "closer": _cell_of(card, "s", "temperature_k")["moved_closer_fraction"],
        "named": card["assessments"]["statistical_consistency"]["o_a_not_below_o_b"],
    }
    # Family 3: nothing moved (O-A equals O-B to the bit).
    card = scorecard(_planted("s", "wind_u_m_s", lat, lon, omb, omb.copy()))
    readings["unmoved"] = {
        "expected": {"verdict": "complete", "moved": False},
        "verdict": card["verdict"],
        "moved": _stat_of(card, "s", "wind_u_m_s")["moved"],
        "unmoved_named": card["assessments"]["statistical_consistency"]["unmoved"],
    }
    # Family 4: the conflicting reports of amendment G: background 0,
    # reports +1 and -3 at equal weight, analysis -2/3.  O-A rms falls
    # (2.03 against 2.24) while the +1 report moves away: half the rows
    # improved, the stream reads moved closer, engineering complete.
    pairs = 20
    omb_c = np.tile([1.0, -3.0], pairs)
    oma_c = omb_c + 2.0 / 3.0
    card = scorecard(_planted("c", "temperature_k", lat, lon, omb_c, oma_c))
    cell = _cell_of(card, "c", "temperature_k")
    readings["conflicting_reports"] = {
        "expected": {"verdict": "complete", "below": True, "closer": 0.5},
        "verdict": card["verdict"],
        "below": _stat_of(card, "c", "temperature_k")["o_a_rms_below_o_b_rms"],
        "closer": cell["moved_closer_fraction"],
        "o_b_rms": cell["o_minus_b"]["rms"], "o_a_rms": cell["o_minus_a"]["rms"],
    }
    # Family 5: Desroziers both directions.  A scalar linear analysis with
    # background error 1.5 and observation error sigma_o: d_ob = e_o - e_b,
    # d_oa = d_ob sigma_o^2 / (sigma_b^2 + sigma_o^2); E[d_oa d_ob] =
    # sigma_o^2 exactly.  Planted sigma_o 1 and 2 must read back within
    # ten percent at n = 4000.
    big = 4000
    lat_big = rng.uniform(-60.0, 60.0, big)
    lon_big = rng.uniform(-180.0, 180.0, big)
    desroziers = {}
    for sigma_o in (1.0, 2.0):
        sigma_b = 1.5
        e_o = rng.normal(0.0, sigma_o, big)
        e_b = rng.normal(0.0, sigma_b, big)
        d_ob = e_o - e_b
        d_oa = d_ob * sigma_o ** 2 / (sigma_b ** 2 + sigma_o ** 2)
        card = scorecard(_planted("d", "temperature_k", lat_big, lon_big, d_ob, d_oa, error=sigma_o))
        c = _cell_of(card, "d", "temperature_k")["consistency"]
        desroziers[f"sigma_o_{sigma_o:g}"] = {
            "planted": sigma_o, "estimate": c["desroziers_sigma_o"], "ratio": c["desroziers_ratio"],
            "plausible": c["plausible"],
        }
    readings["desroziers"] = {"tolerance": 0.10, "readings": desroziers}
    # Family 6: region assignment.  Rows planted in the CONUS box land in
    # global, nh_extratropics and conus and nowhere else; rows planted at
    # 10S 150E land in global and tropics only; the counts add up.
    conus = _planted("r", "temperature_k", lat, lon, omb, 0.5 * omb)
    tropics = _planted(
        "r", "temperature_k", np.full(7, -10.0), np.full(7, 150.0),
        np.ones(7), 0.5 * np.ones(7),
    )
    card = scorecard(Departures.concatenate([conus, tropics]))
    per_region = {
        name: cell["n"]
        for name, cell in card["streams"]["r"]["variables"]["temperature_k"]["regions"].items()
    }
    readings["regions"] = {
        "expected": {
            "global": n + 7, "nh_extratropics": n, "tropics": 7,
            "sh_extratropics": 0, "conus": n,
        },
        "measured": per_region,
    }
    # Family 7: the withheld rows are read separately (the predictive-value
    # assessment) and pooled into the cell: assimilated improve, withheld
    # worsen.
    used = _planted("w", "temperature_k", lat, lon, omb, 0.1 * omb)
    held = _planted(
        "w", "temperature_k", lat[:4], lon[:4], omb[:4], 1.5 * omb[:4], withheld=True,
    )
    card = scorecard(Departures.concatenate([used, held]))
    cell = _cell_of(card, "w", "temperature_k")
    readings["withheld_reported_separately"] = {
        "withheld_n": cell["withheld"]["n"], "expected_withheld_n": 4,
        "withheld_worsened": cell["withheld"]["o_minus_a"]["rms"] > cell["withheld"]["o_minus_b"]["rms"],
        "assimilated_improved": cell["assimilated"]["o_minus_a"]["rms"] < cell["assimilated"]["o_minus_b"]["rms"],
        "predictive_value_rows": card["assessments"]["predictive_value"]["withheld"]["rows"],
        "verdict": card["verdict"], "expected": "complete",
    }
    # Family 8: no stream at all: incomplete, never complete.
    card = scorecard(Departures())
    readings["empty"] = {"expected": "incomplete", "verdict": card["verdict"]}
    # Family 9: two streams, one reached the operators and one did not
    # (its O-A never evaluated): the card is incomplete and names the one.
    reached = _planted("reached", "temperature_k", lat, lon, omb, 0.5 * omb)
    card = scorecard(reached)
    card["streams"]["missing"] = _stream_summary({
        "dewpoint_k": {"regions": {r.name: _cell(Departures(), np.zeros(0, dtype=bool)) for r in REGIONS},
                       "assessments": _assess({"n": 0})},
    }, 0)
    assessments = _card_assessments(card["streams"], None)
    readings["one_stream_never_reached"] = {
        "expected_failures": ["missing/dewpoint_k"],
        "failures": assessments["engineering"]["failures"],
        "verdict": assessments["engineering"]["verdict"],
    }
    # Family 10: the per-cycle merge counts what each cycle said.
    better = scorecard(_planted("s", "temperature_k", lat, lon, omb, 0.5 * omb))
    worse = scorecard(_planted("s", "temperature_k", lat, lon, omb, 2.0 * omb))
    merged = merge_cycles([("c1", better), ("c2", worse), ("c3", better)])
    entry = merged["streams"]["s"]["temperature_k"]
    readings["merge"] = {
        "expected": {"complete": 3, "below": 2, "not_below_cycles": ["c2"], "complete_cycles": 3},
        "complete": entry["complete"], "below": entry["o_a_below_o_b"],
        "not_below_cycles": entry["o_a_not_below_o_b_cycles"],
        "complete_cycles": merged["complete_cycles"],
    }
    return {"schema": SCHEMA, "readings": readings}


def calibration_holds(readings: dict[str, object]) -> list[str]:
    """The bars :func:`calibrate`'s readings must clear; the failures by name."""
    r = readings
    failures = []
    h = r["halved_departures"]
    if not (h["verdict"] == "complete" and h["below"] is True and h["closer"] == 1.0):
        failures.append("halved departures did not read complete, below and every row closer")
    d = r["doubled_departures"]
    if not (d["verdict"] == "complete" and d["below"] is False and d["closer"] == 0.0
            and d["named"] == ["s/temperature_k"]):
        failures.append("doubled departures did not stay complete while reading O-A above O-B")
    u = r["unmoved"]
    if not (u["verdict"] == "complete" and u["moved"] is False and u["unmoved_named"] == ["s/wind_u_m_s"]):
        failures.append("an unmoved stream was not read as unmoved and complete")
    c = r["conflicting_reports"]
    if not (c["verdict"] == "complete" and c["below"] is True and abs(c["closer"] - 0.5) < 1e-12
            and c["o_a_rms"] < c["o_b_rms"]):
        failures.append("the conflicting-report family did not read moved closer with half the rows worsened")
    for name, reading in r["desroziers"]["readings"].items():
        est = reading["estimate"]
        if est is None or abs(est - reading["planted"]) > r["desroziers"]["tolerance"] * reading["planted"]:
            failures.append(f"Desroziers {name}: estimate {est} is not within 10 percent of {reading['planted']}")
        if reading["plausible"] is not True:
            failures.append(f"Desroziers {name}: a well-specified error did not read plausible")
    if r["regions"]["measured"] != r["regions"]["expected"]:
        failures.append(f"region counts {r['regions']['measured']} != {r['regions']['expected']}")
    w = r["withheld_reported_separately"]
    if not (w["withheld_n"] == 4 and w["withheld_worsened"] and w["assimilated_improved"]
            and w["verdict"] == "complete" and w["predictive_value_rows"] == 4):
        failures.append("the withheld rows were not read separately from the assimilated rows")
    if r["empty"]["verdict"] != "incomplete":
        failures.append("an empty card was not incomplete")
    o = r["one_stream_never_reached"]
    if not (o["verdict"] == "incomplete" and o["failures"] == ["missing/dewpoint_k"]):
        failures.append("a stream that never reached the operators was not the one named incomplete")
    m = r["merge"]
    if not (m["complete"] == 3 and m["below"] == 2 and m["not_below_cycles"] == ["c2"] and m["complete_cycles"] == 3):
        failures.append("the per-cycle merge miscounted")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m woof.globe.da_scorecard",
        description="the DA scorecard's calibration, or a card rendered from an assimilation report",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate", help="plant every synthetic family both directions and print the readings")
    show = sub.add_parser("show", help="render the scorecard of an assimilation report")
    show.add_argument("report", help="assimilation-report*.json written by the door")
    show.add_argument("--region", default=VERDICT_REGION)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        result = calibrate()
        failures = calibration_holds(result["readings"])
        result["failures"] = failures
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if not failures else 1
    payload = json.loads(open(args.report, encoding="utf-8").read())
    card = payload.get("scorecard")
    if not isinstance(card, dict):
        raise SystemExit(f"{args.report} carries no scorecard")
    print(render_table(card, region=args.region))
    return 0


__all__ = [
    "DESROZIERS_ASSUMPTIONS",
    "Departures",
    "PLAUSIBLE_RATIO_BAND",
    "REGIONS",
    "RULE",
    "Region",
    "SCHEMA",
    "VERDICT_REGION",
    "calibrate",
    "calibration_holds",
    "merge_cycles",
    "render_table",
    "scorecard",
    "scorecard_from_stream_table",
]


if __name__ == "__main__":
    raise SystemExit(main())
