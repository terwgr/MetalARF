"""How a physics suite's per-band readings become the globe's.

The physics half-step runs the suite one latitude band at a time
(``dynamics.MoistHybridModel.apply_physics``): every band returns a
:class:`~woof.globe.physics.exchange.PhysicsResult` of its own
rows, and what the step reports for the call is assembled here.  Three
kinds of reading exist, and each has exactly one merge that makes the
answer independent of the band count:

- a reading that is the SAME on every band (a call counter, a schedule
  flag, a config-derived string) is checked equal across the bands and
  refused by name when it is not, because a per-call reading that differs
  between bands is a band leaking into arithmetic that must not see it;
- an EXACTLY associative reading (a maximum, a minimum, an integer count)
  folds across the bands in any order;
- a reading that is a flat reduction over a horizontal plane (a grid
  mean, a path sum) is NOT merged from the bands' scalars at all: the
  suite hands the plane back unreduced (``PhysicsResult.planes``), the
  caller assembles the whole plane, and the suite's ``finish`` reduces it
  once -- the same operand in the same order whatever the schedule was.

A floating-point SUM across bands is refused: folded band by band it is a
different number for a different band count, which is the whole thing
this module exists to prevent.  Integer-valued counts carried as floats
(a column count) are exact and are the one sum admitted, under ``count``.
"""
from __future__ import annotations

import math

import numpy as np

SAME = "same"
MAX = "max"
MIN = "min"
COUNT = "count"
COLUMN_MEAN = "column_mean"
SKIP = "skip"
MERGE_RULES = (SAME, MAX, MIN, COUNT, COLUMN_MEAN, SKIP)


def to_host(value):
    """A host numpy view of ``value``, whichever array module holds it."""
    from woof.metal.array import MetalArray

    if isinstance(value, MetalArray):
        return value.to_numpy()
    if hasattr(value, "get") and not isinstance(value, np.ndarray):
        return np.asarray(value.get())
    return np.asarray(value)


def _same(name: str, values: list, what: str):
    first = values[0]
    for other in values[1:]:
        equal = (first == other)
        if isinstance(first, float) and isinstance(other, float):
            equal = equal or (math.isnan(first) and math.isnan(other))
        if not equal:
            raise ValueError(
                f"{what} {name!r} differs between the physics bands "
                f"({first!r} against {other!r}): a per-call reading that "
                "depends on which rows a band holds cannot be reported as "
                "the call's, so it needs a merge rule that names how"
            )
    return first


def merge_band_scalars(band_values: list[dict], rules: dict[str, str], *,
                       columns: list[int] | None = None,
                       what: str = "physics diagnostic") -> dict:
    """The call's scalar readings from the bands', by rule.

    ``rules`` maps a reading's name to one of :data:`MERGE_RULES`; a name
    without a rule is ``same``.  ``columns`` is each band's column count,
    needed by ``column_mean`` (a column-count-weighted mean of per-band
    grid means, the one merge here that is not exact: it is admitted for
    a scheme's own grid-mean readings and stated as such by the suite that
    declares it).
    """
    if not band_values:
        return {}
    if len(band_values) == 1:
        return {k: v for k, v in band_values[0].items() if rules.get(k) != SKIP}
    names: list[str] = []
    for values in band_values:
        for name in values:
            if name not in names:
                names.append(name)
    out = {}
    for name in names:
        rule = rules.get(name, SAME)
        if rule not in MERGE_RULES:
            raise ValueError(f"unknown band merge rule {rule!r} for {name!r}")
        if rule == SKIP:
            continue
        values = [v[name] for v in band_values if name in v]
        if len(values) != len(band_values) and rule != MAX and rule != MIN:
            raise ValueError(
                f"{what} {name!r} is reported by {len(values)} of "
                f"{len(band_values)} physics bands; a reading a band may omit "
                "needs a max or min rule"
            )
        if rule == SAME:
            out[name] = _same(name, values, what)
        elif rule == MAX:
            out[name] = max(values)
        elif rule == MIN:
            out[name] = min(values)
        elif rule == COUNT:
            total = 0
            for value in values:
                if float(value) != math.floor(float(value)):
                    raise ValueError(
                        f"{what} {name!r} is declared a count and a band "
                        f"reported {value!r}, which is not an integer: a "
                        "floating-point sum across bands is not band-count "
                        "independent and is refused"
                    )
                total += int(value)
            out[name] = type(values[0])(total)
        elif rule == COLUMN_MEAN:
            if columns is None or len(columns) != len(values):
                raise ValueError(
                    f"{what} {name!r} is merged as a column-weighted mean "
                    "and the bands' column counts were not supplied"
                )
            total = sum(int(c) for c in columns)
            out[name] = float(sum(
                float(v) * int(c) for v, c in zip(values, columns)
            ) / max(total, 1))
    return out


def merge_band_metadata(band_metadata: list[dict], rules: dict[str, str]) -> dict:
    """The call's namespace metadata from the bands'.

    Every key is ``same`` unless ``rules`` says otherwise: ``max`` for a
    call counter that a band without the columns it counts does not bump
    (the frozen-surface and lead-tile counters), ``skip`` for a record the
    suite assembles itself from planes (the radiation size bounding).
    """
    if not band_metadata:
        return {}
    out = {}
    names: list[str] = []
    for values in band_metadata:
        for name in values:
            if name not in names:
                names.append(name)
    for name in names:
        rule = rules.get(name, SAME)
        if rule == SKIP:
            continue
        values = [m[name] for m in band_metadata if name in m]
        if rule == MAX:
            out[name] = max(values)
        elif rule == MIN:
            out[name] = min(values)
        elif rule == SAME:
            out[name] = _same(name, values, "physics namespace metadata")
        else:
            raise ValueError(
                f"metadata merge rule {rule!r} for {name!r} is not one a "
                "namespace record can take (same, max, min or skip)"
            )
    return out


def finish_alone(suite, exchange, result):
    """A STAND-ALONE call finishes itself: an exchange with no band is
    the whole grid handed to the suite by a harness or a test rather than
    by the model's band loop, so the one result is the call and the
    suite's ``finish`` runs on it here -- the diagnostics complete, the
    namespace metadata assembled, the planes reduced -- and the caller
    reads what it always read."""
    ncol = int(result.theta.shape[-2]) * int(result.theta.shape[-1])
    diagnostics, metadata = suite.finish(
        [dict(result.diagnostics)], [dict(result.physics_state.metadata)],
        dict(result.planes), result.surface, result.physics_state,
        metadata_in=dict(exchange.physics_state.metadata),
        columns=[ncol], dt_s=float(exchange.dt_s),
    )
    result.diagnostics = diagnostics
    result.physics_state.metadata = metadata
    result.planes = {}
    return result


def default_finish(band_diagnostics: list[dict], band_metadata: list[dict],
                   planes: dict[str, object], surface, physics_state, **_):
    """The merge for a suite that declares none: every scalar the same on
    every band, every metadata key the same, every plane a float64 mean
    under its own name."""
    diagnostics = merge_band_scalars(band_diagnostics, {})
    for name, plane in planes.items():
        diagnostics[name] = float(np.mean(np.asarray(to_host(plane), dtype=np.float64)))
    return diagnostics, merge_band_metadata(band_metadata, {})


__all__ = [
    "COLUMN_MEAN", "COUNT", "MAX", "MERGE_RULES", "MIN", "SAME", "SKIP",
    "default_finish", "finish_alone", "merge_band_metadata",
    "merge_band_scalars", "to_host",
]
