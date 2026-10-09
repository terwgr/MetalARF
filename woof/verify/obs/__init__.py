"""Observation-verification battery: the referee and its controls.

This package is the referee half of the observation battery: the statistics,
the harness that runs them, the controls that qualify the instrument, and the
evaluator that turns a set of score files into a verdict.  Fetching and
decoding observations is a separate concern and lives elsewhere; everything
here consumes the seam types in :mod:`woof.verify.obs.contracts` and nothing
else.

Verification *statistics and orchestration* live here.  Diagnostics do not:
the science core for reading and diagnosing history tapes is the ``wrf``
package (distribution ``wrf-rust``), and nothing under this package
reimplements one of its diagnostics.  The one place this package computes a
field itself is :mod:`woof.verify.obs.cross_reader`, whose whole job is to
be an *independent control* on that core -- an instrument that agreed with
the thing it is checking because it called it would measure nothing.

The modules, by mechanism:

:mod:`~woof.verify.obs.contracts`
    the seam: observed fields, station reports, provenance, and the protocols
    a model run and an observation archive satisfy.
:mod:`~woof.verify.obs.fss`
    neighborhood fractions skill score with a validity mask, on the existing
    :mod:`woof.verify.field_metrics` engine.
:mod:`~woof.verify.obs.contingency`
    the 2x2 table and the scores derived from it.
:mod:`~woof.verify.obs.regrid`
    the two registered remap operators, as reusable integer plans.
:mod:`~woof.verify.obs.stations`
    point-observation matching, admission and surface statistics.
:mod:`~woof.verify.obs.model_source`
    one reader for both models' history files, science through the mandated
    core.
:mod:`~woof.verify.obs.registration`
    the pins, hashed before any score is looked at.
:mod:`~woof.verify.obs.battery`
    the scoring pass and the score file it writes.
:mod:`~woof.verify.obs.nowcast`
    the nowcast registration: the same FSS engine over 15 to 60 minute
    leads, with the radar-persistence baseline every first-hour number
    needs beside it.
:mod:`~woof.verify.obs.controls`
    the instrument-qualification controls.
:mod:`~woof.verify.obs.promotion`
    the four-clause promotion evaluator.
:mod:`~woof.verify.obs.stubs`
    loud stand-ins for observations that do not exist yet, which can never
    become a verdict.
:mod:`~woof.verify.obs.cross_reader`
    the independent cross-reader control on the science core (imported
    lazily, like ``model_source``, so the referee imports without it).
:mod:`~woof.verify.obs.t0_parity`
    the t=0 parity check between the two models' initial states.

Nothing in this package names a case.  ``model_source``, ``cross_reader``
and ``t0_parity`` are imported lazily by callers that open history files, so
importing the scorer does not require the science core to be installed.
"""

from woof.verify.obs import (
    battery, contingency, contracts, controls, fss, nowcast, promotion,
    regrid, registration, stations, stubs,
)

__all__ = [
    "battery", "contingency", "contracts", "controls", "fss", "nowcast",
    "promotion", "regrid", "registration", "stations", "stubs",
]
