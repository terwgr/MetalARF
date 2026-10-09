"""Device and host memory sizing for the WOOF global spectral model.

WHY THIS FILE IS NOT ``woof.core.preflight``.  That estimator itemizes a
WRF-shaped experiment -- per-domain state arrays, per-scheme physics
tables, an RRTMGP column workspace, a shared scratch arena.  A spectral
global model has none of those and has one term nothing regional has: the
Legendre basis tables, which are CUBIC in truncation and are the single
largest allocation of a large run.  So the itemization lives here, next to
the shapes it prices, and only the VERDICT VOCABULARY is shared -- free
VRAM measured the same way, the same other-process margin, the same exit
codes, the same "a leg that cannot be measured never passes" rule.

WHAT THE DEVICE FIGURE IS.  The number the run door prints before a byte
is allocated, and the receipt carries beside the measured one, is the
prediction of a model FITTED TO ALLOCATOR-MEASURED PEAKS: the maximum of
the CuPy default pool's live bytes over a whole run, as the runner's
allocator hook reads it (:mod:`woof.globe.device_memory`), on
runs of the native physics suite with the IMEX SSP3 integrator, the
energy ledger sampled and checkpoints written.  It predicts THAT number
and says so; it is not the working set of the dycore alone.  The model it
replaces (2026-09-01) priced the dycore's arrays against a 7.1 GiB
reading that was itself a sampled, not an allocator, figure: it printed
2.21 GiB at T255 where the allocator measures 8.51 GiB and 10.29 GiB at
T533 where the first probe died past 18.6 GiB.  A door sized by it
admitted every card and refused none that mattered.

THE MODEL.  Seven terms.  The first four are structural -- arithmetic on
the config's shapes, with byte counts read off the code that allocates
them -- and the last three are CALIBRATED against the table
:data:`DEVICE_PEAK_CALIBRATION` by least squares, because they are the
transient working sets of the step (kernel scratch, physics workspaces,
the transform's stacked syntheses, the ledger's samples) and no
itemization of those survives a kernel edit:

1. Legendre tables      3 x packed(T, nlat) x float_itemsize
                        + 3 x band x (T+1) x nlat x float_itemsize
2. spectral state       LIVE_SPECTRAL_STACKS x (14 x nlev + 1)
                        x (T+1)^2 x complex_itemsize
3. surface grid fields  SURFACE_GRID_ARRAYS x nlat x nlon x float_itemsize
4. resident grid fields (RESIDENT_GRID_ARRAYS + GRID_TRACER_ARRAYS)
                        x nlev x nlat x nlon x float_itemsize
5. step working set     W x nlev x nlat x nlon x float_itemsize   (fitted)
6. radiation column     R x nlev x min(radiation_column_chunk, nlat x nlon)
   workspace            x float_itemsize                          (fitted)
7. cumulus column       G x nlev x min(cumulus_column_chunk, nlat x nlon)
   workspace            x float_itemsize                          (fitted)
   plus a fixed F bytes                                           (fitted)

Term 6 is the radiation column chunk's workspace: RRTMGP runs the columns
``radiation_column_chunk`` at a time and its per-g-point arrays are sized
by that chunk, not by the grid, so at T63 (18,432 columns, chunk 12,500)
it is most of the peak.  The radiation call is where the peak sat at
every measured truncation up to T255.  Until 2026-09-05 the cloud
preparation ran BEFORE the chunk loop (``cal_cldfra1`` and
``hydrometeor_paths`` in :mod:`woof.globe.core.rrtmgp`) on the whole grid,
its levelled temporaries priced inside W, and at T533 (1.28 M columns,
chunk 5,000) that full-grid preparation was the peak: the T533 probe of
the morning of 2026-09-05 died inside ``hydrometeor_paths`` at 19.25 GiB
under a 20 GiB pool cap, with 50 and 42 levelled 196 MiB arrays named
there by the allocator tracer, and the same probe of the merged tip
(2d12e0b01) at 19.92 GiB.  The preparation now runs on the solver's own
column chunks (the inputs packed per chunk, the cloud fraction, paths
and sizes formed per chunk into their whole-grid arrays), so the
radiation call at T533 asks 3.07 GiB above its entry where it asked
more than 5.9 and never returned, and the T533 peak moved to the
dynamics' right-hand side (18.56 GiB live on the first step under a
20,992 MiB cap) and, under a 22,528 MiB cap on the merged tip, past the
dynamics into the tracer transport (20.74 GiB live, asking 1.66 GiB) and the
cumulus scheme (18.23).  The T533 probe of record still dies under the
20,992 MiB cap, now in the dynamics and by the pool's fragmentation
rather than by live bytes: the free chunks
sit in blocks of the dynamics' own shapes, so handing the pool's cached
blocks back at the step's phase boundaries was tried and refuted (the
dynamics re-grew the pool from 12.4 to 20.3 GiB held inside its own
phase and died at the same place), and T533 stays a floor; every row of
the table was measured on one tree, the merged tip 565b60153 of
2026-09-05 (the chunked preparation, the frozen columns' eleven planes
and the cumulus closure reading all resident), so the table describes
one code.  Without term 6 no single per-column
coefficient fits both ends: the T63 pair (40 and 20 levels) reads 81 MB
per level where the grid arrays account for 12.

Term 7 is the cumulus chunk's workspace, the same shape one chunk size
up: Grell-Freitas packs its columns ``cumulus_column_chunk`` (131,072) at
a time and its per-chunk arrays (the 1,025 MiB and the 16 x nlev x chunk
allocations the tracer names inside ``_cumulus_step`` at T255) grow with
the grid up to that chunk and stop there.  Below the chunk (T63, T127)
they scale like the working set; at and above it (T255 onward) they are
a constant.  Fitted without this term, the three-term model of the
morning of 2026-09-05 read +5.9 percent at T63 and -5.1 percent at T127
against their probes of record, and the excess it had folded into W
carried to T533 as 25.6 GiB where this structure reads 21.6.

``packed(T, nlat)`` is :func:`woof.globe.spectral.legendre.packed_table_
elements`: the tables are stored by bands of orders with the above-
triangle zeros cut off (2026-09-01), so one table is ``(T+1)(T+2)/2 x
nlat`` plus at most ``band - 1`` spare columns per order rather than
``(T+1)^2 x nlat``, and each table's contraction expands one band at a
time into a ``band x (T+1) x nlat`` scratch.  Term 1 is still cubic in
truncation.  That is why a reader whose refusal is tables-bound gets told,
by name, that trimming ``[vertical]`` cannot close it: levels do not
appear in term 1 at all.  Two of the three tables exist after
construction; the derivative table is built by the first gradient, i.e.
the first step, so the runtime figure carries all three.

WHERE THE STRUCTURAL NUMBERS COME FROM.

Terms 1-4 were MEASURED against the artifact on 2026-09-01 and
2026-09-02: the real ``build_transform`` / ``build_model_and_cold_state``
/ ``model.step`` path on the numpy backend, float64, with ``tracemalloc``
filtered to ``numpy.lib.tracemalloc_domain`` so the figure is array bytes
and nothing else.  Six configurations, T21/T31/T42 x nlev 8/12/16,
dealias 1.5.  The coefficients reproduce every one of those six to better
than 0.5%.  Term 1 was re-measured against the packed transform at
T31/T42/T63/T85/T127 in both precisions: it is not fitted, it is the
transform's own byte count.  ``tests/test_arwen_global_sizing.py`` re-takes
that measurement every run.

The numpy backend's step working set is measured directly
(:data:`NUMPY_WORKING_GRID_ARRAYS`, 133.4 array equivalents, a 1.2% spread
across a 5x range of array size) and is a HOST figure: that path opens no
CUDA context and refuses no card.

WHERE THE CALIBRATED NUMBERS COME FROM.  :data:`DEVICE_PEAK_CALIBRATION`
lists every allocator-measured run the fit uses -- truncation, levels,
chunk, the measured peak in bytes, the card and the date -- and
:func:`fit_device_peak_model` solves W, R, G and F from it at import, so
the coefficients cannot be hand-copied beside a table that has since
moved.  :func:`calibration_residuals` reports what the fitted model reads
at each calibration point against what was measured; the door prints the
worst of them beside every prediction.  What the fit does NOT cover is
stated rather than assumed: the native suite at 40 and 20 levels in
float32 at dealias 1.5 with radiation chunks of 12,500 and 5,000 columns
is the measured domain, and a configuration outside it -- another physics
mode, float64, a different dealias factor -- is priced by the same
structure and labelled extrapolated.

WHAT THE FIT IS TRUSTED TO.  Two residuals are printed and they are
different in kind.  The IN-SAMPLE residual is what the fitted model reads
at each row of the table it was solved from.  Rows that share a shape (a
probe and a day of one configuration) contribute the probe-to-day gap
and nothing about the structure, and a table with no more distinct
shapes than coefficients is exactly determined: the table of the morning
of 2026-09-05 -- six rows over four shapes for four coefficients -- read
0.0003 percent at every row by construction and 17 percent at the first
shape measured outside it (T127 at a 5,000-column chunk).  The HELD-OUT
residual (:func:`held_out_residuals`) refits the model with every row of
one shape removed and reads that shape by the refit: it is the error a
shape the table does not contain should expect, and it is the figure a
truncation above the table is labelled with.  On the twelve rows of
2026-09-05 over ten shapes (T63, T127 and T255 at 40 and 20 levels and
at radiation chunks of 12,500 and 5,000, and T255 at 48 levels) the
structure reads every shape within 7.0 percent in sample and 16.1 percent
held out, the T127 shapes
carrying the worst of both (over-read by 7 percent at a 5,000-column
chunk, under-read by 3 to 5 at 12,500); no linear structure of three to six
per-shape regressors tried on those rows did better held out, and a
max-of-phases structure did worse.  A row whose shape is already in the
table tightens the in-sample figure and moves the held-out one only
through the refit, so the table wants NEW SHAPES, not repeats.

THE PROBE OF RECORD.  A day is not needed to read a run's peak, but ten
steps read it only when the radiation is called AGAIN inside them: the
peak sits inside the second radiation call, after a step has left its
state resident, so the ten-step probe of record fires the radiation at
step 0 and step 6 (a 275 s interval at dt 50, 137.5 s at dt 25), samples
the ledger every five steps and writes checkpoints at steps 0, 5 and 10
(:data:`PROBE_RECIPE`).  :data:`PROBE_CALIBRATION` pairs that probe with
a full day on two truncations, in both directions of the instrument:
at T255 the probe reads 9,135,827,456 bytes against the control day's
9,135,851,008 and at T63 3,456,009,728 against the day's 3,456,033,280
(23,552 bytes under at both, 0.0003 and 0.0007 percent), and the same
ten steps with the radiation called once, at step 0, read 7,424,409,088
at T255 and 3,348,668,928 at T63 (18.7 and 3.1 percent under) -- the
recipe the T63 and T127 rows of the first table of 2026-09-05 were
measured with, retired the same day.
:func:`probe_calibration_residuals` re-reads the pairs and the tests
hold them.

WHAT THIS DOES NOT MODEL.  A run's HOST memory during the forecast on the
cupy backend (the device holds the arrays; the host holds the checkpoint
being written), the CUDA context and library workspaces the process holds
outside the pool (measured 2.5 to 3 GB on top of the pool peak at T255),
and the analysis-initialisation decode, which is a mapped-source GRIB read
whose cost belongs to the ingest engine rather than to the spectral model.
The transform-construction host peak IS priced: the recurrence streams
one float64 band at a time, so on the cupy backend the host holds two
float64 band blocks and the per-order solve temporaries and never a whole
table (the dense build held six float64 squares, 9.4 GiB at T533; the
streamed build is priced at 0.21 GiB there).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import sys

import numpy as np

from woof.globe.spectral.grid import GaussianGrid
from woof.globe.spectral.legendre import (
    DEFAULT_BAND,
    expansion_scratch_elements,
    packed_table_elements,
)

from .constants import GRID_TRACERS, SPECTRAL_FIELDS

GIB = 1024 ** 3

#: The recurrence and the Gram solve run in float64 on the host whatever
#: the device precision is, so the host construction figures do not move
#: with ``arwen_global.precision`` even though every device figure does.
_HOST_TABLE_ITEMSIZE = 8

#: Legendre tables the transform keeps resident at run time: the analysis
#: table, the basis and its meridional derivative
#: (``SphericalHarmonicTransform``).  The vector transform's transposed
#: reads are views of the expansion scratch and cost nothing beyond it.
#: Measured 2026-09-01 (T31/T42/T63/T85/T127, numpy, both precisions):
#: 2.003-2.015 packed tables held after construction, and after the first
#: gradient exactly 3 packed tables plus 3 expansion scratches.
DEVICE_LEGENDRE_TABLES = 3

#: Tables that exist when construction returns: the basis and the
#: analysis table.  The derivative is built by the first gradient.
BUILD_LEGENDRE_TABLES = 2

#: The semi-Lagrangian core's second time level (semilag.state
#: TrajectoryState): the previous step's advected winds and scalar, the
#: previous nonlinear residuals (six levelled grid volumes) and one
#: plane (the log surface pressure residual), in the model's float
#: dtype.  Counted off the state class; the integrator names are
#: mirrored from semilag.step so the sizer imports no core.
SEMILAG_TRAJECTORY_LEVELLED_ARRAYS = 6
SEMILAG_TRAJECTORY_PLANES = 1
SEMILAG_INTEGRATOR_NAMES = ("sl_si",)
#: The gather holds three whole-grid stacks of one batch: the staged
#: source batch, the contiguous source stack and the gathered stack
#: (interpolate.gather_batch); the widest batch is the eleven scalars
#: (vapour and the ten grid tracers).  MEASURED 2026-09-07, RTX 5070 Ti,
#: T533 L40 sl_si with every slice parked and sixteen bands: the card ran
#: out inside this gather at 15.96 GB, with the physics banded.
SEMILAG_GATHER_STACKS = 3
SEMILAG_GATHERED_FIELDS = 11

#: Float64 band blocks live on the host at the construction peak: the
#: recurrence's basis band and the analysis band being solved into.
#: Beside them the per-order Gram solve holds its rows, the Gram matrix
#: and LAPACK's copies, priced as :data:`HOST_SOLVE_TEMPORARIES` rows of
#: ``(T+1) x nlat`` float64.  Measured 2026-09-01 on numpy float32
#: (where the resident tables are float32 and the float64 transient shows
#: on its own): peak minus resident was 3.3 MB at T63 against 3.3 MB
#: priced, 5.4 against 5.9 at T85, 8.3 against 13.2 at T127 -- the
#: pricing is conservative by up to a third, in the refusing direction.
HOST_BAND_BLOCKS = 2
HOST_SOLVE_TEMPORARIES = 3

#: Spectral field stacks the run holds BETWEEN steps.  Two, not one:
#: ``runner.run`` keeps ``cold`` referenced for the whole forecast (its
#: diagnostics are the drift gates' reference) alongside the current
#: state.  Measured -- the after-a-step footprint of all six probe
#: configurations agrees with a two-stack resident to 0.07% at the
#: smallest and 0.4% at the largest, 2026-09-01.
RESIDENT_SPECTRAL_STACKS = 2

#: Spectral field stacks live at once at the peak of one SSPRK3 step.
#: Read off ``woof.globe.spectral.timestep.ssprk3_step``: the widest
#: moment holds the incoming state, the stage it was built from, the
#: tendency just evaluated and the combination being formed (``state``,
#: ``y2``, ``k3``, ``e3``).  RK4 is not wider -- it holds ``state``, one
#: stage and at most two tendencies.  The IMEX SSP3 pair is priced by the
#: same count; what it holds beyond it is in the fitted working set.
LIVE_SPECTRAL_STACKS = 4

#: Two-dimensional grid fields the built model holds: the twenty-two
#: :class:`~woof.globe.state.SurfaceState` members (twelve
#: reservoirs plus the ten static surface fields the state grew on
#: 2026-09-01), the surface geopotential, the persistent physics surface
#: state, and the transform's own latitude vectors.  Measured as 37.70
#: array equivalents after the first step with the twelve-member surface
#: (2026-09-01, six configurations); re-measured 62.55 with the
#: twenty-two-member surface the same day over T21/T31/T42 x nlev
#: 8/12/16 (61.60 .. 62.92; the ten new members are held by the cold
#: state, the advancing state and the step's own copies).
SURFACE_GRID_ARRAYS = 62.55

#: Levelled ``(nlev, nlat, nlon)`` grid fields the model holds between
#: steps beyond the grid tracers -- the persistent physics state and the
#: transform's levelled scratch.  Re-measured 2026-09-02 with the grid
#: tracers priced on their own (:data:`GRID_TRACER_ARRAYS`): the resident
#: set of a stepped T31 numpy model minus tables, the two spectral
#: stacks, the surface fields and the twenty tracer arrays is 0.763 /
#: 0.863 / 0.973 array equivalents at nlev 8 / 12 / 16.  The 4.889 of
#: 2026-09-01 was fitted when the ten tracers were spectral coefficients
#: and stood in for what those coefficients' syntheses left resident.
RESIDENT_GRID_ARRAYS = 0.90

#: What the synthesis memo (dynamics.SynthesisMemo) holds between steps:
#: the finished state's syntheses the guards read last, served to the
#: next step's first exchange.  MEASURED 2026-09-05 on the stepped T31
#: numpy model, exactly the memo's entries: the theta and vapor stack (2
#: levelled arrays), temperature (1), the pressure block's full-level
#: pressure, thickness and log ratio (3), plus its half-level pressure
#: (nlev + 1 planes) and the surface pressure and its logarithm (2
#: planes).  An attached in-situ ledger reads the wind, the virtual
#: temperature and the geopotential too (4 more levelled arrays); the
#: model releases every entry before each physics suite call, so the
#: step's peak carries none of this.
SYNTHESIS_MEMO_GRID_ARRAYS = 6
SYNTHESIS_MEMO_PLANES_BEYOND_LEVELS = 3

#: Levelled grid fields live at the PEAK of one step, above the resident
#: set, on the numpy reference backend.  MEASURED, 2026-09-01.
NUMPY_WORKING_GRID_ARRAYS = 133.4

#: The radiation column chunk a config gets when it names none: the
#: native suite's own default (``physics.native_options.NativePhysicsOptions``),
#: read from there rather than typed so the two cannot drift.
def _default_radiation_column_chunk() -> int:
    from .physics.native_options import NativePhysicsOptions

    return int(NativePhysicsOptions.radiation_column_chunk)


#: The cumulus column chunk a config gets when it names none, read from
#: the native suite's own default for the same reason.
def _default_cumulus_column_chunk() -> int:
    from .physics.native_options import NativePhysicsOptions

    return int(NativePhysicsOptions.cumulus_column_chunk)


#: What the calibrated device figure MEASURES, in the words the door and
#: the receipt print.  A number without this sentence beside it is the
#: sampled 7.1 GiB all over again.
DEVICE_PEAK_MEASURES = (
    "maximum live bytes of the CuPy default memory pool over a whole run, "
    "read at the allocator (tables, initial state and every step, the "
    "energy ledger's samples and the checkpoint writes included); fitted "
    "to allocator-measured runs of the native physics suite with the IMEX "
    "SSP3 integrator in float32 at dealias 1.5"
)


def legendre_runtime_bytes(truncation: int, nlat: int, itemsize: int,
                           band: int = DEFAULT_BAND,
                           streaming: bool = False) -> int:
    """Term 1: the three packed tables and their three expansion scratches.

    The band is the transform's own ``legendre_band``, which became a
    reachable door on 2026-09-06 and which the packed table's size
    depends on (the triangle is padded up to whole bands): MEASURED
    analytically at T799 x 1200, 4.297 GiB at band 1 against 4.958 at
    band 128, so pricing every run at the shipped 32 under-charges a
    wider band in the ADMITTING direction.

    ``streaming=True`` holds NO table at all: the basis is regenerated
    one band of orders per call and nothing survives the call, so the
    resident term becomes the working set of one streamed contraction
    (``SphericalHarmonicTransform.streaming_working_bytes``: a float64
    recurrence block, the float64 Gram-solved rows, the backend-dtype
    cast and the dense expansion, each ``band x (T+1) x nlat``).
    Charging the resident tables for a streamed run refuses, on table
    bytes that are never allocated, exactly the truncation the streamed
    transform exists to reach.
    """

    if streaming:
        elements = int(band) * (int(truncation) + 1) * int(nlat)
        return elements * (2 * _HOST_TABLE_ITEMSIZE + 2 * itemsize)
    packed = packed_table_elements(truncation, nlat, band)
    scratch = expansion_scratch_elements(truncation, nlat, band)
    return DEVICE_LEGENDRE_TABLES * (packed + scratch) * itemsize


def legendre_build_bytes(truncation: int, nlat: int, itemsize: int,
                         band: int = DEFAULT_BAND,
                         streaming: bool = False) -> int:
    """What construction leaves resident: the basis and the analysis table.

    Zero under ``streaming``: there is no construction to leave anything.
    """

    if streaming:
        return 0
    return BUILD_LEGENDRE_TABLES * packed_table_elements(
        truncation, nlat, band) * itemsize


def host_transform_transient_bytes(truncation: int, nlat: int,
                                   band: int = DEFAULT_BAND,
                                   streaming: bool = False) -> int:
    """Float64 the host holds at the construction peak beyond the tables.

    A streamed transform pays the same per-band recurrence and solve, one
    band at a time, on every call instead of once at construction, so the
    figure is the same shape and is not zeroed here.
    """

    rows = (int(truncation) + 1) * int(nlat)
    return (HOST_BAND_BLOCKS * expansion_scratch_elements(truncation, nlat, band)
            + HOST_SOLVE_TEMPORARIES * rows) * _HOST_TABLE_ITEMSIZE


def _spectral_stack_coefficients(nlev: int) -> int:
    """Complex coefficients in ONE :class:`MoistHybridState`.

    Four ``(nlev, T+1, T+1)`` fields and one ``(T+1, T+1)`` surface
    field, counted from :data:`SPECTRAL_FIELDS` rather than typed, so a
    field added to the spectral state is priced without touching this
    file.  The ten grid tracers are priced by :data:`GRID_TRACER_ARRAYS`
    among the levelled grid fields.
    """

    return (len(SPECTRAL_FIELDS) - 1) * int(nlev) + 1


#: Levelled ``(nlev, nlat, nlon)`` grid tracers ONE state holds (the
#: condensate species and the number moments, grid-point since
#: 2026-09-02), times the two states the run keeps referenced between
#: steps (:data:`RESIDENT_SPECTRAL_STACKS`: ``cold`` and the advancing
#: state).  Counted from :data:`GRID_TRACERS`, not typed.
GRID_TRACER_ARRAYS = 2 * len(GRID_TRACERS)


#: The ten-step probe that reads a full day's device peak, in the words
#: its configs are written with: the control's physics options, dt 50
#: (25 at T533), duration ten steps, output every five, the energy ledger
#: every five, and the radiation interval set so the second radiation
#: call lands at step 6.  Calibrated in :data:`PROBE_CALIBRATION`.
PROBE_RECIPE = (
    "ten steps of the native suite with the ledger sampled every five "
    "steps, checkpoints at steps 0, 5 and 10, and the radiation called at "
    "step 0 and again at step 6 (radiation_interval_s = 5.5 x dt)"
)

#: The probe of record against the full day it stands for.  Two
#: truncations, so the instrument is held in both directions: a probe
#: that under-reads its day is caught at either end, and the fraction
#: is what a probe row of :data:`DEVICE_PEAK_CALIBRATION` is trusted to.
#: Every figure is a receipt's ``device_memory.peak_used_bytes``.
PROBE_CALIBRATION = (
    {
        "label": "T255, 40 levels",
        "probe_bytes": 9_200_314_368,
        "full_day_bytes": 9_200_513_024,
        "probe": "the 32 GB host the final-t255l40 leg, RTX 5090, 2026-09-05 (the merged tip 565b60153)",
        "full_day": "the merged tip 565b60153's arm of record of the control case, the 32 GB host the control-case arm, RTX 5090, 2026-09-05 (1728 steps at dt 50, hourly checkpoints)",
        # Measured on the whole-grid preparation, retired
        # there as an under-reader; kept as the record of what it read.
        "single_radiation_probe_bytes": 7_424_409_088,
    },
    {
        "label": "T63, 40 levels",
        "probe_bytes": 3_460_084_736,
        "full_day_bytes": 3_460_106_240,
        "probe": "the 32 GB host the final-t63l40 leg, RTX 5090, 2026-09-05 (the merged tip 565b60153)",
        "full_day": "the 32 GB host the final-t63day leg, RTX 5090, 2026-09-05 (1728 steps at dt 50, ledger every 50, hourly checkpoints, the merged tip 565b60153)",
        "single_radiation_probe_bytes": 3_348_668_928,
    },
)


def probe_calibration_residuals() -> tuple[dict[str, object], ...]:
    """What the ten-step probe reads against the full day, per pair.

    ``residual_fraction`` is ``(probe - full_day) / full_day``: negative
    means the probe under-reads the day, the direction that would admit
    a run the card cannot finish.
    """

    rows = []
    for pair in PROBE_CALIBRATION:
        probe = int(pair["probe_bytes"])
        day = int(pair["full_day_bytes"])
        rows.append({
            "label": pair["label"],
            "probe_bytes": probe,
            "full_day_bytes": day,
            "residual_fraction": (probe - day) / day,
            "single_radiation_probe_fraction": (
                (int(pair["single_radiation_probe_bytes"]) - day) / day
                if pair.get("single_radiation_probe_bytes") else None),
        })
    return tuple(rows)


#: Allocator-measured device peaks the calibrated terms are fitted to.
#: Every row is a receipt's ``device_memory.peak_used_bytes`` from
#: :class:`woof.globe.device_memory.DevicePeakTracker` on a run of
#: the native suite (RRTMGP at the named column chunk, sfclay, Noah, YSU,
#: Grell-Freitas at the default 131,072-column chunk, Morrison) with the
#: IMEX SSP3 integrator, float32, dealias 1.5, the energy ledger sampled
#: inside the run and checkpoints written inside it.  Probe rows are the
#: probe of record (:data:`PROBE_RECIPE`, held to the day by
#: :data:`PROBE_CALIBRATION`), so probes and full arms sit in one table.
#: The T63 and T127 rows measured on 2026-09-05 with the radiation called
#: once are retired: that probe read 18.7 percent under the day at T255.
#: The five rows at a 5,000-column radiation chunk or 20 levels at T127
#: and T255 were measured on the afternoon of 2026-09-05 as shapes the
#: morning's table did not contain (the 32 GB host), which is
#: what turned the exactly-determined fit into an over-determined one.
#:
#: Add a row for a NEW SHAPE first (a truncation, level count or chunk
#: the table lacks); a re-measured shape tightens only the in-sample
#: figure.  Never edit a coefficient: the fit below re-solves at import.
DEVICE_PEAK_CALIBRATION = (
    {
        "label": "T63, 40 levels, ten-step probe",
        "truncation": 63, "nlat": 96, "nlon": 192, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 3_460_084_736,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 50 on the merged tip (the 32 GB host the final-t63l40 leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T63, 20 levels, ten-step probe",
        "truncation": 63, "nlat": 96, "nlon": 192, "nlev": 20,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 1_797_618_176,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 50 on the merged tip (the 32 GB host the final-t63l20 leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T127, 40 levels, ten-step probe",
        "truncation": 127, "nlat": 192, "nlon": 384, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 5_014_948_864,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 50 on the merged tip (the 32 GB host the final-t127l40 leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T255, 40 levels, ten-step probe",
        "truncation": 255, "nlat": 384, "nlon": 768, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 9_200_314_368,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 50 on the merged tip (the 32 GB host the final-t255l40 leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T63, 40 levels, 5,000-column chunk, ten-step probe",
        "truncation": 63, "nlat": 96, "nlon": 192, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 5_000, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 1_707_686_912,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 50 on the merged tip (the 32 GB host the final-t63l40c5 leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T127, 20 levels, ten-step probe",
        "truncation": 127, "nlat": 192, "nlon": 384, "nlev": 20,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 2_639_778_304,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 50 on the merged tip (the 32 GB host the final-t127l20 leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T127, 40 levels, 5,000-column chunk, ten-step probe",
        "truncation": 127, "nlat": 192, "nlon": 384, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 5_000, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 2_933_238_784,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 50 on the merged tip (the 32 GB host the final-t127l40c5 leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T255, 20 levels, ten-step probe",
        "truncation": 255, "nlat": 384, "nlon": 768, "nlev": 20,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 4_998_993_408,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 50 on the merged tip (the 32 GB host the final-t255l20 leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T255, 40 levels, 5,000-column chunk, ten-step probe",
        "truncation": 255, "nlat": 384, "nlon": 768, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 5_000, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 7_668_754_432,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 50 on the merged tip (the 32 GB host the final-t255c5 leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T63, 40 levels, 24 h arm",
        "truncation": 63, "nlat": 96, "nlon": 192, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 3_460_106_240,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "1728 steps at dt 50, ledger every 50 steps, hourly checkpoints on the merged tip (the 32 GB host the final-t63day leg, RTX 5090, 2026-09-05, the merged tip 565b60153)",
    },
    {
        "label": "T255, 40 levels, 24 h control",
        "truncation": 255, "nlat": 384, "nlon": 768, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 9_200_513_024,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "1728 steps at dt 50, ledger every 3 steps, hourly checkpoints: the merged tip 565b60153's arm of record on the control case (the 32 GB host the control-case arm, RTX 5090, 2026-09-05)",
    },
    {
        "label": "T383, 40 levels, ten-step probe",
        "truncation": 383, "nlat": 576, "nlon": 1152, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 16_706_958_848,
        "measured_on": "2026-09-06", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 40 (radiation_interval_s = 220 = 5.5 x dt, so the second radiation call lands at step 6), one latitude band, host_spill off, on the scale-out tree (the 32 GB host, the T383 12,500-column one-band probe, RTX 5090, 2026-09-06): the pool held 18,396,111,872 B at that peak and nvidia-smi read 18,300 MiB against the process, which is where this shape's fragmentation and out-of-pool rows come from.  It is the FIRST measured row above T255, and the model read 15.01 GiB against the 15.56 measured -- 3.5 percent under, in the admitting direction, which is the defect this row closes",
    },
    {
        "label": "T383, 40 levels, 5,000-column chunk, ten-step probe",
        "truncation": 383, "nlat": 576, "nlon": 1152, "nlev": 40,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 5_000, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 15_208_199_680,
        "measured_on": "2026-09-06", "device": "NVIDIA RTX 5090",
        "run": "the probe of record at dt 40, one latitude band, host_spill off, on the scale-out tree (the 32 GB host, the T383 5,000-column one-band probe, RTX 5090, 2026-09-06): the pool held 18,279,974,912 B at that peak and nvidia-smi read 18,188 MiB against the process.  It is the radiation chunk's lever measured at the largest truncation the table holds, where the chunk coefficient had only small shapes under it",
    },
    {
        "label": "T255, 48 levels, ten-step probe",
        "truncation": 255, "nlat": 384, "nlon": 768, "nlev": 48,
        "precision": "float32", "dealias_factor": 1.5,
        "radiation_column_chunk": 12_500, "cumulus_column_chunk": 131_072,
        "peak_used_bytes": 10_881_077_248,
        "measured_on": "2026-09-05", "device": "NVIDIA RTX 5090",
        "run": "the ten-step probe of record at dt 50 of the jet_refined 48-level layout on the merged tip (the 32 GB host the final-t255l48 leg, RTX 5090, 2026-09-05, the merged tip 565b60153); the level set's own 24 h arm on its lane tree read 10,826,080,256 B (the 16 GB host the l48 arm, RTX 5070 Ti)",
    },
)


def _calibration_design_row(point: dict) -> tuple[float, float, float, float]:
    """The four fitted regressors of one calibration point.

    ``(nlev x points x itemsize, nlev x min(radiation chunk, points) x
    itemsize, nlev x min(cumulus chunk, points) x itemsize, 1)`` -- the
    step working set's scale, the radiation column workspace's scale,
    the cumulus column workspace's scale, and the fixed term -- in the
    same arithmetic :func:`estimate_global_memory` prices a config with,
    so a point and a prediction can never be built from two different
    formulas.
    """

    itemsize = 4 if point["precision"] == "float32" else 8
    points = int(point["nlat"]) * int(point["nlon"])
    nlev = int(point["nlev"])
    chunk = min(int(point["radiation_column_chunk"]), points)
    cumulus = min(int(point["cumulus_column_chunk"]), points)
    return (float(nlev * points * itemsize), float(nlev * chunk * itemsize),
            float(nlev * cumulus * itemsize), 1.0)


def _structural_bytes(truncation: int, nlat: int, nlon: int, nlev: int,
                      itemsize: int) -> int:
    """Terms 1-4 for one shape: what the fit subtracts before solving."""

    complex_itemsize = 2 * itemsize
    points = nlat * nlon
    coefficients = (int(truncation) + 1) ** 2
    return (
        legendre_runtime_bytes(truncation, nlat, itemsize)
        + math.ceil(LIVE_SPECTRAL_STACKS * _spectral_stack_coefficients(nlev)
                    * coefficients * complex_itemsize)
        + math.ceil(SURFACE_GRID_ARRAYS * points * itemsize)
        + math.ceil((RESIDENT_GRID_ARRAYS + GRID_TRACER_ARRAYS) * nlev
                    * points * itemsize)
    )


def fit_device_peak_model(calibration=DEVICE_PEAK_CALIBRATION
                          ) -> dict[str, float]:
    """Solve the four calibrated coefficients from the table.

    Least squares on ``peak - structural = W x a + R x b + G x c + F``
    over every row, unweighted in bytes: the large configurations, which
    are the ones a card refuses, carry the fit, and the small ones report
    their own residual rather than steering it.  A negative coefficient
    fails loudly -- it would mean the structural terms and the
    measurements disagree, and a model with a negative working set is
    not a number anything should be sized against.
    """

    rows = [_calibration_design_row(point) for point in calibration]
    targets = [
        float(point["peak_used_bytes"]) - _structural_bytes(
            point["truncation"], point["nlat"], point["nlon"], point["nlev"],
            4 if point["precision"] == "float32" else 8)
        for point in calibration
    ]
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    # Column scaling keeps the normal equations well conditioned: the
    # regressors span 1 to 1e11.
    scale = np.max(np.abs(design), axis=0)
    solution, *_ = np.linalg.lstsq(design / scale, target, rcond=None)
    working, physics, cumulus, fixed = (solution / scale).tolist()
    if working <= 0 or physics <= 0 or cumulus <= 0:
        raise ValueError(
            "the device-peak calibration solved to a non-positive working "
            f"set (W={working:.3g}, R={physics:.3g}, G={cumulus:.3g}): the "
            "structural terms and the measured peaks disagree and neither "
            "can be trusted until that is resolved")
    return {
        "working_grid_arrays": working,
        "physics_column_arrays": physics,
        "cumulus_column_arrays": cumulus,
        "fixed_bytes": max(0.0, fixed),
    }


#: The fitted coefficients, solved at import from
#: :data:`DEVICE_PEAK_CALIBRATION`.
DEVICE_PEAK_FIT = fit_device_peak_model()

#: Levelled grid fields live at the peak of one step on the cupy backend,
#: above the resident set: FITTED (term 5).
CUPY_WORKING_GRID_ARRAYS = DEVICE_PEAK_FIT["working_grid_arrays"]

#: Per-column, per-level physics workspace of the radiation chunk, in
#: array equivalents of ``nlev x chunk``: FITTED (term 6).
PHYSICS_COLUMN_ARRAYS = DEVICE_PEAK_FIT["physics_column_arrays"]

#: Per-column, per-level cumulus workspace of the Grell-Freitas chunk, in
#: array equivalents of ``nlev x min(cumulus_column_chunk, points)``:
#: FITTED (term 7).
CUMULUS_COLUMN_ARRAYS = DEVICE_PEAK_FIT["cumulus_column_arrays"]

#: Shape-independent bytes the fit leaves over: FITTED.
FIXED_DEVICE_BYTES = int(round(DEVICE_PEAK_FIT["fixed_bytes"]))


def predict_device_peak_bytes(truncation: int, nlat: int, nlon: int,
                              nlev: int, itemsize: int,
                              radiation_column_chunk: int,
                              cumulus_column_chunk: int | None = None) -> int:
    """The calibrated model's figure for one shape, in bytes."""

    points = nlat * nlon
    chunk = min(int(radiation_column_chunk), points)
    if cumulus_column_chunk is None:
        cumulus_column_chunk = _default_cumulus_column_chunk()
    cumulus = min(int(cumulus_column_chunk), points)
    return int(
        _structural_bytes(truncation, nlat, nlon, nlev, itemsize)
        + math.ceil(CUPY_WORKING_GRID_ARRAYS * nlev * points * itemsize)
        + math.ceil(PHYSICS_COLUMN_ARRAYS * nlev * chunk * itemsize)
        + math.ceil(CUMULUS_COLUMN_ARRAYS * nlev * cumulus * itemsize)
        + FIXED_DEVICE_BYTES)


def calibration_residuals(calibration=DEVICE_PEAK_CALIBRATION
                          ) -> tuple[dict[str, object], ...]:
    """What the fitted model reads at each calibration point.

    One row per point: the measured peak, the model's figure for that
    shape, and the difference in bytes and as a fraction of the
    measurement (positive means the model over-predicts, the refusing
    direction).
    """

    rows = []
    for point in calibration:
        itemsize = 4 if point["precision"] == "float32" else 8
        predicted = predict_device_peak_bytes(
            point["truncation"], point["nlat"], point["nlon"], point["nlev"],
            itemsize, point["radiation_column_chunk"],
            point["cumulus_column_chunk"])
        measured = int(point["peak_used_bytes"])
        rows.append({
            "label": point["label"],
            "measured_bytes": measured,
            "predicted_bytes": predicted,
            "residual_bytes": predicted - measured,
            "residual_fraction": (predicted - measured) / measured,
        })
    return tuple(rows)


def worst_calibration_residual_fraction() -> float:
    """The largest |residual| / measured over the table, IN SAMPLE."""

    return max(abs(row["residual_fraction"]) for row in calibration_residuals())


def _shape_key(point: dict) -> tuple:
    """What makes two rows the same design point: the regressors and the
    structural terms depend on nothing else."""

    points = int(point["nlat"]) * int(point["nlon"])
    return (int(point["truncation"]), int(point["nlat"]), int(point["nlon"]),
            int(point["nlev"]), point["precision"],
            min(int(point["radiation_column_chunk"]), points),
            min(int(point["cumulus_column_chunk"]), points))


def calibration_shapes(calibration=DEVICE_PEAK_CALIBRATION) -> int:
    """Distinct design points in the table.  The fit is over-determined
    only while this exceeds the number of fitted coefficients (four)."""

    return len({_shape_key(point) for point in calibration})


def held_out_residuals(calibration=DEVICE_PEAK_CALIBRATION
                       ) -> tuple[dict[str, object], ...]:
    """What the model reads at each shape when that shape is NOT in the
    fit: every row of the shape is removed, the coefficients are re-solved
    from the rest, and the removed rows are priced by the refit.

    One row per calibration point, ``residual_fraction`` signed as in
    :func:`calibration_residuals` (positive over-predicts).  A shape whose
    removal leaves fewer distinct shapes than coefficients, or a refit
    with a non-positive coefficient, reports ``None``: that shape is the
    only evidence for a term and its held-out error is not a number.
    """

    rows = []
    for point in calibration:
        key = _shape_key(point)
        rest = tuple(other for other in calibration if _shape_key(other) != key)
        fit = None
        if len({_shape_key(other) for other in rest}) >= 4:
            try:
                fit = fit_device_peak_model(rest)
            except (ValueError, np.linalg.LinAlgError):
                fit = None
        itemsize = 4 if point["precision"] == "float32" else 8
        measured = int(point["peak_used_bytes"])
        if fit is None:
            predicted = None
        else:
            design = _calibration_design_row(point)
            predicted = int(
                _structural_bytes(point["truncation"], point["nlat"],
                                  point["nlon"], point["nlev"], itemsize)
                + fit["working_grid_arrays"] * design[0]
                + fit["physics_column_arrays"] * design[1]
                + fit["cumulus_column_arrays"] * design[2]
                + fit["fixed_bytes"])
        rows.append({
            "label": point["label"],
            "measured_bytes": measured,
            "held_out_predicted_bytes": predicted,
            "residual_fraction": (
                None if predicted is None else (predicted - measured) / measured),
        })
    return tuple(rows)


def worst_held_out_residual_fraction() -> float | None:
    """The largest |held-out residual| / measured over the table; ``None``
    when no shape can be held out."""

    values = [abs(row["residual_fraction"]) for row in held_out_residuals()
              if row["residual_fraction"] is not None]
    return max(values) if values else None


#: The measured domain, as a set of ``(name, value)`` pairs a config is
#: compared against; a config outside it is priced by the same structure
#: and labelled extrapolated.
CALIBRATED_DOMAIN = {
    "physics_mode": "arwen-native",
    "precision": "float32",
    "dealias_factor": 1.5,
    "nlev": tuple(sorted({int(point["nlev"]) for point in DEVICE_PEAK_CALIBRATION})),
    "truncation": tuple(sorted({int(point["truncation"])
                               for point in DEVICE_PEAK_CALIBRATION})),
}


def extrapolation_notes(cfg, nlev: int) -> tuple[str, ...]:
    """Why this config's figure is an extrapolation, one note per reason;
    empty inside the measured domain."""

    notes = []
    if cfg.physics_mode != CALIBRATED_DOMAIN["physics_mode"]:
        notes.append(
            f"physics.mode = {cfg.physics_mode!r} is priced as the native "
            "suite the fit measured (a conservative over-prediction for a "
            "suite with no radiation column workspace)")
    if cfg.precision != CALIBRATED_DOMAIN["precision"]:
        notes.append(
            f"precision = {cfg.precision!r} doubles every device term of a "
            "float32 fit; no float64 run was measured")
    if abs(float(cfg.dealias_factor) - CALIBRATED_DOMAIN["dealias_factor"]) > 1e-9:
        notes.append(
            f"dealias_factor = {cfg.dealias_factor:g} moves the grid the fit "
            "was measured on (1.5)")
    if nlev not in CALIBRATED_DOMAIN["nlev"]:
        notes.append(
            f"nlev = {nlev} is outside the measured level counts "
            f"{CALIBRATED_DOMAIN['nlev']}; the level scaling is the "
            "structure's, checked between 20 and 40 at T63 and between "
            "40 and 48 at T255")
    largest = max(CALIBRATED_DOMAIN["truncation"])
    if int(cfg.truncation) > largest:
        held_out = worst_held_out_residual_fraction()
        notes.append(
            f"T{int(cfg.truncation)} is above the largest measured "
            f"truncation (T{largest}); the grid scaling beyond it is the "
            "structure's, not a measurement, and the structure's held-out "
            "error over the measured shapes is "
            + ("not a number" if held_out is None
               else f"up to {100 * held_out:.0f}%"))
        completed = measured_runs_at(cfg.truncation, nlev, cfg.precision)
        if completed:
            worst = max(completed, key=lambda row: int(row["peak_used_bytes"]))
            notes.append(
                f"this shape HAS run: {worst['label']} reached "
                f"{int(worst['peak_used_bytes']) / GIB:.2f} GiB of pool on "
                f"{worst['device']} ({worst['measured_on']}) and completed, "
                "at a recipe the calibration table cannot carry, so it "
                "bounds the figure below without moving the fit")
    return tuple(notes)


def _anchor_working_grid_arrays() -> float:
    """Retired: the working coefficient used to be derived from one 7.1
    GiB sampled reading.  Kept as a name so a stale import fails with a
    sentence rather than an AttributeError."""

    raise RuntimeError(
        "the T533 anchor was retired on 2026-09-05: the working set is "
        "fitted to DEVICE_PEAK_CALIBRATION (allocator-measured peaks), see "
        "fit_device_peak_model")



#: The two legs this preflight decides on.  DELIBERATELY NEW IDENTIFIERS:
#: ``woof.core.preflight.N0_GATE_METRICS`` are pre-registered ledger
#: records that ``verify/nest_gates.py`` reads for the regional model, and
#: a global run reporting under those keys would write a WRF gate record
#: from a run that has no WRF anything in it.
GLOBAL_GATE_METRICS = (
    "arwen_global_legendre_tables_fit_device",
    "arwen_global_step_peak_fits_device",
)

#: What each leg is called in the report.
GATE_DISPLAY = {
    "arwen_global_legendre_tables_fit_device":
        "the Legendre tables fit the device budget",
    "arwen_global_step_peak_fits_device":
        "the run's device peak fits the device budget",
}


@dataclass(frozen=True)
class GlobalMemoryEstimate:
    """Itemized device and host cost of one WOOF global configuration."""

    name: str
    backend: str
    precision: str
    truncation: int
    nlat: int
    nlon: int
    nlev: int
    dealias_factor: float
    float_itemsize: int
    complex_itemsize: int
    legendre_table_bytes: int
    legendre_build_bytes: int
    spectral_state_bytes: int
    surface_grid_bytes: int
    resident_grid_bytes: int
    working_grid_bytes: int
    physics_column_bytes: int
    cumulus_column_bytes: int
    fixed_bytes: int
    host_transform_peak_bytes: int
    working_grid_arrays: float
    radiation_column_chunk: int
    cumulus_column_chunk: int
    extrapolation: tuple[str, ...]
    #: The synthesis memo's entries, held between steps and released
    #: before every physics suite call (:data:`SYNTHESIS_MEMO_GRID_ARRAYS`):
    #: in the resident set, not in the step's peak.  Zero when the run
    #: builds no memo.
    memo_grid_bytes: int = 0
    #: The transform's own band and table residence, so the itemization
    #: names the arithmetic this estimate was taken for rather than the
    #: shipped default.  Appended after every existing field so a
    #: positional construction keeps its meaning.
    legendre_band: int = DEFAULT_BAND
    streaming: bool = False
    #: The semi-Lagrangian core's second time level (semilag.state
    #: TrajectoryState: six levelled grid volumes and one plane), held
    #: for the whole run beside everything above.  Zero for every other
    #: integrator.  Appended after every existing field so a positional
    #: construction keeps its meaning.
    trajectory_bytes: int = 0
    #: The semi-Lagrangian gather's own transient: two stacks of
    #: ``gather_batch`` whole grid volumes (the source batch and the
    #: gathered batch, interpolate.gather_batch) and the staged batch when
    #: the tier holds the tracers.  Whole-grid by construction -- a
    #: departure point reads any row of the globe -- so the band count
    #: does not divide it.  NOT in the one-band peak, where it sits inside
    #: the fitted working envelope (MEASURED: T255 sl_si at one band reads
    #: 8.805 GiB against the IMEX calibration's 8.569, the 0.24 GiB being
    #: the trajectory state, 0.26); it enters the BANDED peak whole, because a
    #: band count that shrinks everything else leaves this standing
    #: (:func:`banded_device_peak_bytes`).  Zero for every other integrator.
    semilag_gather_bytes: int = 0

    @property
    def resident_bytes(self) -> int:
        """Held for the whole run, between steps.

        The peak's spectral term divided down from
        :data:`LIVE_SPECTRAL_STACKS` to :data:`RESIDENT_SPECTRAL_STACKS`;
        every other term above is already a resident one.
        """

        return (self.legendre_table_bytes
                + (self.spectral_state_bytes * RESIDENT_SPECTRAL_STACKS
                   // LIVE_SPECTRAL_STACKS)
                + self.surface_grid_bytes + self.resident_grid_bytes
                + self.memo_grid_bytes + self.trajectory_bytes)

    @property
    def device_peak_bytes(self) -> int:
        """The figure a card must have FREE for this run to complete:
        the calibrated model's prediction of the allocator-measured pool
        peak (:data:`DEVICE_PEAK_MEASURES`).

        Zero on the numpy backend: that path allocates on the host and
        never opens a CUDA context, so there is no device figure to
        compare and no card to refuse.
        """

        if self.backend != "cupy":
            return 0
        return (self.legendre_table_bytes + self.spectral_state_bytes
                + self.surface_grid_bytes + self.resident_grid_bytes
                + self.working_grid_bytes + self.physics_column_bytes
                + self.cumulus_column_bytes + self.fixed_bytes
                + self.trajectory_bytes)

    @property
    def host_peak_bytes(self) -> int:
        """Peak HOST memory, whichever phase binds it.

        On the cupy backend the host holds only the construction
        transient (the float64 recurrence streams one band at a time);
        on the numpy backend the tables are host arrays and the step
        itself is wider still, so both are weighed and the larger is
        reported.
        """

        if self.backend == "cupy":
            return self.host_transform_peak_bytes
        return max(self.host_transform_peak_bytes,
                   self.resident_bytes - self.memo_grid_bytes
                   + self.working_grid_bytes)

    @property
    def calibrated(self) -> bool:
        """Inside the measured domain of the fit."""

        return not self.extrapolation

    def itemization(self) -> tuple[tuple[str, int, str], ...]:
        """``(label, bytes, why)`` rows, largest term first is NOT sorted:
        the order is allocation order, because that is the order a run
        dies in and the first row that does not fit is the one that
        kills it."""

        rows = [
            ("Legendre tables", self.legendre_table_bytes,
             (f"{DEVICE_LEGENDRE_TABLES} x packed (T+1)(T+2)/2 x nlat plus "
              f"{DEVICE_LEGENDRE_TABLES} x {self.legendre_band} x (T+1) x "
              "nlat scratch -- cubic in truncation, and the levels do not "
              "appear in it")
             if not self.streaming else
             (f"streamed: no resident table, one chunk of "
              f"{self.legendre_band} orders regenerated per call, "
              "2 x float64 plus 2 x float scratch of band x (T+1) x nlat")),
            ("spectral state", self.spectral_state_bytes,
             f"{LIVE_SPECTRAL_STACKS} live stacks x "
             f"{_spectral_stack_coefficients(self.nlev)} coefficients"),
            ("surface grid fields", self.surface_grid_bytes,
             f"{SURFACE_GRID_ARRAYS:g} x nlat x nlon"),
            ("levelled grid fields", self.resident_grid_bytes,
             f"{RESIDENT_GRID_ARRAYS + GRID_TRACER_ARRAYS:g} x nlev x nlat x "
             f"nlon ({GRID_TRACER_ARRAYS} grid tracers across two states "
             f"plus {RESIDENT_GRID_ARRAYS:g} physics), held between steps"),
            ("memoized syntheses", self.memo_grid_bytes,
             f"{SYNTHESIS_MEMO_GRID_ARRAYS} x nlev x nlat x nlon plus nlev + "
             f"{SYNTHESIS_MEMO_PLANES_BEYOND_LEVELS} planes, held between "
             "steps and released before every physics call (not at the peak)"),
            ("step working set", self.working_grid_bytes,
             f"{self.working_grid_arrays:.1f} x nlev x nlat x nlon, live at "
             f"the peak of one {self.backend} step"
             + ("" if self.backend != "cupy" else
                " (fitted to the allocator-measured peaks)")),
        ]
        if self.backend == "cupy":
            rows.append(
                ("radiation column workspace", self.physics_column_bytes,
                 f"{PHYSICS_COLUMN_ARRAYS:.0f} x nlev x "
                 f"min(radiation_column_chunk={self.radiation_column_chunk}, "
                 "nlat x nlon): the radiation chunk's per-g-point arrays "
                 "(fitted)"))
            rows.append(
                ("cumulus column workspace", self.cumulus_column_bytes,
                 f"{CUMULUS_COLUMN_ARRAYS:.0f} x nlev x "
                 f"min(cumulus_column_chunk={self.cumulus_column_chunk}, "
                 "nlat x nlon): the Grell-Freitas chunk's per-column arrays, "
                 "a constant from the chunk size up (fitted)"))
            rows.append(
                ("fixed", self.fixed_bytes,
                 "shape-independent bytes the fit leaves over (fitted)"))
        if self.semilag_gather_bytes:
            rows.append(
                ("semi-Lagrangian gather transient", self.semilag_gather_bytes,
                 f"{SEMILAG_GATHER_STACKS} stacks x min(gather_batch, "
                 f"{SEMILAG_GATHERED_FIELDS}) whole grid volumes, the "
                 "departure-point gather's own working set (counted, not "
                 "fitted; inside the one-band envelope above, added whole "
                 "to a banded peak because the band count does not divide it)"))
        if self.trajectory_bytes:
            rows.append(
                ("semi-Lagrangian trajectory state", self.trajectory_bytes,
                 f"{SEMILAG_TRAJECTORY_LEVELLED_ARRAYS} x nlev + "
                 f"{SEMILAG_TRAJECTORY_PLANES} planes of nlat x nlon, the "
                 "second time level the sl_si core holds between steps "
                 "(counted, not fitted)"))
        return tuple(rows)


def _radiation_column_chunk(cfg) -> int:
    options = getattr(cfg, "native_adapter_options", None) or {}
    value = options.get("radiation_column_chunk") if isinstance(options, dict) else None
    return int(value) if value is not None else _default_radiation_column_chunk()


def _cumulus_column_chunk(cfg) -> int:
    options = getattr(cfg, "native_adapter_options", None) or {}
    value = options.get("cumulus_column_chunk") if isinstance(options, dict) else None
    return int(value) if value is not None else _default_cumulus_column_chunk()


def estimate_global_memory(cfg) -> GlobalMemoryEstimate:
    """Price one :class:`~woof.globe.config.ArwenGlobalConfig`.

    Pure arithmetic on the config's own shapes.  Nothing is imported from
    CuPy, no device is touched and no array is built -- the whole point is
    to answer before the allocation that would fail.

    ``nlat``/``nlon`` come from :meth:`GaussianGrid.shape_for`, the same
    formulas the transform builds with, so an estimate can never describe
    a grid the run does not make.
    """

    nlat, nlon = GaussianGrid.shape_for(
        cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
        dealias_factor=cfg.dealias_factor)
    nlev = len(cfg.a_half_pa) - 1
    itemsize = 4 if cfg.precision == "float32" else 8
    complex_itemsize = 2 * itemsize
    coefficients = (int(cfg.truncation) + 1) ** 2
    points = nlat * nlon
    cupy = cfg.backend == "cupy"
    working_arrays = (CUPY_WORKING_GRID_ARRAYS if cupy
                      else NUMPY_WORKING_GRID_ARRAYS)
    chunk = _radiation_column_chunk(cfg)
    cumulus_chunk = _cumulus_column_chunk(cfg)
    # The host holds the construction transient in float64; on the numpy
    # backend the two built tables are host arrays underneath it.
    band = int(getattr(cfg, "legendre_band", DEFAULT_BAND))
    streaming = bool(getattr(cfg, "streaming", False))
    host_peak = host_transform_transient_bytes(
        cfg.truncation, nlat, band, streaming)
    if not cupy:
        host_peak += legendre_build_bytes(
            cfg.truncation, nlat, itemsize, band, streaming)
    return GlobalMemoryEstimate(
        name=cfg.name,
        backend=cfg.backend,
        precision=cfg.precision,
        truncation=int(cfg.truncation),
        nlat=nlat, nlon=nlon, nlev=nlev,
        dealias_factor=float(cfg.dealias_factor),
        float_itemsize=itemsize,
        complex_itemsize=complex_itemsize,
        legendre_table_bytes=legendre_runtime_bytes(
            cfg.truncation, nlat, itemsize, band, streaming),
        legendre_build_bytes=legendre_build_bytes(
            cfg.truncation, nlat, itemsize, band, streaming),
        spectral_state_bytes=math.ceil(
            LIVE_SPECTRAL_STACKS * _spectral_stack_coefficients(nlev)
            * coefficients * complex_itemsize),
        surface_grid_bytes=math.ceil(SURFACE_GRID_ARRAYS * points * itemsize),
        resident_grid_bytes=math.ceil(
            (RESIDENT_GRID_ARRAYS + GRID_TRACER_ARRAYS) * nlev * points * itemsize),
        # The memo is the model's own attribute (dynamics.py:183) and a
        # run with it off builds no SynthesisMemo at all (dynamics.py:230),
        # so its entries are not in that run's resident set.  It is not in
        # device_peak_bytes either way -- the model releases every entry
        # before each physics call -- so this changes what the resident
        # and host figures say, not what the door admits.
        legendre_band=band,
        streaming=streaming,
        memo_grid_bytes=(math.ceil(
            SYNTHESIS_MEMO_GRID_ARRAYS * nlev * points * itemsize
            + (nlev + SYNTHESIS_MEMO_PLANES_BEYOND_LEVELS) * points * itemsize)
            if bool(getattr(cfg, "synthesis_memo", True)) else 0),
        working_grid_bytes=math.ceil(working_arrays * nlev * points * itemsize),
        physics_column_bytes=(
            math.ceil(PHYSICS_COLUMN_ARRAYS * nlev * min(chunk, points) * itemsize)
            if cupy else 0),
        cumulus_column_bytes=(
            math.ceil(CUMULUS_COLUMN_ARRAYS * nlev * min(cumulus_chunk, points)
                      * itemsize)
            if cupy else 0),
        fixed_bytes=FIXED_DEVICE_BYTES if cupy else 0,
        host_transform_peak_bytes=host_peak,
        working_grid_arrays=working_arrays,
        radiation_column_chunk=chunk,
        cumulus_column_chunk=cumulus_chunk,
        extrapolation=extrapolation_notes(cfg, nlev) if cupy else (),
        # The semi-Lagrangian core's second time level, a structural
        # term (counted off semilag.state.TrajectoryState, not fitted):
        # the calibration rows were measured on the IMEX core, and the
        # default core carries this beside everything they measured.
        trajectory_bytes=(
            math.ceil((SEMILAG_TRAJECTORY_LEVELLED_ARRAYS * nlev
                       + SEMILAG_TRAJECTORY_PLANES) * points * itemsize)
            if str(getattr(cfg, "integrator", "")).lower() in SEMILAG_INTEGRATOR_NAMES
            else 0),
        semilag_gather_bytes=(
            math.ceil(SEMILAG_GATHER_STACKS
                      * min(int(getattr(getattr(cfg, "semilag", None), "gather_batch", 8) or 8),
                            SEMILAG_GATHERED_FIELDS)
                      * nlev * points * itemsize)
            if str(getattr(cfg, "integrator", "")).lower() in SEMILAG_INTEGRATOR_NAMES
            else 0),
    )


def worst_model_under_read() -> float:
    """The largest fraction by which the model has UNDER-read a measured
    run, in sample or held out; zero if it has never under-read one.

    ONLY THE UNDER-READS COUNT.  Over-predicting costs a wider band count
    or a parked slice; under-predicting costs the run, and it is the
    direction the model erred in before this lane's refit -- 3.5 percent
    at T383 and 10 to 11 at T533, every time in the admitting direction.
    """

    worst = 0.0
    for row in calibration_residuals():
        worst = max(worst, -float(row["residual_fraction"]))
    for row in held_out_residuals():
        if row["residual_fraction"] is not None:
            worst = max(worst, -float(row["residual_fraction"]))
    return worst


#: The margin the door's prediction is weighed with: one plus the largest
#: amount the model has ever under-read a MEASURED run by.
#:
#: SOLVED, NOT TYPED.  It used to be 1.15, chosen when the model
#: under-read T533 by 10 to 11 percent and nothing above T255 had been
#: measured.  Adding a measured row shrinks it, which is the point: the
#: door's inflation is exactly what the instrument's error justifies, and
#: a margin larger than that refuses runs that fit while a smaller one
#: admits runs that do not.
PREDICTION_MARGIN = 1.0 + worst_model_under_read()

#: A run takes at most three quarters of what the card has free, so a
#: card shared with another process still has room for it.
DEVICE_BUDGET_FRACTION = 0.75


# ---------------------------------------------------------------------------
# What the CARD holds, against what the POOL holds
# ---------------------------------------------------------------------------
#
# Every figure above this line is the CuPy pool's LIVE bytes, because that
# is the quantity the fit was measured on.  A card refuses a run on what
# the CARD holds, and the two differ by two terms this file used to leave
# out of the comparison entirely:
#
#   card held  =  live x fragmentation  +  out-of-pool
#
# FRAGMENTATION is the pool holding blocks it is not lending: it rounds
# every request up to a bin, keeps a freed block in that bin rather than
# returning it, and grows when no bin serves the request.
#
# The OUT-OF-POOL tax is everything on the card the pool never sees: the
# CUDA primary context, the module images, cuBLAS and cuFFT workspaces,
# and the driver's own bookkeeping.
#
# Charging neither is how a run gets admitted and dies: MEASURED
# 2026-09-06 at T533 L40 on a 32 GiB RTX 5090, the door printed 24.45 GiB
# against 30.90 GiB free, admitted the run, and the run reached 27.57 GiB
# LIVE.  Both terms are priced here, each from its own table.
#
# THE TWO TERMS ARE MEASURED AGAINST DIFFERENT THINGS, and charging a
# third would double count.  Fragmentation is pool-held over pool-live;
# the tax is card-held minus pool-held.  Card-held minus pool-LIVE is not
# a third term: it is the sum of these two, and adding it would charge
# the pool's rounding twice.

#: Pool held over pool live, MEASURED on the runs' own receipts
#: (``device_memory.peak_total_bytes`` over ``peak_used_bytes``).
#:
#: ``banded``, ``spilled`` and the radiation ``chunk`` separate the
#: populations, and every one of the three is there because a pair of
#: measurements forced it.
#:
#: Neither the band loop nor the tier moves the ratio one way: T383 L40
#: resident reads x1.1011 and the same shape with all three slices parked
#: at ONE band reads x1.3360, the hardest measured, while the same shape
#: with the tier and EIGHT bands reads x1.0296, the easiest.  A band loop
#: over a tier makes small, uniform allocations that the pool recycles; a
#: whole-globe step staging through a tier makes both kinds at once.
#:
#: The CHUNK moves it too, twice, in the same direction: T383 resident
#: reads x1.1011 at the default 12,500-column radiation chunk and x1.2020
#: at 5,000, and T383 at eight bands with the tier reads x1.0296 at the
#: default and x1.1626 at 5,000.  A smaller chunk takes a gigabyte and a
#: half off the LIVE peak while the pool still grows for the same
#: allocations, so the ratio rises on a run that is smaller.
#:
#: One planning value over all of them would be the largest, and charging
#: that to a banded spilled run at the default chunk refuses the T383
#: forecast this tree completed on a 16 GB card.
POOL_FRAGMENTATION_MEASURES = (
    {
        "label": "T533 L40, sixteen bands with the tier, the physics banded, RTX 5070 Ti",
        "truncation": 533,
        "bands": 16,
        "live_bytes": 9_593_687_552, "held_bytes": 13_573_784_064,
        "chunk": 12500, "banded": True, "spilled": True,
        "where": "the banded-physics lane's sixteen-band T533 run with all "
                 "three slices parked, the card to itself, 2026-09-07 "
                 "(8.935 GiB live, 12.642 held, x1.4149): the largest "
                 "banded-with-tier ratio this tree has read, and the "
                 "planning value of that class.  A band's physics half-step "
                 "allocates and frees its whole working set sixteen times a "
                 "half-step, and the tier stages 42 GiB a step through "
                 "buffers of a band's size; the pool keeps the bins.  On a "
                 "16 GB card at this shape that ratio is the difference "
                 "between a run that fits and one that does not",
    },
    {
        "label": "T383 L40, resident, BEFORE the transport fix",
        "truncation": 383,
        "bands": 1,
        "live_bytes": 16_707_477_504, "held_bytes": 20_637_876_224,
        "chunk": 12500, "banded": False, "spilled": False, "retired": True,
        "where": "the measure lane's profile of 2026-09-06 on the RTX 5090 "
                 "at tip ca57bff43 (15.560 GiB live, 19.220 held, "
                 "x1.2352).  RETIRED FROM THE PLANNING VALUE: that tree's "
                 "tracer transport allocated three whole ten-tracer grid "
                 "volumes a step, which is exactly the shape of request "
                 "that grows a pool, and the band lane removed it.  The "
                 "same shape on the tree that ships reads x1.1011 and "
                 "T255 reads x1.0871, so charging x1.2352 charges this "
                 "tree for a defect that was fixed, and it costs: at T255 "
                 "on a 16 GB card it parks all three slices for a run "
                 "whose card reading is 9.73 GiB of 15.28 free.  Kept as "
                 "the record of what the defect was worth",
    },
    {
        "label": "T383 L40, 12,500-column chunk, resident, the probe of record",
        "truncation": 383,
        "bands": 1,
        "live_bytes": 16_706_958_848, "held_bytes": 18_396_111_872,
        "chunk": 12500, "banded": False, "spilled": False,
        "where": "the same shape on the scale-out tree, RTX 5090, "
                 "2026-09-06 (15.559 GiB live, 17.132 held, x1.1011): the "
                 "live peak reproduces the measure lane's to 0.5 MB and "
                 "the pool holds 2.1 GiB less",
    },
    {
        "label": "T383 L40, resident, 5,000-column chunk, RTX 5090",
        "truncation": 383,
        "bands": 1,
        "live_bytes": 15_208_199_680, "held_bytes": 18_279_974_912,
        "chunk": 5000, "banded": False, "spilled": False,
        "where": "the ten-step probe of record, 2026-09-06 (14.163 GiB "
                 "live, 17.023 held, x1.2020): the largest resident ratio "
                 "this tree has read.  A smaller radiation chunk takes 1.4 "
                 "GiB off the live peak and leaves the pool holding nearly "
                 "as much, which is why the chunk is part of the class",
    },
    {
        "label": "T63 L40, resident, RTX 5070 Ti",
        "truncation": 63,
        "bands": 1,
        "live_bytes": 3_460_084_736, "held_bytes": 4_037_321_728,
        "chunk": 12500, "banded": False, "spilled": False,
        "where": "the BARE run of gate DOOR-1, 2026-09-06 (3.222 GiB live, "
                 "3.760 held, x1.1668): the largest resident ratio this "
                 "tree has read, and the planning value of that class.  "
                 "Its live peak is the calibration table's own T63 row to "
                 "the byte, measured on the other card on the previous "
                 "day, which is the fit's cross-card check at the small "
                 "end",
    },
    {
        "label": "T255 L40, resident, RTX 5070 Ti",
        "truncation": 255,
        "bands": 1,
        "live_bytes": 9_199_464_448, "held_bytes": 10_001_453_056,
        "chunk": 12500, "banded": False, "spilled": False,
        "where": "the ten-step probe of record with the card to itself, "
                 "2026-09-06 (8.568 GiB live, 9.315 held, x1.0871).  It is "
                 "also the fit's cross-card check: the model reads 8.599 "
                 "GiB for this shape, fitted on the OTHER card, 0.4 "
                 "percent above what this one measured",
    },
    {
        "label": "T383 L40, eight bands, nothing parked, RTX 5090",
        "truncation": 383,
        "bands": 8,
        "live_bytes": 16_282_101_248, "held_bytes": 17_413_178_368,
        "chunk": 12500, "banded": True, "spilled": False,
        "where": "the ten-step probe of record, 2026-09-06 (15.164 GiB "
                 "live, 16.217 held, x1.0695).  It is also the band "
                 "count's own relief measured at the probe of record: the "
                 "same shape resident reads 15.559 GiB live, so eight "
                 "bands take 0.396 GiB off a T383 peak",
    },
    {
        "label": "T255 L40, four latitude bands",
        "truncation": 255,
        "bands": 4,
        "live_bytes": 7_479_777_280, "held_bytes": 8_954_919_936,
        "chunk": 12500, "banded": True, "spilled": False,
        "where": "the band lane's step-time leg, RTX 5090, 2026-09-06 "
                 "(x1.1972): a band loop makes more, smaller allocations "
                 "and the pool holds more of them",
    },
    {
        "label": "T533 L40, eight latitude bands",
        "truncation": 533,
        "bands": 8,
        "live_bytes": 27_316_350_976, "held_bytes": 29_954_767_872,
        "chunk": 12500, "banded": True, "spilled": False,
        "where": "the band lane's capacity leg, RTX 5090, 2026-09-06 "
                 "(x1.0966) -- the LOWEST ratio measured, and the shape "
                 "that matters most, so the planning value is not read "
                 "off it",
    },
    {
        "label": "T383 L40, eight bands, with the host tier on",
        "truncation": 383,
        "bands": 8,
        "live_bytes": 12_470_658_560, "held_bytes": 12_839_973_888,
        "chunk": 12500, "banded": True, "spilled": True,
        "where": "the RTX 5070 Ti with the card to itself, 2026-09-06, "
                 "the ten-step probe of record -- the 16 GB row of the "
                 "design's capacity table, measured at the peak a real "
                 "forecast reaches (11.614 GiB live, 11.958 held, "
                 "x1.0296, and nvidia-smi read 12,678 MiB).  A band loop "
                 "over a tier is the TIGHTEST pattern measured: the tier "
                 "removes the persistent blocks and the band loop's "
                 "allocations are small and uniform, so the pool recycles "
                 "them",
    },
    {
        "label": "T383 L40, eight bands with the tier, 5,000-column chunk",
        "truncation": 383,
        "bands": 8,
        "live_bytes": 10_744_860_160, "held_bytes": 12_491_547_648,
        "chunk": 5000, "banded": True, "spilled": True,
        "where": "the RTX 5070 Ti with the card to itself, 2026-09-06, the "
                 "ten-step probe of record (10.007 GiB live, 11.634 held, "
                 "x1.1626): the largest of the three banded-and-spilled "
                 "rows measured that day (the class's planning value at "
                 "T533 and above is the T533 rows', x1.40 to x1.50).  A smaller "
                 "radiation chunk takes 1.6 GiB off the live peak and "
                 "leaves the pool holding nearly as much, so the ratio "
                 "rises even though the run is smaller -- which is why "
                 "the class's planning value is not read off the first "
                 "run of it that was measured",
    },
    {
        "label": "T383 L40, eight bands with the tier, RTX 5090",
        "truncation": 383,
        "bands": 8,
        "live_bytes": 12_470_658_560, "held_bytes": 13_403_030_528,
        "chunk": 12500, "banded": True, "spilled": True,
        "where": "the ten-step probe of record, 2026-09-06 (11.614 GiB "
                 "live, 12.483 held, x1.0748), and the largest of its "
                 "class.  Its live peak is the RTX 5070 Ti's for the same "
                 "shape and plan TO THE BYTE, on the other card",
    },
    {
        "label": "T383 L40, sixteen bands with the tier",
        "truncation": 383,
        "bands": 16,
        "live_bytes": 12_470_658_560, "held_bytes": 12_946_142_208,
        "chunk": 12500, "banded": True, "spilled": True,
        "where": "the RTX 5070 Ti with the card to itself, 2026-09-06 "
                 "(x1.0381).  Its live peak is the eight-band run's to "
                 "the byte, which is the band lane's finding in one line: "
                 "the physics half-step is not banded and owns the peak",
    },
    {
        "label": "T383 L40, sixteen bands with the tier, the physics banded, RTX 5070 Ti",
        "truncation": 383,
        "bands": 16,
        "live_bytes": 10_446_974_464, "held_bytes": 12_746_315_776,
        "chunk": 12500, "banded": True, "spilled": True,
        "where": "the cards lane's probe of the T383 sixteen-band plan "
                 "with all three slices parked on the tree with the "
                 "physics suite banded, the RTX 5070 Ti to itself, "
                 "2026-09-07 (9.730 GiB live, 11.871 held, x1.2201; "
                 "the 16 GB host the probe_t383_16b_tier leg).  The door had "
                 "priced this plan at 18.30 GiB of card by charging it "
                 "the T533 sixteen-band row's x1.4149 on the two-band "
                 "day's 12.41 GiB live, and refused rank 1 of a two-card "
                 "T383 run on a card that holds the plan with 3.4 GiB "
                 "to spare.  An exact-shape row is the plan's own "
                 "evidence (pool_fragmentation_for)",
    },
    {
        "label": "T383 L40, TWO bands with the tier, the bare run",
        "truncation": 383,
        "bands": 2,
        "live_bytes": 12_470_658_560, "held_bytes": 13_564_384_256,
        "chunk": 12500, "banded": True, "spilled": True,
        "where": "the RTX 5070 Ti with the card to itself, 2026-09-06, the "
                 "ten-step probe of record with NO flag set -- the plan a "
                 "bare T383 forecast takes on a 16 GB card (11.614 GiB "
                 "live, 12.633 held, x1.0877, and nvidia-smi read 13,370 "
                 "MiB).  It is the LARGEST of its class and therefore the "
                 "planning value, and it is here because leaving it out "
                 "set that class from three flagged runs while the one run "
                 "the sizer actually chooses fragmented harder than any of "
                 "them: at two bands the band loop's allocations are "
                 "coarser than at eight, so the pool recycles fewer of "
                 "them",
    },
    {
        "label": "T533 L40, resident with the tier, the bare ten-step probe, RTX 5090",
        "truncation": 533,
        "bands": 1,
        "live_bytes": 24_940_278_784, "held_bytes": 30_183_100_928,
        "chunk": 5000, "banded": False, "spilled": True,
        "where": "the banded-physics tree's bare runs of the shipped 25 km "
                 "config on the RTX 5090, 2026-09-07, the card to itself, the "
                 "sizer choosing one band with all three slices parked at "
                 "30.90 GiB free: the ten-step probe 23.23 GiB live and "
                 "28.11 held (x1.2102), the 24 h day 23.23 live and "
                 "28.11 held (x1.2102); the larger is the row.  The tree "
                 "before it read x1.1850 on the same day (21.12 live, 25.03 "
                 "held).  The class had no row at the reduced chunk and "
                 "borrowed the largest of every class (x1.4994), which "
                 "refused this plan on the card it completed on",
    },
    {
        "label": "T383 L40, two bands with the tier, the bare day, RTX 5070 Ti",
        "truncation": 383,
        "bands": 2,
        "live_bytes": 13_322_662_912, "held_bytes": 15_300_097_536,
        "chunk": 12500, "banded": True, "spilled": True,
        "where": "the bare 24 h day of the 34.7 km config on the 16 GB "
                 "card, 2026-09-07 (the 16 GB host, the 24 h day run): the "
                 "sizer chose two bands with all three slices parked at "
                 "15.28 GiB free, 288 steps at 300 s, 12.41 GiB live and "
                 "14.25 held, x1.1484, status pass, 0.29 GiB of the card "
                 "to spare.  Above the two-band probe's x1.0877 and so the "
                 "planning value of the class at T383 up to two bands",
    },
    {
        "label": "T383 L40, resident, with the host tier on",
        "truncation": 383,
        "bands": 1,
        "live_bytes": 12_895_516_160, "held_bytes": 17_228_260_352,
        "chunk": 12500, "banded": False, "spilled": True,
        "where": "the same shape and card as the probe of record the same "
                 "afternoon, with host_spill parking all three slices "
                 "(12.01 GiB live, 16.04 held, x1.3360): the spilled "
                 "population's planning value",
    },
    {
        "label": "T533 L40, two bands with the tier, the physics banded, the bare run, RTX 5090",
        "truncation": 533,
        "bands": 2,
        "live_bytes": 21_152_070_656, "held_bytes": 31_715_355_136,
        "chunk": 5000, "banded": True, "spilled": True,
        "where": "the banded-physics lane's bare T533 run from the door on "
                 "the RTX 5090, 2026-09-07, the shipped 25 km config cut to "
                 "ten steps (the 32 GB host, capacity probe "
                 "t533_bare.shared3): with 28.26 GiB free the sizer chose "
                 "two bands with all three slices parked and the run "
                 "completed at 19.70 GiB live and 29.54 held, x1.4994; two "
                 "more bare runs of the same plan the same hour read "
                 "x1.4037 and x1.4231 (29.69 and 25.74 GiB held at the same "
                 "19.70 live).  Another process computed on the card during "
                 "all three, which refuses their timing and not their "
                 "allocation: the pool's held bytes are this process's own. "
                 "The largest banded-with-tier ratio at the reduced chunk "
                 "and the planning value of that class at T533 and above: "
                 "priced at the T383 row's x1.1626 the door admitted a plan "
                 "at 25.99 GiB that then held 29.5 of the card's 32.6",
    },
    {
        "label": "T533 L40, thirty-two bands with the tier, the physics banded, the bare run, RTX 5070 Ti",
        "truncation": 533,
        "bands": 32,
        "live_bytes": 9_513_141_760, "held_bytes": 13_534_945_792,
        "chunk": 12500, "banded": True, "spilled": True,
        "where": "the banded-physics lane's bare T533 run from the door on "
                 "the 16 GB card, 2026-09-07, the card to itself (the 16 GB "
                 "host, the bare T533 IMEX capacity leg): the sizer chose "
                 "32 bands with all three slices parked, 8.86 GiB live and "
                 "12.61 held, x1.4228, 10.65 s a step.  The row of record "
                 "for a count above sixteen, a shade above the sixteen-band "
                 "probe's x1.4149 as the count rule expects",
    },)

#: A MEASURED ROW OF THIS TABLE'S CLASSES THAT IS NOT ITS PLANNING VALUE,
#: and the open question it raises.
#:
#: MEASURED 2026-09-06, RTX 5090, T255 L40 at eight bands with all three
#: slices parked, the ten-step probe of record: 7,317,473,280 B live and
#: 8,308,735,488 held, **x1.13547**, against a "banded with the host tier
#: at the default radiation chunk" class whose largest T383 row is
#: x1.0877.  The same shape and plan on the RTX 5070 Ti read the same
#: 7,317,473,280 B live TO THE BYTE and 8,217,965,568 held, x1.12306, so
#: it is two cards and not one card's accident.  The gate still held on
#: both legs: the door asked 8.73 GiB and nvidia-smi read 8.46 and 8.07
#: GiB against the two processes, because the model's margin and the flat
#: out-of-pool tax cover more than the class's own spread.
#:
#: AND THE SAME PLAN ON TWO CARDS SAYS WHY, which is the finding that
#: matters more than the row.  MEASURED 2026-09-06, T383 L40 at two bands
#: with all three slices parked, which is the plan a bare 34.7 km forecast
#: takes on a 16 GB card: **12,470,658,560 B of pool live on BOTH cards,
#: to the byte**, and 13,564,384,256 B held on the RTX 5070 Ti against
#: 14,826,533,888 B on the RTX 5090.  x1.0877 and x1.18891 for the same
#: plan and the same live bytes.  nvidia-smi read 13,370 MiB against the
#: one process and 14,892 MiB against the other.
#:
#: So fragmentation is not a property of the plan alone: **the pool grows
#: into the room it has**, and a card with twice the room lets it take
#: 1.5 GiB more for the same work.  That has a consequence in each
#: direction, both MEASURED.  Charging the roomy card's x1.18891 to the
#: tight one prices that plan at 15.60 GiB of card against 15.28 GiB free
#: and REFUSES the forecast the 16 GB card completed at 13.06.  Charging
#: the tight card's ratio to the roomy one is what this table did before
#: x1.0877 was added, and it under-read that run's card by 0.117 GiB
#: (14.426 asked, 14.543 held), which is gate MEM-1 failing.  At x1.0877
#: the same leg clears by 0.047 GiB, which is a pass and is not a margin.
#: The term needs an axis for the free VRAM the pool is growing into, and
#: that is a fitted term over more than one pair.  Named here for whoever
#: fits it; the two legs it needs are the same plan measured on the
#: roomy card and on the tight one.
#:
#: It is NOT the planning value, and the reason is the same reason this
#: table has classes at all.  The four T383 rows of that class read
#: x1.0296 to x1.0877; charging a T383 plan the ratio measured on a T255
#: run refuses the bare 34.7 km forecast this tree has completed on a 16
#: GB card whenever that card carries about a gigabyte of anything else
#: (MEASURED: the plan's card figure moves from 14.43 to 15.20 GiB, and
#: the refusal threshold from 14.43 to 14.93 GiB free).  The ratio rises
#: as the run gets smaller, and there are now two pairs saying so -- this
#: one, and the resident class where the smallest shape measured (T63,
#: x1.1668) is the largest row.  Two pairs is the standard the radiation
#: chunk's own axis was added on, so the class may need a size axis; that
#: is a fitted term and it needs more than two pairs, so it is named for
#: the coordinator rather than guessed here.

#: Card held minus pool held, MEASURED: the bytes on the card that the
#: pool never sees -- the CUDA primary context, the module images, and
#: the cuBLAS and cuFFT workspaces.
OUT_OF_POOL_MEASURES = (
    {"label": "T63 L40", "tax_bytes": int(0.42 * GIB)},
    {"label": "T127 L40", "tax_bytes": int(0.44 * GIB)},
    {"label": "T255 L40", "tax_bytes": int(0.37 * GIB)},
    {"label": "T383 L40", "tax_bytes": int(0.73 * GIB)},
    {
        "label": "T383 L40, eight bands with the tier, 5,000-column chunk",
        "tax_bytes": 450_534_400,
        "where": "the RTX 5070 Ti with the card to itself, 2026-09-06: "
                 "nvidia-smi read 12,344 MiB and the pool held "
                 "12,491,547,648 B (0.4196 GiB outside it)",
    },
    {
        "label": "T383 L40, eight bands with the tier, the probe of record",
        "tax_bytes": 452_398_080,
        "where": "the RTX 5070 Ti with the card to itself, 2026-09-06: "
                 "nvidia-smi read 12,678 MiB against the process and the "
                 "pool held 12,839,973,888 B, so 0.4213 GiB of the card "
                 "was never the pool's.  It is the smallest of the five "
                 "and does not move the planning value; it is here "
                 "because a term charged flat needs its spread on record",
    },
    {
        "label": "T383 L40, eight bands, nothing parked",
        "tax_bytes": 798_489_600,
        "where": "RTX 5090, 2026-09-06: nvidia-smi read 17,368 MiB and the "
                 "pool held 17,413,178,368 B, so 0.7436 GiB was never the "
                 "pool's.  It is the largest of the seven and the planning "
                 "value",
    },
    {
        "label": "T383 L40, eight bands with the tier, RTX 5090",
        "tax_bytes": 782_105_600,
        "where": "RTX 5090, 2026-09-06: 13,528 MiB read against "
                 "13,403,030,528 B held, 0.7284 GiB outside the pool",
    },
    {
        "label": "T383 L40, resident, 5,000-column chunk",
        "tax_bytes": 793_534_464,
        "where": "RTX 5090, 2026-09-06: nvidia-smi read 18,188 MiB and the "
                 "pool held 18,279,974,912 B, so 0.7390 GiB of the card "
                 "was never the pool's.  Two independent T383 resident "
                 "runs at two radiation chunks read 0.7384 and 0.7390, so "
                 "the term does not move with the chunk either",
    },
    {
        "label": "T383 L40, the probe of record",
        "tax_bytes": 792_828_928,
        "where": "RTX 5090, 2026-09-06, the ten-step probe of record at "
                 "one band with the tier off: nvidia-smi read 18,300 MiB "
                 "against this process at its largest and the pool held "
                 "18,396,111,872 B at its own peak, so 0.7384 GiB of the "
                 "card was never the pool's.  The two maxima are taken "
                 "over the same run and need not fall in the same "
                 "instant; they fall in the same phase",
    },
)

#: The planning value: the largest measured out-of-pool tax.
#:
#: It does NOT scale with the shape -- the measurements span a sixteenfold
#: grid and move by a third of a gigabyte -- so it is charged flat.  A
#: shape-scaled form would be a curve drawn through points that do not
#: trend.  Solved from the rows rather than typed.
OUT_OF_POOL_BYTES = max(int(row["tax_bytes"]) for row in OUT_OF_POOL_MEASURES)


def _reduced_chunk(chunk) -> bool:
    """Is this radiation chunk below the shipped default?"""

    return int(chunk or 0) < _default_radiation_column_chunk()


def _fragmentation_class(row) -> tuple[bool, bool, bool]:
    return (bool(row.get("banded", False)), bool(row.get("spilled", False)),
            _reduced_chunk(row.get("chunk")))


def pool_fragmentation_for(spilled: bool = False, banded: bool = False,
                           reduced_chunk: bool = False,
                           bands: int | None = None,
                           truncation: int | None = None) -> tuple[float, str]:
    """``(ratio, evidence)``: the largest MEASURED pool-held-over-live
    ratio for this plan's class, and one phrase saying which rows it came
    from.

    THE CLASS IS THE ALLOCATION PATTERN, and the measurements say it has
    to be.  MEASURED 2026-09-06 across twelve runs of this tree: at the
    default radiation chunk a resident run reads x1.0871 to x1.1668, a
    banded run x1.0695 to x1.1972, a resident run with the pinned tier on
    x1.3360, and a banded run with the tier on x1.0296 to x1.0748; a
    5,000-column radiation chunk moves the resident figure to x1.2020 and
    the banded-with-tier figure to x1.1626, which is why the chunk is part
    of the class.  One planning value over all of them would be the
    largest, and charging x1.3360 to the banded spilled run that
    MEASURED x1.0296 refuses a T383 forecast that this tree completed on
    a 16 GB card with 3.5 GiB to spare.

    The largest of the class, because the term exists to stop a run being
    admitted that the card cannot hold; the middle of a distribution
    would admit half the runs that fail.  A class with no measured row
    falls back to the widest population that has one, and SAYS SO in the
    phrase it returns, so a figure never quietly borrows another class's
    evidence.

    THE BAND COUNT IS PART OF THE PATTERN TOO.  MEASURED 2026-09-07 on
    an RTX 5070 Ti with the physics suite banded: T533 L40 at sixteen
    bands with the tier reads x1.4149, where T383 at eight bands with the
    tier read x1.0296 the day before -- sixteen half-steps a step, each
    allocating and freeing a band's whole working set, leave the pool
    holding bins the eight-band run never opened.  Charging x1.4149 to
    the eight-band T383 run refuses a forecast this tree completed on a
    16 GB card (11.6 GiB live, 12.4 held); charging x1.0296 to the
    sixteen-band T533 run admits one the card holds by 2.9 GiB.  So with
    ``truncation`` given, the rows of the class at that truncation OR
    LARGER are the evidence (a larger grid fragments more, so a smaller
    shape's row would under-charge), and with ``bands`` given, of those
    the rows at that band count OR FEWER (fragmentation grows with the
    count, so a row at fewer bands never over-charges); the largest of
    them is the value.  A shape or count beyond every measured row keeps
    the wider set, and the phrase says which rows spoke.
    """

    want = (bool(banded), bool(spilled), bool(reduced_chunk))
    names = {(False, False): "resident, nothing parked",
             (True, False): "banded, nothing parked",
             (False, True): "resident with the host tier",
             (True, True): "banded with the host tier"}
    name = names[(bool(banded), bool(spilled))] + (
        " at a reduced radiation chunk" if reduced_chunk
        else " at the default radiation chunk")
    # A RETIRED ROW IS HISTORY, NOT A PLANNING VALUE.  It measured a tree
    # whose defect has been fixed, and charging this tree for it refuses
    # runs this tree completes.
    live = [row for row in POOL_FRAGMENTATION_MEASURES
            if not row.get("retired")]
    rows = [row for row in live if _fragmentation_class(row) == want]
    # A ROW AT EXACTLY THIS SHAPE AND BAND COUNT IS THE PLAN'S OWN
    # EVIDENCE, and outranks the rows a larger truncation lends it.  The
    # larger-or-equal rule below exists for a shape with no measurement
    # of its own; applied to one that has one it charges the plan for a
    # grid it is not.  MEASURED 2026-09-07 on the RTX 5070 Ti: T383 L40
    # at sixteen bands with every slice parked, the physics banded, reads
    # x1.2201 (9.73 GiB live, 11.87 held) and completes with 3.4 GiB of
    # the card to spare, while the T533 sixteen-band row's x1.4149 priced
    # it at 18.30 GiB of card and the door refused rank 1 of a two-card
    # T383 run for it.
    if rows and truncation is not None and bands is not None:
        exact = [row for row in rows
                 if int(row.get("truncation", 0) or 0) == int(truncation)
                 and int(row.get("bands", 1) or 1) == int(bands)]
        if exact:
            ratio = max(float(r["held_bytes"]) / float(r["live_bytes"])
                        for r in exact)
            return ratio, (f"the largest of {len(exact)} measured {name} run"
                           + ("" if len(exact) == 1 else "s")
                           + f" at exactly T{int(truncation)} and "
                           f"{int(bands)} band" + ("" if int(bands) == 1 else "s"))
    if rows and truncation is not None:
        # The rows at this truncation or larger; a shape beyond every
        # measured row keeps the whole class.
        larger = [row for row in rows
                  if int(row.get("truncation", 0) or 0) >= int(truncation)]
        if larger:
            rows = larger
            name += f" at T{int(truncation)} or larger"
    if rows and bands is not None and int(bands) > 1:
        # The rows at this band count or fewer; a count beyond every
        # measured row keeps the set above.
        within = [row for row in rows
                  if int(row.get("bands", 1) or 1) <= int(bands)]
        if within:
            rows = within
            name += f" at {int(bands)} bands or fewer"
    if rows:
        ratio = max(float(r["held_bytes"]) / float(r["live_bytes"]) for r in rows)
        return ratio, (f"the largest of {len(rows)} measured "
                       f"{name} run" + ("" if len(rows) == 1 else "s"))
    if not live:
        raise ValueError(
            "no measured fragmentation row at all: a planning value with "
            "nothing under it is a typed constant wearing a measurement's "
            "name")
    ratio = max(float(r["held_bytes"]) / float(r["live_bytes"]) for r in live)
    return ratio, (f"no measured {name} run, so the largest of all "
                   f"{len(live)} measured runs")


#: The resident population's planning value, and the one a figure printed
#: without a plan beside it is charged at.
POOL_FRAGMENTATION = pool_fragmentation_for()[0]


def measured_fragmentation_range() -> tuple[float, float]:
    """The smallest and largest MEASURED ratio over every row."""

    ratios = [float(row["held_bytes"]) / float(row["live_bytes"])
              for row in POOL_FRAGMENTATION_MEASURES
              if not row.get("retired")]
    return min(ratios), max(ratios)


def card_required_bytes(live_bytes: int, *, spilled: bool = False,
                        banded: bool = False, reduced_chunk: bool = False,
                        fragmentation: float | None = None,
                        out_of_pool: int | None = None,
                        margin: float | None = None,
                        bands: int | None = None,
                        truncation: int | None = None) -> int:
    """What a card must have FREE for a run whose pool live peak is
    ``live_bytes``.

    ``live x fragmentation + live x (margin - 1) + out-of-pool``.  This is
    the quantity a refusal is decided on.

    THE MARGIN IS ADDED ON THE LIVE FIGURE AND NOTHING ELSE, because that
    is the only term it is the error bar of: the model predicts the pool's
    live peak, and the fragmentation and the out-of-pool tax are separate
    measurements each already taken at the largest of its class.  Charging
    the model's residual against those too refused a T383 forecast this
    tree completed on a 16 GB card, by 0.7 percent (2026-09-06); and the
    multiplicative form ``live x margin x fragmentation`` this replaced
    charged it against the fragmentation still, and refused a T533
    forecast this tree completed on the same card by 0.3 percent
    (MEASURED 2026-09-07, RTX 5070 Ti: sixteen bands with every slice
    parked, 8.935 GiB live, 12.642 held, 15.51 GiB free, and the door
    read 15.32 to 15.75 GiB of card required).  The fragmentation ratio is
    held over live as MEASURED, so it multiplies the live figure the
    pool actually held; the margin is the live prediction's error bar and
    is worth its own bytes once.

    ``spilled``, ``banded`` and ``reduced_chunk`` name the plan's
    allocation pattern, which is what the fragmentation is measured per
    (:func:`pool_fragmentation_for`).  Zero live bytes is zero card bytes:
    the numpy backend opens no context, so it pays neither term.
    """

    live = int(live_bytes)
    if live <= 0:
        return 0
    if fragmentation is None:
        fragmentation = pool_fragmentation_for(
            bool(spilled), bool(banded), bool(reduced_chunk), bands=bands,
            truncation=truncation)[0]
    tax = OUT_OF_POOL_BYTES if out_of_pool is None else int(out_of_pool)
    weight = PREDICTION_MARGIN if margin is None else float(margin)
    return int(math.ceil(live * float(fragmentation) + live * (weight - 1.0)) + tax)

def estimate_card_required_bytes(estimate, *, spilled: bool = False,
                                 banded: bool = False) -> int:
    """:func:`card_required_bytes` for a whole estimate, charged at ITS
    radiation chunk's fragmentation class.

    Every caller that prices an estimate rather than a plan goes through
    here, so the chunk cannot be dropped on one route and charged on
    another: the door, the gate list, the refusal, the truncation
    bisection and ``woof check`` all read the same class.
    """

    return card_required_bytes(
        estimate.device_peak_bytes, spilled=spilled, banded=banded,
        reduced_chunk=_reduced_chunk(estimate.radiation_column_chunk))


#: The band-count ladder the auto-sizer chooses from.  Powers of two up
#: to 32: the design's capacity table is priced at 1, 4, 8, 16 and 32, and
#: every row of it is far above the four-row band floor (T533 at B = 8 is
#: 100 rows, T799 at B = 32 is 37).
LATITUDE_BAND_LADDER = (1, 2, 4, 8, 16, 32)

#: MEASURED pairs of a run's whole-run live peak at one band and at
#: ``bands`` bands, on the tree whose physics suite runs a band at a time
#: (dynamics.apply_physics), each with the estimate's own per-column
#: working term at that shape (``working_grid_bytes``, the FITTED term
#: that scales with the grid: the step's transient working set and
#: whatever of the persistent grid state the fit folded into it).  What
#: the band count divides is a fraction of that term and nothing else --
#: the tables, the spectral state, the persistent grid state and the
#: fixed terms are held whole whatever the schedule -- so the pair
#: measures the fraction of the working term that does NOT divide:
#:
#:   ``peak(B) = peak(1) - W (1 - f) (1 - 1/B)``, solved for ``f``.
#:
#: The planning value is the LARGEST ``f`` measured: a larger ``f``
#: predicts a larger banded peak, so an error costs a wider band count or
#: a parked slice rather than a run admitted at a peak the card cannot
#: hold, which is the direction the design's R2 names as the defect.
#:
#: The RETIRED form ``peak(B) = peak(1) (r + (1 - r) / B)`` with r =
#: 0.9838 was measured on the tree whose physics half-step was NOT banded
#: (2026-09-06: T255 L40 6.553 GiB at one band against 6.460 at eight on
#: an RTX 5070 Ti, T383 14.164 against 13.768 on an RTX 5090), where the
#: band count moved the whole-run peak by a fixed step and no fraction of
#: anything described it.  On this tree the physics divides and the pair
#: below reads it: 8.805 GiB at one band against 7.855 at eight.
BANDED_PEAK_MEASURES = (
    {
        "label": "T255 L40 native, semi-Lagrangian core, GDAS start, one band against eight, RTX 5070 Ti",
        "truncation": 255,
        "resident_bytes": 9_453_929_984, "banded_bytes": 8_434_387_456,
        "parked_bytes": 0,
        "bands": 8, "working_grid_bytes": 4_254_883_205,
        "where": "the banded-physics lane's ten-step T255 gate (the 16 GB "
                 "host, the semi-Lagrangian core at one band and at eight, "
                 "RTX 5070 Ti, 2026-09-07, tip 5a2979d18): the runs' own receipts, "
                 "device_memory.peak_used_bytes, both legs with the card "
                 "otherwise idle; the working term is the estimate's at "
                 "that config (the T255 semi-Lagrangian gate config)",
    },
    {
        "label": "T533 L40 native, IMEX core, one band against sixteen with every slice parked, RTX 5070 Ti",
        "truncation": 533,
        "resident_bytes": 28_137_857_536, "banded_bytes": 9_593_687_552,
        "parked_bytes": 4_388_300_000,
        "bands": 16, "working_grid_bytes": 18_513_572_311,
        "where": "the resident figure is the band lane's ten-step T533 leg "
                 "at one band (MEASURED_RUNS, RTX 5090, 2026-09-06); the "
                 "banded figure the banded-physics lane's sixteen-band run "
                 "of the same config with all three slices in the pinned "
                 "host tier (4.3883 GiB parked), the card to itself, RTX "
                 "5070 Ti, 2026-09-07, tip bbe2d91de (the 16 GB host, the "
                 "sixteen-band T533 IMEX capacity leg): 8.935 GiB live, "
                 "12.642 held, 9.32 s a step.  The parked bytes are added "
                 "back before the fraction is solved, so the pair measures "
                 "the band count alone",
    },
)


def _banded_fraction_of(row) -> float:
    unspilled = float(row["banded_bytes"]) + float(row.get("parked_bytes", 0))
    return 1.0 - (float(row["resident_bytes"]) - unspilled) / (
        float(row["working_grid_bytes"]) * (1.0 - 1.0 / int(row["bands"])))


def banded_working_resident_fraction(truncation: int | None = None) -> float:
    """The MEASURED fraction of the working term the band count does not
    divide, for a shape at ``truncation`` (:data:`BANDED_PEAK_MEASURES`).

    THE FRACTION IS THE SHAPE'S, NOT ONE NUMBER.  MEASURED: at T255 the
    band count divides 27 percent of the working term (the peak there is
    the tables, the spectral state and the run's own floors as much as
    the step) and at T533 it divides 88 percent (the peak is the step).
    One planning value over both would either refuse T533 on a 16 GB card
    that this tree completed it on, or admit a T255 run at a peak it does
    not reach.  So the value is read at the shape: the measured row at
    the truncation, or the LARGER fraction of the two measured rows the
    truncation lies between (the pessimistic end, a larger fraction
    predicting a larger banded peak), or the nearest measured row's
    beyond either end.  With no truncation given, the largest of all.
    """

    rows = sorted(BANDED_PEAK_MEASURES, key=lambda row: int(row["truncation"]))
    if truncation is None:
        return max(_banded_fraction_of(row) for row in rows)
    t = int(truncation)
    below = [row for row in rows if int(row["truncation"]) <= t]
    above = [row for row in rows if int(row["truncation"]) >= t]
    chosen = []
    if below:
        chosen.append(below[-1])
    if above:
        chosen.append(above[0])
    return max(_banded_fraction_of(row) for row in chosen)


#: The planning value with no shape named: the largest measured
#: non-dividing fraction of the working term (the pessimistic end).
BANDED_WORKING_RESIDENT_FRACTION = banded_working_resident_fraction()

#: The pinned host tier's slices, coldest first: the order the
#: minimum-spill rule sheds them in (spill.SLICES, and the reads-per-step
#: count that makes it the cold order lives there).
SPILL_SLICES = ("physics", "surface", "tracers")


#: The [memory] host_spill values, mirrored from config so the sizer can
#: name them in a refusal without importing the config module.
HOST_SPILL_MODES = ("auto", "on", "off")


#: What one parked byte takes off the CARD, MEASURED.
#:
#: A slice in the pinned host tier removes MORE than its own bytes,
#: because the device copy the physics suite makes of the scheme
#: namespace stops existing beside it.  Three readings, two shapes, two
#: cards, all 2026-09-06:
#:
#:   ``resident_bytes`` the live peak with the slice on the card,
#:   ``spilled_bytes``  the live peak with it parked,
#:   ``parked_bytes``   what the tier holds.
#:
#: The planning value is the SMALLEST of them, the mirror of the
#: fragmentation rule: crediting more relief than the smallest measured
#: would admit a run the card cannot hold, and crediting only the parked
#: bytes -- which is what this modelled before the T383 row was measured
#: -- refuses one it can, by 0.24 GiB on the 16 GB card at T383.
SPILL_RELIEF_MEASURES = (
    {
        "label": "T255 L40 native, one band, RTX 5090",
        "resident_bytes": 4_862_003_609, "spilled_bytes": 3_047_502_479,
        "parked_bytes": 1_083_231_764,
        "where": "the spill lane's section-peak probe, 2026-09-06 (4.528 "
                 "GiB live down to 2.838 against 1.009 parked, x1.675)",
    },
    {
        "label": "T255 L40 native, eight bands, RTX 5090",
        "resident_bytes": 4_636_506_849, "spilled_bytes": 2_858_512_609,
        "parked_bytes": 1_083_231_764,
        "where": "the same probe at eight bands, 2026-09-06 (4.318 GiB "
                 "down to 2.662, x1.641)",
    },
    {
        "label": "T383 L40 native, eight bands, the probe of record, RTX 5090",
        "resident_bytes": 16_282_101_248, "spilled_bytes": 12_470_658_560,
        "parked_bytes": 2_436_312_924,
        "where": "the pair that needs no correction: BOTH arms are eight "
                 "bands at the probe of record on the same card the same "
                 "afternoon, 15.164 GiB with the tier off and 11.614 with "
                 "all three slices parked, so the band loop is not "
                 "credited to the tier by construction rather than by "
                 "subtraction (x1.5644, the smallest of the three and the "
                 "planning value)",
    },
)


def spill_relief_factor() -> float:
    """The smallest MEASURED relief per parked byte.

    Smallest, because this term is subtracted: the fragmentation is
    multiplied and takes the largest of its class, and both rules point
    the same way -- toward the figure that refuses rather than admits.
    """

    return min(
        (float(row["resident_bytes"]) - float(row["spilled_bytes"]))
        / float(row["parked_bytes"])
        for row in SPILL_RELIEF_MEASURES)


#: The planning value: ONE, a parked byte off the card and no more.  The
#: measured x1.56 to x1.68 above belongs to the tree whose physics suite
#: copied the WHOLE namespace on every call; with the suite banded
#: (dynamics.apply_physics) that copy is a band's, so the relief the
#: measurements credited beyond the parked bytes no longer exists to be
#: credited, and crediting it would admit a run the card cannot hold.
#: :func:`spill_relief_factor` keeps the record of what was measured.
SPILL_RELIEF = 1.0


def banded_device_peak_bytes(estimate: GlobalMemoryEstimate, bands: int,
                             *, spilled_bytes: int = 0) -> int:
    """The device peak of ``estimate`` at ``bands`` bands with
    ``spilled_bytes`` of the persistent grid state in the pinned host tier.

    ``peak(B) = peak(1) - W (1 - f) (1 - 1/B) - spilled``: the resident
    peak, less the part of the per-column working term that follows the
    band count (:data:`BANDED_PEAK_MEASURES`, ``f`` the measured fraction
    that does not), less what the card stops holding because the host
    holds it.

    THE SPILL RELIEF IS ONE BYTE PER PARKED BYTE.  On the tree whose
    physics suite ran whole, a parked slice took x1.5644 off the card
    (:data:`SPILL_RELIEF_MEASURES`) because the device copy the suite made
    of the whole namespace stopped existing beside it; the suite now
    makes that copy a band at a time, so the copy a parked byte saves is
    a band's, not the globe's, and the relief credited is the parked byte
    itself.  Crediting the retired factor would admit a run the card
    cannot hold, which is the direction every planning value here refuses.
    """
    b = max(1, int(bands))
    peak = int(estimate.device_peak_bytes)
    if peak == 0:
        return 0
    if b > 1:
        working = int(getattr(estimate, "working_grid_bytes", 0) or 0)
        fraction = banded_working_resident_fraction(
            getattr(estimate, "truncation", None))
        divisible = working * (1.0 - fraction)
        peak = int(peak - divisible * (1.0 - 1.0 / b))
        # The semi-Lagrangian gather's whole-grid transient stands whatever
        # the band count; once the bands have shrunk the rest it is what
        # is left standing.  MEASURED 2026-09-07, RTX 5070 Ti, T533 L40
        # sl_si, sixteen bands, every slice parked: out of memory inside
        # the gather at 15.96 GB, where the same shape on the IMEX core
        # ran at 9.59 GiB live.
        peak += int(getattr(estimate, "semilag_gather_bytes", 0) or 0)
    relief = int(max(0, int(spilled_bytes)) * SPILL_RELIEF)
    return max(0, peak - relief)


#: The native physics namespace, counted array by array on the real
#: thing: MEASURED 2026-09-06, RTX 5090, T533 L40 five-scheme, 120
#: persistent arrays of which 86 belong to the namespace -- 72 planes,
#: four four-layer soil stacks and ten levelled volumes.  In plane
#: equivalents that is ``88 + 10 x nlev`` (2.333 GiB at T533 L40, which
#: the census reproduces exactly).
#:
#: It is a COUNT, not a fit, and it exists because the namespace is EMPTY
#: at run start: the suite seeds it on its first call, after the tier has
#: had to decide what it holds.  A policy that read the cold state would
#: never park the largest slice there is.
NATIVE_NAMESPACE_FIXED_PLANES = 88
NATIVE_NAMESPACE_LEVELLED_ARRAYS = 10


def native_namespace_bytes(cfg) -> int:
    """What the native physics namespace will weigh once it exists."""
    if str(getattr(cfg, "physics_mode", "")) != "arwen-native":
        return 0
    try:
        estimate = estimate_global_memory(cfg)
    except Exception:  # noqa: BLE001 - no estimate is not a refusal
        return 0
    plane = int(estimate.nlat) * int(estimate.nlon) * int(estimate.float_itemsize)
    planes = NATIVE_NAMESPACE_FIXED_PLANES + (
        NATIVE_NAMESPACE_LEVELLED_ARRAYS * int(estimate.nlev)
    )
    return int(planes * plane)


def spill_census(bundle, cfg=None) -> dict[str, int]:
    """The persistent grid state of one bundle, in bytes, by slice.

    MEASURED off the arrays themselves rather than fitted, because the
    fit under-reads it: at T533 L40 native the sizer's own resident-grid
    and surface terms price the physics namespace and the surface
    reservoirs at about 0.47 GiB against a MEASURED 2.476 (2026-09-06,
    RTX 5090, 120 arrays).  A tier sized from that fit would park too
    little and report a relief the run does not get, so the tier is sized
    from the arrays it will hold.  (The fit itself is Lane 5's, and this
    census is the instrument that says by how much.)
    """
    from .constants import GRID_TRACERS

    rows = {name: 0 for name in SPILL_SLICES}
    for name in GRID_TRACERS:
        value = getattr(bundle.atmosphere, name, None)
        if value is not None:
            rows["tracers"] += int(value.nbytes)
    for value in bundle.surface.arrays().values():
        if value is not None:
            rows["surface"] += int(value.nbytes)
    for value in bundle.physics_state.arrays.values():
        if value is not None:
            rows["physics"] += int(value.nbytes)
    if rows["physics"] == 0 and cfg is not None:
        # The namespace has not been seeded yet: price what it will be
        # (:func:`native_namespace_bytes`), because the policy runs at run
        # start and the largest slice does not exist until the first
        # physics call.
        rows["physics"] = native_namespace_bytes(cfg)
    return rows


def spill_slices_for(mode: str, census: dict[str, int], *,
                     peak_bytes: int, budget_bytes: int | None,
                     relief_factor: float = 1.0,
                     ) -> tuple[str, ...]:
    """Which slices the pinned host tier holds: the MINIMUM-SPILL rule.

    ``"on"`` parks all three, ``"off"`` none.  ``"auto"`` parks nothing
    while the predicted peak fits the budget and then takes one slice at
    a time, COLDEST FIRST (:data:`SPILL_SLICES`), stopping the moment the
    prediction fits: mirroring the whole persistent state unconditionally
    pays roughly twice the host traffic the card actually needs, and the
    tracers -- the warmest slice, read by the tracer transport as well as
    by both physics halves -- are the last to go.

    The relief a parked slice buys is its own bytes: every consumer of
    the persistent grid state already builds its own device copy (the
    physics batch, the transport's mass stack, the checkpoint's host
    read), so what stops existing is the original standing beside that
    copy for the life of the run.
    """
    mode = str(mode)
    if mode == "off":
        return ()
    if mode == "on":
        return tuple(SPILL_SLICES)
    if mode != "auto":
        raise ValueError(
            f"host_spill must be one of {list(HOST_SPILL_MODES)}, got {mode!r}"
        )
    if not budget_bytes or not peak_bytes:
        return ()
    taken: list[str] = []
    predicted = float(peak_bytes)
    for name in SPILL_SLICES:
        if predicted <= float(budget_bytes):
            break
        size = int(census.get(name, 0))
        if size <= 0:
            continue
        taken.append(name)
        # ``relief_factor`` states what one parked byte is worth in the
        # units ``peak_bytes`` and ``budget_bytes`` are in.  One, the
        # default, is the pool's own live bytes.  The run planner passes
        # the fragmentation factor, because it weighs CARD bytes and a
        # byte that leaves the pool takes its share of the pool's
        # rounding with it.
        predicted -= size * float(relief_factor)
    return tuple(taken)


@dataclass(frozen=True)
class RunMemoryPlan:
    """What a bare run will do on THIS card, decided once.

    The band count and the pinned host tier are one decision, not two.
    They relieve different halves of the peak -- the band count divides
    the transient working set, the tier removes the persistent grid state
    -- and a run that chose them separately could pay for both where one
    would have done, or take a band count against a peak the tier was
    about to change.  The door prices this plan, the builder is handed
    this plan, and the receipt prints this plan, so the three cannot
    disagree about what the run is.
    """

    #: The band count the run takes, or ``None`` when nothing fits.
    bands: int | None
    #: The slices the pinned host tier holds, coldest first.
    spill_slices: tuple[str, ...]
    #: The pool live peak predicted at that band count and that spill.
    live_peak_bytes: int
    #: What the card must have free for it: ``live x frag + out-of-pool``.
    card_bytes: int
    #: Bytes the tier holds.
    spilled_bytes: int
    #: Three quarters of free, the share a run takes on a shared card.
    budget_bytes: int | None
    free_bytes: int | None
    #: ``"config"`` when the config named it, ``"sizer"`` when it was
    #: chosen here, ``"backend"`` when there is no card to choose against.
    bands_chosen_by: str
    spill_chosen_by: str
    #: One sentence: what was chosen and against what.
    reason: str
    #: Was this plan priced at a radiation chunk below the shipped
    #: default?  The chunk is part of the allocation pattern the
    #: fragmentation ratio is measured per (MEASURED 2026-09-06: T383
    #: resident reads x1.1011 at the default 12,500-column chunk and
    #: x1.2020 at 5,000, and T383 at eight bands with the tier reads
    #: x1.0296 against x1.1626), so it belongs to the plan and not to the
    #: caller: a door that left it out charged a 5,000-column T383 run
    #: the default chunk's ratio and asked 0.53 GiB less of the card than
    #: the measurement says that pattern needs.
    reduced_chunk: bool = False
    #: The shape the plan was priced for, so the fragmentation it reports
    #: is the one it was priced with (rows at this truncation or larger).
    truncation: int | None = None
    #: The completed run whose peak bounded this plan's live figure from
    #: above (:func:`completed_plan_ceiling`), or ``None`` when the model's
    #: own figure stood.  A figure a run measured carries no model margin.
    ceiling: str | None = None

    @property
    def fits(self) -> bool:
        return self.bands is not None

    @property
    def fragmentation(self) -> tuple[float, str]:
        """The measured pool-held-over-live ratio of THIS plan's
        allocation pattern, and the phrase naming the rows behind it.

        One reading, used by the price, by the verdict and by the
        receipt, so a sentence whose two halves do not multiply out
        cannot be printed.
        """

        return pool_fragmentation_for(
            bool(self.spill_slices), bool(self.bands and self.bands > 1),
            bool(self.reduced_chunk), bands=self.bands,
            truncation=self.truncation)

    def receipt(self) -> dict[str, object]:
        return {
            "latitude_bands": self.bands,
            "bands_chosen_by": self.bands_chosen_by,
            "host_spill_slices": list(self.spill_slices),
            "spill_chosen_by": self.spill_chosen_by,
            "predicted_live_peak_bytes": int(self.live_peak_bytes),
            "predicted_card_bytes": int(self.card_bytes),
            "spilled_bytes": int(self.spilled_bytes),
            "budget_bytes": self.budget_bytes,
            "free_bytes": self.free_bytes,
            "pool_fragmentation": self.fragmentation[0],
            "pool_fragmentation_evidence": self.fragmentation[1],
            "reduced_radiation_chunk": bool(self.reduced_chunk),
            "out_of_pool_bytes": OUT_OF_POOL_BYTES,
            "prediction_margin": 1.0 if self.ceiling else PREDICTION_MARGIN,
            "live_peak_ceiling": self.ceiling,
            "reason": self.reason,
        }


#: What the tier ACTUALLY parks, over what the terms below predict it
#: will.  MEASURED 2026-09-06 at T383 L40 on an RTX 5070 Ti: the tier's
#: own receipt reports 2,436,312,924 B parked where the estimate below
#: prices 2,522,957,415, so the estimate reads 3.6 percent high.
#:
#: It is corrected because the estimate is SUBTRACTED: over-stating what
#: the tier holds over-credits the relief and under-states the peak, which
#: is the admitting direction.  One measurement backs it, so the smallest
#: ratio measured is the factor and a second shape would be worth having.
SPILL_CENSUS_MEASURES = (
    {
        "label": "T383 L40, all three slices, RTX 5070 Ti",
        "parked_bytes": 2_436_312_924, "estimated_bytes": 2_522_957_415,
        "where": "the tier's own receipt on the bare run of gate DOOR-1, "
                 "2026-09-06 (2.2692 GiB parked against 2.3496 priced)",
    },
)

#: The factor the census estimate is scaled by: the smallest measured
#: ratio of parked to priced.
SPILL_CENSUS_FACTOR = min(
    float(row["parked_bytes"]) / float(row["estimated_bytes"])
    for row in SPILL_CENSUS_MEASURES)


def _spill_census_estimate(cfg, estimate) -> dict[str, int]:
    """The persistent grid state by slice, priced before it exists.

    The door decides the plan before a single array is allocated, so it
    cannot read the census off the bundle the way ``spill_census`` does.
    The three slices are priced from the same terms the estimate is built
    from, and the physics namespace from the array count measured on the
    real thing (:func:`native_namespace_bytes`), because at run start that
    namespace is EMPTY and a policy reading the cold state would decline
    to park the largest slice there is.
    """

    nlon = getattr(estimate, "nlon", None)
    itemsize = getattr(estimate, "float_itemsize", None)
    surface = getattr(estimate, "surface_grid_bytes", None)
    if nlon is None or itemsize is None or surface is None:
        # An estimate that does not carry the shape terms cannot be priced
        # for spill.  It returns EMPTY rather than a guess, and the plan
        # marks the choice "unpriced" so a reader is never told a tier was
        # weighed when it was not.
        return {}
    plane = int(estimate.nlat) * int(nlon) * int(itemsize)
    levelled = plane * int(estimate.nlev)
    rows = {
        "physics": native_namespace_bytes(cfg),
        "surface": int(surface),
        # The grid tracers of ONE state: the tier parks the bundle it is
        # handed, not both states the run keeps referenced.
        "tracers": int(levelled * len(GRID_TRACERS)),
    }
    # Scaled to what the tier MEASURABLY parks (:data:`SPILL_CENSUS_FACTOR`).
    # The estimate is subtracted from the peak, so reading it high credits
    # relief the run does not get.
    return {name: int(value * SPILL_CENSUS_FACTOR) for name, value in rows.items()}


def plan_run_memory(cfg, free_bytes: int | None, estimate=None,
                    census: dict[str, int] | None = None) -> RunMemoryPlan:
    """Choose the band count and the spill for ``cfg`` on this card.

    THE SEARCH ORDER IS THE MEASURED COST ORDER.  Band counts ascend, and
    at each count the tier takes the minimum number of slices, coldest
    first; the first combination whose CARD bytes fit is taken.  So the
    tier is paid before the band count is widened, and the reason is
    measured on both sides: banding costs 18 to 48 percent of the step at
    T255 and 26 percent at T533 (RTX 5090, 2026-09-06, eight profiled
    steps), while the tier's exposed transfer time measured 5.9 percent
    of a T383 step on the RTX 5070 Ti the same day.  Cheaper relief first.

    A count or a mode the CONFIG named is honoured, not searched: the
    plan then prices what the run will do rather than what it should.
    """

    if estimate is None:
        estimate = estimate_global_memory(cfg)
    mode = str(getattr(cfg, "host_spill", "auto") or "auto")
    if mode not in HOST_SPILL_MODES:
        raise ValueError(
            f"host_spill must be one of {list(HOST_SPILL_MODES)}, got {mode!r}"
        )
    declared_bands = int(getattr(cfg, "latitude_bands", 0) or 0)
    sizes = dict(census) if census else _spill_census_estimate(cfg, estimate)
    priced_spill = bool(sizes) and any(int(v) > 0 for v in sizes.values())
    # THE RADIATION CHUNK IS PART OF THE ALLOCATION PATTERN, so it is part
    # of what the plan is priced at.  MEASURED 2026-09-06, twice in the
    # same direction: T383 resident reads x1.1011 at the default
    # 12,500-column chunk and x1.2020 at 5,000, and T383 at eight bands
    # with the tier reads x1.0296 at the default and x1.1626 at 5,000 -- a
    # smaller chunk takes a gigabyte and a half off the LIVE peak while
    # the pool still grows for the same allocations, so the ratio rises on
    # a run that is smaller.  Leaving it out charged a 5,000-column T383
    # run the default chunk's ratio: 18.43 GiB of card asked where the
    # measurement says 18.96, against a run that held 17.76.
    reduced = _reduced_chunk(estimate.radiation_column_chunk)

    def priced(bands: int, slices: tuple[str, ...]
               ) -> tuple[int, int, int, str | None]:
        spilled = sum(int(sizes.get(name, 0)) for name in slices)
        live = banded_device_peak_bytes(estimate, bands, spilled_bytes=spilled)
        # A COMPLETED RUN OF A LIGHTER PLAN BOUNDS THIS ONE FROM ABOVE
        # (:data:`COMPLETED_PLAN_PEAKS`): a model figure above what the
        # card held for the same shape at fewer bands and fewer parked
        # slices is a figure the card has refuted, and it is priced at the
        # measurement, with no model margin, because a live peak is the
        # plan's own and read the same on every run of it.
        ceiling = completed_plan_ceiling(estimate, bands, slices)
        # A row at exactly this plan IS the figure, above or below the
        # model; a lighter row bounds it from above.
        capped = ceiling is not None and (
            completed_plan_is_exact(estimate, bands, slices, ceiling)
            or int(ceiling["peak_used_bytes"]) < live)
        if capped:
            live = int(ceiling["peak_used_bytes"])
        # A plan that parks anything is priced at the SPILLED population's
        # fragmentation, because the tier is what makes a run fragment
        # that way: crediting the parked bytes and then charging the
        # resident ratio would price a run nobody has measured.
        return live, card_required_bytes(
            live, spilled=bool(slices), banded=bands > 1,
            reduced_chunk=reduced, bands=bands,
            truncation=int(estimate.truncation),
            margin=1.0 if capped else None), spilled, (
                str(ceiling["label"]) if capped else None)

    # No card to decide against: one band, no tier, and the plan says so
    # rather than reporting a choice it did not make.
    if estimate.backend != "cupy" or not free_bytes:
        bands = declared_bands or 1
        slices = tuple(SPILL_SLICES) if mode == "on" else ()
        live, card, spilled, ceiling = priced(bands, slices)
        return RunMemoryPlan(
            bands=bands, spill_slices=slices, live_peak_bytes=live,
            card_bytes=card, spilled_bytes=spilled, budget_bytes=None,
            free_bytes=free_bytes,
            bands_chosen_by="config" if declared_bands else "backend",
            spill_chosen_by="config" if mode != "auto" else "backend",
            reduced_chunk=reduced,
            truncation=int(estimate.truncation),
            ceiling=ceiling,
            reason=("no card could be read, so the run is priced as it is "
                    "configured and nothing is chosen against a measurement"),
        )

    budget = int(DEVICE_BUDGET_FRACTION * float(free_bytes))
    widest = _widest_band_count(estimate.nlat)
    ladder = [b for b in LATITUDE_BAND_LADDER if b <= widest] or [1]
    if declared_bands > 0:
        ladder = [declared_bands]

    def candidates() -> list[tuple[str, ...]]:
        """The spill choices this mode allows, cheapest first.

        ``auto`` walks the coldest-first prefixes: nothing, then the
        physics namespace, then the surface reservoirs, then the tracers
        -- the warmest slice, read by the tracer transport as well as by
        both physics halves, and so the last to go.  A slice with no
        bytes is skipped rather than parked empty.

        EVERY CANDIDATE IS PRICED BY ``priced``, the same function that
        prices the answer.  An earlier form asked ``spill_slices_for`` to
        choose the slices and then priced the result separately, and the
        two disagreed: the chooser weighed the unparked run at the
        SPILLED fragmentation, decided nothing needed parking, and handed
        back a plan the pricer then refused.  One function, one figure.
        """
        if mode == "off":
            return [()]
        present = [n for n in SPILL_SLICES if int(sizes.get(n, 0)) > 0]
        if mode == "on":
            return [tuple(present)]
        return [tuple(present[:k]) for k in range(len(present) + 1)]

    # ONE PASS, AND IT IS MONOTONE.  The plan is the cheapest candidate
    # whose card figure, with the model's margin, fits what the card has
    # free.  Because the candidates are enumerated in cost order, more
    # free VRAM can only ever buy a cheaper plan.
    #
    # THE QUARTER-FREE BUDGET IS REPORTED, NOT PREFERRED, and the shape of
    # the answer is why.  Choosing the cheapest plan that fits three
    # quarters of free, and falling back to the whole card only when none
    # does, is NOT monotone once the host tier is a candidate: just below
    # a threshold the fallback hands back a light plan, and just above it
    # the budget pass hands back a heavier one, so a user who frees memory
    # watches the model start spilling.  Swept over T255, T383 and T533
    # from two to thirty-four gigabytes free, that rule flipped the parked
    # slice count three times on one truncation.  The budget survives as a
    # figure in the receipt and a line at the door -- whether the plan
    # leaves a quarter of the card free -- which is what a reader on a
    # shared card actually needs.
    for bands in ladder:
        for slices in candidates():
            live, card, spilled, ceiling = priced(bands, slices)
            # The margin is already inside ``card``: it multiplies the
            # LIVE figure the model predicts, not the measured terms that
            # turn it into card bytes (:func:`card_required_bytes`).
            if card > int(free_bytes):
                continue
            parked = (", ".join(slices) + " in the pinned host tier"
                      if slices else "nothing parked")
            inside = card <= budget
            bounded = ("" if ceiling is None else
                       f" (the live figure is the {live / GIB:.2f} GiB a "
                       f"completed run of this shape held: {ceiling})")
            return RunMemoryPlan(
                bands=bands, spill_slices=slices, live_peak_bytes=live,
                card_bytes=card, spilled_bytes=spilled,
                budget_bytes=budget, free_bytes=int(free_bytes),
                bands_chosen_by="config" if declared_bands else "sizer",
                spill_chosen_by=(
                    "config" if mode != "auto"
                    else ("sizer" if priced_spill else "unpriced")),
                reduced_chunk=reduced,
                truncation=int(estimate.truncation),
                ceiling=ceiling,
                reason=(
                    f"{bands} latitude band{'' if bands == 1 else 's'} with "
                    f"{parked}, priced at {card / GIB:.2f} GiB of card "
                    f"against {int(free_bytes) / GIB:.2f} GiB free{bounded}"
                    + ("" if inside else
                       f"; it does NOT leave the quarter-free share "
                       f"({budget / GIB:.2f} GiB), so a second process "
                       "arriving mid-run has nowhere to go")),
            )

    widest_bands = ladder[-1]
    slices = candidates()[-1]
    live, card, spilled, ceiling = priced(widest_bands, slices)
    return RunMemoryPlan(
        bands=None, spill_slices=slices, live_peak_bytes=live,
        card_bytes=card, spilled_bytes=spilled, budget_bytes=budget,
        free_bytes=int(free_bytes),
        bands_chosen_by="sizer",
        spill_chosen_by="sizer" if priced_spill else "unpriced",
        reduced_chunk=reduced,
        truncation=int(estimate.truncation),
        ceiling=ceiling,
        reason=(
            f"nothing fits: the widest band count the grid allows "
            f"({widest_bands})"
            + ("" if not slices else " with every slice parked")
            + f" still prices at {card / GIB:.2f} GiB of card against "
            f"{int(free_bytes) / GIB:.2f} GiB free"),
    )


def latitude_bands_for(cfg, free_bytes: int | None, estimate=None):
    """``(bands, live_peak_bytes)`` this run will take on a card with
    ``free_bytes`` free, or ``(None, resident_peak)`` when nothing fits.

    A thin reading of :func:`plan_run_memory`, which decides the band
    count and the pinned host tier together.  Kept because the door, the
    builder and the band-pipeline tests all ask this question in these
    words; the plan is what answers it, so the count the gate weighs is
    the count the run takes.
    """
    if estimate is None:
        estimate = estimate_global_memory(cfg)
    plan = plan_run_memory(cfg, free_bytes, estimate)
    if plan.bands is None:
        return None, estimate.device_peak_bytes
    return plan.bands, plan.live_peak_bytes


def auto_latitude_bands(cfg, free_bytes: int | None = None) -> int:
    """The band count a bare run takes: one when the resident run fits.

    The rule, and why it is a rule rather than a flag: a run whose
    resident peak does not fit the card is admitted today and dies in the
    allocator (MEASURED 2026-09-06, T533 L40 on a 32 GiB RTX 5090: the
    door predicted 24.45 GiB, the card had 30.90 GiB free, the run
    started and the live peak reached 27.57 before ``transport.py``
    asked for one more ten-tracer grid volume).  With a band count the
    same run streams grid space through the card and fits.  Choosing that
    count is the sizer's job, so a BARE run at T533 starts (fixed means
    default); naming ``latitude_bands`` in the config overrides it.

    IT NEVER REFUSES, and the door does.  A refusal belongs at the door,
    once, before a byte is allocated, where it can print the itemization
    and the largest truncation the card fits; raising here would refuse
    from inside ``build_model_and_cold_state``, which every tool and test
    that builds a model calls.  When nothing fits, this returns the widest
    count the grid allows, which is the best the card can be given, and
    ``run_memory_gate`` is what says no.

    Returns one on the numpy backend and whenever no free-byte figure can
    be measured: there is no card to size against, and a band count
    chosen against nothing would be a number rather than a decision.
    """
    if getattr(cfg, "backend", "numpy") != "cupy":
        return 1
    declared = int(getattr(cfg, "latitude_bands", 0) or 0)
    if declared > 0:
        return declared
    try:
        estimate = estimate_global_memory(cfg)
    except Exception:  # noqa: BLE001 - a prediction that cannot be made is not a refusal
        return 1
    if free_bytes is None:
        free_bytes = _free_device_bytes()
    if not free_bytes:
        return 1
    bands, _peak = latitude_bands_for(cfg, free_bytes, estimate)
    if bands is not None:
        return bands
    ladder = [
        b for b in LATITUDE_BAND_LADDER
        if b <= _widest_band_count(estimate.nlat)
    ]
    return ladder[-1] if ladder else 1


def widest_band_count(nlat: int) -> int:
    """The largest band count the grid allows (bands.widest_band_count)."""
    from .bands import widest_band_count as _widest

    return _widest(int(nlat))


def _widest_band_count(nlat: int) -> int:
    from .bands import widest_band_count

    return widest_band_count(int(nlat))


def _free_device_bytes() -> int | None:
    """What the card has free, read WITHOUT opening a context here.

    Asked in process this would stand up a CUDA primary context -- MEASURED
    0.486 GiB on the RTX 5090 -- which the run then holds for the whole
    forecast, and it would read the card at whatever instant the model
    build happened to reach rather than before the run allocated anything.
    Both matter: a reading taken after the Legendre tables are on the card
    is several GiB lower than the reading the door took, and the two
    disagreeing is how a run gets one band count from the door and another
    from the builder.  The short-lived subprocess probe is the tree's own
    instrument for exactly this.
    """
    try:
        from woof.core.preflight import device_memory_probe_subprocess

        probe = device_memory_probe_subprocess()
    except Exception:  # noqa: BLE001 - no card, no driver, no figure
        return None
    if not probe:
        return None
    # The runner's parking decision reads this figure, and it credits the
    # blocks this process's own pool already holds exactly as the door's
    # gate does (pool_bytes_held_unused): MEASURED 2026-09-07 on the RTX
    # 5090, a DA run launched with 30.2 GiB free and a 21 GiB pre-grown
    # pool read 9 GiB here, parked the tracers to fit an 11.6 GiB
    # requirement into it, and died in the semi-Lagrangian gather.
    return int(probe["free_bytes"]) + pool_bytes_held_unused()


def evaluate_global_gates(estimate: GlobalMemoryEstimate,
                          budget_bytes: int | None
                          ) -> dict[str, bool | None]:
    """The two legs, keyed by :data:`GLOBAL_GATE_METRICS`.

    ``None`` marks a leg with nothing to weigh -- no budget could be
    measured or declared -- and a leg that was never evaluated can never
    report a pass, exactly as ``evaluate_alloc_gates`` treats an
    unmeasured allocation.

    The tables leg is not redundant beside the peak leg.  It names WHICH
    allocation dies first: the tables are built by
    ``SphericalHarmonicTransform.__post_init__``, before a single
    prognostic field exists, so a card that cannot hold them fails during
    construction and no amount of trimming the vertical column or the
    tracer set moves it.
    """

    if budget_bytes is None:
        return {metric: None for metric in GLOBAL_GATE_METRICS}
    return {
        "arwen_global_legendre_tables_fit_device":
            estimate.legendre_table_bytes <= budget_bytes,
        # CARD bytes, not pool bytes.  The pool's live peak is what the
        # model predicts; what a card must hold is that times the measured
        # fragmentation plus the measured bytes outside the pool, and a
        # leg that compared the first against free VRAM passed runs the
        # card could not hold (MEASURED 2026-09-06: printed 24.45 GiB,
        # reached 27.57 GiB live, died).
        "arwen_global_step_peak_fits_device":
            estimate_card_required_bytes(estimate) <= budget_bytes,
    }


def largest_truncation_within(cfg, budget_bytes: int) -> int | None:
    """The largest truncation whose device peak fits ``budget_bytes``.

    THE FIRST OVER-BUDGET LEVER, in the same spirit as
    ``recommend_column_chunk``: a number the reader can type, not an
    instruction to make the run smaller.  ``None`` when even the smallest
    truncation the loader admits (3) does not fit, which means the levels
    or the precision have to move instead.

    Monotone in truncation -- every term grows with it -- so a bisection
    is exact rather than a scan.
    """

    import dataclasses

    def fits(truncation: int) -> bool:
        candidate = dataclasses.replace(
            cfg, truncation=truncation, nlat=None, nlon=None,
            zonal_wavenumber=min(cfg.zonal_wavenumber, truncation),
            diffusion_preserve_degree=min(
                cfg.diffusion_preserve_degree, truncation))
        return estimate_card_required_bytes(
            estimate_global_memory(candidate)) <= budget_bytes

    if not fits(3):
        return None
    low, high = 3, int(cfg.truncation)
    if fits(high):
        return high
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


#: Runs that COMPLETED at a shape the fit has no row for.
#:
#: A calibration row has to be the probe of record or it moves the fit
#: (the same ten steps with ONE radiation call read 18.7 percent under
#: the day at T255), so a run taken at any other recipe cannot join
#: :data:`DEVICE_PEAK_CALIBRATION`.  It is still a measurement, and it
#: still answers the only question the fitted-domain refusal asks: has
#: anything weighed this shape on a card?
#:
#: These rows lift that refusal, and only when the model's figure sits at
#: or ABOVE the measured peak.  If the model reads under a run that
#: actually happened, the refusal stands and says which run.
MEASURED_RUNS = (
    {
        "label": "T533, 40 levels, one band",
        "truncation": 533, "nlat": 801, "nlon": 1602, "nlev": 40,
        "precision": "float32", "radiation_column_chunk": 12_500,
        "peak_used_bytes": 28_137_857_536,
        "held_bytes": 32_908_712_960,
        "measured_on": "2026-09-06", "device": "NVIDIA RTX 5090",
        "run": "the band lane's ten-step capacity leg (the 32 GB host, "
               "the T533 one-band leg), dt 30, the ledger and "
               "the checkpoints of that config, and the radiation called "
               "at step 0 only -- so this peak is BELOW what the probe of "
               "record would read at this shape, which is the direction a "
               "lower bound has to err in.  The tree it cut from died "
               "here with an OutOfMemoryError at step 1",
    },
    {
        "label": "T533, 40 levels, eight bands",
        "truncation": 533, "nlat": 801, "nlon": 1602, "nlev": 40,
        "precision": "float32", "radiation_column_chunk": 12_500,
        "peak_used_bytes": 27_316_350_976,
        "held_bytes": 29_954_767_872,
        "measured_on": "2026-09-06", "device": "NVIDIA RTX 5090",
        "run": "the same leg at eight bands (the 32 GB host, the T533 "
               "eight-band leg): 25.44 GiB live, 27.90 "
               "held, ten steps, passed",
    },
)


def measured_runs_at(truncation: int, nlev: int, precision: str) -> tuple:
    """The completed runs recorded for this shape, if any."""

    return tuple(
        row for row in MEASURED_RUNS
        if int(row["truncation"]) == int(truncation)
        and int(row["nlev"]) == int(nlev)
        and row["precision"] == precision)


#: Completed runs with their PLAN: the band count, the parked slices, the
#: core and the radiation chunk beside the peak the pool actually held.
#: A run that completed at a plan bounds every heavier plan of the same
#: shape from ABOVE: more bands can only shrink the physics half-step's
#: working set, a superset of parked slices can only take bytes off the
#: card, a smaller radiation chunk can only shrink the radiation's
#: transient, and every other phase of the step is the same phase.  So a
#: model figure above such a run's peak is a figure the card has already
#: refuted, and the plan is priced at the measurement instead
#: (:func:`completed_plan_ceiling`).  MEASURED 2026-09-07: the model read
#: 15.08 GiB live for T383 L40 at two bands with every slice parked and
#: refused the 34.7 km day on a 16 GB card that had just completed it at
#: 12.41 GiB live; and 22.81 GiB for T533 L40 at one band with every slice
#: parked, at a pool ratio borrowed from another class, where the 25 km
#: day completed at 21.12.  Same core only: the semi-Lagrangian core holds
#: a second time level and a whole-grid gather the Eulerian core does not,
#: so neither bounds the other.
COMPLETED_PLAN_PEAKS = (
    {
        "label": "T533 L40 on the semi-Lagrangian core, one band, every slice parked, the 25 km day",
        "truncation": 533, "nlat": 801, "nlon": 1602, "nlev": 40,
        "precision": "float32", "core": "semilag",
        "radiation_column_chunk": 5_000,
        "bands": 1, "spill_slices": ("physics", "surface", "tracers"),
        "peak_used_bytes": 24_940_350_464, "held_bytes": 30_183_100_928,
        "measured_on": "2026-09-07", "device": "NVIDIA RTX 5090",
        "run": "the bare 24 h day of configs/verify/"
               "arwen_global_gdas_t533_native_sl_si_24h.toml on the tree "
               "with the physics suite banded (the 32 GB host, capacity "
               "probe t533_day): 288 steps at 300 s, "
               "2.98 s a step, 953 s of wall, status pass, "
               "23.23 GiB live.  The tree before it read 21.12 GiB "
               "live and 25.03 held on the same config, card and plan "
               "(the 32 GB host, the same 24 h day before the banding, "
               "2.95 s a step, 998 s): "
               "the banded suite at one band fills full-latitude waists "
               "after its loop that the resident call analysed in place, "
               "and the ceiling carries the reading of the tree that ships",
    },
    {
        "label": "T533 L40 on the semi-Lagrangian core, two bands, every slice parked, the ten-step probe",
        "truncation": 533, "nlat": 801, "nlon": 1602, "nlev": 40,
        "precision": "float32", "core": "semilag",
        "radiation_column_chunk": 5_000,
        "bands": 2, "spill_slices": ("physics", "surface", "tracers"),
        "peak_used_bytes": 21_152_070_656, "held_bytes": 31_715_355_136,
        "measured_on": "2026-09-07", "device": "NVIDIA RTX 5090",
        "run": "the banded-physics lane's three bare ten-step completions "
               "of the same config cut to ten steps (the 32 GB host, "
               "capacity probes t533_bare.shared1 to 3), the same live "
               "peak each time; another process computed "
               "on the card during each, which refuses their timing and "
               "not their allocation",
    },
    {
        "label": "T383 L40 on the semi-Lagrangian core, two bands, every slice parked, the 34.7 km day",
        "truncation": 383, "nlat": 576, "nlon": 1152, "nlev": 40,
        "precision": "float32", "core": "semilag",
        "radiation_column_chunk": 12_500,
        "bands": 2, "spill_slices": ("physics", "surface", "tracers"),
        "peak_used_bytes": 13_322_662_912, "held_bytes": 15_300_097_536,
        "measured_on": "2026-09-07", "device": "NVIDIA RTX 5070 Ti",
        "run": "the bare 24 h day of configs/verify/"
               "arwen_global_gdas_t383_native_sl_si_24h.toml on the 16 GB "
               "card (the 16 GB host, the 24 h day run): 288 steps at 300 "
               "s, 3.35 s a step, 1,022 s of wall, status pass, the door "
               "having priced 11.93 GiB live and 14.72 of card",
    },
    {
        "label": "T383 L40 on the semi-Lagrangian core, sixteen bands, every slice parked, the physics banded",
        "truncation": 383, "nlat": 576, "nlon": 1152, "nlev": 40,
        "precision": "float32", "core": "semilag",
        "radiation_column_chunk": 12_500,
        "bands": 16, "spill_slices": ("physics", "surface", "tracers"),
        "peak_used_bytes": 10_446_974_464, "held_bytes": 12_746_315_776,
        "measured_on": "2026-09-07", "device": "NVIDIA RTX 5070 Ti",
        "run": "the cards lane's ten-step probe of the T383 GDAS "
               "semi-Lagrangian config (t383-slsi-cards.toml) at "
               "sixteen bands with all three slices parked on the "
               "tree with the physics suite banded, the 16 GB card to "
               "itself (the 16 GB host the probe_t383_16b_tier leg): status "
               "pass, 4.26 s a step, 9.73 GiB live, 11.87 held (x1.2201), "
               "2.27 GiB parked.  The plan a two-card T383 run's rank on "
               "this card runs, which the door refused at 18.30 GiB of "
               "card before this row and the exact-shape ratio existed",
    },
)


def _estimate_core(estimate) -> str:
    """``"semilag"`` when the estimate carries the semi-Lagrangian core's
    second time level, ``"eulerian"`` otherwise."""

    return "semilag" if int(getattr(estimate, "trajectory_bytes", 0) or 0) else "eulerian"


def completed_plan_ceiling(estimate, bands: int, slices) -> dict | None:
    """The completed run whose peak bounds ``(bands, slices)`` from above
    for this shape, or ``None``.

    A row applies when it is the same shape, the same core and the same
    precision, its band count is at most ``bands``, its parked slices are
    a subset of ``slices``, and its radiation chunk is at least the
    estimate's.  Of the rows that apply, the smallest peak is the bound.
    """

    want = set(str(name) for name in (slices or ()))
    chunk = int(getattr(estimate, "radiation_column_chunk", 0) or 0)
    core = _estimate_core(estimate)
    rows = [
        row for row in COMPLETED_PLAN_PEAKS
        if int(row["truncation"]) == int(estimate.truncation)
        and int(row["nlev"]) == int(estimate.nlev)
        and str(row["precision"]) == str(getattr(estimate, "precision", "float32"))
        and str(row["core"]) == core
        and int(row["bands"]) <= int(bands)
        and set(row["spill_slices"]) <= want
        and int(row["radiation_column_chunk"]) >= chunk
    ]
    if not rows:
        return None
    # A row at EXACTLY this plan is the plan's measurement and outranks a
    # lighter row's bound: the one-band 25 km day on this tree read 23.23
    # GiB live where the model read 22.81 (MEASURED 2026-09-07, RTX 5090),
    # and a figure the card held is the figure, above or below the model.
    exact = [row for row in rows
             if int(row["bands"]) == int(bands)
             and set(row["spill_slices"]) == want
             and int(row["radiation_column_chunk"]) == chunk]
    if exact:
        return max(exact, key=lambda row: int(row["peak_used_bytes"]))
    return min(rows, key=lambda row: int(row["peak_used_bytes"]))


def completed_plan_is_exact(estimate, bands: int, slices, row) -> bool:
    """Is ``row`` a measurement of exactly ``(bands, slices)`` at this
    estimate's radiation chunk, rather than a lighter plan's bound?"""

    return (row is not None
            and int(row["bands"]) == int(bands)
            and set(row["spill_slices"]) == set(str(n) for n in (slices or ()))
            and int(row["radiation_column_chunk"])
            == int(getattr(estimate, "radiation_column_chunk", 0) or 0))


def fitted_truncation_ceiling() -> int:
    """The largest truncation the device-peak model has been MEASURED at.

    Read off :data:`DEVICE_PEAK_CALIBRATION` rather than typed, so adding
    a measured row raises the ceiling and no constant has to be found.
    """

    return max(CALIBRATED_DOMAIN["truncation"])


def above_fitted_domain(cfg, nlev: int | None = None) -> str | None:
    """The refusal for a shape above the fitted domain, or ``None``.

    THE BREAKAGE THIS PREVENTS, MEASURED 2026-09-06: above the largest
    measured truncation the model's grid scaling is the structure's
    arithmetic and not a measurement, and it has only ever erred in the
    admitting direction there -- the door printed 24.45 GiB, the card had
    30.90 GiB free, the run was admitted, and the live peak reached 27.57
    before the allocator killed it several minutes in.  A refusal at the
    door costs a sentence.  Being wrong the other way costs the run, and
    on a shared card it costs whatever else was on the card with it.

    It is a REFUSAL, not a warning, because a warning is what the
    extrapolation notes already were: they printed on that run, and the
    run started anyway.  The way past it is a measurement -- one ten-step
    probe of the shape, added to :data:`DEVICE_PEAK_CALIBRATION` -- and
    the sentence says so, because a door that cannot be got past by doing
    the right thing is a wall rather than a gate.
    """

    ceiling = fitted_truncation_ceiling()
    truncation = int(getattr(cfg, "truncation", 0) or 0)
    if truncation <= ceiling:
        return None
    if nlev is None:
        nlev = len(getattr(cfg, "b_half", ()) or ()) - 1
    # A shape a run has actually COMPLETED on a card is not a shape
    # nothing has weighed.  The refusal lifts, but only while the model
    # sits at or above what that run measured: a model reading under a
    # run that happened is exactly the failure this refusal exists for,
    # and then the sentence names the run instead of the extrapolation.
    completed = measured_runs_at(truncation, nlev, cfg.precision)
    if completed:
        # The comparison is made at the ROW's own radiation chunk, not at
        # this config's: a smaller chunk is a smaller run, and asking
        # whether the model reads under a completed run has to price the
        # run that happened rather than the one being started.
        itemsize = 4 if cfg.precision == "float32" else 8
        under = []
        for row in completed:
            try:
                figure = predict_device_peak_bytes(
                    int(row["truncation"]), int(row["nlat"]), int(row["nlon"]),
                    int(row["nlev"]), itemsize,
                    int(row["radiation_column_chunk"]))
            except Exception:  # noqa: BLE001 - no figure is not a licence
                figure = 0
            if figure < int(row["peak_used_bytes"]):
                under.append((row, figure))
        if not under:
            return None
        worst_row, predicted = max(
            under, key=lambda pair: int(pair[0]["peak_used_bytes"]) - pair[1])
        return (
            f"T{truncation} is above the largest truncation this card "
            f"model has been fitted at (T{ceiling}), and the one run that "
            f"reached this shape reads ABOVE the model: "
            f"{int(worst_row['peak_used_bytes']) / GIB:.2f} GiB measured "
            f"({worst_row['label']}, {worst_row['device']}, "
            f"{worst_row['measured_on']}) against "
            f"{predicted / GIB:.2f} GiB predicted.  A model that reads "
            f"under a run that happened is exactly what admits a run the "
            f"card cannot hold.  Measure the shape at the probe of record "
            f"({PROBE_RECIPE}) and add its receipt's "
            f"device_memory.peak_used_bytes to "
            f"sizing.DEVICE_PEAK_CALIBRATION."
        )
    worst = worst_held_out_residual_fraction()
    return (
        f"T{truncation} is above the largest truncation this card model "
        f"has been measured at (T{ceiling}), and above it the model has "
        f"only ever erred in the direction that admits a run the card "
        f"cannot hold: MEASURED 2026-09-06, the door printed 24.45 GiB "
        f"for a T533 run on a 32 GiB card with 30.90 GiB free, the run "
        f"was admitted, and its live peak reached 27.57 GiB before the "
        f"allocator killed it inside the tracer transport.  The model's "
        f"held-out error over the shapes it HAS measured is "
        + ("not a number" if worst is None else f"{100 * worst:.0f} percent")
        + f", and nothing bounds it above T{ceiling}.  Measure the shape "
        f"and the ceiling moves: run the ten-step probe of record "
        f"({PROBE_RECIPE}) at this shape on the card, then add its "
        f"receipt's device_memory.peak_used_bytes to "
        f"sizing.DEVICE_PEAK_CALIBRATION as a new row.  `woof check` "
        f"still prices this configuration and prints the itemization; "
        f"what is refused is starting a forecast on a number nothing has "
        f"weighed."
    )


class GlobalMemoryRefusal(RuntimeError):
    """This card cannot hold this configuration; the run is not started.

    ``RuntimeError`` deliberately: both WOOF global doors already convert
    that into a one-line refusal at exit 2 (``cli._REFUSALS`` and
    ``cli._DOOR_REFUSALS``), so the message below is the whole user
    experience and there is no traceback to read past.
    """


def prediction_sentence(estimate: GlobalMemoryEstimate) -> str:
    """``X GiB device peak predicted (...)``: the figure with what it is."""

    worst = worst_calibration_residual_fraction()
    held_out = worst_held_out_residual_fraction()
    calibration = (
        f"calibrated model, worst calibration residual "
        f"{100 * worst:.1f}% in sample and "
        + ("not a number" if held_out is None else f"{100 * held_out:.1f}%")
        + f" held out over {len(DEVICE_PEAK_CALIBRATION)} measured runs of "
        f"{calibration_shapes()} shapes"
    )
    if not estimate.calibrated:
        calibration += "; EXTRAPOLATED: " + "; ".join(estimate.extrapolation)
    return (f"{_format_bytes(estimate.device_peak_bytes).strip()} device peak "
            f"predicted ({calibration})")


def run_memory_gate(cfg, *, probe=None) -> dict:
    """Price ``cfg`` against THIS card, before the transform is built.

    Returns ``{"verdict", "refuse", "free_bytes", "estimate", "plan"}``.

    WHAT IS WEIGHED IS WHAT THE CARD MUST HOLD, and that is the change
    this gate needed.  The estimate's figure is the CuPy pool's live
    peak, because that is the quantity it was fitted to; a card must also
    hold the pool's fragmentation and everything outside the pool, and
    charging neither is how the 2026-09-06 T533 run was admitted at a
    printed 24.45 GiB and died at a 27.57 GiB live peak
    (:func:`card_required_bytes`).

    The rule is otherwise ``woof go``'s memory gate: refuse only on a
    genuine out-of-memory prediction -- the plan the sizer would take
    does not fit even the WHOLE card's free VRAM -- and never refuse on a
    card that could not be read.  A gate that fired on the margin would
    turn the model's residual into a refusal of runs that fit.

    Above the largest measured truncation it refuses for a different
    reason, named separately (:func:`above_fitted_domain`): there the
    figure itself is unweighed.

    The card is read in a SHORT-LIVED SUBPROCESS
    (:func:`device_memory_probe_subprocess`).  Asked in-process it would
    stand up a CUDA primary context -- measured 0.486 GiB on the RTX 5090
    -- which this process then holds for the entire forecast, as a
    consumer no term of the estimate it had just computed names.
    """

    from woof.core.preflight import device_memory_probe_reason
    from woof.core.preflight import device_memory_probe_subprocess
    from woof.local_gpu import no_local_gpu

    estimate = estimate_global_memory(cfg)
    if estimate.backend != "cupy":
        return {"verdict": (
            f"memory: {_format_bytes(estimate.host_peak_bytes).strip()} host "
            f"peak, no device (backend='numpy')"),
            "refuse": False, "free_bytes": None, "estimate": estimate,
            "plan": plan_run_memory(cfg, None, estimate)}
    unfitted = above_fitted_domain(cfg, estimate.nlev)
    predicted = prediction_sentence(estimate)
    if probe is None:
        # ``no_local_gpu`` is asked HERE and not above, so a probe handed
        # in by a caller is still honoured: a supplied reading is not
        # device contact, and the flag exists to stop this process from
        # opening the card, not to make the gate unusable.
        if no_local_gpu():
            return {"verdict": (
                f"memory: {predicted}; card not read (GPUWM_NO_LOCAL_GPU is set)"),
                "refuse": bool(unfitted), "unfitted": unfitted,
                "free_bytes": None, "estimate": estimate,
                "plan": plan_run_memory(cfg, None, estimate)}
        probe = device_memory_probe_subprocess()
    if probe is None:
        return {"verdict": (
            f"memory: {predicted}; not weighed ({device_memory_probe_reason()})"),
            "refuse": bool(unfitted), "unfitted": unfitted,
            "free_bytes": None, "estimate": estimate,
            "plan": plan_run_memory(cfg, None, estimate)}
    held = pool_bytes_held_unused()
    free = int(probe["free_bytes"]) + held
    # "predicted": the calibrated model's figure, printed before a byte is
    # allocated.  The measured peak is the run's own, read at the
    # allocator and printed after the run (device_memory.py).
    held_text = (
        "" if held == 0
        else f" ({_format_bytes(held).strip()} of it held in this "
             "process's pool)")
    # THE GATE PRICES THE PLAN IT WILL START, not the resident run it
    # would have started before grid space could stream.  A door that
    # weighed the resident peak would refuse T533 L40 on a 32 GiB card --
    # the run the band count exists to admit -- and print a number the run
    # was never going to reach.
    plan = plan_run_memory(cfg, free, estimate)
    if plan.bands in (None, 1) and not plan.spill_slices:
        plan_text = ""
    else:
        plan_text = "; " + plan.reason
    # The ratio the verdict quotes is the PLAN's, not the module's default:
    # a banded run with the tier is charged its own class, and printing
    # the resident class's figure beside a banded plan's card total is a
    # sentence whose two halves do not multiply out.
    frag, frag_evidence = plan.fragmentation
    return {
        "verdict": (
            f"memory: {predicted}; "
            f"{_format_bytes(plan.card_bytes).strip()} of card required "
            f"({frag:.4f} pool fragmentation, {frag_evidence}, and "
            f"{_format_bytes(OUT_OF_POOL_BYTES).strip()} outside the pool, "
            "measured) against "
            f"{_format_bytes(free).strip()} free{held_text}{plan_text}"),
        "refuse": bool(unfitted) or not plan.fits,
        "unfitted": unfitted,
        "latitude_bands": plan.bands,
        "banded_peak_bytes": plan.live_peak_bytes,
        "card_required_bytes": plan.card_bytes,
        "host_spill_slices": plan.spill_slices,
        "plan": plan,
        "free_bytes": free,
        "estimate": estimate,
    }


def pool_bytes_held_unused() -> int:
    """The bytes THIS process already holds in its CuPy pool and is not
    using.

    A launcher on a shared card pre-grows the pool by one block before the
    model starts (a reservation, so another launch cannot take the job's
    memory mid-run; the arm of record of 2026-09-04 ran that way at 9.5
    GiB).  The gate's subprocess probe reads the card AFTER that block was
    taken, so it reads the reservation as another tenant's memory and
    refuses the very run the reservation was made for: a T533 probe with
    19.5 GiB held against a 24.2 GiB figure read 12 GiB free and exited
    at the door (2026-09-05).  Those bytes are the run's own to use and
    are credited here.  Read from ``sys.modules`` and never imported: a
    process that has not touched cupy has no pool and must not stand up a
    CUDA context to learn it (that is what the subprocess probe exists to
    avoid).  Zero on any doubt.
    """

    import sys

    cupy = sys.modules.get("cupy")
    if cupy is None:
        return 0
    try:
        # The pool the process is actually spending, not the default one by
        # name: a run under the driver's async pool or the slab holds its
        # reservation somewhere else, and crediting the default pool's
        # (empty) free bytes refuses the run the reservation was made for.
        from .device_memory import installed_pool

        pool = installed_pool(cupy)
        return max(0, int(pool.total_bytes()) - int(pool.used_bytes()))
    except Exception:  # noqa: BLE001 - a pool that cannot be read holds nothing the gate may count
        return 0


def refusal_sentence(estimate: GlobalMemoryEstimate, free_bytes: int,
                     *, config_path, plan: "RunMemoryPlan | None" = None) -> str:
    """What an undersized card is told, with the failing term named.

    The comparison is CARD bytes against free VRAM, and the sentence says
    which of the three terms carries it: the pool's live peak, the pool's
    fragmentation, or the bytes outside the pool.  A refusal that quoted
    only the live peak would leave a reader looking at a figure smaller
    than the free VRAM printed beside it and no way to tell why the run
    was refused.
    """

    # The figure quoted is the PLAN's when there is one, because that is
    # what was weighed; the resident peak's own card figure only when
    # there is not.
    spilled = bool(getattr(plan, "spill_slices", ()))
    banded = bool(getattr(plan, "bands", None) and plan.bands > 1)
    frag = (plan.fragmentation[0] if plan is not None
            else pool_fragmentation_for(
                spilled, banded,
                _reduced_chunk(estimate.radiation_column_chunk))[0])
    card = (int(plan.card_bytes) if plan is not None and plan.card_bytes
            else estimate_card_required_bytes(
                estimate, spilled=spilled, banded=banded))
    if estimate.legendre_table_bytes > free_bytes:
        where = (
            f"the Legendre tables alone are "
            f"{_format_bytes(estimate.legendre_table_bytes).strip()}, so "
            f"this would die inside "
            f"SphericalHarmonicTransform.__post_init__ before one "
            f"prognostic field exists")
    elif estimate.resident_bytes > free_bytes:
        where = (
            f"the tables fit but the resident set "
            f"({_format_bytes(estimate.resident_bytes).strip()}) does not, "
            f"so this would die building the initial state")
    else:
        where = (
            f"the resident set "
            f"({_format_bytes(estimate.resident_bytes).strip()}) fits and "
            f"the step's working set does not, so this would die as a CuPy "
            f"OutOfMemoryError inside the first step -- in the physics "
            f"exchange, a stacked synthesis or the ledger's first sample -- "
            f"with a failure receipt carrying the peak up to the death")
    tried = "" if plan is None else f"  The sizer tried: {plan.reason}."
    return (
        f"this card cannot hold this configuration: T{estimate.truncation} "
        f"at {estimate.nlon}x{estimate.nlat}x{estimate.nlev} in "
        f"{estimate.precision} peaks at "
        f"{_format_bytes(estimate.device_peak_bytes).strip()} of pool by the "
        f"calibrated model ({DEVICE_PEAK_MEASURES.split(';')[0]}), and the "
        f"cheapest plan the sizer can build for it needs "
        f"{_format_bytes(card).strip()} of CARD once the measured "
        f"{frag:.4f} pool fragmentation of that plan's own allocation "
        f"pattern and the measured "
        f"{_format_bytes(OUT_OF_POOL_BYTES).strip()} outside the pool are "
        f"charged, against {_format_bytes(free_bytes).strip()} free VRAM, "
        f"and {where}.{tried}  "
        f"Refused here rather than as a CuPy OutOfMemoryError partway "
        f"through.  Run `woof check {config_path}` for the itemization "
        f"and the largest truncation this card fits."
    )


# ---------------------------------------------------------------------------
# `woof check CONFIG` for an Arwen Global run config
# ---------------------------------------------------------------------------

def _format_bytes(n: int | None) -> str:
    return "n/a" if n is None else f"{n / GIB:7.2f} GiB"


def _leg_text(value: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "not measured"}[value]


def _device_budget(free_bytes: int | None) -> int | None:
    """What the device peak is weighed against: free VRAM less the margin.

    The SAME pair ``woof check`` compares a regional envelope against
    (``core.preflight``'s ``envelope_budget``) and the same one
    ``woof go``'s memory gate admits on.  ``EXTERNAL_MARGIN_BYTES`` and
    nothing else is subtracted, because this estimate already models
    every byte this process puts on the card; what it does not model is
    other processes, which is exactly what that margin is for.
    """

    from woof.core.preflight import EXTERNAL_MARGIN_BYTES

    if free_bytes is None:
        return None
    return max(0, int(free_bytes) - EXTERNAL_MARGIN_BYTES)


def global_check_main(args) -> int:
    """``woof check CONFIG`` where CONFIG is a WOOF global run config.

    Exit codes are the regional preflight's, restricted to the verdicts
    this estimator can actually reach: 0 = every leg weighed and PASSED;
    1 = a leg FAILED; 2 = fail-closed, nothing to weigh against (no
    budget declared and no card measurable) or a requested measurement
    this door cannot perform.  There is no code 4 here and inventing one
    would be hollow: the regional 4 separates "the itemized allocation
    gate passed but the machine envelope did not", and this estimator has
    one figure, which IS the envelope, so the two verdicts cannot part.
    """

    from woof.core.preflight import (EXTERNAL_MARGIN_BYTES,
                                      cap_free_to_physical,
                                      device_rail_free_bytes,
                                      device_wide_used_bytes)
    # Not from preflight: `measured_free_vram_bytes` is one of the symbols a
    # published engine does not carry yet, and the compat seam refuses by
    # name rather than measuring the card with a second instrument that
    # would disagree with the first.
    from .engine_compat import measured_free_vram_bytes
    from woof.local_gpu import no_local_gpu

    from .config import load_config

    cfg = load_config(args.config)
    estimate = estimate_global_memory(cfg)
    card_total_gib = getattr(args, "vram_gib", None)
    budget_gib = getattr(args, "budget_gib", None)
    machine = bool(getattr(args, "json", False))

    if getattr(args, "alloc", False):
        # A REQUESTED MEASUREMENT THAT WILL NOT HAPPEN, refused rather
        # than quietly downgraded to an estimate.  `--alloc` constructs
        # the regional experiment's persistent allocation set on the card
        # and reports measured-against-estimate; there is no such
        # construction for a spectral run short of building the transform
        # itself, which IS the expensive thing this preflight exists to
        # decide about.  Printing an estimate under a flag that promises a
        # measurement is how a report comes to be trusted for something it
        # never did.
        print("woof check: REFUSED (exit 2): --alloc has no measurement to "
              "make for a WOOF global config -- it constructs the "
              "regional experiment's device allocation set, and this model "
              "has none.  Building this configuration's transform IS the "
              "allocation, so the measurement is the run.",
              file=sys.stderr)
        print(f"  remedy: size it without the flag, then run it:\n"
              f"    woof check {args.config}\n"
              f"    woof global run {args.config}", file=sys.stderr)
        return 2

    # Read the card BEFORE anything in this process touches CUDA, exactly
    # as the regional preflight does: the residency the rail must respect
    # is other processes', and our own context would inflate it.
    rail_bytes = (None if getattr(args, "rail_mib", None) is None
                  else int(args.rail_mib) * 1024 ** 2)
    other_process_bytes = (None if rail_bytes is None
                           else device_wide_used_bytes())

    capped_to = None
    if budget_gib is not None:
        # DECLARED, never measured, and it must not print under the same
        # label as a measurement.  Unlike the regional door there is no
        # reserve to add back: this estimator's figure is an envelope, so
        # the declared budget IS the envelope budget and the free figure
        # it implies is that plus the other-process margin.
        free = int(budget_gib * GIB) + EXTERNAL_MARGIN_BYTES
        free_source = "declared (--budget-gib)"
        if card_total_gib is not None:
            free, capped_to = cap_free_to_physical(
                free, card_total_bytes=int(card_total_gib * GIB),
                measured_total_bytes=None)
        if capped_to is not None:
            free_source = ("declared (--budget-gib), capped at the card's "
                           "physical total")
    elif no_local_gpu():
        # THE DOCUMENTED NEVER-OPEN-THE-LOCAL-DEVICE SWITCH.  Reading free
        # VRAM means ``cudaMemGetInfo``, which cannot be asked without
        # standing a CUDA primary context up on the card this flag exists
        # to leave alone.  Unmeasurable, therefore, and unmeasurable fails
        # closed below with the declared-budget remedy -- never as a pass.
        free, free_source = None, "not measured (GPUWM_NO_LOCAL_GPU is set)"
    else:
        free, free_source, capped_to = measured_free_vram_bytes(
            card_total_gib=card_total_gib)

    rail = None
    if rail_bytes is not None:
        rail_free = device_rail_free_bytes(
            rail_bytes, other_process_bytes=other_process_bytes)
        rail = {"rail_bytes": rail_bytes,
                "other_process_bytes": other_process_bytes,
                "rail_free_bytes": rail_free}
        free = rail_free if free is None else min(int(free), rail_free)

    budget = _device_budget(free)
    gates = evaluate_global_gates(estimate, budget)
    lever = (None if budget is None or estimate.device_peak_bytes <= budget
             else largest_truncation_within(cfg, budget))

    if machine:
        _print_json(args, cfg, estimate, gates,
                    free=free, free_source=free_source, budget=budget,
                    capped_to=capped_to, rail=rail, lever=lever)
    else:
        _print_report(args, cfg, estimate, gates,
                      free=free, free_source=free_source, budget=budget,
                      capped_to=capped_to, rail=rail, lever=lever)

    if estimate.backend != "cupy":
        # NO CARD IS INVOLVED, so there is nothing to fail closed on.  The
        # numpy backend runs a complete forecast on the host, and refusing
        # it for an unmeasurable GPU would refuse the one path that needs
        # no GPU at all.
        return 0
    evaluable = [leg for leg in gates.values() if leg is not None]
    if not evaluable:
        print("woof check: REFUSED (exit 2, fail-closed): no leg could be "
              "weighed -- no VRAM budget was declared and no card could be "
              "measured in this machine (CuPy or a CUDA device is absent), "
              "so the estimate above has nothing to be verified against.",
              file=sys.stderr)
        print(f"  remedy: declare the card this config is sized for and "
              f"re-run:\n"
              f"    woof check {args.config} --budget-gib <N> "
              f"--vram-gib <card GiB>\n"
              f"  # this configuration's device peak is "
              f"{estimate.device_peak_bytes / GIB:.2f} GiB of pool and "
              f"{estimate_card_required_bytes(estimate) / GIB:.2f} "
              f"GiB of card; a budget at or above the card figure passes.",
              file=sys.stderr)
        return 2
    if not all(evaluable):
        return 1
    return 0


def _print_report(args, cfg, estimate, gates, *, free, free_source, budget,
                  capped_to, rail, lever) -> None:
    print(f"woof check: WOOF global memory preflight for "
          f"{estimate.name!r} (T{estimate.truncation}, "
          f"{estimate.nlon}x{estimate.nlat}x{estimate.nlev}, "
          f"{estimate.precision}, {estimate.backend} backend)")
    for flag in _ignored_regional_flags(args):
        print(f"  {flag}")
    if estimate.backend == "cupy":
        print(f"  free VRAM      {_format_bytes(free)}  ({free_source})")
        print(f"  device budget  {_format_bytes(budget)}  (free VRAM less "
              f"the {_format_bytes(_external_margin())} other-process "
              f"margin, which is all this estimate does not already model)")
        print(f"  device peak    {_format_bytes(estimate.device_peak_bytes)}"
              f"  ({prediction_sentence(estimate).split(' device peak predicted ')[1]})")
        print(f"  measures       {DEVICE_PEAK_MEASURES}")
        low, high = measured_fragmentation_range()
        # The ratio PRINTED is the ratio CHARGED, and the shape's own
        # radiation chunk decides which class that is: a page quoting the
        # default chunk's x1.1668 beside a figure built from the
        # 5,000-column class's x1.2020 is a sentence whose two halves do
        # not multiply out.
        shown, shown_evidence = pool_fragmentation_for(
            reduced_chunk=_reduced_chunk(estimate.radiation_column_chunk))
        print(f"  card required  "
              f"{_format_bytes(estimate_card_required_bytes(estimate))}"
              f"  (the device peak is the POOL's live bytes; a card also "
              f"holds the pool's fragmentation, x{shown:.4f} MEASURED "
              f"({shown_evidence}), and "
              f"{_format_bytes(OUT_OF_POOL_BYTES).strip()} outside the pool "
              f"MEASURED -- the context, the module images and the cuBLAS "
              f"and cuFFT workspaces).  A banded run, or one with the "
              f"pinned host tier, is charged its own class at the door")
        print(f"  fragmentation  x{low:.4f} to x{high:.4f} MEASURED over the "
              f"runs in POOL_FRAGMENTATION_MEASURES; the planning value is "
              f"the largest, because the term exists to stop a run being "
              f"admitted that the card cannot hold")
        ceiling = fitted_truncation_ceiling()
        if int(estimate.truncation) > ceiling:
            print(f"  UNFITTED: T{estimate.truncation} is above T{ceiling}, "
                  f"the largest truncation this model has been measured at. "
                  f"`woof check` still prices it; the run doors REFUSE it, "
                  f"and one ten-step probe of this shape added to "
                  f"DEVICE_PEAK_CALIBRATION is what lifts that")
    else:
        # NO ``device peak 0.00 GiB`` LINE HERE.  The itemization below is
        # real -- those arrays are built, on the host -- and heading it
        # with a zero device figure reads as "this configuration costs
        # nothing", which is the opposite of what the rows say.
        print("  NO DEVICE VERDICT: arwen_global.backend = 'numpy' runs the "
              "CPU reference path, which opens no CUDA context and puts "
              "nothing on a card.  Every term below is HOST memory.")
        print(f"  host peak      {_format_bytes(estimate.host_peak_bytes)}")
    for label, nbytes, why in estimate.itemization():
        print(f"    {label:<25s} {_format_bytes(nbytes)}   {why}")
    if estimate.backend == "cupy":
        print(f"  host peak      "
              f"{_format_bytes(estimate.host_transform_peak_bytes)}  "
              f"(building the Legendre tables one band at a time; the "
              f"recurrence runs in float64 whatever the device precision "
              f"is -- so this figure does NOT halve with "
              f"precision='float32')")
        for row in calibration_residuals():
            print(f"  calibration    {row['label']}: measured "
                  f"{_format_bytes(row['measured_bytes']).strip()}, model "
                  f"{_format_bytes(row['predicted_bytes']).strip()} "
                  f"({100 * row['residual_fraction']:+.1f}%)")
        for row in held_out_residuals():
            if row["residual_fraction"] is None:
                print(f"  held out       {row['label']}: not a number (the "
                      "only evidence for a term)")
            else:
                print(f"  held out       {row['label']}: refit without this "
                      f"shape reads "
                      f"{_format_bytes(row['held_out_predicted_bytes']).strip()} "
                      f"({100 * row['residual_fraction']:+.1f}%)")
        for row in MEASURED_RUNS:
            print(f"  completed      {row['label']}: "
                  f"{_format_bytes(row['peak_used_bytes']).strip()} of pool "
                  f"on {row['device']} ({row['measured_on']}), at a recipe "
                  "the calibration table cannot carry")
    if capped_to is not None:
        print(f"  CAPPED: the declared free figure exceeded the card's "
              f"physical total and was clamped to "
              f"{_format_bytes(capped_to)}; free VRAM cannot exceed the card")
    if rail is not None:
        print(f"  DEVICE RAIL {_format_bytes(rail['rail_bytes'])} "
              f"whole-machine; other processes hold "
              f"{_format_bytes(rail['other_process_bytes'])}, leaving "
              f"{_format_bytes(rail['rail_free_bytes'])} for this run")
    if estimate.backend != "cupy":
        return
    for metric in GLOBAL_GATE_METRICS:
        print(f"  {GATE_DISPLAY[metric]}: {_leg_text(gates[metric])}")
    if gates["arwen_global_legendre_tables_fit_device"] is False:
        # THE FIRST ALLOCATION, and the one no vertical trim can move.
        print(f"  OVER BUDGET AT CONSTRUCTION: the Legendre tables alone are "
              f"{_format_bytes(estimate.legendre_table_bytes)} against a "
              f"{_format_bytes(budget)} budget, and the transform builds "
              f"them before one prognostic field exists -- this "
              f"configuration dies in "
              f"SphericalHarmonicTransform.__post_init__, not at step 1.")
        print("  The tables are 3 x (T+1)(T+2)/2 x nlat plus their band "
              "scratch: CUBIC in truncation, and vertical levels do not "
              "appear in them at all.  Trimming [vertical] cannot close "
              "this one.")
    elif gates["arwen_global_step_peak_fits_device"] is False:
        card = estimate_card_required_bytes(estimate)
        print(f"  OVER BUDGET: the device peak "
              f"{_format_bytes(estimate.device_peak_bytes)} of pool is "
              f"{_format_bytes(card)} of card, which exceeds the "
              f"{_format_bytes(budget)} budget by "
              f"{_format_bytes(card - budget)}.")
    if gates["arwen_global_step_peak_fits_device"] is False:
        _print_levers(cfg, estimate, budget, lever)


def _print_levers(cfg, estimate, budget, lever) -> None:
    if lever is not None:
        print(f"  remedy (first lever, grid.truncation): T{lever} fits this "
              f"budget at {estimate.nlev} levels in {estimate.precision} "
              f"-- set [grid] truncation = {lever}")
    if estimate.precision == "float64":
        print("  remedy (second lever, precision): "
              "[arwen_global] precision = \"float32\" halves every device "
              "term above.  It moves the config identity hash and needs its "
              "own transform gate to pass, which the run checks for itself.")
    else:
        print("  remedy (second lever, vertical levels): the spectral state, "
              "the resident grid fields, the working set and the physics "
              "column workspace are all linear in nlev; the Legendre tables "
              "are not, so the floor this lever can reach is "
              f"{_format_bytes(estimate.legendre_table_bytes)}.")
    if lever is None:
        print("  no truncation this loader admits (>= 3) fits the budget: "
              "the card, not the configuration, is what has to change")


def _external_margin() -> int:
    from woof.core.preflight import EXTERNAL_MARGIN_BYTES

    return EXTERNAL_MARGIN_BYTES


def _ignored_regional_flags(args) -> list[str]:
    """One line per regional-only flag this config cannot honour.

    Silence would be the defect: ``--column-chunk`` names the RRTMGP
    column workspace and ``--reserve-gib`` the regional allocation
    reserve, and neither exists in a spectral run.  A reader who typed one
    and saw an unchanged number would reasonably conclude the lever did
    not help, rather than that it was never read.
    """

    said = []
    if getattr(args, "column_chunk", None) is not None:
        said.append("--column-chunk is not read here: it sizes the RRTMGP "
                    "column workspace of the regional model; this model's "
                    "radiation chunk is [physics] native_adapter_options "
                    "radiation_column_chunk, which the estimate reads from "
                    "the config")
    if getattr(args, "reserve_gib", None) is not None:
        said.append("--reserve-gib is not read here: this estimate is an "
                    "envelope and is weighed against free VRAM less the "
                    "other-process margin, with no allocation reserve to "
                    "override")
    return said


def calibration_report() -> dict[str, object]:
    """The calibration as the JSON report and the receipt carry it."""

    return {
        "measures": DEVICE_PEAK_MEASURES,
        "points": [dict(point) for point in DEVICE_PEAK_CALIBRATION],
        "fitted": dict(DEVICE_PEAK_FIT),
        "residuals": [dict(row) for row in calibration_residuals()],
        "worst_residual_fraction": worst_calibration_residual_fraction(),
        "shapes": calibration_shapes(),
        "held_out_residuals": [dict(row) for row in held_out_residuals()],
        "worst_held_out_residual_fraction": worst_held_out_residual_fraction(),
        "completed_runs": [dict(row) for row in MEASURED_RUNS],
        "numpy_working_grid_arrays": NUMPY_WORKING_GRID_ARRAYS,
    }


def _print_json(args, cfg, estimate, gates, *, free, free_source, budget,
                capped_to, rail, lever) -> None:
    payload = {
        "config": str(args.config),
        "config_family": "arwen_global",
        "experiment": estimate.name,
        "config_hash": cfg.config_hash,
        "backend": estimate.backend,
        "precision": estimate.precision,
        "truncation": estimate.truncation,
        "nlat": estimate.nlat, "nlon": estimate.nlon, "nlev": estimate.nlev,
        "dealias_factor": estimate.dealias_factor,
        "radiation_column_chunk": estimate.radiation_column_chunk,
        "cumulus_column_chunk": estimate.cumulus_column_chunk,
        "legendre_table_bytes": estimate.legendre_table_bytes,
        "legendre_build_bytes": estimate.legendre_build_bytes,
        "spectral_state_bytes": estimate.spectral_state_bytes,
        "surface_grid_bytes": estimate.surface_grid_bytes,
        "resident_grid_bytes": estimate.resident_grid_bytes,
        "memo_grid_bytes": estimate.memo_grid_bytes,
        "working_grid_bytes": estimate.working_grid_bytes,
        "physics_column_bytes": estimate.physics_column_bytes,
        "cumulus_column_bytes": estimate.cumulus_column_bytes,
        "fixed_bytes": estimate.fixed_bytes,
        "trajectory_bytes": estimate.trajectory_bytes,
        "semilag_gather_bytes": estimate.semilag_gather_bytes,
        "resident_bytes": estimate.resident_bytes,
        "device_peak_bytes": estimate.device_peak_bytes,
        "device_peak_measures": DEVICE_PEAK_MEASURES,
        "card_required_bytes": estimate_card_required_bytes(estimate),
        "pool_fragmentation": POOL_FRAGMENTATION,
        "out_of_pool_bytes": OUT_OF_POOL_BYTES,
        "fitted_truncation_ceiling": fitted_truncation_ceiling(),
        "above_fitted_domain": above_fitted_domain(cfg, estimate.nlev),
        "extrapolation": list(estimate.extrapolation),
        "host_transform_peak_bytes": estimate.host_transform_peak_bytes,
        "host_peak_bytes": estimate.host_peak_bytes,
        "measured_free_bytes": free,
        "free_bytes_source": free_source,
        "free_bytes_capped_to_physical_bytes": capped_to,
        "sized_for_hardware_not_present": (
            getattr(args, "budget_gib", None) is not None),
        "device_budget_bytes": budget,
        "external_margin_bytes": _external_margin(),
        "gates": gates,
        "calibration": calibration_report(),
        "largest_truncation_within_budget": lever,
    }
    if rail is not None:
        payload["device_rail"] = rail
    notes = _ignored_regional_flags(args)
    if notes:
        payload["flags_not_read"] = notes
    print(json.dumps(payload, indent=2))


__all__ = [
    "BUILD_LEGENDRE_TABLES", "CALIBRATED_DOMAIN", "CUMULUS_COLUMN_ARRAYS",
    "OUT_OF_POOL_BYTES", "OUT_OF_POOL_MEASURES", "POOL_FRAGMENTATION",
    "POOL_FRAGMENTATION_MEASURES", "RunMemoryPlan",
    "above_fitted_domain", "pool_fragmentation_for", "worst_model_under_read",
    "SPILL_RELIEF", "SPILL_RELIEF_MEASURES", "spill_relief_factor",
    "MEASURED_RUNS", "measured_runs_at", "COMPLETED_PLAN_PEAKS",
    "completed_plan_ceiling", "completed_plan_is_exact", "SPILL_CENSUS_FACTOR",
    "SPILL_CENSUS_MEASURES",
    "card_required_bytes", "fitted_truncation_ceiling",
    "measured_fragmentation_range", "plan_run_memory",
    "CUPY_WORKING_GRID_ARRAYS",
    "DEVICE_LEGENDRE_TABLES", "DEVICE_PEAK_CALIBRATION", "DEVICE_PEAK_FIT",
    "DEVICE_PEAK_MEASURES", "FIXED_DEVICE_BYTES", "GATE_DISPLAY",
    "GLOBAL_GATE_METRICS", "GlobalMemoryEstimate", "GlobalMemoryRefusal",
    "HOST_BAND_BLOCKS", "HOST_SOLVE_TEMPORARIES", "LIVE_SPECTRAL_STACKS",
    "NUMPY_WORKING_GRID_ARRAYS", "PHYSICS_COLUMN_ARRAYS",
    "RESIDENT_GRID_ARRAYS", "RESIDENT_SPECTRAL_STACKS", "SPILL_SLICES",
    "spill_census", "spill_slices_for", "native_namespace_bytes",
    "estimate_card_required_bytes",
    "PROBE_CALIBRATION", "PROBE_RECIPE",
    "SURFACE_GRID_ARRAYS", "calibration_report", "calibration_residuals",
    "estimate_global_memory", "evaluate_global_gates",
    "extrapolation_notes", "fit_device_peak_model", "global_check_main",
    "host_transform_transient_bytes", "largest_truncation_within",
    "legendre_build_bytes", "legendre_runtime_bytes",
    "predict_device_peak_bytes", "prediction_sentence",
    "probe_calibration_residuals", "refusal_sentence",
    "run_memory_gate", "worst_calibration_residual_fraction",
    "calibration_shapes", "held_out_residuals",
    "worst_held_out_residual_fraction",
]
