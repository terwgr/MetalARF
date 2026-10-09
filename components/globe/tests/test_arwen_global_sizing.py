"""Sizing verdicts for the global spectral model, and the doors they reach.

The model could be configured to want more VRAM than any card in the world
and nothing would say so until CuPy raised ``OutOfMemoryError`` from inside
``SphericalHarmonicTransform.__post_init__`` -- after the config had been
accepted, the output directory prepared, and (on the analysis route) a GRIB
analysis decoded.  `woof check` could not even be pointed at one of these
configs: it died in ``ValueError: unknown table(s)/top-level key(s)
['arwen_global', 'grid', ...]``, a traceback where a verdict belongs.

Three things are pinned here and they are different in kind:

* THE MODEL IS CORRECT ABOUT REAL ARRAYS.  ``test_the_estimate_matches_a
  _really_built_model`` builds the real transform and the real cold state
  on the numpy backend and weighs them with ``tracemalloc`` filtered to
  numpy's own allocation domain.  That is the measurement the estimator's
  structural coefficients were fitted to, re-taken every run, so a change
  to the field set or the surface state that moves the footprint fails
  here instead of silently mis-pricing a card.
* THE CALIBRATION HOLDS.  The device figure is a model FITTED to
  allocator-measured pool peaks (``sizing.DEVICE_PEAK_CALIBRATION``: T63,
  T127 and T255 at 40 and 20 levels and at two radiation chunks, from
  receipts written by the runner's allocator hook), and
  ``test_the_model_reads_every_calibration_point_within_its_stated_residual``
  is what stops a coefficient edit -- or a structural edit that does not
  re-fit -- from quietly moving the model off the measurements.  The
  table must hold more distinct shapes than the fit has coefficients,
  and the held-out residual (each shape priced by a refit without it) is
  the accuracy the door states for a shape outside the table: a table of
  four shapes for four coefficients read 0.0003 percent by construction
  and 17 percent at the first shape measured outside it.  The retired anchor (a sampled 7.1 GiB at T533
  that read 2.21 GiB at T255 where the allocator measures 8.51) is pinned
  as retired.
* THE VERDICTS ARE VERDICTS.  Exit codes, a leg that cannot be weighed
  never passing, and every refusal naming the concrete breakage: which
  allocation dies first, and what to type instead.
"""

from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import argparse
import json
from pathlib import Path
import sys
import tracemalloc

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from woof import cli
from woof.globe import cli as arwen_cli
from woof.globe import sizing
from woof.globe.config import load_config
from woof.globe.runner import build_model_and_cold_state, build_transform
from woof.core import preflight as core_preflight
from woof.experiment import is_experiment_toml_bytes

# The global route sniff is patch item 08 of the series the carve wrote for
# the engine: `woof/experiment.py` on a published 2.7 carries the experiment
# and legacy sniffs beside each other and not this one.  Imported this way so
# the rest of this module still collects; a module-scope ImportError here is a
# COLLECTION error, which abandons the whole run and reports nothing.
try:
    from woof.experiment import is_arwen_global_toml_bytes
except ImportError:  # pragma: no cover - depends on the installed engine
    is_arwen_global_toml_bytes = None

needs_the_global_route_sniff = pytest.mark.skipif(
    is_arwen_global_toml_bytes is None,
    reason="the installed woof's experiment module does not carry "
           "is_arwen_global_toml_bytes: patch item 08 of the series the "
           "carve wrote for the engine")
from woof.globe.spectral.grid import GaussianGrid
from woof.globe.spectral.transform import SphericalHarmonicTransform

#: The engine's `check` door routes a config to a loader by SNIFFING it, and
#: the sniff that recognises a global config -- `is_arwen_global_toml_bytes`
#: -- is patch item 08 of the series the carve wrote for the engine.  Without
#: it, `woof check` hands a global TOML to the regional loader and refuses it
#: as nine unknown tables.  That is an engine gap, not a defect in this
#: package's sizer: everything under `woof global run` prices the same card
#: through the same code and is proved above.
_ENGINE_ROUTES_GLOBAL_CONFIGS = is_arwen_global_toml_bytes is not None

_needs_the_engine_sniff = pytest.mark.skipif(
    not _ENGINE_ROUTES_GLOBAL_CONFIGS,
    reason="the installed woof's check door cannot route a global config: "
           "is_arwen_global_toml_bytes is patch item 08 of the series the "
           "carve wrote for the engine")


SMOKE_CONFIG = _shipped_configs() / "arwen_global_moist_smoke.toml"
QUICKSTART_CONFIG = (_shipped_configs() / "arwen_global_t255_quickstart.toml")
GIB = 1024 ** 3


def _check_args(config, **overrides) -> argparse.Namespace:
    """The namespace ``woof check`` builds, with this door's defaults."""

    values = dict(
        config=Path(config), alloc=False, column_chunk=None,
        reserve_gib=None, budget_gib=None, vram_gib=None, rail_mib=None,
        forcing_interval_s=core_preflight.DEFAULT_FORCING_INTERVAL_SECONDS,
        json=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def _retruncate(text: str, truncation: int) -> str:
    return text.replace("truncation = 3", f"truncation = {truncation}")


# --- the model against really built arrays ---------------------------------

def _numpy_arrays_held() -> int:
    """Live numpy array bytes right now, and nothing else.

    ``tracemalloc``'s numpy domain is the instrument: numpy registers its
    buffer allocations there, so this counts arrays without counting the
    Python objects around them.
    """

    domain = np.lib.tracemalloc_domain
    snapshot = tracemalloc.take_snapshot().filter_traces(
        [tracemalloc.DomainFilter(True, domain)])
    return sum(stat.size for stat in snapshot.statistics("filename"))


def test_the_estimate_matches_a_really_built_model(tmp_path):
    """The structural terms price arrays that are actually allocated.

    Built and STEPPED, not stubbed: the real ``build_transform``,
    ``build_model_and_cold_state`` and ``MoistHybridModel.step``, weighed
    through numpy's own allocation domain.  This is the measurement the
    coefficients were fitted to, re-taken every run, so an edit to the
    field set or the surface state that moves the footprint fails here
    rather than silently mis-pricing a card.

    T31 with eight levels rather than the T3 smoke config: at T3 the
    tables are 2,304 bytes and the transform's fixed small vectors
    (latitudes, the eigenvalue row, the triangle mask) are half the
    figure, so the test would be measuring the terms this model
    deliberately does not carry.
    """

    config = tmp_path / "t31.toml"
    config.write_text(
        _retruncate(SMOKE_CONFIG.read_text(encoding="utf-8"), 31)
        .replace("nlev = 4", "nlev = 8")
        .replace("zonal_wavenumber = 2", "zonal_wavenumber = 2"),
        encoding="utf-8")
    cfg = load_config(config)

    tracemalloc.start()
    transform = build_transform(cfg)
    tables = _numpy_arrays_held()
    model, cold = build_model_and_cold_state(cfg, transform)
    # BOTH stacks stay referenced, which is what `runner.run` does: it
    # keeps `cold` for the whole forecast (its diagnostics are the drift
    # gates' reference) beside the state it is advancing.
    state, _ = model.step(cold, cfg.dt_s)
    resident = _numpy_arrays_held()
    assert state.step == 1
    tracemalloc.stop()

    estimate = sizing.estimate_global_memory(cfg)
    # Construction leaves the basis and the analysis table; the first
    # step's gradient adds the derivative and the expansion scratches, so
    # the runtime tables term is what the stepped resident set carries.
    assert estimate.legendre_build_bytes == pytest.approx(tables, rel=0.05)
    assert estimate.resident_bytes == pytest.approx(resident, rel=0.05)


def test_the_grid_shape_is_the_one_the_transform_builds():
    """No second copy of the dealias formulas.

    An estimator with its own ``nlat``/``nlon`` arithmetic prices a grid
    the run does not make, and a dealias change moves one and not the
    other.  Both come from :meth:`GaussianGrid.shape_for`.
    """

    for truncation in (3, 21, 42):
        built = SphericalHarmonicTransform.create(truncation, backend="numpy")
        assert GaussianGrid.shape_for(truncation) == (built.grid.nlat,
                                                      built.grid.nlon)


def test_the_legendre_table_count_is_the_transform_s():
    """Two packed tables at build, three plus scratch at run time, and a
    float64 construction transient the host pays whatever the precision.

    Term 1 is not a fitted coefficient: it is the transform's own byte
    count (``legendre_table_nbytes``), and this pins the estimator's
    formulas to it at T63 in both precisions.  The host transient is the
    one figure that does NOT halve with ``precision='float32'``: the
    recurrence and the Gram solve run in float64 on the host, so a reader
    sizing host RAM off the device figure would be short.  The transient
    is priced conservatively (within a third, never under), measured
    2026-09-01 at T63/T85/T127.
    """

    truncation = 63
    nlat = GaussianGrid.shape_for(truncation)[0]
    for precision, itemsize in (("float64", 8), ("float32", 4)):
        tracemalloc.start()
        tracemalloc.reset_peak()
        built = SphericalHarmonicTransform.create(truncation, backend="numpy",
                                                  precision=precision)
        resident, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert built.truncation == truncation
        held = built.legendre_table_nbytes
        assert held["derivative"] == 0 and held["scratch"] == 0
        assert sizing.legendre_build_bytes(truncation, nlat, itemsize) == (
            held["basis"] + held["analysis"])
        assert resident == pytest.approx(held["basis"] + held["analysis"], rel=0.05)

        # A step analyses, synthesises and takes gradients: all three
        # tables and all three expansion scratches exist after it.
        built.forward(built.inverse(built.zeros()))
        built.gradient(built.zeros())
        held = built.legendre_table_nbytes
        assert sizing.legendre_runtime_bytes(truncation, nlat, itemsize) == (
            held["basis"] + held["analysis"] + held["derivative"] + held["scratch"])

        transient = peak - resident
        priced = sizing.host_transform_transient_bytes(truncation, nlat)
        assert transient <= priced
        if precision == "float32":
            # Under float64 the recurrence's band IS the resident block
            # (no cast, no copy), so the transient only shows on its own
            # where the resident tables are float32.
            assert transient >= 0.6 * priced


# --- the calibration -------------------------------------------------------

#: The calibrated model's stated IN-SAMPLE accuracy: every calibration
#: point reads back within this fraction of its allocator-measured peak
#: (7.0 percent over the twelve rows of 2026-09-05, T127 at a
#: 5,000-column chunk the worst), and the door prints the worst residual
#: beside every prediction.
STATED_RESIDUAL = 0.08

#: The stated HELD-OUT accuracy: every shape, priced by a refit of the
#: model without that shape's rows, reads within this fraction (16.1
#: percent over the ten shapes of 2026-09-05).  This, not the in-sample
#: figure, is what a truncation above the table is labelled with.
STATED_HELD_OUT_RESIDUAL = 0.20


def test_the_model_reads_every_calibration_point_within_its_stated_residual(
        tmp_path):
    """The fit, end to end through the real config loader.

    Each row of ``DEVICE_PEAK_CALIBRATION`` is rebuilt as a config the
    loader accepts (truncation, levels, precision, radiation chunk) and
    priced by ``estimate_global_memory``; the figure must sit within the
    stated residual of the receipt's ``peak_used_bytes``.  A structural
    coefficient edited without re-fitting, or a fit that no longer
    explains its own table, fails here.
    """

    for point in sizing.DEVICE_PEAK_CALIBRATION:
        config = tmp_path / f"{point['label'].replace(' ', '_')}.toml"
        config.write_text(
            _retruncate(SMOKE_CONFIG.read_text(encoding="utf-8"),
                        point["truncation"])
            .replace('backend = "numpy"', 'backend = "cupy"')
            .replace('precision = "float64"', f'precision = "{point["precision"]}"')
            .replace("nlev = 4", f"nlev = {point['nlev']}")
            .replace("transform_roundtrip_relative_linf = 5.0e-11",
                     "transform_roundtrip_relative_linf = 5.0e-5")
            .replace("transform_parseval_relative_error = 5.0e-12",
                     "transform_parseval_relative_error = 5.0e-5"),
            encoding="utf-8")
        import dataclasses

        cfg = dataclasses.replace(
            load_config(config), physics_mode="arwen-native",
            native_adapter_options={
                "radiation_column_chunk": point["radiation_column_chunk"],
                "cumulus_column_chunk": point["cumulus_column_chunk"]})
        estimate = sizing.estimate_global_memory(cfg)
        assert (estimate.nlat, estimate.nlon, estimate.nlev) == (
            point["nlat"], point["nlon"], point["nlev"]), point["label"]
        assert estimate.radiation_column_chunk == point["radiation_column_chunk"]
        assert estimate.cumulus_column_chunk == point["cumulus_column_chunk"]
        assert estimate.device_peak_bytes == pytest.approx(
            point["peak_used_bytes"], rel=STATED_RESIDUAL), point["label"]
    assert sizing.worst_calibration_residual_fraction() <= STATED_RESIDUAL


def test_the_fit_is_over_determined_and_says_what_it_reads_held_out():
    """Four coefficients need more than four distinct shapes, or the
    residual the door prints is an identity: the table of the morning of
    2026-09-05 held six rows over four shapes and read 0.0003 percent at
    every one by construction, then 17 percent at the first new shape.
    The held-out residual is the accuracy a shape outside the table can
    expect, every shape must read within its stated bound, and the door's
    sentence carries both figures."""

    shapes = sizing.calibration_shapes()
    assert shapes > 4, shapes
    assert shapes >= 8
    rows = sizing.held_out_residuals()
    assert len(rows) == len(sizing.DEVICE_PEAK_CALIBRATION)
    priced = [row for row in rows if row["residual_fraction"] is not None]
    assert priced, "no shape could be held out"
    for row in priced:
        assert abs(row["residual_fraction"]) <= STATED_HELD_OUT_RESIDUAL, row
    worst = sizing.worst_held_out_residual_fraction()
    assert worst is not None
    assert worst >= sizing.worst_calibration_residual_fraction()
    # Chunks and level counts are both represented, so neither term is
    # carried by one shape.
    chunks = {int(point["radiation_column_chunk"])
              for point in sizing.DEVICE_PEAK_CALIBRATION}
    levels = {int(point["nlev"]) for point in sizing.DEVICE_PEAK_CALIBRATION}
    assert len(chunks) >= 2 and len(levels) >= 2

    import dataclasses

    base = dataclasses.replace(
        load_config(SMOKE_CONFIG), backend="cupy", precision="float32",
        physics_mode="arwen-native")
    inside = sizing.estimate_global_memory(dataclasses.replace(
        base, truncation=63, nlat=None, nlon=None, zonal_wavenumber=2,
        diffusion_preserve_degree=1))
    sentence = sizing.prediction_sentence(inside)
    assert "in sample and" in sentence and "held out over" in sentence
    assert f"of {shapes} shapes" in sentence


def test_the_t533_floor_is_the_dynamics_floor_of_the_chunked_preparation():
    """The T533 probe of the morning of 2026-09-05 reached 19.25 GiB inside
    its first radiation call before a 20 GiB pool cap stopped it.  With the
    cloud preparation chunked (woof.globe.core.rrtmgp, the afternoon of
    2026-09-05) the radiation call returns at 17.12 GiB and the same probe
    died in the first step's dynamics under the 20,992 MiB cap with 18.56
    GiB live: the pool's fragmentation, not the live bytes.  T533 stays a
    floor, the floor names the dynamics, no T533 row exists, the model's
    T533 figure sits above the floor, and the extrapolation note carries
    it.  Every row of the table was re-measured on the chunked
    preparation the same afternoon."""

    import dataclasses

    # THE FLOOR IS SUPERSEDED, and that is the point of this assertion
    # now: T533 L40 has since RUN to completion on an RTX 5090, twice, so
    # a bound taken from a capped probe that died is the weaker evidence
    # and the reader is given the completed run instead.  The row stays in
    # the table as the record of the cap.
    assert sizing.measured_floor_for(533, 40, "float32") is None
    completed = sizing.measured_runs_at(533, 40, "float32")
    assert completed and all(int(row["peak_used_bytes"]) > 0
                             for row in completed)
    floor = next(row for row in sizing.MEASURED_FLOORS
                 if int(row["truncation"]) == 533)
    assert floor["reached_bytes"] < floor["pool_cap_bytes"]
    # The merged tip of 2026-09-05 under a 22,528 MiB cap on an empty card
    # passed the dynamics and died in the tracer transport at 20.74 GiB live
    # asking 1.66 GiB: the floor the table carries is that one.
    assert floor["pool_cap_bytes"] == 22_528 * 1024 ** 2
    assert "tracer transport" in floor["where"] and "17.12 GiB" in floor["where"]
    assert floor["reached_bytes"] == 22_268_853_248
    assert all(int(point["truncation"]) != 533
               for point in sizing.DEVICE_PEAK_CALIBRATION)
    # Every row is measured on the chunked cloud preparation or later.
    # The bound is a date rather than one date because a refit adds rows:
    # what must not happen is a row measured BEFORE the preparation was
    # chunked, which would be a different model of the radiation call.
    assert all(point["measured_on"] >= "2026-09-05"
               for point in sizing.DEVICE_PEAK_CALIBRATION)
    cfg = dataclasses.replace(
        load_config(SMOKE_CONFIG), backend="cupy", precision="float32",
        physics_mode="arwen-native", truncation=533, nlat=None, nlon=None,
        native_adapter_options={"radiation_column_chunk": 5000})
    notes = sizing.extrapolation_notes(cfg, 40)
    assert any("held-out error" in note for note in notes)
    assert any("this shape HAS run" in note for note in notes)
    assert not any("a capped probe of this shape" in note for note in notes)
    # The model still has to sit above the floor the capped probe reached
    # and above the completed runs, which is why both are kept: a model
    # reading under either would be the defect the fitted-domain refusal
    # exists for.
    predicted = sizing.predict_device_peak_bytes(
        533, 801, 1602, 40, 4, 5000, 131_072)
    assert predicted > floor["reached_bytes"]
    for row in completed:
        at_its_own_chunk = sizing.predict_device_peak_bytes(
            533, 801, 1602, 40, 4, int(row["radiation_column_chunk"]),
            131_072)
        assert at_its_own_chunk >= int(row["peak_used_bytes"])
    assert sizing.measured_floor_for(255, 40, "float32") is None


def test_a_capped_probe_is_a_floor_the_door_prints_not_a_row(monkeypatch):
    """The floor mechanism, on a planted shape: a probe a pool cap stopped
    is a floor on that shape's peak, not its peak.  It must not be a
    calibration row, the model's figure for the shape must sit above it,
    and the extrapolation note must carry it."""

    import dataclasses

    planted = {
        "truncation": 639, "nlat": 960, "nlon": 1920, "nlev": 40,
        "precision": "float32", "radiation_column_chunk": 5_000,
        "cumulus_column_chunk": 131_072,
        "reached_bytes": 12 * 1024 ** 3, "pool_cap_bytes": 14 * 1024 ** 3,
        "where": "inside a planted call,",
        "measured_on": "2026-09-05", "device": "planted",
        "run": "a planted capped probe",
    }
    monkeypatch.setattr(sizing, "MEASURED_FLOORS", (planted,))
    floor = sizing.measured_floor_for(639, 40, "float32")
    assert floor is not None
    assert floor["reached_bytes"] < floor["pool_cap_bytes"]
    assert all(int(point["truncation"]) != 639
               for point in sizing.DEVICE_PEAK_CALIBRATION)
    cfg = dataclasses.replace(
        load_config(SMOKE_CONFIG), backend="cupy", precision="float32",
        physics_mode="arwen-native", truncation=639, nlat=None, nlon=None,
        native_adapter_options={"radiation_column_chunk": 5000})
    notes = sizing.extrapolation_notes(cfg, 40)
    assert any("held-out error" in note for note in notes)
    assert any("12.00 GiB" in note and "at least that" in note
               for note in notes)
    predicted = sizing.predict_device_peak_bytes(
        639, 960, 1920, 40, 4, 5000, 131_072)
    assert predicted > floor["reached_bytes"]


#: What a probe row of the calibration table is trusted to: the ten-step
#: probe of record reads the full day's peak within this fraction, on
#: every pair the instrument was held against.
PROBE_RESIDUAL = 0.02


def test_the_probe_of_record_reads_the_full_day_on_every_pair():
    """The instrument behind every probe row, held in both directions.

    ``PROBE_CALIBRATION`` pairs the ten-step probe with a full day on
    more than one truncation; each pair must agree within
    :data:`PROBE_RESIDUAL`, and the retired single-radiation probe must
    still read as the under-reader it was on every pair (18.7 percent
    under the T255 day, 3.1 under the T63 day: the gap grows with the
    grid, which is why the single-radiation T63 rows looked plausible),
    so a table row measured that way can never be mistaken for one of
    record.
    """

    rows = sizing.probe_calibration_residuals()
    assert len(rows) >= 2, "the probe is held on at least two truncations"
    assert len({row["label"].split(",")[0] for row in rows}) >= 2
    for row in rows:
        assert abs(row["residual_fraction"]) <= PROBE_RESIDUAL, row
    retired = {row["label"]: row["single_radiation_probe_fraction"]
               for row in rows if row["single_radiation_probe_fraction"] is not None}
    assert retired and all(fraction < -PROBE_RESIDUAL
                           for fraction in retired.values()), retired
    assert retired["T255, 40 levels"] < -0.15
    assert "step 0 and again at step 6" in sizing.PROBE_RECIPE
    probe_rows = [point for point in sizing.DEVICE_PEAK_CALIBRATION
                  if "probe" in point["label"]]
    assert probe_rows and all("probe of record" in point["run"]
                              for point in probe_rows)


def test_the_calibrated_terms_are_solved_from_the_table_not_typed():
    """Move a measurement, and the coefficients move with it.

    A hand-copied constant beside a measurement that had been re-taken is
    exactly the stale-number failure this tree's ledger rules exist to
    stop, so the three fitted terms are solved from
    :data:`DEVICE_PEAK_CALIBRATION` at import rather than written down,
    and the retired anchor's derivation refuses by name.
    """

    fit = sizing.fit_device_peak_model()
    assert fit == sizing.DEVICE_PEAK_FIT
    assert sizing.CUPY_WORKING_GRID_ARRAYS == fit["working_grid_arrays"] > 0
    assert sizing.PHYSICS_COLUMN_ARRAYS == fit["physics_column_arrays"] > 0
    assert sizing.CUMULUS_COLUMN_ARRAYS == fit["cumulus_column_arrays"] > 0
    assert sizing.FIXED_DEVICE_BYTES == int(round(fit["fixed_bytes"])) >= 0

    moved = tuple(
        dict(point, peak_used_bytes=int(point["peak_used_bytes"] * 1.5))
        for point in sizing.DEVICE_PEAK_CALIBRATION)
    refit = sizing.fit_device_peak_model(moved)
    assert refit["working_grid_arrays"] > fit["working_grid_arrays"]

    with pytest.raises(RuntimeError, match="retired"):
        sizing._anchor_working_grid_arrays()


def test_the_model_says_what_it_measures_and_when_it_extrapolates():
    """A figure without its measurement sentence is the sampled 7.1 GiB
    all over again; a figure outside the fitted domain says so."""

    import dataclasses

    base = dataclasses.replace(
        load_config(SMOKE_CONFIG), backend="cupy", precision="float32",
        physics_mode="arwen-native")
    inside = sizing.estimate_global_memory(dataclasses.replace(
        base, truncation=63, nlat=None, nlon=None, zonal_wavenumber=2,
        diffusion_preserve_degree=1))
    # T63 at the smoke config's four levels: the level count is outside
    # the measured set, and nothing else is.
    assert inside.extrapolation == (
        sizing.extrapolation_notes(base, 4)) and any(
        "nlev = 4" in note for note in inside.extrapolation)
    sentence = sizing.prediction_sentence(inside)
    assert "device peak predicted (calibrated model" in sentence
    assert "worst calibration residual" in sentence
    assert "EXTRAPOLATED" in sentence

    float64 = sizing.estimate_global_memory(
        dataclasses.replace(base, precision="float64"))
    assert any("float64" in note for note in float64.extrapolation)
    reference = sizing.estimate_global_memory(
        dataclasses.replace(base, physics_mode="reference"))
    assert any("physics.mode" in note for note in reference.extrapolation)
    assert "allocator" in sizing.DEVICE_PEAK_MEASURES


def test_the_retired_sizing_line_is_gone():
    """2.21 GiB at T255 where the allocator measures 8.51 GiB: the figure
    the doors printed until 2026-09-05.  The T255 production shape must
    now price within the stated residual of the control's receipt."""

    cfg = load_config(QUICKSTART_CONFIG)
    estimate = sizing.estimate_global_memory(cfg)
    control = next(point for point in sizing.DEVICE_PEAK_CALIBRATION
                   if point["label"].startswith("T255"))
    assert estimate.device_peak_bytes == pytest.approx(
        control["peak_used_bytes"], rel=STATED_RESIDUAL)
    assert estimate.device_peak_bytes > 3 * 2376043592


def test_the_tables_term_is_cubic_and_carries_no_levels():
    """The claim the refusal prose makes, checked rather than asserted.

    "Trimming [vertical] cannot close a tables-bound refusal" is only
    true if levels really do not appear in the tables term, and "the
    tables are what makes a large truncation unaffordable" is only true
    if that term grows faster than the rest.
    """

    import dataclasses

    from woof.globe.vertical import HybridCoordinate

    base = load_config(SMOKE_CONFIG)

    def priced(truncation, nlev):
        coordinate = HybridCoordinate.pressure_blend(nlev, 100.0)
        return sizing.estimate_global_memory(dataclasses.replace(
            base, truncation=truncation, backend="cupy", precision="float32",
            a_half_pa=tuple(coordinate.a_half_pa),
            b_half=tuple(coordinate.b_half),
            zonal_wavenumber=min(base.zonal_wavenumber, truncation),
            diffusion_preserve_degree=min(base.diffusion_preserve_degree,
                                          truncation)))

    thin, thick = priced(63, 8), priced(63, 64)
    assert thin.legendre_table_bytes == thick.legendre_table_bytes
    assert thick.device_peak_bytes > thin.device_peak_bytes

    small, large = priced(127, 20), priced(255, 20)
    table_growth = large.legendre_table_bytes / small.legendre_table_bytes
    peak_growth = large.device_peak_bytes / small.device_peak_bytes
    # (T+1)(T+2)/2 x nlat against a doubled truncation is ~8x and the
    # band scratch beside it ~4x (6.3x together at T127 -> T255);
    # everything else is ~4x, so the tables outgrow the whole estimate.
    assert table_growth > 6.0
    assert table_growth > peak_growth


# --- gates -----------------------------------------------------------------

def test_the_gate_keys_are_not_the_regional_ledger_s():
    """New identifiers, deliberately.

    ``N0_GATE_METRICS`` are pre-registered ledger records that
    ``verify/nest_gates.py`` reads for the regional model.  A global run
    reporting under those keys writes a WRF gate record from a run with no
    WRF anything in it.
    """

    assert not (set(sizing.GLOBAL_GATE_METRICS)
                & set(core_preflight.N0_GATE_METRICS))
    assert set(sizing.GATE_DISPLAY) == set(sizing.GLOBAL_GATE_METRICS)


def test_an_unweighable_leg_is_none_and_never_passes():
    estimate = sizing.estimate_global_memory(load_config(SMOKE_CONFIG))
    gates = sizing.evaluate_global_gates(estimate, None)
    assert set(gates) == set(sizing.GLOBAL_GATE_METRICS)
    assert all(leg is None for leg in gates.values())
    assert not any(leg is True for leg in gates.values())


def test_the_largest_truncation_lever_is_exact(tmp_path):
    """The number the refusal tells a reader to type actually fits.

    And the next one up does not -- a lever that names a truncation one
    step below the true boundary sends the reader round again.
    """

    config = tmp_path / "big.toml"
    config.write_text(
        _retruncate(SMOKE_CONFIG.read_text(encoding="utf-8"), 400)
        .replace('backend = "numpy"', 'backend = "cupy"')
        .replace('precision = "float64"', 'precision = "float32"')
        .replace("transform_roundtrip_relative_linf = 5.0e-11",
                 "transform_roundtrip_relative_linf = 5.0e-5")
        .replace("transform_parseval_relative_error = 5.0e-12",
                 "transform_parseval_relative_error = 5.0e-5"),
        encoding="utf-8")
    cfg = load_config(config)
    # THREE GiB, not half of one.  The budget a truncation is weighed
    # against is now CARD bytes, and a card pays a MEASURED 0.73 GiB
    # outside the pool before one array is allocated (the context alone
    # measured 0.486 GiB on the RTX 5090), so a sub-gigabyte budget fits
    # no truncation at all and the lever correctly has nothing to name.
    budget = 3 * GIB
    lever = sizing.largest_truncation_within(cfg, budget)
    assert lever is not None and 3 <= lever < cfg.truncation

    import dataclasses

    def peak(truncation):
        return sizing.card_required_bytes(
            sizing.estimate_global_memory(dataclasses.replace(
                cfg, truncation=truncation, nlat=None, nlon=None,
                zonal_wavenumber=min(cfg.zonal_wavenumber, truncation),
                diffusion_preserve_degree=min(cfg.diffusion_preserve_degree,
                                              truncation))).device_peak_bytes)

    assert peak(lever) <= budget < peak(lever + 1)


def test_no_truncation_fits_a_hopeless_budget():
    cfg = load_config(SMOKE_CONFIG)
    import dataclasses

    cfg = dataclasses.replace(cfg, backend="cupy", precision="float32")
    assert sizing.largest_truncation_within(cfg, 1024) is None


# --- routing ---------------------------------------------------------------

@needs_the_global_route_sniff
def test_the_sniffer_separates_the_three_config_families():
    payload = SMOKE_CONFIG.read_bytes()
    assert is_arwen_global_toml_bytes(payload)
    assert not is_experiment_toml_bytes(payload)
    experiment = b'[experiment]\nname = "x"\n'
    assert not is_arwen_global_toml_bytes(experiment)
    # Bytes that are not TOML at all answer False rather than raising:
    # this sniff is the FIRST thing `woof check` asks, and a decode
    # traceback here would replace every other family's own refusal.
    assert not is_arwen_global_toml_bytes(b"this is not toml at all\x00")


@_needs_the_engine_sniff
def test_the_input_preflight_says_not_applicable_and_advances(capsys):
    """Stage one of the composed `woof check`.

    Returning anything but 0 stops the memory estimator that CAN certify
    this config from ever running.
    """

    from woof.ingest.preflight import _check_command

    assert _check_command(_check_args(SMOKE_CONFIG)) == 0
    printed = capsys.readouterr().out
    assert "not applicable" in printed
    assert "WOOF global run config" in printed
    assert "Continuing to the memory preflight" in printed


@_needs_the_engine_sniff
def test_check_main_routes_a_global_config_out_of_the_regional_loader(capsys):
    """The defect this wiring closes, stated as a test.

    ``_load_experiment_any`` accepts an experiment TOML or a legacy
    RunConfig and nothing else, so this config used to arrive at
    ``woof.config``'s strict loader and die in ``unknown table(s)``.
    """

    code = core_preflight.check_main(
        _check_args(SMOKE_CONFIG, budget_gib=8.0))
    printed = capsys.readouterr().out
    assert code == 0
    assert "WOOF global memory preflight" in printed
    assert "unknown table" not in printed


@needs_the_global_route_sniff
@_needs_the_engine_sniff
def test_a_regional_config_still_reaches_the_regional_estimator(capsys):
    """The routing is additive: nothing else changes lane."""

    regional = REPO_ROOT / "configs" / "conus3km.toml"
    assert is_experiment_toml_bytes(regional.read_bytes())
    assert not is_arwen_global_toml_bytes(regional.read_bytes())
    code = core_preflight.check_main(
        _check_args(regional, budget_gib=8.0, vram_gib=24.0))
    printed = capsys.readouterr().out
    assert "memory preflight for" in printed
    assert "WOOF global" not in printed
    assert code in (0, 1, 4)


# --- verdicts --------------------------------------------------------------

def _global_check(config, capsys, **overrides):
    code = core_preflight.check_main(_check_args(config, **overrides))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@_needs_the_engine_sniff
def test_a_card_that_fits_passes(capsys):
    code, out, _ = _global_check(QUICKSTART_CONFIG, capsys,
                                 budget_gib=12.0, vram_gib=16.0)
    assert code == 0
    assert "the Legendre tables fit the device budget: PASS" in out
    assert "the run's device peak fits the device budget: PASS" in out
    assert "calibrated model" in out
    assert "measures       maximum live bytes of the CuPy default memory pool" in out
    assert "calibration    T255, 40 levels, 24 h control: measured" in out


@_needs_the_engine_sniff
def test_a_card_the_retired_line_admitted_is_refused(capsys):
    """8 GiB was a pass at T255 by the 2.21 GiB line; the allocator
    measures 8.51 GiB there, so it is a FAIL with the breakage named."""

    code, out, _ = _global_check(QUICKSTART_CONFIG, capsys,
                                 budget_gib=8.0, vram_gib=12.0)
    assert code == 1
    assert "the run's device peak fits the device budget: FAIL" in out
    assert "OVER BUDGET:" in out


@_needs_the_engine_sniff
def test_an_undersized_card_is_refused_with_the_failing_term_named(
        capsys, tmp_path):
    config = tmp_path / "t533.toml"
    config.write_text(
        QUICKSTART_CONFIG.read_text(encoding="utf-8")
        .replace("truncation = 255", "truncation = 533"), encoding="utf-8")
    # The packed T533 float32 tables are 1.50 GiB (runtime, with scratch),
    # so a 1 GiB budget is the one that dies at construction.
    code, out, _ = _global_check(config, capsys, budget_gib=1.0, vram_gib=4.0)
    assert code == 1
    assert "the Legendre tables fit the device budget: FAIL" in out
    assert "OVER BUDGET AT CONSTRUCTION" in out
    # The concrete breakage: WHERE it dies, and why one obvious lever
    # cannot move it.
    assert "SphericalHarmonicTransform.__post_init__" in out
    assert "Trimming [vertical] cannot close this one" in out
    # A ONE-GIGABYTE BUDGET NAMES NO TRUNCATION, and that is the right
    # answer rather than a missing one: a card pays a MEASURED 0.73 GiB
    # outside the pool before an array is allocated, so no truncation the
    # loader admits fits this budget and the remedy is the card.
    assert ("no truncation this loader admits (>= 3) fits the budget: the "
            "card, not the configuration, is what has to change") in out


@_needs_the_engine_sniff
def test_a_peak_bound_refusal_names_the_truncation_lever(capsys, tmp_path):
    config = tmp_path / "t533.toml"
    config.write_text(
        QUICKSTART_CONFIG.read_text(encoding="utf-8")
        .replace("truncation = 255", "truncation = 533"), encoding="utf-8")
    code, out, _ = _global_check(config, capsys, budget_gib=5.5, vram_gib=8.0)
    assert code == 1
    assert "the Legendre tables fit the device budget: PASS" in out
    assert "the run's device peak fits the device budget: FAIL" in out
    assert "OVER BUDGET:" in out
    assert "remedy (first lever, grid.truncation): T" in out


@_needs_the_engine_sniff
def test_no_budget_and_no_card_fails_closed_with_a_typeable_remedy(
        capsys, monkeypatch):
    """Exit 2, and a refusal that names what it could not do.

    ``GPUWM_NO_LOCAL_GPU`` is the documented never-open-the-local-device
    switch, so under it the card is unmeasurable BY CONSTRUCTION -- and
    unmeasurable must never read as a pass.
    """

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    code, out, err = _global_check(QUICKSTART_CONFIG, capsys)
    assert code == 2
    assert "not measured" in out
    assert "REFUSED (exit 2, fail-closed)" in err
    assert "--budget-gib" in err
    assert "device peak is" in err


@_needs_the_engine_sniff
def test_alloc_is_refused_rather_than_downgraded(capsys):
    """A flag that promises a measurement must not print an estimate."""

    code, _, err = _global_check(QUICKSTART_CONFIG, capsys, alloc=True)
    assert code == 2
    assert "--alloc has no measurement to make" in err
    assert "woof global run" in err


@_needs_the_engine_sniff
def test_regional_only_flags_are_said_out_loud(capsys):
    code, out, _ = _global_check(QUICKSTART_CONFIG, capsys, budget_gib=12.0,
                                 column_chunk=512, reserve_gib=2.0)
    assert code == 0
    assert "--column-chunk is not read here" in out
    assert "--reserve-gib is not read here" in out


@_needs_the_engine_sniff
def test_the_numpy_backend_gets_no_device_verdict(capsys):
    """The CPU reference path opens no context and refuses no card."""

    code, out, _ = _global_check(SMOKE_CONFIG, capsys)
    assert code == 0
    assert "NO DEVICE VERDICT" in out
    assert "Every term below is HOST memory" in out
    assert "device budget" not in out


@_needs_the_engine_sniff
def test_the_json_report_carries_the_itemization_and_the_calibration(capsys):
    code, out, _ = _global_check(QUICKSTART_CONFIG, capsys, budget_gib=12.0,
                                 json=True)
    payload = json.loads(out[out.index("{"):])
    assert code == 0
    assert payload["config_family"] == "arwen_global"
    assert payload["gates"] == {metric: True
                                for metric in sizing.GLOBAL_GATE_METRICS}
    assert payload["device_peak_bytes"] == sum(
        payload[key] for key in ("legendre_table_bytes",
                                 "spectral_state_bytes",
                                 "surface_grid_bytes",
                                 "resident_grid_bytes",
                                 "working_grid_bytes",
                                 "physics_column_bytes",
                                 "cumulus_column_bytes",
                                 "fixed_bytes",
                                 "trajectory_bytes"))
    # The gather transient is reported and priced into the BANDED peak
    # only (sizing.banded_device_peak_bytes); the one-band peak is the fit's.
    assert "semilag_gather_bytes" in payload
    calibration = payload["calibration"]
    assert calibration["points"] == [dict(point) for point in
                                     sizing.DEVICE_PEAK_CALIBRATION]
    assert calibration["fitted"] == sizing.DEVICE_PEAK_FIT
    assert calibration["measures"] == sizing.DEVICE_PEAK_MEASURES
    assert len(calibration["residuals"]) == len(sizing.DEVICE_PEAK_CALIBRATION)
    assert calibration["shapes"] == sizing.calibration_shapes()
    assert len(calibration["held_out_residuals"]) == len(
        sizing.DEVICE_PEAK_CALIBRATION)
    assert calibration["worst_held_out_residual_fraction"] == (
        sizing.worst_held_out_residual_fraction())
    assert calibration["floors"] == [dict(row) for row in sizing.MEASURED_FLOORS]
    assert payload["device_peak_measures"] == sizing.DEVICE_PEAK_MEASURES
    # The quickstart runs the reference physics suite: priced as the
    # native suite the fit measured, and the report says so.
    assert payload["extrapolation"] == list(sizing.extrapolation_notes(
        load_config(QUICKSTART_CONFIG), payload["nlev"]))
    assert any("physics.mode" in note for note in payload["extrapolation"])
    assert payload["free_bytes_source"].startswith("declared")


def test_the_estimator_touches_no_device(monkeypatch):
    """Pure arithmetic on the config's shapes.

    The whole point is to answer BEFORE the allocation that would fail, so
    an estimator that imports CuPy to do it has already lost the argument
    -- and on a box carrying a wheel for the wrong CUDA major it would
    raise where it is supposed to report.  Proved by making the import
    impossible: the estimate for a ``backend = "cupy"`` config still comes
    out, with the same numbers.
    """

    import builtins
    import dataclasses

    cfg = dataclasses.replace(load_config(SMOKE_CONFIG), backend="cupy",
                              precision="float32")
    expected = sizing.estimate_global_memory(cfg)

    real_import = builtins.__import__

    def refuse_cupy(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise AssertionError("the estimator imported CuPy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_cupy)
    assert sizing.estimate_global_memory(cfg) == expected


# --- the run door ----------------------------------------------------------

def test_the_run_gate_refuses_a_card_that_cannot_hold_the_tables(tmp_path):
    config = tmp_path / "t533.toml"
    config.write_text(
        QUICKSTART_CONFIG.read_text(encoding="utf-8")
        .replace("truncation = 255", "truncation = 533"), encoding="utf-8")
    cfg = load_config(config)

    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": 1 * GIB})
    assert gate["refuse"] is True
    sentence = sizing.refusal_sentence(gate["estimate"], gate["free_bytes"],
                                       config_path=config)
    assert "SphericalHarmonicTransform.__post_init__" in sentence
    assert "woof check" in sentence

    # 16 GiB was roomy by the retired line (it read 10.29 GiB at T533);
    # the calibrated model puts T533 at forty levels above 20 GiB (the
    # table re-measured on the chunked cloud preparation on 2026-09-05
    # reads 25.9 GiB for this config; there is no measured T533 row, T533
    # is a floor of 20.74 GiB under a 22,528 MiB cap), so a 16 GiB card is
    # refused with the step named and a 40 GiB one admitted.  The bound is
    # the model's own figure, not a typed number: a re-fit that moved the
    # T533 reading below 20 GiB would be a measurement, and this test
    # would then be asking the wrong question.
    assert gate["estimate"].device_peak_bytes > 20 * GIB
    tight = sizing.run_memory_gate(cfg, probe={"free_bytes": 16 * GIB})
    assert tight["refuse"] is True
    assert "calibrated model" in tight["verdict"]
    assert "OutOfMemoryError inside the first step" in sizing.refusal_sentence(
        tight["estimate"], tight["free_bytes"], config_path=config)
    roomy = sizing.run_memory_gate(cfg, probe={"free_bytes": 40 * GIB})
    assert roomy["refuse"] is False


def test_the_run_gate_never_refuses_a_card_it_could_not_read(monkeypatch):
    """A gate that refuses on an unreadable card refuses every CPU box.

    ``woof go``'s rule, for the same reason: a genuine OOM prediction is
    a refusal, an absent measurement is not.
    """

    cfg = load_config(QUICKSTART_CONFIG)
    monkeypatch.setattr(core_preflight, "device_memory_probe_subprocess",
                        lambda **kwargs: None)
    monkeypatch.setattr(core_preflight, "device_memory_probe_reason",
                        lambda **kwargs: "no CUDA device answered")
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    gate = sizing.run_memory_gate(cfg)
    assert gate["refuse"] is False
    assert "not weighed" in gate["verdict"]


def test_the_run_gate_does_not_price_a_card_for_a_numpy_run(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("the numpy backend must not read a card")

    monkeypatch.setattr(core_preflight, "device_memory_probe_subprocess",
                        forbidden)
    gate = sizing.run_memory_gate(load_config(SMOKE_CONFIG))
    assert gate["refuse"] is False
    assert "backend='numpy'" in gate["verdict"]


def test_the_run_door_refuses_before_it_allocates(tmp_path, monkeypatch,
                                                  capsys):
    """One sentence at EXIT_DEVICE, and the run never starts.

    The output directory is the witness: ``runner.run`` prepares it before
    the first step, so an empty (indeed absent) outdir proves the refusal
    landed ahead of every allocation this configuration would make.
    """

    config = tmp_path / "t533.toml"
    config.write_text(
        QUICKSTART_CONFIG.read_text(encoding="utf-8")
        .replace("truncation = 255", "truncation = 533"), encoding="utf-8")
    outdir = tmp_path / "out"
    monkeypatch.setattr(core_preflight, "device_memory_probe_subprocess",
                        lambda **kwargs: {"free_bytes": 2 * GIB})
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)

    code = arwen_cli.main(["run", str(config), "--outdir", str(outdir)])
    captured = capsys.readouterr()
    # 4, not 2.  A device that will not hold the run is the one refusal a
    # caller can act on without a human -- ask for a smaller truncation --
    # so it has its own code rather than being folded in with a bad config.
    assert code == arwen_cli.EXIT_DEVICE
    assert not outdir.exists()
    assert "this card cannot hold this configuration" in captured.err
    assert captured.err.startswith("woof global: ")
    assert "Traceback" not in captured.err


def test_the_run_door_prints_the_verdict_on_a_run_that_proceeds(tmp_path,
                                                                capsys):
    """The gate is not silent when it admits, either.

    A numpy smoke run integrates for real here, so the line is proved
    against the door rather than against the gate function.
    """

    outdir = tmp_path / "out"
    code = arwen_cli.main(["run", str(SMOKE_CONFIG),
                     "--outdir", str(outdir)])
    printed = capsys.readouterr().out
    assert code == 0
    assert "woof global: memory:" in printed
    assert "host peak" in printed or "backend='numpy'" in printed
    assert (outdir / "arwen-global-receipt.json").exists()


def test_the_gate_credits_the_bytes_this_process_already_holds(monkeypatch):
    """A launcher that pre-grew the pool (a reservation on a shared card)
    took those bytes from the card before the subprocess probe read it;
    they are the run's own.  The T533 probe of 2026-09-05 with 19.5 GiB
    held against a 24.2 GiB figure read 12 GiB free and was refused at
    the door.  Read from ``sys.modules``, never imported: no pool, no
    credit, no CUDA context."""

    import dataclasses
    import sys
    from types import SimpleNamespace

    cfg = dataclasses.replace(
        load_config(QUICKSTART_CONFIG), backend="cupy", precision="float32",
        physics_mode="arwen-native", truncation=533, nlat=None, nlon=None)
    estimate = sizing.estimate_global_memory(cfg)
    # THE BOUNDARY IS THE CHEAPEST PLAN, NOT THE RESIDENT PEAK.  The door
    # prices the band count and the pinned tier the run will take, so the
    # free figure that separates a refusal from a run is the card figure
    # of the cheapest plan the sizer can build, not the resident one it
    # will not choose.  Three gigabytes below it is a refusal on any plan.
    parked = sum(sizing._spill_census_estimate(cfg, estimate).values())
    thinnest_live = sizing.banded_device_peak_bytes(
        estimate, sizing.LATITUDE_BAND_LADDER[-1], spilled_bytes=parked)
    thinnest_card = sizing.card_required_bytes(
        thinnest_live, spilled=True, banded=True)
    cheapest = sizing.plan_run_memory(cfg, thinnest_card * 4, estimate)
    assert cheapest.fits
    # What the door has to see free for the thinnest plan to pass: the
    # card figure with the model's own margin on it.
    predicted = int(sizing.PREDICTION_MARGIN * thinnest_card)
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    assert sizing.pool_bytes_held_unused() == 0
    bare = sizing.run_memory_gate(cfg, probe={"free_bytes": predicted - 3 * GIB})
    assert bare["refuse"] is True and "held in this process" not in bare["verdict"]

    class _Pool:
        def total_bytes(self):
            return 4 * GIB

        def used_bytes(self):
            return GIB

    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(get_default_memory_pool=_Pool))
    assert sizing.pool_bytes_held_unused() == 3 * GIB
    credited = sizing.run_memory_gate(
        cfg, probe={"free_bytes": predicted - 3 * GIB})
    assert credited["refuse"] is False
    assert credited["free_bytes"] == predicted
    assert "3.00 GiB of it held in this process's pool" in credited["verdict"]

    class _Broken:
        def total_bytes(self):
            raise RuntimeError("no context")

        def used_bytes(self):
            return 0

    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(get_default_memory_pool=_Broken))
    assert sizing.pool_bytes_held_unused() == 0


def test_an_exact_shape_fragmentation_row_outranks_a_larger_truncations(monkeypatch):
    # The larger-or-equal rule lends a shape with no row of its own the
    # rows of a larger grid; a shape that HAS a row at its truncation and
    # band count is priced at that row (MEASURED 2026-09-07: T383 at
    # sixteen bands with the tier reads x1.2201 on the RTX 5070 Ti, and the
    # T533 sixteen-band row's x1.4149 refused a plan the card holds).
    rows = (
        {"truncation": 533, "bands": 16, "live_bytes": 100, "held_bytes": 141,
         "chunk": 12500, "banded": True, "spilled": True},
        {"truncation": 383, "bands": 16, "live_bytes": 100, "held_bytes": 122,
         "chunk": 12500, "banded": True, "spilled": True},
        {"truncation": 383, "bands": 8, "live_bytes": 100, "held_bytes": 103,
         "chunk": 12500, "banded": True, "spilled": True},
    )
    monkeypatch.setattr(sizing, "POOL_FRAGMENTATION_MEASURES", rows)
    ratio, phrase = sizing.pool_fragmentation_for(True, True, False, bands=16, truncation=383)
    assert ratio == 1.22 and "exactly T383 and 16 bands" in phrase
    # no exact row at twelve bands: the rows at T383 or larger and twelve or fewer
    ratio, phrase = sizing.pool_fragmentation_for(True, True, False, bands=12, truncation=383)
    assert ratio == 1.03 and "exactly" not in phrase
    # no row at T255 at all: the rows at T255 or larger and sixteen or fewer, the largest
    ratio, phrase = sizing.pool_fragmentation_for(True, True, False, bands=16, truncation=255)
    assert ratio == 1.41 and "exactly" not in phrase
