"""Gates BIT-1 to BIT-4 and HALO-1: the band count is a layout knob.

Grid space is cut into latitude bands, spectral space stays whole, and
the two meet at a full-latitude Fourier waist.  The Legendre contraction
therefore runs at ``K = N = nlat`` at every band count, so a banded run
must reproduce a resident run BIT FOR BIT -- no new pin, no new identity
field, no tolerance.  Everything below is that claim, on the real
executable and its real operators:

BIT-1  the shipped step, resident against banded, every array of the
       advanced bundle and every scalar metric
BIT-3  the same case run twice at one band count: a nondeterministic
       reduction that passed BIT-1 by luck fails here
BIT-4  band count against band count, and both against resident
HALO-1  the meridional sweep behind its deep halo, per sub-step,
       including the polar band construction and the global Courant scan

and the two things the band count must NOT do: appear in an identity, or
leave a row of a reduction buffer unwritten.

THE DEFECT THIS LANE REMOVES, pinned by name below: at T533 L40 float32
the transport divided the whole stacked tracer mass by the whole
pseudo-density into a fresh ten-tracer volume, 2,053,123,584 B, and that
allocation is the one that failed at step 2 on a 32 GiB RTX 5090
(MEASURED 2026-09-06).  ``test_the_transport_stops_allocating_the_volume_
that_killed_a_t533_step`` measures that the volume is gone, not merely
smaller.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import hashlib
import tracemalloc
from dataclasses import replace

import numpy as np
import pytest

from woof.globe.bands import (
    MIN_BAND_ROWS,
    BandPipeline,
    band_edges,
    widest_band_count,
)
from woof.globe.config import load_config
from woof.globe.runner import build_model_and_cold_state, resolve_latitude_bands
from woof.globe.transport import GridTracerTransport
from woof.globe.vertical import HybridCoordinate
from woof.globe.spectral.transform import (
    SphericalHarmonicTransform,
    latitude_band_edges,
)

CONFIG = str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml")
TRACERS = ("qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng")


# ---------------------------------------------------------------- schedule


def test_the_band_schedule_is_a_pure_function_of_the_grid_and_the_count():
    for nlat in (33, 65, 384, 801):
        for bands in (1, 2, 3, 4, 8, 16):
            edges = band_edges(nlat, bands)
            assert edges[0] == 0 and edges[-1] == nlat
            assert edges == sorted(edges)
            assert edges == [(k * nlat) // bands for k in range(bands)] + [nlat]
            if bands > widest_band_count(nlat):
                continue
            pipeline = BandPipeline(nlat, bands)
            assert [s.start for s in pipeline.slices()] == edges[:-1]
            assert [s.stop for s in pipeline.slices()] == edges[1:]
            assert pipeline.resident == (bands == 1)


def test_a_band_narrower_than_the_measured_floor_is_refused_by_name():
    # The floor is a MEASURED copy-rate floor (RTX 5070 Ti, 2026-09-06:
    # below four latitude rows a T533 band slab copies at half the
    # contiguous rate), and a run that asks past it is refused rather
    # than silently clamped -- a clamp would run a different band count
    # from the one the receipt records.
    with pytest.raises(ValueError, match="below the 4-row floor"):
        BandPipeline(33, 16)
    assert widest_band_count(33) == 33 // MIN_BAND_ROWS == 8
    BandPipeline(33, 8)
    # The floor is the RUN's, not the gate's: a determinism test that
    # wants one row a band says so.
    assert len(BandPipeline(33, 33, minimum_rows=1)) == 33


def test_the_deep_halo_clips_at_the_poles_and_names_what_it_added():
    pipeline = BandPipeline(40, 4)
    first, lead, trail = pipeline.halo(slice(0, 10), 4)
    assert (first, lead, trail) == (slice(0, 14), 0, 4)
    middle, lead, trail = pipeline.halo(slice(10, 20), 4)
    assert (middle, lead, trail) == (slice(6, 24), 4, 4)
    last, lead, trail = pipeline.halo(slice(30, 40), 4)
    assert (last, lead, trail) == (slice(26, 40), 4, 0)
    # A halo wider than the grid clips to the grid rather than indexing
    # past the pole.
    assert pipeline.halo(slice(10, 20), 100)[0] == slice(0, 40)


def test_a_band_is_a_contiguous_slab_and_not_a_stride():
    """The defect that reached the ten-step T255 checkpoint gate.

    MEASURED 2026-09-06 on an RTX 5070 Ti at T255 and T533 shapes, float32
    and float64: a cupy longitude reduction over a STRIDED latitude slice
    of a levelled array returns different bits from the same rows in a
    contiguous array, at every band count, while the same reduction over a
    contiguous band is byte-identical to the whole reduction's rows.  The
    band count never entered it; the layout did.  So a band is
    materialised where it is cut, and this is the assertion that keeps it
    that way.
    """
    from woof.globe.dynamics import band_view

    value = np.arange(4 * 12 * 6, dtype=np.float64).reshape(4, 12, 6)
    whole = band_view(np, value, slice(0, 12))
    assert whole is value, "one band is the array itself, not a copy of it"
    band = band_view(np, value, slice(3, 9))
    assert band.flags["C_CONTIGUOUS"]
    assert np.array_equal(band, value[:, 3:9, :])
    assert not np.shares_memory(band, value)
    plane = np.arange(12 * 6, dtype=np.float64).reshape(12, 6)
    assert band_view(np, plane, slice(3, 9)).flags["C_CONTIGUOUS"]


# ------------------------------------------------------------------- waist


def test_a_waist_filled_band_by_band_is_the_whole_analysis():
    """The incremental fill against :meth:`forward`, exactly (FFT-1's
    band-fed form: the band pipeline never holds the field whole)."""
    transform = SphericalHarmonicTransform.create(
        truncation=21, dealias_factor=1.5, backend="numpy", precision="float64"
    )
    nlat = transform.grid.nlat
    rng = np.random.default_rng(0)
    field = rng.standard_normal((3, 5, nlat, transform.grid.nlon))
    reference = transform.forward(field)
    for bands in (1, 2, 3, 5, 7, nlat):
        waist = transform.open_waist((3, 5), bands=bands)
        for r0, r1 in latitude_band_edges(nlat, bands):
            waist.fill_band(r0, r1, field[..., r0:r1, :])
        assert np.array_equal(transform.contract_waist(waist.close()), reference)


def test_a_waist_with_an_unwritten_row_is_refused_rather_than_contracted():
    transform = SphericalHarmonicTransform.create(
        truncation=21, dealias_factor=1.5, backend="numpy", precision="float64"
    )
    nlat = transform.grid.nlat
    field = np.zeros((nlat, transform.grid.nlon))
    waist = transform.open_waist((), bands=4)
    edges = latitude_band_edges(nlat, 4)
    for r0, r1 in edges[:-1]:
        waist.fill_band(r0, r1, field[r0:r1])
    with pytest.raises(ValueError, match="never filled"):
        waist.close()
    # And two bands writing one row is a scheduling defect, named here
    # rather than discovered as an answer that moves with the band count.
    waist = transform.open_waist((), bands=4)
    r0, r1 = edges[0]
    waist.fill_band(r0, r1, field[r0:r1])
    with pytest.raises(ValueError, match="already filled"):
        waist.fill_band(r0, r1, field[r0:r1])


def test_the_banded_gradient_is_the_whole_gradient():
    transform = SphericalHarmonicTransform.create(
        truncation=21, dealias_factor=1.5, backend="numpy", precision="float64"
    )
    nlat = transform.grid.nlat
    rng = np.random.default_rng(1)
    coeff = transform.forward(
        rng.standard_normal((2, 4, nlat, transform.grid.nlon))
    )
    east_ref, north_ref = transform.gradient(coeff)
    for bands in (1, 2, 3, 5, 11):
        waists = transform.gradient_waists(coeff, bands=bands)
        east = np.empty_like(east_ref)
        north = np.empty_like(north_ref)
        for r0, r1 in latitude_band_edges(nlat, bands):
            e, n = transform.gradient_band(waists, r0, r1)
            east[..., r0:r1, :] = e
            north[..., r0:r1, :] = n
        assert np.array_equal(east, east_ref)
        assert np.array_equal(north, north_ref)


# ------------------------------------------------------------------ HALO-1


def _transport_case(truncation=42, nlev=6, seed=0, meridional=800.0, wind=300.0):
    transform = SphericalHarmonicTransform.create(
        truncation=truncation, dealias_factor=1.5, backend="numpy",
        precision="float64",
    )
    vertical = HybridCoordinate.pressure_blend(nlev)
    nlat, nlon = transform.grid.shape
    rng = np.random.default_rng(seed)
    shape = (nlev, nlat, nlon)
    dp = 2000.0 + 500.0 * rng.random(shape)
    tracers = {}
    for index, name in enumerate(TRACERS):
        base = rng.random(shape) ** 3
        base[:, : nlat // 3, :] = 0.0
        tracers[name] = base * (1.0 + index)
    lat = np.linspace(-1.0, 1.0, nlat)[None, :, None]
    u = wind * (1.0 - lat * lat) + 5.0 * rng.standard_normal(shape)
    v = meridional * rng.standard_normal(shape)
    omega = np.zeros((nlev + 1, nlat, nlon))
    omega[1:-1] = 0.2 * rng.standard_normal((nlev - 1, nlat, nlon))
    return transform, vertical, tracers, dp, dp * u, dp * v, omega


@pytest.mark.parametrize("courant_limit", (0.25, 0.05))
@pytest.mark.parametrize("step", (0, 1))
def test_halo_1_the_meridional_sweep_behind_its_deep_halo(courant_limit, step):
    """Every advanced tracer, the pseudo-density and every metric, at every
    band count, byte for byte against the resident sweep.

    The meridional sweep is the one whose stencil crosses a band edge:
    ``_walled_step`` reads ``q[j-2 .. j+2]``, so ``n`` sub-steps need
    ``2n`` rows either side.  A smaller Courant limit forces more
    sub-steps and therefore a deeper halo, which is the case the carry
    buffer exists for.
    """
    transform, vertical, tracers, dp, fu, fv, omega = _transport_case()
    nlat = transform.grid.nlat
    reference = None
    for bands in (1, 2, 3, 4, 5, 8):
        transport = GridTracerTransport(
            transform=transform, vertical=vertical, courant_limit=courant_limit
        )
        transport.pipeline = BandPipeline(nlat, bands)
        out, metrics = transport.advance(
            {k: v.copy() for k, v in tracers.items()},
            dp.copy(), fu.copy(), fv.copy(), omega.copy(), 30.0, step=step,
        )
        payload = {name: out[name].tobytes() for name in TRACERS}
        payload["pseudo_density"] = metrics["pseudo_density"].tobytes()
        scalars = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float, str))
        }
        if reference is None:
            reference = (payload, scalars)
            # The halo has to be deeper than one row for this to test
            # anything: at courant_limit 0.05 the sweep sub-cycles.
            assert metrics["substeps_y"] >= (2 if courant_limit < 0.25 else 1)
            continue
        assert payload == reference[0], f"bands={bands}"
        assert scalars == reference[1], f"bands={bands}"


def test_the_transport_stops_allocating_the_volume_that_killed_a_t533_step():
    """The named defect: the whole ten-tracer ratio volume.

    MEASURED 2026-09-06 on an RTX 5090, T533 L40 float32: the transport's
    floor section asked for 2,053,123,584 B -- one whole ten-tracer grid
    volume -- and the allocation failed at step 2.  Here the same shape
    argument runs on the CPU at a size a test can hold, and the measure is
    that the volume is GONE rather than smaller: the banded peak sits
    below the resident peak by more than one stacked mass array.
    """
    transform, vertical, tracers, dp, fu, fv, omega = _transport_case(
        truncation=42, nlev=6
    )
    nlat = transform.grid.nlat
    stack_bytes = 10 * dp.size * dp.dtype.itemsize
    peaks = {}
    for bands in (1, 8):
        transport = GridTracerTransport(transform=transform, vertical=vertical)
        transport.pipeline = BandPipeline(nlat, bands)
        arguments = (
            {k: v.copy() for k, v in tracers.items()},
            dp.copy(), fu.copy(), fv.copy(), omega.copy(),
        )
        tracemalloc.start()
        try:
            transport.advance(*arguments, 30.0, step=0)
            peaks[bands] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    saved = peaks[1] - peaks[8]
    assert saved > stack_bytes, (
        f"banding saved {saved} B, less than one whole stacked mass "
        f"({stack_bytes} B): the ratio volume is still being allocated"
    )


# ------------------------------------------------- BIT-1, BIT-3 and BIT-4


def _digest(host, value) -> str:
    array = np.asarray(host(value))
    return hashlib.sha256(
        array.dtype.str.encode() + str(array.shape).encode()
        + np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def _inventory(model, bundle) -> dict[str, str]:
    host = model.transform.backend.to_numpy
    out = {}
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure", "qv"):
        out["atmosphere." + name] = _digest(host, getattr(bundle.atmosphere, name))
    for name, value in bundle.atmosphere.grid_tracers().items():
        out["tracer." + name] = _digest(host, value)
    for name, value in bundle.surface.arrays().items():
        out["surface." + name] = _digest(host, value)
    for name, value in sorted(bundle.physics_state.arrays.items()):
        out["physics." + name] = _digest(host, value)
    return out


def _scalars(metrics, prefix=""):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (bool, str)):
            out[prefix + key] = value
        elif isinstance(value, (int, float)):
            out[prefix + key] = float(value)
        elif isinstance(value, dict):
            out.update(_scalars(value, prefix + key + "."))
    return out


def _stepped(bands: int, steps: int = 2):
    cfg = load_config(CONFIG)
    model, bundle = build_model_and_cold_state(
        replace(cfg, latitude_bands=bands)
    )
    assert model.latitude_bands == bands
    assert model.pipeline.bands == bands
    assert model.transport.pipeline is model.pipeline
    trace = []
    for _ in range(steps):
        bundle, metrics = model.step(bundle, cfg.dt_s)
        trace.append(_scalars(metrics))
    return _inventory(model, bundle), trace


@pytest.mark.parametrize("bands", (2, 3, 4, 8))
def test_bit_1_and_bit_4_the_step_is_the_resident_step_at_every_band_count(bands):
    """The whole shipped step, reference physics, two steps.

    Forty-four arrays: the five spectral fields, the ten grid tracers, the
    surface reservoirs and the physics namespace, plus every scalar the
    step reports.
    """
    resident, resident_trace = _stepped(1)
    banded, banded_trace = _stepped(bands)
    assert set(resident) == set(banded)
    differing = sorted(k for k in resident if resident[k] != banded[k])
    assert not differing, f"bands={bands} moved {differing}"
    assert banded_trace == resident_trace


def test_bit_3_the_same_band_count_twice_is_the_same_answer():
    """Catches a nondeterministic reduction that passed BIT-1 by luck."""
    first, first_trace = _stepped(4)
    second, second_trace = _stepped(4)
    assert first == second
    assert first_trace == second_trace


def test_the_band_pipeline_actually_runs_the_bands_it_is_given():
    """Positive evidence of work: a band count that changed no code path
    would pass every compare above by doing nothing."""
    from woof.globe.spectral import transform as transform_module

    drains = {"count": 0}
    original = transform_module.SphericalHarmonicTransform.waist_band_to_grid

    def counting(self, waist, r0, r1):
        drains["count"] += 1
        return original(self, waist, r0, r1)

    transform_module.SphericalHarmonicTransform.waist_band_to_grid = counting
    try:
        cfg = load_config(CONFIG)
        model, bundle = build_model_and_cold_state(
            replace(cfg, latitude_bands=4)
        )
        before = drains["count"]
        model.step(bundle, cfg.dt_s)
        banded = drains["count"] - before

        model, bundle = build_model_and_cold_state(
            replace(cfg, latitude_bands=1)
        )
        before = drains["count"]
        model.step(bundle, cfg.dt_s)
        resident = drains["count"] - before
    finally:
        transform_module.SphericalHarmonicTransform.waist_band_to_grid = original
    assert banded > 0, "a four-band step drained no waist band"
    assert resident == 0, (
        "the resident step drained a waist band: the shipped default is "
        "supposed to keep the whole-grid route it has today"
    )


# ------------------------------------------------------------- the identity


def test_the_band_count_is_absent_from_every_identity_at_every_value():
    cfg = load_config(CONFIG)
    base = cfg.config_hash
    assert "latitude_bands" not in cfg.config_identity
    for bands in (0, 1, 2, 4, 8, 32):
        moved = replace(cfg, latitude_bands=bands)
        assert "latitude_bands" not in moved.config_identity
        assert moved.config_hash == base, (
            "the band count moved the config hash: it is a streaming "
            "granularity, and the Legendre contraction keeps K = N = nlat "
            "at every value of it"
        )


def test_the_door_carries_the_band_count_and_the_receipt_records_it(tmp_path):
    from woof.globe.cli import main

    outdir = tmp_path / "banded"
    assert main([
        "run", CONFIG, "--outdir", str(outdir), "--until-s", "1080",
        "--latitude-bands", "4",
    ]) == 0
    import json

    receipt = json.loads(
        (outdir / "arwen-global-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["latitude_bands"]["latitude_bands"] == 4
    assert receipt["latitude_bands"]["chosen_by"] == "config"
    assert receipt["latitude_bands"]["resident"] is False
    assert receipt["latitude_bands"]["widest_band_rows"] == 9
    with pytest.raises(SystemExit):
        main(["run", CONFIG, "--outdir", str(tmp_path / "x"),
              "--latitude-bands", "not-a-number"])


def test_the_sizer_returns_the_resident_run_where_there_is_no_card():
    """On the numpy backend there is no card to size against, and a band
    count chosen against nothing would be a number rather than a
    decision."""
    cfg = load_config(CONFIG)
    assert cfg.latitude_bands == 0
    assert resolve_latitude_bands(cfg) == 1
    assert resolve_latitude_bands(replace(cfg, latitude_bands=4)) == 4


def test_the_refusal_lives_at_the_door_and_the_sizer_never_raises():
    """Refuse rather than admit -- but refuse ONCE, at the door.

    The MEASURED 2026-09-06 defect is a door that predicted 24.45 GiB, saw
    30.90 GiB free, admitted the run and watched the live peak reach
    27.57.  The fix is the gate pricing the band count the run will take.
    The sizer itself never raises: it is called from inside
    ``build_model_and_cold_state``, which every tool and test that builds
    a model calls, and a refusal from there is a traceback rather than a
    sentence.
    """
    from woof.globe import sizing

    cfg = replace(
        load_config(CONFIG), backend="cupy", truncation=1279, nlat=None,
        nlon=None,
    )
    estimate = sizing.estimate_global_memory(cfg)
    bands, _peak = sizing.latitude_bands_for(cfg, 8 * 2**30, estimate)
    assert bands is None, "a shape nothing fits must report no band count"
    chosen = sizing.auto_latitude_bands(cfg, free_bytes=8 * 2**30)
    assert chosen == min(
        max(sizing.LATITUDE_BAND_LADDER),
        sizing.widest_band_count(estimate.nlat),
    ), "when nothing fits the sizer hands back the best the card can be given"
    assert chosen > 1


def test_the_banded_peak_model_divides_only_what_the_band_count_divides():
    from woof.globe import sizing

    class Estimate:
        backend = "cupy"
        nlat = 800
        nlev = 40
        truncation = 533
        device_peak_bytes = 32 * 2**30
        working_grid_bytes = 16 * 2**30

    one = sizing.banded_device_peak_bytes(Estimate(), 1)
    assert one == Estimate.device_peak_bytes
    f = sizing.banded_working_resident_fraction(Estimate.truncation)
    assert 0.0 < f < 1.0
    previous = one
    for bands in (2, 4, 8, 16):
        peak = sizing.banded_device_peak_bytes(Estimate(), bands)
        divisible = Estimate.working_grid_bytes * (1.0 - f)
        assert peak == int(Estimate.device_peak_bytes - divisible * (1.0 - 1.0 / bands))
        assert peak < previous
        previous = peak
    # It never divides below the part that does not divide: the tables,
    # the spectral state, the persistent grid state, the fixed terms and
    # the measured fraction of the working term the band count leaves.
    assert previous > int(Estimate.device_peak_bytes
                          - Estimate.working_grid_bytes * (1.0 - f))
    # A parked byte is one byte off the card, no more.
    assert sizing.SPILL_RELIEF == 1.0
    assert sizing.banded_device_peak_bytes(
        Estimate(), 8, spilled_bytes=2**30) == previous_at_8(Estimate(), sizing) - 2**30


def previous_at_8(estimate, sizing):
    return sizing.banded_device_peak_bytes(estimate, 8)


def test_the_banded_peak_fraction_is_the_pessimistic_end_of_its_measurements():
    """The constant is derived, and it is derived in the safe direction.

    ``peak(B) = peak(1) - W (1 - f) (1 - 1/B)`` solved on every measured
    pair of whole-run live peaks on the tree whose physics is banded.  A
    LARGER f predicts a larger banded peak, so the pessimistic end is the
    largest of them: taking a smaller one under-predicts, which is the
    admitting direction the design's R2 names as the defect it is fixing.
    """
    from woof.globe import sizing

    def solve(row):
        relief = row["resident_bytes"] - row["banded_bytes"] - row.get("parked_bytes", 0)
        return 1.0 - relief / (row["working_grid_bytes"] * (1.0 - 1.0 / row["bands"]))

    values = [solve(row) for row in sizing.BANDED_PEAK_MEASURES]
    assert sizing.BANDED_WORKING_RESIDENT_FRACTION == max(values)
    t255 = next(row for row in sizing.BANDED_PEAK_MEASURES if row["truncation"] == 255)
    t533 = next(row for row in sizing.BANDED_PEAK_MEASURES if row["truncation"] == 533)
    assert round(solve(t255), 4) == 0.7262
    assert round(solve(t533), 3) == 0.184
    for row in sizing.BANDED_PEAK_MEASURES:
        assert row["banded_bytes"] + row.get("parked_bytes", 0) < row["resident_bytes"]
    # The fraction is the shape's: the measured row at its truncation, the
    # larger of the two rows a truncation lies between, the nearest row
    # beyond either end.
    assert sizing.banded_working_resident_fraction(255) == solve(t255)
    assert sizing.banded_working_resident_fraction(533) == solve(t533)
    assert sizing.banded_working_resident_fraction(383) == max(solve(t255), solve(t533))
    assert sizing.banded_working_resident_fraction(799) == solve(t533)
    assert sizing.banded_working_resident_fraction(63) == solve(t255)


def test_the_semi_lagrangian_core_is_priced_with_its_second_time_level():
    """The default core carries the trajectory state beside everything
    the IMEX calibration measured: six levelled volumes and one plane."""
    from woof.globe import sizing

    cfg = load_config(CONFIG)
    imex = sizing.estimate_global_memory(replace(cfg, integrator="imex_ssp3"))
    sl = sizing.estimate_global_memory(replace(cfg, integrator="sl_si"))
    assert imex.trajectory_bytes == 0 and imex.semilag_gather_bytes == 0
    points = sl.nlat * sl.nlon
    assert sl.trajectory_bytes == (6 * sl.nlev + 1) * points * sl.float_itemsize
    assert sl.semilag_gather_bytes == 3 * min(cfg.semilag.gather_batch, 11) * sl.nlev * points * sl.float_itemsize
    assert sl.resident_bytes == imex.resident_bytes + sl.trajectory_bytes
    if sl.backend == "cupy":
        assert sl.device_peak_bytes == imex.device_peak_bytes + sl.trajectory_bytes
    assert any("trajectory" in row[0] for row in sl.itemization())
    assert any("gather" in row[0] for row in sl.itemization())
    # The gather transient enters the banded peak whole and the one-band peak not at all.
    for bands in (1, 8):
        both = sizing.banded_device_peak_bytes(sl, bands)
        one = sizing.banded_device_peak_bytes(imex, bands)
        if sl.backend != "cupy":
            assert both == one == 0
        elif bands == 1:
            assert both == one + sl.trajectory_bytes
        else:
            assert both == one + sl.trajectory_bytes + sl.semilag_gather_bytes


def test_the_door_prices_the_band_count_the_run_will_take():
    """A door that weighed the RESIDENT peak would refuse the very run the
    band count and the host tier exist to admit.

    The peak is priced in CARD bytes, not pool bytes: the door refuses on
    what the card must hold, which is the pool's live peak times the
    measured fragmentation of the plan's allocation pattern plus the
    measured out-of-pool tax (``sizing.card_required_bytes``).  Charging
    neither is how the 2026-09-06 T533 run was admitted at a printed 24.45
    GiB and died at a 27.57 GiB live peak.
    """
    import dataclasses

    from woof.globe import sizing
    from woof.globe.config import load_config

    cfg = dataclasses.replace(
        load_config(str(_shipped_configs() / "arwen_global_t255_quickstart.toml")),
        backend="cupy", precision="float32", physics_mode="arwen-native",
        truncation=383, nlat=None, nlon=None, latitude_bands=0)
    estimate = sizing.estimate_global_memory(cfg)
    resident_card = sizing.card_required_bytes(estimate.device_peak_bytes)

    # A card that cannot hold the resident run, but can hold a plan.
    free = int(resident_card * 0.92)
    plan = sizing.plan_run_memory(cfg, free, estimate)
    assert plan.fits
    # The model's margin is inside the card figure already: it multiplies
    # the LIVE peak the model predicts, not the measured terms that turn
    # it into card bytes.
    assert resident_card > free
    assert plan.card_bytes <= free
    assert plan.live_peak_bytes < estimate.device_peak_bytes
    assert plan.spill_slices or (plan.bands or 1) > 1

    # latitude_bands_for reads that plan and nothing else, so the count the
    # gate weighs is the count the run takes.
    bands, peak = sizing.latitude_bands_for(cfg, free, estimate)
    assert bands == plan.bands and peak == plan.live_peak_bytes

    # A card that cannot hold it at any count and any spill returns no
    # count, and the raising form turns that into a refusal rather than a
    # run.
    none_bands, _ = sizing.latitude_bands_for(cfg, 2 * 2**30, estimate)
    assert none_bands is None

    # A config that names the count is taken at its word.
    named = dataclasses.replace(cfg, latitude_bands=8)
    assert sizing.latitude_bands_for(named, resident_card * 4, estimate)[0] == 8


def test_a_failed_run_records_the_band_schedule_it_died_on(tmp_path, monkeypatch):
    """An out-of-memory death is exactly when the schedule is wanted.

    MEASURED 2026-09-06, RTX 5090: a T383 run at eight bands lost the
    card to another tenant and died at step 0; its failure receipt
    carried `latitude_bands: null`, so nothing in the output said which
    schedule was streaming when it ran out.
    """
    import json

    from woof.globe import runner as runner_module
    from woof.globe.runner import RECEIPT_NAME, run

    cfg = replace(load_config(CONFIG), latitude_bands=3)
    original = runner_module.MoistHybridModel.step

    def explode(self, bundle, dt_s, **kwargs):
        raise RuntimeError("the card went away")

    monkeypatch.setattr(runner_module.MoistHybridModel, "step", explode)
    try:
        with pytest.raises(RuntimeError):
            run(cfg, tmp_path / "died")
    finally:
        monkeypatch.setattr(runner_module.MoistHybridModel, "step", original)

    receipt = json.loads(
        (tmp_path / "died" / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "error"
    record = receipt["latitude_bands"]
    assert record["latitude_bands"] == 3
    assert record["chosen_by"] == "config"
    assert record["resident"] is False
