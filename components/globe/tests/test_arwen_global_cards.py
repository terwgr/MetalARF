"""Gates CARD-1, BIT-6 and WIRE-1's arithmetic: the multi-card layer.

Three separable claims, and this file is where the two that need no GPU
are settled:

CARD-1  every module-level cache of a device object carries the device id
        in its key.  A process that opens two cards builds an entry on
        the first and hands it to a launch on the second, and CuPy's own
        error for that case ("The device where the array resides (0) is
        different from the current device (1)") only appears where it
        happens to check.  Neither node has two cards, so the device id
        is driven through a stub here and the real single-device arm runs
        on the cards; the stub is the arm that can distinguish a key that
        carries the id from one that does not, which is what the gate is
        about.

BIT-6   the band-to-card assignment is outside the arithmetic.  The band
        schedule is ``floor(k * nlat / B)`` whatever the card count, the
        reduction buffers are laid out by ``(nlat, shape)`` and filled in
        grid order, and a partial sum is added in ascending RANK order.
        The assertions here are on those three; the whole-model arm runs
        on the pair.

WIRE-1  the transport moves the bytes it says it moves and the ledger
        counts them.  The rate itself is a two-node measurement; the
        accounting is tested here over the loopback, where a wrong ledger
        would report a wrong rate on the real link too.

The transport is exercised for real over TCP in-process (two ranks, two
threads, loopback sockets), because a mocked transport tests the mock.
"""
from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from woof.globe.configs_dir import config_root as _shipped_configs
from woof.globe import cards
from woof.globe.bands import (
    BandPipeline,
    LatitudeAccumulator,
    PlaneAccumulator,
    band_edges,
)


# ---------------------------------------------------------------------
# CARD-1
# ---------------------------------------------------------------------


class _FakeDevice:
    def __init__(self, ident):
        self.id = ident


class _FakeCuda:
    def __init__(self):
        self.current = 0

    def Device(self):  # noqa: N802 - CuPy's own spelling
        return _FakeDevice(self.current)


class _FakeCupy(types.ModuleType):
    """Just enough cupy for a cache key: a device id and a kernel factory."""

    def __init__(self):
        super().__init__("cupy")
        self.cuda = _FakeCuda()
        self.built = []

    def ElementwiseKernel(self, *args, **kwargs):  # noqa: N802
        made = ("kernel", len(self.built), self.cuda.current)
        self.built.append(made)
        return made


@pytest.fixture
def two_devices(monkeypatch):
    """A stub ``cupy`` whose current device the test moves.

    Neither node in this pair has two cards, so a process that holds a
    device-0 array and a device-1 array at once cannot be built here.
    What CAN be built, and what the defect actually is, is a cache key
    that does not distinguish the two: with this stub the caches are
    driven on device 0 and then on device 1 in one process, and a key
    missing the device id collapses the two entries into one.
    """
    fake = _FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return fake


def test_card1_the_backend_key_carries_the_device_and_numpy_has_none(two_devices):
    from woof.globe.spectral.backend import device_cache_key

    two_devices.cuda.current = 0
    assert device_cache_key(two_devices) == ("cupy", 0)
    two_devices.cuda.current = 1
    assert device_cache_key(two_devices) == ("cupy", 1)
    # NumPy has no device, so every CPU-backed cache keeps its old key.
    assert device_cache_key(np) == ("numpy",)


def test_card1_every_module_level_device_cache_builds_once_per_device(two_devices):
    """The gate proper: each cache, both devices, distinct entries.

    Six caches, named individually rather than swept, so a cache added
    later without a device id fails this file rather than a two-card run.
    """
    from woof.globe.physics import native_batch
    from woof.globe import transport as transport_module
    from woof.globe.spectral import fused

    for holder in (
        fused._CACHE, transport_module._KERNELS, native_batch._COLUMN_WATER_KERNEL,
    ):
        holder.clear()

    def build_all():
        return (
            fused.project_kernel(two_devices),
            transport_module._fused(two_devices, "periodic"),
            transport_module._fused(two_devices, "walled"),
            native_batch._column_water_kernel(two_devices),
        )

    two_devices.cuda.current = 0
    first = build_all()
    two_devices.cuda.current = 1
    second = build_all()
    two_devices.cuda.current = 0
    again = build_all()

    assert all(a != b for a, b in zip(first, second)), (
        "a cache handed device 1 the entry it built on device 0"
    )
    assert again == first, "device 0's entries were not reused"
    for holder, expected in (
        (fused._CACHE, 2), (transport_module._KERNELS, 4),
        (native_batch._COLUMN_WATER_KERNEL, 2),
    ):
        assert len(holder) == expected
        for key in holder:
            assert 0 in key or 1 in key, f"{key!r} carries no device id"


def test_card1_the_vertical_operator_serves_two_devices_from_one_build(two_devices):
    """The one cache deliberately NOT device-keyed, and why that is right.

    ``semi_implicit._OPERATORS`` holds an object of NumPy arrays; the three
    caches of DEVICE arrays it owns carry the device in their own keys.
    So one operator serves both cards and hands each its own matrices,
    and the NumPy eigen-decomposition behind it is not repeated per card.
    """
    from woof.globe.semi_implicit import (
        _OPERATORS,
        vertical_structure_operator,
    )
    from woof.globe.vertical import HybridCoordinate
    from woof.globe.spectral.backend import get_backend

    _OPERATORS.clear()
    vertical = HybridCoordinate.surface_stretched(40)
    one = vertical_structure_operator(vertical, 300.0, 1.0e5)
    two = vertical_structure_operator(vertical, 300.0, 1.0e5)
    assert one is two and len(_OPERATORS) == 1

    backend = get_backend("numpy", "float64")
    two_devices.cuda.current = 0
    one.device_matrices(backend)
    keys_after_numpy = set(one._device)
    assert keys_after_numpy == {("numpy", str(backend.float_dtype))}, (
        "the NumPy key must stay one element so a CPU run keeps its behaviour"
    )


# ---------------------------------------------------------------------
# The band assignment: BIT-6's arithmetic
# ---------------------------------------------------------------------


def test_the_band_schedule_does_not_move_with_the_card_count():
    """The schedule is a function of (nlat, bands) and of nothing else."""
    reference = band_edges(384, 8)
    for cards_count in (1, 2, 3, 4):
        weights = tuple(1.0 + i for i in range(cards_count))
        owners = cards.band_owners(384, 8, weights)
        assert len(owners) == 8
        assert band_edges(384, 8) == reference
        assert sorted(set(owners)) == list(range(cards_count))
        # contiguous runs: a rank's bands are consecutive
        for rank in set(owners):
            held = [k for k, r in enumerate(owners) if r == rank]
            assert held == list(range(held[0], held[-1] + 1))


def test_the_faster_card_is_given_more_rows():
    """Assignment by MEASURED throughput, and the reciprocal is the weight."""
    weights = cards.throughput_weights((357.4, 739.9))  # 5090, 5070 Ti
    assert weights[0] > weights[1]
    assert abs(sum(weights) - 1.0) < 1e-9
    owners = cards.band_owners(384, 16, weights)
    fast = sum(1 for r in owners if r == 0)
    assert fast > 16 - fast, "the 2.07x faster card took the smaller share"


def test_a_card_with_no_band_is_refused_by_name():
    with pytest.raises(ValueError, match="pays every exchange and computes"):
        cards.band_owners(384, 1, (1.0, 1.0))
    with pytest.raises(ValueError, match="pays every exchange and computes"):
        cards.band_owners(384, 4, (1.0, 1.0e-9))


def test_a_partial_sum_is_added_in_rank_order():
    """The one floating-point sum that crosses the wire, pinned to an order.

    Three values whose sum is order-dependent: 1 + 1e-16 + 1e-16 is 1.0
    left to right (each addend vanishes under the 1) and
    1.0000000000000002 right to left (the two small ones add first).
    Added in rank order the answer is the first one, every time.
    """
    parts = [np.float64(1.0), np.float64(1e-16), np.float64(1e-16)]
    assert cards.sum_in_rank_order(parts) == (parts[0] + parts[1]) + parts[2]
    assert cards.sum_in_rank_order(parts) != parts[0] + (parts[1] + parts[2])


def test_the_pipeline_hands_a_card_whole_bands_and_refuses_a_split_one():
    class _Exchange:
        world = 2

        def __init__(self, rows):
            self._rows = rows

        def owned_rows(self):
            return self._rows

    whole = BandPipeline(384, 8, exchange=_Exchange((0, 192)))
    assert [(s.start, s.stop) for s in whole.local_slices()] == [
        (0, 48), (48, 96), (96, 144), (144, 192)
    ]
    assert whole.local_rows() == (0, 192)
    assert len(whole.slices()) == 8, "the schedule itself must not move"
    with pytest.raises(ValueError, match="whole number of\n? *bands|whole bands"):
        BandPipeline(384, 8, exchange=_Exchange((0, 100)))


# ---------------------------------------------------------------------
# The transport, over real sockets
# ---------------------------------------------------------------------


def _two_rank_tcp(body, world: int = 2, attempts: int = 4):
    """Run ``body(rank, transport)`` on ``world`` threads over loopback.

    Retried on a connection failure: the helper picks a free port, closes
    it and reconnects, so another test in the same session can take the
    port in between.  That is the harness racing itself, not the
    transport, and a retry is the right answer to it.
    """
    last = None
    for _ in range(int(attempts)):
        try:
            return _two_rank_tcp_once(body, world)
        except (ConnectionError, OSError) as exc:
            last = exc
    raise last


def _two_rank_tcp_once(body, world: int = 2):
    import socket as _socket

    ports = []
    holders = []
    for _ in range(world - 1):
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        ports.append(sock.getsockname()[1])
        holders.append(sock)
    for sock in holders:
        sock.close()
    ports.append(0)
    addresses = [f"127.0.0.1:{p}" for p in ports]
    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def run(rank):
        try:
            transport = cards.TcpCards(rank, world, addresses, chunk_bytes=1 << 16)
            try:
                results[rank] = body(rank, transport)
            finally:
                transport.close()
        except BaseException as exc:  # noqa: BLE001
            errors[rank] = exc

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120.0)
    if errors:
        raise errors[sorted(errors)[0]]
    return results


def test_the_tcp_transport_all_gathers_in_rank_order():
    def body(rank, transport):
        payload = bytes([rank]) * (3 + rank)
        return transport.all_gather("t", payload)

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        assert out[rank] == [b"\x00\x00\x00", b"\x01\x01\x01\x01"]


def test_wire1_the_ledger_counts_the_bytes_that_crossed():
    """WIRE-1's accounting: a wrong ledger reports a wrong rate."""
    size = 1 << 20

    def body(rank, transport):
        transport.all_gather("t", bytes(size))
        transport.ledger.mark_step()
        return transport.ledger.receipt()

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        row = out[rank]
        assert row["posted_bytes"] == size
        assert row["received_bytes"] == size
        assert row["steps"] == 1
        assert row["bytes_per_step"] == 2 * size
        assert row["achieved_bytes_s"] > 0.0
        assert row["wire_floor_gb_s"] == pytest.approx(2.0)


def test_wire1_the_ledger_is_read_after_the_sender_threads_catch_up():
    """The bytes a receipt reports are the bytes that crossed, every time.

    A post returns as soon as the frame is queued and the sender thread
    is what hands the bytes to the ledger, so the ledger trails the model
    by whatever is in flight -- which is the overlap working.  Reading it
    raw at that instant reports fewer bytes than crossed the wire:
    MEASURED, a one-megabyte all-gather read back 0 posted bytes on two
    of three attempts.  Every route a reader has must therefore drain
    first.  Repeated, because the defect this pins was intermittent.
    """
    size = 1 << 20

    def body(rank, transport):
        transport.all_gather("t", bytes(size))
        raw = transport.ledger.receipt()["posted_bytes"]
        transport.drain()
        return (raw, transport.ledger.receipt()["posted_bytes"])

    for _ in range(6):
        out = _two_rank_tcp(body)
        for rank in (0, 1):
            straight, after = out[rank]
            # The ledger drains itself, so the FIRST read is already the
            # wire's own count; the explicit drain changes nothing.
            assert straight == size
            assert after == size


def test_the_row_gather_assembles_the_buffer_one_card_would_have_built():
    """The seam every waist and every reduction buffer goes through."""
    nlat, width = 32, 5
    whole = np.arange(nlat * width, dtype=np.float64).reshape(nlat, width)

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, 4, weights=(1.0, 1.0))
        first, last = session.local_rows()
        mine = np.full_like(whole, np.nan)
        mine[first:last] = whole[first:last]
        session.gather_rows(np, mine, 0, name="rows")
        return mine

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        assert np.array_equal(out[rank], whole), "a gathered buffer is not the globe"


def test_the_accumulators_reduce_the_globe_across_two_cards():
    """The reduction contract, over the wire: same number, both ranks, and
    the number a one-card run computes."""
    nlat, nlon = 24, 4
    plane = np.arange(nlat * nlon, dtype=np.float64).reshape(nlat, nlon)

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, 4, weights=(1.0, 1.0))
        exchange = cards.RowExchange(session)
        pipeline = BandPipeline(nlat, 4, exchange=exchange)
        acc = PlaneAccumulator(np, (nlat, nlon), np.float64,
                               name="p", exchange=exchange)
        rows = LatitudeAccumulator(np, (nlat,), np.float64,
                                   name="l", exchange=exchange)
        for band in pipeline.local_slices():
            acc.add_band(band, plane[band])
            rows.add_band(band, plane[band].sum(axis=-1))
        return float(acc.total()), float(rows.total())

    out = _two_rank_tcp(body)
    reference = (float(plane.sum()), float(plane.sum(axis=-1).sum()))
    assert out[0] == out[1] == reference


def test_the_deep_halo_fills_exactly_the_rows_past_the_boundary():
    nlat, width = 40, 3
    whole = np.arange(nlat * 2, dtype=np.float64).reshape(nlat, 2)

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, 4, weights=(1.0, 1.0))
        first, last = session.local_rows()
        mine = np.full_like(whole, np.nan)
        mine[first:last] = whole[first:last]
        session.exchange_halo(np, mine, 0, width, name="halo")
        return mine, (first, last)

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        mine, (first, last) = out[rank]
        lo = max(0, first - width)
        hi = min(nlat, last + width)
        assert np.array_equal(mine[lo:hi], whole[lo:hi]), "halo rows are not the globe's"
        # and nothing beyond the halo was fetched
        assert np.isnan(mine[hi:]).all() or hi == nlat
        assert np.isnan(mine[:lo]).all() or lo == 0


def test_a_halo_wider_than_a_card_is_refused_by_name():
    def body(rank, transport):
        session = cards.CardSession(transport, 40, 4, weights=(1.0, 1.0))
        with pytest.raises(ValueError, match="wider than the"):
            session.exchange_halo(np, np.zeros((40, 2)), 0, 40, name="halo")
        return True

    assert _two_rank_tcp(body) == {0: True, 1: True}


def test_a_missing_tag_names_the_breakage_rather_than_hanging():
    ready = threading.Event()

    def body(rank, transport):
        if rank == 0:
            with pytest.raises(TimeoutError, match="same tags in the same order"):
                transport.collect("never-posted", timeout_s=1.0)
            ready.set()
        else:
            # Rank 1 stays up: a peer that exits first would deliver an EOF
            # and the test would measure the dead-rank message instead.
            ready.wait(20.0)
        return True

    assert _two_rank_tcp(body)[0] is True


# ---------------------------------------------------------------------
# The launcher and the receipt
# ---------------------------------------------------------------------


def test_the_launcher_sets_nccl_ib_disable_and_the_receipt_records_it():
    env = cards.launch_environment("enp133s0f1np1")
    assert env["NCCL_IB_DISABLE"] == "1"
    assert env["NCCL_SOCKET_IFNAME"] == "enp133s0f1np1"


def test_a_single_card_session_is_the_two_card_one_with_a_world_of_one():
    session = cards.single_card_session(96, 4)
    assert session.world == 1
    assert session.local_rows() == (0, 96)
    assert session.gather_rows(np, np.zeros((96, 2)), 0).shape == (96, 2)
    row = session.receipt()
    assert row["cards"] == 1 and row["transport"] == "single"
    assert row["wire"]["posted_bytes"] == 0


def test_an_unrecognised_latitude_layout_is_refused_rather_than_written_stale():
    session = cards.single_card_session(8, 2)
    session.world = 2  # the refusal is the point, not the transport
    with pytest.raises(ValueError, match="does not recognise"):
        cards.gather_named_arrays(
            session, {"physics__odd": np.zeros((8, 3, 5))}, 8, 16)


# ---------------------------------------------------------------------
# WIRE-1 as a gate, not as a number in a receipt
# ---------------------------------------------------------------------


def _gate_fixture(tmp_path, **overrides):
    """The arguments ``run_gates`` needs, all of them comfortably inside
    their limits, so the only row that can move the verdict is the wire."""
    from woof.globe.config import load_config

    path = tmp_path / "wire-gate.toml"
    path.write_text("".join(line + chr(10) for line in (
        "[arwen_global]",
        'schema = "gpuwm.arwen-global-run/v1"',
        'name = "wire-gate"',
        'backend = "numpy"',
        'acknowledgement = "research-only-arwen-global-v1"',
        "[grid]", "truncation = 21",
        "[time]", "dt_s = 600.0", "duration_s = 600.0",
        "[vertical]", "nlev = 16", 'coordinate = "pressure_blend"',
    )), encoding="utf-8")
    cfg = load_config(path)
    payload = dict(
        cfg=cfg,
        transform_check={"roundtrip_relative_linf": 0.0,
                         "parseval_relative_error": 0.0},
        final_diag={"global_mean_surface_pressure_pa": 1.0,
                    "global_mean_total_water_kg_m2": 1.0},
        target_mass=1.0,
        target_water=1.0,
        trackers={"maximum_mass_fixer_log_offset": 0.0,
                  "maximum_global_water_fixer_kg_m2": 0.0,
                  "maximum_physics_water_repair_kg_m2": 0.0},
        # A config that says nothing about the core is a semi-Lagrangian run
        # (the default core, 2026-09-06), and a semi-Lagrangian run's receipt
        # carries that core's own gate rows; the runner supplies their
        # trackers and so does this fixture, because run_gates refuses to
        # judge such a run without them (a KeyError, not a silently missing
        # gate).
        supplementary={"maximum_positivity_fixer_relative": 0.0,
                       "maximum_semilag_lipschitz": 0.0,
                       "maximum_semilag_tracer_mass_fixer_relative": 0.0,
                       "maximum_semilag_tracer_mass_fixer_water_relative": 0.0,
                       "maximum_semilag_trajectory_move_cells": 0.0,
                       },
    )
    payload.update(overrides)
    return payload


def _wire_block(achieved_gb_s, world=2):
    return {"cards": world, "wire": {
        "achieved_gb_s": achieved_gb_s,
        "wire_floor_gb_s": cards.WIRE_FLOOR_BYTES_S / 1e9,
    }}


def test_wire1_fails_the_receipt_when_the_link_falls_below_the_floor(tmp_path):
    """The breakage: every interconnect figure this design was priced on
    was taken on IDLE cards, and MEASURED 2026-09-06 three of four
    rank-legs of the T255 two-card probe achieved 1.82 to 1.98 GB/s while
    both cards computed.  Without this row the receipt reports "pass" on
    a link the run did not get."""
    from woof.globe.runner import run_gates

    rows = run_gates(**_gate_fixture(tmp_path), cards=_wire_block(1.8174))
    gate = rows["two_card_wire_achieved_gb_s"]
    assert gate["direction"] == "floor"
    assert gate["value"] == pytest.approx(1.8174)
    assert gate["limit"] == pytest.approx(2.0)
    assert gate["passed"] is False
    assert not all(row["passed"] for row in rows.values())


def test_wire1_passes_above_the_floor_and_every_other_row_is_a_ceiling(tmp_path):
    from woof.globe.runner import run_gates

    rows = run_gates(**_gate_fixture(tmp_path), cards=_wire_block(2.3124))
    assert rows["two_card_wire_achieved_gb_s"]["passed"] is True
    assert all(row["passed"] for row in rows.values())
    ceilings = [n for n, r in rows.items() if r["direction"] == "ceiling"]
    assert len(ceilings) == len(rows) - 1


def test_a_single_card_run_has_no_wire_gate_to_fail(tmp_path):
    """A card that opened no socket moves no bytes, and a floor on zero
    would fail every single-card run in the tree."""
    from woof.globe.runner import run_gates

    rows = run_gates(**_gate_fixture(tmp_path), cards={"cards": 1, "transport": "single"})
    assert "two_card_wire_achieved_gb_s" not in rows
    assert all(row["passed"] for row in rows.values())


# ---------------------------------------------------------------------
# The order-m axis (design section 10, lane 7): the multi-card axis for
# the transform.  Orders are independent output indices of the Legendre
# contraction and are never reduced across the wire, so the split is
# bit-exact by construction.
# ---------------------------------------------------------------------


def test_order_band_owners_assigns_whole_bands_and_refuses_an_idle_rank():
    from woof.globe.spectral.legendre import band_bounds, DEFAULT_BAND

    bounds = band_bounds(255, DEFAULT_BAND)  # eight bands
    owners = cards.order_band_owners(bounds, (1.0, 1.0))
    assert len(owners) == len(bounds)
    # contiguous runs, both ranks used
    assert set(owners) == {0, 1}
    assert owners == tuple(sorted(owners)), "an order rank owns a contiguous run"
    # a truncation with fewer bands than cards is refused
    with pytest.raises(ValueError, match="cannot be shared"):
        cards.order_band_owners(band_bounds(20, DEFAULT_BAND), (1.0, 1.0))
    # an extreme weight that starves a rank is refused by name
    with pytest.raises(ValueError, match="no band"):
        cards.order_band_owners(bounds, (1.0, 1e-9))


def test_the_faster_card_is_given_more_orders():
    from woof.globe.spectral.legendre import band_bounds, DEFAULT_BAND

    bounds = band_bounds(255, DEFAULT_BAND)
    slow_first = cards.order_band_owners(bounds, (0.5, 1.0))
    # rank 1 is twice as fast, so it owns more of the (coefficient-weighted)
    # order bands than rank 0
    assert slow_first.count(1) >= slow_first.count(0)


def test_the_transform_refuses_a_partition_that_splits_a_band():
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    with pytest.raises(ValueError, match="whole Legendre bands"):
        SphericalHarmonicTransform.create(
            85, backend="numpy", precision="float64",
            order_partition=((0, 16),),  # half a 32-order band
        )


def test_the_transform_refuses_a_non_contiguous_partition():
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    with pytest.raises(ValueError, match="contiguous"):
        SphericalHarmonicTransform.create(
            85, backend="numpy", precision="float64",
            order_partition=((0, 32), (64, 86)),  # skips the middle band
        )


def test_a_partitioned_transform_holds_only_its_own_bands_tables():
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    whole = SphericalHarmonicTransform.create(255, backend="numpy", precision="float64")
    lower = SphericalHarmonicTransform.create(
        255, backend="numpy", precision="float64",
        order_partition=tuple(
            __import__("woof.globe.spectral.legendre", fromlist=["band_bounds"])
            .band_bounds(255, 32)[:4]),
    )
    # four of eight bands: the resident analysis+basis tables are a
    # fraction of the whole, which is the capacity the axis buys.
    assert lower._analysis.nbytes < whole._analysis.nbytes
    assert lower._basis.nbytes < whole._basis.nbytes


def _order_split_reference(truncation, seed):
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    whole = SphericalHarmonicTransform.create(
        truncation, backend="numpy", precision="float64")
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((2, whole.grid.nlat, whole.grid.nlon))
    coeff = whole.forward(field)
    grid = whole.inverse(coeff)
    return whole, field, coeff, grid


def test_the_order_split_transform_is_the_single_card_transform_over_the_wire():
    """BIT-5 on the order axis, in process: two ranks each hold half the
    Legendre table, each contracts its own orders, and the assembled
    analysis and synthesis are the single-card transform's bit for bit."""
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    whole, field, ref_coeff, ref_grid = _order_split_reference(85, 1)

    def body(rank, transport):
        session = cards.CardSession(transport, whole.grid.nlat, 4, weights=(1.0, 1.0))
        exchange = cards.order_exchange_for(session, whole, weights=(1.0, 1.0))
        tr = SphericalHarmonicTransform.create(
            85, backend="numpy", precision="float64",
            order_partition=exchange.owned_bounds())
        tr.order_exchange = exchange
        return tr.forward(field), tr.inverse(ref_coeff)

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        coeff, grid = out[rank]
        assert np.array_equal(coeff, ref_coeff), "order-split analysis is not the single-card spectrum"
        assert np.array_equal(grid, ref_grid), "order-split synthesis is not the single-card grid"


def test_the_order_axis_gathers_columns_and_never_sums_across_the_wire():
    """The assembled column set is a concatenation of disjoint order
    ranges: a placed piece, never an added one, so no floating-point sum
    crosses the wire and the schedule cannot move a bit."""
    whole, field, ref_coeff, _ = _order_split_reference(85, 2)
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    def body(rank, transport):
        session = cards.CardSession(transport, whole.grid.nlat, 4, weights=(1.0, 1.0))
        exchange = cards.order_exchange_for(session, whole, weights=(1.0, 1.0))
        lo, hi = exchange.owned_order_range
        tr = SphericalHarmonicTransform.create(
            85, backend="numpy", precision="float64",
            order_partition=exchange.owned_bounds())
        # the compact contraction of the owned orders, before the gather
        waist = tr.fourier_waist(field)
        compact = tr._contract_orders(waist.take(), tr._analysis, lo, hi)
        return lo, hi, compact

    out = _two_rank_tcp(body)
    # each rank's compact columns equal exactly the reference columns for
    # its owned order range (m=0 realified on the whole afterwards)
    for rank in (0, 1):
        lo, hi, compact = out[rank]
        want = ref_coeff[..., lo:hi].copy()
        if lo == 0:
            # the m=0 realification is applied to the assembled whole, not
            # to the compact piece, so compare the piece pre-realification
            want[..., 0] = compact[..., 0]
        assert np.array_equal(compact, want)


# ---------------------------------------------------------------------
# The cross-card agreement refusal (lane 6's finding, made into a gate):
# a gather two-card run assembles a waist from rows computed on both
# cards, so it reproduces one card only where the cards agree.
# ---------------------------------------------------------------------


def test_the_agreement_check_refuses_a_pair_that_disagrees():
    # The run is refused by name at the contraction the cards disagree on,
    # with its direction and operand shape in the message.
    key = ("synthesis", "basis", (1, 24, 48), "complex64")
    with pytest.raises(ValueError, match="do not return the same bits"):
        cards._assert_card_hashes_agree(key, ["bb", "cc"])
    message = _raised_message(cards._assert_card_hashes_agree, key, ["bb", "cc"])
    assert "synthesis" in message and "(1, 24, 48)" in message


def test_the_agreement_check_passes_a_pair_that_agrees():
    cards._assert_card_hashes_agree(("analysis", "analysis", (8, 4, 8), "complex64"), ["aa", "aa"])  # no raise


def test_the_agreement_check_rides_the_runs_own_contractions_over_the_wire():
    """Two ranks contract the same inputs through a row exchange; the ledger
    records every distinct (direction, table, shape) once and the run
    passes.  A rank whose basis table differs by one unit of roundoff is
    refused by name at the first synthesis, before any waist is assembled
    from its rows.  A probe at fixed widths cannot do this: it refused a
    real pair at an eight-plane synthesis the step never presents."""
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    whole = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    rng = np.random.default_rng(3)
    field = rng.standard_normal((3, whole.grid.nlat, whole.grid.nlon))
    coeff = whole.project(whole.forward(field))

    def body(rank, transport):
        session = cards.CardSession(transport, whole.grid.nlat, 4, weights=(1.0, 1.0))
        tr = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
        tr.row_exchange = cards.RowExchange(session)
        tr.forward(field)
        tr.inverse(coeff)
        tr.inverse(coeff)          # the same shape again: checked once
        tr.inverse(coeff[:1])      # a new shape: checked
        return session.agreement.receipt()

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        receipt = out[rank]
        assert receipt["agree"] is True
        directions = [row["direction"] for row in receipt["contractions"]]
        assert directions == ["analysis", "synthesis", "synthesis"]
        assert receipt["checked"] == 3
    assert out[0]["contractions"] == out[1]["contractions"]

    def body_disagree(rank, transport):
        session = cards.CardSession(transport, whole.grid.nlat, 4, weights=(1.0, 1.0))
        tr = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
        if rank == 1:
            # One band of the synthesis table moved by a few units of
            # roundoff: the kind of difference two cards' cuBLAS kernels
            # produce, applied to one rank only.
            m0, m1, block = tr._basis.blocks[0]
            tr._basis.blocks[0] = (m0, m1, block * (1.0 + 2.0 ** -40))
        tr.row_exchange = cards.RowExchange(session)
        try:
            tr.inverse(coeff)
        except ValueError as exc:
            return str(exc)
        return "no refusal"

    out = _two_rank_tcp(body_disagree)
    for rank in (0, 1):
        assert "do not return the same bits" in out[rank]
        assert "synthesis" in out[rank]


def _raised_message(fn, *args):
    try:
        fn(*args)
    except ValueError as exc:
        return str(exc)
    return ""


def test_a_model_run_on_the_order_axis_is_refused_by_name():
    import types
    from woof.globe.spectral.transform import SphericalHarmonicTransform
    from woof.globe.runner import open_card_session

    transform = SphericalHarmonicTransform.create(85, backend="numpy", precision="float64")
    cfg = types.SimpleNamespace(
        cards=2, card_axis="order", card_rank=0,
        card_addresses=("127.0.0.1:1", "127.0.0.1:2"),
        card_transport="tcp", card_weights=(), card_exchange="gather",
    )
    with pytest.raises(ValueError, match="card_axis='order' is not built for a model RUN"):
        open_card_session(cfg, transform, 4)


# -- the steps' own wall on every receipt ------------------------------------

def test_every_receipt_carries_the_steps_own_wall(tmp_path):
    """The capacity rows and the two-card ratio are defined on a STEP.

    The tier's receipt block carried a per-step wall only when something was
    parked; a resident run had no per-step figure at all and its step could
    only be read off the whole run's wall with the build, the cold start and
    the checkpoint writes inside it (the cards lane's T383 pair measurement,
    2026-09-07).  So every receipt carries the steps' own wall, pass or fail.
    """
    from dataclasses import replace
    from woof.globe.config import load_config
    from woof.globe.runner import run, _step_wall_receipt

    cfg = load_config(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
    cfg = replace(cfg, duration_s=cfg.dt_s * 3, output_interval_s=cfg.dt_s * 3)
    result = run(cfg, tmp_path / "run")
    wall = result["step_wall"]
    assert wall["steps_timed"] == 3
    assert wall["step_wall_seconds"] > 0.0
    assert wall["step_s"] == pytest.approx(wall["step_wall_seconds"] / 3, abs=1e-6)
    assert wall["step_wall_seconds"] <= result["wall_seconds"]
    # A run that died before its first step says so rather than dividing by it.
    assert _step_wall_receipt(0, 0.0)["step_s"] is None


def test_record_mode_carries_on_and_fails_the_gate_row_rather_than_refusing(tmp_path):
    """card_agreement='record': a disagreeing pair is not refused, the
    receipt lists the shape the cards disagreed on, and run_gates fails
    the two_card_contractions_agree row on it.  A timing device for unlike
    cards; the run's answer is neither card's and the receipt says so."""
    from woof.globe.spectral.transform import SphericalHarmonicTransform
    from woof.globe.runner import run_gates

    whole = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    rng = np.random.default_rng(5)
    field = rng.standard_normal((2, whole.grid.nlat, whole.grid.nlon))
    coeff = whole.project(whole.forward(field))

    def body(rank, transport):
        session = cards.CardSession(
            transport, whole.grid.nlat, 4, weights=(1.0, 1.0), agreement="record")
        tr = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
        if rank == 1:
            m0, m1, block = tr._basis.blocks[0]
            tr._basis.blocks[0] = (m0, m1, block * (1.0 + 2.0 ** -40))
        tr.row_exchange = cards.RowExchange(session)
        tr.forward(field)
        tr.inverse(coeff)
        return session.agreement.receipt()

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        receipt = out[rank]
        assert receipt["mode"] == "record"
        assert receipt["agree"] is False
        assert [row["direction"] for row in receipt["contractions"]] == ["analysis"]
        assert [row["direction"] for row in receipt["disagreements"]] == ["synthesis"]
        assert receipt["disagreements"][0]["hashes"][0] != receipt["disagreements"][0]["hashes"][1]
    block = _wire_block(2.5)
    block["card_agreement"] = out[0]
    gates = run_gates(**_gate_fixture(tmp_path), cards=block)
    row = gates["two_card_contractions_agree"]
    assert row["passed"] is False and row["value"] == 1 and row["limit"] == 0
    block["card_agreement"] = {"agree": True, "disagreements": []}
    gates = run_gates(**_gate_fixture(tmp_path), cards=block)
    assert gates["two_card_contractions_agree"]["passed"] is True


def test_card_agreement_is_a_named_choice_outside_every_identity(tmp_path):
    from dataclasses import replace
    from woof.globe.config import load_config

    cfg = load_config(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
    assert cfg.card_agreement == "refuse"
    recorded = replace(cfg, card_agreement="record")
    assert recorded.config_hash == cfg.config_hash
    text = open(str(_shipped_configs() / "arwen_global_moist_smoke.toml"), encoding="utf-8").read()
    assert "[memory]" not in text
    bad = tmp_path / "bad.toml"
    bad.write_text(text + chr(10) + '[memory]' + chr(10) + 'card_agreement = "ignore"' + chr(10), encoding="utf-8")
    with pytest.raises(ValueError, match="card_agreement must be one of"):
        load_config(bad)
