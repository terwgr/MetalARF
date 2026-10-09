"""The four engine memory levers, their doors, and the two bit gates the
scale-out design rests on (FFT-1 and CHUNK-1) on the numpy backend.

The determinism spine is proved on the CPU before a card is touched: the
band schedule, the row-locality of the longitude FFT, and the field-stack
width of the Legendre contraction.  The card runs the same gates at
production truncations through
``tools/arwen_global_fft_band_probe.py`` and
``tools/arwen_global_band_probe.py --spectral-chunks``.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from woof.globe.cli import build_parser
from woof.globe.config import DEFAULT_SPECTRAL_CHUNK, load_config
from woof.globe.dynamics import MoistHybridModel
from woof.globe.runner import build_model_and_cold_state, build_transform
from woof.globe.spectral.backend import get_backend
from woof.globe.spectral.legendre import DEFAULT_BAND
from woof.globe.spectral.transform import SphericalHarmonicTransform

from tools.arwen_global_fft_band_probe import band_edges

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")

#: Every config shipped in the tree, hashed at the tip this table was
#: written on.  The levers arrived with the rule that a config that names
#: none of them keeps the hash it had, so every checkpoint written before
#: the [memory] table existed still restarts.
SHIPPED_CONFIG_HASHES = {
    # the quickstart moved to the default core on 2026-09-06 (sl_si at 300 s,
    # order 16 at 720 s, off-centring 0.55); this is its hash since then
    str(_shipped_configs() / "arwen_global_t255_quickstart.toml"):
        "ce5629409342af640456f621304ae4d09b91f1dbc0171198d45b8757fba8c3fb",
    # the jet48 door is the config of record with its vertical table changed
    # and nothing else, so it went bare on the same day (and, like the
    # record, stopped writing the drain when the merge of 2026-09-07 made
    # both read the core's own order 16 at 720 s: ecd354f3... before that)
    str(_shipped_configs() / "arwen_global_gdas_t255_jet48_24h.toml"):
        "9f6477f2f1b891b04ed59052df59ddd9a8061e10d1758bb80dc25989c2f10b41",
    # the config of record is bare since 2026-09-06 (it runs the shipped
    # default core, the semi-Lagrangian one at 300 s): its hash moved with
    # its content (afec3fa6... while it still wrote order 8 at 2,160 s; the
    # door's own drain since 2026-09-07), and the [memory] table still adds
    # nothing to it
    str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml"):
        "93e5ee7500b8a57a20ef27c8e985561069d9e83d72a3cdf77e05aee8b5a9f69f",
    str(_shipped_configs() / "arwen_global_gdas_t533_24h.toml"):
        "85e60b9c8dcd0f884d533b3b2f61842c5d4bf7f5c808f777b0307127555031f3",
}


def test_shipped_config_hashes_do_not_move_with_the_memory_table():
    for path, expected in SHIPPED_CONFIG_HASHES.items():
        if not Path(path).exists():
            pytest.skip(f"{path} is not in this tree")
        assert load_config(path).config_hash == expected, path


def test_the_defaults_are_the_shipped_engine_values():
    cfg = load_config(CONFIG)
    assert cfg.spectral_chunk == DEFAULT_SPECTRAL_CHUNK == 6
    assert cfg.synthesis_memo is True
    assert cfg.legendre_band == DEFAULT_BAND
    assert cfg.streaming is False
    assert MoistHybridModel.spectral_chunk == cfg.spectral_chunk
    assert MoistHybridModel.synthesis_memo == cfg.synthesis_memo
    assert SphericalHarmonicTransform.legendre_band == cfg.legendre_band
    assert SphericalHarmonicTransform.streaming == cfg.streaming


def test_the_memory_table_is_read_and_is_strict(tmp_path):
    text = Path(CONFIG).read_text(encoding="utf-8")
    named = tmp_path / "named.toml"
    named.write_text(
        text + "\n[memory]\nspectral_chunk = 2\nsynthesis_memo = false\n"
        "legendre_band = 8\nstreaming = true\n",
        encoding="utf-8")
    cfg = load_config(named)
    assert (cfg.spectral_chunk, cfg.synthesis_memo) == (2, False)
    assert (cfg.legendre_band, cfg.streaming) == (8, True)

    typo = tmp_path / "typo.toml"
    typo.write_text(text + "\n[memory]\nspectral_chunck = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="memory"):
        load_config(typo)

    zero = tmp_path / "zero.toml"
    zero.write_text(text + "\n[memory]\nspectral_chunk = 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="spectral_chunk"):
        load_config(zero)


def test_the_bit_neutral_levers_leave_the_identity_at_every_value():
    """A run that moves a MEASURED bit-neutral lever keeps its lineage.

    synthesis_memo: the shipped bit-neutrality gate, and a ten-step T255
    native A/B on the RTX 5070 Ti (2026-09-06) whose 126 checkpoint
    arrays, metadata included, are byte-identical.
    streaming: the streamed analysis and synthesis of a single plane and
    of a forty-level stack at T255 float32 are byte-identical to the
    resident-table calls (MEASURED 2026-09-06, RTX 5070 Ti).
    """
    cfg = load_config(CONFIG)
    for field, value in (("synthesis_memo", False), ("streaming", True)):
        moved = replace(cfg, **{field: value})
        assert moved.config_hash == cfg.config_hash, field
        assert field not in moved.config_identity, field


def test_the_arithmetic_levers_are_carried_when_they_are_moved():
    """spectral_chunk is the Legendre GEMM's M and legendre_band is its
    batch count, and both are MEASURED to move bits, so a config that
    moves either does not share a checkpoint lineage with one that does
    not, while the shipped values keep every hash ever written."""
    cfg = load_config(CONFIG)
    assert "spectral_chunk" not in cfg.config_identity
    assert "legendre_band" not in cfg.config_identity
    for field, value in (("spectral_chunk", 2), ("legendre_band", 16)):
        moved = replace(cfg, **{field: value})
        assert moved.config_identity[field] == value, field
        assert moved.config_hash != cfg.config_hash, field


def test_the_transform_identity_carries_a_non_default_legendre_band():
    """MEASURED 2026-09-06, RTX 5070 Ti, T255 float32: the analysis of a
    single plane -- the shape the dycore transforms ln ps in -- differs by
    6.2e-06 between band 32 and bands 8 and 16, and a ten-step T255 native
    run at band 16 left 66 of the 125 checkpoint state arrays differing from the
    band-32 run.  Two transforms that compute different bits may not carry
    one identity hash."""
    default = SphericalHarmonicTransform.create(15, backend="numpy")
    moved = SphericalHarmonicTransform.create(
        15, backend="numpy", legendre_band=8)
    streamed = SphericalHarmonicTransform.create(
        15, backend="numpy", streaming=True)
    assert "legendre_band" not in default.identity
    assert moved.identity["legendre_band"] == 8
    assert moved.identity_hash != default.identity_hash
    assert streamed.identity_hash == default.identity_hash


def test_the_levers_reach_the_transform_and_the_model():
    """The defect these doors close: build_transform passed neither
    transform lever and build_model_and_cold_state passed neither model
    lever, so a named value was parsed and never applied."""
    cfg = replace(
        load_config(CONFIG),
        legendre_band=8, streaming=True, spectral_chunk=2,
        synthesis_memo=False,
    )
    transform = build_transform(cfg)
    assert transform.legendre_band == 8
    assert transform.streaming is True
    model, _ = build_model_and_cold_state(cfg)
    assert model.spectral_chunk == 2
    assert model.synthesis_memo is False
    assert model._memo is None


def test_the_run_door_carries_the_levers_and_refuses_a_value_it_cannot_run():
    from woof.globe.cli import _memory_lever_overrides

    parser = build_parser()
    args = parser.parse_args([
        "run", CONFIG, "--spectral-chunk", "2", "--synthesis-memo", "off",
        "--legendre-band", "8", "--streaming", "on",
    ])
    assert _memory_lever_overrides(args) == {
        "spectral_chunk": 2, "synthesis_memo": False,
        "legendre_band": 8, "streaming": True,
    }
    bare = parser.parse_args(["run", CONFIG])
    assert _memory_lever_overrides(bare) == {}

    refused = parser.parse_args(["run", CONFIG, "--spectral-chunk", "0"])
    with pytest.raises(ValueError, match="spectral-chunk"):
        _memory_lever_overrides(refused)


def test_band_edges_are_a_pure_function_of_nlat_and_the_band_count():
    """Rule 1 of the design: the band count is a streaming granularity.
    The edges may not depend on the card count, the order they run in, or
    anything else, so they are floor(k x nlat / B) and nothing more."""
    for nlat in (384, 576, 801):
        for bands in (1, 2, 4, 8, 16, 32):
            edges = band_edges(nlat, bands)
            assert edges[0][0] == 0
            assert edges[-1][1] == nlat
            assert all(a == previous for (a, _), (_, previous)
                       in zip(edges[1:], edges[:-1]))
            assert sum(b - a for a, b in edges) == nlat
            assert edges == band_edges(nlat, bands)


@pytest.mark.parametrize("truncation", (21, 31))
@pytest.mark.parametrize("bands", (2, 4, 8))
def test_fft_1_the_longitude_transform_is_row_local(truncation, bands):
    """Gate FFT-1 on the numpy backend.  The whole bit-identity claim of
    the scale-out design rests on this: the transform length never moves,
    only the batch, so a field fed band by band returns the whole-field
    bits."""
    backend = get_backend("numpy", "float64")
    xp = backend.xp
    transform = SphericalHarmonicTransform.create(truncation, backend="numpy")
    nlat, nlon = transform.grid.shape
    t1 = truncation + 1
    rng = np.random.default_rng(20260906)
    field = xp.asarray(rng.standard_normal((3, nlat, nlon)))
    coeff = xp.asarray(
        rng.standard_normal((3, nlat, t1))
        + 1j * rng.standard_normal((3, nlat, t1)))

    whole = xp.fft.rfft(field, axis=-1)
    spectrum = xp.zeros((3, nlat, nlon // 2 + 1), dtype=whole.dtype)
    xp.multiply(nlon, coeff, out=spectrum[..., :t1])
    whole_inverse = xp.fft.irfft(spectrum, n=nlon, axis=-1)

    banded = xp.empty_like(whole)
    banded_inverse = xp.empty_like(whole_inverse)
    for a, b in band_edges(nlat, bands):
        banded[..., a:b, :] = xp.fft.rfft(field[..., a:b, :], axis=-1)
        sub = xp.zeros((3, b - a, nlon // 2 + 1), dtype=whole.dtype)
        xp.multiply(nlon, coeff[..., a:b, :], out=sub[..., :t1])
        banded_inverse[..., a:b, :] = xp.fft.irfft(sub, n=nlon, axis=-1)

    assert np.array_equal(whole, banded)
    assert np.array_equal(whole_inverse, banded_inverse)


class _ChunkShim:
    """Only the two attributes ``MoistHybridModel._chunked`` reads, so the
    loop under test is the dycore's own loop and not a transcription."""

    def __init__(self, transform, chunk):
        self.transform = transform
        self.spectral_chunk = chunk


def _chunk_arms(transform, nlev, chunks, fields=12):
    xp = transform.backend.xp
    nlat, nlon = transform.grid.shape
    rng = np.random.default_rng(1)
    stack = xp.asarray(rng.standard_normal((fields, nlev, nlat, nlon)))
    one = _ChunkShim(transform, 1)
    forward = MoistHybridModel._chunked(one, transform.forward, stack)
    spectral = transform.project(forward)
    inverse = MoistHybridModel._chunked(one, transform.inverse, spectral)
    out = {}
    for chunk in chunks:
        shim = _ChunkShim(transform, chunk)
        out[chunk] = (
            np.array_equal(
                forward, MoistHybridModel._chunked(
                    shim, transform.forward, stack)),
            np.array_equal(
                inverse, MoistHybridModel._chunked(
                    shim, transform.inverse, spectral)),
        )
    return out


@pytest.mark.parametrize("chunk", (1, 2, 6, 12))
def test_chunk_1_the_stack_width_holds_on_the_shipped_level_ladder(chunk):
    """Gate CHUNK-1 at the level count the model runs: forty levels make
    the synthesis GEMM's M at least forty at every chunk width, and the
    bits hold."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy")
    analysis, synthesis = _chunk_arms(transform, 40, (chunk,))[chunk]
    assert analysis, "analysis moved at spectral_chunk=%d" % chunk
    assert synthesis, "synthesis moved at spectral_chunk=%d" % chunk


def test_chunk_1_is_not_guaranteed_across_vertical_ladders():
    """...and why the chunk therefore stays in the config identity.

    The chunk is the Legendre GEMM's M dimension and BLAS blocks on M, so
    whether a width moves bits depends on the library and the ladder.
    MEASURED 2026-09-06: on numpy 2.2.6 at T21 float64 the synthesis moves
    1 ulp (2.22e-16 against values of order 1) at two, five and ten levels
    and holds at forty; on the CPU host's numpy 2.5.2 the same sweep
    moves at five levels only.  A configuration can ask for a short ladder
    today -- the pressure_blend coordinate ships with six levels -- so the
    identity carries the chunk rather than resting on one host's BLAS.

    What this asserts is the part that does not move with the library: the
    sweep is run, and every width that disagrees with the reference is a
    width whose config identity differs from the reference's.
    """
    transform = SphericalHarmonicTransform.create(21, backend="numpy")
    cfg = load_config(CONFIG)
    for nlev in (2, 5, 10, 40):
        arms = _chunk_arms(transform, nlev, (1, 2, 3, 4, 6, 12))
        for chunk, (_, synthesis) in arms.items():
            if synthesis or chunk == 1:
                continue
            assert replace(cfg, spectral_chunk=chunk).config_hash != replace(
                cfg, spectral_chunk=1).config_hash, (nlev, chunk)


def test_the_device_peak_hook_reads_the_pool_its_allocator_spends():
    """The measurement defect this closes: the hook read
    ``get_default_memory_pool()``, so a run under the driver's async pool
    reported 0.00 GiB -- zero exactly where the peak was the reason the
    pool had been swapped."""
    from woof.globe.device_memory import DevicePeakTracker

    class _Pool:
        def __init__(self):
            self.live = 0

        def malloc(self, size):
            self.live += size
            return size

        def used_bytes(self):
            return self.live

        def total_bytes(self):
            return self.live

    pool = _Pool()
    tracker = DevicePeakTracker(pool)
    tracker(1024)
    tracker(2048)
    row = tracker.receipt()
    assert row["peak_used_bytes"] == 3072
    assert row["pool"] == "_Pool"
    assert "_Pool" in row["measures"]


# --- the sizer prices what the levers actually allocate -----------------
#
# The door moves the levers BEFORE the run is sized (cli._run), for the
# stated reason that a lever the door reads and the estimate does not is
# a flag that was parsed and ignored.  The estimate read neither: it
# priced the Legendre tables at DEFAULT_BAND and as resident whatever the
# config said, so `--streaming on` -- the entry point documented for a
# truncation whose tables do not fit -- was refused on table bytes a
# streamed run never allocates, and a wider band was under-charged in the
# admitting direction.

def _sized(**overrides):
    from woof.globe.sizing import estimate_global_memory
    cfg = replace(load_config(CONFIG), backend="cupy", truncation=799,
                  **overrides)
    return estimate_global_memory(cfg)


def test_the_sizer_prices_a_streamed_run_without_a_resident_table():
    """A streamed transform holds no table (transform.__post_init__ builds
    StreamedLegendreTable, whose own contract is 'NOTHING between calls'),
    so charging it three resident tables refuses the truncation the lever
    exists to reach."""
    resident = _sized()
    streamed = _sized(streaming=True)
    assert streamed.legendre_table_bytes < resident.legendre_table_bytes / 5
    assert streamed.legendre_build_bytes == 0
    assert streamed.device_peak_bytes < resident.device_peak_bytes
    # what one streamed contraction holds, from the transform's own
    # formula: band x (T+1) x nlat of float64 recurrence, float64 Gram
    # rows, the cast and the expansion.
    band, nlat, t1 = DEFAULT_BAND, streamed.nlat, streamed.truncation + 1
    itemsize = streamed.float_itemsize
    assert streamed.legendre_table_bytes == band * t1 * nlat * (16 + 2 * itemsize)


def test_the_sizer_prices_the_band_the_run_will_build():
    """packed_table_elements pads the triangle up to whole bands, so the
    table is bigger at a wider band.  Pricing every run at the shipped 32
    under-charges band 128 by 1.5 GiB at T799, in the admitting
    direction."""
    narrow = _sized(legendre_band=8)
    shipped = _sized()
    wide = _sized(legendre_band=128)
    assert narrow.legendre_table_bytes < shipped.legendre_table_bytes
    assert shipped.legendre_table_bytes < wide.legendre_table_bytes
    assert wide.device_peak_bytes > shipped.device_peak_bytes


def test_a_run_with_no_memo_does_not_carry_the_memo_in_its_resident_set():
    """MoistHybridModel builds no SynthesisMemo when the lever is off, so
    its entries are in no resident set.  The memo is released before every
    physics call, so this moves the resident and host figures and not the
    peak the door admits on."""
    on = _sized()
    off = _sized(synthesis_memo=False)
    assert off.memo_grid_bytes == 0
    assert on.memo_grid_bytes > 0
    assert off.resident_bytes == on.resident_bytes - on.memo_grid_bytes
    assert off.device_peak_bytes == on.device_peak_bytes


def test_the_default_estimate_is_the_one_that_was_calibrated():
    """Every figure the fit was taken on stays exactly where it was."""
    from woof.globe.sizing import (
        estimate_global_memory, legendre_build_bytes, legendre_runtime_bytes,
    )
    cfg = replace(load_config(CONFIG), backend="cupy", truncation=533)
    est = estimate_global_memory(cfg)
    assert est.legendre_table_bytes == legendre_runtime_bytes(
        533, est.nlat, est.float_itemsize)
    assert est.legendre_build_bytes == legendre_build_bytes(
        533, est.nlat, est.float_itemsize)
    assert est.legendre_band == DEFAULT_BAND and est.streaming is False
