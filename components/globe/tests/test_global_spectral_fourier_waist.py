"""Gate WAIST-1: the Fourier waist returns the resident transform's bits.

The scale-out design cuts grid space into latitude bands and joins it to
a whole spectral space at a full-latitude Fourier buffer.  Everything
downstream of that -- the band pipeline, the host spill, the second card
-- inherits its bit-identity claim from this one seam, so this file
holds the seam to the resident transform at every band count and every
stack width the dycore transforms.

The claim in one line: the FFT along longitude is row-local, so it can
be fed and drained band by band; the Legendre contraction is not, so it
is never split and always sees ``K = N = nlat``.

Numpy at T21/T31 here (the determinism spine, before a card is
touched).  ``tools/arwen_global_waist_probe.py`` runs the same
comparisons on the real card at T255/T383/T533.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.globe.spectral.transform import (
    FourierWaist,
    SphericalHarmonicTransform,
    latitude_band_edges,
)
from woof.globe.spectral.vector import VorticityDivergenceOperator


#: Band counts, including uneven ones and counts that exceed the latitudes.
BANDS = (1, 2, 3, 4, 5, 7, 8, 16, 32, 100)

#: Every leading shape a grid field carries into the transform in
#: ``MoistHybridModel.step``, read off the call sites at the tip:
#: a single plane (``forward(log ps)``, ``inverse(log_surface_pressure)``),
#: a levelled field (``inverse(divergence)``, ``forward(bernoulli)``), a
#: spectral chunk of a stacked synthesis (``_chunked``, up to
#: ``spectral_chunk`` rows), the vector pair (``vordiv_from_wind``,
#: ``gradient(stack([psi, chi]))``) and the advection's stacked flux pair
#: (``_scalar_tendency`` -> ``_wind_fourier``).  ``nlev`` is cut to 4 here
#: because the seam does not read the level axis; the probe runs 40.
LEADS = ((), (4,), (1, 4), (2, 4), (6, 4), (2, 2, 4), (2, 6, 4))


def _transform(truncation=21, bands=1, precision="float64", **kwargs):
    return SphericalHarmonicTransform.create(
        truncation, latitude_bands=bands, precision=precision, **kwargs
    )


def _field(transform, lead, seed=0):
    rng = np.random.default_rng(seed)
    shape = (*lead, transform.grid.nlat, transform.grid.nlon)
    return rng.normal(size=shape).astype(
        np.dtype(transform.backend.float_dtype)
    )


def _coeff(transform, lead, seed=0):
    return transform.forward(_field(transform, lead, seed))


# --------------------------------------------------------------- schedule


def test_the_band_schedule_tiles_the_latitudes_exactly_once():
    for nlat in (1, 2, 7, 32, 64, 801):
        for bands in (1, 2, 3, 5, 8, 32, 1000):
            edges = latitude_band_edges(nlat, bands)
            assert edges[0][0] == 0
            assert edges[-1][1] == nlat
            assert all(b > a for a, b in edges)
            assert all(
                edges[i][1] == edges[i + 1][0] for i in range(len(edges) - 1)
            )
            assert len(edges) == min(bands, nlat)


def test_the_band_schedule_is_a_pure_function_of_the_latitudes_and_the_count():
    # It is what makes a banded answer independent of the schedule: the
    # edges cannot move with a card count, a completion order or a call
    # order, because nothing else is in scope.
    first = latitude_band_edges(801, 8)
    assert first == [(0, 100), (100, 200), (200, 300), (300, 400),
                     (400, 500), (500, 600), (600, 700), (700, 801)]
    assert latitude_band_edges(801, 8) == first


@pytest.mark.parametrize("bands", (0, -1))
def test_a_band_count_below_one_is_refused_by_name(bands):
    with pytest.raises(ValueError, match="bands must be >= 1"):
        latitude_band_edges(64, bands)
    with pytest.raises(ValueError, match="latitude_bands must be >= 1"):
        _transform(bands=bands)


# ----------------------------------------------------------- gate WAIST-1


@pytest.mark.parametrize("lead", LEADS)
def test_the_waist_head_and_tail_reproduce_the_whole_analysis(lead):
    """WAIST-1: ``contract_waist(fourier_waist(f))`` is ``forward(f)``."""
    t = _transform()
    field = _field(t, lead)
    reference = t.forward(field)
    for bands in BANDS:
        got = t.contract_waist(t.fourier_waist(field, bands=bands))
        assert np.array_equal(got, reference), (
            f"lead {lead}, {bands} bands: the waist changed the analysis"
        )


@pytest.mark.parametrize("lead", LEADS)
def test_the_waist_head_and_tail_reproduce_the_whole_synthesis(lead):
    t = _transform()
    coeff = _coeff(t, lead)
    reference = t.inverse(coeff)
    for bands in BANDS:
        got = t.waist_to_grid(t.contract_to_waist(coeff, bands=bands))
        assert np.array_equal(got, reference), (
            f"lead {lead}, {bands} bands: the waist changed the synthesis"
        )


@pytest.mark.parametrize("lead", LEADS)
def test_a_banded_transform_returns_the_resident_bits(lead):
    """The band count reached through the transform's own field."""
    resident = _transform()
    field = _field(resident, lead)
    forward = resident.forward(field)
    inverse = resident.inverse(forward)
    for bands in BANDS:
        banded = _transform(bands=bands)
        assert np.array_equal(banded.forward(field), forward)
        assert np.array_equal(banded.inverse(forward), inverse)


@pytest.mark.parametrize("bands", BANDS)
def test_draining_the_waist_band_by_band_equals_draining_it_whole(bands):
    t = _transform()
    coeff = _coeff(t, (2, 4))
    reference = t.inverse(coeff)
    waist = t.contract_to_waist(coeff, bands=bands)
    pieces = [
        t.waist_band_to_grid(waist, r0, r1) for r0, r1 in waist.edges
    ]
    assert np.array_equal(np.concatenate(pieces, axis=-2), reference)


def test_the_bands_of_one_drain_do_not_leak_into_each_other():
    """A stale half-spectrum between bands would show as a wrong tail.

    cuFFT's complex-to-real transform destroys its input, so a drain
    that reused a buffer without restoring the zero pad above the
    truncation would synthesise the previous band's spectrum there.
    Draining the same waist twice, and draining it band by band, must
    both give the resident answer.
    """
    t = _transform()
    coeff = _coeff(t, (3, 4))
    reference = t.inverse(coeff)
    waist = t.contract_to_waist(coeff, bands=8)
    first = [t.waist_band_to_grid(waist, r0, r1) for r0, r1 in waist.edges]
    second = [t.waist_band_to_grid(waist, r0, r1) for r0, r1 in waist.edges]
    assert np.array_equal(np.concatenate(first, axis=-2), reference)
    assert np.array_equal(np.concatenate(second, axis=-2), reference)


@pytest.mark.parametrize("truncation", (21, 31))
@pytest.mark.parametrize("precision", ("float64", "float32"))
def test_the_bands_hold_at_both_precisions_and_two_truncations(
    truncation, precision
):
    resident = _transform(truncation, precision=precision)
    field = _field(resident, (2, 4))
    forward = resident.forward(field)
    inverse = resident.inverse(forward)
    for bands in (2, 5, 8, 32):
        banded = _transform(truncation, bands=bands, precision=precision)
        assert np.array_equal(banded.forward(field), forward)
        assert np.array_equal(banded.inverse(forward), inverse)


def test_the_derivative_and_gradient_syntheses_band_too():
    resident = _transform()
    coeff = _coeff(resident, (4,))
    meridional = resident.inverse_meridional_derivative(coeff)
    zonal = resident.inverse_zonal_derivative(coeff)
    east, north = resident.gradient(coeff)
    for bands in (2, 3, 8, 32):
        banded = _transform(bands=bands)
        assert np.array_equal(
            banded.inverse_meridional_derivative(coeff), meridional
        )
        assert np.array_equal(banded.inverse_zonal_derivative(coeff), zonal)
        got_east, got_north = banded.gradient(coeff)
        assert np.array_equal(got_east, east)
        assert np.array_equal(got_north, north)


def test_the_streaming_transform_bands_too():
    """Streaming holds no Legendre table; the waist is orthogonal to that."""
    resident = _transform(streaming=True)
    field = _field(resident, (2, 4))
    forward = resident.forward(field)
    inverse = resident.inverse(forward)
    for bands in (2, 5, 32):
        banded = _transform(bands=bands, streaming=True)
        assert np.array_equal(banded.forward(field), forward)
        assert np.array_equal(banded.inverse(forward), inverse)
    # And a streamed call on a resident transform.
    assert np.array_equal(
        _transform(bands=8).forward_streaming(field),
        resident.forward_streaming(field),
    )
    assert np.array_equal(
        _transform(bands=8).inverse_streaming(forward),
        resident.inverse_streaming(forward),
    )


@pytest.mark.parametrize("lead", LEADS)
def test_the_waist_reproduces_the_head_and_tail_it_replaced(lead):
    """The transform as it is now against the transform as it was.

    Every other case here holds band counts against each other and
    against ``forward``/``inverse`` as they are today, which cannot see
    a change that moved both sides.  This one compares against the
    expressions the analysis and the synthesis carried before the
    split, transcribed in the card probe from tip ca57bff43, so a
    reader of the design gets the comparison the design asks for
    without a second checkout.
    """
    from tools.arwen_global_waist_probe import (
        legacy_analysis_head,
        legacy_synthesis_tail,
    )

    t = _transform()
    field = _field(t, lead)
    assert np.array_equal(
        legacy_analysis_head(t, field), t.fourier_waist(field).values
    )
    waist = t.contract_to_waist(_coeff(t, lead))
    legacy = legacy_synthesis_tail(t, waist.values)
    assert np.array_equal(legacy, t.waist_to_grid(waist))


# -------------------------------------------------------- the vector seam


def test_the_vector_analysis_routes_through_the_shared_waist():
    t = _transform()
    pair = _field(t, (2, 4))
    waist = VorticityDivergenceOperator(t)._wind_fourier(pair)
    assert isinstance(waist, FourierWaist)
    assert waist.shape == (2, 4, t.grid.nlat, t.truncation + 1)


@pytest.mark.parametrize("lead", ((), (4,), (2, 4)))
def test_the_vector_analysis_returns_the_resident_bits_at_every_band_count(lead):
    resident = _transform()
    pair = _field(resident, (2, *lead))
    zeta, divergence = VorticityDivergenceOperator(resident).vordiv_from_wind(
        pair[0], pair[1]
    )
    for bands in BANDS:
        banded = _transform(bands=bands)
        got_zeta, got_div = VorticityDivergenceOperator(
            banded
        ).vordiv_from_wind(pair[0], pair[1])
        assert np.array_equal(got_zeta, zeta), f"lead {lead}, {bands} bands"
        assert np.array_equal(got_div, divergence), f"lead {lead}, {bands} bands"


def test_the_vector_second_half_still_takes_a_raw_zonal_spectrum():
    """The seam accepts the object it accepted before, cut or uncut."""
    t = _transform()
    pair = _field(t, (2, 4))
    operator = VorticityDivergenceOperator(t)
    reference = operator.vordiv_from_wind(pair[0], pair[1])
    raw = np.fft.rfft(pair, axis=-1) / t.grid.nlon
    got = operator._vordiv_from_fourier(raw)
    assert np.array_equal(got[0], reference[0])
    assert np.array_equal(got[1], reference[1])


# ------------------------------------------------------- the waist object


def test_a_consumed_waist_refuses_a_second_drain_by_name():
    t = _transform()
    waist = t.contract_to_waist(_coeff(t, (4,)))
    t.waist_to_grid(waist)
    with pytest.raises(RuntimeError, match="consumed"):
        waist.values
    with pytest.raises(RuntimeError, match="consumed"):
        t.waist_to_grid(waist)


def test_a_consumed_waist_still_answers_its_shape():
    """The band pipeline reads the schedule after the last band drains."""
    t = _transform()
    waist = t.contract_to_waist(_coeff(t, (2, 4)), bands=4)
    lead, shape, nbytes, edges = waist.lead, waist.shape, waist.nbytes, waist.edges
    t.waist_to_grid(waist)
    assert (waist.lead, waist.shape, waist.nbytes, waist.edges) == (
        lead, shape, nbytes, edges
    )


@pytest.mark.parametrize("bands", (1, 2, 4, 8))
def test_the_waist_is_its_own_contiguous_buffer_at_every_band_count(bands):
    """One path for every band count, and never a view onto a wider buffer.

    Before the split the analysis carried the retained orders as a slice
    of the full-width scaled zonal spectrum, so that whole spectrum
    stayed alive as the slice's base through both GEMMs.  The waist is
    ``T+1`` wide and owns its own memory at one band as at eight, so the
    wider buffer dies with the rfft that produced it.
    """
    t = _transform()
    waist = t.fourier_waist(_field(t, (2, 4)), bands=bands)
    assert waist.values.base is None
    assert waist.values.shape == (2, 4, t.grid.nlat, t.truncation + 1)
    assert waist.values.flags["C_CONTIGUOUS"]
    assert waist.bands == min(bands, t.grid.nlat)


def test_the_drain_scratch_is_zero_padded_above_the_truncation():
    t = _transform()
    scratch = t.contract_to_waist(_coeff(t, (2, 4))).drain_scratch(3)
    assert scratch.shape == (2, 4, 3, t.grid.nlon // 2 + 1)
    assert not np.any(scratch)


def test_the_waist_is_smaller_than_the_grid_stack_it_stands_in_for():
    """The design's pivot: naming the seam costs less memory than not.

    Four fields at forty levels, float32: the waist is
    ``nlat x (T+1)`` complex64 per plane against ``nlat x nlon``
    float32 of grid.
    """
    t = _transform(31, precision="float32")
    waist = t.fourier_waist(_field(t, (4, 4)))
    grid_bytes = 4 * 4 * t.grid.nlat * t.grid.nlon * 4
    assert waist.nbytes < grid_bytes


# ------------------------------------------------------------- identities


@pytest.mark.parametrize("bands", BANDS)
def test_the_band_count_is_absent_from_the_identity_at_every_value(bands):
    """It moves no operand shape of any GEMM, so it stamps no hash.

    The contrast that makes this a measurement and not a convention is
    ``legendre_band``, which was documented as layout, measured on
    2026-09-06 to move bits on cupy, and now JOINS the identity.  The
    band count does not, because the contraction never sees it.
    """
    resident = _transform()
    banded = _transform(bands=bands)
    assert banded.identity == resident.identity
    assert banded.identity_hash == resident.identity_hash
    assert "latitude_bands" not in banded.identity


def test_a_non_default_legendre_band_still_joins_the_identity():
    """The waist did not loosen the lever lane's correction."""
    assert (
        _transform(bands=8, legendre_band=16).identity_hash
        != _transform(bands=8).identity_hash
    )


def test_the_probe_reads_the_transform_s_own_band_schedule():
    """One definition of the schedule, or a divergence no gate would catch."""
    from tools import arwen_global_fft_band_probe as probe

    assert probe.band_edges is latitude_band_edges


# ------------------------------------------------- what the fill hands the FFT


def test_the_whole_field_band_hands_the_fft_the_field_and_not_a_slice(
    monkeypatch,
):
    """The one-band fill must not wrap the field in a view.

    MEASURED 2026-09-06, RTX 5070 Ti, CuPy 14.2.0: ``rfft`` on the array
    peaks at 94,617,600 live pool bytes and ``rfft`` on
    ``f[..., 0:nlat, :]`` peaks at 188,989,440, the difference being a
    94,371,840 B copy of the field (T255, two fields at forty levels,
    single precision).  The slice is every row of the input and reports
    itself C-contiguous, so the copy buys nothing, and it cancelled the
    waist's saving to a tenth of a percent on a default run.  Numpy does
    not make that copy, so this asserts the call rather than the bytes.
    """
    t = _transform()
    field = _field(t, (2, 4))
    seen = []
    real_rfft = np.fft.rfft

    def spy(a, *args, **kwargs):
        seen.append(a)
        return real_rfft(a, *args, **kwargs)

    monkeypatch.setattr(np.fft, "rfft", spy)
    t.fourier_waist(field)
    assert len(seen) == 1
    assert seen[0] is field, "the whole-field band wrapped the field in a view"

    seen.clear()
    t.fourier_waist(field, bands=2)
    edges = latitude_band_edges(t.grid.nlat, 2)
    assert [a.shape[-2] for a in seen] == [r1 - r0 for r0, r1 in edges]
    assert all(a is not field for a in seen)
