"""The packed-band transform returns the dense transform's bits.

``woof.globe.spectral`` stores its Legendre tables in bands of orders
with the above-triangle zeros cut off (half the memory) and builds the
derivative table lazily.  The model's arithmetic identity -- checkpoint
restarts, receipts, the A/B arms -- rides on the transform returning the
same bits it did with the dense tables, so this file runs the frozen
dense implementation (``global_spectral_dense_reference.py``, commit
8f7c5d7d7) beside the live one on the same machine and demands EXACT
equality of the tables and of every operation on random inputs at T21
and T63, float64 and float32, numpy backend.

Why exact equality against a co-run reference and not only a hash pin:
OpenBLAS's GEMM/GEMV results depend on the operand shapes and the thread
partition (measured 2026-09-01: a per-order GEMM on the packed
``(nlat, T+1-m)`` block differs from the dense one in 32 of 64 orders at
T63), and they differ between CPU kernels too, so a hash of the outputs
is a fact about one host.  The hashes below are that fact, recorded on
the host named beside them; on a host whose reference hashes differ the
pin test says so and stands down, while the equality test still judges.
The basis and derivative tables touch no BLAS of their own, but they are
built ON the Gaussian latitudes, and those come out of
``numpy.polynomial.legendre.leggauss``, which is a companion-matrix
eigensolve plus a Newton refinement -- LAPACK, and an implementation that
numpy is free to change.  MEASURED 2026-09-07 across five numpy versions
on two operating systems: the nodes move between 2.2.6 and 2.3.0 and are
identical from 2.3.0 through 2.5.3 on Windows and Linux alike, so the table
hashes below are a fact about a numpy VERSION rather than about a host, and
the recorded pair was recorded under 2.2.6.  They are therefore guarded the same way
the output hashes are, and the exact-equality test against the co-run dense
reference is the judge on every host.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

import global_spectral_dense_reference as dense
from woof.globe.spectral.legendre import DEFAULT_BAND
from woof.globe.spectral.transform import SphericalHarmonicTransform
from woof.globe.spectral.vector import VorticityDivergenceOperator

CASES = [(21, "float64"), (21, "float32"), (63, "float64"), (63, "float32")]

#: sha256 of every output of :func:`_operations`, in order, as measured
#: 2026-09-01 on AMD Zen 5 (family 26 model 68), numpy 2.2.6 with its
#: bundled scipy-openblas 0.3.29 (DYNAMIC_ARCH), Windows 11.
PINNED_OUTPUT_DIGESTS = {
    "T21-float64": "eda8bc057ce4467e06ad42aa1417e513cb7da40fb918ea30508523416b743b04",
    "T21-float32": "a8e13a147e28b96070774060c23502f68bf311401293ff516de20274600aa0f1",
    "T63-float64": "e4dd9da3a4e053329a09d5f77c2426dcd3ebdc01285821fe1a356a16a6273dfe",
    "T63-float32": "2f3872e989a096c9aa2bf07e7c077e645e432a56ae0b30ee786262635d64f328",
}

#: sha256 of the float64 basis and derivative tables, as measured
#: 2026-09-01 on the host named above, under numpy 2.2.6.  NOT
#: host-independent: see the module docstring on leggauss.
PINNED_TABLE_DIGESTS = {
    21: "4ae109659655dbb556eb69038d8a4daa0234756e4a4187b03d41a62469abc4c6",
    63: "6afec8c0a1567239c9ff34bec9a392d199e4cf47bc9f1b79c8fdde8a07c6a29c",
}


def _digest(arrays) -> str:
    h = hashlib.sha256()
    for name, array in arrays:
        a = np.ascontiguousarray(np.asarray(array))
        h.update(f"{name}:{a.dtype.str}:{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _same(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _operations(transform, vector_cls, truncation: int, precision: str):
    """Every transform operation the model calls, on seeded random inputs.

    Leading shapes (), (3,) and (2, 5): the single-field case reaches
    BLAS as a GEMV and the stacked cases as GEMMs, and both must hold.
    """
    rng = np.random.default_rng(truncation)
    fd = np.float64 if precision == "float64" else np.float32
    nlat, nlon = transform.grid.shape
    t1 = truncation + 1
    operator = vector_cls(transform)
    out = []
    for lead in ((), (3,), (2, 5)):
        field = rng.standard_normal((*lead, nlat, nlon)).astype(fd)
        coeff = transform.project(
            rng.standard_normal((*lead, t1, t1))
            + 1j * rng.standard_normal((*lead, t1, t1))
        )
        u = rng.standard_normal((*lead, nlat, nlon)).astype(fd)
        v = rng.standard_normal((*lead, nlat, nlon)).astype(fd)
        east, north = transform.gradient(coeff)
        zeta, div = operator.vordiv_from_wind(u, v)
        uu, vv = operator.wind_from_vordiv(coeff, coeff * 0.5)
        out += [
            (f"forward{lead}", transform.forward(field)),
            (f"inverse{lead}", transform.inverse(coeff)),
            (f"gradient-east{lead}", east),
            (f"gradient-north{lead}", north),
            (f"zonal-derivative{lead}", transform.inverse_zonal_derivative(coeff)),
            (f"laplacian{lead}", transform.laplacian(coeff)),
            (f"inverse-laplacian{lead}", transform.inverse_laplacian(coeff)),
            (f"vordiv-zeta{lead}", zeta),
            (f"vordiv-div{lead}", div),
            (f"wind-u{lead}", uu),
            (f"wind-v{lead}", vv),
        ]
    return out


@pytest.mark.parametrize("truncation,precision", CASES)
@pytest.mark.parametrize("band", [1, 7, 32])
def test_the_packed_tables_are_the_dense_tables(truncation, precision, band):
    reference = dense.SphericalHarmonicTransform.create(truncation, precision=precision)
    live = SphericalHarmonicTransform.create(
        truncation, precision=precision, legendre_band=band
    )
    assert live._dbasis is None, "the derivative table must not exist before a gradient"
    assert _same(reference._analysis_mjn, live._analysis.dense())
    assert _same(reference._basis_mnj, live._basis.dense())
    assert _same(reference._dbasis_mnj, live._derivative_basis.dense())
    assert live._dbasis is not None


@pytest.mark.parametrize("truncation,precision", CASES)
@pytest.mark.parametrize("band,streaming", [(7, False), (32, False), (7, True), (32, True)])
def test_every_operation_returns_the_dense_transform_s_bits(
    truncation, precision, band, streaming
):
    # streaming=True holds no table at all and regenerates each band per
    # call; it must return the dense reference's bits like the resident
    # packed tables do.
    reference = dense.SphericalHarmonicTransform.create(truncation, precision=precision)
    live = SphericalHarmonicTransform.create(
        truncation, precision=precision, legendre_band=band, streaming=streaming
    )
    expected = _operations(reference, dense.VorticityDivergenceOperator, truncation, precision)
    got = _operations(live, VorticityDivergenceOperator, truncation, precision)
    for (name, want), (_, have) in zip(expected, got, strict=True):
        assert np.asarray(have).dtype == np.asarray(want).dtype, name
        assert _same(want, have), f"{name} differs from the dense reference"
    # Under numpy every order is its own BLAS call whatever the band, so
    # the numbers above match at both bands.  The identity is deliberately
    # not backend-dependent: under cupy the band IS the strided-batched
    # GEMM's batch count and MEASURED to move bits (2026-09-06, RTX 5070
    # Ti, T255 float32), so a non-default band carries its own hash on
    # every backend and only the shipped band shares the dense
    # reference's.
    if band == DEFAULT_BAND:
        assert live.identity_hash == reference.identity_hash
    else:
        assert live.identity_hash != reference.identity_hash
        assert live.identity["legendre_band"] == band


@pytest.mark.parametrize("truncation,precision", CASES)
def test_the_output_digests_are_the_recorded_ones(truncation, precision):
    key = f"T{truncation}-{precision}"
    reference = dense.SphericalHarmonicTransform.create(truncation, precision=precision)
    reference_digest = _digest(
        _operations(reference, dense.VorticityDivergenceOperator, truncation, precision)
    )
    if reference_digest != PINNED_OUTPUT_DIGESTS[key]:
        pytest.skip(
            f"the dense reference itself digests to {reference_digest} here "
            f"against the recorded {PINNED_OUTPUT_DIGESTS[key]}: this host's "
            "BLAS sums in a different order than the recording host's, so "
            "the recorded digest cannot be judged here (the exact-equality "
            "test beside this one is the judge on every host)"
        )
    live = SphericalHarmonicTransform.create(truncation, precision=precision)
    live_digest = _digest(
        _operations(live, VorticityDivergenceOperator, truncation, precision)
    )
    assert live_digest == PINNED_OUTPUT_DIGESTS[key]


@pytest.mark.parametrize("truncation", [21, 63])
def test_the_table_digests_are_the_recorded_ones(truncation):
    # The frozen dense reference builds its tables from the same Gaussian
    # latitudes by the same arithmetic, so when ITS digest does not
    # reproduce the recording host's the pin cannot be judged here and the
    # equality test above is the judge instead.  Without this the suite went
    # red on every host with a numpy whose leggauss differs from 2.2.6's,
    # and the verdict described the recording host rather than this tree.
    reference = dense.SphericalHarmonicTransform.create(
        truncation, precision="float64")
    reference_digest = _digest([
        ("basis", reference._basis_mnj),
        ("derivative", reference._dbasis_mnj),
    ])
    if reference_digest != PINNED_TABLE_DIGESTS[truncation]:
        pytest.skip(
            f"the frozen dense reference itself digests to "
            f"{reference_digest} here against the recorded "
            f"{PINNED_TABLE_DIGESTS[truncation]}: this environment's "
            "Gaussian latitudes are not the recording environment's bits "
            "(numpy's leggauss is a LAPACK eigensolve and its result moved "
            "between numpy 2.2.6 and 2.3.0), so the recorded digest cannot "
            "be judged here")
    live = SphericalHarmonicTransform.create(truncation, precision="float64")
    digest = _digest([
        ("basis", live._basis.dense()),
        ("derivative", live._derivative_basis.dense()),
    ])
    assert digest == PINNED_TABLE_DIGESTS[truncation]
