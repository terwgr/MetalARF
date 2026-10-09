"""Triangular spherical-harmonic analysis/synthesis on a Gaussian grid."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .backend import Backend, get_backend
from .grid import GaussianGrid
from .legendre import (
    DEFAULT_BAND,
    legendre_bands,
    legendre_derivative_block,
)


def expand_band(scratch, layout: str, previous: int, m0: int, m1: int, block):
    """Write one packed band into ``scratch`` as the dense table held it.

    Returns the ``(m1-m0, nlat, T+1)`` (``'jn'``) or ``(m1-m0, T+1, nlat)``
    (``'nj'``) view.  The zero prefix is maintained incrementally: band b
    needs zeros over degrees ``[0, m0_b)``, and after band b-1 the scratch
    holds zeros over ``[0, m0_{b-1})`` and that band's data over
    ``[m0_{b-1}, T+1)``, so only the ``[m0_{b-1}, m0_b)`` slice is cleared.
    The first band (``previous = m0 = 0``) writes every entry, so the
    buffer's state before the first call never matters.  Shared by the
    resident tables and the streamed ones so both hand the GEMM the same
    operand.
    """
    dense = scratch[: m1 - m0]
    if layout == "jn":
        dense[:, :, previous:m0] = 0
        dense[:, :, m0:] = block
    else:
        dense[:, previous:m0, :] = 0
        dense[:, m0:, :] = block
    return dense


def latitude_band_edges(nlat: int, bands: int) -> list[tuple[int, int]]:
    """The band schedule: ``floor(k * nlat / B)`` for ``k = 0..B``.

    A pure function of ``(nlat, bands)`` and of nothing else -- not the
    completion order, not how many devices are running -- which is what
    makes a banded answer independent of the schedule that produced it.
    Bands that would be empty (more bands than latitudes) are dropped
    rather than yielded, so every returned band holds at least one row
    and the list tiles ``[0, nlat)`` exactly once in ascending order.

    One definition, read by the transform, by the probes and by the
    band pipeline above it: two copies of a band schedule that drift
    apart is a divergence no bit-identity gate would catch, because
    both halves would be self-consistent.
    """
    n = int(nlat)
    b = int(bands)
    if n < 1:
        raise ValueError(f"nlat must be >= 1, got {nlat}")
    if b < 1:
        raise ValueError(
            f"bands must be >= 1, got {bands}: a band count below one has no "
            f"schedule, and silently treating it as one band would hide a "
            f"caller that computed its band count wrong"
        )
    cuts = [(k * n) // b for k in range(b + 1)]
    return [(a, c) for a, c in zip(cuts, cuts[1:]) if c > a]


def analysis_band(
    block: np.ndarray, m0: int, weights: np.ndarray, truncation: int
) -> np.ndarray:
    """The Gram-solved analysis rows of one band, ``(orders, nlat, T+1-m0)``.

    Gauss-quadrature exactness holds in real arithmetic only: the
    computed basis values and weights carry O(truncation * eps)
    correlated error, so the raw analysis of a resolved field leaks
    ~1e-13 into modes that must vanish (measured at T9, numpy 2.2.6).
    Solving against the analysis-synthesis Gram matrix per order makes
    forward(inverse(x)) = x on the resolved subspace at working
    precision on any in-spec numpy, which is the transform's own
    correctness claim.  Float64 whatever the backend precision; the one
    function feeds the resident table and the streamed analysis, so the
    two cannot compute different rows.
    """
    count, _, nlat = block.shape
    table = np.zeros((count, nlat, int(truncation) + 1 - m0), dtype=np.float64)
    for i in range(count):
        rows = block[i, i:, :] * weights[None, :] * (2.0 * math.pi)
        gram = rows @ block[i, i:, :].T
        table[i, :, i:] = np.linalg.solve(gram, rows).T
    return table


class BandedLegendreTable:
    """A Legendre table packed in bands of consecutive orders.

    ``layout='nj'`` holds synthesis blocks ``(orders, T+1-m0, nlat)`` (the
    basis and its derivative, contracted over degree); ``layout='jn'``
    holds the analysis blocks ``(orders, nlat, T+1-m0)`` (contracted over
    latitude).  Each block is the band's slice of the dense table with its
    leading ``m0`` all-zero degrees cut off.

    WHY THE CONTRACTION EXPANDS EACH BAND BACK TO DENSE.  OpenBLAS's GEMM
    and GEMV results depend on the operand shapes: the edge-tile kernels
    and the thread partition move with N and K, so a per-order GEMM on the
    packed ``(nlat, T+1-m)`` block does not return the bits the dense
    ``(nlat, T+1)`` one did (measured 2026-09-01: 32 of 64 orders differ
    at T63 float64, every packing granularity up to 32 still differs
    somewhere).  The model's arithmetic identity is pinned on the dense
    contraction, so :meth:`expanded` rebuilds the dense ``(orders, nlat,
    T+1)`` stack for one band at a time in a scratch buffer -- the zeros
    are written, not stored -- and the GEMM reads exactly what it read
    before.  Under numpy that is the same BLAS call per order; under cupy
    one strided-batched GEMM per band.

    THAT IS NOT ENOUGH ON CUPY.  Expanding to dense fixes the OPERANDS,
    and cuBLAS also picks its kernel and tiling by the BATCH count, which
    is the band, and by the GEMM's M, which is the width of the field
    stack being transformed.  MEASURED 2026-09-06 on an RTX 5070 Ti at
    T255 float32: with identical dense operands, band 8 and band 16
    return different bits from band 32 for a stack of one, two, four and
    eight levels, and the same bits at forty.  The one field the dycore
    transforms as a single plane is the surface pressure, so a ten-step
    run at another band ends in another state.  This is why the band is
    in the transform identity above the shipped default, and why a probe
    that sweeps only the band at forty levels reads it as neutral.
    """

    def __init__(self, xp, dtype, truncation: int, nlat: int, layout: str, band: int):
        if layout not in ("nj", "jn"):
            raise ValueError(f"layout must be 'nj' or 'jn', got {layout!r}")
        self.xp = xp
        self.dtype = dtype
        self.truncation = int(truncation)
        self.nlat = int(nlat)
        self.layout = layout
        self.band = int(band)
        self.blocks: list[tuple[int, int, object]] = []
        self._scratch = None

    def append(self, m0: int, m1: int, block) -> None:
        expected = (
            (m1 - m0, self.truncation + 1 - m0, self.nlat)
            if self.layout == "nj"
            else (m1 - m0, self.nlat, self.truncation + 1 - m0)
        )
        if tuple(block.shape) != expected:
            raise ValueError(
                f"band {m0}..{m1} block has shape {tuple(block.shape)}, "
                f"expected {expected} for layout {self.layout!r}"
            )
        self.blocks.append((m0, m1, block))

    @property
    def nbytes(self) -> int:
        """Resident bytes of the packed blocks (the scratch is separate)."""
        return sum(int(block.nbytes) for _, _, block in self.blocks)

    @property
    def scratch_nbytes(self) -> int:
        return 0 if self._scratch is None else int(self._scratch.nbytes)

    def output_size(self, *, transposed: bool) -> int:
        """Trailing size of ``values @ table`` for one order."""
        if self.layout == "jn" or transposed:
            return self.truncation + 1
        return self.nlat

    def expanded(self):
        """Yield ``(m0, m1, dense)``: the band as the dense table held it.

        ``dense`` is ``(m1-m0, nlat, T+1)`` for ``'jn'`` and ``(m1-m0,
        T+1, nlat)`` for ``'nj'``, a view into one scratch buffer reused
        for every band (see :func:`expand_band`).
        """
        if self._scratch is None:
            self._scratch = self.xp.zeros(
                scratch_shape(self.layout, self.band, self.truncation, self.nlat),
                dtype=self.dtype,
            )
        previous = 0
        for m0, m1, block in self.blocks:
            yield m0, m1, expand_band(self._scratch, self.layout, previous, m0, m1, block)
            previous = m0

    def dense(self):
        """The whole dense table, assembled in host numpy.  Tests only."""
        return _assemble_dense(self)


def scratch_shape(layout: str, band: int, truncation: int, nlat: int) -> tuple[int, int, int]:
    t1 = int(truncation) + 1
    return (band, nlat, t1) if layout == "jn" else (band, t1, nlat)


def _assemble_dense(table) -> np.ndarray:
    t1 = table.truncation + 1
    shape = (t1, table.nlat, t1) if table.layout == "jn" else (t1, t1, table.nlat)
    out = np.zeros(shape, dtype=table.dtype)
    for m0, m1, dense in table.expanded():
        out[m0:m1] = dense if table.xp is np else table.xp.asnumpy(dense)
    return out


class StreamedLegendreTable:
    """A Legendre table that is never resident: each band is generated on
    demand from the recurrence, contracted, and discarded.

    Same ``expanded()`` / ``output_size()`` contract as
    :class:`BandedLegendreTable`, so :meth:`SphericalHarmonicTransform.
    _batched_contract` reads either without knowing which.  ``kind`` is
    ``'basis'`` (synthesis, ``'nj'``), ``'analysis'`` (Gram-solved
    analysis rows, ``'jn'``) or ``'derivative'`` (``dP/dphi``, ``'nj'``).

    Memory while a contraction runs is one order chunk: the float64
    recurrence block, the float64 analysis rows (analysis only), the
    backend-dtype copy and the dense expansion scratch, each
    ``order_chunk x (T+1) x nlat`` at most -- and NOTHING between calls.
    At T1534 on the 1536-latitude GFS grid that is 0.6 GB per block for a
    chunk of 32 against the 28.9 GB dense table that could not be held.

    The bits are the resident table's: the recurrence, the Gram solve,
    the dtype cast and :func:`expand_band` are the same functions, and
    under numpy the GEMM runs per order whatever the chunk, so
    ``forward_streaming`` returns exactly what ``forward`` does on a
    resident table (pinned by ``tests/test_global_spectral_streaming.py``
    and the dense-reference bit-identity test).  The recurrence is
    re-run on every call: for a synthesis that is elementwise work of
    the same order as the contraction's own read of the block, for an
    analysis it is the per-order Gram solve the resident build paid once
    at construction, paid again per call (the tools/ probe records the
    wall time beside the peak).
    """

    def __init__(self, transform: "SphericalHarmonicTransform", kind: str, order_chunk: int):
        if kind not in ("basis", "analysis", "derivative"):
            raise ValueError(f"kind must be 'basis', 'analysis' or 'derivative', got {kind!r}")
        if int(order_chunk) < 1:
            raise ValueError(f"order_chunk must be >= 1, got {order_chunk}")
        self.transform = transform
        self.kind = kind
        self.layout = "jn" if kind == "analysis" else "nj"
        self.band = int(order_chunk)
        self.xp = transform.backend.xp
        self.dtype = transform.backend.float_dtype
        self.truncation = int(transform.grid.truncation)
        self.nlat = int(transform.grid.nlat)

    #: Nothing is resident between calls.
    nbytes = 0
    scratch_nbytes = 0

    def output_size(self, *, transposed: bool) -> int:
        if self.layout == "jn" or transposed:
            return self.truncation + 1
        return self.nlat

    def blocks(self):
        """Yield ``(m0, m1, block)`` in the backend dtype, one chunk live."""
        grid = self.transform.grid
        backend = self.transform.backend
        for m0, m1, block in legendre_bands(self.truncation, grid.sin_lat, band=self.band):
            if self.kind == "analysis":
                host = analysis_band(block, m0, grid.quadrature_weights, self.truncation)
            elif self.kind == "derivative":
                host = legendre_derivative_block(grid.sin_lat, m0, block)
            else:
                host = block
            # Neither frame may keep a chunk alive past its contraction:
            # the next chunk's recurrence would otherwise land beside it.
            del block
            device = backend.asarray(host, dtype=self.dtype)
            del host
            yield m0, m1, device
            del device

    def expanded(self):
        scratch = self.xp.empty(
            scratch_shape(self.layout, self.band, self.truncation, self.nlat),
            dtype=self.dtype,
        )
        previous = 0
        for m0, m1, block in self.blocks():
            dense = expand_band(scratch, self.layout, previous, m0, m1, block)
            del block
            yield m0, m1, dense
            previous = m0

    def dense(self):
        """The whole dense table, assembled in host numpy.  Tests only."""
        return _assemble_dense(self)


class _RowsFilled:
    """Which latitude rows a band fill has written.  See :meth:`FourierWaist.close`."""

    def __init__(self, nlat: int):
        self._written = np.zeros(int(nlat), dtype=bool)

    def claim(self, r0: int, r1: int) -> None:
        if r1 <= r0 or r0 < 0 or r1 > self._written.size:
            raise ValueError(
                f"latitude band [{r0}, {r1}) is not a nonempty run of the "
                f"{self._written.size} rows"
            )
        if bool(self._written[r0:r1].any()):
            raise ValueError(
                f"latitude rows {r0}..{r1} were already filled: two bands "
                "writing one row of the waist would contract the second "
                "band's spectrum where the first band's belongs"
            )
        self._written[r0:r1] = True

    def claim_external(self, r0: int, r1: int) -> None:
        """ Rows another card filled and sent, marked written.

        The bytes are the bytes that card computed for those rows, which
        are the bytes a one-card run would have written there, so the
        assembled waist is the one-card waist and the contraction that
        reads it is the one-card contraction.
        """
        self._written[int(r0):int(r1)] = True

    def complete(self) -> bool:
        return bool(self._written.all())

    def require_complete(self) -> None:
        if not bool(self._written.all()):
            missing = int((~self._written).sum())
            raise ValueError(
                f"{missing} of {self._written.size} latitude rows of the "
                "Fourier waist were never filled, so the Legendre "
                "contraction would read uninitialised memory into every "
                "coefficient of the field"
            )


class FourierWaist:
    """The full-latitude Fourier buffer where grid space and spectral space meet.

    ``(*lead, nlat, T+1)`` complex: on the way in, the zonal spectrum of
    a grid field scaled by ``1/nlon`` and cut to the retained orders; on
    the way out, the Legendre contraction's output before the inverse
    FFT reads it.  It is the object the transform already built as an
    unnamed temporary between its ``rfft`` and its contraction, and
    between its contraction and its ``irfft``, given a name and a
    lifetime so grid space can be fed to it and drained from it a
    latitude band at a time.

    WHY IT IS WHOLE IN LATITUDE.  The FFT along longitude is row-local,
    so feeding it band by band changes no arithmetic in any row
    (MEASURED 2026-09-06, gate FFT-1: 49 of 49 cases bit-exact, 0 ULP,
    RTX 5070 Ti, CuPy 14.2.0 / CUDA 13.2, band counts 2 to 100 including
    uneven ones).  The Legendre contraction is not row-local: it reduces
    the latitude axis on the way in (``K = nlat``) and produces it on
    the way out (``N = nlat``), and the GEMM's result depends on those
    shapes -- :class:`BandedLegendreTable` carries the measurement that
    forces the point.  So the waist stays whole, the contraction keeps
    the operand shapes it has today, and a run that cuts grid space into
    bands returns the bits the resident run returned.

    It is also smaller than the grid stack it stands in for: at T533 a
    four-field forty-level waist is 0.510 GiB against 0.765 GiB of grid,
    so naming it costs less memory than not naming it, before a band
    count is chosen.

    A waist is consumed by the call that drains its last band
    (:meth:`take`): the buffer must die before the inverse FFT allocates
    its output, not beside it, which is the lifetime ``_synthesize``
    kept by hand before the split.
    """

    @classmethod
    def open(cls, transform, lead, edges) -> "FourierWaist":
        """An empty waist a band pipeline fills a band at a time.

        :meth:`SphericalHarmonicTransform.fourier_waist` fills the waist
        from a field that already exists whole.  A band pipeline never
        has that field: it builds one band of grid space, hands it over,
        and frees it before the next band exists, so the waist is opened
        first and filled by :meth:`fill_band`.

        THE BITS ARE THE SAME EXPRESSION either way.  ``fill_band``
        evaluates the divide-into-the-waist route
        ``fourier_waist`` takes above one band, and at one band it takes
        the divide-into-a-new-array route ``fourier_waist`` takes there,
        for the memory reason recorded in that method: the band loop's
        full-width transient has nothing to amortise over a single band.
        """
        waist = cls.__new__(cls)
        waist.transform = transform
        waist._values = None
        waist.edges = list(edges)
        waist.lead = tuple(int(v) for v in lead)
        waist.shape = (
            *waist.lead, transform.grid.nlat, transform.truncation + 1
        )
        waist.nbytes = int(
            np.prod(waist.shape) * transform.backend.complex_dtype().itemsize
        )
        waist._open = True
        waist._filled = _RowsFilled(transform.grid.nlat)
        return waist

    def fill_band(self, r0: int, r1: int, rows) -> None:
        """Write the zonal spectrum of one latitude band into the waist.

        ``rows`` is the band's grid field, ``(*lead, r1 - r0, nlon)``
        real.  The FFT along longitude is row-local, so this band's
        columns are the columns a whole-field analysis would have put
        there (MEASURED, gate FFT-1).
        """
        if not getattr(self, "_open", False):
            raise RuntimeError(
                "this FourierWaist was not opened for a band fill: a waist "
                "built from a whole field is already full"
            )
        xp = self.transform.backend.xp
        r0 = int(r0)
        r1 = int(r1)
        self._filled.claim(r0, r1)
        from woof.metal.array import MetalArray
        if isinstance(rows, MetalArray):
            transform = self.transform
            context = getattr(transform, "metal_context", None)
            if (context is None or rows.buffer.context is not context
                    or rows.dtype != np.dtype(np.float64)):
                raise TypeError(
                    "resident Global waist fill requires same-context FP64 rows"
                )
            expected = (*self.lead, r1 - r0, transform.grid.nlon)
            if tuple(rows.shape) != expected:
                raise ValueError(
                    f"band rows {tuple(rows.shape)} are not the waist's {expected}"
                )
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs

            nlon = int(transform.grid.nlon)
            nfreq = int(transform.truncation + 1)
            nlat = int(transform.grid.nlat)
            batch = int(np.prod(self.lead, dtype=np.int64)) if self.lead else 1
            if self._values is None:
                self._values = MetalArrayModule(context).empty(
                    (*self.lead, nlat, nfreq), dtype=np.complex128
                )
            table_cache = getattr(transform, "_metal_longitude_table_cache", {})
            table_key = (nlon, nfreq)
            if table_key not in table_cache:
                mode = np.arange(nfreq, dtype=np.float64)[:, None]
                longitude = np.arange(nlon, dtype=np.float64)[None, :]
                angle = (2.0 * np.pi / nlon) * mode * longitude
                twiddles = np.stack((np.cos(angle), -np.sin(angle)), axis=-1)
                table_cache[table_key] = (
                    np.ascontiguousarray(twiddles, dtype=np.float64),
                    np.asarray([1.0 / nlon], dtype=np.float64),
                )
                transform._metal_longitude_table_cache = table_cache
            twiddle_host, scale_host = table_cache[table_key]
            metal = MetalArrayModule(context)
            context.submit_kernel(
                "global_longitude_transform_resident_fragment",
                {"grid": rows.view((rows.size,), dtype=np.float64),
                 "twiddles": metal.asarray(
                     twiddle_host.reshape(-1)
                 ),
                 "spectrum": self._values.view(
                     (2 * self._values.size,), dtype=np.float64),
                 "scale": metal.asarray(scale_host)},
                {"nlat": int(r1 - r0), "nlon": nlon, "nfreq": nfreq,
                 "batch": batch, "full_nlat": nlat, "start_row": r0},
                exact_policy=False,
                grid_size=batch * (r1 - r0) * nfreq,
                threads_per_group=(
                    kernel_specs.GLOBAL_LONGITUDE_TRANSFORM_RESIDENT_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            return
        f = xp.asarray(rows, dtype=self.transform.backend.float_dtype)
        if tuple(f.shape) != (*self.lead, r1 - r0, self.transform.grid.nlon):
            raise ValueError(
                f"band rows {tuple(f.shape)} are not the waist's "
                f"{(*self.lead, r1 - r0, self.transform.grid.nlon)}"
            )
        t1 = self.transform.truncation + 1
        if len(self.edges) == 1 and self._values is None:
            # One band covering the globe: divide into a new array, so the
            # full-width scaled spectrum the band loop cannot amortise
            # over a single band never exists (fourier_waist's own
            # measurement, 0.9898 GiB against 1.5831 at T383 six fields).
            self._values = xp.divide(
                xp.fft.rfft(f, axis=-1)[..., :t1], self.transform.grid.nlon
            )
            return
        if self._values is None:
            self._values = xp.empty(
                self.shape, dtype=self.transform.backend.complex_dtype
            )
        xp.divide(
            xp.fft.rfft(f, axis=-1)[..., :t1],
            self.transform.grid.nlon,
            out=self._values[..., r0:r1, :],
        )

    def close(self) -> "FourierWaist":
        """Fill the rows the other cards hold, refuse a waist with an
        unwritten row, then hand it back.

        A band schedule that skipped a row would contract whatever the
        allocator left in the buffer into every coefficient of the field,
        which is a wrong answer rather than a crash.

        THIS IS THE ONLY PLACE A SECOND CARD TOUCHES THE ANALYSIS.  A
        multi-card run partitions grid space and replicates spectral
        space, so each rank fills the waist rows of the bands it owns and
        the rest arrive here, as the bytes the rank that owns them
        produced.  The contraction that follows then runs at ``K = nlat``
        on a complete waist, exactly as a one-card run does, which is why
        two cards return one card's bits and no pin moves (gate BIT-5).
        The synthesis direction needs no counterpart: it starts from the
        replicated spectral state, so every rank contracts to its own rows
        without a gather, and the wire carries the analysis alone.

        ``transform.row_exchange`` is a duck-typed hook rather than an
        import because ``woof.globe.spectral`` is the layer below
        ``woof.globe`` and must not learn that cards exist; it is
        ``None`` on a single-card run and the branch below is not taken.
        """
        if getattr(self, "_open", False):
            exchange = getattr(self.transform, "row_exchange", None)
            if (
                exchange is not None
                and int(getattr(exchange, "world", 1)) > 1
                and not self._filled.complete()
            ):
                values = self.values
                exchange.fill_rows(
                    self.transform.backend.xp, values, values.ndim - 2,
                    name="waist",
                )
                self._filled.claim_external(0, self.transform.grid.nlat)
            self._filled.require_complete()
            self._open = False
        return self

    def __init__(self, transform, values, edges):
        self.transform = transform
        self._values = values
        self._open = False
        self._filled = None
        #: The band schedule this waist was filled with, or is to be
        #: drained with: ``[(r0, r1), ...]`` tiling ``[0, nlat)``.
        self.edges = list(edges)
        # Cached at construction so the shape questions still answer
        # after take() has handed the buffer away.
        self.lead = tuple(values.shape[:-2])
        self.shape = tuple(values.shape)
        self.nbytes = int(values.nbytes)

    @property
    def bands(self) -> int:
        return len(self.edges)

    @property
    def values(self):
        if self._values is None:
            raise RuntimeError(
                "this FourierWaist was consumed: its buffer was handed to a "
                "drain and released.  Draining a waist twice would read freed "
                "memory or a stale spectrum; build a new waist instead"
            )
        return self._values

    def take(self):
        """Hand the buffer over and forget it: the caller is now its only owner."""
        values = self.values
        self._values = None
        return values

    def band(self, r0: int, r1: int):
        """The waist rows of one latitude band, a view."""
        return self.values[..., int(r0):int(r1), :]

    def drain_scratch(self, rows: int):
        """A zero-padded ``(*lead, rows, nlon//2+1)`` half spectrum for one band.

        The waist owns the padding contract because it owns the
        truncation: the orders above ``T`` are the zeros the inverse
        real FFT reads as the vanishing tail of the spectrum, and a band
        drain that forgot them would synthesise noise there.

        A fresh buffer per band, not a cached one.  Caching it would
        save two thirds of one ``memset`` per band (the retained orders
        are overwritten wholesale by the caller's multiply, so only the
        pad above ``T`` would need clearing, 268 of 802 columns at T533)
        -- about 0.1 ms a band on a 5090, against holding the last
        band's spectrum alive past the call, which is the lifetime the
        one-band case must not lose.  Measure and cache if a band loop
        ever shows it.
        """
        xp = self.transform.backend.xp
        return xp.zeros(
            (*self.lead, int(rows), self.transform.grid.nlon // 2 + 1),
            dtype=self.transform.backend.complex_dtype,
        )


@dataclass
class SphericalHarmonicTransform:
    #: The multi-card row exchange, or None.  NOT a dataclass field and
    #: NOT in :meth:`identity`: it changes who computes a latitude row,
    #: never what the row is, so it belongs in no hash (the precedent this
    #: follows is ``legendre_band`` and ``streaming``, deliberately absent
    #: from the identity for the same reason).  ``woof.globe`` sets
    #: it on the transform at run start; this layer only calls it.
    row_exchange = None
    #: The multi-card ORDER exchange, or None.  Dual to ``row_exchange``:
    #: it gathers the coefficient (analysis) and waist (synthesis) columns
    #: of the orders this rank did not contract, from the ranks that did.
    #: Like ``row_exchange`` it is NOT a dataclass field and enters no
    #: identity -- it changes who contracts an order, never what the order
    #: is.  ``None`` on a single-card or band-axis run.
    order_exchange = None
    #: Optional staged Metal route context. Like the exchange endpoints this
    #: selects an execution route without changing transform identity.
    metal_context = None

    grid: GaussianGrid
    backend: Backend
    # Opt-in TF32 tensor-core compute for the Legendre GEMM contractions.
    # TF32 truncates the float32 mantissa to 10 bits inside the GEMM, so this
    # CHANGES NUMERICS: it stays default-off, joins the identity hash only
    # when enabled, and needs its own transform_check gate re-measurement on
    # the GPU before any production default flips.
    tensor_core_contractions: bool = False
    # Orders per packed Legendre band (see BandedLegendreTable).  Under
    # numpy every order is its own BLAS call whatever the band, so there
    # the band is layout only.  Under cupy the band is the strided-batched
    # GEMM's batch count, and it is ARITHMETIC: MEASURED 2026-09-06 on an
    # RTX 5070 Ti at T255 float32, an analysis of a single plane (the
    # GEMM's M = 1, which is what ln ps is) differs by 6.2e-06 between
    # band 32 and bands 8 and 16, and the synthesis of the same plane by
    # 1.4e-05; the differences appear and vanish with M (M = 1, 2, 4 and 8
    # each move somewhere, M = 40 moves nowhere), which is the M-dependent
    # kernel and tile choice the BandedLegendreTable docstring below
    # already records for N and K.  A ten-step T255 native run at band 16
    # left 66 of the 125 checkpoint state arrays differing from the
    # band-32 run, the divergence entering at step 0 through
    # forward(log(ps)).  So a
    # non-default band JOINS the identity, and the default keeps every
    # hash that was ever written.
    legendre_band: int = DEFAULT_BAND
    # A streaming transform holds NO Legendre table: every analysis,
    # synthesis and derivative regenerates the basis one chunk of
    # ``legendre_band`` orders at a time (see StreamedLegendreTable) and
    # runs in O(legendre_band x (T+1) x nlat) memory.  The instrument's
    # entry point for a truncation whose tables do not fit -- T1534 on the
    # native GFS grid wants 28.9 GB per dense table -- at the cost of
    # re-running the recurrence (and, for the analysis, the per-order
    # Gram solve) on every call.  Same bits as the resident tables --
    # MEASURED 2026-09-06, RTX 5070 Ti, T255 float32: the streamed
    # analysis and the streamed synthesis of a single plane and of a
    # forty-level stack are byte-identical to the resident-table calls --
    # so absent from the identity hash at every value.
    streaming: bool = False
    # Latitude bands the longitude FFT is fed and drained in.  1 is the
    # whole field in one call, which is the expression the transform
    # evaluated before the waist was named, so the default is today's
    # arithmetic and today's memory.  Above 1 the rfft runs on
    # floor(k*nlat/B) row slices and the irfft drains the waist the same
    # way, and the whole (*lead, nlat, nlon//2+1) zonal spectrum never
    # exists at full size.  It is a STREAMING GRANULARITY, NOT A
    # DECOMPOSITION: the Legendre contraction is never split, so the
    # GEMM keeps K = N = nlat and every band count returns the same
    # bits.  MEASURED 2026-09-06 (gate FFT-1, RTX 5070 Ti, CuPy 14.2.0 /
    # CUDA 13.2): 49 of 49 cases bit-exact at 0 ULP over T255/T383/T533,
    # band counts 2 to 100 including uneven ones down to 8-row bands,
    # 3-D and 4-D stacks, float32 and float64; and gate WAIST-1 on this
    # branch, every stack width the dycore transforms.  So it is ABSENT
    # FROM THE IDENTITY at every value -- unlike legendre_band, which
    # was documented as layout and measured on 2026-09-06 to be
    # arithmetic, and which joins the identity above.  The difference is
    # not a convention: the band count moves no operand shape of any
    # GEMM, and the Legendre band moves the batch count of one.
    latitude_bands: int = 1
    # The Legendre ORDER bands this transform holds a table for, as
    # ``((m0, m1), ...)``, or empty for the whole table.  This is the
    # order-m multi-card axis (design section 10): the orders are
    # independent output indices of the contraction and are never reduced,
    # so a rank that holds only its own bands' tables and contracts only
    # its own orders returns exactly the bits a whole-table rank returns
    # for those orders, and the whole spectrum is the concatenation of the
    # ranks' disjoint order ranges.  Holding only the owned bands' tables
    # is the capacity the axis buys past the point where a whole Legendre
    # table stops fitting one card.  A partition changes WHICH bands live
    # here, never the arithmetic of any band, so it enters NO identity --
    # the same rule ``latitude_bands`` follows, and for the same reason.
    # WHOLE bands only: the contraction runs one strided-batched GEMM per
    # band and cuBLAS keys its kernel on the batch, so a partition that
    # split a band would move bits where whole bands do not.
    order_partition: tuple = ()

    def __post_init__(self) -> None:
        if self.tensor_core_contractions:
            if self.backend.name != "cupy":
                raise ValueError(
                    "tensor_core_contractions=True requires backend='cupy': "
                    "numpy has no tensor-core path, so the flag would stamp "
                    "a TF32 identity onto arithmetic that never changes"
                )
            if np.dtype(self.backend.float_dtype) != np.dtype(np.float32):
                raise ValueError(
                    "tensor_core_contractions=True requires precision="
                    "'float32': CuPy's TF32 compute type acts on float32 "
                    "GEMMs only, so under float64 the flag would move the "
                    "identity hash while every contraction stays fp64"
                )
        if int(self.legendre_band) < 1:
            raise ValueError(f"legendre_band must be >= 1, got {self.legendre_band}")
        if int(self.latitude_bands) < 1:
            raise ValueError(
                f"latitude_bands must be >= 1, got {self.latitude_bands}: a "
                f"band count below one has no schedule, and a transform that "
                f"quietly rounded it up to one band would run a caller's "
                f"memory plan at full grid width"
            )
        self._owned_order_bounds = self._validate_order_partition()
        self._weights = self.backend.asarray(
            self.grid.quadrature_weights, dtype=self.backend.float_dtype
        )
        if self.streaming:
            self._basis = StreamedLegendreTable(self, "basis", self.legendre_band)
            self._analysis = StreamedLegendreTable(self, "analysis", self.legendre_band)
        else:
            self._basis, self._analysis = self._build_tables()
        # The meridional-derivative table is built by the first gradient or
        # vector operation (see _derivative_basis).  An analysis-only
        # transform -- the verification harness at T767 on the native GFS
        # grid -- never pays its third of the table memory.
        self._dbasis = None
        self._zonal_wavenumber = self.backend.asarray(
            1j * np.arange(self.grid.truncation + 1, dtype=np.float64),
            dtype=self.backend.complex_dtype,
        )
        n = np.arange(self.grid.truncation + 1, dtype=np.float64)
        eigen = -(n * (n + 1.0)) / (self.grid.radius_m * self.grid.radius_m)
        self._laplacian_eigen = self.backend.asarray(eigen, dtype=self.backend.float_dtype)
        tri = np.tri(self.grid.truncation + 1, dtype=bool)
        self._tri_mask = self.backend.asarray(tri, dtype=bool)
        # Per-coefficient action table for the fused cupy project(): 0 above
        # the triangle (zeroed), 2 on the in-triangle m=0 column (imaginary
        # part dropped), 1 elsewhere in the triangle (copied).  The numpy
        # project() below stays the specification and never reads it.
        code = np.where(tri, 1, 0).astype(np.int8)
        code[:, 0] = np.where(tri[:, 0], 2, 0)
        self._project_code = self.backend.asarray(code)
        self._coslat = self.backend.asarray(
            self.grid.cos_lat[:, None], dtype=self.backend.float_dtype
        )
        self._degree = self.backend.asarray(n, dtype=self.backend.float_dtype)
        self._order = self.backend.asarray(n, dtype=self.backend.float_dtype)

    def _validate_order_partition(self) -> tuple:
        """Normalise ``order_partition`` to a contiguous run of whole bands.

        Empty means the whole table.  A non-empty partition must be a
        contiguous subsequence of ``band_bounds(T, legendre_band)`` -- whole
        bands only, because the contraction runs one GEMM per band and its
        batch count is the band; and contiguous, because a rank owns one
        interval of orders whose columns are one slice of the waist and one
        slice of the spectrum.
        """
        raw = tuple(self.order_partition or ())
        if not raw:
            return ()
        if self.streaming:
            raise ValueError(
                "order_partition with streaming=True is not built: the order "
                "axis is the resident-table capacity path (each rank holds "
                "only its own bands' tables) and streaming is the other one "
                "(no table at all).  Choose one"
            )
        from .legendre import band_bounds

        schedule = [(int(a), int(b)) for a, b in band_bounds(
            self.grid.truncation, self.legendre_band)]
        owned = [(int(a), int(b)) for a, b in raw]
        if any(pair not in schedule for pair in owned):
            raise ValueError(
                f"order_partition {owned} is not a set of whole Legendre "
                f"bands of {schedule}: the contraction's GEMM batch is the "
                "band, so a partition that splits one moves bits"
            )
        first = schedule.index(owned[0])
        if owned != schedule[first:first + len(owned)]:
            raise ValueError(
                f"order_partition {owned} is not a contiguous run of "
                f"{schedule}: a rank owns one interval of orders"
            )
        return tuple(owned)

    @property
    def owned_order_range(self) -> tuple[int, int]:
        """``(m_lo, m_hi)`` orders this transform holds a table for.

        The whole retained range ``(0, T+1)`` when unpartitioned.
        """
        if not self._owned_order_bounds:
            return (0, self.grid.truncation + 1)
        return (self._owned_order_bounds[0][0], self._owned_order_bounds[-1][1])

    def _build_tables(self) -> tuple[BandedLegendreTable, BandedLegendreTable]:
        """The basis and the Gram-solved analysis table, band by band.

        Gauss-quadrature exactness holds in real arithmetic only: the
        computed basis values and weights carry O(truncation * eps)
        correlated error, so the raw analysis of a resolved field leaks
        ~1e-13 into modes that must vanish (measured at T9, numpy 2.2.6).
        Solving against the analysis-synthesis Gram matrix per order makes
        forward(inverse(x)) = x on the resolved subspace at working
        precision on any in-spec numpy, which is the transform's own
        correctness claim.

        Streams: one float64 band of the recurrence is live at a time, its
        two blocks are cast to the backend dtype and appended, and the
        float64 band is dropped.  The host never holds a full float64
        table, whatever the device precision.  The rows themselves come
        from :func:`analysis_band`, the function the streamed analysis
        reads too.
        """
        grid = self.grid
        t = grid.truncation
        xp = self.backend.xp
        dtype = self.backend.float_dtype
        basis = BandedLegendreTable(xp, dtype, t, grid.nlat, "nj", self.legendre_band)
        analysis = BandedLegendreTable(xp, dtype, t, grid.nlat, "jn", self.legendre_band)
        weights = grid.quadrature_weights
        owned = set(self._owned_order_bounds)
        for m0, m1, block in legendre_bands(t, grid.sin_lat, band=self.legendre_band):
            # The recurrence is sequential across bands, so every band is
            # computed; only the OWNED bands are cast and kept resident.
            # That is the order axis's capacity win: a partitioned transform
            # holds one rank's fraction of the table memory, which is what
            # brings a truncation past the whole-table wall onto a card.
            if owned and (m0, m1) not in owned:
                continue
            table = analysis_band(block, m0, weights, t)
            basis.append(m0, m1, self.backend.asarray(block, dtype=dtype))
            analysis.append(m0, m1, self.backend.asarray(table, dtype=dtype))
        return basis, analysis

    @property
    def _derivative_basis(self):
        """``dP/dphi`` in the basis layout, built on first use.

        Re-runs the float64 recurrence band by band rather than keeping a
        float64 basis on the host for it: the recurrence is deterministic,
        so the derivative comes out identical to one built at construction,
        and an analysis-only transform never pays for it.  A streaming
        transform never holds it at all.
        """
        if self.streaming:
            return StreamedLegendreTable(self, "derivative", self.legendre_band)
        if self._dbasis is None:
            grid = self.grid
            table = BandedLegendreTable(
                self.backend.xp, self.backend.float_dtype, grid.truncation,
                grid.nlat, "nj", self.legendre_band,
            )
            owned = set(self._owned_order_bounds)
            for m0, m1, block in legendre_bands(
                grid.truncation, grid.sin_lat, band=self.legendre_band
            ):
                if owned and (m0, m1) not in owned:
                    continue
                table.append(
                    m0, m1,
                    self.backend.asarray(
                        legendre_derivative_block(grid.sin_lat, m0, block),
                        dtype=self.backend.float_dtype,
                    ),
                )
            self._dbasis = table
        return self._dbasis

    @property
    def legendre_table_nbytes(self) -> dict[str, int]:
        """Resident table bytes by name, plus the expansion scratch buffers.

        ``derivative`` is 0 until the first gradient/vector operation.
        """
        scratch = self._basis.scratch_nbytes + self._analysis.scratch_nbytes
        derivative = 0
        if self._dbasis is not None:
            derivative = self._dbasis.nbytes
            scratch += self._dbasis.scratch_nbytes
        return {
            "basis": self._basis.nbytes,
            "analysis": self._analysis.nbytes,
            "derivative": derivative,
            "scratch": scratch,
        }

    @classmethod
    def create(
        cls,
        truncation: int,
        *,
        nlat: int | None = None,
        nlon: int | None = None,
        dealias_factor: float = 1.5,
        radius_m: float | None = None,
        grid: GaussianGrid | None = None,
        backend: str = "numpy",
        precision: str = "float64",
        tensor_core_contractions: bool = False,
        legendre_band: int = DEFAULT_BAND,
        streaming: bool = False,
        latitude_bands: int = 1,
        order_partition: tuple = (),
    ) -> "SphericalHarmonicTransform":
        """Build the transform for ``truncation``.

        Without ``grid`` the Gaussian grid is derived from the truncation
        and the dealias rule (``nlat``/``nlon`` may widen it).  With
        ``grid`` the transform analyses fields living on THAT grid at a
        truncation the grid did not derive -- the capped exact analysis of
        a product on its native grid, e.g. T767 on the 1536x3072 GFS grid.
        The grid is asked for the exactness condition and refuses when the
        projection would alias (:meth:`GaussianGrid.truncated_to`).

        ``streaming=True`` builds no Legendre table: ``forward``,
        ``inverse``, ``gradient`` and the vector operators regenerate the
        basis ``legendre_band`` orders at a time and run in
        O(legendre_band x (T+1) x nlat) memory whatever the truncation --
        the entry point for an analysis whose tables do not fit, e.g.
        T1534 on the native GFS grid::

            native = GaussianGrid.for_shape(1536, 3072)
            transform = SphericalHarmonicTransform.create(
                1534, grid=native, streaming=True)
            coeff = transform.forward(field)   # same bits as a resident table

        A resident transform can also stream one call through
        :meth:`forward_streaming` / :meth:`inverse_streaming`.
        """
        if grid is not None:
            if nlat is not None or nlon is not None or radius_m is not None \
                    or dealias_factor != 1.5:
                raise ValueError(
                    "grid= carries its own shape and radius: nlat, nlon, "
                    "dealias_factor and radius_m cannot be combined with it"
                )
            grid = grid.truncated_to(truncation)
        else:
            kwargs = {}
            if radius_m is not None:
                kwargs["radius_m"] = radius_m
            grid = GaussianGrid.create(
                truncation,
                nlat=nlat,
                nlon=nlon,
                dealias_factor=dealias_factor,
                **kwargs,
            )
        return cls(
            grid,
            get_backend(backend, precision),
            tensor_core_contractions=tensor_core_contractions,
            legendre_band=legendre_band,
            streaming=streaming,
            latitude_bands=latitude_bands,
            order_partition=tuple(order_partition or ()),
        )

    def streaming_working_bytes(self, order_chunk: int | None = None) -> dict[str, int]:
        """Bytes one streamed contraction holds for a chunk of orders.

        ``recurrence`` is the float64 band block, ``analysis`` the float64
        Gram-solved rows (0 for a synthesis), ``device`` the backend-dtype
        copy when the cast copies (0 under numpy float64, where the host
        array is handed over), ``scratch`` the dense expansion.  Every
        term is ``order_chunk x (T+1) x nlat`` elements at most: the
        formula the tests hold the measured peak against, so the claim
        "O(order_chunk x (T+1) x nlat)" is a number and not a slogan.
        """
        chunk = self.legendre_band if order_chunk is None else int(order_chunk)
        elements = chunk * (self.truncation + 1) * self.grid.nlat
        itemsize = np.dtype(self.backend.float_dtype).itemsize
        casts = self.backend.name != "numpy" or itemsize != 8
        return {
            "recurrence": elements * 8,
            "analysis": elements * 8,
            "device": elements * itemsize if casts else 0,
            "scratch": elements * itemsize,
        }

    @property
    def truncation(self) -> int:
        return self.grid.truncation

    @property
    def spectral_shape(self) -> tuple[int, int]:
        n = self.truncation + 1
        return n, n

    @property
    def geometry_identity(self) -> dict:
        return {
            "truncation": int(self.truncation),
            "nlat": int(self.grid.nlat),
            "nlon": int(self.grid.nlon),
            "radius_m": float(self.grid.radius_m),
            "normalization": "complex-orthonormal-condon-shortley",
            "latitude_grid": "gauss-legendre-in-sin-latitude",
        }

    @property
    def geometry_hash(self) -> str:
        raw = json.dumps(
            self.geometry_identity, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    @property
    def identity(self) -> dict:
        payload = {
            **self.geometry_identity,
            "backend": self.backend.name,
            "float_dtype": str(np.dtype(self.backend.float_dtype)),
        }
        # TF32 joins the identity only when enabled: it changes the GEMM
        # arithmetic, so an enabled transform must not share a hash with the
        # fp32 transform it diverges from, while the disabled default keeps
        # every pre-existing identity hash byte-identical.
        if self.tensor_core_contractions:
            payload["tensor_core_contractions"] = True
        # The Legendre band is the strided-batched GEMM's batch count on
        # cupy and it MOVES BITS at the stack widths the dycore actually
        # transforms (see the field comment above), so a transform built
        # on a non-default band must not share a hash with the one it
        # diverges from.  The default keeps every identity hash ever
        # written, and the numpy backend, where the band is layout only,
        # pays the same conservative rule rather than carrying a
        # backend-dependent identity.
        if int(self.legendre_band) != DEFAULT_BAND:
            payload["legendre_band"] = int(self.legendre_band)
        return payload

    @property
    def identity_hash(self) -> str:
        raw = json.dumps(
            self.identity, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    def zeros(self, *leading: int):
        return self.backend.xp.zeros(
            (*leading, *self.spectral_shape), dtype=self.backend.complex_dtype
        )

    def _validate_grid(self, field) -> None:
        if tuple(field.shape[-2:]) != self.grid.shape:
            raise ValueError(
                f"grid field has trailing shape {field.shape[-2:]}, expected {self.grid.shape}"
            )

    def _validate_spectral(self, coeff) -> None:
        if tuple(coeff.shape[-2:]) != self.spectral_shape:
            raise ValueError(
                f"spectral field has trailing shape {coeff.shape[-2:]}, "
                f"expected {self.spectral_shape}"
            )

    def project(self, coeff):
        xp = self.backend.xp
        from woof.metal.array import MetalArray, MetalArrayModule

        if isinstance(coeff, MetalArray):
            context = getattr(self, "metal_context", None)
            if (context is None or coeff.buffer.context is not context
                    or coeff.dtype != np.dtype(np.complex128)):
                raise TypeError(
                    "resident Global spectral projection requires complex128 "
                    "coefficients on this transform's Metal context"
                )
            self._validate_spectral(coeff)
            from woof.metal.generated import kernel_specs

            count = int(np.prod(coeff.shape[:-2], dtype=np.int64)) * (
                self.grid.truncation + 1
            ) * (self.grid.truncation + 1)
            output = MetalArrayModule(context).empty(
                coeff.shape, dtype=np.complex128
            )
            context.submit_kernel(
                "global_spectral_project_fragment",
                {"input": coeff.view((count, 2), dtype=np.float64),
                 "output": output.view((count, 2), dtype=np.float64)},
                {"count": count, "ndegree": self.grid.truncation + 1,
                 "nfreq": self.grid.truncation + 1},
                exact_policy=False, grid_size=count,
                threads_per_group=(
                    kernel_specs.GLOBAL_SPECTRAL_PROJECT_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            return output
        if self.backend.name == "cupy":
            # One fused launch replaces the asarray+copy+mask-multiply+
            # realify chain (four launches per call, ~470 calls/step at
            # T533).  Same arithmetic as the numpy specification below for
            # every finite input; the kernel always writes a fresh output,
            # matching .copy().  One divergence: above-triangle entries
            # become exact zeros, where the mask multiply would turn a
            # non-finite entry there into NaN (in-triangle non-finites
            # still propagate identically on both paths).
            from .fused import project_kernel

            arr = xp.asarray(coeff, dtype=self.backend.complex_dtype)
            self._validate_spectral(arr)
            return project_kernel(xp)(arr, self._project_code)
        arr = xp.asarray(coeff, dtype=self.backend.complex_dtype).copy()
        self._validate_spectral(arr)
        arr *= self._tri_mask
        arr[..., :, 0] = arr[..., :, 0].real
        return arr

    def _contraction_scope(self):
        """Scope the Legendre GEMM contractions onto TF32 tensor cores.

        Default (flag off) is a no-op, so every hash-checked fp32/fp64
        identity is untouched.  With the flag on (cupy float32 only, enforced
        in ``__post_init__``) the float32 compute type is switched to TF32
        for the duration of the contraction and restored afterwards: CuPy's
        compute-type table is process-global state, and leaving it set would
        silently change the arithmetic of every other float32 GEMM in the
        process.
        """
        if not self.tensor_core_contractions:
            return contextlib.nullcontext()
        # backend='cupy' already imported cupy; CPU paths never reach this.
        from cupy import _core as cupy_core
        from cupy._core import _routines_linalg as cupy_linalg

        xp = self.backend.xp

        @contextlib.contextmanager
        def scope():
            previous = cupy_core.get_compute_type(xp.float32)
            cupy_core.set_compute_type(
                xp.float32, cupy_linalg.COMPUTE_TYPE_TF32
            )
            try:
                yield
            finally:
                cupy_core.set_compute_type(xp.float32, previous)

        return scope()

    def _batched_contract(self, values, table, *, transposed=False):
        """Contract ``values[..., k, m] * table[m, k, n] -> out[..., n, m]``.

        Each band rides one strided-batched real GEMM per real/imaginary
        component: ``matmul`` on an (orders, batch, k) stack against the
        band's dense (orders, k, n) expansion lands on cuBLAS
        gemmStridedBatched under the cupy backend (where the TF32
        compute-type scope can apply) and on per-order BLAS GEMM under
        numpy.  An einsum with the order axis shared between operands has
        no guaranteed GEMM lowering, which is why this formulation was
        chosen.  The dense expansion carries the above-triangle zeros, so
        every order's GEMM sees the operand the whole-table contraction
        gave it and returns the same bits (see BandedLegendreTable).

        ``transposed=True`` reads an ``'nj'`` table as ``(m, j, n)``: the
        F-contiguous inner matrices of the transposed view map onto the
        BLAS/cuBLAS transpose flags without a copy.
        """
        from woof.metal.array import MetalArray, MetalArrayModule

        metal_context = getattr(self, "metal_context", None)
        if (isinstance(values, MetalArray) and metal_context is not None
                and self.backend.name == "numpy"
                and values.buffer.context is metal_context
                and values.dtype == np.dtype(np.complex128)):
            lead = tuple(values.shape[:-2])
            ninput, nmodes = map(int, values.shape[-2:])
            batch = int(np.prod(lead, dtype=np.int64)) if lead else 1
            noutput = int(table.output_size(transposed=transposed))
            metal = MetalArrayModule(metal_context)
            result = metal.empty((*lead, noutput, nmodes), dtype=np.complex128)
            source_words = values.view((2 * values.size,), dtype=np.float64)
            result_words = result.view((2 * result.size,), dtype=np.float64)
            from woof.metal.generated import kernel_specs

            for m0, m1, dense in table.expanded():
                if transposed:
                    dense = dense.transpose(0, 2, 1)
                dense = np.ascontiguousarray(dense, dtype=np.float64)
                norder = int(m1 - m0)
                metal_context.submit_kernel(
                    "global_latitude_transform_resident_fragment",
                    {"values": source_words,
                     "legendre": metal.asarray(dense.reshape(-1)),
                     "spectral": result_words},
                    {"norder": norder, "batch": batch,
                     "ninput": ninput, "noutput": noutput,
                     "nmodes": nmodes, "order_start": int(m0)},
                    exact_policy=False,
                    grid_size=norder * batch * noutput,
                    threads_per_group=(
                        kernel_specs.GLOBAL_LATITUDE_TRANSFORM_RESIDENT_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
            return result

        xp = self.backend.xp
        lead = values.shape[:-2]
        k = values.shape[-2]
        m_count = values.shape[-1]
        n_count = table.output_size(transposed=transposed)
        stacked = xp.moveaxis(values, -1, 0).reshape(m_count, -1, k)
        real_in = xp.ascontiguousarray(stacked.real)
        imag_in = xp.ascontiguousarray(stacked.imag)
        batch = stacked.shape[1]
        # The complex input is dead once its real/imaginary operands are
        # copied out.  When the caller handed in a temporary (the weighted
        # Fourier pair of a vector analysis, 1.53 GiB per six-field flux
        # chunk at T533 float32; the waist an analysis hands over through
        # FourierWaist.take, 0.765 GiB at the same shape) this reference
        # was the last one, and dropping it here keeps it from sitting
        # beside the GEMMs (memory only).  The analysis operand used to be
        # a SLICE of the full-width zonal spectrum and cost 1.15 GiB
        # there, which is the figure this comment carried until the
        # transform split at the waist: the slice kept its base alive, and
        # what arrives now is the waist itself at two thirds the width.
        # Both figures are the shape arithmetic of a six-field forty-level
        # stack at T533 in single precision.
        del values, stacked
        real = xp.empty((m_count, batch, n_count), dtype=self.backend.float_dtype)
        imag = xp.empty_like(real)
        metal_context = getattr(self, "metal_context", None)
        metal_specs = None
        if metal_context is not None and self.backend.name == "numpy":
            from woof.metal.generated import kernel_specs as metal_specs
        metal_latitude_ready = (
            metal_specs is not None
            and hasattr(metal_specs, "GLOBAL_LATITUDE_TRANSFORM_FRAGMENT_RESOURCES")
            and np.dtype(self.backend.float_dtype) == np.dtype(np.float64)
        )
        if metal_latitude_ready:
            from woof.metal.array import MetalArrayModule

            metal = MetalArrayModule(metal_context)
            for m0, m1, dense in table.expanded():
                if transposed:
                    dense = dense.transpose(0, 2, 1)
                dense_host = np.ascontiguousarray(dense, dtype=np.float64)
                band_real = np.ascontiguousarray(real_in[m0:m1], dtype=np.float64)
                band_imag = np.ascontiguousarray(imag_in[m0:m1], dtype=np.float64)
                norder = int(m1 - m0)
                band_batch, ninput = int(band_real.shape[1]), int(band_real.shape[2])
                resources = {
                    "real_values": metal.asarray(band_real.reshape(-1)),
                    "imag_values": metal.asarray(band_imag.reshape(-1)),
                    "legendre": metal.asarray(dense_host.reshape(-1)),
                    "real_spectral": metal.empty((norder * band_batch * n_count,), dtype=np.float64),
                    "imag_spectral": metal.empty((norder * band_batch * n_count,), dtype=np.float64),
                }
                metal_context.submit_kernel(
                    "global_latitude_transform_fragment", resources,
                    {"norder": norder, "batch": band_batch,
                     "ninput": ninput, "noutput": int(n_count)},
                    exact_policy=False,
                    grid_size=norder * band_batch * int(n_count),
                    threads_per_group=metal_specs.GLOBAL_LATITUDE_TRANSFORM_FRAGMENT_THREADS_PER_GROUP,
                )
                real[m0:m1] = resources["real_spectral"].to_numpy().reshape(
                    norder, band_batch, n_count
                )
                imag[m0:m1] = resources["imag_spectral"].to_numpy().reshape(
                    norder, band_batch, n_count
                )
        else:
            for m0, m1, dense in table.expanded():
                if transposed:
                    dense = dense.transpose(0, 2, 1)
                real[m0:m1] = real_in[m0:m1] @ dense
                imag[m0:m1] = imag_in[m0:m1] @ dense
        # The GEMM operands are dead once every band has run; releasing
        # them before the complex output exists keeps the two from being
        # live together (memory only: at T533 float32 a six-field synthesis
        # held 0.51 GiB of operands beside a 0.76 GiB output).
        del real_in, imag_in
        out = xp.empty(real.shape, dtype=self.backend.complex_dtype)
        out.real = real
        out.imag = imag
        return xp.moveaxis(out.reshape(m_count, *lead, n_count), 0, -1)

    def open_waist(self, lead, *, bands: int | None = None) -> FourierWaist:
        """An empty waist on this transform's grid, for a band pipeline to fill.

        ``lead`` is the leading (field, level, ...) shape; the waist is
        ``(*lead, nlat, T+1)`` complex.  Fill it with
        :meth:`FourierWaist.fill_band` one band at a time, close it, and
        hand it to :meth:`contract_waist`: the contraction then reads a
        full-latitude object whatever the band count filled it, which is
        the bit-identity argument of the whole streaming design.
        """
        return FourierWaist.open(
            self,
            lead,
            latitude_band_edges(
                self.grid.nlat, self.latitude_bands if bands is None else bands
            ),
        )

    def fourier_waist(self, field, *, bands: int | None = None) -> FourierWaist:
        """Fill the Fourier waist from a grid field: the head of an analysis.

        Grid fields are real, so the real FFT's m = 0..nlon/2 output
        covers every retained order (nlon >= 2*truncation+2) at half the
        work of the complex transform it replaces.  Orders above the
        truncation are dropped from every column: the zonal projection
        is exact by FFT truncation and never by subsampling longitudes.

        THE BITS ARE THE EXPRESSION THE ANALYSIS EVALUATED BEFORE THE
        SPLIT.  That expression divided the whole rfft output by
        ``nlon`` and then sliced; dividing only the retained orders
        computes the same quotient for every element that survived the
        slice.  ``nlon`` is a Python int, so no promotion moves between
        the two forms (checked at both precisions, and pinned by the
        ten-step T255 native run against the tree this branch cut from).

        TWO ROUTES, EACH MEASURED.  MEASURED 2026-09-06, RTX 5070 Ti,
        T383 six fields at forty levels float32, peak live device bytes
        of the head alone (waist 0.3955 GiB, full-width scaled spectrum
        0.5943 GiB):

        ==================  =====  ==========  ============
        fill                bands  peak GiB    held after
        ==================  =====  ==========  ============
        divide then slice       1      1.1886        0.5943   (before the split)
        divide into a new       1      0.9898        0.3955   <- one band
        band loop into it       1      1.5831        0.3955
        band loop into it       2      0.9893        0.3955
        band loop into it       8      0.5440        0.3955
        band loop into it      32      0.4326        0.3955
        ==================  =====  ==========  ============

        So one band divides into a NEW array and more than one band
        divides into a preallocated waist: the band loop's own chain
        costs a full-width transient it cannot amortise over a single
        band, and at one band that transient is the whole point of the
        exercise.  Both routes are the same three operations and the
        gate holds them against each other at every band count.

        THE TABLE IS THE HEAD ALONE, AND THE ROUTE ONLY REACHES IT WHEN
        THE WHOLE-FIELD BAND IS HANDED THE FIELD.  Written with a slice
        the saving does not reach a caller: MEASURED 2026-09-06, RTX
        5070 Ti, peak live pool bytes of one whole ``forward`` call at
        the shipped one band, against the same call in the tree this
        branch cut from --

        =====================  ==============  ==============  =========
        stack                  before split    slice at fill   this fill
        =====================  ==============  ==============  =========
        T255 two by six by 40      1.0574 GiB      1.0561 GiB   0.8803
        T255 two by 40             0.1762          0.1760       0.1467
        T383 six by 40             1.7818          1.7808       1.5831
        =====================  ==============  ==============  =========

        -- because the rfft copies what it is handed when that is a
        view, so the whole-field slice put a second grid stack beside
        the output and cancelled the waist exactly.  See the comment on
        the branch below for the isolated measurement.

        Either way the full-width ``(*lead, nlat, nlon//2+1)`` scaled
        spectrum stops surviving the head: before the split it was the
        slice's base and stayed alive through both GEMMs (1.15 GiB per
        six-field analysis at T533 float32, 2.47 GB for the advection's
        stacked flux pair), and it is now 0.77 GiB of waist at T533,
        with only ``1/B`` of the rfft output live beside it.
        """
        xp = self.backend.xp
        from woof.metal.array import MetalArray
        if isinstance(field, MetalArray):
            if (getattr(self, "metal_context", None) is None
                    or field.buffer.context is not self.metal_context
                    or field.dtype != np.dtype(np.float64)):
                raise TypeError(
                    "resident Global longitude transform requires its FP64 Metal context"
                )
            f = field
        else:
            f = xp.asarray(field, dtype=self.backend.float_dtype)
        self._validate_grid(f)
        edges = latitude_band_edges(
            self.grid.nlat, self.latitude_bands if bands is None else bands
        )
        t1 = self.truncation + 1
        if getattr(self, "metal_context", None) is not None:
            if self.backend.name != "numpy" or f.dtype != np.dtype(np.float64):
                raise TypeError(
                    "Global Metal longitude transform currently requires NumPy FP64 fields"
                )
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs

            context = self.metal_context
            metal = MetalArrayModule(context)
            nlon = int(self.grid.nlon)
            nfreq = int(t1)
            table_cache = getattr(self, "_metal_longitude_table_cache", {})
            table_key = (nlon, nfreq)
            if table_key not in table_cache:
                mode = np.arange(nfreq, dtype=np.float64)[:, None]
                longitude = np.arange(nlon, dtype=np.float64)[None, :]
                angle = (2.0 * np.pi / nlon) * mode * longitude
                twiddles = np.stack((np.cos(angle), -np.sin(angle)), axis=-1)
                table_cache[table_key] = (
                    np.ascontiguousarray(twiddles, dtype=np.float64),
                    np.asarray([1.0 / nlon], dtype=np.float64),
                )
                self._metal_longitude_table_cache = table_cache
            twiddle_host, scale_host = table_cache[table_key]
            twiddles = metal.asarray(twiddle_host)
            scale = metal.asarray(scale_host)
            batch = int(np.prod(f.shape[:-2], dtype=np.int64))
            values = metal.empty(
                (*f.shape[:-2], self.grid.nlat, nfreq), dtype=np.complex128
            )
            grid = (
                metal.asarray(f).view((f.size,), dtype=np.float64)
                if isinstance(f, MetalArray)
                else metal.asarray(np.ascontiguousarray(
                    f.reshape(batch, self.grid.nlat, nlon), dtype=np.float64
                ))
            )
            grid_words = grid.view((grid.size,), dtype=np.float64)
            twiddle_words = twiddles.view((twiddles.size,), dtype=np.float64)
            spectrum_words = values.view((2 * values.size,), dtype=np.float64)
            for r0, r1 in edges:
                context.submit_kernel(
                    "global_longitude_transform_resident_fragment",
                    {
                        "grid": grid_words,
                        "twiddles": twiddle_words,
                        "spectrum": spectrum_words,
                        "scale": scale,
                    },
                    {
                        "nlat": int(r1 - r0),
                        "nlon": nlon,
                        "nfreq": nfreq,
                        "batch": batch,
                        "full_nlat": int(self.grid.nlat),
                        "start_row": int(r0),
                    },
                    exact_policy=False,
                    grid_size=batch * (r1 - r0) * nfreq,
                    threads_per_group=kernel_specs.GLOBAL_LONGITUDE_TRANSFORM_RESIDENT_FRAGMENT_THREADS_PER_GROUP,
                )
            return FourierWaist(self, values, edges)
        if len(edges) == 1:
            r0, r1 = edges[0]
            # THE WHOLE-FIELD BAND IS HANDED THE FIELD, NOT A SLICE OF
            # IT.  cupy's rfft copies a sliced input even when the slice
            # covers every row and reports itself C-contiguous, and the
            # copy is the whole grid stack: MEASURED 2026-09-06, RTX
            # 5070 Ti, T255 two fields at forty levels float32, peak live
            # pool bytes of one rfft -- 94,617,600 B on the array, and
            # 188,989,440 B on f[..., 0:nlat, :], which is the output
            # plus a 94,371,840 B copy of the field.  Written the second
            # way the one-band fill paid that copy on every analysis and
            # it cancelled the waist's saving exactly: the head peaked at
            # 0.1760 GiB against the pre-split expression's 0.1762.
            # Written this way it peaks at 0.1467.  A band that is not
            # the whole field pays a copy of its own rows and cannot
            # avoid it (ascontiguousarray measured the same peak), but
            # that copy divides with the band count.
            rows = f if (r0, r1) == (0, self.grid.nlat) else f[..., r0:r1, :]
            return FourierWaist(
                self,
                xp.divide(
                    xp.fft.rfft(rows, axis=-1)[..., :t1],
                    self.grid.nlon,
                ),
                edges,
            )
        waist = xp.empty(
            (*f.shape[:-2], self.grid.nlat, t1), dtype=self.backend.complex_dtype
        )
        for r0, r1 in edges:
            xp.divide(
                xp.fft.rfft(f[..., r0:r1, :], axis=-1)[..., :t1],
                self.grid.nlon,
                out=waist[..., r0:r1, :],
            )
        return FourierWaist(self, waist, edges)

    def _agree_across_cards(self, direction: str, table, array) -> None:
        """Hand a contraction's output to the card layer once per shape.

        A multi-card gather run reproduces one card only where the cards
        return the same bits for the same contraction, and which shapes
        agree is a property of the operand shapes THIS configuration
        presents (unlike cards agree on every contraction of a T255 L40
        step and disagree on an eight-plane synthesis at the same
        truncation).  So the check rides the contractions themselves: the
        first time a (direction, table, shape) is contracted the row
        exchange hashes the output on every rank and refuses the run by
        name if they differ.  ``row_exchange`` is duck-typed and ``None``
        on a single-card run, where this is one attribute read.
        """
        exchange = self.row_exchange
        hook = getattr(exchange, "agree_once", None)
        if hook is None or int(getattr(exchange, "world", 1)) <= 1:
            return
        name = (
            "analysis" if table is self._analysis
            else "basis" if table is self._basis
            else getattr(table, "name", None) or f"table@{id(table):x}"
        )
        hook(
            self.backend.xp,
            (direction, name, tuple(int(v) for v in array.shape), str(array.dtype)),
            array,
        )

    def contract_waist(self, waist: FourierWaist, table=None, *,
                       keep_resident: bool = False):
        """Contract a filled waist to spectral coefficients: the tail of an analysis.

        Consumes the waist, handing its buffer to the contraction as the
        sole reference so the GEMM operands are not copied out beside it.
        The contraction reads a full-latitude object whatever the band
        count that filled it, which is the whole bit-identity argument:
        ``K = nlat``, the batch and the order bands of 32 are the ones a
        resident analysis hands it.
        """
        table = self._analysis if table is None else table
        with self._contraction_scope():
            out = self._batched_contract(waist.take(), table)
        from woof.metal.array import MetalArray
        if isinstance(out, MetalArray):
            if not keep_resident:
                # The public transform API keeps its NumPy return contract;
                # resident callers may carry coefficients to the next stage.
                out = out.to_numpy()
        if not isinstance(out, MetalArray):
            out[..., :, 0] = out[..., :, 0].real
        self._agree_across_cards("analysis", table, out)
        return out

    def _analyze(self, field, table, *, bands: int | None = None,
                 keep_resident: bool = False):
        # No contraction scope here: contract_waist opens its own, and the
        # fill is an FFT and a divide.  _contraction_scope sets CuPy's
        # process-global float32 compute type and its own docstring scopes
        # that to the duration of the contraction, so wrapping the fill in
        # it as well held TF32 over an FFT that never reads the setting --
        # no bits moved either way, and the narrow form is the one the
        # pre-split analysis had.
        return self.contract_waist(
            self.fourier_waist(field, bands=bands), table,
            keep_resident=keep_resident,
        )

    def forward(self, field, *, keep_resident: bool = False):
        if self.order_exchange is not None and int(
                getattr(self.order_exchange, "world", 1)) > 1:
            return self.forward_orders(field)
        return self._analyze(field, self._analysis,
                             keep_resident=keep_resident)

    def _contract_orders(self, values, table, m_lo, m_hi, *, transposed=False):
        """Contract only orders ``[m_lo, m_hi)`` of ``values``, compactly.

        The same per-band GEMM :meth:`_batched_contract` runs, band for
        band, for the orders this rank owns -- ``real_in[band] @ dense``
        with the band's ABSOLUTE ``m0`` still setting the dense table's
        above-triangle zeros -- so the columns this returns are the columns
        the whole-table contraction returned there, bit for bit.  It
        allocates output for the owned orders alone, never the full width,
        which is what keeps a partitioned rank inside its fraction of the
        transient as well as its fraction of the table.

        ``values`` carries the whole order axis last; ``[m_lo, m_hi)`` is
        the contiguous run of orders whose whole bands ``table`` holds.
        Returns ``(*lead, n, m_hi - m_lo)``.
        """
        xp = self.backend.xp
        sub = values[..., m_lo:m_hi]
        lead = sub.shape[:-2]
        k = sub.shape[-2]
        width = m_hi - m_lo
        n_count = table.output_size(transposed=transposed)
        stacked = xp.moveaxis(sub, -1, 0).reshape(width, -1, k)
        real_in = xp.ascontiguousarray(stacked.real)
        imag_in = xp.ascontiguousarray(stacked.imag)
        batch = stacked.shape[1]
        del sub, stacked
        real = xp.empty((width, batch, n_count), dtype=self.backend.float_dtype)
        imag = xp.empty_like(real)
        for m0, m1, dense in table.expanded():
            if transposed:
                dense = dense.transpose(0, 2, 1)
            real[m0 - m_lo:m1 - m_lo] = real_in[m0 - m_lo:m1 - m_lo] @ dense
            imag[m0 - m_lo:m1 - m_lo] = imag_in[m0 - m_lo:m1 - m_lo] @ dense
        del real_in, imag_in
        out = xp.empty(real.shape, dtype=self.backend.complex_dtype)
        out.real = real
        out.imag = imag
        return xp.moveaxis(out.reshape(width, *lead, n_count), 0, -1)

    def forward_orders(self, field, exchange=None):
        """Analysis on the order axis: each rank contracts its own orders.

        The longitude FFT is row-local, so each rank builds the whole waist
        from the whole grid field it holds (grid space is NOT partitioned
        on this axis).  Each rank then contracts only the orders its table
        covers, and the ranks all-gather the resulting coefficient columns,
        which are disjoint order ranges concatenated -- no sum crosses the
        wire, so the assembled spectrum is the single-card spectrum bit for
        bit.

        With no exchange, or a world of one, this IS :meth:`forward`: the
        owned range is the whole retained range and the contraction is the
        whole-table one.
        """
        exchange = self.order_exchange if exchange is None else exchange
        waist = self.fourier_waist(field)
        values = waist.take()
        m_lo, m_hi = self._exchange_range(exchange)
        with self._contraction_scope():
            compact = self._contract_orders(values, self._analysis, m_lo, m_hi)
        full = self._gather_order_columns(
            exchange, compact, self.grid.truncation + 1, name="analysis_orders"
        )
        full[..., :, 0] = full[..., :, 0].real
        return full

    def inverse_orders(self, coeff, exchange=None):
        """Synthesis on the order axis: each rank contracts its own orders.

        Each rank contracts the orders its table covers into their waist
        columns, the ranks all-gather those columns into the whole waist
        (again a concatenation of disjoint order ranges, no sum), and the
        inverse FFT along longitude -- which is where the sum over orders
        actually happens -- then runs WHOLE and identically on every rank.
        So this returns :meth:`inverse` bit for bit.
        """
        exchange = self.order_exchange if exchange is None else exchange
        xp = self.backend.xp
        c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        m_lo, m_hi = self._exchange_range(exchange)
        with self._contraction_scope():
            compact = self._contract_orders(c, self._basis, m_lo, m_hi)
        values = self._gather_order_columns(
            exchange, compact, self.grid.truncation + 1, name="synthesis_orders"
        )
        waist = FourierWaist(
            self, values, latitude_band_edges(self.grid.nlat, self.latitude_bands)
        )
        return self.waist_to_grid(waist)

    def _exchange_range(self, exchange) -> tuple[int, int]:
        """The order range this rank owns, from the exchange or the table."""
        if exchange is not None and hasattr(exchange, "owned_order_range"):
            return tuple(int(v) for v in exchange.owned_order_range)
        return self.owned_order_range

    def _gather_order_columns(self, exchange, compact, width, *, name):
        """Assemble the whole ``(*lead, X, width)`` from the ranks' columns.

        ``compact`` is this rank's ``(*lead, X, owned_m)``.  With no
        exchange the owned columns ARE the whole width and nothing crosses
        the wire.
        """
        if exchange is None or int(getattr(exchange, "world", 1)) <= 1:
            return compact
        return exchange.gather_columns(
            self.backend.xp, compact, width, name=name
        )

    def forward_streaming(self, field, *, order_chunk: int | None = None):
        """:meth:`forward` without a resident analysis table.

        Each chunk of ``order_chunk`` orders (default ``legendre_band``)
        is generated by the recurrence, Gram-solved, contracted with the
        Fourier-transformed field for those orders and discarded, so the
        call runs in O(order_chunk x (T+1) x nlat) memory whatever the
        truncation (:meth:`streaming_working_bytes` prices it).  Returns
        the bits :meth:`forward` returns on a resident table.
        """
        return self._analyze(
            field,
            StreamedLegendreTable(
                self, "analysis", self.legendre_band if order_chunk is None else order_chunk
            ),
        )

    def inverse_streaming(self, coeff, *, order_chunk: int | None = None):
        """:meth:`inverse` without a resident basis table; see :meth:`forward_streaming`."""
        return self._synthesize(
            coeff,
            StreamedLegendreTable(
                self, "basis", self.legendre_band if order_chunk is None else order_chunk
            ),
        )

    def contract_to_waist(
        self, coeff, table=None, *, bands: int | None = None
    ) -> FourierWaist:
        """Contract spectral coefficients into the waist: the head of a synthesis.

        The contraction produces the latitude axis (``N = nlat``) and is
        never split, so it runs at the operand shapes a resident
        synthesis gives it whatever band count the waist will be drained
        in.  The returned waist carries that drain schedule.
        """
        xp = self.backend.xp
        from woof.metal.array import MetalArray

        if isinstance(coeff, MetalArray):
            context = getattr(self, "metal_context", None)
            if context is None or coeff.buffer.context is not context:
                raise ValueError(
                    "Global synthesis input belongs to another Metal context"
                )
            if (self.backend.name != "numpy"
                    or coeff.dtype != np.dtype(np.complex128)):
                raise TypeError(
                    "resident Global synthesis currently requires a "
                    "NumPy FP64 transform and complex128 coefficients"
                )
            c = coeff
        else:
            c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        table = self._basis if table is None else table
        with self._contraction_scope():
            values = self._batched_contract(c, table)
        self._agree_across_cards("synthesis", table, values)
        return FourierWaist(
            self,
            values,
            latitude_band_edges(
                self.grid.nlat, self.latitude_bands if bands is None else bands
            ),
        )

    def _rows_to_grid(self, values, waist: FourierWaist, r0: int, r1: int,
                      *, keep_resident: bool = False):
        """One latitude band of a waist, in grid space.

        Only orders 0..truncation are populated and the negative-m half
        of the old full spectrum was the conjugate mirror, so the
        half-spectrum inverse real FFT reproduces ifft(full).real to fp
        roundoff (irfft reads only the real part of the m=0 bin, exactly
        the part .real kept) at half the FFT work and half the temporary.

        Memory only, same arithmetic: the half spectrum is allocated
        after the contraction (not beside its operands) and at the
        BAND's width, the nlon scaling is written straight into its
        retained-order slice instead of through a full-size temporary
        (the same multiply(nlon, values) ufunc call on the same values,
        so the same bits), and the waist is released before the FFT when
        the caller handed over the last reference.  At T533 float32 a
        six-field synthesis carried 1.15 + 0.76 GiB of those two beside
        the FFT's own buffers.
        """
        from woof.metal.array import MetalArray

        metal_context = getattr(self, "metal_context", None)
        if isinstance(values, MetalArray) and metal_context is not None:
            if values.buffer.context is not metal_context:
                raise ValueError("Global inverse transform waist belongs to another Metal context")
            if values.dtype != np.dtype(np.complex128):
                raise TypeError("resident Global inverse transform requires complex128")
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs

            nlat, nlon = int(r1 - r0), int(self.grid.nlon)
            nfreq = int(self.truncation + 1)
            batch = int(np.prod(waist.lead, dtype=np.int64)) if waist.lead else 1
            table_cache = getattr(self, "_metal_inverse_longitude_table_cache", {})
            table_key = (nlon, nfreq)
            if table_key not in table_cache:
                mode = np.arange(nfreq, dtype=np.float64)[:, None]
                longitude = np.arange(nlon, dtype=np.float64)[None, :]
                angle = (2.0 * np.pi / nlon) * mode * longitude
                table_cache[table_key] = np.ascontiguousarray(
                    np.stack((np.cos(angle), np.sin(angle)), axis=-1) / nlon,
                    dtype=np.float64,
                )
                self._metal_inverse_longitude_table_cache = table_cache
            metal = MetalArrayModule(metal_context)
            grid = metal.empty((batch * nlat * nlon,), dtype=np.float64)
            metal_context.submit_kernel(
                "global_inverse_longitude_transform_resident_fragment",
                {"spectrum": values.view((2 * values.size,), dtype=np.float64),
                 "twiddles": metal.asarray(
                     table_cache[table_key].reshape(-1)),
                 "grid": grid},
                {"nlat": nlat, "nlon": nlon, "nfreq": nfreq,
                 "batch": batch, "full_nlat": int(self.grid.nlat),
                 "start_row": int(r0)},
                exact_policy=False, grid_size=batch * nlat * nlon,
                threads_per_group=(
                    kernel_specs.GLOBAL_INVERSE_LONGITUDE_TRANSFORM_RESIDENT_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            if keep_resident:
                if r0 != 0 or r1 != int(self.grid.nlat):
                    raise ValueError(
                        "resident Global synthesis requires a whole-latitude launch"
                    )
                return grid.view((*waist.lead, nlat, nlon), dtype=np.float64)
            return grid.to_numpy().reshape(*waist.lead, nlat, nlon)

        xp = self.backend.xp
        spectrum = waist.drain_scratch(r1 - r0)
        xp.multiply(
            self.grid.nlon,
            values[..., r0:r1, :],
            out=spectrum[..., : self.truncation + 1],
        )
        del values
        metal_context = getattr(self, "metal_context", None)
        if metal_context is not None and self.backend.name == "numpy":
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs

            nlat = int(r1 - r0)
            nlon = int(self.grid.nlon)
            nfreq = int(self.truncation + 1)
            batch = int(np.prod(waist.lead, dtype=np.int64)) if waist.lead else 1
            spectrum_host = np.ascontiguousarray(
                spectrum[..., :nfreq].reshape(batch, nlat, nfreq),
                dtype=np.complex128,
            )
            table_cache = getattr(
                self, "_metal_inverse_longitude_table_cache", {}
            )
            table_key = (nlon, nfreq)
            if table_key not in table_cache:
                mode = np.arange(nfreq, dtype=np.float64)[:, None]
                longitude = np.arange(nlon, dtype=np.float64)[None, :]
                angle = (2.0 * np.pi / nlon) * mode * longitude
                table_cache[table_key] = np.ascontiguousarray(
                    np.stack((np.cos(angle), np.sin(angle)), axis=-1) / nlon,
                    dtype=np.float64,
                )
                self._metal_inverse_longitude_table_cache = table_cache
            metal = MetalArrayModule(metal_context)
            real_spectrum = metal.asarray(spectrum_host.real.reshape(-1))
            imag_spectrum = metal.asarray(spectrum_host.imag.reshape(-1))
            twiddles = metal.asarray(table_cache[table_key].reshape(-1))
            grid = metal.empty((batch * nlat * nlon,), dtype=np.float64)
            metal_context.submit_kernel(
                "global_inverse_longitude_transform_fragment",
                {"real_spectrum": real_spectrum, "imag_spectrum": imag_spectrum,
                 "twiddles": twiddles, "grid": grid},
                {"nlat": nlat, "nlon": nlon, "nfreq": nfreq,
                 "batch": batch},
                exact_policy=False,
                grid_size=batch * nlat * nlon,
                threads_per_group=(
                    kernel_specs.GLOBAL_INVERSE_LONGITUDE_TRANSFORM_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            return grid.to_numpy().reshape(*waist.lead, nlat, nlon)
        if self.backend.name == "cupy":
            # cuFFT's complex-to-real transform destroys its input, so
            # cupy.fft.irfft copies the spectrum first; the spectrum here
            # is this call's own buffer and dead afterwards, so the
            # SciPy-style entry point is told it may be overwritten.  Same
            # cuFFT plan, same inverse scaling code path
            # (cupy.fft._fft._fft with the same arguments), so the same
            # bits, without the 1.15 GiB copy per six-field synthesis at
            # T533 float32 (allocator probe, 2026-09-02).
            from cupyx.scipy import fft as device_fft

            grid = device_fft.irfft(
                spectrum, n=self.grid.nlon, axis=-1, overwrite_x=True
            )
        else:
            grid = xp.fft.irfft(spectrum, n=self.grid.nlon, axis=-1)
        del spectrum
        # copy=False: the cast is a no-op copy when the FFT already returned
        # the backend dtype (cupy keeps single precision), and the same
        # float64 -> float32 rounding as before when it did not (numpy).
        return grid.astype(self.backend.float_dtype, copy=False)

    def waist_band_to_grid(self, waist: FourierWaist, r0: int, r1: int):
        """Drain one latitude band of a waist without consuming it.

        The entry point a band pipeline uses: the waist stays alive for
        the bands still to come, and only this band's half spectrum and
        grid rows are allocated.
        """
        return self._rows_to_grid(waist.values, waist, r0, r1)

    def waist_to_grid(self, waist: FourierWaist, *, keep_resident: bool = False):
        """Drain a whole waist to a whole grid field, band by band.

        Consumes the waist: the last band takes the buffer, so a
        one-band drain releases the contraction output before the FFT
        allocates its output rather than beside it -- the lifetime the
        synthesis kept by hand before the split.
        """
        if keep_resident:
            from woof.metal.array import MetalArray

            if isinstance(waist.values, MetalArray):
                values = waist.take()
                return self._rows_to_grid(
                    values, waist, 0, self.grid.nlat, keep_resident=True
                )
        xp = self.backend.xp
        edges = waist.edges
        last = len(edges) - 1
        if last == 0:
            r0, r1 = edges[0]
            return self._rows_to_grid(waist.take(), waist, r0, r1)
        out = xp.empty(
            (*waist.lead, self.grid.nlat, self.grid.nlon),
            dtype=self.backend.float_dtype,
        )
        for index, (r0, r1) in enumerate(edges):
            out[..., r0:r1, :] = self._rows_to_grid(
                waist.take() if index == last else waist.values, waist, r0, r1
            )
        return out

    def _synthesize(self, coeff, table, *, bands: int | None = None,
                    keep_resident: bool = False):
        return self.waist_to_grid(
            self.contract_to_waist(coeff, table, bands=bands),
            keep_resident=keep_resident,
        )

    def inverse(self, coeff, *, keep_resident: bool = False):
        if self.order_exchange is not None and int(
                getattr(self.order_exchange, "world", 1)) > 1:
            return self.inverse_orders(coeff)
        return self._synthesize(
            coeff, self._basis, keep_resident=keep_resident
        )

    def inverse_meridional_derivative(self, coeff):
        """Return ∂field/∂latitude in grid space."""
        from woof.metal.array import MetalArray

        resident = isinstance(coeff, MetalArray)
        result = self._synthesize(
            coeff, self._derivative_basis, keep_resident=resident
        )
        if resident and (
            not isinstance(result, MetalArray)
            or result.dtype != np.dtype(np.float64)
        ):
            raise TypeError(
                "resident Global meridional derivative synthesis left FP64 Metal"
            )
        return result

    def _resident_zonal_derivative_coefficients(self, coeff):
        """Apply the resident zonal spectral derivative in FP64 Metal."""
        from woof.metal.array import MetalArray, MetalArrayModule

        context = getattr(self, "metal_context", None)
        if (not isinstance(coeff, MetalArray)
                or context is None
                or coeff.buffer.context is not context
                or coeff.dtype != np.dtype(np.complex128)):
            raise TypeError(
                "resident Global zonal derivative requires complex128 coefficients "
                "on this transform's Metal context"
            )
        self._validate_spectral(coeff)
        if coeff.size > 0x7FFFFFFF:
            raise ValueError("Global spectral derivative exceeds the int32 ABI")
        from woof.metal.generated import kernel_specs

        derivative = MetalArrayModule(context).empty(
            coeff.shape, dtype=np.complex128
        )
        context.submit_kernel(
            "global_semilag_zonal_derivative_fragment",
            {"coefficients": coeff.view((2 * coeff.size,), dtype=np.float64),
             "derivative": derivative.view((2 * derivative.size,), dtype=np.float64)},
            {"truncation": int(self.truncation),
             "coefficient_count": int(coeff.size)},
            exact_policy=False, grid_size=int(coeff.size),
            threads_per_group=(
                kernel_specs.GLOBAL_SEMILAG_ZONAL_DERIVATIVE_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        return derivative

    def inverse_zonal_derivative(self, coeff):
        xp = self.backend.xp
        from woof.metal.array import MetalArray

        if isinstance(coeff, MetalArray):
            derivative = self._resident_zonal_derivative_coefficients(coeff)
            result = self.inverse(derivative, keep_resident=True)
            if (not isinstance(result, MetalArray)
                    or result.dtype != np.dtype(np.float64)):
                raise TypeError(
                    "resident Global zonal derivative synthesis left FP64 Metal"
                )
            return result
        c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        # The 1j*m diagonal is a precomputed vector broadcast along the
        # order axis; above-triangle entries are zero in every projected
        # state, so scaling the full rectangle equals the retired per-order
        # loop.
        return self.inverse(c * self._zonal_wavenumber)

    def gradient(self, coeff):
        """Return physical eastward and northward derivatives (per metre)."""
        from woof.metal.array import MetalArray, MetalArrayModule

        if isinstance(coeff, MetalArray):
            context = getattr(self, "metal_context", None)
            east_waist, north_waist = self.gradient_waists(coeff, bands=1)
            east_raw = self.waist_to_grid(east_waist, keep_resident=True)
            north_raw = self.waist_to_grid(north_waist, keep_resident=True)
            if (not isinstance(east_raw, MetalArray)
                    or not isinstance(north_raw, MetalArray)):
                raise TypeError("resident Global gradient synthesis left Metal")
            from woof.metal.generated import kernel_specs

            metal = MetalArrayModule(context)
            east = metal.empty(tuple(east_raw.shape), dtype=np.float64)
            north = metal.empty(tuple(north_raw.shape), dtype=np.float64)
            count = int(east.size)
            radius = metal.asarray(np.asarray([self.grid.radius_m], dtype=np.float64))
            coslat = metal.asarray(np.ascontiguousarray(
                self._coslat.reshape(-1), dtype=np.float64
            ))
            context.submit_kernel(
                "global_semilag_gradient_metric_fragment",
                {"east_raw": east_raw.view((count,), dtype=np.float64),
                 "north_raw": north_raw.view((count,), dtype=np.float64),
                 "coslat": coslat, "radius": radius,
                 "east": east.view((count,), dtype=np.float64),
                 "north": north.view((count,), dtype=np.float64)},
                {"count": count, "nlat": int(self.grid.nlat),
                 "nlon": int(self.grid.nlon)},
                exact_policy=False, grid_size=count,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_GRADIENT_METRIC_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            return (east.view(tuple(east_raw.shape), dtype=np.float64),
                    north.view(tuple(north_raw.shape), dtype=np.float64))
        east = self.inverse_zonal_derivative(coeff) / (
            self.grid.radius_m * self._coslat
        )
        north = self.inverse_meridional_derivative(coeff) / self.grid.radius_m
        return east, north

    def gradient_waists(self, coeff, *, bands: int | None = None):
        """The two contractions of :meth:`gradient`, before their drains.

        A band pipeline needs the gradient one latitude band at a time,
        and both of its contractions produce the latitude axis, so both
        run whole and hand back waists for :meth:`gradient_band` to drain.
        The metric factors are per latitude row, so the band applies its
        own rows of them; the arithmetic is the whole call's, row by row.
        """
        xp = self.backend.xp
        from woof.metal.array import MetalArray

        if isinstance(coeff, MetalArray):
            derivative = self._resident_zonal_derivative_coefficients(coeff)
            return (
                self.contract_to_waist(derivative, self._basis, bands=bands),
                self.contract_to_waist(coeff, self._derivative_basis, bands=bands),
            )
        c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        return (
            self.contract_to_waist(
                c * self._zonal_wavenumber, self._basis, bands=bands
            ),
            self.contract_to_waist(c, self._derivative_basis, bands=bands),
        )

    def gradient_band(self, waists, r0: int, r1: int):
        """One latitude band of :meth:`gradient`, drained from its waists."""
        east_waist, north_waist = waists
        east = self.waist_band_to_grid(east_waist, r0, r1) / (
            self.grid.radius_m * self._coslat[int(r0):int(r1)]
        )
        north = self.waist_band_to_grid(north_waist, r0, r1) / self.grid.radius_m
        return east, north

    def laplacian(self, coeff):
        from woof.metal.array import MetalArray

        if isinstance(coeff, MetalArray):
            context = getattr(self, "metal_context", None)
            if (context is None or coeff.buffer.context is not context
                    or coeff.dtype != np.dtype(np.complex128)):
                raise TypeError(
                    "resident Global Laplacian requires complex128 coefficients "
                    "on this transform's Metal context"
                )
            self._validate_spectral(coeff)
            from woof.metal.global_spectral_diffusion import (
                submit_global_spectral_diffusion,
            )
            return submit_global_spectral_diffusion(
                coeff, np.asarray(self._laplacian_eigen, dtype=np.float64)
            )
        xp = self.backend.xp
        c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        return c * self._laplacian_eigen[:, None]

    def inverse_laplacian(self, coeff):
        from woof.metal.array import MetalArray

        if isinstance(coeff, MetalArray):
            context = getattr(self, "metal_context", None)
            if (context is None or coeff.buffer.context is not context
                    or coeff.dtype != np.dtype(np.complex128)):
                raise TypeError(
                    "resident Global inverse Laplacian requires complex128 "
                    "coefficients on this transform's Metal context"
                )
            self._validate_spectral(coeff)
            factors = np.zeros(self.truncation + 1, dtype=np.float64)
            eigen = np.asarray(self._laplacian_eigen, dtype=np.float64)
            factors[1:] = 1.0 / eigen[1:]
            from woof.metal.global_spectral_diffusion import (
                submit_global_spectral_diffusion,
            )
            return submit_global_spectral_diffusion(coeff, factors)
        xp = self.backend.xp
        c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        out = xp.zeros_like(c)
        eig = self._laplacian_eigen
        out[..., 1:, :] = c[..., 1:, :] / eig[1:, None]
        return self.project(out)

    def spectral_mean_square(self, coeff) -> float:
        c = self.backend.to_numpy(coeff)
        self._validate_spectral(c)
        weight = np.ones(self.truncation + 1, dtype=np.float64)
        weight[1:] = 2.0
        total = np.sum(np.abs(c) ** 2 * weight[None, :], axis=(-2, -1))
        return float(np.asarray(total).mean() / (4.0 * np.pi))

    def power_by_degree(self, coeff) -> np.ndarray:
        c = self.backend.to_numpy(coeff)
        weight = np.ones(self.truncation + 1, dtype=np.float64)
        weight[1:] = 2.0
        return np.sum(np.abs(c) ** 2 * weight[None, :], axis=-1)

    def constant_coeff(self, value: float):
        out = self.zeros()
        out[0, 0] = float(value) * math.sqrt(4.0 * math.pi)
        return out

    def add_grid_constant(self, coeff, value: float):
        out = self.backend.xp.asarray(coeff).copy()
        out[..., 0, 0] += float(value) * math.sqrt(4.0 * math.pi)
        return out

    def transform_check(self, *, seed: int = 0, max_degree: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        t = self.truncation if max_degree is None else min(self.truncation, int(max_degree))
        coeff = np.zeros(self.spectral_shape, dtype=np.complex128)
        for n in range(t + 1):
            for m in range(n + 1):
                coeff[n, m] = rng.normal() + (0.0j if m == 0 else 1j * rng.normal())
        grid = self.inverse(self.backend.asarray(coeff, dtype=self.backend.complex_dtype))
        back = self.backend.to_numpy(self.forward(grid))
        scale = max(1.0, float(np.max(np.abs(coeff))))
        error = float(np.max(np.abs(back - coeff)) / scale)
        spatial_ms = self.grid.global_mean(self.backend.to_numpy(grid) ** 2)
        spectral_ms = float(
            np.sum(
                np.abs(coeff) ** 2
                * np.where(np.arange(self.truncation + 1)[None, :] == 0, 1.0, 2.0)
            )
            / (4.0 * np.pi)
        )
        parseval = abs(spatial_ms - spectral_ms) / max(1.0, abs(spectral_ms))
        return {
            "truncation": self.truncation,
            "nlat": self.grid.nlat,
            "nlon": self.grid.nlon,
            "roundtrip_relative_linf": error,
            "parseval_relative_error": float(parseval),
        }
