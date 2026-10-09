"""Latitude bands: the deterministic reduction contract, and the device slab.

Grid space is cut into latitude bands; spectral space stays whole.  Two
things in that sentence are arithmetic and therefore live here rather
than in the operators that use them.

**The reduction contract.**  A global reduction evaluated band by band in
whatever order the bands happen to finish is a different number every
time it is asked.  Every global reduction in the step is written here as
TWO stages:

1. a band-local stage that reduces only the axes that live inside one
   band (levels, longitude, species) and WRITES its result into a
   resident buffer whose layout is a pure function of ``(nlat, shape)``
   and never of the band count, the band order or the card;
2. a whole stage that reduces that resident buffer once, in grid order.

Stage 2 sees the same operand in the same layout whether the run took one
band or thirty-two, so the answer does not move.  The accumulators below
are that contract, and they are used on the RESIDENT path first: a
resident run fills one band covering every row, so the resident run and
the banded run execute the same two stages over the same buffer and
compute the identical number rather than two numbers that happen to
agree.

Reductions that are exactly associative in floating point (minimum,
maximum, all, any) need no buffer and fold in any order;
:class:`AssociativeAccumulator` carries them and REFUSES a sum, because a
floating-point sum folded band by band is exactly the defect the other
two classes exist to prevent.

**The device slab.**  CuPy's pool holds more than the run has live: the
gap MEASURED 2026-09-06 on an RTX 5090 was 1.235 at T383 (19.220 GiB held
against 15.560 GiB live) and 1.086 at T533, and every capacity figure in
the scale-out design pays for it.  :class:`SlabAllocator` takes one
contiguous arena from the driver before the first model byte and serves
every allocation out of it, so the bytes the card holds for the run are
the arena, once, and the receipt reports the run's own internal
fragmentation instead of a pool's opaque held total.
"""
from __future__ import annotations

import threading

import numpy as np

GIB = 2**30

# The smallest band the strided-copy measurement of 2026-09-06 reaches
# full contiguous bandwidth at (RTX 5070 Ti): below four latitude rows a
# T533 band slab copies at half rate.
MIN_BAND_ROWS = 4


def band_edges(nlat: int, bands: int) -> list[int]:
    """``floor(k * nlat / B)`` for ``k = 0..B``: the band boundaries.

    A pure function of ``(nlat, bands)``.  It is never a function of the
    card count or of which card runs which band, which is what makes a
    two-card answer equal to a one-card answer.
    """
    n = int(nlat)
    b = int(bands)
    if n < 1:
        raise ValueError(f"nlat must be >= 1, got {nlat!r}")
    if b < 1:
        raise ValueError(f"bands must be >= 1, got {bands!r}")
    if b > n:
        raise ValueError(
            f"{b} bands over {n} latitude rows would leave a band empty: an "
            "empty band writes no rows, so its reduction buffer would keep "
            "uninitialised values"
        )
    edges = [(k * n) // b for k in range(b + 1)]
    edges[-1] = n
    return edges


def band_slices(nlat: int, bands: int) -> list[slice]:
    """The band edges as slices, in ascending latitude-row order."""
    edges = band_edges(nlat, bands)
    return [slice(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


class BandPipeline:
    """How many latitude bands grid-space work is streamed through, and
    which rows each one holds.

    The whole of the streaming decomposition is here: a band count, the
    edges :func:`band_edges` derives from it, and the halo width a sweep
    that reads its neighbours needs.  Nothing else in the model chooses a
    row range, so a band schedule is a pure function of ``(nlat, bands)``
    and never of the card that runs a band or the order the bands finish
    in -- which is what lets a banded answer be the resident answer.

    ONE BAND IS THE RESIDENT RUN, and the code says so rather than
    branching around it: :attr:`resident` is ``True`` when the single band
    covers every row, and the operators use it to keep the whole-globe
    memory route the shipped default has today (a route choice, measured;
    the arithmetic is the same either way and gate BIT-4 holds the two
    against each other).

    The floor of :data:`MIN_BAND_ROWS` rows a band is the strided-copy
    measurement of 2026-09-06 (RTX 5070 Ti: below four latitude rows a
    T533 band slab copies at half rate), and it is a REFUSAL rather than a
    silent clamp: a run that asked for more bands than the grid can carry
    at that width has asked for a schedule this model will not give it,
    and clamping would return a different band count from the one the
    receipt records.
    """

    def __init__(self, nlat: int, bands: int = 1, *, minimum_rows: int = MIN_BAND_ROWS,
                 exchange=None):
        n = int(nlat)
        b = int(bands)
        if b < 1:
            raise ValueError(
                f"latitude_bands must be >= 1, got {bands!r}: zero or negative "
                "bands is not a streaming granularity"
            )
        floor = max(1, int(minimum_rows))
        if b > 1 and n // b < floor:
            raise ValueError(
                f"{b} latitude bands over {n} rows gives {n // b} rows a band, "
                f"below the {floor}-row floor where a band slab copies at half "
                "the contiguous rate (MEASURED 2026-09-06, RTX 5070 Ti, T533); "
                f"the widest band count for this grid is {max(1, n // floor)}"
            )
        self.nlat = n
        self.bands = b
        self._slices = tuple(band_slices(n, b))
        #: The row exchange of a multi-card run, or None.  It owns which
        #: bands this rank executes; the SCHEDULE above is unchanged by
        #: it, which is the point (gate BIT-6).
        self.exchange = None
        self.attach_exchange(exchange)

    def attach_exchange(self, exchange) -> None:
        """Name the cards this rank shares the band schedule with.

        The schedule itself does not move: :func:`band_edges` is a pure
        function of ``(nlat, bands)`` and never of the card count, so
        attaching an exchange changes only WHICH of those bands this
        process executes and leaves every row boundary where it was.
        """
        self.exchange = exchange
        if exchange is None or getattr(exchange, "world", 1) <= 1:
            self._local = self._slices
            self._local_rows = (0, self.nlat)
            return
        first, last = exchange.owned_rows()
        local = tuple(
            s for s in self._slices if s.start >= first and s.stop <= last
        )
        if not local:
            raise ValueError(
                f"this card owns latitude rows {first}..{last} and no whole "
                f"band of the {self.bands}-band schedule falls inside them: "
                "a card must own whole bands, because a band is the unit the "
                "reduction buffers and the waist are written in"
            )
        covered = sum(s.stop - s.start for s in local)
        if covered != last - first:
            raise ValueError(
                f"this card's rows {first}..{last} are not a whole number of "
                f"bands of the {self.bands}-band schedule (covered {covered} "
                f"of {last - first}): the band assignment and the row range "
                "must be the same partition"
            )
        self._local = local
        self._local_rows = (int(first), int(last))

    def local_slices(self) -> tuple[slice, ...]:
        """The bands THIS card executes, in ascending latitude order.

        One card is every band, which is why the operators loop over this
        rather than over :meth:`slices` and the single-card path is the
        multi-card path with a world of one.
        """
        return self._local

    def local_rows(self) -> tuple[int, int]:
        """``(first, last)`` grid rows this card owns."""
        return self._local_rows

    @property
    def cards(self) -> int:
        return 1 if self.exchange is None else int(getattr(self.exchange, "world", 1))

    @property
    def resident(self) -> bool:
        """True when one band covers the globe on this card: the shipped
        default, and never true on a multi-card run (a card that held
        every row would leave the other card nothing)."""
        return len(self._slices) == 1 and self.cards == 1

    def slices(self) -> tuple[slice, ...]:
        return self._slices

    def __iter__(self):
        return iter(self._slices)

    def __len__(self) -> int:
        return len(self._slices)

    def rows(self, index: int) -> slice:
        return self._slices[int(index)]

    def widest(self) -> int:
        return max(s.stop - s.start for s in self._slices)

    def halo(self, rows: slice, width: int) -> tuple[slice, int, int]:
        """``(extended, lead, trail)`` for a sweep that reads ``width``
        neighbouring rows either side of ``rows``.

        The extension is clipped at the poles, where the meridional sweep
        has a wall rather than a neighbour, and the two counts say how many
        of the extended rows are halo so the caller can drop them again.
        A band computed on ``extended`` and cut back to ``rows`` holds the
        values the whole sweep would have written there, bit for bit,
        because the stencil that produced them read identical inputs --
        which is the deep halo's whole argument (gate HALO-1).
        """
        w = int(width)
        if w < 0:
            raise ValueError(f"halo width must be >= 0, got {width!r}")
        start = max(0, int(rows.start) - w)
        stop = min(self.nlat, int(rows.stop) + w)
        return slice(start, stop), int(rows.start) - start, stop - int(rows.stop)

    def receipt(self) -> dict[str, object]:
        return {
            "latitude_bands": int(self.bands),
            "latitude_rows": int(self.nlat),
            "widest_band_rows": int(self.widest()),
            "narrowest_band_rows": int(
                min(s.stop - s.start for s in self._slices)
            ),
            "resident": bool(self.resident),
            "cards": int(self.cards),
            "local_bands": len(self._local),
            "local_rows": list(self._local_rows),
        }


def widest_band_count(nlat: int, *, minimum_rows: int = MIN_BAND_ROWS) -> int:
    """The largest band count :class:`BandPipeline` will accept for a grid."""
    return max(1, int(nlat) // max(1, int(minimum_rows)))


class _RowCoverage:
    """Which latitude rows an accumulator has been given.

    A reduction over a buffer with an unwritten row reads whatever the
    allocator handed back, so the reduce refuses instead: a band schedule
    that skips a row, or two bands that claim one, is a scheduling defect
    and it is named at the reduction rather than discovered later as an
    answer that moves with the band count.
    """

    def __init__(self, nlat: int, name: str):
        self._nlat = int(nlat)
        self._name = str(name)
        self._written = np.zeros(int(nlat), dtype=bool)

    def claim(self, rows: slice) -> None:
        start, stop, stride = rows.indices(self._nlat)
        if stride != 1:
            raise ValueError(
                f"{self._name}: a band is a contiguous run of latitude rows, "
                f"got stride {stride}"
            )
        if stop <= start:
            raise ValueError(f"{self._name}: empty band {rows!r}")
        if bool(self._written[start:stop].any()):
            raise ValueError(
                f"{self._name}: latitude rows {start}..{stop} were already "
                "written by another band; two bands claiming one row would "
                "double-count the reduction"
            )
        self._written[start:stop] = True

    def claim_external(self, first: int, last: int) -> None:
        """ Rows another card computed and sent, marked written.

        A card gather fills the rows this rank does not own with the bytes
        the rank that owns them produced, so the buffer is complete in
        exactly the sense the reduce needs: every row holds the value the
        whole-globe run would have put there.
        """
        self._written[int(first):int(last)] = True

    def require_complete(self) -> None:
        if not bool(self._written.all()):
            missing = int((~self._written).sum())
            raise ValueError(
                f"{self._name}: {missing} of {self._nlat} latitude rows were "
                "never written, so the whole-buffer reduction would read "
                "uninitialised memory"
            )

    def complete(self) -> bool:
        return bool(self._written.all())

    def reset(self) -> None:
        self._written[:] = False


class LatitudeAccumulator:
    """A resident ``(..., nlat)`` buffer, written per band, reduced whole.

    The band-local stage hands in the reduction of everything inside its
    rows (levels, longitude, species); the whole stage runs
    :meth:`total`, one weighted sum over latitude in grid order.  The
    buffer's shape and dtype do not depend on the band count, so the sum
    stage sees one operand whatever the schedule was.
    """

    def __init__(
        self, xp, shape, dtype, *, nlat: int | None = None, name: str = "latitude",
        exchange=None,
    ):
        self._xp = xp
        self._exchange = exchange
        self._shape = tuple(int(s) for s in shape)
        if not self._shape:
            raise ValueError("a latitude accumulator needs at least one axis")
        self._nlat = int(self._shape[-1] if nlat is None else nlat)
        if self._nlat != self._shape[-1]:
            raise ValueError(
                f"the last axis of {self._shape} is not nlat={self._nlat}"
            )
        # empty, not zeros: every row is written before the reduce, and
        # the coverage check refuses the case where it is not.
        self._buffer = xp.empty(self._shape, dtype=dtype)
        self._coverage = _RowCoverage(self._nlat, name)
        self.name = str(name)
        self._gathered = False

    @property
    def buffer(self):
        return self._buffer

    def _gather_cards(self) -> None:
        """ Fill the rows the other cards own before the whole stage runs.

        The whole stage reduces the buffer in GRID order, so it has to see
        every row whoever computed it.  The buffer is small by
        construction (32 KB for the transport column masses, 256 KB for
        the level water masses at T533) and it crosses the wire once per
        reduction, which is why the reduction contract survives a card
        split at a cost that does not appear in the step.
        """
        if self._gathered or self._exchange is None:
            return
        if int(getattr(self._exchange, "world", 1)) <= 1:
            return
        if self._coverage.complete():
            # Every row is already here, so there is nothing to fetch: the
            # buffer was filled from a quantity synthesised whole out of
            # the replicated spectral state rather than computed band by
            # band out of partitioned grid state.  Both cards reach this
            # branch together, because coverage is a function of the code
            # path and both cards run the same one.
            self._gathered = True
            return
        self._exchange.fill_rows(
            self._xp, self._buffer, self._buffer.ndim - 1, name=self.name)
        self._coverage.claim_external(0, self._nlat)
        self._gathered = True

    @property
    def nlat(self) -> int:
        return self._nlat

    def add_band(self, rows: slice, value) -> None:
        """Write one band's rows: ``value`` carries the buffer's shape
        with ``rows.stop - rows.start`` in place of ``nlat``."""
        self._coverage.claim(rows)
        self._buffer[..., rows] = value

    def complete(self):
        """The filled buffer itself, for the reductions whose whole stage
        is not a sum: the zonal sweep's per-row Courant maximum is scanned
        as a vector, and the scan has to see every row."""
        self._gather_cards()
        self._coverage.require_complete()
        return self._buffer

    def total(self, weights=None, *, axis: int = -1):
        """The whole-latitude reduction, once, in grid order."""
        self._gather_cards()
        self._coverage.require_complete()
        if weights is None:
            return self._xp.sum(self._buffer, axis=axis)
        return self._xp.sum(self._buffer * weights, axis=axis)

    def reset(self) -> None:
        self._coverage.reset()
        self._gathered = False


class PlaneAccumulator:
    """A resident ``(..., nlat, nlon)`` plane, written per band, reduced whole.

    For the reductions whose whole stage is a flat sum over a horizontal
    plane (the column-hole filler's created water, the mass fixer's
    surface pressure): a flat sum cannot be folded band by band without
    changing its last bits, so the plane itself is the accumulator and
    the flat sum runs once over it.  At T533 the plane is 5.13 MB.
    """

    def __init__(self, xp, shape, dtype, *, name: str = "plane", exchange=None):
        self._xp = xp
        self._exchange = exchange
        self._shape = tuple(int(s) for s in shape)
        if len(self._shape) < 2:
            raise ValueError("a plane accumulator needs (..., nlat, nlon)")
        self._nlat = int(self._shape[-2])
        self._buffer = xp.empty(self._shape, dtype=dtype)
        self._coverage = _RowCoverage(self._nlat, name)
        self.name = str(name)
        self._gathered = False

    @property
    def plane(self):
        if (
            not self._gathered
            and self._exchange is not None
            and int(getattr(self._exchange, "world", 1)) > 1
        ):
            if self._coverage.complete():
                # Filled whole from the replicated spectral state; see
                # LatitudeAccumulator._gather_cards.
                self._gathered = True
            else:
                self._exchange.fill_rows(
                    self._xp, self._buffer, self._buffer.ndim - 2,
                    name=self.name)
                self._coverage.claim_external(0, self._nlat)
                self._gathered = True
        self._coverage.require_complete()
        return self._buffer

    @property
    def nlat(self) -> int:
        return self._nlat

    def add_band(self, rows: slice, value) -> None:
        self._coverage.claim(rows)
        self._buffer[..., rows, :] = value

    def total(self, weights=None):
        plane = self.plane
        if weights is None:
            return self._xp.sum(plane)
        return self._xp.sum(plane * weights)

    def reset(self) -> None:
        self._coverage.reset()
        self._gathered = False


class AssociativeAccumulator:
    """A fold of an EXACTLY associative reduction over the bands.

    Minimum, maximum, all and any are exact in floating point in any
    order, so these fold as the bands finish and no buffer is needed.  A
    sum is refused by name: a floating-point sum folded band by band is a
    different number for a different band count, which is the whole
    reason :class:`LatitudeAccumulator` exists.
    """

    _OPS = ("min", "max", "all", "any")
    _REFUSED = ("sum", "add", "mean", "prod", "product")

    def __init__(self, xp, op: str, *, name: str = "associative", exchange=None):
        key = str(op)
        if key in self._REFUSED:
            raise ValueError(
                f"{key!r} is not exactly associative in floating point, so "
                "folding it band by band makes the answer depend on the band "
                "count; use LatitudeAccumulator or PlaneAccumulator"
            )
        if key not in self._OPS:
            raise ValueError(f"op must be one of {self._OPS}, got {op!r}")
        self._xp = xp
        self._op = key
        self._value = None
        self.bands = 0
        self.name = str(name)
        self._exchange = exchange

    def add_band(self, value) -> None:
        xp = self._xp
        if self._op == "min":
            band = xp.min(value)
            self._value = band if self._value is None else xp.minimum(self._value, band)
        elif self._op == "max":
            band = xp.max(value)
            self._value = band if self._value is None else xp.maximum(self._value, band)
        elif self._op == "all":
            band = xp.all(value)
            self._value = band if self._value is None else (self._value & band)
        else:
            band = xp.any(value)
            self._value = band if self._value is None else (self._value | band)
        self.bands += 1

    def total(self):
        if self._value is None:
            raise ValueError(f"{self.name}: no band was folded")
        if (
            self._exchange is not None
            and int(getattr(self._exchange, "world", 1)) > 1
        ):
            # Exact in any order, so crossing the wire costs the answer
            # nothing; it still folds in ascending rank order so a rerun
            # and a receipt read the same.
            return self._exchange.fold(
                self._xp, self._value, self._op, name=self.name)
        return self._value


def associative_over(xp, op: str, value, *, name: str = "associative",
                     exchange=None):
    """One band's fold of an exactly associative reduction, in the class
    the banded path uses.

    The resident path is the single-band case, so both paths run the same
    fold and there is one implementation of each of these reductions
    rather than two that happen to agree.
    """
    accumulator = AssociativeAccumulator(xp, op, name=name, exchange=exchange)
    accumulator.add_band(value)
    return accumulator.total()


# ---------------------------------------------------------------------
# The two-phase slab allocator
# ---------------------------------------------------------------------

ALIGNMENT = 512

#: The first segment is a fraction of the door's device-peak prediction
#: rather than the whole of it.  MEASURED 2026-09-06 on an RTX 5070 Ti:
#: the prediction over-read a T85 reference run's live peak by 6.2x
#: (3.57 GiB predicted against 0.572 GiB live), so an arena taken at the
#: prediction held seven times what the run used.  The slab reserves a
#: slice and EXTENDS to what the run actually asks for.
INITIAL_FRACTION = 0.125

#: An extension is at least this fraction of what is already held, so the
#: number of driver calls stays logarithmic in the peak while the held
#: total stays inside this fraction of the high-water mark.
GROWTH_FRACTION = 0.125

#: No segment smaller than this: a hundred tiny extensions cost more
#: driver calls than they save bytes.
MIN_SEGMENT_BYTES = 64 * 2**20

#: And none larger, unless one request needs it.  MEASURED 2026-09-06 on
#: an RTX 5070 Ti, ten-step T255 native: with the extension free to be an
#: eighth of what was already held, the last one added 0.99 GiB to reach a
#: 6.553 GiB live peak and the arena finished at 8.956 GiB, held over live
#: 1.367 -- worse than the CuPy default pool's 1.222 on the same run.  The
#: cap bounds that overshoot at a quarter of a GiB, at the cost of a few
#: dozen more driver calls during the first step's ramp.
MAX_SEGMENT_BYTES = 256 * 2**20


class _Block:
    """One cut of a segment.  ``free`` blocks are the free list's.

    ``offset`` is the block's place in the allocator's own flat address
    space, in which segment k occupies ``[origin_k, origin_k + size_k)``.
    A block never spans two segments, because a new segment starts life as
    one free root block and ``prev``/``next`` are only ever linked inside
    one segment.
    """

    __slots__ = ("offset", "size", "free", "prev", "next", "segment")

    def __init__(self, offset: int, size: int, segment: int):
        self.offset = int(offset)
        self.size = int(size)
        self.segment = int(segment)
        self.free = True
        self.prev = None
        self.next = None


class SlabHandle:
    """The owner object a served pointer keeps alive.

    CuPy frees an :class:`cupy.cuda.UnownedMemory` by dropping its owner,
    so the block returns to the free list exactly when the last array
    referencing it goes away, with no finaliser thread and no polling.
    """

    __slots__ = ("_allocator", "_offset", "_segment", "__weakref__")

    def __init__(self, allocator: "SlabAllocator", offset: int, segment):
        self._allocator = allocator
        self._offset = int(offset)
        # The segment is kept alive by every block served out of it, so a
        # run that still holds an array cannot have the ground freed under
        # it when the allocator is uninstalled at the end of the run.
        self._segment = segment

    def __del__(self):  # pragma: no cover - interpreter teardown paths
        try:
            self._allocator._release(self._offset)
        except Exception:
            pass


class SlabAllocator:
    """Two-phase device allocator: reserve an arena, then serve from it.

    **Phase 1, reserve.**  A contiguous device segment is taken straight
    from the driver before the first model byte, sized at
    :data:`INITIAL_FRACTION` of the door's own device-peak prediction and
    clamped to what the card can give.

    **Phase 2, serve and extend.**  Every allocation is a cut of the
    arena: an exact-size free block first (the step asks for the same
    shapes every step, so this is the steady-state path and it is O(1)),
    otherwise the smallest free block that fits, split at
    :data:`ALIGNMENT`.  A release coalesces with its free neighbours
    immediately, so a run cannot walk itself into a sawtooth.  A request
    no segment can serve EXTENDS the arena by one more segment rather
    than the size being guessed up front: MEASURED 2026-09-06, the door's
    prediction over-read a T85 run's live peak by 6.2x, and an arena taken
    at a prediction is only ever as good as that prediction.  What the
    card holds for the run therefore tracks what the run reached, and the
    receipt reports the gap between the two.

    A request that even an extension cannot satisfy falls through to a
    CuPy pool and is counted, sized and named in the receipt rather than
    silently changing what the run holds.  ``used_bytes`` and
    ``total_bytes`` carry the CuPy pool interface, so the shipped
    device-peak hook (:mod:`woof.globe.device_memory`) measures a
    slab run exactly as it measures a pool run.

    Not stream-safe by construction: it serves one stream, which is what
    the model uses today.  A caller that introduces a second stream owns
    the synchronisation, and this docstring is the notice.
    """

    name = "slab"

    def __init__(
        self,
        arena_bytes: int,
        *,
        ceiling_bytes: int | None = None,
        device_id: int | None = None,
        alignment: int = ALIGNMENT,
        fallback=None,
        arena_reason: str = "",
    ):
        self._requested_arena_bytes = int(arena_bytes)
        self._ceiling_bytes = None if ceiling_bytes is None else int(ceiling_bytes)
        self._alignment = int(alignment)
        self._device_id = device_id
        self._fallback = fallback
        self._arena_reason = str(arena_reason)
        self._lock = threading.RLock()
        self._segments: list = []
        self._segment_origin: list[int] = []
        self._segment_size: list[int] = []
        self._arena_bytes = 0
        self._blocks: dict[int, _Block] = {}
        self._free_by_size: dict[int, dict[int, _Block]] = {}
        self._live_bytes = 0
        self._live_peak_bytes = 0
        self._live_blocks = 0
        self.allocations = 0
        self.extensions = 0
        self.overflow_allocations = 0
        self.overflow_peak_bytes = 0
        self._overflow_live: dict[int, int] = {}
        self._previous_allocator = None
        self._installed = False

    # -- the arena -----------------------------------------------------

    def _align(self, size: int) -> int:
        a = self._alignment
        return ((int(size) + a - 1) // a) * a

    def _extend(self, minimum: int) -> bool:
        """Take one more segment, at least ``minimum`` bytes.  False when
        the card or the ceiling will not give it."""
        import cupy as cp

        minimum = self._align(max(int(minimum), self._alignment))
        growth = min(
            max(
                self._align(int(self._arena_bytes * GROWTH_FRACTION)),
                MIN_SEGMENT_BYTES if self._arena_bytes else 0,
            ),
            MAX_SEGMENT_BYTES,
        )
        want = self._align(max(minimum, growth))
        if self._ceiling_bytes is not None:
            room = self._ceiling_bytes - self._arena_bytes
            if room < minimum:
                return False
            want = min(want, self._align(room))
            if want < minimum:
                want = minimum
        try:
            segment = cp.cuda.memory.Memory(want)
        except Exception:  # noqa: BLE001 - a card that will not extend is not a crash
            if want <= minimum:
                return False
            try:
                want = minimum
                segment = cp.cuda.memory.Memory(want)
            except Exception:  # noqa: BLE001
                return False
        index = len(self._segments)
        origin = self._arena_bytes
        self._segments.append(segment)
        self._segment_origin.append(origin)
        self._segment_size.append(int(want))
        self._arena_bytes += int(want)
        if self._device_id is None:
            self._device_id = int(cp.cuda.Device().id)
        root = _Block(origin, int(want), index)
        self._blocks[origin] = root
        self._insert_free(root)
        if index:
            self.extensions += 1
        return True

    def _open(self) -> None:
        import cupy as cp

        if self._segments:
            return
        if self._fallback is None:
            self._fallback = cp.get_default_memory_pool()
        first = self._align(max(self._requested_arena_bytes, self._alignment))
        if not self._extend(first):
            raise MemoryError(
                f"the slab allocator could not reserve its first {first} B "
                "segment on this card"
            )

    def _pointer_of(self, block: _Block) -> int:
        index = block.segment
        return int(self._segments[index].ptr) + (
            block.offset - self._segment_origin[index]
        )

    # -- the free list -------------------------------------------------

    def _insert_free(self, block: _Block) -> None:
        block.free = True
        self._free_by_size.setdefault(block.size, {})[block.offset] = block

    def _remove_free(self, block: _Block) -> None:
        bucket = self._free_by_size.get(block.size)
        if bucket is not None:
            bucket.pop(block.offset, None)
            if not bucket:
                del self._free_by_size[block.size]
        block.free = False

    def _take(self, size: int) -> _Block | None:
        exact = self._free_by_size.get(size)
        if exact:
            block = next(iter(exact.values()))
            self._remove_free(block)
            return block
        best = None
        for candidate_size in self._free_by_size:
            if candidate_size >= size and (best is None or candidate_size < best):
                best = candidate_size
        if best is None:
            return None
        block = next(iter(self._free_by_size[best].values()))
        self._remove_free(block)
        if block.size - size >= self._alignment:
            rest = _Block(block.offset + size, block.size - size, block.segment)
            rest.prev = block
            rest.next = block.next
            if block.next is not None:
                block.next.prev = rest
            block.next = rest
            block.size = size
            self._blocks[rest.offset] = rest
            self._insert_free(rest)
        return block

    def _release(self, offset: int) -> None:
        with self._lock:
            if self._overflow_live.pop(offset, None) is not None:
                return
            block = self._blocks.get(offset)
            if block is None or block.free:
                return
            self._live_bytes -= block.size
            self._live_blocks -= 1
            self._insert_free(block)
            nxt = block.next
            if nxt is not None and nxt.free:
                self._remove_free(block)
                self._remove_free(nxt)
                block.size += nxt.size
                block.next = nxt.next
                if nxt.next is not None:
                    nxt.next.prev = block
                del self._blocks[nxt.offset]
                self._insert_free(block)
            prev = block.prev
            if prev is not None and prev.free:
                self._remove_free(prev)
                self._remove_free(block)
                prev.size += block.size
                prev.next = block.next
                if block.next is not None:
                    block.next.prev = prev
                del self._blocks[block.offset]
                self._insert_free(prev)

    # -- the allocator interface ---------------------------------------

    def malloc(self, size):
        import cupy as cp

        want = self._align(max(int(size), 1))
        with self._lock:
            if not self._segments:
                self._open()
            block = self._take(want)
            if block is None and self._extend(want):
                block = self._take(want)
            if block is not None:
                block.free = False
                self._live_bytes += block.size
                self._live_blocks += 1
                if self._live_bytes > self._live_peak_bytes:
                    self._live_peak_bytes = self._live_bytes
                self.allocations += 1
                handle = SlabHandle(self, block.offset, self._segments[block.segment])
                memory = cp.cuda.UnownedMemory(
                    self._pointer_of(block), block.size, handle, self._device_id,
                )
                return cp.cuda.MemoryPointer(memory, 0)
            # Overflow: neither the arena nor an extension can serve this.
            self.allocations += 1
            self.overflow_allocations += 1
            pointer = self._fallback.malloc(want)
            live = int(self._fallback.used_bytes())
            if live > self.overflow_peak_bytes:
                self.overflow_peak_bytes = live
            return pointer

    def used_bytes(self) -> int:
        """Live bytes handed out: the slab's plus any overflow pool's."""
        overflow = 0 if self._fallback is None else int(self._fallback.used_bytes())
        return int(self._live_bytes) + overflow

    def total_bytes(self) -> int:
        """Bytes held from the driver: the arena plus any overflow pool's."""
        overflow = 0 if self._fallback is None else int(self._fallback.total_bytes())
        return int(self._arena_bytes) + overflow

    def free_all_blocks(self) -> None:
        """The pool interface's release hook.  The arena is the run's, so
        nothing is handed back; an overflow pool's cached blocks are."""
        if self._fallback is not None:
            self._fallback.free_all_blocks()

    # -- lifecycle -----------------------------------------------------

    def install(self) -> "SlabAllocator":
        import cupy as cp

        if self._installed:
            raise RuntimeError("the slab allocator is already installed")
        self._open()
        self._previous_allocator = cp.cuda.get_allocator()
        cp.cuda.set_allocator(self.malloc)
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        import cupy as cp

        cp.cuda.set_allocator(self._previous_allocator)
        self._installed = False

    def close(self) -> None:
        """Drop the allocator's own hold on the arena.

        The segments' device bytes go back when the LAST block served out
        of each one goes away, not here: every :class:`SlabHandle` keeps
        its segment alive, so a run that still holds an array at teardown
        keeps valid memory under it.
        """
        self.uninstall()
        with self._lock:
            self._blocks.clear()
            self._free_by_size.clear()
            self._segments = []
            self._segment_origin = []
            self._segment_size = []

    # -- the invariant SLAB-1 asserts ----------------------------------

    def audit(self) -> dict[str, int]:
        """The block table's own arithmetic, for the no-double-hand-out gate.

        Walks each segment front to back: every block is accounted for
        once, the sizes add up to the segment, no two live blocks overlap,
        and the live total equals the sum of the live blocks.
        """
        with self._lock:
            offsets = sorted(self._blocks)
            walked = 0
            live = 0
            live_blocks = 0
            free_blocks = 0
            adjacent_free_pairs = 0
            previous_free = False
            previous_segment = -1
            for offset in offsets:
                block = self._blocks[offset]
                if block.segment != previous_segment:
                    if previous_segment >= 0 and walked != (
                        self._segment_origin[previous_segment]
                        + self._segment_size[previous_segment]
                    ):
                        raise AssertionError(
                            f"slab segment {previous_segment} is covered to "
                            f"{walked} B, not to its end"
                        )
                    walked = self._segment_origin[block.segment]
                    previous_free = False
                    previous_segment = block.segment
                if offset != walked:
                    raise AssertionError(
                        f"slab block table has a hole or an overlap at "
                        f"{offset} (expected {walked})"
                    )
                walked += block.size
                if block.free:
                    free_blocks += 1
                    if previous_free:
                        adjacent_free_pairs += 1
                else:
                    live_blocks += 1
                    live += block.size
                previous_free = block.free
            if self._segments and walked != self._arena_bytes:
                raise AssertionError(
                    f"slab blocks cover {walked} B of a {self._arena_bytes} B arena"
                )
            if live != self._live_bytes:
                raise AssertionError(
                    f"slab live total {self._live_bytes} B against {live} B in "
                    "the block table"
                )
            return {
                "blocks": len(offsets),
                "segments": len(self._segments),
                "live_blocks": live_blocks,
                "free_blocks": free_blocks,
                "adjacent_free_pairs": adjacent_free_pairs,
                "live_bytes": live,
                "arena_bytes": int(self._arena_bytes),
                "live_peak_bytes": int(self._live_peak_bytes),
            }

    # -- the receipt ---------------------------------------------------

    def receipt(self) -> dict[str, object]:
        peak = int(self._live_peak_bytes)
        return {
            "allocator": self.name,
            "arena_bytes": int(self._arena_bytes),
            "arena_gib": round(self._arena_bytes / GIB, 3),
            "arena_first_segment_bytes": int(self._requested_arena_bytes),
            "arena_ceiling_bytes": (
                None if self._ceiling_bytes is None else int(self._ceiling_bytes)
            ),
            "arena_reason": self._arena_reason,
            "segments": len(self._segments),
            "extensions": int(self.extensions),
            "alignment_bytes": int(self._alignment),
            "slab_live_peak_bytes": peak,
            "slab_live_peak_gib": round(peak / GIB, 3),
            "arena_over_slab_live_peak": (
                None if peak <= 0 else round(self._arena_bytes / peak, 4)
            ),
            "allocations": int(self.allocations),
            "overflow_allocations": int(self.overflow_allocations),
            "overflow_peak_bytes": int(self.overflow_peak_bytes),
            "measures": (
                "device bytes served out of contiguous driver segments the "
                "allocator reserves and extends: arena_bytes is what the card "
                "holds for the run, slab_live_peak_bytes what the run had live "
                "at its highest, and arena_over_slab_live_peak the gap between "
                "them, which is the run's internal fragmentation plus whatever "
                "the last extension over-reserved"
            ),
        }


__all__ = [
    "ALIGNMENT",
    "GIB",
    "GROWTH_FRACTION",
    "INITIAL_FRACTION",
    "MAX_SEGMENT_BYTES",
    "MIN_BAND_ROWS",
    "MIN_SEGMENT_BYTES",
    "AssociativeAccumulator",
    "BandPipeline",
    "LatitudeAccumulator",
    "PlaneAccumulator",
    "SlabAllocator",
    "SlabHandle",
    "associative_over",
    "band_edges",
    "band_slices",
    "widest_band_count",
]
