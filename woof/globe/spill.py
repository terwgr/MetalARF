"""The pinned host tier: the persistent grid state the card is not holding.

A latitude band count divides the TRANSIENT part of a step -- the
syntheses, the batches, the flux stacks -- because each of those exists
inside one operator and dies there.  It does not divide the PERSISTENT
part: the ten grid tracers, the surface reservoirs and the native physics
namespace are alive from one step to the next whatever the band count is,
so they sit on the card for the whole run and every capacity figure pays
for them.

MEASURED 2026-09-06, RTX 5090, T533 L40 native at eight bands: that
persistent part is 4.388 GiB (tracers 1.912, physics namespace 2.333,
surface 0.143) of a 20.433 GiB live peak.

**What this module does.**  It parks the coldest of those arrays in PINNED
HOST MEMORY and hands the card a copy only where a copy is made anyway.
Every consumer of the persistent grid state builds its own device copy
already:

* ``NativeColumnBatch.from_exchange`` copies every levelled field into
  bottom-to-top contiguous float32 (its ``volume()``), copies the surface
  (``SurfaceState.copy()``) and hands the namespace to
  ``PersistentNativeState``, which copies every array it touches;
* the tracer transport builds its own stacks and returns new arrays;
* the checkpoint writer reads to host.

So a parked array's device copy is the copy the consumer was going to
make.  What stops existing is the ORIGINAL sitting on the card beside it
for the whole run.  That is the design's minimum-spill rule: shed exactly
enough, and shed the coldest first.

**Coldest first, and the order is counted rather than asserted.**  The
grid passes of one step (``dynamics.MoistHybridModel.step``) read:

===========================  ==========================  ===============
slice                        reads per step              T533 GiB
===========================  ==========================  ===============
native physics namespace     two (the physics halves)    2.333
surface reservoirs           two (the physics halves)    0.143
grid tracers                 three (the halves and the   1.912
                             tracer transport)
===========================  ==========================  ===============

so the tier fills in that order and stops as soon as the budget is met.

**Bit identity.**  Nothing here is arithmetic.  A parked array holds the
same bytes in host memory that it held on the card, a staged copy is a
memcpy, and the consumer runs the same kernel on the same values.  The
ten-step T255 checkpoint gate runs with the tier forced on and is
compared to the resident run byte for byte (gate BIT-1, spill on).

**Overlap.**  Transfers run on their own streams -- one for
host-to-device and one for device-to-host, which the PCIe link runs
concurrently (MEASURED 2026-09-06: 55.6/56.8 GB/s in, 42.3/42.6 out,
70.8/73.6 together) -- and the tier double buffers, so the copy of the
next array is in flight while the current one is being consumed.  The
receipt records the achieved transfer time against the step time (gate
OVERLAP-1): a spill configuration that is NOT hidden says so rather than
running a campaign fifteen percent slow.
"""
from __future__ import annotations

import os

import numpy as np

#: Diagnostic switch: retire every write-back before returning from
#: :meth:`HostTier.store`.  It removes the overlap and is not a shipping
#: mode; it exists so a suspected ordering defect can be separated from a
#: logic one by one run rather than by argument.
SYNCHRONOUS_STORES = bool(os.environ.get("ARWEN_SPILL_SYNCHRONOUS_STORES"))

#: Diagnostic switch: after every park and every stage, synchronize and
#: compare the two copies, naming the first slot whose bytes do not
#: match.  Not a shipping mode -- it costs a synchronization and a whole
#: comparison per transfer -- and it is what turns "the answer moved"
#: into "this slot did".
VERIFY_TRANSFERS = bool(os.environ.get("ARWEN_SPILL_VERIFY"))

#: Diagnostic switch: stage on the compute stream instead of the inbound
#: one.  It removes the overlap on the inbound side and is not a shipping
#: mode; like the two above it exists to separate an ordering defect from
#: a logic one by measurement.
SYNCHRONOUS_STAGES = bool(os.environ.get("ARWEN_SPILL_SYNCHRONOUS_STAGES"))

GIB = 2**30

#: The tier's slices, coldest first.  The order IS the minimum-spill
#: policy: a run that needs two GiB of relief parks the physics namespace
#: and stops, rather than mirroring everything and paying the traffic of
#: the slices it did not need.
SLICES = ("physics", "surface", "tracers")

#: How many times one step reads each slice back onto the card, on the
#: EULERIAN core: the physics namespace and the surface by the two
#: physics halves, the tracers by those two and by the flux-form tracer
#: transport.  This is what makes the order above the cold-first order.
#:
#: The shipped semi-Lagrangian core reads the tracers FOUR times: the two
#: physics halves, the trajectory gather that is the sweep itself, and the
#: mass fixer's before-state (which the water reading beside it shares, so
#: it is one read and not two).  MEASURED 2026-09-07, RTX 5070 Ti, T255
#: L40 native at three steps with all three slices parked: 23.49 GiB
#: staged for 1.0085 GiB parked, 7.83 GiB a step.  Nothing admits or
#: refuses on these counts; they are the traffic's description, and the
#: receipt reports what the run actually staged rather than this table.
SLICE_READS_PER_STEP = {"physics": 2, "surface": 2, "tracers": 3}


def _is_cupy(xp) -> bool:
    return getattr(xp, "__name__", "") == "cupy"


def pinned_host_array(shape, dtype, *, xp=None) -> np.ndarray:
    """A host array in PAGE-LOCKED memory, shaped and typed as asked.

    Pinned pages are what make a transfer asynchronous and what reach the
    MEASURED 55.6 to 56.8 GB/s inbound rate; pageable host memory stages
    through a driver bounce buffer and cannot overlap with compute
    (MEASURED 4.2 to 6.2 GB/s to allocate, so the whole T799 tier costs
    about four seconds of allocation, once, at run start).

    One allocation per parked array rather than one arena cut with a bump
    pointer: what the tier holds is decided by the run, and the native
    physics namespace GROWS on its first call (the suite seeds the arrays
    a restart would have carried).  An arena sized before that first call
    would either refuse the run or be sized by a guess; a slot per array
    is sized by the array.

    On the numpy backend there is nothing to pin and this is a plain host
    allocation, so the CPU gates exercise the same object the card runs.
    """
    dtype = np.dtype(dtype)
    count = 1
    for extent in shape:
        count *= int(extent)
    nbytes = count * dtype.itemsize
    if xp is not None and _is_cupy(xp) and nbytes > 0:
        import cupy as cp

        memory = cp.cuda.alloc_pinned_memory(nbytes)
        view = np.frombuffer(memory, dtype=dtype, count=count).reshape(shape)
        return view
    return np.empty(shape, dtype=dtype)


class SpilledArray:
    """A persistent grid array whose truth is its pinned host copy.

    It carries the array interface a reader needs in order to decide what
    to do with it (``shape``, ``dtype``, ``ndim``, ``nbytes``) and
    NOTHING that would let it be mistaken for a device array: an operator
    handed one by accident raises rather than silently running the step
    on the host, which is the failure a transparent proxy would hide.
    """

    __slots__ = ("host", "name", "tier", "version", "__weakref__")

    def __init__(self, name: str, host: np.ndarray, tier: "HostTier"):
        self.name = str(name)
        self.host = host
        self.tier = tier
        #: Bumped by every write into the slot.  The physics boundary
        #: fingerprints a parked array by ``(name, version)`` instead of
        #: reducing it: a sum/min/max over the whole tier twice per
        #: physics call is 10 GiB of HOST reads a step at T533, and the
        #: version detects any write exactly rather than probabilistically.
        self.version = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.host.shape)

    @property
    def dtype(self):
        return self.host.dtype

    @property
    def ndim(self) -> int:
        return int(self.host.ndim)

    @property
    def nbytes(self) -> int:
        return int(self.host.nbytes)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"SpilledArray({self.name!r}, shape={self.shape}, "
            f"dtype={self.dtype}, host)"
        )

    def __array__(self, dtype=None, copy=None):
        """The host copy: what ``np.asarray`` and the checkpoint see.

        The slot's outstanding write is waited on first: an outbound copy
        is asynchronous, and a reader that skipped this would see half of
        one step and half of the next.
        """
        self.tier.host_ready(self)
        if dtype is None:
            return self.host
        return self.host.astype(dtype, copy=False)

    def stage(self, rows: slice | None = None):
        """A DEVICE copy of the whole array, or of one latitude band."""
        return self.tier.stage(self, rows)

    def store(self, value, rows: slice | None = None) -> None:
        """Write a device band (or the whole array) back into the tier."""
        self.tier.store(self, value, rows)

    def copy(self) -> "SpilledArray":
        """THE SAME PARKED ARRAY, and the reason is the tier's contract.

        ``SurfaceState.copy`` and ``PhysicsState.copy`` copy every array
        they hold, defensively: the step hands the copy on and never
        writes into it.  A parked array is never written in place either
        -- the step replaces a whole slot (:meth:`store`) and the replaced
        values are the ones every later reader wants -- so duplicating the
        host copy would double the tier and copy 2.3 GiB of host memory
        four times a step at T533 for a value nothing distinguishes.

        The one place a copy IS written, the water fixer's surface
        reservoir, replaces the slot explicitly rather than mutating it.
        A checkpoint takes its own host copy (``checkpoint.bundle_arrays``)
        because the writer thread outlives the step that produced it.
        """
        return self


def spilled(value) -> bool:
    """True for an array the tier holds rather than the card."""
    return isinstance(value, SpilledArray)


def prefetch(value, rows: slice | None = None) -> None:
    """Start a parked array's copy now; a no-op for one already on the card.

    THE DOUBLE BUFFER, called at the one place a caller knows what it
    will read next: a loop over a named list of arrays.  The copy of
    entry ``j + 1`` runs on the inbound stream while entry ``j`` is being
    consumed, so the step blocks on the link only for what the link could
    not finish in time.  MEASURED 2026-09-06 without it, T255 L40 native
    at one band on an RTX 5090: the step waited 0.301 s a step against
    0.217 s of inbound transfer -- MORE than the link was busy, because a
    copy issued immediately before its use has nothing to hide behind.

    It costs one extra staged buffer live at a time, which is the double
    buffer's own price and is bounded by construction.
    """
    if isinstance(value, SpilledArray):
        value.tier.prefetch(value, rows)


def resident(xp, value, rows: slice | None = None):
    """``value`` as a device array: the identity for one already there.

    This is the single door between the tier and every operator.  A
    reader that needs bytes on the card calls it; a reader that does not
    (the checkpoint writer, an exporter, the DA door's host build) reads
    ``.host`` and never touches the card at all.
    """
    if isinstance(value, SpilledArray):
        return value.stage(rows)
    if rows is None:
        return value
    from .dynamics import band_view

    return band_view(xp, value, rows)


class HostTier:
    """The pinned host tier, and the transfer pipeline that hides it.

    Three streams: the model's own (compute), one for host-to-device and
    one for device-to-host.  A staged copy is issued on the inbound
    stream, an event is recorded on it, and the compute stream waits on
    that event -- so the copy overlaps whatever the compute stream is
    still doing and the consumer never reads a half-arrived array.  A
    write-back waits on an event recorded on the compute stream, so the
    tier never reads a buffer the model is still writing.

    :meth:`prefetch` is the double buffer: a caller working through a
    list of parked arrays asks for the next one before consuming the
    current, and the two transfers queue behind each other on the
    inbound stream while the compute stream works.

    The device buffer a stage hands back is an ordinary allocation from
    whatever allocator the run installed, and it is freed when the
    consumer drops it.  The tier holds no device memory of its own, which
    is what makes it a capacity mechanism rather than a cache.
    """

    def __init__(self, xp, *, enabled: bool = True):
        self.xp = xp
        self.enabled = bool(enabled)
        self._cupy = _is_cupy(xp)
        self._in_stream = None
        self._out_stream = None
        self._pending: dict[int, tuple] = {}
        self.parked: dict[str, SpilledArray] = {}
        self.parked_bytes = 0
        self.staged_bytes = 0
        self.stored_bytes = 0
        self.stage_calls = 0
        self.store_calls = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.slices: list[str] = []
        self.transfer_s = 0.0
        self.transfer_in_s = 0.0
        self.transfer_out_s = 0.0
        #: The time the COMPUTE stream spent blocked on a transfer,
        #: measured with a pair of events either side of its wait.  This
        #: is the EXPOSED cost: what the step actually pays for the tier,
        #: against ``transfer_in_s``, what the link spent.  One minus
        #: their ratio is the achieved overlap, and it is gate OVERLAP-1.
        self.exposed_s = 0.0
        self.steps = 0
        self._out_pending: list = []
        self._timed: list = []
        # The last transfer event touching each slot, by direction.  A
        # slot is one host buffer that BOTH directions use, so the two
        # streams must be ordered against each other on it: an inbound
        # copy that starts while the outbound copy of the same slot is
        # still writing reads a torn array, and the values that come back
        # are half this step and half the last.  MEASURED as exactly that:
        # a T533 native step read a grid tracer at -1.35e-04 against a
        # field maximum of 0.0712, which the roundoff floor refused.
        self._slot_in: dict[int, object] = {}
        self._slot_out: dict[int, object] = {}
        if self.enabled and self._cupy:
            import cupy as cp

            self._in_stream = cp.cuda.Stream(non_blocking=True)
            self._out_stream = cp.cuda.Stream(non_blocking=True)

    # -- parking -------------------------------------------------------

    def park(self, name: str, value) -> SpilledArray:
        """Move one array off the card and into the tier's named slot.

        A slot of the right shape and dtype is reused, so a step that
        replaces the whole persistent state -- which every step does --
        allocates pinned memory once and writes into it forever after.
        The device array is released by the caller dropping its last
        reference; nothing here keeps one, which is the whole point.
        """
        if not self.enabled:
            raise RuntimeError(
                "the host tier is off, so nothing may be parked in it: a run "
                "that parked with the tier off would hold the array in "
                "pageable host memory and transfer it synchronously"
            )
        if isinstance(value, SpilledArray) and value.tier is self:
            self.parked[name] = value
            return value
        handle = self.parked.get(name)
        shape = tuple(int(v) for v in value.shape)
        dtype = np.dtype(value.dtype)
        if handle is None or handle.shape != shape or handle.dtype != dtype:
            host = pinned_host_array(shape, dtype, xp=self.xp)
            handle = SpilledArray(name, host, self)
            self.parked[name] = handle
            self.parked_bytes += int(host.nbytes)
        self.store(handle, value)
        if VERIFY_TRANSFERS:
            self._verify(handle, value, "park")
        return handle

    def _verify(self, handle, value, what: str) -> None:
        self.synchronize()
        want = np.asarray(value.get() if hasattr(value, "device") else value)
        got = np.asarray(handle.host)
        if want.shape != got.shape or not np.array_equal(want, got):
            bad = int(np.count_nonzero(want != got)) if want.shape == got.shape else -1
            raise AssertionError(
                f"host tier {what} of {handle.name!r} did not round trip: "
                f"{bad} of {got.size} cells differ (want "
                f"[{float(want.min()):.6g}, {float(want.max()):.6g}], got "
                f"[{float(got.min()):.6g}, {float(got.max()):.6g}])"
            )

    def open(self, name: str, shape, dtype) -> SpilledArray:
        """A slot of ``shape`` and ``dtype`` under ``name``, with nothing
        written into it yet: the destination of a state assembled a band
        at a time (:meth:`store` with ``rows``), which is how the physics
        half-step returns a namespace array that did not exist before the
        call (the suite seeds its namespace on its first call).  A slot of
        the right shape already under the name is reused, as
        :meth:`park` reuses it."""
        if not self.enabled:
            raise RuntimeError(
                "the host tier is off, so no slot may be opened in it: a run "
                "that parked with the tier off would hold the array in "
                "pageable host memory and transfer it synchronously"
            )
        shape = tuple(int(v) for v in shape)
        dtype = np.dtype(dtype)
        handle = self.parked.get(name)
        if handle is None or handle.shape != shape or handle.dtype != dtype:
            host = pinned_host_array(shape, dtype, xp=self.xp)
            handle = SpilledArray(name, host, self)
            self.parked[name] = handle
            self.parked_bytes += int(host.nbytes)
        return handle

    def hold(self, prefix: str, mapping) -> dict[str, SpilledArray]:
        """Park a whole named slice, returning it by its own keys.

        Double buffered: the write-back of one array retires while the
        next is being issued, so the tier fills at the link's rate rather
        than at the rate of one synchronous copy after another.
        """
        out: dict[str, SpilledArray] = {}
        for key, value in mapping.items():
            if value is None:
                out[key] = value
                continue
            out[key] = self.park(f"{prefix}__{key}", value)
        return out

    # -- transfers -----------------------------------------------------

    @staticmethod
    def _band(host: np.ndarray, rows: slice | None):
        if rows is None:
            return host
        return host[..., rows, :]

    def _order_in(self, handle) -> None:
        """The inbound stream waits for the slot's last outbound write."""
        event = self._slot_out.get(id(handle))
        if event is not None and self._in_stream is not None:
            self._in_stream.wait_event(event)

    def _order_out(self, handle) -> None:
        """The outbound stream waits for the slot's last inbound read."""
        event = self._slot_in.get(id(handle))
        if event is not None and self._out_stream is not None:
            self._out_stream.wait_event(event)

    def host_ready(self, handle) -> None:
        """Block until the slot's host bytes are final.

        Every read of ``handle.host`` from the CPU -- the checkpoint, a
        band cut, an exporter -- goes through here first: an outbound copy
        is asynchronous and the bytes are not there until it retires.
        """
        event = self._slot_out.get(id(handle))
        if event is not None:
            event.synchronize()

    def stage(self, handle: SpilledArray, rows: slice | None = None):
        """A device copy of ``handle``, or of one band of it."""
        pending = self._pending.pop(id(handle), None)
        if pending is not None:
            if pending[0] == rows:
                self.prefetch_hits += 1
                if pending[2] is not None:
                    self._wait_on(pending[2])
                return pending[1]
            # A prefetch of a band nobody asked for: finish it before its
            # source buffer is dropped, and count it rather than hiding it.
            self.prefetch_misses += 1
            if pending[2] is not None:
                pending[2].synchronize()
        if rows is not None:
            # A band is cut on the CPU, so the slot's bytes must be final
            # before the cut rather than merely ordered on a stream.
            self.host_ready(handle)
        source = np.ascontiguousarray(self._band(handle.host, rows))
        self.stage_calls += 1
        self.staged_bytes += int(source.nbytes)
        if not self._cupy:
            # A COPY, because on the card a stage is one: the numpy
            # backend is where the determinism gates run, and if a stage
            # aliased the slot there then a consumer that writes into a
            # staged array would change the tier on the CPU and not on
            # the card, and the gate that exists to catch exactly that
            # divergence would be the one hiding it.  ``ascontiguousarray``
            # of a contiguous slot returns the slot itself, so the copy is
            # taken here rather than relied upon above.
            return source.copy() if np.may_share_memory(source, handle.host)                 else source
        device = self._issue_in(source, handle)
        if VERIFY_TRANSFERS:
            self.xp.cuda.Stream.null.synchronize()
            got = device.get()
            if not np.array_equal(got, source):
                import cupy as cp

                again = cp.asarray(source).get()
                raise AssertionError(
                    f"host tier stage of {handle.name!r} did not round trip: "
                    f"{int(np.count_nonzero(got != source))} of {source.size} "
                    f"cells differ; async read [{float(got.min()):.6g}, "
                    f"{float(got.max()):.6g}] against host "
                    f"[{float(source.min()):.6g}, {float(source.max()):.6g}]; "
                    f"a synchronous re-copy "
                    f"{'matches' if np.array_equal(again, source) else 'does NOT match'}"
                    f"; stream {cp.cuda.get_current_stream()!r}"
                )
        return device

    def _issue_in(self, source: np.ndarray, handle=None):
        import cupy as cp

        if SYNCHRONOUS_STAGES:
            if handle is not None:
                self.host_ready(handle)
            return cp.asarray(source)
        device = cp.empty(source.shape, dtype=source.dtype)
        event = self._copy_in(device, source, handle)
        self._wait_on(event)
        return device

    def _wait_on(self, event) -> None:
        """The compute stream waits for one transfer, and the wait is timed.

        The two events sit either side of the wait on the COMPUTE stream,
        so the elapsed between them is the time the step was blocked on
        the link rather than the time the link was busy.  That difference
        is the whole of gate OVERLAP-1: a transfer that is hidden costs
        the step nothing here and shows up only in ``transfer_in_s``.
        """
        import cupy as cp

        stream = cp.cuda.get_current_stream()
        start = stream.record()
        stream.wait_event(event)
        stop = stream.record()
        self._timed.append(("exposed", start, stop))

    def _copy_in(self, device, source: np.ndarray, handle=None):
        """Issue one host-to-device copy on the inbound stream, ORDERED.

        Two orderings, and both were measured rather than assumed.

        1. The inbound stream waits for everything the compute stream has
           already queued.  The staging buffer comes from the pool, and
           the pool hands back a block whose PREVIOUS tenant a queued
           kernel may still be reading; an unordered copy overwrites it
           under that kernel.  MEASURED 2026-09-06, T255 L40 native on an
           RTX 5090: without this wait the physics batch read three of
           its thirteen tracer volumes wrong (qc, qg and nc), the ice and
           graupel fields came back different and the run died four
           operators later in the hybrid pressure.
        2. The inbound stream waits for the slot's own last write-back,
           because both directions use one host buffer.

        The consumer then waits for the copy, which is what makes a
        prefetch overlap: the copy is queued now and the compute stream
        only blocks on it when the array is actually read.
        """
        import cupy as cp

        guard = cp.cuda.get_current_stream().record()
        self._in_stream.wait_event(guard)
        if handle is not None:
            self._order_in(handle)
        with self._in_stream:
            start = self._in_stream.record()
            device.set(source, stream=self._in_stream)
            event = self._in_stream.record()
        self._timed.append(("in", start, event))
        if handle is not None:
            self._slot_in[id(handle)] = event
        return event

    def prefetch(self, handle: SpilledArray, rows: slice | None = None) -> None:
        """Start ``handle``'s transfer now; :meth:`stage` collects it.

        This is the double buffer.  Called for entry ``j + 1`` while
        entry ``j`` is being consumed, the copy runs on the inbound
        stream against the compute stream's work.
        """
        if not self.enabled or not self._cupy:
            return
        if id(handle) in self._pending:
            return
        import cupy as cp

        if rows is not None:
            self.host_ready(handle)
        source = np.ascontiguousarray(self._band(handle.host, rows))
        self.stage_calls += 1
        self.staged_bytes += int(source.nbytes)
        device = cp.empty(source.shape, dtype=source.dtype)
        event = self._copy_in(device, source, handle)
        # ``source`` must outlive the copy, so the pending entry holds it.
        self._pending[id(handle)] = (rows, device, event, source)

    def store(self, handle: SpilledArray, value, rows: slice | None = None) -> None:
        """Write a device array (or a band of one) back into the tier.

        The write is issued on the outbound stream after an event on the
        compute stream, so it overlaps the next operator; the tier holds
        a reference to the device array until the copy retires, because a
        pool that reused those bytes under an in-flight copy would write
        the next operator's values into the checkpoint.  :meth:`drain`
        and :meth:`synchronize` release them.
        """
        self.store_calls += 1
        handle.version += 1
        target = self._band(handle.host, rows)
        # ``hasattr(value, "device")`` is not the device test it reads as:
        # every numpy array has carried a ``device`` attribute since numpy
        # 2.0 (it answers "cpu"), so a host array reaching this method on
        # the cupy backend took the device branch and died in
        # ``source.get`` with an AttributeError instead of being written
        # into the slot.  ``get`` is the method the device branch actually
        # calls, so asking for it is the test that matches the branch.
        if not self._cupy or not hasattr(value, "get"):
            # A CPU write into the slot: every transfer either way must
            # have retired first, because neither stream is ordered
            # against the host's own store.
            self.host_ready(handle)
            event = self._slot_in.get(id(handle))
            if event is not None:
                event.synchronize()
            target[...] = np.asarray(value)
            self.stored_bytes += int(target.nbytes)
            return
        import cupy as cp

        source = value if value.flags.c_contiguous else cp.ascontiguousarray(value)
        if not target.flags.c_contiguous:
            # A band of a levelled array is strided in host memory (the
            # latitude axis is the middle one), so the copy is a 2-D one:
            # ``nlev`` rows of ``rows_n * nlon`` contiguous elements at a
            # pitch of ``nlat * nlon``.  Written out rather than staged
            # through a contiguous buffer, because staging would cost a
            # whole host-side pass over the band on every write.
            pass
        entry_event = cp.cuda.get_current_stream().record()
        self._order_out(handle)
        with self._out_stream:
            self._out_stream.wait_event(entry_event)
            start = self._out_stream.record()
            if target.flags.c_contiguous:
                source.get(out=target, stream=self._out_stream)
            else:
                self._memcpy_strided_out(source, target)
            done = self._out_stream.record()
        self._timed.append(("out", start, done))
        self._slot_out[id(handle)] = done
        self._out_pending.append((done, source, target))
        self.stored_bytes += int(target.nbytes)
        if SYNCHRONOUS_STORES:
            self.drain()
        elif len(self._out_pending) > 2:
            self._retire_one()

    def _memcpy_strided_out(self, source, target: np.ndarray) -> None:
        """Device-to-host copy into a strided host band, as a 2-D copy."""
        import cupy as cp

        itemsize = int(target.dtype.itemsize)
        if target.ndim < 2 or target.strides[-1] != itemsize:
            raise ValueError(
                f"the host tier cannot write {target.shape} with strides "
                f"{target.strides}: a parked band is written as a 2-D copy "
                "whose innermost axis is contiguous, and this one is not"
            )
        width = int(np.prod(target.shape[-2:])) * itemsize
        height = int(np.prod(target.shape[:-2])) if target.ndim > 2 else 1
        dpitch = int(target.strides[-3]) if target.ndim > 2 else width
        cp.cuda.runtime.memcpy2DAsync(
            target.ctypes.data, dpitch,
            int(source.data.ptr), width,
            width, height,
            cp.cuda.runtime.memcpyDeviceToHost,
            self._out_stream.ptr,
        )

    def _retire_one(self) -> None:
        event, source, _target = self._out_pending.pop(0)
        event.synchronize()
        del source

    def drain(self) -> None:
        """Retire every outstanding write-back, releasing its device buffer."""
        while self._out_pending:
            self._retire_one()

    def step_boundary(self) -> None:
        """Close the step: retire the writes and read the transfer clock.

        The tier's own streams are timed with CUDA events around each
        copy, so ``transfer_in_s`` and ``transfer_out_s`` are the time the
        LINK spent, not a rate multiplied by a byte count.  The receipt
        divides them by the step time, which is gate OVERLAP-1: a spill
        configuration whose transfers are not hidden reports it rather
        than running a campaign fifteen percent slow.
        """
        if not self.enabled:
            return
        self.steps += 1
        self.drain()
        if not self._cupy or not self._timed:
            return
        import cupy as cp

        for direction, start, end in self._timed:
            try:
                end.synchronize()
                elapsed = float(cp.cuda.get_elapsed_time(start, end)) / 1000.0
            except Exception:  # noqa: BLE001 - a dropped event is not a run
                continue
            if direction == "in":
                self.transfer_in_s += elapsed
            elif direction == "out":
                self.transfer_out_s += elapsed
            else:
                self.exposed_s += elapsed
        self.transfer_s = self.transfer_in_s + self.transfer_out_s
        self._timed.clear()

    def synchronize(self) -> None:
        """Every outstanding transfer complete.

        Called before a checkpoint reads the tier and at the end of the
        run: a host copy that is still in flight is not a checkpoint.
        """
        for entry in list(self._pending.values()):
            if entry[2] is not None:
                entry[2].synchronize()
        self._pending.clear()
        self.drain()
        if self._cupy:
            if self._in_stream is not None:
                self._in_stream.synchronize()
            if self._out_stream is not None:
                self._out_stream.synchronize()

    def close(self) -> None:
        self.synchronize()
        self.parked.clear()

    # -- the receipt ---------------------------------------------------

    def receipt(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "pinned": bool(self._cupy),
            "slices": list(self.slices),
            "parked_arrays": int(len(self.parked)),
            "parked_gib": round(self.parked_bytes / GIB, 4),
            "staged_gib": round(self.staged_bytes / GIB, 4),
            "stored_gib": round(self.stored_bytes / GIB, 4),
            "stage_calls": int(self.stage_calls),
            "store_calls": int(self.store_calls),
            "prefetch_hits": int(self.prefetch_hits),
            "prefetch_misses": int(self.prefetch_misses),
            "steps": int(self.steps),
            "transfer_in_s": round(self.transfer_in_s, 6),
            "transfer_out_s": round(self.transfer_out_s, 6),
            "transfer_s": round(self.transfer_s, 6),
            "exposed_s": round(self.exposed_s, 6),
            "exposed_s_per_step": round(
                self.exposed_s / self.steps, 6) if self.steps else 0.0,
            # One minus what the step paid over what the link spent
            # inbound: gate OVERLAP-1's own number.  Negative would mean
            # the step waited longer than the link was busy, which is a
            # queue behind another transfer rather than a hidden one.
            "achieved_overlap": round(
                1.0 - self.exposed_s / self.transfer_in_s, 4
            ) if self.transfer_in_s > 0 else None,
            "transfer_s_per_step": round(
                self.transfer_s / self.steps, 6) if self.steps else 0.0,
            "staged_gib_per_step": round(
                self.staged_bytes / GIB / self.steps, 4) if self.steps else 0.0,
            "stored_gib_per_step": round(
                self.stored_bytes / GIB / self.steps, 4) if self.steps else 0.0,
        }


__all__ = [
    "GIB", "SLICES", "SLICE_READS_PER_STEP", "HostTier", "SpilledArray",
    "pinned_host_array", "prefetch", "resident", "spilled",
]
