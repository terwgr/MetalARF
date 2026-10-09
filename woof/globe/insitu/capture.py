"""Per-component physics tendency capture: device reductions only.

At the seam where a physics suite runs its components in sequence (the
reference suite's process functions in ``ReferencePhysics.step``, the
native runtime's scheme calls in ``NativePhysicsRuntime.run``) each
component marks the live prognostic fields; the capture records, per
component per call, the global-mean and max-abs change of theta, qv, u
and v since the previous mark.  The results stay on the device as one
small vector per mark and cross to the host with the ledger's batched
flush.

Marks copy the four fields: the reference suite's convective adjustment
and every native kernel update their arrays IN PLACE, so a capture that
held references to "before" would read exactly zero change and report a
component as inert - the silent-instrument failure the 2026-08-31 night
taught (five instruments gave confident wrong answers).  The copy is the
price of a measurement that cannot lie that way.

"Global mean" here is the area-weighted mean over the grid averaged over
levels with equal weight (no dp weighting: the marks carry no dp on the
reference path and the native batch is bottom-to-top float32; level
ordering does not matter to either reduction).

THE PHYSICS RUNS A LATITUDE BAND AT A TIME (dynamics.apply_physics), so a
mark arrives with ``band=(first, last)`` and carries that band's rows of
the four fields.  The record of a call is still ONE vector per component:
the per-row stage of each mean (the zonal mean, then the level mean, both
row-local) is written into a resident ``(nlat,)`` buffer as the bands
arrive, and the area-weighted sum over latitude runs once when the suite
closes the call (:meth:`ComponentCapture.close_call`), in grid order --
the same operand the whole-grid mark reduces, so the number does not
depend on the band count.  The max-abs stage is exactly associative and
folds as it goes.  A whole-grid mark (``band=None``, a hand-driven suite)
records as it always did, immediately.
"""
from __future__ import annotations

import time

import numpy as np

CAPTURE_FIELDS = ("theta", "qv", "u", "v")
CAPTURE_STATS = ("mean_change", "max_abs_change")
CAPTURE_NAMES = tuple(
    f"{field}_{stat}" for field in CAPTURE_FIELDS for stat in CAPTURE_STATS
)
START_MARK = "start"


class ComponentCapture:
    def __init__(self, quadrature_weights):
        self._weights_host = np.asarray(quadrature_weights, dtype=np.float64)
        self._weights_by_module: dict[int, object] = {}
        self._metal_weights_by_context: dict[int, object] = {}
        self._nlat = int(self._weights_host.shape[0])
        self.active = True
        self.records: list[tuple[int, str, object]] = []
        self._previous: dict[str, object] | None = None
        self._previous_band: tuple[int, int] | None = None
        self._call = -1
        # The open banded call's per-component partials, in the order the
        # components first marked: name -> [rows covered, {field: (nlat,)
        # meridional buffer}, {field: max-abs so far}].
        self._pending: dict[str, list] = {}
        self._pending_order: list[str] = []
        self._open = False
        # Host wall time spent inside marks (on cupy: launch time only).
        self.wall_s = 0.0

    def begin_step(self) -> None:
        """Reset the call counter so marks inside the next step number
        the physics calls from zero (0 = first Strang half, 1 = second)."""
        self._call = -1
        self._previous = None
        self._previous_band = None
        self._pending = {}
        self._pending_order = []
        self._open = False

    def _weights(self, xp):
        key = id(xp)
        weights = self._weights_by_module.get(key)
        if weights is None:
            weights = xp.asarray(self._weights_host, dtype=xp.float64)
            self._weights_by_module[key] = weights
        return weights

    def _rows(self, band, fields) -> tuple[int, int]:
        rows = int(fields[CAPTURE_FIELDS[0]].shape[-2])
        if band is None:
            if rows != self._nlat:
                raise ValueError(
                    f"a whole-grid component mark carries {rows} rows against a "
                    f"grid of {self._nlat}; a band mark names its rows with band="
                )
            return (0, self._nlat)
        first, last = int(band[0]), int(band[1])
        if first < 0 or last > self._nlat or last - first != rows:
            raise ValueError(
                f"component mark band {band!r} does not cover the {rows} rows "
                f"of the fields it arrived with (grid of {self._nlat} rows)"
            )
        return first, last

    def _vector(self, xp, buffers, maxima):
        weights = self._weights(xp)
        stats = []
        for key in CAPTURE_FIELDS:
            stats.append(0.5 * xp.sum(buffers[key] * weights))
            stats.append(maxima[key])
        return xp.stack(stats).astype(xp.float64)

    def _metal_pack(self, fields, context):
        """Pack the four marked FP64 fields with device-to-device copies."""
        from woof.metal.array import MetalArrayModule

        arrays = [fields[key] for key in CAPTURE_FIELDS]
        shape = tuple(arrays[0].shape)
        if (len(shape) != 3 or any(tuple(array.shape) != shape
                                   for array in arrays)
                or any(array.dtype != np.dtype(np.float64)
                       or array.buffer.context is not context
                       for array in arrays)):
            raise TypeError(
                "resident in-situ capture requires four same-shape FP64 "
                "fields on one Metal context"
            )
        count = int(arrays[0].size)
        packed = MetalArrayModule(context).empty(
            (len(CAPTURE_FIELDS) * count,), dtype=np.float64
        )
        pairs = [
            (array.view((count,), dtype=np.float64), packed.view(
                (count,), dtype=np.float64,
                byte_offset=index * count * np.dtype(np.float64).itemsize,
            ))
            for index, array in enumerate(arrays)
        ]
        context.submit_bookkeeping(pairs)
        return packed, shape

    def _metal_statistics(self, current, previous, shape, row_start):
        """Return the eight component-change statistics on Metal."""
        from woof.metal.array import MetalArrayModule
        from woof.metal.generated import kernel_specs

        nlevel, nlat, nlon = map(int, shape)
        count = int(current.size // len(CAPTURE_FIELDS))
        chunk_size = 256
        partial_count = max(1, (count + chunk_size - 1) // chunk_size)
        context = current.buffer.context
        metal = MetalArrayModule(context)
        weights = self._metal_weights_by_context.get(id(context))
        if weights is None:
            weights = metal.asarray(np.ascontiguousarray(
                self._weights_host, dtype=np.float64
            ))
            self._metal_weights_by_context[id(context)] = weights
        partial_stats = metal.empty(
            (len(CAPTURE_FIELDS) * 2 * partial_count,), dtype=np.float64
        )
        context.submit_kernel(
            "global_insitu_capture_partial_fragment",
            {"current": current,
             "previous": previous,
             "quadrature_weights": weights,
             "partial_stats": partial_stats},
            {"nlevel": nlevel, "nlat": nlat, "nlon": nlon,
             "row_start": int(row_start), "chunk_size": chunk_size,
             "partial_count": partial_count}, exact_policy=False,
            grid_size=len(CAPTURE_FIELDS) * partial_count,
            threads_per_group=(
                kernel_specs.GLOBAL_INSITU_CAPTURE_PARTIAL_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        statistics = metal.empty((len(CAPTURE_FIELDS) * 2,), dtype=np.float64)
        context.submit_kernel(
            "global_insitu_capture_finalize_fragment",
            {"partial_stats": partial_stats,
             "statistics": statistics},
            {"partial_count": partial_count, "nlevel": nlevel,
             "nlon": nlon}, exact_policy=False,
            grid_size=len(CAPTURE_FIELDS),
            threads_per_group=(
                kernel_specs.GLOBAL_INSITU_CAPTURE_FINALIZE_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        return statistics

    @staticmethod
    def _metal_merge(accumulator, band_statistics):
        from woof.metal.generated import kernel_specs

        accumulator.buffer.context.submit_kernel(
            "global_insitu_capture_merge_fragment",
            {"accumulator": accumulator,
             "band_statistics": band_statistics},
            {"nfield": len(CAPTURE_FIELDS)}, exact_policy=False, grid_size=1,
            threads_per_group=(
                kernel_specs.GLOBAL_INSITU_CAPTURE_MERGE_FRAGMENT_THREADS_PER_GROUP
            ),
        )

    def _mark_metal(self, name, fields, band, first, last) -> None:
        from woof.metal.array import MetalArray

        started = time.perf_counter()
        context = fields[CAPTURE_FIELDS[0]].buffer.context
        if name == START_MARK:
            if band is None or not self._open:
                self._call += 1
                self._open = band is not None
            previous, shape = self._metal_pack(fields, context)
            self._previous = previous
            self._previous_band = (first, last)
            self._metal_shape = shape
            self.wall_s += time.perf_counter() - started
            return
        if (not isinstance(self._previous, MetalArray)
                or self._previous_band != (first, last)):
            raise ValueError(
                f"component mark {name!r} for rows {(first, last)} arrived "
                "before a matching resident 'start' mark"
            )
        current, shape = self._metal_pack(fields, context)
        if shape != self._metal_shape:
            raise ValueError("resident in-situ capture field shape changed within a band")
        statistics = self._metal_statistics(
            current, self._previous, shape, first
        )
        self._previous = current
        if band is None:
            self.records.append((self._call, name, statistics))
        else:
            entry = self._pending.get(name)
            if entry is None:
                self._pending[name] = [last - first, statistics]
                self._pending_order.append(name)
            else:
                entry[0] += last - first
                self._metal_merge(entry[1], statistics)
        self.wall_s += time.perf_counter() - started

    def mark(self, name: str, fields: dict[str, object], xp, *, band=None) -> None:
        """Record the change since the previous mark; ``start`` opens a
        call (with ``band``, one band of the open call)."""
        if not self.active:
            return
        started = time.perf_counter()
        first, last = self._rows(band, fields)
        from woof.metal.array import MetalArray

        resident = isinstance(fields[CAPTURE_FIELDS[0]], MetalArray)
        if resident:
            if not all(isinstance(fields[key], MetalArray)
                       for key in CAPTURE_FIELDS):
                raise TypeError("resident in-situ capture fields cannot mix backends")
            self._mark_metal(name, fields, band, first, last)
            return
        if name == START_MARK:
            if band is None or not self._open:
                # A new call: the whole grid's, or a banded call's first
                # band.  Later bands of the same call start without
                # advancing the counter.
                self._call += 1
                self._open = band is not None
            self._previous = {key: fields[key].copy() for key in CAPTURE_FIELDS}
            self._previous_band = (first, last)
            self.wall_s += time.perf_counter() - started
            return
        if self._previous is None or self._previous_band != (first, last):
            raise ValueError(
                f"component mark {name!r} for rows {(first, last)} arrived "
                f"before a {START_MARK!r} mark for those rows; the capture "
                "cannot difference against nothing"
            )
        if band is None:
            buffers = {key: xp.empty(self._nlat, dtype=xp.float64) for key in CAPTURE_FIELDS}
            maxima: dict[str, object] = {}
        else:
            entry = self._pending.get(name)
            if entry is None:
                entry = [
                    0,
                    {key: xp.empty(self._nlat, dtype=xp.float64) for key in CAPTURE_FIELDS},
                    {key: None for key in CAPTURE_FIELDS},
                ]
                self._pending[name] = entry
                self._pending_order.append(name)
            buffers, maxima = entry[1], entry[2]
        for key in CAPTURE_FIELDS:
            current = fields[key]
            delta = current - self._previous[key]
            zonal = xp.mean(delta, axis=-1, dtype=xp.float64)
            # (nlev, rows) -> level-equal mean -> these rows of the (nlat,)
            # buffer the area weighting reduces once, in grid order.
            buffers[key][first:last] = xp.mean(zonal, axis=0)
            band_max = xp.max(xp.abs(delta))
            held = maxima.get(key)
            maxima[key] = band_max if held is None else xp.maximum(held, band_max)
            self._previous[key] = current.copy()
        if band is None:
            self.records.append((self._call, name, self._vector(xp, buffers, maxima)))
        else:
            entry[0] += last - first
        self.wall_s += time.perf_counter() - started

    def close_call(self, xp) -> None:
        """The banded call's marks have all arrived: reduce every
        component's buffers once and record the vectors, in the order the
        components first marked.  A component whose marks did not cover
        the globe is refused by name rather than recorded on the rows it
        happened to see."""
        if not self.active or not self._open:
            return
        started = time.perf_counter()
        for name in self._pending_order:
            entry = self._pending[name]
            if len(entry) == 2:
                covered, statistics = entry
                if covered != self._nlat:
                    raise ValueError(
                        f"component {name!r} marked {covered} of {self._nlat} "
                        "latitude rows"
                    )
                self.records.append((self._call, name, statistics))
                continue
            covered, buffers, maxima = entry
            if covered != self._nlat:
                raise ValueError(
                    f"component {name!r} marked {covered} of {self._nlat} rows "
                    "in this physics call; a component that marks on some bands "
                    "and not others cannot be reported as the call's"
                )
            self.records.append((self._call, name, self._vector(xp, buffers, maxima)))
        self._pending = {}
        self._pending_order = []
        self._open = False
        self._previous = None
        self._previous_band = None
        self.wall_s += time.perf_counter() - started

    def drain(self) -> list[tuple[int, str, object]]:
        records = self.records
        self.records = []
        return records


def attach_capture(physics, capture: ComponentCapture) -> list[str]:
    """Install the capture on every physics object that declares an
    ``observer`` slot (the reference suite; the native column suite behind
    the bridge).  Returns the class names it attached to."""
    attached: list[str] = []
    targets = [physics]
    adapter = getattr(physics, "adapter", None)
    if adapter is not None:
        targets.append(adapter)
    for target in targets:
        if target is not None and hasattr(type(target), "observer"):
            target.observer = capture
            attached.append(type(target).__name__)
    return attached


__all__ = [
    "CAPTURE_FIELDS",
    "CAPTURE_NAMES",
    "CAPTURE_STATS",
    "ComponentCapture",
    "START_MARK",
    "attach_capture",
]
