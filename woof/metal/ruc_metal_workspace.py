"""RUC's packed Metal workspace and zero-dispatch field transfers.

The fused CUDA driver passes pointer tables for more than eighty state fields
and 815 generated SFCTMP aliases. Metal kernels bind the latter through a
single typed word slab. Public field arrays are copied into and out of that
slab with the runtime's existing Metal blit path; no device pointer values
are exposed to shader code.
"""

from __future__ import annotations

import math

import numpy as np

from .array import MetalArray
from .ruc_metal_layout import (
    ALIVE_BASE_UNITS,
    INTEGER_BASE_UNITS,
    INTEGER_ROWS,
    RUN_BASE_UNITS,
    TABLE_BASE_UNITS,
    driver_field_row,
)


_I32_MAX = np.iinfo(np.int32).max
_FLOAT = np.dtype(np.float32)
_INT = np.dtype(np.int32)
_U32 = np.dtype(np.uint32)


def ruc_workspace_bytes(n: int, *, table_words: int = 0) -> int:
    """The packed slab, pending rain plane and status allocation."""
    if type(n) is not int or not 0 <= n <= _I32_MAX:
        raise ValueError("RUC Metal workspace extent must fit int32")
    if type(table_words) is not int or table_words < 0:
        raise ValueError("RUC Metal table extent must be a nonnegative int")
    words = (TABLE_BASE_UNITS + 1) * max(n, 1) + table_words
    return max(4, words * 4) + 35 * 8


class RucMetalWorkspace:
    """The packed RUC workspace retained by one forecast's parameter owner."""

    def __init__(self, context, n: int, *, table_words: int = 0):
        if type(n) is not int or not 0 <= n <= _I32_MAX:
            raise ValueError("RUC Metal workspace extent must fit int32")
        if type(table_words) is not int or table_words < 0:
            raise ValueError("RUC Metal table extent must be a nonnegative int")
        self.context = context
        self.n = n
        # Field rows scale with the column count. The six parameter tables
        # occupy a separate tail whose offsets are supplied by the caller.
        self.table_base = TABLE_BASE_UNITS * max(n, 1)
        self.table_words = table_words
        self.pending_base = self.table_base + table_words
        words = self.pending_base + max(n, 1)
        self.buffer = context.buffer(ruc_workspace_bytes(n, table_words=table_words) - 35 * 8)
        self.raw = MetalArray(self.buffer, (words,), _U32)
        self.pending_rainbl = self._view(self.pending_base, n, _FLOAT)
        flag_buffer = context.buffer(35 * 8)
        self.flag_slab = MetalArray(flag_buffer, (35,), np.uint64)
        self.driver_flags = self.flag_slab.view((20,), np.uint64)
        self.driver_flag_words = self.flag_slab.view((40,), np.uint32)
        self.sf_flags = self.flag_slab.view((15,), np.uint64, byte_offset=20 * 8)
        self.flag_slab.fill(0)

    def _view(self, word_offset: int, count: int, dtype) -> MetalArray:
        return MetalArray(self.buffer, (count,), dtype,
                          byte_offset=word_offset * np.dtype(dtype).itemsize)

    def field(self, name: str, *, levels: int = 1) -> MetalArray:
        """Get the packed float32 field view in column or layer-major form."""
        row, profiled = driver_field_row(name)
        expected_levels = 9 if profiled else 1
        if levels != expected_levels:
            raise ValueError(
                f"RUC field {name} expects {expected_levels} level(s)")
        count = self.n * levels
        return MetalArray(self.buffer, (count,), _FLOAT,
                          byte_offset=row * self.n * _FLOAT.itemsize)

    def field_view(self, name: str, shape, *, dtype=_FLOAT) -> MetalArray:
        """Get a shape-preserving typed view for a packed driver row."""
        row, profiled = driver_field_row(name)
        shape = tuple(int(value) for value in shape)
        expected = (9, self.n) if profiled else (self.n,)
        if math.prod(shape) != math.prod(expected):
            raise ValueError(f"RUC packed field {name} shape does not match {expected}")
        if np.dtype(dtype) != _FLOAT:
            raise TypeError("RUC driver numeric rows use float32 storage")
        return MetalArray(self.buffer, shape, dtype,
                          byte_offset=row * self.n * _FLOAT.itemsize)

    def integer(self, row: int) -> MetalArray:
        if type(row) is not int or not 0 <= row < INTEGER_ROWS:
            raise ValueError("RUC integer workspace row must be in 0..4")
        return self._view((INTEGER_BASE_UNITS + row) * max(self.n, 1),
                          self.n, _INT)

    @property
    def run(self) -> MetalArray:
        return self._view(RUN_BASE_UNITS * max(self.n, 1), self.n, _U32)

    @property
    def alive(self) -> MetalArray:
        return self._view(ALIVE_BASE_UNITS * max(self.n, 1), self.n, _U32)

    def table(self, offset: int, count: int, *, dtype=_FLOAT) -> MetalArray:
        """Return a typed view into the table tail of the workspace."""
        if type(offset) is not int or type(count) is not int or offset < 0 or count < 0:
            raise ValueError("RUC table offsets and extents must be nonnegative ints")
        if offset + count > self.table_words:
            raise ValueError("RUC table view exceeds the allocated workspace tail")
        dtype = np.dtype(dtype)
        if dtype not in (_FLOAT, _INT, _U32):
            raise TypeError("RUC Metal tables support only float32/int32/uint32")
        return self._view(self.table_base + offset, count, dtype)

    def upload_tables(self, tables) -> dict[str, int]:
        """Copy ordered one-dimensional parameter tables into the tail slab."""
        if not isinstance(tables, dict):
            raise TypeError("RUC Metal tables must be a name-to-array mapping")
        offsets = {}
        cursor = 0
        transfers = []
        uploads = []
        for name, values in tables.items():
            if isinstance(values, MetalArray):
                source = values
                if source.ndim != 1 or not source.flags.c_contiguous:
                    raise ValueError(f"RUC Metal table {name} must be flat and contiguous")
                dtype, count = source.dtype, source.size
            else:
                source = np.ascontiguousarray(values)
                if source.ndim != 1 or source.dtype not in (_FLOAT, _INT, _U32):
                    raise TypeError(f"RUC Metal table {name} must be flat float32/int32/uint32")
                dtype, count = source.dtype, source.size
            target = self.table(cursor, count, dtype=dtype)
            offsets[name] = cursor
            cursor += count
            if isinstance(source, MetalArray):
                transfers.append((source, target))
            else:
                uploads.append((target, source))
        if cursor > self.table_words:
            raise ValueError("RUC Metal table data exceeds the workspace tail")
        self.transfer(transfers)
        for target, source in uploads:
            target[...] = source
        return offsets

    def transfer(self, pairs) -> None:
        """Blit same-typed contiguous source/destination field views."""
        pairs = tuple(pairs)
        for source, destination in pairs:
            if not isinstance(source, MetalArray) or not isinstance(destination, MetalArray):
                raise TypeError("RUC Metal workspace transfers require MetalArray values")
            if source.dtype != destination.dtype or source.size != destination.size:
                raise ValueError("RUC Metal workspace transfer shapes or dtypes differ")
            if (source.buffer.context is not self.context
                    or destination.buffer.context is not self.context):
                raise ValueError("RUC Metal workspace transfer context differs")
        # The native bookkeeping ABI carries at most 64 independent slices.
        # It copies buffers, not elements, so one-column and profile views use
        # the same path and keep the kernel resource ABI bounded.
        for start in range(0, len(pairs), 64):
            self.context.submit_bookkeeping(pairs[start:start + 64])

    def pack_fields(self, fields, *, categories=("ivgtyp", "isltyp")) -> None:
        """Copy named public values to their packed driver rows.

        `fields` is keyed by the exact LSMRUC argument name. The two WRF
        category arrays are stored in integer rows 0 and 1; all other values
        map to the source-generated `d_<name>` field row.
        """
        pairs = []
        for name, source in fields.items():
            if name in categories:
                row = tuple(categories).index(name)
                target = self.integer(row)
            else:
                source_shape = tuple(source.shape)
                row, profiled = driver_field_row(name)
                expected = (9, self.n) if profiled else (self.n,)
                if math.prod(source_shape) != math.prod(expected):
                    raise ValueError(
                        f"RUC Metal {name} shape {source_shape} does not match {expected}")
                target = self.field_view(name, source_shape)
            pairs.append((source, target))
        self.transfer(pairs)

    def unpack_fields(self, fields, *, categories=("ivgtyp", "isltyp")) -> None:
        """Copy packed driver rows back to their caller-owned arrays."""
        pairs = []
        for name, target in fields.items():
            if name in categories:
                source = self.integer(tuple(categories).index(name))
            else:
                row, profiled = driver_field_row(name)
                shape = (9, self.n) if profiled else (self.n,)
                source = self.field_view(name, shape)
            pairs.append((source, target))
        self.transfer(pairs)


__all__ = ["RucMetalWorkspace"]
