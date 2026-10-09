"""Metal views over MYNN's declared, state-owned scratch slots.

Layer and face leaves are exposed in physical level-major order
``(..., nz, ncol)``. ``MetalArray`` is contiguous-only, so unlike the CUDA
scratch holder these views do not pretend to have logical ``(ncol, nz)``
strides.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import prod

import numpy as np

from woof.core.mynn_pbl_scratch import (
    MYNN_PBL_CONSTANT_ZERO_SLOTS,
    MYNN_PBL_FLAG_SLOTS,
    mynn_pbl_flag_shapes,
    mynn_pbl_index_shapes,
    mynn_pbl_scratch_shapes,
)
from woof.metal.array import MetalArray


class MynnPblMetalScratch:
    """One eagerly allocated MYNN workspace owned by a ``DomainState``.

    Constant-zero feeds are obtained from the same named state slots and rely
    on state scratch's zero initialization; ordinary work slots retain the
    kernels' existing write-before-read obligation.
    """

    constant_zero_slots = MYNN_PBL_CONSTANT_ZERO_SLOTS

    def __init__(self, state, chunk: int, nz: int):
        self.chunk = int(chunk)
        self.nz = int(nz)
        if self.chunk < 1 or self.nz < 1:
            raise ValueError("MYNN scratch needs chunk >= 1 and nz >= 1")
        if not getattr(state, "_metal_setup_state", False):
            raise TypeError("MYNN Metal scratch requires a Metal-backed DomainState")

        # Calling all declared slots through state.scratch preserves named
        # ownership and whatever accounting/arena policy the state provides.
        self._buffers: dict[str, MetalArray] = {}
        for slot, shape in mynn_pbl_scratch_shapes(self.chunk, self.nz).items():
            self._buffers[slot] = self._state_scratch(
                state, slot, shape, np.float32)
        for slot, shape in mynn_pbl_index_shapes(self.chunk, self.nz).items():
            self._buffers[slot] = self._state_scratch(
                state, slot, shape, np.int32)
        for slot, shape in mynn_pbl_flag_shapes().items():
            self._buffers[slot] = self._state_scratch(
                state, slot, shape, np.int32)

    @staticmethod
    def _state_scratch(state, slot: str, shape, dtype) -> MetalArray:
        array = state.scratch(shape, slot, dtype=dtype)
        if not isinstance(array, MetalArray):
            raise TypeError("MYNN Metal scratch requires a Metal-backed DomainState")
        if array.shape != tuple(shape) or array.dtype != np.dtype(dtype):
            raise ValueError(f"MYNN scratch slot {slot!r} has incompatible storage")
        return array

    def _backing(self, slot: str, values: int, dtype) -> MetalArray:
        buf = self._buffers[slot]
        dtype = np.dtype(dtype)
        if buf.dtype != dtype:
            raise TypeError(f"MYNN scratch slot {slot!r} is {buf.dtype}, requested {dtype}")
        if buf.size < values:
            raise ValueError(
                f"MYNN scratch slot {slot!r} holds {buf.size} values, needs {values}")
        return buf

    @staticmethod
    def _physical_shape(shape):
        shape = tuple(int(extent) for extent in shape)
        if any(extent < 0 for extent in shape):
            raise ValueError("MYNN view extents cannot be negative")
        if len(shape) >= 2:
            return (*shape[:-2], shape[-1], shape[-2])
        return shape

    @staticmethod
    def _subview(buf: MetalArray, offset_elements: int, shape,
                 *, level_major: bool) -> MetalArray:
        shape = tuple(int(extent) for extent in shape)
        physical_shape = (MynnPblMetalScratch._physical_shape(shape)
                          if level_major else shape)
        byte_offset = buf.byte_offset + offset_elements * buf.dtype.itemsize
        return MetalArray(buf.buffer, physical_shape, buf.dtype,
                          byte_offset=byte_offset)

    def group(self, slot: str, names: Iterable[str], shape) -> dict[str, MetalArray]:
        """Return level-major contiguous leaves from a packed float slot."""
        names = tuple(names)
        shape = tuple(int(extent) for extent in shape)
        unit = prod(shape)
        buf = self._backing(slot, len(names) * unit, np.float32)
        return {
            name: self._subview(buf, i * unit, shape, level_major=True)
            for i, name in enumerate(names)
        }

    def one(self, slot: str, shape) -> MetalArray:
        """Return the first level-major leaf from a packed float slot."""
        shape = tuple(int(extent) for extent in shape)
        unit = prod(shape)
        buf = self._backing(slot, unit, np.float32)
        return self._subview(buf, 0, shape, level_major=True)

    def index(self, slot: str, shape) -> MetalArray:
        """Return a contiguous int32 index view (normally one value per column)."""
        shape = tuple(int(extent) for extent in shape)
        unit = prod(shape)
        buf = self._backing(slot, unit, np.int32)
        return self._subview(buf, 0, shape, level_major=False)

    def flags(self, slot: str = MYNN_PBL_FLAG_SLOTS[0]) -> MetalArray:
        """Clear and return persistent validation words, as the CUDA holder does."""
        buf = self._backing(slot, mynn_pbl_flag_shapes()[slot][0], np.int32)
        buf.fill(0)
        return buf


__all__ = ["MynnPblMetalScratch"]
