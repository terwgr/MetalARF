"""WOOF's small contiguous Metal storage substrate.

This deliberately implements contiguous storage and transfers for selected
WOOF dtypes. It is not a general CuPy replacement; unimplemented indexing and
arithmetic fail loudly. FP64 and complex128 are storage only. Complex64 has
four explicit kernels elsewhere, but no general array operators or NumPy
exceptional-complex semantics. Accepting a host dtype does not imply complete
or verified numerical behavior.
"""

from __future__ import annotations

import math
import numpy as np

from .runtime import MetalBuffer, MetalContext


class _Flags:
    __slots__ = ("c_contiguous",)

    def __init__(self):
        self.c_contiguous = True


class MetalArray:
    """A shaped contiguous view over an owned Metal buffer allocation."""

    __slots__ = ("buffer", "shape", "dtype", "byte_offset", "strides",
                 "size", "nbytes", "ndim", "flags")

    def __init__(self, buffer: MetalBuffer, shape, dtype=np.float32,
                 *, byte_offset: int = 0):
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.complex64),
                         np.dtype(np.int32),
                         np.dtype(np.uint32), np.dtype(np.float64),
                         np.dtype(np.int64), np.dtype(np.uint64),
                         np.dtype(np.complex128)):
            raise TypeError(f"Metal slice does not support dtype {dtype}")
        shape = tuple(int(extent) for extent in shape)
        if any(extent < 0 for extent in shape):
            raise ValueError("Metal array extents cannot be negative")
        size = math.prod(shape)
        nbytes = size * dtype.itemsize
        alignment = min(dtype.alignment, 8)
        if (byte_offset < 0 or byte_offset % alignment
                or byte_offset + nbytes > buffer.size):
            raise ValueError("Metal array view is unaligned or outside its buffer")
        self.buffer = buffer
        self.shape = shape
        self.dtype = dtype
        self.byte_offset = byte_offset
        stride = dtype.itemsize
        strides = []
        for extent in reversed(shape):
            strides.append(stride)
            stride *= max(extent, 1)
        self.strides = tuple(reversed(strides))
        self.size = size
        self.nbytes = nbytes
        self.ndim = len(shape)
        self.flags = _Flags()

    def view(self, shape, dtype=None, *, byte_offset=None) -> MetalArray:
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        offset = self.byte_offset if byte_offset is None else byte_offset
        return MetalArray(self.buffer, shape, dtype, byte_offset=offset)

    def view_uint64_bits(self) -> MetalArray:
        """View FP64 words as uint64 without changing any stored bits.

        This is for software-defined binary64 arithmetic that uses ulong
        resources. It performs no numeric conversion and adds no allocation:
        the returned view has the same shape, byte offset, Metal context, and
        owning buffer as this array. The constructor enforces uint64 alignment
        and buffer bounds. Only float64 arrays may use this convenience view.
        """
        if self.dtype != np.dtype(np.float64):
            raise TypeError("uint64 bit views require float64 source storage")
        return self.view(self.shape, np.uint64)

    def copy(self) -> MetalArray:
        """Return an independent device copy without staging through the host."""
        context = self.buffer.context
        target = MetalArray(
            context.buffer(max(self.nbytes, self.dtype.itemsize)),
            self.shape,
            self.dtype,
        )
        if self.size:
            context.submit_bookkeeping(((self, target),))
        return target

    def fill(self, value) -> None:
        scalar = np.asarray(value, dtype=self.dtype)
        if scalar.ndim != 0:
            raise ValueError("Metal array fill requires a scalar value")
        if self.size == 0:
            return
        pattern = np.frombuffer(scalar.tobytes(), dtype=np.uint32)
        self.buffer.context.submit_fill_pattern(
            self.buffer, self.nbytes // 4, pattern,
            target_offset=self.byte_offset)

    def __setitem__(self, key, value) -> None:
        if key is not Ellipsis and key != slice(None):
            raise NotImplementedError(
                "the first Metal slice supports only whole-array assignment")
        if isinstance(value, MetalArray):
            if value.shape != self.shape or value.dtype != self.dtype:
                raise ValueError("Metal copy requires equal shape and dtype")
            if self.size == 0:
                return
            self.buffer.context.submit_bookkeeping(((value, self),))
        else:
            host = np.asarray(value, dtype=self.dtype)
            if host.ndim == 0:
                self.fill(host)
                return
            if host.shape != self.shape:
                raise ValueError(
                    f"Metal host upload shape {host.shape} != {self.shape}")
            host = np.ascontiguousarray(host)
            self.buffer.write(host.tobytes(), offset=self.byte_offset)

    def __getitem__(self, key) -> MetalArray:
        """Return a contiguous view selected along the leading axis.

        Global spectral stacks are stored as contiguous fields.  The SL
        forecast slices those fields and their level slabs without changing
        their layout; exposing those views keeps the stack on Metal and does
        not add array arithmetic or implicit host conversion.
        """
        if self.ndim == 0:
            raise IndexError("a scalar MetalArray has no leading axis")
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.ndim:
            raise IndexError("too many indices for MetalArray")
        first = key[0]
        trailing = key[1:]
        if any(not (isinstance(item, slice)
                    and item.start is None and item.stop is None
                    and item.step is None) for item in trailing):
            raise NotImplementedError(
                "MetalArray views support indexing only the leading axis"
            )
        stride = self.strides[0]
        if isinstance(first, (int, np.integer)):
            index = int(first)
            if index < 0:
                index += self.shape[0]
            if index < 0 or index >= self.shape[0]:
                raise IndexError("MetalArray leading index is out of range")
            return self.view(
                self.shape[1:], dtype=self.dtype,
                byte_offset=self.byte_offset + index * stride,
            )
        if not isinstance(first, slice):
            raise TypeError("MetalArray leading index must be an int or slice")
        start, stop, step = first.indices(self.shape[0])
        if step != 1:
            raise NotImplementedError("MetalArray leading slices require step=1")
        return self.view(
            (max(0, stop - start), *self.shape[1:]), dtype=self.dtype,
            byte_offset=self.byte_offset + start * stride,
        )

    def to_numpy(self) -> np.ndarray:
        raw = self.buffer.read(self.nbytes, offset=self.byte_offset)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape).copy()

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "implicit Metal-to-host conversion is disabled; call to_numpy()")


class MetalArrayModule:
    """Allocation module passed to ``DomainState`` for Metal slice setup."""

    __woof_metal_array_module__ = True
    float32 = np.float32
    float64 = np.float64
    complex64 = np.complex64
    complex128 = np.complex128
    int32 = np.int32
    int64 = np.int64
    uint32 = np.uint32
    uint64 = np.uint64

    def __init__(self, context: MetalContext | None = None):
        self.context = context or MetalContext()
        self.fft = MetalFFT(self)

    @staticmethod
    def _shape(shape) -> tuple[int, ...]:
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        result = tuple(int(extent) for extent in shape)
        if any(extent < 0 for extent in result):
            raise ValueError("Metal allocation extents cannot be negative")
        return result

    def _allocate(self, shape, dtype, pattern=None) -> MetalArray:
        shape = self._shape(shape)
        dtype = np.dtype(dtype)
        nbytes = math.prod(shape) * dtype.itemsize
        # Metal rejects zero-byte MTLBuffers, but NumPy/CuPy callers can
        # legitimately request empty logical arrays.
        array = MetalArray(self.context.buffer(max(nbytes, dtype.itemsize)),
                           shape, dtype)
        if array.size and pattern is not None:
            self.context.submit_fill_pattern(
                array.buffer, array.nbytes // 4, pattern)
        return array

    def asarray(self, value, dtype=None, order=None, *, copy=None) -> MetalArray:
        """Upload host data or return a compatible Metal array.

        This is intentionally a transfer operation, not a CuPy-compatible
        lazy conversion.  Host inputs are copied into owned contiguous
        storage and the write completes before this method returns.
        """
        if isinstance(value, MetalArray):
            target_dtype = value.dtype if dtype is None else np.dtype(dtype)
            if target_dtype != value.dtype:
                raise NotImplementedError(
                    "MetalArray dtype conversion is not implemented")
            if value.buffer.context is not self.context:
                raise ValueError("MetalArray belongs to a different context")
            if copy is True:
                result = self.zeros(value.shape, value.dtype)
                result.buffer.context.submit_bookkeeping(((value, result),))
                return result
            return value
        host = np.asarray(value, dtype=dtype)
        if host.dtype not in (np.dtype(np.float32), np.dtype(np.complex64),
                              np.dtype(np.int32),
                              np.dtype(np.uint32), np.dtype(np.float64),
                              np.dtype(np.int64), np.dtype(np.uint64),
                              np.dtype(np.complex128)):
            raise TypeError(f"Metal slice does not support dtype {host.dtype}")
        if order not in (None, "C", "A", "K"):
            raise ValueError("MetalArray supports only C-contiguous storage")
        host = np.ascontiguousarray(host)
        result = self.zeros(host.shape, host.dtype)
        if host.size:
            result.buffer.write(host.tobytes(), offset=result.byte_offset)
        return result

    def zeros(self, shape, dtype=np.float32) -> MetalArray:
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.complex64),
                         np.dtype(np.int32),
                         np.dtype(np.uint32), np.dtype(np.float64),
                         np.dtype(np.int64), np.dtype(np.uint64),
                         np.dtype(np.complex128)):
            raise TypeError(f"Metal slice does not support dtype {dtype}")
        return self._allocate(shape, dtype, (0,))

    def empty(self, shape, dtype=np.float32) -> MetalArray:
        """Allocate uninitialized contiguous storage for all-write kernels.

        Callers must ensure every element is written before any read. This
        avoids a full-buffer fill for large producer outputs and bounded
        scratch whose kernels establish their own initialization order.
        """
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.complex64),
                         np.dtype(np.int32), np.dtype(np.uint32),
                         np.dtype(np.float64), np.dtype(np.int64),
                         np.dtype(np.uint64), np.dtype(np.complex128)):
            raise TypeError(f"Metal array does not support dtype {dtype}")
        return self._allocate(shape, dtype)

    def ones(self, shape, dtype=np.float32) -> MetalArray:
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.complex64),
                         np.dtype(np.int32),
                         np.dtype(np.uint32), np.dtype(np.float64),
                         np.dtype(np.int64), np.dtype(np.uint64),
                         np.dtype(np.complex128)):
            raise TypeError(f"Metal slice does not support dtype {dtype}")
        pattern = np.frombuffer(np.asarray(1, dtype=dtype).tobytes(),
                                dtype=np.uint32)
        return self._allocate(shape, dtype, pattern)


class MetalFFT:
    """The first native FFT slice: contiguous FP32 rfft2 and Complex64 irfft2."""

    def __init__(self, arrays: MetalArrayModule):
        self._arrays = arrays

    @staticmethod
    def _axes(axes, ndim: int) -> None:
        if axes is None:
            return
        axes = tuple(axes)
        allowed = {(-2, -1), (ndim - 2, ndim - 1)}
        if axes not in allowed:
            raise NotImplementedError("Metal FFT currently transforms the last two axes")

    @staticmethod
    def _shape(s, default):
        if s is None:
            return tuple(default)
        if isinstance(s, (int, np.integer)):
            raise ValueError("Metal rfft2/irfft2 shape must contain two extents")
        shape = tuple(int(extent) for extent in s)
        if len(shape) != 2 or any(extent <= 0 for extent in shape):
            raise ValueError("Metal rfft2/irfft2 shape must contain two positive extents")
        return shape

    @staticmethod
    def _check_norm(norm) -> None:
        if norm not in (None, "backward"):
            raise NotImplementedError(
                "Metal FFT currently supports NumPy's backward normalization only")

    def _output(self, shape, dtype) -> MetalArray:
        dtype = np.dtype(dtype)
        nbytes = math.prod(shape) * dtype.itemsize
        return MetalArray(self._arrays.context.buffer(max(nbytes, dtype.itemsize)),
                          shape, dtype)

    def rfft2(self, a: MetalArray, s=None, axes=None, norm=None) -> MetalArray:
        if not isinstance(a, MetalArray):
            raise TypeError("Metal rfft2 requires a MetalArray input")
        if a.dtype != np.dtype(np.float32):
            raise TypeError("Metal rfft2 currently requires float32 input")
        if not 2 <= a.ndim <= 16:
            raise ValueError("Metal rfft2 requires rank between two and sixteen")
        if any(extent <= 0 for extent in a.shape):
            raise ValueError("Metal rfft2 does not support empty dimensions")
        self._axes(axes, a.ndim)
        self._check_norm(norm)
        requested = self._shape(s, a.shape[-2:])
        if requested != a.shape[-2:]:
            raise NotImplementedError(
                "Metal rfft2 does not resize or pad input dimensions")
        output_shape = a.shape[:-1] + (a.shape[-1] // 2 + 1,)
        result = self._output(output_shape, np.complex64)
        self._arrays.context.submit_fft2(a, result, inverse=False)
        return result

    def irfft2(self, a: MetalArray, s=None, axes=None, norm=None) -> MetalArray:
        if not isinstance(a, MetalArray):
            raise TypeError("Metal irfft2 requires a MetalArray input")
        if a.dtype != np.dtype(np.complex64):
            raise TypeError("Metal irfft2 currently requires complex64 input")
        if not 2 <= a.ndim <= 16:
            raise ValueError("Metal irfft2 requires rank between two and sixteen")
        if any(extent <= 0 for extent in a.shape):
            raise ValueError("Metal irfft2 does not support empty dimensions")
        self._axes(axes, a.ndim)
        self._check_norm(norm)
        default_width = 2 * (a.shape[-1] - 1)
        if s is None and default_width <= 0:
            raise ValueError("irfft2 needs an explicit s when the half-spectrum width is one")
        output_shape = self._shape(s, (a.shape[-2], default_width))
        if output_shape[0] != a.shape[-2]:
            raise NotImplementedError(
                "Metal irfft2 does not resize or pad input rows")
        if a.shape[-1] != output_shape[-1] // 2 + 1:
            raise ValueError("irfft2 input width must equal nx//2+1")
        target_shape = a.shape[:-2] + output_shape
        result = self._output(target_shape, np.float32)
        self._arrays.context.submit_fft2(a, result, inverse=True)
        return result
