"""The slice of CuPy that the SFIRE drivers use, over Metal storage.

Arrays are ``MetalArray`` buffers. Leading-axis views alias the buffer; every
other index and every reduction goes through the host, which shares memory on
Apple silicon. Kernels do all of the model arithmetic.
"""
from __future__ import annotations

import operator
import types

import numpy as np

from .array import MetalArray, MetalArrayModule

_state = {"module": None}


def bind(context=None):
    """Select the Metal context that new arrays are allocated on."""
    module = _state["module"]
    if context is None:
        if module is None:
            module = MetalArrayModule()
    elif module is None or module.context is not context:
        module = MetalArrayModule(context)
    _state["module"] = module
    return module


def _module():
    return _state["module"] or bind()


def _host(value):
    return value.to_numpy() if isinstance(value, MetalArray) else np.asarray(value)


def _wrap(array: MetalArray) -> "SArray":
    return SArray(array.buffer, array.shape, array.dtype, byte_offset=array.byte_offset)


def _upload(host, dtype=None) -> "SArray":
    host = np.asarray(host, dtype=dtype, order="C")
    return _wrap(_module().asarray(host))


class SArray(MetalArray):
    __slots__ = ()

    def view(self, shape, dtype=None, *, byte_offset=None):
        return _wrap(super().view(shape, dtype, byte_offset=byte_offset))

    def get(self):
        return self.to_numpy()

    def item(self, *index):
        host = self.to_numpy()
        return host.item(*index)

    def astype(self, dtype):
        return _upload(self.to_numpy().astype(dtype))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(np.empty(self.shape, dtype=bool).reshape(shape).shape)
        return self.view(shape)

    def ravel(self):
        return self.view((self.size,))

    def __len__(self):
        return self.shape[0]

    def __bool__(self):
        return bool(self.to_numpy())

    def __float__(self):
        return float(self.to_numpy())

    def __int__(self):
        return int(self.to_numpy())

    def __getitem__(self, key):
        try:
            return _wrap(super().__getitem__(key))
        except (NotImplementedError, TypeError, IndexError):
            host = self.to_numpy()[key]
            return _upload(host) if isinstance(host, np.ndarray) else host

    def __setitem__(self, key, value):
        if key is Ellipsis or (isinstance(key, slice) and key == slice(None)):
            return super().__setitem__(Ellipsis, value)
        host = self.to_numpy()
        host[key] = _host(value)
        super().__setitem__(Ellipsis, host)

    def copy(self):
        return _wrap(super().copy())


def _out(result):
    """Arrays go back to Metal; booleans and scalars stay on the host."""
    if isinstance(result, np.ndarray) and result.ndim and result.dtype != np.bool_:
        return _upload(result)
    return result


def _reduction(name):
    def method(self, *args, **kwargs):
        result = getattr(self.to_numpy(), name)(*args, **kwargs)
        return _out(result)
    method.__name__ = name
    return method


for _name in ("max", "min", "sum", "any", "all", "mean"):
    setattr(SArray, _name, _reduction(_name))


def _binary(op):
    def method(self, other):
        return _out(op(self.to_numpy(), _host(other)))
    return method


def _reverse(op):
    def method(self, other):
        return _out(op(_host(other), self.to_numpy()))
    return method


for _dunder, _op in (("add", operator.add), ("sub", operator.sub), ("mul", operator.mul),
                     ("truediv", operator.truediv), ("lt", operator.lt), ("le", operator.le),
                     ("gt", operator.gt), ("ge", operator.ge), ("and", operator.and_),
                     ("or", operator.or_)):
    setattr(SArray, f"__{_dunder}__", _binary(_op))
    if _dunder in ("add", "sub", "mul", "truediv"):
        setattr(SArray, f"__r{_dunder}__", _reverse(_op))
SArray.__eq__ = _binary(operator.eq)
SArray.__ne__ = _binary(operator.ne)
SArray.__hash__ = object.__hash__
SArray.__neg__ = lambda self: _upload(-self.to_numpy())
SArray.__invert__ = lambda self: _out(~self.to_numpy())
SArray.__abs__ = lambda self: _upload(np.abs(self.to_numpy()))


def asarray(value, dtype=None, order=None):
    if isinstance(value, MetalArray):
        if dtype is not None and np.dtype(dtype) != value.dtype:
            return _upload(value.to_numpy(), dtype)
        return value if isinstance(value, SArray) else _wrap(value)
    return _upload(np.asarray(value), dtype)


ascontiguousarray = asarray


def _fill(shape, dtype, value):
    array = _wrap(_module().zeros(shape, dtype))
    if value:
        array.fill(value)
    return array


def zeros(shape, dtype=np.float32):
    return _fill(shape, dtype, 0)


empty = zeros


def full(shape, value, dtype=np.float32):
    host = _host(value)
    if host.ndim:
        return _upload(np.broadcast_to(host, shape), dtype)
    return _fill(shape, dtype, host.item())


def zeros_like(array, dtype=None):
    return zeros(array.shape, array.dtype if dtype is None else dtype)


empty_like = zeros_like


def ones_like(array, dtype=None):
    return _fill(array.shape, array.dtype if dtype is None else dtype, 1)


def copyto(dst, src):
    dst[...] = _host(src) if not isinstance(src, MetalArray) else src


def asnumpy(value):
    return _host(value)


def stack(arrays, axis=0):
    return _upload(np.stack([_host(a) for a in arrays], axis=axis))


def concatenate(arrays, axis=0):
    return _upload(np.concatenate([_host(a) for a in arrays], axis=axis))


def broadcast_to(array, shape):
    return _upload(np.broadcast_to(_host(array), shape))


def isfinite(array):
    return np.isfinite(_host(array))


def any(array, *args, **kwargs):
    return np.any(_host(array), *args, **kwargs)


def all(array, *args, **kwargs):
    return np.all(_host(array), *args, **kwargs)


def minimum(a, b):
    return _upload(np.minimum(_host(a), _host(b)))


def maximum(a, b):
    return _upload(np.maximum(_host(a), _host(b)))


def min(array, *args, **kwargs):
    return np.min(_host(array), *args, **kwargs)


def array_equal(a, b):
    return bool(np.array_equal(_host(a), _host(b)))


class _Device:
    id = 0

    def __init__(self, *_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Null:
    @staticmethod
    def synchronize():
        context = _module().context
        context.wait()


cuda = types.SimpleNamespace(
    Device=_Device,
    Stream=types.SimpleNamespace(null=_Null),
    runtime=types.SimpleNamespace(
        getDeviceProperties=lambda _id: {"name": b"Apple Metal"}),
)

float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64
uint32 = np.uint32
bool_ = np.bool_
ndarray = MetalArray
