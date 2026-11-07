"""Resident dispatch for the retained RRFS dycore kernel ABIs."""
import numpy as np
from woof.metal.array import MetalArray, MetalArrayModule
from .metal_rrfs_arguments import ARGUMENTS
from . import metal_v841_launch


def launch(area, symbol, count, arguments, *, levels=1, species=1):
    if count * levels * species == 0:
        return
    resources, params, original = {}, {}, []
    for (name, pointer, dtype), value in zip(ARGUMENTS[area][symbol], arguments, strict=True):
        if pointer:
            if not isinstance(value, MetalArray) or value.dtype != np.dtype(dtype):
                raise TypeError(f'{symbol}.{name} requires resident {dtype}')
            resources[name] = value.view((value.size,))
            original.append(value)
        else:
            params[name] = int(value) if dtype == 'int32' else float(np.float32(value))
    params.update(dispatch_x=int(count), dispatch_levels=int(levels), dispatch_species=int(species))
    entrypoint = 'hexv841_rrfs_' + area + '_' + symbol
    context = original[0].buffer.context
    context.submit_kernel(entrypoint, resources, params, exact_policy=False,
                          grid_size=int(count*levels*species), threads_per_group=128)
    hook = metal_v841_launch.POST_LAUNCH
    if hook is not None:
        with metal_v841_launch.suspended():
            hook(entrypoint, original)


def transpose(value):
    """Copy an immutable native stencil to slot-major device storage."""
    from .metal_v841_launch import launch as dispatch
    rows, columns = value.shape
    xp = MetalArrayModule(value.buffer.context)
    result = xp.empty((columns, rows), value.dtype)
    dispatch(value.buffer.context, 'hexv841_rrfs_transpose', value.size,
             {'src': value.view((value.size,), np.uint32),
              'dst': result.view((value.size,), np.uint32)},
             {'rows': rows, 'columns': columns})
    return result


class Arrays(MetalArrayModule):
    """Allocation/copy subset used by the retained RRFS launch schedules."""
    float32 = np.float32
    int32 = np.int32
    ndarray = MetalArray

    def empty_like(self, value):
        return self.empty(value.shape, value.dtype)

    def zeros_like(self, value):
        return self.zeros(value.shape, value.dtype)

    def ones_like(self, value):
        return self.ones(value.shape, value.dtype)

    def array(self, value, *, copy=False):
        return value.copy() if copy else self.asarray(value)

    def ascontiguousarray(self, value):
        return value
