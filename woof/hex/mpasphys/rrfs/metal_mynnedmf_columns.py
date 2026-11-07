"""Batch resident MYNN-EDMF solve columns into the registered Metal ABI."""
from math import prod

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule


_MAX_I32 = np.iinfo(np.int32).max


def _transfer(context, fields, packed, first, count, nc, mapping, *, scatter):
    copies, calls = [], []
    offset = 0
    for spec, array, rows in fields:
        size = rows * count
        if not scatter or spec['intent'] != 'in':
            compact = packed.view((size,), byte_offset=packed.byte_offset + offset * 4)
            if count == nc or rows == 1:
                native = array.view((size,), np.uint32, byte_offset=array.byte_offset + first * 4)
                copies.append((compact, native) if scatter else (native, compact))
            else:
                # The shared transport kernel moves FP32 words without arithmetic.
                # The only integer carrier (kpbl) has one row and uses bookkeeping.
                native = array.view((array.size - first,), byte_offset=array.byte_offset + first * 4)
                compact = compact.view(compact.shape, np.float32)
                src, dst = (compact, native) if scatter else (native, compact)
                calls.append(context.prepare_kernel_call('hexv842_mpas_transport_rows_f32',
                    {'src': src, 'dst': dst, 'mapping': mapping},
                    {'nrows': 1, 'nlev': rows, 'nsolve': count, 'nfull': nc, 'scatter': int(scatter)},
                    exact_policy=False, grid_size=size))
        offset += size
    for first_call in range(0, len(copies), 64):
        context.submit_bookkeeping(copies[first_call:first_call + 64])
    for first_call in range(0, len(calls), 64):
        context.submit_kernel_batch(calls[first_call:first_call + 64], label='RRFS MYNN-EDMF transport')


def launch_columns(arguments, *, source, entrypoint):
    nc = int(arguments['ite']) - int(arguments['its']) + 1
    if (nc < 1 or arguments['jts'] != arguments['jte']
            or arguments['kme'] < arguments['kte'] + 1):
        raise ValueError('Metal RRFS MYNN-EDMF requires a solve row with the upper interface level')
    context = arguments['pblh'].buffer.context
    xp = MetalArrayModule(context)
    array_specs = tuple((name, spec) for name, spec in source.ARGUMENTS.items()
                        if spec['dimensions'])
    scalars = {}
    for name, spec in source.ARGUMENTS.items():
        if spec['dimensions'] or name in ('errflg', 'errmsg'):
            continue
        value = arguments[name]
        if spec['optional']:
            scalars['present_' + name] = int(value is not None)
            if value is None:
                value = 0
        scalars[name] = np.float32(value) if spec['kind'] == 'float' else int(value)
    fields = []
    words_per_column = 0
    channel_starts = np.zeros(len(array_specs), np.int32)
    masks = [0, 0, 0]
    for index, (name, spec) in enumerate(array_specs):
        array = arguments[name]
        if array is None and spec['optional']:
            continue
        dims = spec['dimensions']
        # The compact views omit the one j row and any non-solve i columns.
        native_extents = source.argument_shape(spec, arguments)
        shape = tuple(size for dim, size in zip(reversed(dims), reversed(native_extents))
                      if dim not in ('ims:ime', 'jms:jme')) + (nc,)
        rows = prod(shape[:-1])
        dtype = np.int32 if spec['kind'] == 'int' else np.float32
        if (not isinstance(array, MetalArray) or array.buffer.context is not context
                or array.dtype != np.dtype(dtype) or array.shape != shape or rows < 1):
            raise ValueError(f'RRFS MYNN-EDMF {name} requires a resident {np.dtype(dtype)} {shape}')
        if words_per_column + rows > _MAX_I32:
            raise MemoryError('RRFS MYNN-EDMF column carrier exceeds signed word addressing')
        channel_starts[index] = words_per_column
        words_per_column += rows
        masks[index // 32] |= 1 << (index % 32)
        fields.append((spec, array, rows))
    stride = source.workspace_bytes(arguments)
    if stride < 1 or stride % 8 or stride > _MAX_I32:
        raise ValueError('RRFS MYNN-EDMF arena stride exceeds signed byte addressing')
    fixed_bytes = (nc + len(array_specs) + 1) * 4
    max_buffer = int(context.max_buffer_length)
    working_set = int(context.recommended_working_set_size) // 8
    batch = min(nc, max_buffer // stride, max_buffer // (4 * words_per_column),
                _MAX_I32 // words_per_column,
                (working_set - fixed_bytes) // (stride + 4 * words_per_column))
    if batch < 1 or nc * 4 > max_buffer:
        raise MemoryError('RRFS MYNN-EDMF has no Metal workspace for a solve column')
    packed = xp.empty((batch * words_per_column,), np.uint32)
    workspace = xp.zeros((batch * stride // 4,), np.uint32)
    errors = xp.empty((nc,), np.int32)
    mapping = xp.zeros((1,), np.int32)
    offsets = xp.asarray(channel_starts * batch)
    scalars.update(workspace_stride=stride, **{f'presence{i}': int(np.uint32(mask).view(np.int32))
                                            for i, mask in enumerate(masks)})
    for first in range(0, nc, batch):
        count = min(batch, nc - first)
        if count != batch:
            offsets = xp.asarray(channel_starts * count)
        _transfer(context, fields, packed, first, count, nc, mapping, scatter=False)
        context.submit_kernel(entrypoint,
            {'fields': packed, 'workspace': workspace, 'error_flags': errors, 'offsets': offsets},
            dict(scalars, batch_first=first, batch_count=count), exact_policy=True, grid_size=count)
        _transfer(context, fields, packed, first, count, nc, mapping, scatter=True)
    status = errors.to_numpy()
    if np.any(status == -1):
        raise ValueError('RRFS MYNN-EDMF vertical extent ends inside a source height scan')
    if np.any(status):
        raise ValueError('RRFS MYNN-EDMF reported a native column error')
    return errors
