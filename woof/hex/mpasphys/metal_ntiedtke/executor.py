"""Resident MPAS New Tiedtke executor."""

import numpy as np

from woof.hex.mpasphys.ntiedtke_v842 import LEVEL_OUTPUTS
from woof.metal.array import MetalArray
from .downstream_executor import execute
from .downstream_layouts import STAGES
from .early_executor import SURFACE, prepare


_LEVEL_INPUTS = (
    't3d', 'pcps', 'dz8w', 'rho3d', 'qv3d', 'exner', 'rthdynten', 'rthblten',
    'rthratenlw', 'rthratensw', 'rqvdynten', 'rqvblten', 'u3d', 'v3d', 'qc3d', 'qi3d',
)


def _require_views(mapping, extents, context):
    for names, shape, dtype in extents:
        for name in names:
            value = mapping[name]
            if (not isinstance(value, MetalArray) or value.shape != shape
                    or value.dtype != np.dtype(dtype) or value.buffer.context is not context):
                raise TypeError(f'New Tiedtke {name} requires a same-context {np.dtype(dtype)} view of {shape}')


def _check_scratch(context, ncol, nz):
    """Bound the resident stage packets and their packed source rows."""
    levels = ncol * nz
    interfaces = ncol * (nz + 1)
    slab = ncol * (nz + 2)
    fields = {name for stage in STAGES for group in stage['groups'] for name in group}
    fields.update(('ztt', 'zqq', 'ztenu', 'ztenv'))
    state_words = sum(ncol if name in SURFACE else slab for name in fields)
    packet_words = [
        sum(ncol if name in SURFACE else slab for name in group)
        for stage in STAGES for group in stage['groups']
    ]
    # The prefix through cloud-depth kernels also own their input and output packets.
    early_words = 53 * levels + 5 * interfaces + 75 * slab + 38 * ncol + nz + 3
    needed = 4 * (early_words + state_words + sum(packet_words) + 1)
    if needed > context.recommended_working_set_size:
        raise MemoryError('New Tiedtke scratch exceeds the recommended Metal working set')
    # Cloud ascent owns the largest early-stage packet, with thirteen slabs.
    max_words = max(13 * slab, *packet_words)
    if max_words > np.iinfo(np.int32).max:
        raise ValueError('New Tiedtke scratch indexing exceeds int32')
    if 4 * max_words > context.max_buffer_length:
        raise MemoryError('New Tiedtke scratch buffer exceeds the Metal buffer limit')


def launch(module, binding, constants, *, itimestep):
    """Run all source stages and publish their outputs after the status check."""
    nz, ncol = binding.fields['t3d'].shape
    if ncol < 1 or nz < 4:
        raise ValueError('New Tiedtke requires columns and at least four levels')
    shape = (nz, ncol)
    context = module.context
    _require_views(binding.fields, (
        (_LEVEL_INPUTS, shape, np.float32),
        (('p8w', 'w'), (nz + 1, ncol), np.float32),
        (('hfx', 'qfx', 'xland', 'meshDensity'), (ncol,), np.float32),
    ), context)
    _require_views(binding.target, (
        (LEVEL_OUTPUTS, shape, np.float32),
        (('raincv', 'cuprec'), (ncol,), np.float32),
        (('cu_act_flag',), (ncol,), np.int32),
    ), context)
    _check_scratch(context, ncol, nz)

    status = module.zeros((1,), dtype=np.uint32)
    state = prepare(module, binding, constants, itimestep=itimestep)
    # Keep the dispatch packets live through the status readback.
    packets = execute(module, state, ncol=ncol, nz=nz, dt=binding.dt_dyn,
                      stepcu=binding.stepcu, constants=constants, status=status)
    if int(status.to_numpy()[0]):
        raise ValueError('New Tiedtke active downdraft has no half-level within 6000 Pa of the surface')
    del packets

    result = {}
    for name in LEVEL_OUTPUTS:
        target = binding.target[name]
        if (name == 'rqccuten' and not binding.f_qc) or (name == 'rqicuten' and not binding.f_qi):
            result[name] = target
        else:
            slab = state[name]
            result[name] = slab.view(shape, byte_offset=slab.byte_offset + 4 * ncol)
            target[...] = result[name]
    for output, target in (('raincv', 'raincv'), ('pratec', 'cuprec')):
        result[output] = state[output]
        binding.target[target][...] = result[output]
    binding.target['cu_act_flag'].fill(np.int32(1))
    return result
