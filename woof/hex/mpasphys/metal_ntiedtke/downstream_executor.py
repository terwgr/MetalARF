"""Enqueue source-named stages from downdraft through output conversion."""
import numpy as np
from .downstream_layouts import STAGES


def _flat(array):
    return array.view((array.size,))


def execute(module, state, *, ncol, nz, dt, stepcu, constants,
            tiedtke_closure=False, status=None):
    """Enqueue all stages from cudlfsn through post_run without host readback.

    ``state`` is a mutable dictionary of owning MetalArrays keyed by source
    name. Its caller allocates and seeds them once from the verified early
    stages, including distinct ztt/zqq and driver-reference arrays. Aliases
    in STAGES deliberately point to the same dictionary names.
    """
    ctx = module.context
    delt = np.float32(np.float32(dt) * np.float32(stepcu))
    common = dict(ncol=ncol, nz=nz, klev=nz, ktopm2=2, dt=np.float32(dt),
                  stepcu=stepcu, delt=delt, ztmst=delt,
                  tiedtke_closure=int(tiedtke_closure))
    common.update({key: np.float32(getattr(constants, key))
                   for key in ('cp', 'rd', 'rv', 'xlv', 'xls', 'xlf', 'grav')})
    live = []
    for stage in STAGES:
        if stage['name'] == 'hexv842_nt_cududvn':
            ctx.submit_bookkeeping(((_flat(state['ptenu']), _flat(state['ztenu'])),
                                    (_flat(state['ptenv']), _flat(state['ztenv']))))
        buffers, writes, pres = {}, [], []
        for group_index, ((resname, dtype, _), names) in enumerate(zip(stage['resources'], stage['groups'])):
            fields = [_flat(state[name]) for name in names]
            packed = module.empty((sum(a.size for a in fields),), dtype={'float': np.float32, 'float32': np.float32, 'int32': np.int32, 'uint32': np.uint32}[dtype])
            buffers[resname] = packed
            offset = 0
            for field in fields:
                view = packed.view((field.size,), dtype=field.dtype,
                                   byte_offset=packed.byte_offset + offset * 4)
                pres.append((field, view))
                if group_index in stage['outputs']:
                    writes.append((view, field))
                offset += field.size
        # Ordered queue submissions make the packs visible to the following
        # dispatch and the dispatch visible to its unpack, without a wait.
        if pres:
            ctx.submit_bookkeeping(pres)
        if stage['name'] == 'hexv842_nt_cuddrafn':
            buffers['status'] = status
        params = {name: common[name] for name, _ in stage['parameters']}
        ctx.submit_kernel(stage['name'], buffers, params, exact_policy=False,
                          grid_size=ncol, threads_per_group=32)
        if writes:
            ctx.submit_bookkeeping(writes)
        live.extend(buffers.values())
    # Keep scratch owners reachable until the final queued dispatch completes.
    return live
