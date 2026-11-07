"""Resident FP32 IAU rate formation and mass coupling."""
import numpy as np

from .array import MetalArray, MetalArrayModule


_SLOW_SLOTS = dict(u='ru_t', v='rv_t', w='rw_t', php='rph_t', thp='rth_t')
_MAPS = dict(u='msfu', v='msfv', w='msft', php='msft', thp='msft')
_MAX_INDEX = np.iinfo(np.int32).max


def _resident(array, context, shape):
    return (isinstance(array, MetalArray) and array.buffer.context is context
            and array.dtype == np.dtype(np.float32) and array.shape == shape)


class ResidentIau:
    """Keep the attached analysis nodes resident across steps of their window."""

    def __init__(self, context, nodes):
        self.context = context
        self.arrays = MetalArrayModule(context)
        self.deltas = None
        fields = {name: delta for node in nodes for name, delta in node.increments.items()}
        all_deltas = [delta for node in nodes for delta in node.increments.values()]
        required = sum(a.nbytes for a in all_deltas) + sum(a.nbytes for a in fields.values())
        if (required > int(context.recommended_working_set_size) // 16
                or any(a.dtype != np.dtype(np.float32) or a.ndim != 3
                       or a.nbytes > int(context.max_buffer_length)
                       or not 0 < a.size <= _MAX_INDEX
                       or not np.isfinite(a).all() for a in all_deltas)):
            return
        self.deltas = [{name: self.arrays.asarray(delta)
                        for name, delta in node.increments.items()} for node in nodes]

    def fold(self, state, cfg, weights, dt, boundary_rows):
        if self.deltas is None or state.mup.ndim != 2:
            return None
        context = self.context
        ny, nx = state.mup.shape
        if (not nx or not ny or not _resident(state.mup, context, (ny, nx))
                or not _resident(state.mub2d, context, (ny, nx))):
            return None
        boundary = bool(getattr(cfg, 'specified', False) or getattr(cfg, 'nested', False))
        terms, layouts = [], {}
        for deltas, weight in zip(self.deltas, weights):
            if weight <= 0.0:
                continue
            with np.errstate(over='ignore', invalid='ignore'):
                factor = np.float32(weight / dt)
            if not np.isfinite(factor):
                return None
            for name, delta in deltas.items():
                terms.append((name, delta, factor))
                if name in layouts:
                    continue
                stagger = 1 if name == 'u' else 2 if name == 'v' else 0
                full = name in ('w', 'php')
                nz = delta.shape[0]
                horizontal = ny + (stagger == 2), nx + (stagger == 1)
                c1, c2 = (state.c1f, state.c2f) if full else (state.c1h, state.c2h)
                has_map = name in _MAPS and bool(getattr(state, 'has_msf', False))
                map_factor = getattr(state, _MAPS[name]) if has_map else state.mub2d
                if (delta.shape != (nz, *horizontal)
                        or not _resident(c1, context, (nz,))
                        or not _resident(c2, context, (nz,))
                        or (has_map and not _resident(map_factor, context, horizontal))):
                    return None
                mask_x = mask_y = 0
                if boundary:
                    if not full or boundary_rows > 1:
                        mask_y = min(boundary_rows, horizontal[0])
                        mask_x = min(boundary_rows, horizontal[1])
                elif name == 'u' and getattr(cfg, 'open_x', False):
                    mask_x = 1
                elif name == 'v' and getattr(cfg, 'open_y', False):
                    mask_y = 1
                layouts[name] = (c1, c2, map_factor,
                                 dict(n=delta.size, nz=nz, ny=ny, nx=nx, stagger=stagger,
                                      mask_y=mask_y, mask_x=mask_x, has_map=int(has_map)))
        rates = {}
        for name, delta, factor in terms:
            rate = rates.get(name)
            accumulate = rate is not None
            if not accumulate:
                rate = rates[name] = self.arrays.empty(delta.shape, dtype=np.float32)
            context.submit_kernel('da_iau_rate',
                                  dict(delta=delta.view((delta.size,)), rate=rate.view((rate.size,))),
                                  dict(n=rate.size, factor=float(factor), accumulate=int(accumulate)),
                                  exact_policy=False, grid_size=rate.size)
        slow, scalars = {}, {}
        for name, rate in rates.items():
            c1, c2, map_factor, params = layouts[name]
            context.submit_kernel('da_iau_couple',
                                  dict(rate=rate.view((rate.size,)), mub=state.mub2d, mup=state.mup,
                                       c1=c1, c2=c2, map=map_factor),
                                  params, exact_policy=False, grid_size=rate.size)
            slot = _SLOW_SLOTS.get(name)
            if slot is None:
                scalars[name] = rate
            else:
                slow[slot] = rate
        return slow, scalars
