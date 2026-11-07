"""Strict regional WSM6 component submissions."""
import numpy as np

from woof.metal.strict_regional_kessler_family import _submit


def prepare(state, *, theta, rho, pii, dz, z8w):
    nz, ny, nx = state.p.shape
    return _submit(
        'strict_regional_wsm6_prep',
        {'thb': state.thb, 'thp': state.thp, 'alt': state.alt,
         'pressure': state.p, 'phb': state.phb, 'php': state.php,
         't': theta, 'rho': rho, 'pii': pii, 'dz': dz, 'zi': z8w,
         'saved': state.h_diabatic},
        {'nz': nz, 'nc': ny * nx,
         'thb3d': int(state.thb.shape == (nz, ny, nx)),
         'phb3d': int(state.phb.shape == (nz + 1, ny, nx))}, ny * nx)


def column(*, dt, hail_opt, **arrays):
    from woof.core.wsm6_constants import wrf_init_words
    from woof.metal.array import MetalArrayModule

    theta = arrays['theta']
    nz, ny, nx = theta.shape
    context = theta.buffer.context
    if not hasattr(context, '_physics_cache'):
        context._physics_cache = {}
    key = ('wsm6_init', int(hail_opt))
    if key not in context._physics_cache:
        context._physics_cache[key] = MetalArrayModule(context).asarray(
            wrf_init_words(hail_opt))
    arrays['coef'] = context._physics_cache[key]
    flat = {name: value.view((int(value.size),)) if value.ndim != 1 else value
            for name, value in arrays.items()}
    return context.submit_kernel(
        'strict_regional_wsm6_deep' if nz > 64 else 'strict_regional_wsm6', flat,
        {'delt': np.float32(dt), 'nz': nz, 'ny': ny, 'nx': nx},
        exact_policy=True, grid_size=ny * nx, threads_per_group=32)
