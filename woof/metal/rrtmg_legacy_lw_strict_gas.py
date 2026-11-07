"""Registered strict regional longwave gas-band submission."""
import numpy as np
from woof.core.rrtmg_lw import GPU_BAND_TABS


def submit_band(band, laytrop, fs, isv, wx, chi_mls, tables, gas_tau, fracs, *, oneminus):
    """Submit one source-ordered band, using resident ``kgNN/name`` tables.

    ``fs`` and ``isv`` are SETCOEF's 33 and 6 slot arrays. The preceding
    regional INATM and SETCOEF stages use a separate 38-slot molecular array.
    """
    if band not in GPU_BAND_TABS:
        raise ValueError('band must be in 1..16')
    if fs.ndim != 3 or fs.shape[0] != 33:
        raise ValueError('fs must have shape (33, ncol, nl)')
    _, ncol, nl = fs.shape
    if ncol < 1 or nl < 1:
        raise ValueError('ncol and nl must be positive')
    resources = {'laytrop_v': laytrop, 'fs': fs, 'isv': isv, 'wx': wx,
                 'chi_mls': chi_mls, 'gas_tau': gas_tau, 'fracs': fracs}
    resources.update({name: tables[f'kg{band:02d}/{name}'] for name in GPU_BAND_TABS[band]})
    return fs.buffer.context.submit_kernel(
        f'rrtmg_legacy_lw_strict_band{band}', resources,
        {'ncol': ncol, 'nl': nl, 'oneminus': np.float32(oneminus)},
        exact_policy=True, grid_size=ncol * nl, threads_per_group=128)
