"""MPAS v8.4.2 transport-to-microphysics handoff on Metal."""
from __future__ import annotations
import math
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _i, _require
from .metal_tendencies_v842 import _launch


def source_baseline(*, before, source, rho_old, dt, n_cells_solve):
    nlev, ncells = rho_old.shape
    _require('rho_old', rho_old, np.float32, (nlev, ncells))
    _require('before', before, np.float32)
    _require('source', source, np.float32, before.shape)
    if before.ndim != 3 or before.shape[1:] != (nlev, ncells):
        raise ValueError('MPAS source baseline must match old density extent')
    if not 0 < n_cells_solve <= ncells or not math.isfinite(dt) or dt <= 0:
        raise ValueError('MPAS source baseline requires positive dt and solve extent')
    context = _context(before)
    baseline = _arrays(context).empty(before.shape, np.float32)
    _launch('hexv842_mpas_transport_baseline_f32', before.size, context,
            dict(before=before, source=source, rho_old=rho_old, baseline=baseline),
            dict(nlev=_i(nlev), ncells=_i(ncells), nsolve=_i(n_cells_solve),
                 nspecies=_i(before.shape[0]), dt=np.float32(dt)))
    return baseline


def handoff(*, before, after, theta_m, scalar_names, rqvdynten, rthdynten,
            convection_scheme, monotonic, dt, n_cells_solve):
    names = tuple(scalar_names)
    if len(set(names)) != len(names) or 'qv' not in names:
        raise ValueError('MPAS scalar names must be unique and include qv')
    nlev, ncells = theta_m.shape
    _require('theta_m', theta_m, np.float32, (nlev, ncells))
    for name, value in (('before', before), ('after', after)):
        _require(name, value, np.float32, (len(names), nlev, ncells))
    for name, value in (('rqvdynten', rqvdynten), ('rthdynten', rthdynten)):
        _require(name, value, np.float32, (nlev, ncells))
    if not 0 < n_cells_solve <= ncells or not math.isfinite(dt) or dt <= 0:
        raise ValueError('MPAS transport requires positive dt and solve extent')
    _launch('hexv842_mpas_transport_handoff_f32', nlev*ncells, _context(theta_m),
            dict(before=before, after=after, theta_m=theta_m, rqvdynten=rqvdynten, rthdynten=rthdynten),
            dict(nlev=_i(nlev), ncells=_i(ncells), nsolve=_i(n_cells_solve), nspecies=_i(len(names)),
                 iqv=_i(names.index('qv')), handoff=_i(convection_scheme in ('cu_grell_freitas', 'cu_ntiedtke')),
                 monotonic=_i(monotonic), dt=np.float32(dt), rv=np.float32(461.6), rd=np.float32(287.),
                 rvord=np.float32(np.float32(461.6)/np.float32(287.))))
