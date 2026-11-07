"""Publish RRFS TEMPO's native cold-start aerosols through Metal pools."""

import numpy as np

from woof.metal.array import MetalArray

from .metal_tempo_v303 import initialize_native_aerosols


def initialize_tempo_aerosols(session, configs):
    """Initialize level-one aerosol state and surface sources on solve cells."""
    if (configs['config_microp_scheme'].strip() != 'mp_tempo'
            or not configs.get('config_tempo_aerosolaware', False)
            or configs['config_do_restart']):
        return
    pools, native = session.pools, session.native
    dims = session.layout.dimensions
    solve, nz = int(dims['nCellsSolve']), int(dims['nVertLevels'])

    def resident(pool, name, level=1):
        row = pools.rows[pool, name]
        key = (pool, row['pool_name'], level)
        item = pools.buffers.get(key)
        value = item[1] if item is not None else pools.field(pool, name, level=level)
        if not isinstance(value, MetalArray) or value.dtype != np.float32:
            raise TypeError(f'RRFS TEMPO startup requires a Metal float32 {pool}.{name}')
        return key, value

    state_key, state = resident('state', 'scalars', level=1)
    _, zgrid = resident('mesh', 'zgrid')
    _, area = resident('mesh', 'areaCell')
    w_key, nwfa2d = resident('diag_physics', 'nwfa2d')
    i_key, nifa2d = resident('diag_physics', 'nifa2d')
    if (state.ndim != 3 or state.shape[1] != nz or not 0 <= solve <= state.shape[2]
            or zgrid.ndim != 2 or zgrid.shape[0] != nz + 1 or zgrid.shape[1] < solve
            or area.ndim != 1 or nwfa2d.ndim != 1 or nifa2d.ndim != 1
            or min(area.size, nwfa2d.size, nifa2d.size) < solve):
        raise ValueError('RRFS TEMPO startup pools do not contain the solve extent')
    indices = {}
    for name in ('nwfa', 'nifa'):
        value = int(native._dimension_values['state', 'index_' + name]) - 1
        if not 0 <= value < state.shape[0]:
            raise ValueError(f'RRFS TEMPO startup requires a valid {name} scalar index')
        indices[name] = value
    if indices['nwfa'] == indices['nifa']:
        raise ValueError('RRFS TEMPO startup aerosol scalar indices must differ')

    # A cached full field is authoritative too. Pack its solve columns without
    # asking MetalPoolViews for a conflicting second extent.
    xp = pools.arrays
    state_host = None
    if state.shape[2] == solve:
        profiles = {name: state[i] for name, i in indices.items()}
    else:
        state_host = state.to_numpy()
        profiles = {name: xp.asarray(np.ascontiguousarray(state_host[i, :, :solve]))
                    for name, i in indices.items()}
    zsolve = zgrid if zgrid.shape[1] == solve else xp.asarray(np.ascontiguousarray(zgrid.to_numpy()[:, :solve]))
    wsolve, isolve, asolve = nwfa2d[:solve], nifa2d[:solve], area[:solve]
    reduce = getattr(session, 'rank_reduce', None)
    extrema = {}
    for name, values in profiles.items():
        words = values.to_numpy()
        low, high = (float(words.min()), float(words.max())) if solve else (np.inf, -np.inf)
        if reduce is not None:
            low, high = reduce(low, 'min'), reduce(high, 'max')
        extrema[name] = low, high
    initialize_native_aerosols(profiles['nwfa'], profiles['nifa'], wsolve, isolve, zsolve, asolve,
                               solve_cells=solve, global_nwfa_extrema=extrema['nwfa'],
                               global_nifa_extrema=extrema['nifa'], do_restart=False)
    if not solve:
        return
    if state_host is not None:
        for name, column in profiles.items():
            state_host[indices[name], :, :solve] = column.to_numpy()
        state[...] = state_host

    # Publish only the modified fields. The cached selection records whether
    # each authority contains solve columns or the full local allocation.
    for key in (state_key, w_key, i_key):
        host, device, selection = pools.buffers[key]
        host[selection] = device.to_numpy()
        native.put_field(key[0], key[1], host, level=key[2],
                         nlevels=native._fields[key[0], key[1]][2])
