"""Retained RRFS startup directions, accumulators and species flags on Metal."""
from operator import index

import numpy as np

from woof.metal.array import MetalArray

from .driver_rrfs import _INIT_FLUXES, _INIT_SPECIES
from .framework_init_rrfs import FrameworkInitializationRRFS


def initialize(pools, options, dimensions, scalar_indices):
    """Initialize full resident pool views and return native scalar flags.

    The caller must bind full entity strides and synchronize pool ownership.
    """
    mesh, diag, sfc = pools['mesh'], pools['diag_physics'], pools['sfc_input']
    cells, solve, lags = (index(dimensions[name]) for name in ('nCells', 'nCellsSolve', 'nLags'))
    east, north = mesh['east'], mesh['north']
    if (not isinstance(east, MetalArray) or east.ndim != 2 or east.shape[0] != 3
            or north.shape != east.shape or not 0 <= solve <= cells <= east.shape[1]
            or east.size >= 2**31):
        raise ValueError('RRFS startup requires full direction arrays and valid cell bounds')
    extent = east.shape[1]
    if mesh['latCell'].shape != (extent,) or mesh['lonCell'].shape != (extent,):
        raise ValueError('RRFS startup lat/lon and direction strides must match')
    ctx = east.buffer.context
    if cells:
        ctx.submit_kernel('hexrrfs_init_directions',
                          {'lat': mesh['latCell'], 'lon': mesh['lonCell'], 'east': east, 'north': north},
                          {'cells': cells, 'extent': extent}, exact_policy=True, grid_size=cells)
    if 'config_microphysics_top' in options:
        levels = index(dimensions['nVertLevels'])
        if levels < 1 or mesh['rdzw'].size < levels or mesh['dzu'].size < levels:
            raise ValueError('RRFS startup microphysics top requires complete vertical coordinates')
        ctx.submit_kernel('hexrrfs_init_microphysics_top',
                          {'rdzw': mesh['rdzw'][:levels], 'dzu': mesh['dzu'][:levels],
                           'level': pools['tend_physics']['mp_top_level'].view((1,))},
                          {'levels': levels, 'top': np.float32(options['config_microphysics_top'])},
                          exact_policy=True, grid_size=1)
    flags = {'f_' + name: int(index(scalar_indices['index_' + name]) > -1) for name in _INIT_SPECIES}
    flags.update(f_qoz=0, f_nbca=0)
    if bool(options['config_do_restart']) or not solve:
        return flags

    def zero(name):
        diag[name][:solve].fill(0)

    def copy(name, src):
        diag[name][:solve][...] = src[:solve]

    if options['config_convection_scheme'].strip() != 'off':
        zero('i_rainc')
    if options['config_microp_scheme'].strip() != 'off':
        zero('i_rainnc')
    for name in _INIT_FLUXES:
        zero('i_' + name)
        zero(name)
    if not bool(options['config_do_DAcycling']):
        copy('xicem', sfc['xice'])
        copy('sstsk', sfc['sst'])
    if bool(options['config_oml1d']):
        copy('t_oml', sfc['sst'])
        copy('t_oml_initial', sfc['sst'])
        depth = np.float32(options['config_oml_hml0'])
        if depth > 0:
            diag['h_oml'][:solve].fill(depth)
            diag['h_oml_initial'][:solve].fill(depth)
        else:
            copy('h_oml', diag['h_oml_initial'])
        if depth >= 0:
            zero('hu_oml')
            zero('hv_oml')
            size = sfc['sst'].size
            ctx.submit_kernel('hexrrfs_init_ocean_temperature',
                              {'sst': sfc['sst'], 'deep': diag['t_oml_200m_initial']},
                              {'extent': size, 'solve': solve}, exact_policy=True, grid_size=solve)
    for name in ('nsteps_accum', 'ndays_accum', 'tday_accum', 'tyear_accum'):
        zero(name)
    copy('tyear_mean', sfc['tmn'])
    if diag['tlag'].shape[0] < lags:
        raise ValueError('RRFS startup tlag storage is shorter than nLags')
    for lag in range(lags):
        diag['tlag'][lag][:solve][...] = sfc['tmn'][:solve]
    return flags


def prepare_pools(session):
    """Initialize and publish full Registry pools before scheme startup."""
    native, views, layout = session.native, session.pools, session.layout
    fields = dict(FrameworkInitializationRRFS.PREPARED_FIELDS)
    if 'config_microphysics_top' in layout.configs:
        fields['mesh'] = (*fields['mesh'], 'rdzw', 'dzu')
        fields['tend_physics'] = ('mp_top_level',)
    host = {pool: {name: native.field_storage(pool, name)
                   for name in names if (pool, name) in native._fields}
            for pool, names in fields.items()}
    total = sum(value.nbytes for arrays in host.values() for value in arrays.values())
    if total > views.arrays.context.recommended_working_set_size:
        raise MemoryError('RRFS startup workspace exceeds the Metal device budget')
    work = {pool: {name: views.arrays.asarray(np.ascontiguousarray(value))
                   for name, value in arrays.items()}
            for pool, arrays in host.items()}
    indices = {name: int(value) for (pool, name), value in native._dimension_values.items()
               if pool == 'state' and name.startswith('index_')}
    flags = initialize(work, layout.configs, layout.dimensions, indices)

    for pool, arrays in work.items():
        if pool == 'sfc_input':
            continue
        for name, value in arrays.items():
            original = host[pool][name]
            original[...] = value.to_numpy().reshape(original.shape)
            cached = views.buffers.get((pool, name, 1))
            if cached is not None:
                _, device, selection = cached
                device[...] = np.ascontiguousarray(original[selection]).reshape(device.shape)
    for name, value in flags.items():
        if name in native.abi['variables']:
            native.put_variable(name, np.asarray(value, np.int32))
    return flags
