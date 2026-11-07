"""Native MPAS aerosol-aware Thompson slot for Metal."""
import numpy as np
from woof.hex.metal_dynamics_v841 import _arrays, _context, _require
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from .metal_thompson_column_v842 import MetalMpasThompson, bind_session as bind_common, step
from woof.hex.mpasphys.metal_thompson_v842 import launch_conversion, launch_precip
from woof.hex.mpasphys import thompson_v842 as common, thompson_aero_v842 as aerosol
from woof.hex.mpasphys.driver_v842 import correct_source, slot_corrections

NAMES = ('nc', 'nifa', 'nwfa')
TENDENCIES = ('rncmpten', 'rnifampten', 'rnwfampten')
EMISSIONS = ('nifa2d', 'nwfa2d')


def bind_session(session, tables, *, diagnostics_requested=False):
    """Bind existing common and aerosol native storage; numerical state stays owned by pools."""
    p = bind_common(session, tables, diagnostics_requested=diagnostics_requested)
    pools = session.slot_pools
    state = session.pools.field('state', 'scalars', level=2)
    model, before, prepared = {}, {}, {}
    for name, tendency in zip(NAMES, TENDENCIES):
        index = session.layout.indices['state', 'index_' + name]
        if not 1 <= index <= state.shape[0]:
            raise ValueError('invalid native scalar index ' + name)
        model[name] = state[index - 1]
        prepared[name] = pools['prepared'][name + '_p'][:model[name].shape[0]]
        before[name] = pools['tend_physics'][tendency]
    p['aerosol_model'] = model
    p['aerosol_before'] = before
    p['aerosol_prepared'] = prepared
    p['aerosol_emission'] = {name: pools['diag_physics'][name] for name in EMISSIONS}
    return p


def _validate_and_create_views(p, dt):
    nz, nc = p['fields']['qv'].shape
    context = _context(p['fields']['qv'])
    _require('qv', p['fields']['qv'], np.float32, (nz, nc))
    if not 2 <= nz <= common.MAX_LEVELS or nc < 1:
        raise ValueError('native aerosol Thompson requires 2..256 levels and columns')
    step = np.float32(dt)
    if not np.isfinite(step) or step <= 0:
        raise ValueError('native aerosol tendency requires positive finite dynamics dt')
    resources = {}
    for name in NAMES:
        for prefix, group in (('', 'aerosol_model'), ('before_', 'aerosol_before'),
                              ('prepared_', 'aerosol_prepared')):
            value = p[group][name]
            _require(prefix + name, value, np.float32, (nz, nc))
            if _context(value) is not context:
                raise ValueError('aerosol arrays must share the Thompson context')
            resources[prefix + name] = value
    for name, value in p['aerosol_emission'].items():
        _require(name, value, np.float32, (nc,))
        if _context(value) is not context:
            raise ValueError('aerosol emissions must share the Thompson context')
        resources[name] = value
    words = 3 * nz * nc + 2 * nc
    if words * 4 > min(context.max_buffer_length, context.recommended_working_set_size):
        raise MemoryError('native Thompson aerosol slab exceeds Metal working set')
    slab = _arrays(context).empty((words,), np.float32)
    plane = nz * nc
    views = {name: slab.view((nz, nc), byte_offset=slab.byte_offset + i * plane * 4)
             for i, name in enumerate(NAMES)}
    views.update({name: slab.view((nc,), byte_offset=slab.byte_offset + (3 * plane + i * nc) * 4)
                  for i, name in enumerate(EMISSIONS)})
    return context, slab, resources, views, nz, nc, step


def launch_slot(call):
    """Submit the shared native wrapper and packed aerosol-aware column.

    The column receives five consecutive slab views and emits internally once
    per substep. Unsupported output routes refuse before either packing or the
    shared common conversion can write pool state.
    """
    if (call.slot, call.scheme) != ('microphysics', aerosol.SCHEME):
        raise ValueError('MPAS Metal aerosol Thompson requires its microphysics slot')
    p = call.pools['thompson_aero_v842']
    if p.get('nest_look') is not None:
        raise NotImplementedError('native aerosol Thompson nest look is not yet implemented on Metal')
    if type(p['n_microp']) is not int or p['n_microp'] < 1:
        raise ValueError('native manager must supply a positive integer microphysics substep count')
    microp_dt = np.float32(p['dt_microp'])
    if not np.isfinite(microp_dt) or microp_dt <= 0:
        raise ValueError('native aerosol Thompson substep must be positive and finite')
    corrections = slot_corrections(call, common.CORRECT_MODE_SOURCE)
    correct_source(common._PRECIP, common.CORRECT_MODE_SOURCE, corrections, 'thompson_v842')
    context, slab, resources, views, nz, nc, dt = _validate_and_create_views(p, call.dt)
    fields = dict(p['fields']); fields.update({name: views[name] for name in NAMES})
    surface = dict(p['surface']); surface.update({name: views[name] for name in EMISSIONS})
    conversion = (p['model'], p['mesh'], p['diagnostics'], p['tendencies'], fields, p['constants'])
    launch_precip('from', fields, surface, p['precip'], l_acrain=p['l_acrain'],
                  bucket=call.options['config_bucket_rainnc'])
    launch_conversion('from', *conversion)
    _launch('hexv842_mpas_thompson_aerosol_from_f32', nz*nc, context,
            dict(resources, slab=slab), dict(nz=np.int32(nz), columns=np.int32(nc)))
    for _ in range(p['n_microp']):
        step(fields, surface, p['tables'], dt=microp_dt, aerosol_aware=True,
                    **{name: p[name] for name in ('has_reqc', 'has_reqi', 'has_reqs')})
    if call.flags.diagnostics:
        from .metal_thompson_column_v842 import launch_radar, relative_humidity
        launch_radar(fields, p['diagnostics'])
        n = nz * nc
        relative_humidity(*(fields[name].view((n,)) for name in ('qv', 'th', 'pii', 'p')),
                          p['diagnostics']['relhum'].view((n,)))
    launch_precip('to', fields, surface, p['precip'], l_acrain=p['l_acrain'],
                  bucket=call.options['config_bucket_rainnc'], corrections=corrections)
    launch_conversion('to', *conversion, mp_top_level=p['mp_top_level'], dt_dyn=dt)
    outputs = {name: value for name, value in resources.items() if name not in EMISSIONS}
    _launch('hexv842_mpas_thompson_aerosol_to_f32', nz*nc, context,
            dict(outputs, slab=slab), dict(nz=np.int32(nz), columns=np.int32(nc), dt_dyn=dt))


class MetalMpasThompsonAerosol(MetalMpasThompson):
    """Native aerosol slot with checkpoint-owned physical pools."""
    def bind_pools(self, pools, call):
        return {'thompson_aero_v842': bind_session(pools.session, self.tables, diagnostics_requested=call.flags.diagnostics)}

    launch_slot = staticmethod(launch_slot)
