"""Native MPAS-A v8.4.2 Kessler microphysics on resident Metal pools."""

import numpy as np

from ..metal_dynamics_v841 import _arrays, _context, _require
from . import kessler_v842 as native
from .driver_v842 import slot_corrections
from .metal_tendencies_v842 import _launch


_SURFACE_FIELDS = frozenset({
    'surface_pressure', 'tend_sfc_pressure', 'rainnc', 'rainncv', 'precipw', 'i_rainnc',
})
_KERNEL = 'hexv842_mpas_kessler_'


class MetalMpasKessler:
    corrections = frozenset(native.CORRECT_MODE_SOURCE)

    def bind_pools(self, provider, call):
        top = int(provider['tend_physics']['mp_top_level'].to_numpy().reshape(-1)[0])
        bound = native.bind_native_pools(provider, call, writeback_levels=top)
        bound['solve_cells'] = int(provider.session.layout.dimensions['nCellsSolve'])
        return bound

    def launch_slot(self, call):
        if (call.slot, call.scheme) != ('microphysics', 'mp_kessler'):
            raise ValueError('native Kessler microphysics slot required')
        bound = call.pools
        state = bound['state']
        constants = {name: np.float32(value) for name, value in bound['constants'].items()}
        nz, nc = state['theta_m'].shape
        solve, top = bound['solve_cells'], bound['mp_top_level']
        profile_words = nz * nc
        work_words = 6 * profile_words
        if (nz < 2 or nc < 1 or not 0 <= solve <= nc or not 1 <= top <= nz
                or work_words > np.iinfo(np.int32).max):
            raise ValueError('invalid Kessler native extents')
        dt, dt_dyn = np.float32(call.dt), np.float32(bound['dt_dyn'])
        if not np.isfinite(dt) or not np.isfinite(dt_dyn) or min(dt, dt_dyn) <= 0:
            raise ValueError('Kessler requires positive finite physics and dynamics timesteps')

        context = _context(state['theta_m'])
        for name, value in state.items():
            shape = (nc,) if name in _SURFACE_FIELDS else (nz + 1, nc) if name == 'zgrid' else (nz, nc)
            _require('Kessler ' + name, value, np.int32 if name == 'i_rainnc' else np.float32, shape)
            if _context(value) is not context or not value.flags.c_contiguous:
                raise ValueError('Kessler requires contiguous arrays in the same Metal context')
        if not solve:
            return
        if (4 * work_words > context.max_buffer_length
                or 4 * (14 * profile_words + 2 * nc) > context.recommended_working_set_size):
            raise MemoryError('Kessler workspace exceeds context allocation limits')

        xp = _arrays(context)
        local = {name: xp.empty((nz, nc), np.float32)
                 for name in ('t', 'qvp', 'qcp', 'qrp', 'rho', 'pii', 'z', 'dz8w')}
        local['rain_local'] = xp.zeros((nc,), np.float32)
        local['last_split'] = xp.zeros((nc,), np.float32)
        local['work'] = xp.empty((6, nz, nc), np.float32)
        arrays = dict(state, **local)
        common = dict(nz=np.int32(nz), nc=np.int32(nc), ns=np.int32(solve))

        def launch(stage, count, names, aliases, **params):
            fields = {name: arrays[aliases.get(name, name)] for name in names}
            _launch(_KERNEL + stage + '_f32', count, context, fields, dict(common, **params))

        launch('from', profile_words,
               ('theta_m', 'qv', 'qc', 'qr', 'rho_zz', 'zz', 'exner', 'zgrid',
                't', 'qvp', 'qcp', 'qrp', 'rho', 'pii', 'z', 'dz'),
               {'dz': 'dz8w'}, rvord=constants['rvord'])
        launch('column', nc,
               ('t', 'qv', 'qc', 'qr', 'rho', 'pii', 'z', 'dz8w', 'rainnc', 'rainncv', 'work'),
               {'qv': 'qvp', 'qc': 'qcp', 'qr': 'qrp', 'rainnc': 'rain_local', 'rainncv': 'last_split'},
               dt=dt, **{name: constants[name] for name in
                         ('xlv', 'cp', 'ep2', 'svp1', 'svp2', 'svp3', 'svpt0', 'rhowater')})
        if call.flags.diagnostics:
            # Kessler RH uses post-column temperature/vapour and pre-writeback pressure.
            _launch(_KERNEL + 'relhum_f32', profile_words, context,
                    {name: arrays[name] for name in ('t', 'pii', 'qvp', 'pressure_b', 'pressure_p', 'relhum')},
                    dict(n=np.int32(profile_words), nc=common['nc'], ns=common['ns']))
        corrections = slot_corrections(call, native.CORRECT_MODE_SOURCE)
        launch('precip', nc, ('rho', 'qvp', 'dz', 'rain_local', 'rainnc', 'rainncv', 'precipw', 'i_rainnc'),
               {'dz': 'dz8w'}, bucket=np.float32(call.options['config_bucket_rainnc']),
               accumulate=np.int32(call.flags.rain_bucket), correct_wb3=np.int32('WB3' in corrections))
        launch('to', nc,
               ('theta_m', 'qv', 'qc', 'qr', 'rho_zz', 'zz', 'exner', 'exner_b', 'rtheta_b',
                'rtheta_p', 'pressure_p', 'pressure_b', 'zgrid', 'surface_pressure',
                'tend_sfc_pressure', 'rt_diabatic_tend', 'dtheta_dt_mp', 't', 'qvp', 'qcp', 'qrp'),
               {}, dt_dyn=dt_dyn, mp_top=np.int32(top),
               **{name: constants[name] for name in ('rvord', 'rd', 'p0', 'rcv', 'gravity')})

    def restart_state(self):
        # Carry the published state through the shared native pools and manager.
        return {}

    def restore_restart_state(self, state):
        if state:
            raise ValueError('Kessler state belongs to native pools')
