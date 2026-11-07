"""Native MPAS-A v8.4.2 WSM6 microphysics on resident Metal pools."""

import numpy as np

from woof.hex.metal_dynamics_v841 import _arrays, _context, _require
from woof.hex.mpasphys import wsm6_v842 as native
from woof.hex.mpasphys.driver_v842 import slot_corrections
from woof.hex.mpasphys.metal_tendencies_v842 import _launch


_STATE_WORDS = 89
_SURFACE_FIELDS = frozenset((
    'surface_pressure', 'tend_sfc_pressure', 'precipw', 'rainnc', 'rainncv',
    'snownc', 'snowncv', 'graupelnc', 'graupelncv', 'sr',
    'refl10cm_max', 'refl10cm_1km', 'refl10cm_1km_max',
))


class MetalMpasWsm6:
    corrections = frozenset({'WB3', 'WB5'})

    def __init__(self):
        self._cache = None

    def bind_pools(self, provider, call):
        top = int(provider['tend_physics']['mp_top_level'].to_numpy().reshape(-1)[0])
        return native.bind_native(provider, call, writeback_levels=top)

    def launch_slot(self, call):
        if (call.slot, call.scheme) != ('microphysics', 'mp_wsm6'):
            raise ValueError('native WSM6 microphysics slot required')
        pools = call.pools
        fields, options = pools['fields'], pools['options']
        corrections = slot_corrections(call, native.CORRECT_MODE_SOURCE)

        constants = np.asarray(options['host_constants'], np.float32)
        if constants.shape != (26,) or not np.all(np.isfinite(constants)):
            raise ValueError('WSM6 requires 26 finite native constants')
        hail = options['hail_opt']
        if hail not in (0, 1):
            raise ValueError('invalid WSM6 hail selector')
        nz, nc = fields['theta_m'].shape
        solve, top = int(options['solve_cells']), int(options['mp_top_level'])
        if (not 2 <= nz <= native.MAX_LEVELS or not 0 <= solve <= nc
                or nc < 1 or not 1 <= top <= nz or 10 * nz * nc >= 2**31):
            raise ValueError('invalid WSM6 native extents')
        dt = np.float32(options['dt_microp'])
        dt_dyn = np.float32(options['dt_dyn'])
        if not np.isfinite(dt) or not np.isfinite(dt_dyn) or min(dt, dt_dyn) <= 0:
            raise ValueError('WSM6 requires positive finite physics and dynamics timesteps')

        context = _context(fields['theta_m'])
        arrays = {name: fields[name] for name in native.FIELD_NAMES}
        for name, value in arrays.items():
            shape = (nz + 1, nc) if name == 'zgrid' else (nc,) if name in _SURFACE_FIELDS else (nz, nc)
            _require('WSM6 ' + name, value, np.float32, shape)
        counter, device_constants = fields['i_rainnc'], pools['constants']
        _require('WSM6 i_rainnc', counter, np.int32, (nc,))
        _require('WSM6 constants', device_constants, np.float32, (26,))
        if any(_context(value) is not context or not value.flags.c_contiguous
               for value in (*arrays.values(), counter, device_constants)):
            raise ValueError('WSM6 requires contiguous arrays in the same Metal context')
        if not solve:
            return

        # Ten 3-level planes, 98 level planes and the source column I/O scratch.
        workspace_words = 138 * nz + 331
        sizes = (nc * workspace_words * 4, 10 * nz * nc * 4, 7 * nc * 4,
                 3 * nz * nc * 4, 2 * nz * nc * 4, nz * nc * 4,
                 _STATE_WORDS * nc * 4, nc * 4, nc * 4)
        if (max(sizes) > context.max_buffer_length
                or sum(sizes) > context.recommended_working_set_size):
            raise MemoryError('WSM6 workspace exceeds context allocation limits')
        key = context, nz, nc, constants.tobytes(), int(hail)
        if self._cache is None or self._cache[0] != key:
            xp = _arrays(context)
            work = dict(
                v=xp.empty((10, nz, nc), np.float32),
                s=xp.empty((7, nc), np.float32),
                r=xp.empty((3, nz, nc), np.float32),
                prod=xp.empty((2, nz, nc), np.float32),
                pi=xp.empty((nz, nc), np.float32),
                workspace=xp.empty((nc * workspace_words,), np.float32),
                state=xp.empty((_STATE_WORDS * nc,), np.float32),
                status=xp.empty((nc,), np.int32),
            )
            hail_by_column = xp.asarray(np.full(nc, hail, np.int32))
            _launch('hexv842_wsm6_state_f32', nc, context,
                    dict(constants=device_constants, hail=hail_by_column,
                         state=work['state'], status=work['status']),
                    dict(count=np.int32(nc)))
            self._cache = key, work

        work = self._cache[1]
        common = dict(nz=np.int32(nz), nc=np.int32(nc), ns=np.int32(solve),
                      rv_rd=np.float32(constants[4] / constants[3]))
        packed = {name: work[name] for name in ('v', 's', 'r', 'prod')}
        _launch('hexv842_wsm6_from_f32', nc, context,
                dict(arrays, **packed, pi=work['pi']), common)
        column_kernel = 'hexv842_wsm6_column_wb5_f32' if 'WB5' in corrections else 'hexv842_wsm6_column_f32'
        _launch(column_kernel, nc, context,
                dict(v=work['v'], s=work['s'], pi=work['pi'], constants=device_constants,
                     state=work['state'], workspace=work['workspace'], status=work['status']),
                dict(nz=common['nz'], nc=common['nc'], ns=common['ns'], wrapper=np.int32(1),
                     workspace_words=np.int32(workspace_words), dt=dt))
        if all(options[name] == 1 for name in ('has_reqc', 'has_reqi', 'has_reqs')):
            _launch('hexv842_wsm6_radii_f32', nc, context,
                    dict(v=work['v'], pi=work['pi'], constants=device_constants,
                         state=work['state'], radii=work['r']),
                    dict(nz=common['nz'], nc=common['nc'], ns=common['ns'],
                         wrapper=np.int32(1), enabled=np.int32(1)))
        if options['l_diags']:
            from .metal_wsm6_radar_v842 import launch_radar, relative_humidity
            radar_fields = {name: work['v'][index] for name,index in
                            (('th',0),('qv',1),('qr',4),('qs',5),('qg',6),('p',8))}
            radar_fields.update(pii=work['pi'],zgrid=fields['zgrid'])
            launch_radar(radar_fields,fields,device_constants,state=work['state'],
                         solve_cells=solve,running_max=True)
            relative_humidity(radar_fields,fields['relhum'],solve_cells=solve)
        writeback_kernel = 'hexv842_wsm6_to_wb3_f32' if 'WB3' in corrections else 'hexv842_wsm6_to_f32'
        _launch(writeback_kernel, nc, context,
                dict(arrays, **packed, c=device_constants, i_rainnc=counter),
                dict(common, top=np.int32(top), dt_dyn=dt_dyn, rd=constants[3],
                     rvord=np.float32(options['rvord']), rcv=np.float32(options['rcv']),
                     p0=np.float32(options['p0']), g=constants[0],
                     bucket=np.float32(options['bucket_rainnc']),
                     acrain=np.int32(options['l_acrain'])))

    def restart_state(self):
        # Initialization depends only on the resolved constants and hail option.
        return {}

    def restore_restart_state(self, state):
        if state:
            raise ValueError('WSM6 state belongs to native pools')
