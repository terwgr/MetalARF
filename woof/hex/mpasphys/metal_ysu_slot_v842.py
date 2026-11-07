"""Native MPAS v8.4.2 YSU using prepared profiles and Registry pools."""

import numpy as np

from woof.hex.metal_dynamics_v841 import _arrays, _context, _require
from woof.hex.mpasphys.metal_tendencies_v842 import _launch


class MetalMpasYsu:
    corrections = frozenset()

    def __init__(self, context):
        self.context = context

    def restart_state(self):
        return {}

    def restore_restart_state(self, payload):
        if payload:
            raise ValueError('YSU state belongs to native pools')

    def launch_slot(self, call):
        if (call.slot, call.scheme) != ('pbl', 'bl_ysu'):
            raise ValueError('YSU slot requires native bl_ysu')
        pools = call.pools
        prepared = pools['prepared']
        diag = pools['diag_physics']
        tend = pools['tend_physics']
        surface = pools['sfc_input']
        raw = pools['constants']
        flags = pools['scalar_state']
        nz = int(pools['nVertLevels'])
        nc = int(pools['nCellsSolve'])
        if nz < 3 or nc < 1 or not flags['f_qc'] or not flags['f_qi']:
            raise ValueError('bounded native YSU slot requires active qc/qi, columns and at least three levels')
        ctx = self.context
        xp = _arrays(ctx)
        word_bytes = np.dtype(np.float32).itemsize
        # Ten full profiles and ten surface vectors remain live with the workspace.
        profile_bytes = nz * nc * word_bytes
        scratch_bytes = 10 * (profile_bytes + nc * word_bytes)
        workspace_per_column = 61 * (nz + 2) * word_bytes
        # The shader indexes both profiles and workspace with signed words.
        index_limit = np.iinfo(np.int32).max
        workspace_budget = min(ctx.recommended_working_set_size - scratch_bytes,
                               ctx.max_buffer_length, index_limit * word_bytes)
        tile = min(nc, workspace_budget // workspace_per_column)
        if (tile < 1 or profile_bytes > ctx.max_buffer_length
                or (nz + 1) * nc > index_limit):
            raise MemoryError('YSU scratch exceeds Metal buffer or working-set limit')
        if _context(prepared['t_p']) is not ctx:
            raise ValueError('YSU prepared arrays belong to another context')

        profiles = {
            'dz8w2d': 'dz_p', 'pi2d': 'pi_p', 'tx': 't_p', 'qvx': 'qv_p',
            'qcx': 'qc_p', 'qix': 'qi_p', 'p2d': 'pres_hyd_p',
            'p2di': 'pres2_hyd_p', 'ux': 'u_p', 'vx': 'v_p',
        }
        a = {name: prepared[src][:nz + int(name == 'p2di')] for name, src in profiles.items()}
        for name in ('znt', 'ust', 'hfx', 'qfx', 'wspd', 'br', 'u10', 'v10'):
            a[name] = diag[name].copy() if name in ('u10', 'v10') else diag[name]
        a.update(psim=diag['fm'], psih=diag['fh'], xland=surface['xland'], psfcpa=prepared['psfc_p'])
        if call.options['config_ysu_pblmix']:
            dz = a['dz8w2d'].to_numpy()
            if (not np.all(np.isfinite(dz) & (dz > 0))
                    or not np.all(np.cumsum(dz, axis=0, dtype=np.float32)[-1] > np.float32(500))):
                raise ValueError('YSU top-down requires finite positive layers and an interface above 500 m')
        for name, value in a.items():
            shape = (nz + int(name == 'p2di'), nc) if name in profiles else (nc,)
            _require(name, value, np.float32, shape)
        for name in ('rthratenlw', 'rthratensw'):
            _require(name, tend[name], np.float32, (nz, nc))

        a['rthraten'] = xp.empty((nz, nc), np.float32)
        _launch('hexv842_mpas_ysu_radiation_f32', nz * nc, ctx,
                dict(lw=tend['rthratenlw'], sw=tend['rthratensw'], rthraten=a['rthraten']),
                dict(count=np.int32(nz * nc)))
        a['ctopo'] = xp.ones((nc,), np.float32)
        a['ctopo2'] = xp.ones((nc,), np.float32)
        a['uox'] = xp.zeros((nc,), np.float32)
        a['vox'] = xp.zeros((nc,), np.float32)
        for name in ('utnp', 'vtnp', 'ttnp', 'qvtnp', 'qctnp', 'qitnp', 'exch_hx', 'exch_mx'):
            a[name] = xp.zeros((nz, nc), np.float32)
        for name in ('hpbl', 'wstar', 'delta'):
            a[name] = xp.zeros((nc,), np.float32)
        a['kpbl1d'] = xp.zeros((nc,), np.int32)
        aliases = {'g': 'gravity', 'rovcp': 'rcp', 'rovg': 'rdg', 'rd': 'R_d',
                   'rv': 'R_v', 'ep1': 'ep_1', 'ep2': 'ep_2'}
        names = ('cp', 'g', 'rovcp', 'rovg', 'rd', 'xlv', 'rv', 'ep1', 'ep2', 'karman')
        constants = {name: np.float32(raw[aliases.get(name, name)]) for name in names}
        a['workspace'] = xp.empty((61 * (nz + 2) * tile,), np.float32)
        for offset in range(0, nc, tile):
            count = min(tile, nc - offset)
            _launch('hexv842_mpas_ysu_column_f32', count, ctx, a,
                    dict(ncol=np.int32(nc), offset=np.int32(offset), count=np.int32(count),
                         nz=np.int32(nz), f_qc=np.int32(1), f_qi=np.int32(1),
                         topdown=np.int32(bool(call.options['config_ysu_pblmix'])),
                         dt=np.float32(call.dt), **constants))
        theta = xp.empty((nz, nc), np.float32)
        _launch('hexv842_mpas_ysu_theta_f32', nz * nc, ctx,
                dict(ttnp=a['ttnp'], pi=a['pi2d'], rthblten=theta), dict(count=np.int32(nz * nc)))
        result = {'rublten': a['utnp'], 'rvblten': a['vtnp'], 'rthblten': theta,
                  'rqvblten': a['qvtnp'], 'rqcblten': a['qctnp'], 'rqiblten': a['qitnp']}
        for name, value in result.items():
            tend[name][...] = value
        for target, src in (('hpbl', 'hpbl'), ('kpbl', 'kpbl1d'), ('kzh', 'exch_hx'),
                            ('kzm', 'exch_mx'), ('kzq', 'exch_hx'), ('delta', 'delta'),
                            ('wstar', 'wstar')):
            diag[target][...] = a[src]
        diag['exch_h'].fill(0)
        return result
