"""Native MPAS MM5 surface-layer slot and fractional ice exchange."""

import numpy as np

from woof.hex.metal_dynamics_v841 import _arrays
from woof.hex.mpasphys.sfclay_mm5_v842 import (
    CONSTANTS, FIELDS, INPUTS, OUTPUTS, SCHEMES, SEA_EXCHANGE, _MPAS_NAMES,
)


def bind_native_pools(provider, call):
    native = provider.native
    aliases = {'g': 'gravity', 'rovcp': 'rcp', 'r': 'R_d',
               'ep1': 'ep_1', 'ep2': 'ep_2', 'p1000mb': 'P0'}
    constants = provider.cp.asarray(
        [native.get_constant(aliases.get(name, name)) for name in CONSTANTS],
        dtype=np.float32,
    )
    profiles = {'ux': 'u_p', 'vx': 'v_p', 't1d': 't_p', 'qv1d': 'qv_p',
                'p1d': 'pres_hyd_p', 'dz8w1d': 'dz_p'}
    atmosphere = {name: provider.prepared(source)[0] for name, source in profiles.items()}
    atmosphere['psfcpa'] = provider.prepared('psfc_p')
    return dict(
        atmosphere=atmosphere, diag_physics=provider['diag_physics'],
        sfc_input=provider['sfc_input'], mesh_density=provider.field('mesh', 'meshDensity'),
        constants=constants, config_len_disp=native.get_real_config('config_len_disp'),
        xice_threshold=provider['scalar_state']['xice_threshold'],
        sea_exchange=({name: provider.prepared(name + '_sea') for name in SEA_EXCHANGE}
                      if call.options['config_frac_seaice'] else None),
    )


class MetalMpasMm5:
    bind_pools = staticmethod(bind_native_pools)
    corrections = frozenset({'RP3'})

    def __init__(self, context, tables, scheme):
        if scheme not in SCHEMES:
            raise ValueError('native MM5 surface-layer scheme required')
        self.context = context
        self.tables = tables
        self.scheme = scheme
        self.column_kernel = ('hexv842_mm5_revised' if scheme == 'sf_monin_obukhov_rev'
                              else 'hexv842_mm5_classic')

    def restart_state(self):
        return {}

    def restore_restart_state(self, payload):
        if payload:
            raise ValueError('MM5 state belongs to native pools')

    def submit(self, name, count, arrays, params):
        self.context.submit_kernel(name, arrays, params, exact_policy=False,
                                   grid_size=count, threads_per_group=128)

    def launch_slot(self, call):
        if (call.slot, call.scheme) != ('surface_layer', self.scheme):
            raise ValueError('configured MM5 surface-layer scheme required')
        pools = call.pools
        atmosphere, diag, surface = (pools[x] for x in ('atmosphere', 'diag_physics', 'sfc_input'))
        n = int(atmosphere['ux'].size)
        if not n:
            return diag
        frac = bool(call.options['config_frac_seaice'])
        vector_bytes = n * np.dtype(np.float32).itemsize
        field_bytes = len(FIELDS) * vector_bytes
        scratch_bytes = 2 * field_bytes + vector_bytes if frac else field_bytes
        if (field_bytes > self.context.max_buffer_length
                or scratch_bytes > self.context.recommended_working_set_size
                or len(FIELDS) * n > np.iinfo(np.int32).max):
            raise MemoryError('MM5 scratch exceeds Metal buffer or working-set limit')

        xp = _arrays(self.context)
        main = xp.empty((len(FIELDS), n), np.float32)
        for j, name in enumerate(OUTPUTS):
            if name in ('q2', 't2', 'th2', 'u10', 'v10', 'cd', 'cda', 'ck', 'cka'):
                main[j].fill(0)
            else:
                main[j][...] = diag[_MPAS_NAMES.get(name, name)]
        values = {**atmosphere, 'tsk': surface['skintemp'], 'xland': surface['xland'],
                  'pblh': diag['hpbl'], 'mavail': diag['mavail']}
        for name in INPUTS:
            if name not in ('dx', 'lakemask', 'water_depth'):
                main[FIELDS.index(name)][...] = values[name]
        sea = xp.empty((len(FIELDS), n), np.float32) if frac else main
        held = diag['regime'].copy() if frac else main[0]
        sst = surface['sst'] if frac else main[0]
        ice = surface['xice'] if frac else main[0]
        common = dict(n=np.int32(n), threshold=np.float32(pools['xice_threshold']))
        self.submit(
            'hexv842_mm5_prepare', n,
            dict(a=main, sea=sea, density=pools['mesh_density'], sst=sst, ice=ice),
            {**common, 'frac': np.int32(frac),
             'rp3': np.int32('RP3' in getattr(call, 'corrections', ())),
             'len_disp': np.float32(pools['config_len_disp'])},
        )
        options = dict(n=np.int32(n), isfflx=np.int32(1), isftcflx=np.int32(0),
                       iz0tlnd=np.int32(0), scm_force_flux=np.int32(0),
                       shalwater_z0=np.int32(0), optional_mask=np.int32(255))
        for fields in (main, sea) if frac else (main,):
            self.submit(self.column_kernel, n,
                        dict(a=fields, constants=pools['constants'], tables=self.tables), options)
        if frac:
            for name in SEA_EXCHANGE:
                pools['sea_exchange'][name][...] = sea[FIELDS.index(name)]
            self.submit('hexv842_mm5_blend', n,
                        dict(a=main, sea=sea, ice=ice, regime_hold=held), common)
        for j, name in enumerate(OUTPUTS):
            diag[_MPAS_NAMES.get(name, name)][...] = main[j]
        return diag
