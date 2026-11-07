"""The retained RRFS framework startup order with Metal scheme owners."""

import numpy as np

from ..metal_framework_v842 import MetalFrameworkInitialization
from ..metal_noah_init_v842 import MetalNoahInitialization
from ..route_table import route_row
from .framework_init_rrfs import FrameworkInitializationRRFS
from .metal_landuse_rrfs import initialize_landuse_rrfs
from .metal_ruc_init import initialize_ruc
from .metal_startup_pools import prepare_pools
from .metal_tempo_slot import MetalMpasTempo
from .metal_tempo_startup import initialize_tempo_aerosols
from .tempo_v303 import CONFIG_ARGUMENTS


class MetalFrameworkStartupRRFS(FrameworkInitializationRRFS):
    """Use the fork's registry, manager and scheme initialization callbacks."""

    def __init__(self, session, noah_resources):
        super().__init__(session)
        if self.configs['config_o3climatology'] and session.ozone_service is None:
            raise RuntimeError('RRFS Metal startup requires its ozone tables')
        self.noah = MetalNoahInitialization(self, noah_resources, session.pools.arrays.context)
        self.ozone = MetalFrameworkInitialization(session)

    def physics_tables_init(self, call):
        if self.native.get_text_field('sfc_input', 'mminlu') != self.noah.resources['landuse']:
            raise ValueError('RRFS land-use resources differ from the native classification')
        rows = route_row(self.session.config)['slots']
        state = {}
        for slot, scheme in self.session.config.selectors.items():
            if not self.session.config.active(slot):
                continue
            descriptor = rows[slot].get('cuda_entries', {}).get(scheme, {})
            if not descriptor.get('resources'):
                continue
            service = self.session.slot_services.get((slot, scheme))
            if service is None:
                raise RuntimeError(f'RRFS Metal startup requires a resource owner for {slot}/{scheme}')
            if (slot, scheme) == ('microphysics', 'mp_tempo'):
                flags = tuple(bool(self.configs[name]) for name in CONFIG_ARGUMENTS)
                if (not isinstance(service, MetalMpasTempo) or service.tables is None
                        or service.flags != flags or service._table_sha256 is None
                        or service._table_pin_set != self.native.pin_set
                        or service.tables.buffer.context is not self.session.pools.arrays.context):
                    raise ValueError('RRFS Metal TEMPO startup requires matching authenticated table words')
            state[scheme] = service
        # The retained registry and startup callbacks read native storage.
        self.session.pools.publish()
        self.native.scheme_state = state
        self.scalar('l_mp_tables', True)

    def _refresh_startup_field(self, pool, name):
        cached = self.session.pools.buffers.get((pool, name, 1))
        if cached is not None:
            host, device, selection = cached
            device[...] = np.ascontiguousarray(host[selection]).reshape(device.shape)

    def physics_registry_init(self, call):
        super().physics_registry_init(call)
        scheme = self.configs['config_lsm_scheme'].strip()
        if not self.configs['config_do_restart'] and scheme in ('sf_noah', 'sf_noahmp', 'sf_ruc'):
            self._refresh_startup_field('sfc_input', 'dzs')

    def prepare_pools(self):
        prepare_pools(self.session)

    def initialize_landuse(self):
        self.noah.initialize_landuse()

    def initialize_landuse_rrfs(self, julday):
        initialize_landuse_rrfs(self, self.noah)

    def initialize_ozone(self, call):
        self.ozone.initialize_ozone(call)

    def o3climatology_from_mpas(self, call):
        return self.ozone.o3climatology_from_mpas(call)

    def initialize_ruc(self):
        initialize_ruc(self.session)

    def initialize_noah(self):
        self.noah.initialize_noah()

    def initialize_convection(self):
        super().initialize_convection()
        if not self.configs['config_do_restart'] and self.configs['config_convection_scheme'].strip() == 'cu_kain_fritsch':
            self._refresh_startup_field('diag_physics', 'nca')

    def initialize_microphysics(self):
        scheme = self.configs['config_microp_scheme'].strip()
        if scheme != 'mp_tempo':
            raise NotImplementedError(f'RRFS Metal startup has no initializer for {scheme}')
        if self.native.scheme_state.get(scheme) is not self.session.slot_services['microphysics', scheme]:
            raise RuntimeError('RRFS Metal TEMPO requires table initialization before physics startup')
        initialize_tempo_aerosols(self.session, self.configs)

    def initialize_ngw(self):
        raise NotImplementedError('RRFS Metal startup has no nonorographic gravity-wave initializer')
