"""Explicit initialization callbacks for the unfinished Metal framework.

The common registry, manager, initialization and finalization callbacks require
decoded Noah resources and preinitialized resource-owning Metal services.
Complete scheme wiring remains caller-bound.
Shared directions and metadata initialization still executes on the host.
"""
from .framework_init_v842 import FrameworkInitialization


class MetalFrameworkInitialization:
    def __init__(self, session):
        self.session = session

    def initialize_ozone(self, call):
        if self.session.ozone_service is None:
            raise RuntimeError('Metal ozone initialization requires decoded native tables')
        self.session.ozone_service.initialize()

    def o3climatology_from_mpas(self, call):
        if self.session.ozone_service is None:
            raise RuntimeError('Metal ozone update requires decoded native tables')
        return self.session.ozone_service.update(call)


class MetalFrameworkStartup(FrameworkInitialization):
    """Common framework callbacks with explicit native Noah resources.

    Scheme resources must already belong to explicit registered Metal services.
    """
    def __init__(self, session, noah_resources):
        super().__init__(session)
        if self.configs['config_o3climatology'] and session.ozone_service is None:
            raise RuntimeError('native physics initialization requires ozone tables')
        from .metal_noah_init_v842 import MetalNoahInitialization
        self.noah = MetalNoahInitialization(self, noah_resources,
                                            session.pools.arrays.context)
        self.ozone = MetalFrameworkInitialization(session)

    def physics_tables_init(self, call):
        """Record preinitialized resource owners in the native table-startup order.

        Services already bind their tables or compiled scheme arithmetic; this
        callback never invokes CUDA loaders or substitutes absent resources.
        """
        from ..physics_suites import load_table
        rows=load_table()['mpasphys_v842']['slots'];state={}
        for slot,scheme in self.session.config.selectors.items():
            if not self.session.config.active(slot):continue
            if rows[slot]['cuda_entries'][scheme].get('resources'):
                service=self.session.slot_services.get((slot,scheme))
                if service is None:
                    raise RuntimeError('native Metal table startup requires a resource-owning service: '+slot+'/'+scheme)
                state[scheme]=service
        self.native.scheme_state=state
        self.scalar('l_mp_tables',False)

    def initialize_landuse(self):
        self.noah.initialize_landuse()

    def initialize_noah(self):
        self.noah.initialize_noah()

    def initialize_ozone(self, call):
        self.ozone.initialize_ozone(call)

    def initialize_thompson_aerosols(self):
        from .metal_aerosol_start_v842 import initialize
        initialize(self)

    def physics_init(self, call):
        super().physics_init(call)
        # The common startup writes native host pools; refresh all cached
        # views before a resident scheme can read them.
        self.session.pools.refresh()
