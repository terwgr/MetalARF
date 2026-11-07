"""Hydrated MPAS framework session for the unfinished Metal physics route.

A caller must supply a source-faithful driver and any selected services that
the session cannot bind from the native route. Provider-specific layout
preparation must already have been applied.
"""
from __future__ import annotations
from importlib import import_module
from .route_table import framework_entry, route_row
from .pool_v842 import NativePoolHydration
from .metal_pools_v842 import MetalPoolViews
from .metal_transaction_v842 import MetalPhysicsTransaction
from .metal_surface_v842 import MetalSurfaceService, MetalSurfaceServiceRRFS
from .metal_clock_v842 import MetalClockCheckpoint
from .metal_resident_v842 import MetalResidentPreparation
from .metal_ozone_v842 import MetalOzoneService
from .metal_coupling_v842 import MetalResidentCoupling
from .metal_handoff_v842 import MetalResidentTransport
import numpy as np


class _PreparedCheckpoint:
    def __init__(self, pools): self.pools = pools
    def restart_state(self): return self.pools.prepared_restart_state()
    def restore_restart_state(self, state): self.pools.restore_prepared_restart_state(state)


def rrfs_slot_services(config, supplied=None):
    """Bind resource-free fork slots from the selected route descriptors."""
    row = route_row(config)
    services = dict(supplied or {})
    scheme = config.selectors['gwdo']
    descriptor = row['slots']['gwdo']['cuda_entries'].get(scheme, {})
    if (descriptor.get('module') == 'ugwp_oro_v842'
            and not config.execution_options.get('config_ngw_scheme', False)
            and ('gwdo', scheme) not in services):
        from .rrfs.metal_ugwp_oro import MetalMpasUgwpOro
        services['gwdo', scheme] = MetalMpasUgwpOro()
    scheme = config.selectors['pbl']
    descriptor = row['slots']['pbl']['cuda_entries'].get(scheme, {})
    if scheme == 'bl_mynnedmf' and ('pbl', scheme) not in services:
        from .rrfs.metal_mynnedmf_slot import MetalMpasMynnEdmf, SOURCE_KERNELS
        module = descriptor.get('module')
        if module in SOURCE_KERNELS:
            services['pbl', scheme] = MetalMpasMynnEdmf(module)
    scheme = config.selectors['surface_layer']
    descriptor = row['slots']['surface_layer']['cuda_entries'].get(scheme, {})
    if (descriptor.get('module') == 'rrfs.sf_mynnsfc_v600'
            and ('surface_layer', scheme) not in services):
        from .rrfs.metal_sf_mynnsfc_v600 import MetalMpasSfMynnV600
        services['surface_layer', scheme] = MetalMpasSfMynnV600()
    return services


class MetalPhysicsSession:
    def __init__(self, *, layout, payload, context, dt, start, stop, driver_factory, resources=None, config=None,
                 ozone_tables=None, noah_resources=None, slot_services=None):
        if not callable(driver_factory):
            raise TypeError('MPAS Metal session requires an explicit physics driver factory')
        self.layout, self.dt, self.config = layout, float(dt), config
        self.slot_services = dict(slot_services or {})
        row = route_row(config)
        self.driver_api = import_module(row.get('driver_module', __package__ + '.driver_v842'))
        rrfs = self.driver_api.__name__ == __package__ + '.rrfs.driver_rrfs'
        if rrfs:
            self.slot_services = rrfs_slot_services(config, self.slot_services)
        self.native = framework_entry(config, 'backend')(layout=layout, resources=resources)
        self.hydration = NativePoolHydration(self.native, layout, payload)
        self.hydration.populate()
        self.native.clock(start=start, stop=stop, dt=dt, calendar=layout.configs['config_calendar_type'])
        self.native.streams()
        self.hydration.register()
        d = layout.dimensions
        self.native.mpas_atm_set_dims(nvertlevels_val=d['nVertLevels'], maxedges_val=d['maxEdges'],
                                      maxedges2_val=d['maxEdges2'], num_scalars_val=d['num_scalars'])
        self.pools = MetalPoolViews(self.native, layout, context)
        from .metal_slot_pools_v842 import MetalSlotPools
        self.slot_pools = MetalSlotPools(self)
        if noah_resources is not None:
            self.slot_pools.noah_v842_resources = noah_resources
        self.sources = {name: np.zeros((d['nVertLevels'], d[entity]+1), dtype=np.float32, order='F')
                        for name,entity in (('tend_ru_physics','nEdges'),
                                            ('tend_rtheta_physics','nCells'),('tend_rho_physics','nCells'))}
        self.arguments = self.driver_api.NativePoolArguments(self.native,
            pools={**self.sources, **{name:name for name in layout.pools | {'configs'}}},
            its=1, ite=d['nCellsSolve'], synchronize=lambda entry,call:self.pools.publish())
        self.calendar_callback = self.driver_api.NativeCalendarState(self.native, None, self.arguments)
        self.surface_service = (MetalSurfaceServiceRRFS if rrfs else MetalSurfaceService)(self)
        self.clock_checkpoint = MetalClockCheckpoint(self.native)
        self.resident_preparation = None if config is None else MetalResidentPreparation(self, config)
        self.resident_coupling = None if config is None else MetalResidentCoupling(self)
        self.resident_transport = None if config is None else MetalResidentTransport(self)
        self.ozone_service = None if ozone_tables is None else MetalOzoneService(self, ozone_tables)
        self.framework_startup = None
        if noah_resources is not None:
            if rrfs:
                from .rrfs.metal_framework_startup import MetalFrameworkStartupRRFS
                self.device_pools = self.slot_pools
                self.framework_startup = MetalFrameworkStartupRRFS(self, noah_resources)
            else:
                from .metal_framework_v842 import MetalFrameworkStartup
                self.framework_startup = MetalFrameworkStartup(self, noah_resources)
        self.precipitation_window = None
        if config is not None and config.correct('RP6') and config.active('land_surface'):
            from .metal_precip_window_v842 import MetalSurfacePrecipitationWindow
            self.precipitation_window = MetalSurfacePrecipitationWindow(self)
        self._driver_factory = driver_factory
        driver = driver_factory(self)
        services = {'surface':self.surface_service, 'clock':self.clock_checkpoint}
        services.update({slot + '/' + scheme: service
                         for (slot, scheme), service in self.slot_services.items()})
        if self.precipitation_window is not None:
            services['precipitation_window'] = self.precipitation_window
        if self.resident_preparation is not None:
            if config.moist_physics:
                services['prepared'] = _PreparedCheckpoint(self.pools)
            services['coupling'] = self.resident_coupling
            services['transport'] = self.resident_transport
        if self.ozone_service is not None:
            services['ozone'] = self.ozone_service
        self.transaction = MetalPhysicsTransaction(driver, self.pools, lambda: driver_factory(self),
            stateful_services=services)

    def bind_hooks(self, *, external, save_fields, restore_fields, after_call):
        """Bind actual Metal surface and calendar callbacks into native driver hooks.

        Remaining framework and scheme operations must be explicitly supplied
        by the caller; unsupported operations are never replaced by a no-op.
        """
        from .metal_surface_diagnostics_v842 import run_native_diagnostics
        callbacks = {'physics_timetracker_state': self.calendar_callback,
                     'atmphys_sfc_diagnostics': lambda call: run_native_diagnostics(self.slot_pools, call),
                     **self.surface_service.callbacks()}
        if self.resident_preparation is not None:
            callbacks['MPAS_to_physics'] = self.resident_preparation.prepare
            callbacks['physics_get_tend'] = self.resident_coupling.get_tend
            callbacks['post_transport_prepare'] = self.resident_transport.prepare
            # The resident arrays have one lifetime, as in ResidentPhysicsGlue.
            for stem in self.resident_preparation.allocation_stems():
                callbacks['allocate_' + stem] = lambda call: None
                callbacks['deallocate_' + stem] = lambda call: None
        if self.ozone_service is not None:
            callbacks['o3climatology_from_MPAS'] = self.ozone_service.update
        if self.framework_startup is not None:
            for operation in ('physics_tables_init', 'physics_registry_init', 'physics_run_init',
                              'physics_init', 'atmphys_finalize'):
                callbacks[operation] = getattr(self.framework_startup, operation)
        if self.precipitation_window is not None:
            callbacks.update(self.precipitation_window.operations())
        for service in self.slot_services.values():
            operations = getattr(service, 'callbacks', None)
            if operations is not None:
                callbacks.update(operations())
        callbacks.update(external)
        return self.driver_api.NativeDriverHooks(self.native, self.arguments, after_call=after_call,
            external=callbacks, save_fields=save_fields, restore_fields=restore_fields)

    def register_slots(self, registry):
        """Bind built-in preparation and explicit source-backed service instances.

        Services own their scheme-private restart state; the session supplies
        shared resident views. All other active slots remain required.
        """
        if self.resident_preparation is None:
            raise RuntimeError('MPAS Metal slot registration requires a driver config')
        from .metal_registry_v842 import metal_slot_definitions
        from dataclasses import replace
        for (slot, scheme), (implementation, corrections) in metal_slot_definitions(
                self.slot_services, self.config).items():
            if isinstance(implementation, str):
                callback = getattr(self.resident_preparation, implementation)
            else:
                def callback(call, service=implementation):
                    pools = self.slot_pools.for_slot(call)
                    binder = getattr(service, 'bind_pools', None)
                    if binder is not None:
                        pools = binder(pools, call)
                    return service.launch_slot(replace(call, pools=pools))
            if hasattr(registry, 'register_metal'):
                registry.register_metal(slot, scheme, callback, corrections=corrections)
            else:
                registry.register(slot, scheme, 'metal', callback, corrections=corrections)

    @property
    def driver(self):
        return self.transaction.driver

    def initialize(self):
        self.driver.initialize(self)
        if self.resident_preparation is not None and self.config.moist_physics:
            self.resident_preparation.activate()

    def close(self):
        self.native.close()
