"""Initialize the bounded native MPAS Metal regional physics session."""
from __future__ import annotations

from copy import copy
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
from woof.hex.cuda_regional_forecast_v842 import RegionalNativeDycoreConfigV842
from woof.hex.lbc_stream import initial_paths
from woof.hex.metal_driver_v841 import assemble_regional_metal_dycore_v841
from woof.hex.metal_physics_prep_v841 import MetalMpasToPhysGeometryV841
from woof.hex.mpasphys.driver_v842 import resolve_namelist
from woof.hex.mpasphys.forecast_v842 import bind_limited_area, set_namelist_options
from woof.hex.mpasphys.metal_diagnostics_v842 import initialize_thermodynamics
from woof.hex.mpasphys.metal_native_coupling_v842 import MetalNativeCoupling
from woof.hex.mpasphys.metal_registry_v842 import (
    MetalSlotRegistry, require_metal_native_profile, resolve_metal_namelist,
)
from woof.hex.mpasphys.metal_session_v842 import MetalPhysicsSession, rrfs_slot_services
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.hex.mpasphys.metal_winds_v842 import MetalMpasWinds
from woof.hex.mpasphys.pool_v842 import decode_registry_pools
from woof.hex.mpasphys.route_table import DEFAULT_ROW, framework_entry, route_row
from woof.hex.mpasphys.runtime_v842 import (fold_lake_class, load_native_regional_inputs,
                                            native_mesh_state_exclusions, remap_limited_area_sentinels)
from woof.hex.mesh import Mesh
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from woof.hex.mpasphys.metal_resources_v842 import load_resources
from woof.hex.mpasphys.metal_noah_v842 import MetalMpasNoah
from woof.hex.mpasphys.metal_seaice_v842 import MetalMpasSeaice
from woof.hex.mpasphys.metal_bl_mynn_slot_v842 import MetalMpasBlMynn
from woof.hex.mpasphys.metal_thompson_column_v842 import MetalMpasThompson
from woof.hex.mpasphys.metal_gf_v842 import MetalMpasGrellFreitas
from woof.hex.mpasphys.metal_rrtmg_lw_slot_v842 import MetalMpasLongwave
from woof.hex.mpasphys.metal_rrtmg_sw_slot_v842 import MetalMpasShortwave
from woof.hex.mpasphys.metal_native_step_v842 import MetalNativeStep
from woof.hex.mpasphys.pool_v842 import NativePoolHydration
from woof.hex.mpasphys.moving_v842 import STATE_FIELDS, SAVED_FIELDS
from woof.hex.metal_containers_v841 import MetalPrognosticState, MetalSavedDiagnostics

@contextmanager
def initialize_native_metal(args, *, payload, prepared=None, reader=None, sources=None, namelist=None,
                            resource_packet=None, resource_cache=None, checkpoint_route=None, stream_intervals=None):
    """Build a bounded regional native session and its dycore stepper.

    Before a scheme runs, its diagnostic location words are initialized to zero;
    the native source does not establish their unused initial values. Active
    branches subsequently record their source I/J. On failure discard the
    session, pools, dycore, and stepper together. The final tuple contains the
    source mesh, grid and scalar names needed for checkpoint export. A prepared
    (layout, config, namelist) from the forecast runner reuses its decoded pool.
    """
    args = copy(args)
    if checkpoint_route is not None and checkpoint_route.carrier is not None:
        args.start = checkpoint_route.carrier['valid_time']
    payload = Path(payload)
    if prepared is None:
        if reader is None or sources is None or namelist is None:
            raise ValueError('native Metal startup requires reader, sources and namelist')
        text = bind_limited_area(args, namelist)
        if getattr(args, 'physics_interval', None) is not None:
            text = set_namelist_options(text, {'physics': {name: args.physics_interval for name in
                ('config_radtlw_interval', 'config_radtsw_interval', 'config_conv_interval')}})
        config = resolve_namelist(text, physics_mode=getattr(args, 'physics_mode', 'exact'),
                                  correct_skip=getattr(args, 'physics_correct_skip', None))
        if payload.exists():
            raise ValueError('regional payload destination must be new')
        source = framework_entry(config, 'registry_source')()
        layout = decode_registry_pools(reader, source, config, text, sources, payload, threads=2)
        if checkpoint_route is not None and checkpoint_route.carrier is not None:
            checkpoint_route.mark_restart(layout)
    else:
        layout, config, text = prepared
    require_metal_native_profile(config)
    table_key = getattr(config, 'table_key', DEFAULT_ROW)
    rrfs = table_key != DEFAULT_ROW
    microphysics = config.selectors["microphysics"]
    if (config.selectors['convection'] == 'cu_grell_freitas'
            and (config.options['config_gfconv_closure_deep'], config.options['config_gfconv_closure_shallow']) != (0, 8)):
        raise ValueError('native Metal GF requires the default closure selectors')
    # Verify both immediate source lineage links. Static is a complete mesh;
    # the existing regional loader overlays init and validates static -> init.
    Mesh.from_netcdf(args.grid, args.static, exclude_variables=native_mesh_state_exclusions(layout))
    regional_args = copy(args)
    regional_args.grid = args.static
    mesh, grid, profiles, state, reference, saved, names = load_native_regional_inputs(regional_args, layout, payload)
    options = {f.name: layout.configs[f.name] for f in fields(RegionalNativeDycoreConfigV842)
               if f.init and f.name in layout.configs}
    options.update(config_dt=args.dt, config_moist_physics=config.moist_physics, config_apply_lbcs=True)
    dycore_config = RegionalNativeDycoreConfigV842(**options)
    if rrfs:
        from woof.hex.dycore_variants import resolve_dycore_variant

        row = route_row(config)
        resolved = resolve_dycore_variant(dycore_config, row)
        overrides = {field.name: layout.configs[field.name] for field in fields(resolved)
                     if field.init and field.name in layout.configs}
        dycore_config = resolve_dycore_variant(dycore_config, row, overrides=overrides)
        unsupported = [name for name in ("config_scalar_eddy_mix", "config_rayleigh_damp_u",
                                         "config_mpas_cam_coef")
                       if getattr(dycore_config, name)]
        if unsupported:
            raise ValueError("native Metal RRFS does not implement " + ", ".join(unsupported))
    if (dycore_config.config_scalar_advection and config.moist_physics
            and (dycore_config.config_monotonic or dycore_config.config_positive_definite)):
        raise ValueError(
            "native Metal regional scalar advection requires unlimited transport "
            "(config_monotonic=false, config_positive_definite=false)")
    paths = initial_paths(args.lbc_dir)
    if not paths:
        raise ValueError('regional input requires actual LBC files')
    noah_seed = np.zeros(4, dtype=np.int32)
    seaice_seed = np.zeros(2, dtype=np.int32)
    hydration = NativePoolHydration(None, layout, payload)
    landuse = (hydration.records['sfc_input', 'mminlu']['text'].rstrip(' ')
               if config.moist_physics else None)

    with MetalContext(batch_kernel_calls=False) as context:
        driver = assemble_regional_metal_dycore_v841(context, mesh, state,
            grid.vertical_grid, reference, saved, grid.terrain_metrics, dycore_config,
            reference_wind_profiles=profiles, lbc_paths=paths,
            start_time=datetime.strptime(args.start, '%Y-%m-%d_%H:%M:%S'),
            scalar_names=tuple('lbc_' + name for name in names))
        if rrfs:
            from .rrfs.metal_resources_rrfs import load_resources as load_rrfs_resources

            resources = load_rrfs_resources(context, config=config, landuse=landuse,
                                            ozone_levels=layout.dimensions['nOznLevels'],
                                            packet=resource_packet)
            resources['slot_services'] = rrfs_slot_services(config, resources['slot_services'])
        elif config.moist_physics:
            tables = load_resources(context, landuse=landuse,
                ozone_levels=layout.dimensions['nOznLevels'], packet=resource_packet,
                cache=resource_cache, microphysics=microphysics)
            if microphysics == 'mp_kessler':
                from .metal_kessler_v842 import MetalMpasKessler
                microphysics_service = MetalMpasKessler()
            elif microphysics == 'mp_wsm6':
                from .metal_wsm6_v842 import MetalMpasWsm6
                microphysics_service = MetalMpasWsm6()
            elif microphysics == 'mp_thompson_aerosols':
                from .metal_thompson_aerosol_slot_v842 import MetalMpasThompsonAerosol
                microphysics_service = MetalMpasThompsonAerosol(tables['thompson'])
            else:
                microphysics_service = MetalMpasThompson(tables['thompson'])
            xp = MetalArrayModule(context)
            services = {('land_surface', 'sf_noah'): MetalMpasNoah(tables['noah']['tables'], xp.asarray(noah_seed)),
                ('seaice', 'sf_noah'): MetalMpasSeaice(context, module_locations=seaice_seed, i_start=1, j_index=1),
                ('microphysics', microphysics): microphysics_service,
                ('radiation_lw', 'rrtmg_lw'): MetalMpasLongwave(context, tables['longwave'], config=config),
                ('radiation_sw', 'rrtmg_sw'): MetalMpasShortwave(context, tables['shortwave'], config=config)}
            if config.selectors['pbl'] == 'bl_ysu':
                from .metal_ysu_slot_v842 import MetalMpasYsu
                from .metal_mm5_slot_v842 import MetalMpasMm5

                table_rows = 1001
                scheme = config.selectors['surface_layer']
                revised = scheme == 'sf_monin_obukhov_rev'
                mm5_tables = xp.zeros(((4 if revised else 6) * table_rows,), np.float32)
                table_kernel = 'hexv842_mm5_tables' if revised else 'hexv842_mm5_classic_tables'
                _launch(table_kernel, table_rows, context, dict(tables=mm5_tables),
                        dict(n=np.int32(table_rows)))
                services['pbl', 'bl_ysu'] = MetalMpasYsu(context)
                services['surface_layer', scheme] = MetalMpasMm5(context, mm5_tables, scheme)
            else:
                services['pbl', 'bl_mynn'] = MetalMpasBlMynn(context)
            if config.selectors['convection'] == 'cu_grell_freitas':
                services['convection', 'cu_grell_freitas'] = MetalMpasGrellFreitas()
            elif config.selectors['convection'] == 'cu_ntiedtke':
                from .metal_ntiedtke.service import MetalMpasNewTiedtke

                services['convection', 'cu_ntiedtke'] = MetalMpasNewTiedtke()
            resources = dict(slot_services=services, noah_resources=tables['noah'], ozone_tables=tables['ozone'])
        else:
            resources = dict(slot_services={})
        final_config = resolve_metal_namelist(text, slot_services=resources['slot_services'],
                                               physics_mode=config.mode, correct_skip=config.correct_skip,
                                               table_key=table_key)
        profile_name = config.options.get('hex_physics_profile')
        if profile_name is not None:
            from .profiles_v842 import finish_profile, resolve_profile
            final_config = finish_profile(
                profile_name, resolve_profile(profile_name, final_config.mode),
                final_config, backend='metal')
        if dict(final_config.execution_options) != dict(config.execution_options):
            raise ValueError('settled Metal options differ from decoded Registry options')
        config = final_config
        if checkpoint_route is not None:
            checkpoint_route.bind_inputs((mesh, grid, profiles, state, reference, saved, names), config)

        def factory(session):
            registry = session.driver_api.SlotRegistry() if rrfs else MetalSlotRegistry()
            session.register_slots(registry)
            # Native non-OPENACC microphysics pre/post bodies are empty.
            empty = lambda call: None
            external = {'pre_microphysics': empty, 'post_microphysics': empty}
            if not config.active('convection'):
                external['update_convection_step1'] = empty
            hooks = session.bind_hooks(external=external, save_fields=lambda: {},
                restore_fields=lambda state: None, after_call=lambda entry, call: None)
            manager = session.driver_api.PhysicsManager(config, args.dt, streams=stream_intervals)
            return session.driver_api.PhysicsDriver(config, manager, registry, hooks)

        session = MetalPhysicsSession(layout=layout, payload=payload, context=context,
            dt=args.dt, start=args.start, stop=args.stop, driver_factory=factory,
            config=config, **resources)
        try:
            fold_lake_class(session.native, layout)
            remap_limited_area_sentinels(session.native, layout)
            atmosphere = driver.atmosphere
            initialize_thermodynamics(atmosphere)
            pools = session.pools
            xp = pools.arrays

            def copy_solve(pool, name, value):
                target = pools.field(pool, name)
                if value.ndim != 2 or target.ndim != 2 or value.shape[0] != target.shape[0]:
                    raise ValueError(f'core field layout differs: {pool}.{name}')
                if value.shape[1] not in (target.shape[1], target.shape[1] + 1):
                    raise ValueError(f'core field extent differs: {pool}.{name}')
                _launch('hexv842_mpas_transport_rows_f32', target.size, context,
                    dict(src=value, dst=target, mapping=xp.asarray(np.array([0], np.int32))),
                    dict(nrows=np.int32(1), nlev=np.int32(target.shape[0]),
                         nsolve=np.int32(target.shape[1]), nfull=np.int32(value.shape[1]),
                         scatter=np.int32(0)))

            for name, value in [('rho_base', atmosphere.reference.rho_base),
                ('rtheta_base', atmosphere.reference.rho_theta_base),
                ('pressure_base', atmosphere.reference.pressure_base),
                ('exner_base', atmosphere.reference.exner_base)]:
                copy_solve('diag', name, value)
            copy_solve('mesh', 'zz', atmosphere.vertical.zz)
            copy_solve('mesh', 'dss', atmosphere.vertical.dss)
            copy_solve('mesh', 'edgeNormalVectors', xp.asarray(
                np.ascontiguousarray(mesh.arrays['edgeNormalVectors'].T, dtype=np.float32)))
            geometry = MetalMpasToPhysGeometryV841.from_host(context, driver.regional_v841.padded)
            coupling = MetalNativeCoupling(session, names, reconstruct_winds=MetalMpasWinds(geometry))
            coupling.synchronize(atmosphere, level=1)
            # Shared framework startup reads host pools and refreshes cached Metal
            # views, so publish the source core words before invoking it.
            pools.publish()
            session.initialize()
            if checkpoint_route is not None and checkpoint_route.carrier is not None:
                carrier = checkpoint_route.carrier
                if tuple(names) != tuple(carrier['scalar_names']):
                    raise ValueError('native restart scalar names differ')
                state = driver.atmosphere.state.to_host()
                saved = driver.atmosphere.saved.to_host()
                for target, nameset in ((state, STATE_FIELDS), (saved, SAVED_FIELDS)):
                    for name in nameset:
                        array, value = getattr(target, name), carrier['state'][name]
                        if array.shape != value.shape or array.dtype != value.dtype:
                            raise ValueError(f'native restart layout differs: {name}')
                        array[...] = value
                state.time_seconds = float(carrier['time_seconds'])
                driver.atmosphere = replace(driver.atmosphere,
                    state=MetalPrognosticState.from_host(context, state),
                    saved=MetalSavedDiagnostics.from_host(context, saved))
                checkpoint_route.restore_physics(session, layout, None, mesh)
            step_offset, seconds_offset = ((0, 0.0) if checkpoint_route is None
                                           else checkpoint_route.physics_clock_offset(args.dt))
            stepper = MetalNativeStep(session, driver, geometry, names,
                physics_step_offset=step_offset, physics_seconds_offset=seconds_offset)
            if microphysics == 'mp_thompson_aerosols' and not layout.configs['config_do_restart']:
                # MPAS feeds the startup scalar changes back into the dynamics
                # carrier before the first synchronization can overwrite them.
                stepper._copy_rows(pools.field('state', 'scalars', level=1),
                                   driver.atmosphere.state.scalars, stepper.indices, scatter=True)
            yield session, driver, stepper, (mesh, grid, names)
        finally:
            session.close()

