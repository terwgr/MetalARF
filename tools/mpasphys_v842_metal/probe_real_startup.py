"""Regional native core and physics startup using real decoded input streams.

This stops before the first physics step. Location seeds are fixture inputs;
their unchanged values do not establish native module cold-state parity.
"""
from __future__ import annotations

import argparse
from copy import copy
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from woof.hex.cuda_regional_forecast_v842 import RegionalNativeDycoreConfigV842
from woof.hex.lbc_stream import initial_paths
from woof.hex.metal_driver_v841 import assemble_regional_metal_dycore_v841
from woof.hex.metal_physics_prep_v841 import MetalMpasToPhysGeometryV841
from woof.hex.mpasphys.driver_v842 import PhysicsDriver, PhysicsManager, resolve_namelist
from woof.hex.mpasphys.forecast_v842 import bind_limited_area, set_namelist_options
from woof.hex.mpasphys.framework_backend_v842 import registry_source
from woof.hex.mpasphys.metal_diagnostics_v842 import initialize_thermodynamics
from woof.hex.mpasphys.metal_native_coupling_v842 import MetalNativeCoupling
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry, resolve_metal_namelist
from woof.hex.mpasphys.metal_session_v842 import MetalPhysicsSession
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.hex.mpasphys.metal_winds_v842 import MetalMpasWinds
from woof.hex.mpasphys.pool_v842 import decode_registry_pools
from woof.hex.mpasphys.runtime_v842 import (fold_lake_class, load_native_regional_inputs,
                                            native_mesh_state_exclusions, remap_limited_area_sentinels)
from woof.hex.mesh import Mesh
from woof.metal.runtime import MetalContext
from tools.mpasphys_v842_metal.real_services import real_services


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--static', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--lbc-dir', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--noah-packet', type=Path, required=True)
    parser.add_argument('--thompson-directory', type=Path, required=True)
    parser.add_argument('--ozone-packet', type=Path, required=True)
    parser.add_argument('--noah-locations', type=int, nargs=4, required=True)
    parser.add_argument('--seaice-locations', type=int, nargs=2, required=True)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--stop', required=True)
    parser.add_argument('--physics-mode', choices=('exact', 'correct'), default='exact')
    parser.add_argument('--physics-interval')
    parser.add_argument('--compare-source', action='store_true', help='compare live scheme calls with selected host source')
    return parser.parse_args()


@contextmanager
def real_startup(args, checkpoint_route=None):
    text = bind_limited_area(args, "&physics\nconfig_physics_suite='convection_permitting'\n/\n")
    if args.physics_interval is not None:
        text = set_namelist_options(text, {'physics': {name: args.physics_interval for name in
            ('config_radtlw_interval', 'config_radtsw_interval', 'config_conv_interval')}})
    config = resolve_namelist(text, physics_mode=args.physics_mode)
    if args.payload.exists():
        raise ValueError('regional payload destination must be new')
    layout = decode_registry_pools(args.reader, registry_source(), config, text,
                                   args.source, args.payload, threads=2)
    if checkpoint_route is not None and checkpoint_route.carrier is not None:
        checkpoint_route.mark_restart(layout)
    # Verify both immediate source lineage links. Static is a complete mesh;
    # the existing regional loader overlays init and validates static -> init.
    Mesh.from_netcdf(args.grid, args.static, exclude_variables=native_mesh_state_exclusions(layout))
    regional_args = copy(args)
    regional_args.grid = args.static
    mesh, grid, profiles, state, reference, saved, names = load_native_regional_inputs(regional_args, layout, args.payload)
    options = {f.name: layout.configs[f.name] for f in fields(RegionalNativeDycoreConfigV842)
               if f.init and f.name in layout.configs}
    options.update(config_dt=args.dt, config_moist_physics=config.moist_physics, config_apply_lbcs=True)
    dycore_config = RegionalNativeDycoreConfigV842(**options)
    paths = initial_paths(args.lbc_dir)
    if not paths:
        raise ValueError('regional input requires actual LBC files')
    noah_seed = np.asarray(args.noah_locations, dtype=np.int32)
    seaice_seed = np.asarray(args.seaice_locations, dtype=np.int32)

    with MetalContext(batch_kernel_calls=False) as context:
        driver = assemble_regional_metal_dycore_v841(context, mesh, state,
            grid.vertical_grid, reference, saved, grid.terrain_metrics, dycore_config,
            reference_wind_profiles=profiles, lbc_paths=paths,
            start_time=datetime.strptime(args.start, '%Y-%m-%d_%H:%M:%S'),
            scalar_names=tuple('lbc_' + name for name in names))
        resources = real_services(context, config,
            noah_resources=json.loads(args.noah_packet.read_text()),
            thompson_resources={'table_directory': str(args.thompson_directory)},
            ozone_resources=json.loads(args.ozone_packet.read_text()),
            ozone_levels=layout.dimensions['nOznLevels'], noah_locations=noah_seed,
            seaice_locations=seaice_seed)
        final_config = resolve_metal_namelist(text, slot_services=resources['slot_services'],
                                               physics_mode=args.physics_mode)
        if dict(final_config.execution_options) != dict(config.execution_options):
            raise ValueError('settled Metal options differ from decoded Registry options')
        config = final_config
        if checkpoint_route is not None:
            checkpoint_route.metal_inputs = (mesh, grid, profiles, state, reference, saved, names)
            checkpoint_route.bind_inputs(checkpoint_route.metal_inputs, config)

        def factory(session):
            registry = MetalSlotRegistry()
            session.register_slots(registry)
            # Native non-OPENACC microphysics pre/post bodies are empty.
            empty = lambda call: None
            hooks = session.bind_hooks(external={'pre_microphysics': empty,
                'post_microphysics': empty}, save_fields=lambda: {},
                restore_fields=lambda state: None, after_call=lambda entry, call: None)
            return PhysicsDriver(config, PhysicsManager(config, args.dt), registry, hooks)

        session = MetalPhysicsSession(layout=layout, payload=args.payload, context=context,
            dt=args.dt, start=args.start, stop=args.stop, driver_factory=factory,
            config=config, **resources)
        try:
            lakes = fold_lake_class(session.native, layout)
            sentinels = remap_limited_area_sentinels(session.native, layout)
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
            operations = [entry['operation'] for entry in session.driver.calls]
            expected = ['physics_tables_init', 'physics_registry_init',
                        'physics_run_init', 'physics_init']
            if operations != expected:
                raise RuntimeError(f'unexpected initialization callbacks: {operations}')
            services = resources['slot_services']
            locations = {name: services[slot].restart_state()['module_locations'].tolist()
                         for name, slot in [('noah', ('land_surface', 'sf_noah')),
                                            ('seaice', ('seaice', 'sf_noah'))]}
            if locations != {'noah': noah_seed.tolist(), 'seaice': seaice_seed.tolist()}:
                raise RuntimeError(f'initialization changed location seeds: {locations}')
            print(json.dumps({'operations': operations, 'dimensions': layout.dimensions,
                'lake_cells_folded': lakes, 'remapped_sentinels': sentinels,
                'location_fixture_seeds_unchanged': locations,
                'identity': 'grid-static and static-init checked; original attributes retained',
                'scope': 'real regional core and framework startup; no forecast step'}), flush=True)
            yield session, driver, geometry, names
        finally:
            session.close()


def main():
    with real_startup(parse_args()):
        pass


if __name__ == '__main__':
    main()
