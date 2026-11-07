"""Initialize the native default profile from decoded real MPAS inputs.

The six location words are explicit diagnostic fixture seeds, not evidence of
native cold-state parity. This exercises framework initialization only; it is
not a forecast or a source-comparison claim. The payload must have been made
by the existing native Registry reader for the same sources and namelist.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np

from woof.hex.mpasphys.driver_v842 import PhysicsDriver, PhysicsManager, resolve_namelist
from woof.hex.mpasphys.framework_backend_v842 import registry_source
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.hex.mpasphys.metal_session_v842 import MetalPhysicsSession
from woof.hex.mpasphys.pool_v842 import RegistryLayout
from woof.metal.runtime import MetalContext
from tools.mpasphys_v842_metal.real_services import real_services


TEXT = "&physics\nconfig_physics_suite='convection_permitting'\n/\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--noah-packet', type=Path, required=True)
    parser.add_argument('--thompson-directory', type=Path, required=True)
    parser.add_argument('--ozone-packet', type=Path, required=True)
    parser.add_argument('--noah-locations', type=int, nargs=4, required=True)
    parser.add_argument('--seaice-locations', type=int, nargs=2, required=True)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--stop', required=True)
    args = parser.parse_args()

    abi = subprocess.check_output([str(args.reader), '--abi'], text=True).strip()
    if abi != 'gpuwm-rw-mpas-physics-pool-v1':
        raise ValueError('unexpected native Registry reader ABI')
    dimensions = {}
    for source in args.source:
        dimensions.update(json.loads(subprocess.check_output(
            [str(args.reader), '--inspect', str(source)], text=True)))
    config = resolve_namelist(TEXT)
    config = replace(config, backends={**config.backends, **{
        slot: 'metal' for slot in config.selectors if config.active(slot)}})
    layout = RegistryLayout(registry_source(), config, TEXT, dimensions)
    noah_seed = np.asarray(args.noah_locations, dtype=np.int32)
    seaice_seed = np.asarray(args.seaice_locations, dtype=np.int32)
    noah_packet = json.loads(args.noah_packet.read_text())
    ozone_packet = json.loads(args.ozone_packet.read_text())

    with MetalContext(batch_kernel_calls=False) as context:
        resources = real_services(context, config, noah_resources=noah_packet,
            thompson_resources={'table_directory': str(args.thompson_directory)},
            ozone_resources=ozone_packet, ozone_levels=layout.dimensions['nOznLevels'],
            noah_locations=noah_seed, seaice_locations=seaice_seed)

        def factory(session):
            registry = MetalSlotRegistry()
            session.register_slots(registry)
            # These native hooks have empty bodies without MPAS_OPENACC.
            empty = lambda call: None
            hooks = session.bind_hooks(external={'pre_microphysics': empty,
                'post_microphysics': empty}, save_fields=lambda: {},
                restore_fields=lambda state: None, after_call=lambda entry, call: None)
            return PhysicsDriver(config, PhysicsManager(config, args.dt), registry, hooks)

        session = MetalPhysicsSession(layout=layout, payload=args.payload, context=context,
            dt=args.dt, start=args.start, stop=args.stop, driver_factory=factory,
            config=config, **resources)
        try:
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
                'active_selectors': {slot: scheme for slot, scheme in config.selectors.items()
                                     if config.active(slot)},
                'location_fixture_seeds_unchanged': locations,
                'scope': 'real-payload framework initialization; no forecast parity'}), flush=True)
        finally:
            session.close()


if __name__ == '__main__':
    main()
