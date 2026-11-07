"""Run one native Metal step on the real regional startup inputs."""
import json
from collections import Counter
import numpy as np
from tools.mpasphys_v842_metal.probe_real_startup import parse_args, real_startup
from woof.hex.mpasphys.metal_native_step_v842 import MetalNativeStep
from woof.hex.mpasphys.metal_diagnostics_v842 import output_diagnostics
from woof.hex.mpasphys.metal_winds_v842 import MetalMpasWinds


def main():
    args = parse_args()
    if args.compare_source and args.physics_mode != 'exact':
        raise ValueError('live selected-source oracles currently cover exact mode only')
    with real_startup(args) as (session, driver, geometry, names):
        oracles = {}
        if args.compare_source:
            from tools.mpasphys_v842_metal.oracle_coupled_mynn import MynnOracle
            from tools.mpasphys_v842_metal.oracle_coupled_noah import NoahOracle
            from tools.mpasphys_v842_metal.oracle_coupled_thompson import ThompsonOracle
            for slot, scheme, oracle in (
                    ('pbl', 'bl_mynn', MynnOracle()),
                    ('land_surface', 'sf_noah', NoahOracle()),
                    ('microphysics', 'mp_thompson', ThompsonOracle(args.thompson_directory, driver.nlev))):
                service = session.slot_services[slot, scheme]
                launch = service.launch_slot
                if slot == 'land_surface':
                    service.launch_slot = lambda call, oracle=oracle, launch=launch, service=service: oracle.compare(call, launch, service)
                else:
                    service.launch_slot = lambda call, oracle=oracle, launch=launch: oracle.compare(call, launch)
                oracles[slot] = oracle
        start = driver.atmosphere.state.time_seconds
        before = driver.atmosphere.state.scalars.to_numpy()
        native_time = np.float32(session.native.get_field('state', 'Time', level=1))
        clock = session.native.clock_state.current_time.stamp()
        coupling = MetalNativeStep(session, driver, geometry, names)
        coupling.step(1)
        state, saved = driver.atmosphere.state, driver.atmosphere.saved
        if state.time_seconds != start + session.dt or session.transaction.phase != 'boundary':
            raise AssertionError('native step did not commit its outer time boundary')
        if np.float32(session.native.get_field('state', 'Time', level=1)) != np.float32(native_time + np.float32(session.dt)):
            raise AssertionError('native Registry time did not advance')
        if session.native.clock_state.current_time.stamp() == clock:
            raise AssertionError('native clock did not advance')
        arrays = {'rho': state.rho, 'rho_theta': state.rho_theta,
                  'scalars': state.scalars, 'theta_m': saved.theta_m,
                  'exner': saved.exner, 'pressure_p': saved.pressure_perturbation}
        counts = {}
        for name, array in arrays.items():
            host = array.to_numpy()
            if not np.isfinite(host).all():
                raise AssertionError(f'{name} contains nonfinite values')
            counts[name] = host.size
        theta, rho, pressure = output_diagnostics(driver.atmosphere, names)
        wind = MetalMpasWinds(geometry)(driver.atmosphere)
        endpoints = [('state', 'rho_zz', state.rho), ('state', 'theta_m', saved.theta_m),
            ('state', 'u', saved.normal_velocity), ('state', 'w', saved.vertical_velocity),
            ('diag', 'rho_p', saved.density_perturbation), ('diag', 'rtheta_p', saved.rho_theta_perturbation),
            ('diag', 'pressure_p', saved.pressure_perturbation), ('diag', 'exner', saved.exner),
            ('diag', 'uReconstructZonal', wind.u_p), ('diag', 'uReconstructMeridional', wind.v_p),
            ('diag', 'rho', rho), ('diag', 'theta', theta), ('diag', 'pressure', pressure)]
        checked = 0
        for pool, name, expected in endpoints:
            actual = session.pools.field(pool, name, level=1).to_numpy()
            want = expected.to_numpy()[..., :actual.shape[-1]]
            if not np.array_equal(actual.view(np.uint32), want.view(np.uint32)):
                raise AssertionError(f'published endpoint differs: {pool}.{name}')
            checked += actual.size
        actual = session.pools.field('state', 'scalars', level=1).to_numpy()
        scalar_rows = [names.index(name) for name in coupling.native_names]
        want = state.scalars.to_numpy()[scalar_rows, :, :actual.shape[-1]]
        if not np.array_equal(actual.view(np.uint32), want.view(np.uint32)):
            raise AssertionError('published endpoint differs: state.scalars')
        checked += actual.size
        boundary_count = 0
        if coupling.boundary_columns is not None:
            indices = coupling.boundary_columns.to_numpy()
            indices = indices[indices < session.layout.dimensions['nCellsSolve']]
            for name in ('rthdynten', 'rqvdynten'):
                values = session.pools.field('tend_physics', name).to_numpy()
                if np.any(values[:, indices].view(np.uint32) != 0):
                    raise AssertionError(f'LB1 boundary forcing is nonzero: {name}')
            boundary_count = int(indices.size)
        calls = [entry['operation'] for entry in session.driver.calls]
        counts_by_call = Counter(calls)
        required = ('physics_get_tend', 'post_transport_prepare', 'microphysics',
                    'cloudiness', 'radiation_lw', 'radiation_sw', 'pbl',
                    'surface_layer', 'land_surface', 'convection', 'seaice', 'gwdo')
        if any(counts_by_call[name] != 1 for name in required):
            raise AssertionError(f'unexpected native callback counts: {counts_by_call}')
        changed = int(np.count_nonzero(before.view(np.uint32) != state.scalars.to_numpy().view(np.uint32)))
        if not changed:
            raise AssertionError('native step did not change scalar words')
        print(json.dumps({'scope': 'one regional native Metal step on decoded inputs',
                          'time_seconds': state.time_seconds, 'finite_words': counts,
                          'endpoint_words_equal': checked, 'lb1_boundary_cells': boundary_count, 'driver_calls': dict(counts_by_call), 'changed_scalar_words': changed,
                          'selected_host_source': {name: {'calls': oracle.calls, 'words': oracle.words} for name, oracle in oracles.items()}}), flush=True)


if __name__ == '__main__':
    main()
