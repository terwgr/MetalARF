"""Exercise cadence and a fresh-process carrier continuation of live native dynamics.

The default uses three steps and checkpoints after the first. The supplied
physics interval must cross a due/held/due sequence. This checks continuation
against the same Metal execution, not an independent CUDA forecast oracle.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import mkdtemp

import numpy as np
from tools.mpasphys_v842_metal.probe_real_startup import parse_args, real_startup
from woof.hex.metal_containers_v841 import MetalPrognosticState, MetalSavedDiagnostics
from woof.hex.mpasphys.checkpoint_v842 import CheckpointRoute
from woof.hex.mpasphys.metal_checkpoint_v842 import MetalCheckpointRoute
from woof.hex.mpasphys.metal_native_step_v842 import MetalNativeStep
from woof.hex.mpasphys.moving_v842 import load_carrier, STATE_FIELDS, SAVED_FIELDS


def compare(actual, expected, label='state'):
    if isinstance(expected, np.ndarray):
        if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
            different = int(np.count_nonzero(actual.ravel(order='C').view(np.uint8) != expected.ravel(order='C').view(np.uint8))) if actual.shape == expected.shape and actual.dtype == expected.dtype else None
            raise AssertionError(f'{label} differs: unequal bytes={different}')
        return expected.size
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f'{label} keys differ')
        return sum(compare(actual[key], value, f'{label}.{key}') for key, value in expected.items())
    if isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f'{label} sequence differs')
        return sum(compare(a, b, f'{label}[{i}]') for i, (a, b) in enumerate(zip(actual, expected)))
    if actual != expected:
        raise AssertionError(f'{label} differs: {actual!r} != {expected!r}')
    return 0


def dycore_state(driver):
    atmosphere = driver.atmosphere
    result = {name: getattr(atmosphere.state, name).to_numpy() for name in STATE_FIELDS}
    result.update({name: getattr(atmosphere.saved, name).to_numpy() for name in SAVED_FIELDS})
    if not all(np.isfinite(value).all() for value in result.values()):
        raise AssertionError('nonfinite native dycore state')
    return result


def compare_endpoint(current, expected, offset):
    words = compare(current['dycore'], expected['dycore'], 'dycore')
    words += compare(current['physics']['pools'], expected['physics']['pools'], 'physics.pools')
    for name, service in current['physics']['services'].items():
        if name != 'clock':
            words += compare(service, expected['physics']['services'][name], 'services.' + name)
    compare(current['physics']['services']['clock']['text'], expected['physics']['services']['clock']['text'], 'clock.text')
    compare(current['boundary'], expected['boundary'], 'boundary')
    compare(current['cadence'], expected['cadence'], 'cadence')
    from fractions import Fraction
    actual_manager = current['physics']['driver']['manager']
    expected_manager = expected['physics']['driver']['manager']
    for name, alarm in actual_manager['alarms'].items():
        if Fraction(alarm['next_ring']) + Fraction(str(offset)) != Fraction(expected_manager['alarms'][name]['next_ring']):
            raise AssertionError(f'manager phase differs: {name}')
    return words


def run(options, arguments):
    sys.argv = [sys.argv[0], *arguments]
    args = parse_args()
    if args.physics_mode != 'correct':
        raise ValueError('uninterrupted continuation comparison requires correct-mode alarm phase')
    route = MetalCheckpointRoute.from_route(CheckpointRoute())
    route.origin_time = args.start
    route.frame_from_cli, route.frame_seconds = True, args.dt
    if options.phase == 'resume':
        route.carrier = load_carrier(options.carrier)
        route.restart_metadata = json.loads(options.carrier.with_suffix(options.carrier.suffix + '.json').read_text())['checkpoint']
        route.offset_seconds = route.carrier['time_seconds']
        route.cli_restart = True
        args.start = route.carrier['valid_time']
    route.resolve_schedule(options.steps * args.dt, args.dt, [])
    with real_startup(args, checkpoint_route=route) as (session, driver, geometry, names):
        if options.phase == 'resume':
            if tuple(names) != tuple(route.carrier['scalar_names']):
                raise ValueError('native restart scalar names differ')
            # The sealed regional carriers include their dummy columns. Restore
            # those complete arrays after cold assembly, without padding twice.
            state = driver.atmosphere.state.to_host()
            saved = driver.atmosphere.saved.to_host()
            values = route.carrier['state']
            for name in STATE_FIELDS:
                target = getattr(state, name)
                if target.shape != values[name].shape or target.dtype != values[name].dtype:
                    raise ValueError(f'native restart dycore layout differs: {name}')
                target[...] = values[name]
            for name in SAVED_FIELDS:
                target = getattr(saved, name)
                if target.shape != values[name].shape or target.dtype != values[name].dtype:
                    raise ValueError(f'native restart saved layout differs: {name}')
                target[...] = values[name]
            state.time_seconds = float(route.carrier['time_seconds'])
            driver.atmosphere = replace(driver.atmosphere,
                state=MetalPrognosticState.from_host(driver.context, state),
                saved=MetalSavedDiagnostics.from_host(driver.context, saved))
            route.restore_physics(session, session.layout, None, route.metal_inputs[0])
            print('restored sealed live dycore and physics carrier', flush=True)
        coupling = MetalNativeStep(session, driver, geometry, names)
        cadence, endpoints, matched = [], {}, {}
        if options.phase == 'resume':
            with options.reference.open('rb') as stream:
                reference = pickle.load(stream)
        remaining = options.steps if options.phase == 'write' else options.steps - options.split
        for step in range(1, remaining + 1):
            coupling.step(step)
            if coupling.boundary_columns is None:
                raise AssertionError('correct-mode regional LB1 is not enabled')
            indices = coupling.boundary_columns.to_numpy()
            indices = indices[indices < session.layout.dimensions['nCellsSolve']]
            if not indices.size:
                raise AssertionError('regional LB1 has no solved boundary columns')
            for name in ('rthdynten', 'rqvdynten'):
                values = session.pools.field('tend_physics', name).to_numpy()
                if np.any(values[:, indices].view(np.uint32) != 0):
                    raise AssertionError(f'LB1 boundary forcing is nonzero: {name}')
            cadence.append({name: sum(call['operation'] == name for call in session.driver.calls)
                            for name in ('radiation_lw', 'radiation_sw', 'convection')})
            print(json.dumps({'phase': options.phase, 'step': step, 'cadence': cadence[-1]}), flush=True)
            global_step = step if options.phase == 'write' else step + options.split
            if global_step > options.split:
                regional = driver.regional_v841
                current = {'dycore': dycore_state(driver), 'physics': session.transaction.checkpoint(),
                    'cadence': cadence[-1], 'boundary': {
                        'clock': regional.clock.isoformat(),
                        'interval_end': regional.driving.interval_end.isoformat(),
                        'remaining_word': np.asarray([regional.driving.remaining_seconds(regional.clock, np.float32(0))], np.float32)}}
                if options.phase == 'write':
                    endpoints[global_step] = current
                else:
                    matched[global_step] = compare_endpoint(current, reference[global_step], route.offset_seconds)
            if options.phase == 'write' and step == options.split:
                mesh, grid, *_ = route.metal_inputs
                route.export(options.carrier, session=session, dycore=driver,
                    mesh=mesh, vertical=grid.vertical_grid, names=names,
                    layout=session.layout, step=step, dt=args.dt)
        if options.phase == 'write':
            expected_cadence = [{'radiation_lw': due, 'radiation_sw': due, 'convection': due} for due in (1, 0, 1)]
            if cadence != expected_cadence:
                raise AssertionError(f'expected due/held/due, got {cadence}')
            with options.reference.open('wb') as stream:
                pickle.dump(endpoints, stream, protocol=5)
        else:
            if driver.atmosphere.state.time_seconds != options.steps * args.dt:
                raise AssertionError('resumed global dycore time differs')
            print(json.dumps({'scope': 'live same-Metal fresh-process continuation across radiation and convection cadence',
                'equal_array_elements_by_endpoint': matched, 'resumed_cadence': cadence,
                'clock': session.native.clock_state.current_time.stamp(),
                'independent_forecast_parity': False}), flush=True)



def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phase', choices=('both', 'write', 'resume'), default='both')
    parser.add_argument('--carrier', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--split', type=int, default=1)
    options, arguments = parser.parse_known_args()
    if options.steps != 3 or options.split != 1:
        raise ValueError('this discriminating probe requires three steps with a first-step checkpoint')
    if options.phase == 'both':
        root = Path(mkdtemp(prefix='mpas-native-live-resume-'))
        print(json.dumps({'retained_probe_directory': str(root)}), flush=True)
        for phase in ('write', 'resume'):
            child = list(arguments)
            index = child.index('--payload') + 1
            child[index] = str(root / (phase + '-payload'))
            subprocess.run([sys.executable, '-m', __spec__.name, '--phase', phase,
                '--carrier', str(root / 'checkpoint.pkl'), '--reference', str(root / 'expected.pkl'),
                *child], check=True)

    else:
        if options.carrier is None or options.reference is None:
            raise ValueError('child phase requires carrier and reference paths')
        run(options, arguments)


if __name__ == '__main__':
    main()
