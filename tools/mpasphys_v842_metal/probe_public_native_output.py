"""Compare public native history and its sealed final checkpoint with saved endpoints."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
import numpy as np
from netCDF4 import Dataset, chartostring
from tools.mpasphys_v842_metal.probe_native_resume import compare
from woof.hex.mpasphys.moving_v842 import load_carrier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--origin-checkpoint', type=Path)
    args = parser.parse_args()
    with args.reference.open('rb') as stream:
        expected = pickle.load(stream)
    if args.origin_checkpoint is not None:
        origin = load_carrier(args.origin_checkpoint)
        metal = origin['physics']['metal']
        expected[1] = {'dycore': origin['state'], 'physics': {
            'pools': metal['pools'], 'services': metal['services'],
            'driver': {'manager': origin['physics']['manager']}}}
    receipt = json.loads((args.out / 'native-physics-receipt.json').read_text())
    manifest = json.loads((args.payload / 'manifest.json').read_text())
    rows = {row['name']: row for row in manifest['fields']}
    carrier = load_carrier(args.checkpoint)
    final_step = round(carrier['time_seconds'] / receipt['dt'])
    offset = final_step - receipt['steps_completed']
    endpoint = expected[final_step]
    elements = compare(carrier['state'], endpoint['dycore'], 'checkpoint.dycore')
    saved = carrier['physics']['metal']
    elements += compare(saved['pools'], endpoint['physics']['pools'], 'checkpoint.pools')
    for name, value in saved['services'].items():
        if name != 'clock':
            elements += compare(value, endpoint['physics']['services'][name], 'checkpoint.services.' + name)
    compare(saved['services']['clock']['text'], endpoint['physics']['services']['clock']['text'], 'checkpoint.clock.text')
    from fractions import Fraction
    elapsed = Fraction(carrier['physics']['manager_elapsed'])
    phase_offset = Fraction(str(carrier['time_seconds'])) - elapsed
    actual_manager = carrier['physics']['manager']
    expected_manager = endpoint['physics']['driver']['manager']
    for name, alarm in actual_manager['alarms'].items():
        if Fraction(alarm['next_ring']) + phase_offset != Fraction(expected_manager['alarms'][name]['next_ring']):
            raise AssertionError(f'public checkpoint alarm phase differs: {name}')
    history, compared_fields, skipped = {}, set(), []
    for record in receipt['streams']:
        index, step = record['record_index'], record['step'] + offset
        if step == 0:
            skipped.append({'record': record['record_index'], 'reason': 'initial first-call diagnostic frame has no retained endpoint reference'})
            continue
        with Dataset(record['path']) as dataset:
            if dataset.getncattr('backend') != 'metal':
                raise AssertionError('history backend metadata differs')
            wanted_time = (datetime.strptime(carrier['origin_time'], '%Y-%m-%d_%H:%M:%S') + timedelta(seconds=step * receipt['dt'])).strftime('%Y-%m-%d_%H:%M:%S')
            actual_time = str(chartostring(dataset.variables['xtime'][index:index + 1])[0]).rstrip(' \x00')
            if actual_time != wanted_time:
                raise AssertionError(f'history clock differs: {actual_time} != {wanted_time}')
            count = 0
            for name, variable in dataset.variables.items():
                if name in ('xtime', 'initial_time'):
                    continue
                variable.set_auto_mask(False)
                if name in carrier['scalar_names']:
                    row = rows['scalars']
                    value = expected[step]['physics']['pools']["state.scalars.1"][carrier['scalar_names'].index(name)]
                    dimensions = row['dimensions'][1:]
                else:
                    row = rows[name]
                    value = expected[step]['physics']['pools'][f"{row['pool']}.{row['name']}.1"]
                    dimensions = row['dimensions']
                target_dimensions = tuple(d for d in variable.dimensions if d != 'Time')
                selection = tuple(index if d == 'Time' else slice(None) for d in variable.dimensions)
                actual = np.asarray(variable[selection])
                value = np.asarray(value)[tuple(slice(0, dataset.dimensions[d].size) for d in dimensions)]
                if tuple(dimensions) != target_dimensions:
                    value = value.transpose(tuple(dimensions.index(d) for d in target_dimensions))
                count += compare(actual, value, f'history.{step}.{name}')
                compared_fields.add(name)
            history[step] = count
    if set(history) != set(range(offset + 1, final_step + 1)):
        raise AssertionError(f'missing held or due public history endpoint: {history}')
    manifest = {'checkpoint_dycore_fields': sorted(carrier['state']),
        'checkpoint_pool_fields': sorted(saved['pools']),
        'checkpoint_service_fields': sorted(saved['services']),
        'history_fields': sorted(compared_fields), 'excluded_history_records': skipped,
        'transformations': 'trim native dummy extents and permute declared dimensions; no numerical conversion',
        'clock_comparison': 'native text and absolute alarm phase; new segment start/stop and elapsed origins differ by design'}
    (args.out / 'native-metal-comparison.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'checkpoint_array_elements_equal': elements,
        'history_array_elements_equal_by_endpoint': history,
        'comparison': 'byte-identical to saved uninterrupted Metal endpoints; not independent CUDA parity'}), flush=True)


if __name__ == '__main__':
    main()
