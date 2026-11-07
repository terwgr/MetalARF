"""Bounded native Metal forecast using the shared MPAS door and stream writer.

Scientific input reconstruction, Registry control and stream encoding are host
work. Resident dynamics and selected native schemes execute on Metal. This
route does not turn allocated diagnostics into computed output authorities.
"""
from __future__ import annotations

from copy import copy
from datetime import datetime
from fractions import Fraction
import json
import sys
from pathlib import Path

import numpy as np


# Each admitted option is consumed here or by the shared native door. The
# ordinary Metal row has its own option consumers; both use the same refusal
# helper so explicit unsupported requests cannot disappear silently.
OPTIONS = frozenset({
    'backend', 'physics_backend', 'physics_suite', 'physics_pin_set', 'physics_profile',
    'physics_namelist', 'physics',
    'physics_mode', 'physics_correct_skip', 'physics_resources',
    'native_pool_reader', 'mpas_source', 'grid', 'static', 'init', 'lbc_dir',
    'orographic_statistics', 'start_time', 'hours', 'dt',
    'time_integration_order', 'acoustic_substeps', 'monotonic',
    'streams', 'stream_list_directory', 'history_every_minutes',
    'checkpoint_hours', 'restart_from', 'out', 'scratch', 'receipt',
    'init_source', 'repo', 'stop_on_refusal', 'wind_farms',
    'output_format', 'output_config', 'icechunk_store',
})

_UNAVAILABLE_STREAM_FIELDS = frozenset({
    'relhum', 'refl10cm', 'refl10cm_max', 'refl10cm_1km', 'refl10cm_1km_max',
})


def run_forecast(arguments, config, text):
    from ..metal_forecast import _refuse_unimplemented
    from ..forecast_door import (_refuse, bind_native_step, bind_native_dynamics)
    from .forecast_v842 import run_forecast as shared_forecast
    from .metal_registry_v842 import require_metal_native_profile
    from .route_table import DEFAULT_ROW
    from .. import numerics
    _refuse_unimplemented(arguments, _refuse, implemented=OPTIONS)
    if numerics.is_strict():
        raise _refuse('Metal Hex strict numerics has no complete arithmetic contract')
    if getattr(config, 'table_key', DEFAULT_ROW) != DEFAULT_ROW:
        try:
            require_metal_native_profile(config)
        except ValueError as error:
            raise _refuse(str(error)) from error
    if 'hex_physics_profile' in config.options:
        from .metal_bl_mynn_slot_v842 import validate_mynn_options
        validate_mynn_options(config.options)
    required = ('grid', 'static', 'init', 'lbc_dir') if getattr(arguments, 'lbc_dir', None) is not None else ('grid', 'init')
    for name in required:
        value = getattr(arguments, name, None)
        if value is None or not Path(value).exists():
            raise _refuse(f'native Metal requires an existing --{name.replace("_", "-")}')
    text = bind_native_dynamics(arguments, bind_native_step(arguments, text))
    return shared_forecast(arguments, config, text, runtime=run_dynamics)


def run_dynamics(arguments, config, layout, payload, streams, library,
                 start, stop, dt, steps, out, *, moving=None, namelist):
    from .metal_forecast_v842 import initialize_native_metal
    from .driver_v842 import interval_seconds
    from .route_table import route_row
    from .streams_output_v842 import PhysicsStreams, _FailedStartReflectivity

    class MetalPhysicsStreams(PhysicsStreams):
        def _capture_views(self):
            # Metal publishes native host pools before either capture.
            return None

        @property
        def first_frame_from_first_call(self):
            return getattr(self, "_first_frame_from_first_call", False)

        @first_frame_from_first_call.setter
        def first_frame_from_first_call(self, value):
            self._first_frame_from_first_call = bool(value)

        def _capture(self, selected):
            self.session.pools.publish()
            return super()._capture(selected)

        def _start_reflectivity(self, snapshot):
            try:
                from .metal_refl_start_v842 import start_frame_reflectivity
                pending = start_frame_reflectivity(self.layout.configs, snapshot,
                                                   context=self.session.pools.arrays.context,
                                                   constants=self.session.slot_pools['constants'])
                if pending is not None:
                    return pending
            except Exception as error:
                return _FailedStartReflectivity(error)
            return _FailedStartReflectivity(RuntimeError("start-frame reflectivity is not implemented on Metal"))
    from .. import metal_v841_launch
    if library is not None:
        raise ValueError('native Metal uses the resident Python framework, not a native CPU physics library')
    wsm6 = config.selectors['microphysics'] == 'mp_wsm6'
    kessler = config.selectors['microphysics'] == 'mp_kessler'
    ysu = config.selectors['pbl'] == 'bl_ysu'
    outputs = [s for s in streams if 'output' in s.directions and s.output_interval != 'none']
    intervals = None
    if ysu and any(s.name in ('output', 'diagnostics') for s in outputs):
        raise NotImplementedError('native Metal YSU/MM5 requires state streams without the MPAS diagnostic alarm')
    missing = _UNAVAILABLE_STREAM_FIELDS.intersection(
        name for s in outputs if s.name != 'restart' for name in s.fields)
    microphysics_diagnostics = bool(missing) and config.selectors['microphysics'] in (
        'mp_thompson', 'mp_thompson_aerosols', 'mp_wsm6', 'mp_tempo')
    if microphysics_diagnostics:
        missing = frozenset()
    if kessler:
        if any(s.name == 'output' for s in outputs):
            raise NotImplementedError('native Metal Kessler uses the diagnostics stream for its MPAS alarm')
        period = next((interval_seconds(s.output_interval) for s in outputs
                       if s.name == 'diagnostics'), None)
        if period is not None:
            if period % Fraction(str(dt)):
                raise NotImplementedError('native Metal Kessler diagnostics cadence must align with the timestep')
            missing -= {'relhum'}
    if missing:
        raise NotImplementedError(
            'native Metal output lacks computed authority for ' + ', '.join(sorted(missing)))
    if wsm6 or ysu or kessler or microphysics_diagnostics:
        intervals = {s.name: (s.input_interval if s.name == 'surface' else s.output_interval)
                     for s in streams}
    if moving is not None:
        from .checkpoint_v842 import CheckpointRoute
        from .metal_checkpoint_v842 import MetalCheckpointRoute
        if not isinstance(moving, CheckpointRoute) or moving.operator or moving.control is not None or moving.shock:
            raise ValueError('native Metal moving-mesh execution is not implemented')
        moving = MetalCheckpointRoute.from_route(moving)
        moving.begin(start)
    args = copy(arguments)
    args.start, args.stop, args.dt = start, stop, dt
    args.physics_mode = config.mode
    # Public mode/skip resolution has already consumed command-line and
    # namelist options. The factory receives those exact resolved choices.
    args.physics_correct_skip = tuple(config.correct_skip)
    sources = [Path(value) for key in ('grid', 'static', 'init', 'orographic_statistics')
               if (value := getattr(args, key, None)) is not None]
    receipt = {'schema': 'mpas-native-metal-forecast/v1', 'backend': 'metal',
        'start': start, 'stop': stop, 'dt': dt, 'steps_completed': 0,
        'streams': [], 'init_source': getattr(args, 'init_source', None),
        'host_computation': ['scientific input reconstruction and Registry control',
                             'table decoding and transport', 'calendar and monthly interpolation weights',
                             'longwave layer counts and column grouping',
                             'output capture and Rust encoding',
                             'checkpoint serialization and identity hashing'],
        'diagnostic_location_initialization': 'deterministic zero before first source branch; no scientific consumer'}
    if ysu and config.options['config_ysu_pblmix']:
        receipt['host_computation'].append('YSU top-down prepared vertical extent validation')
    if config.selectors['convection'] == 'cu_ntiedtke':
        receipt['host_computation'].append('New Tiedtke scalar timestep and downdraft status validation')
    with metal_v841_launch.preserve_post_launch():
        with initialize_native_metal(args, reader=None, sources=sources, payload=Path(payload),
                namelist=namelist, resource_packet=getattr(args, 'physics_resources', None),
                checkpoint_route=moving, prepared=(layout, config, namelist),
                stream_intervals=intervals) as (session, driver, stepper, inputs):
            receipt['physics'] = session.config.receipt()
            print(f'native Metal physics mode {session.config.mode}; applied corrections: '
                  f'{", ".join(sorted(session.config.corrections)) or "none"}', file=sys.stderr, flush=True)
            for gap in session.config.gaps:
                print('native Metal correction gap: ' + json.dumps(gap.as_dict()),
                      file=sys.stderr, flush=True)
            request = out / 'native-physics-request.json'
            plan = json.loads(request.read_text())
            plan.update(backend='metal', physics=receipt['physics'], identity_checked=True)
            request.write_text(json.dumps(plan, indent=2) + '\n')
            mesh, grid, names = inputs
            if config.selectors['microphysics'] == 'mp_tempo':
                # driver_rrfs.bind_native_output_diagnostics: stream-selected hourly radar.
                included = any('output' in s.directions and 'refl10cm_1km_max' in s.fields
                               for s in streams)
                session.pools.field('diag_physics', 'do_diag_dbz')[...] = np.int32(included)
            row = route_row(session.config)
            resets = row.get('diagnostic_resets', ('refl10cm_1km_max',))
            reset_values = row.get('diagnostic_reset_values', {})
            from .metal_output_diagnostics_v842 import MetalOutputDiagnostics
            diagnostics = (MetalOutputDiagnostics(session, driver)
                           if config.selectors['microphysics'] == 'mp_tempo' and any(
                               MetalOutputDiagnostics.produced.intersection(s.fields) for s in outputs) else None)
            output = MetalPhysicsStreams(session, streams, out, diagnostics=diagnostics, final_time=datetime.strptime(stop, '%Y-%m-%d_%H:%M:%S'))
            output.first_frame_from_first_call = session.config.mode == 'correct' and moving is None
            session.output_streams = output
            from ..wind_farm import attach_metal_native
            wind = attach_metal_native(arguments, session, output, dt=dt)
            if wind is not None:
                receipt['wind_farms'] = json.loads(wind.metadata())
            output.attrs['backend'] = 'metal'
            receipt['output_capture_transport'] = 'published native host pool words; synchronous Metal download'
            try:
                if moving is not None and moving.resume_continues:
                    from .checkpoint_v842 import CLOCK_START_WORDS
                    continuation = moving.carrier['physics'].get('continuation') or {}
                    clock_start = continuation.get('clock_start') or {}
                    expected_clock = set(CLOCK_START_WORDS) if session.config.moist_physics else set()
                    if set(clock_start) != expected_clock:
                        raise ValueError('Metal correct-mode restart requires the checkpoint physics start hour')
                    if continuation.get('scheme_flags'):
                        raise ValueError('Metal restart cannot restore foreign scheme SAVE flags')
                    if continuation.get('output_accumulators') and diagnostics is None:
                        raise ValueError('Metal restart cannot restore output diagnostic accumulators')
                    for name, value in clock_start.items():
                        table = session.native.get_table(name)[0]
                        if np.asarray(value).shape != table.shape or np.asarray(value).dtype != table.dtype:
                            raise ValueError(f'Metal restart physics clock layout differs at {name}')
                    receipt['checkpoint_continuation'] = moving.restore_continuation(
                        session=session, diagnostics=diagnostics, dt=dt)
                for step in range(steps + 1):
                    if step:
                        stepper.step(step)
                        if step == 1:
                            session.pools.publish()
                            output.complete_first_frame()
                        receipt['steps_completed'] = step
                    diagnostics_written = output.write(step=step, dt=dt, receipt=receipt)
                    if step and diagnostics_written:
                        for name in resets:
                            key = ('diag_physics', name)
                            if key in session.native._fields:
                                session.pools.fill(*key, np.float32(reset_values.get(name, 0)))
                        session.pools.publish()
                    if moving is not None:
                        moving.after_step(step, session=session, dycore=driver, mesh=mesh,
                            vertical=grid.vertical_grid, names=names, layout=layout,
                            dt=dt, steps=steps, columns=None, out=out)
                    if step:
                        with (out / 'physics-calls.jsonl').open('a') as stream:
                            stream.write(json.dumps({'step': step, 'calls': session.driver.calls}) + '\n')
                if moving is not None:
                    moving.finish(step=steps, session=session, dycore=driver, mesh=mesh,
                        vertical=grid.vertical_grid, names=names, layout=layout, dt=dt)
                    receipt['checkpoint'] = moving.receipt()
            finally:
                output.close()
            receipt['output_metadata'] = output.metadata_receipt
            receipt['physics'] = session.config.receipt()
    target = out / 'native-physics-receipt.json'
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    named = getattr(arguments, 'receipt', None)
    if named is not None:
        Path(named).write_text(target.read_text())
    print(json.dumps({'status': 'complete', 'backend': 'metal',
                      'steps': receipt['steps_completed'], 'receipt': str(target)}), flush=True)
    return 0
