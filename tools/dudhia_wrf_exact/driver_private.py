"""Strict PhysicsDriver probe using the shared upstream host solar path.

This probes the Python API; it does not claim a public CLI or WRF solar parity.
"""
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

os.environ['GPUWM_WRF_EXACT'] = '1'
from woof.core import physics
from woof.core.dudhia import DudhiaShortwaveRadiation, wrf_solar_geometry, _wrf_cssca
from woof.core.kernels import module_source
from tools.thompson_real_column_parity.host_backend import HostModule, CXX_FLAGS
from woof.core.grid import make_vertical_coord, make_base_state
from woof.core.state import init_at_rest
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.suites import configuration

original = MetalContext._submit_kernel_batch
def sync(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if result is not None: result.wait()
    return result
MetalContext._submit_kernel_batch = sync

with MetalContext() as context:
    xp = MetalArrayModule(context)
    cfg = replace(configuration((0, 0, 0, 0, 1, 'periodic'), sfclay_option=0),
                  radt=5.0/60, solar_generation='wrf_461', ra_sw_eclipse=0)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 300. + .003*np.asarray(z), cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base, array_module=xp)
    state.qv[...] = xp.asarray(np.full((cfg.nz, cfg.ny, cfg.nx), .012, np.float32))
    from woof.core.diagnostics import update_diagnostics
    update_diagnostics(state, cfg.hypsometric_opt)
    latitude = np.full((cfg.ny,cfg.nx), 35., np.float32)
    longitude = np.full_like(latitude, 0.)
    rad = DudhiaShortwaveRadiation(datetime(2026,1,1,12,tzinfo=timezone.utc), latitude, longitude)
    driver = physics.initialize_physics(state, cfg, glw=300., radiation=rad)
    reference = HostModule("driver_dudhia_reference", module_source("dudhia_sw")).get_function("dudhia_sw_column")
    resumed = None
    held = None
    temporary = TemporaryDirectory()
    for step in range(7):
        state.elapsed_seconds = float(step)
        if step in (0, 5):
            atmosphere = physics._prepare_atmosphere(state)
            def h(value): return value.to_numpy()
            shape, ncol = state.p.shape, cfg.ny*cfg.nx
            zeros = np.zeros(shape, np.float32)
            values = [h(atmosphere[name]) for name in ('temperature','pressure','qv','qc')]
            values += [h(value) if value is not None else zeros for value in (getattr(state, name, None) for name in ('qr', 'qi', 'qs', 'qg'))]
            values += [h(atmosphere['dz']), h(atmosphere['exner'])]
            mu, solcon = wrf_solar_geometry(rad.start_time + timedelta(seconds=step), latitude, longitude,
                                            hour_offset_seconds=2.5, generation='wrf_461')
            mu = np.asarray(mu, np.float32)
            albedo = h(driver.fields['albedo'])
            expected_heat = np.zeros(shape, np.float32)
            expected_gsw = np.zeros((cfg.ny,cfg.nx), np.float32)
            expected_sw = np.zeros_like(expected_gsw)
            reference(((ncol+127)//128,), (128,), (*values, mu, albedo, expected_heat, expected_gsw, expected_sw,
                      np.float32(solcon), _wrf_cssca(rad.swrad_scat), np.int32(rad.icloud),
                      np.int32(1), np.int32(ncol), np.int32(cfg.nz), np.int64(1),
                      np.int64(-ncol), np.int64((cfg.nz-1)*ncol)))
        driver.compute(state, cfg)
        mass = np.add(np.multiply(state.c1h.to_numpy()[:, None, None],
                                  state.total_mu().to_numpy()[None], dtype=np.float32),
                      state.c2h.to_numpy()[:, None, None], dtype=np.float32)
        rate = expected_heat if step in (0, 5) else np.divide(held, mass, dtype=np.float32)
        coupled = np.multiply(mass, rate, dtype=np.float32)
        held = np.multiply(rate, mass, dtype=np.float32)
        np.testing.assert_array_equal(driver.radiation_tendencies.rtheta.to_numpy().view(np.uint32),
                                      coupled.view(np.uint32))
        np.testing.assert_array_equal(driver._wrf_rthraten.to_numpy().view(np.uint32), held.view(np.uint32))
        np.testing.assert_array_equal(driver.tendencies.rtheta.to_numpy().view(np.uint32),
                                      coupled.view(np.uint32))
        assert driver.call_counts['radiation'] == (1 if step < 5 else 2)
        if step in (0, 5):
            for actual, expected in ((driver.rthratensw, expected_heat),
                                     (driver.fields['swdown'], expected_sw)):
                np.testing.assert_array_equal(actual.to_numpy().view(np.uint32), expected.view(np.uint32))
        if step == 2:
            from woof.io.restart import write_restart, restore_restart
            path = Path(temporary.name) / 'checkpoint.npz'
            write_restart(path, state, cfg)
            resumed = init_at_rest(cfg, coord, base, array_module=xp)
            update_diagnostics(resumed, cfg.hypsometric_opt)
            rad2 = DudhiaShortwaveRadiation(rad.start_time, latitude, longitude)
            driver2 = physics.initialize_physics(resumed, cfg, glw=300., radiation=rad2)
            restore_restart(path, resumed, cfg)
            for name in ('rthratenlw', 'rthratensw', 'rthraten', '_wrf_rthraten'):
                np.testing.assert_array_equal(getattr(driver2, name).to_numpy().view(np.uint32),
                                              getattr(driver, name).to_numpy().view(np.uint32))
        elif resumed is not None:
            resumed.elapsed_seconds = float(step)
            driver2.compute(resumed, cfg)
            for name in ('rthratenlw', 'rthratensw', 'rthraten', '_wrf_rthraten'):
                np.testing.assert_array_equal(getattr(driver2, name).to_numpy().view(np.uint32),
                                              getattr(driver, name).to_numpy().view(np.uint32))
        if resumed is not None:
            for name in ('swdown', 'glw'):
                np.testing.assert_array_equal(driver2.fields[name].to_numpy().view(np.uint32),
                                              driver.fields[name].to_numpy().view(np.uint32))
        print('step', step+1, 'calls', driver.call_counts['radiation'],
              'carrier_coupled', driver._wrf_rthraten_coupled,
              'heating', float(driver.rthraten.to_numpy()[0,0,0]), flush=True)
    print('Dudhia CUDA host flags', CXX_FLAGS, '; shared host solar exercised, not independently verified')
