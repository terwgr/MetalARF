"""Boundary transaction around the existing MPAS v8.4.2 physics driver.

The driver and its hooks own scheme state. The pool snapshot alone is not a
complete physics restart; a caller must supply a driver whose hook restart
contract covers every active scheme and any session-owned service not listed
in stateful_services before using this for a forecast.
An error during restore or rollback leaves the transaction unusable; the caller
must discard its driver and pools instead of continuing the forecast.
"""
from __future__ import annotations


class MetalPhysicsTransaction:
    def __init__(self, driver, pools, recreate_driver, *, stateful_services=None):
        if not callable(recreate_driver):
            raise TypeError('rollback requires a driver recreation callback')
        self.driver, self.pools, self.recreate_driver = driver, pools, recreate_driver
        self.stateful_services = dict(stateful_services or {})
        self.phase = 'boundary'
        self._saved = None
        self._sources = None

    def checkpoint(self):
        if self.phase != 'boundary':
            raise RuntimeError('MPAS physics checkpoint requires a step boundary')
        pools = self.pools.snapshot()
        return {'driver': self.driver.restart_state(), 'pools': pools,
                'services': {name: service.restart_state() for name, service in self.stateful_services.items()}}

    def restore(self, checkpoint):
        if self.phase != 'boundary':
            raise RuntimeError('MPAS physics restore requires a step boundary')
        self.phase = 'failed'
        self.pools.restore(checkpoint['pools'])
        self.driver.restore_restart_state(checkpoint['driver'])
        self._restore_services(checkpoint)
        self.phase = 'boundary'

    def begin_step(self, *, itimestep, time_seconds, driver_pools):
        if self.phase != 'boundary':
            raise RuntimeError('MPAS physics step already active')
        self._saved = self.checkpoint()
        self.phase = 'begun'
        return self.driver.begin_step(itimestep=itimestep, time_seconds=time_seconds, pools=driver_pools)

    def sources(self, *, driver_pools):
        if self.phase != 'begun':
            raise RuntimeError('MPAS held sources require an active step')
        self._sources = self.driver.physics_get_tend(rk_step=1, dynamics_substep=1, pools=driver_pools)
        self.phase = 'sources'
        return self._sources

    def finish_step(self, *, driver_pools):
        if self.phase != 'sources':
            raise RuntimeError('MPAS post-transport physics requires held sources')
        result = self.driver.finish_step(pools=driver_pools)
        self.phase = 'finished'
        return result

    def commit(self):
        """Publish resident pools before the next native driver boundary."""
        if self.phase != 'finished':
            raise RuntimeError('MPAS commit requires post-transport physics')
        self.phase = 'failed'
        self.pools.publish()
        self.phase, self._saved, self._sources = 'boundary', None, None

    def rollback(self):
        if self.phase not in ('begun', 'sources', 'finished'):
            raise RuntimeError('no MPAS physics step available to roll back')
        self.phase = 'failed'
        self.pools.restore(self._saved['pools'])
        replacement = self.recreate_driver()
        replacement.restore_restart_state(self._saved['driver'])
        self._restore_services(self._saved)
        self.driver = replacement
        self.phase, self._saved, self._sources = 'boundary', None, None

    def _restore_services(self, checkpoint):
        states = checkpoint['services']
        if set(states) != set(self.stateful_services):
            raise ValueError('MPAS transaction service checkpoint set differs')
        for name, service in self.stateful_services.items():
            service.restore_restart_state(states[name])
