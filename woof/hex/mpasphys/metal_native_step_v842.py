"""Resident native MPAS physics around an actual Metal dycore step.

After any failed step, discard and recreate the session, its pools, the
dycore and this wrapper. Pool publication and dycore publication are not an
atomic cross-object commit.
"""
from dataclasses import replace
import os
import numpy as np
from ..metal_dynamics_v841 import _add, _context, _require
from .metal_native_coupling_v842 import MetalNativeCoupling
from .metal_tendencies_v842 import MetalMpasSourcesV842, _launch
from .metal_winds_v842 import MetalMpasWinds


class MetalNativeStep:
    def __init__(self, session, dycore, wind_geometry, scalar_names, *, physics_step_offset=0, physics_seconds_offset=0.0):
        self.session, self.dycore = session, dycore
        self.physics_step_offset = physics_step_offset
        self.physics_seconds_offset = physics_seconds_offset
        self.names = tuple(scalar_names)
        self.coupling = MetalNativeCoupling(session, self.names,
                                           reconstruct_winds=MetalMpasWinds(wind_geometry))
        self.native_names = self.coupling.native_names
        self.indices = tuple(self.names.index(name) for name in self.native_names)
        self.held = None
        self.boundary_columns = self._boundary_columns()

    def _boundary_columns(self):
        regional = getattr(self.dycore, 'regional_v841', None)
        if regional is None or self.session.config.mode != 'correct' or os.environ.get('WOOF_HEX_MPAS_LB1', '1') == '0':
            return None
        mask = np.asarray(regional.masks_host.bdy_mask_cell, dtype=np.int32)
        if mask.ndim != 1 or mask.size + 1 != self.dycore.ncells:
            raise ValueError('regional boundary mask differs from padded cell extent')
        indices = np.flatnonzero(np.append(mask > 0, True)).astype(np.int32)
        return self.dycore.arrays.asarray(indices)

    def _copy_rows(self, source, target, mapping, *, scatter):
        """Copy solved columns; the dycore may also have a dummy column."""
        context = _context(source)
        _require('source', source, np.float32)
        _require('target', target, np.float32)
        if _context(target) is not context:
            raise ValueError('native step arrays belong to different Metal contexts')
        nlev = self.dycore.nlev
        full = target.shape[-1] if scatter else source.shape[-1]
        solved = source.shape[-1] if scatter else target.shape[-1]
        nrows = len(mapping)
        expected_source = (nrows, nlev, solved) if scatter and source.ndim == 3 else None
        if source.ndim not in (2, 3) or target.ndim != source.ndim or solved not in (full, full - 1):
            raise ValueError('native step source and target extents differ')
        if source.shape[-2] != nlev or target.shape[-2] != nlev:
            raise ValueError('native step vertical extents differ')
        if source.ndim == 2 and mapping != (0,):
            raise ValueError('native step scalar-plane mapping differs')
        if expected_source is not None and source.shape != expected_source:
            raise ValueError('native step source scalar rows differ')
        if not scatter and target.ndim == 3 and target.shape[0] != nrows:
            raise ValueError('native step target scalar rows differ')
        mapped_rows = target.shape[0] if scatter and target.ndim == 3 else source.shape[0] if source.ndim == 3 else 1
        if any(i < 0 or i >= mapped_rows for i in mapping):
            raise ValueError('native step scalar row mapping is out of range')
        if not nrows:
            return
        indices = self.dycore.arrays.asarray(np.asarray(mapping, dtype=np.int32))
        _launch('hexv842_mpas_transport_rows_f32', nrows * nlev * solved, context,
                dict(src=source, dst=target, mapping=indices),
                dict(nrows=np.int32(nrows), nlev=np.int32(nlev), nsolve=np.int32(solved),
                     nfull=np.int32(full), scatter=np.int32(scatter)))

    def _padded(self, source, shape, mapping=(0,)):
        target = self.dycore.arrays.zeros(shape, np.float32)
        self._copy_rows(source, target, mapping, scatter=True)
        return target

    def _stage(self, *, rk_step, dynamics_substep, state, saved, diagnostics):
        if self.held is None:
            if (rk_step, dynamics_substep) != (1, 1):
                raise RuntimeError('native sources require the first RK stage')
            pools = self.session.pools
            self._copy_rows(state.rho, pools.field('state', 'rho_zz', level=2), (0,), scatter=False)
            self._copy_rows(diagnostics.h_edge, pools.field('diag', 'rho_edge'), (0,), scatter=False)
            sources = self.session.transaction.sources(driver_pools=self.session.slot_pools)
            diabatic = (pools.field('tend', 'rt_diabatic_tend')
                        if self.session.config.moist_physics else None)
            self.held = MetalMpasSourcesV842(
                self._padded(sources.rho, state.rho.shape),
                self._padded(sources.rho_u, saved.normal_velocity.shape),
                self._padded(sources.rho_theta, state.rho.shape),
                self._padded(sources.scalars, state.scalars.shape, self.indices),
                diabatic_theta=None if diabatic is None else self._padded(diabatic, state.rho.shape))
        return self.held

    def _post_transport(self, *, state, saved, outer_state, modified_theta_dynamics_rate):
        transport = self.session.resident_transport
        config = self.dycore.config
        def synchronize(current, carried):
            candidate = replace(self.dycore.atmosphere, state=current, saved=carried)
            self.coupling.synchronize(candidate, level=2)
        if not self.session.config.moist_physics:
            if modified_theta_dynamics_rate is not None:
                self._copy_rows(modified_theta_dynamics_rate,
                    self.session.pools.field('tend_physics', 'rthdynten'), (0,), scatter=False)
            self.session.transaction.finish_step(driver_pools=self.session.slot_pools)
            return
        transport.bind(current=state, carried=saved, outer=outer_state,
            rate=modified_theta_dynamics_rate, held=self.held,
            scalar_advection=config.config_scalar_advection, monotonic=config.config_monotonic,
            positive_definite=config.config_positive_definite,
            convection=self.session.config.selectors['convection'],
            scalar_names=self.names, synchronize=synchronize,
            boundary_columns=self.boundary_columns)
        self.session.transaction.finish_step(driver_pools=self.session.slot_pools)
        if self.session.config.active('microphysics'):
            pools = self.session.pools
            self._copy_rows(pools.field('state', 'scalars', level=2), state.scalars, self.indices, scatter=True)
            for target, pool, name in ((saved.theta_m, 'state', 'theta_m'),
                                       (saved.exner, 'diag', 'exner'),
                                       (saved.rho_theta_perturbation, 'diag', 'rtheta_p'),
                                       (saved.pressure_perturbation, 'diag', 'pressure_p')):
                self._copy_rows(pools.field(pool, name, level=2 if pool == 'state' else 1), target, (0,), scatter=True)
            state.rho_theta[...] = saved.rho_theta_perturbation
            _add(self.dycore, state.rho_theta, self.dycore.atmosphere.reference.rho_theta_base)

    def step(self, itimestep):
        """Run one outer step; discard the session and dycore on any error."""
        if self.held is not None:
            raise RuntimeError('native Metal step already active')
        self.coupling.synchronize(self.dycore.atmosphere, level=1)
        transaction = self.session.transaction
        transaction.begin_step(itimestep=itimestep + self.physics_step_offset,
            time_seconds=self.physics_seconds_offset + float(self.session.native.clock_state.elapsed), driver_pools=self.session.slot_pools)
        result = self.dycore.step_device_native(physics_stage_callback=self._stage,
            physics_post_transport_callback=self._post_transport,
            physics_mass_loading_count=self.session.layout.indices['state', 'moist_end'])
        self.coupling.synchronize(result, level=1)
        self.session.native.update_state_time(self.session.dt, source_level=1, target_level=1)
        self.session.native.advance_clock()
        transaction.commit()
        self.held = None
        return result
