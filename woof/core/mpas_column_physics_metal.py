"""A bounded Metal implementation of the generic MPAS column contract."""
from __future__ import annotations

from woof.core.mpas_column_batch_metal import MetalMpasColumnBatchPhysics
from dataclasses import fields
from types import MappingProxyType
from woof.core.mpas_column_suite import column_physics_plan, MpasColumnSuiteTendencies

from woof.core.mpas_column_metal_profiles import supports_physics


class MetalMpasColumnPhysics(MetalMpasColumnBatchPhysics):
    """Generic identity and inventory over the equivalent resident batch."""

    def __init__(self, *, physics, surface=None, statics=None,
                 cell_area_m2=None, array_module=None, **kwargs):
        if not supports_physics(physics,
                physics_mode=kwargs.get('physics_mode', 'exact'),
                physics_correct_skip=kwargs.get('physics_correct_skip', ())):
            raise NotImplementedError('generic Metal column plan is not admitted')
        if statics:
            raise ValueError('this generic plan has no orographic statics')
        # The selected schemes do not consume cell area. Keep generic's
        # validation, even though the existing batch needs no device copy.
        if cell_area_m2 is not None:
            import numpy as np
            area = np.asarray(cell_area_m2, dtype=np.float64)
            if (area.shape != (int(kwargs['n_columns']),)
                    or not np.all(np.isfinite(area)) or np.any(area <= 0)):
                raise ValueError('cell_area_m2 must be finite and positive per column')
        given = {} if surface is None else dict(surface)
        permitted = {'landmask', 'xland', 'ivgtyp', 'isltyp', 'vegfra', 'tsk',
                     'tmn', 'xice', 'snow', 'snow_depth', 'soil_temperature',
                     'soil_moisture', 'xice_threshold'}
        if set(given) - permitted:
            raise ValueError('surface contains fields outside this Metal plan')
        plan = column_physics_plan(physics, n_levels=kwargs['n_levels'],
            dt=kwargs['dt'], radiation_seconds=kwargs['radiation_seconds'],
            surface_pbl_seconds=kwargs['surface_pbl_seconds'],
            cumulus_seconds=kwargs.get('cumulus_seconds'))
        for name in ('soil_temperature', 'soil_moisture'):
            if name not in given or tuple(given[name].shape) != (plan.n_soil_levels, int(kwargs['n_columns'])):
                raise ValueError(f'surface {name} must contain the selected soil column')
        super().__init__(microphysics_scheme='wsm6', cumulus_scheme=None,
                         array_module=array_module, **given, **kwargs)
        self._plan = plan
        self.last_scalar_rates = MappingProxyType({})
        self._identity.update(schema=plan.identity['schema'],
                              physics=dict(plan.physics),
                              species=list(plan.species),
                              static_fields=list(plan.static_fields),
                              n_soil_levels=plan.n_soil_levels,
                              backend='metal')

    def _physics_settings(self):
        settings = super()._physics_settings()
        settings.update(icloud=2, gwd_opt=0)
        return settings

    def run_phase1(self, *, species=None, **kwargs):
        if species is not None:
            if set(species) != set(self._species):
                raise ValueError('phase-one species do not match the generic plan')
            kwargs.update(species)
        result = super().run_phase1(**kwargs)
        shape = (self._nz, self._ncol)
        for name in ('du', 'dv'):
            self._out[name][...] = self._driver.pbl_raw_rates[name].view(shape)
        return MpasColumnSuiteTendencies(
            **{item.name: getattr(result, item.name) for item in fields(result)},
            scalar_rates=self.last_scalar_rates, dw=None)

    def run_phase2(self, *, species=None, w=None, u=None, v=None, **kwargs):
        if species is not None:
            if set(species) != set(self._species):
                raise ValueError('phase-two species do not match the generic plan')
            kwargs.update(species)
        if w is not None:
            self._require_input('w', w, (self._nz + 1, self._ncol))
        if (u is None) != (v is None):
            raise ValueError('phase-two winds must be given as a pair')
        if u is not None:
            self._require_input('u', u, (self._nz, self._ncol))
            self._require_input('v', v, (self._nz, self._ncol))
        # The generic contract keeps allocator-owned state buffers and only
        # publishes the completed phase back to the caller.
        shape = (self._nz, self._ncol)
        target_theta = kwargs['theta']
        target_species = {name: kwargs[name] for name in self._species}
        self._state.thp.view(shape)[...] = target_theta
        self._state.p.view(shape)[...] = kwargs['pressure']
        for name, value in target_species.items():
            self._species_buffers[name].view(shape)[...] = value
        kwargs['theta'] = self._state.thp.view(shape)
        kwargs['pressure'] = self._state.p.view(shape)
        kwargs.update({name: self._species_buffers[name].view(shape) for name in self._species})
        receipt = super().run_phase2(**kwargs)
        target_theta[...] = self._state.thp.view(shape)
        for name, target in target_species.items():
            target[...] = self._species_buffers[name].view(shape)
        return receipt

    def radii(self):
        return {name: getattr(self._state, name).view((self._nz, self._ncol))
                for name in self._plan.radii}

    def soil_state(self):
        return {name: self._driver.fields[name].view((self._plan.n_soil_levels, self._ncol)).copy()
                for name in ('tslb', 'smois')}

    @property
    def n_soil_levels(self):
        return self._plan.n_soil_levels

    @property
    def plan(self):
        return self._plan

    @property
    def species(self):
        return self._species

    @property
    def physics(self):
        return dict(self._plan.physics)


def run_mpas_column_physics_metal(**kwargs):
    return MetalMpasColumnPhysics(**kwargs)
