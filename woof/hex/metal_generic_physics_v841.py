"""Bounded generic column provider for resident Metal Hex state."""
from __future__ import annotations

from woof.core.mpas_column_physics_metal import (
    run_mpas_column_physics_metal)
from woof.core.mpas_column_metal_profiles import supports_generic_profile, supports_physics
from .cuda_suite_physics_v841 import SealedSuiteConstructorV841, _suite_for_row
from .errors import ConfigurationRefusal
from .metal_arwen_physics_v841 import MetalTwoPhasePhysicsBackendV841



class MetalGenericPhysicsBackendV841(MetalTwoPhasePhysicsBackendV841):
    def __init__(self, *, constructor_values, prep_geometry, array_module,
                 gwdo_static=None, rollback=True):
        if constructor_values.get('generation'):
            raise NotImplementedError('generic Metal generation overrides are not admitted')
        if gwdo_static is not None:
            raise ValueError('this generic plan has no drag statics')
        self._constructor = SealedSuiteConstructorV841.from_mapping(constructor_values)
        if not supports_physics(self._constructor.physics,
                physics_mode=self._constructor.physics_mode,
                physics_correct_skip=self._constructor.physics_correct_skip):
            raise NotImplementedError('generic Metal constructor plan is not admitted')
        prep_geometry.validate()
        self._values = self._constructor.arwen_kwargs()
        self._values['physics_mode'] = self._constructor.physics_mode
        self._values['physics_correct_skip'] = self._constructor.physics_correct_skip
        self._n_levels = self._constructor.n_levels
        self._n_columns = self._constructor.n_columns
        self._dt = self._constructor.dt
        if prep_geometry.n_cells != self._n_columns:
            raise ValueError('preparation geometry and column counts differ')
        self._xp = array_module
        self._factory = run_mpas_column_physics_metal
        self._prep_geometry = prep_geometry
        self._gwdo_static = None
        self._seam = self._new_seam()
        self._rollback = bool(rollback)
        self._phase = 'boundary'
        self._snapshot = self._step_start = None
        self._scalar_target = self._scalar_backup = None
        self._pending_gwdo = self.last_gwdo = None
        self._pending_refl10cm = self._committed_refl10cm = None
        self._gf_forcing_seen = False

    def _new_seam(self):
        seam = self._factory(**self._values, array_module=self._xp)
        if dict(seam.surface_classification) != dict(self._constructor.expected_surface_classification()):
            raise ValueError('generic surface classification differs from sealed constructor')
        self.noahmp_statics_receipt = {'bound': False, 'fields': {}}
        return seam


def build_column_backend(row, *, seam_options=None, **kwargs):
    profile = _suite_for_row(row.name)
    constructor = SealedSuiteConstructorV841.from_mapping(kwargs['constructor_values'])
    if (seam_options or not supports_generic_profile(profile)
            or not supports_physics(constructor.physics,
                physics_mode=constructor.physics_mode,
                physics_correct_skip=constructor.physics_correct_skip)
            or constructor.selection != dict(profile['selection'])
            or tuple(row.model_scalar_names()) != tuple(profile['species'])
            or row.batch_module != profile['engine_batch']['module']
            or row.batch_entrypoint != profile['engine_batch']['entrypoint']):
        raise ConfigurationRefusal('physics_backend', row.name,
                                   'this generic selection has no Metal admission',
                                   'a supported generic Metal plan')
    return MetalGenericPhysicsBackendV841(**kwargs)


def additional_memory_bytes(row, n_cells, n_levels):
    from .metal_arwen_physics_v841 import additional_memory_bytes as batch_bytes
    profile = _suite_for_row(row.name)
    if not supports_generic_profile(profile):
        raise ConfigurationRefusal('physics_backend', row.name,
                                   'this generic selection has no Metal admission',
                                   'a supported generic Metal plan')
    return batch_bytes(row, n_cells, n_levels)
