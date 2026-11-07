"""The selectable column-suite provider on resident Metal Hex state."""
from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
import math

import numpy as np

from woof.core.mpas_column_suite_metal import run_mpas_column_suite_metal
from .cuda_suite_physics_v841 import SealedSuiteConstructorV841, _suite_for_row
from .errors import ConfigurationRefusal
from .metal_arwen_physics_v841 import MetalTwoPhasePhysicsBackendV841


from .physics_suites import supports_synthesized_metal_profile as supports_synthesized_profile


def _metal_suite_profile(row):
    profile = _suite_for_row(row.name)
    if profile.get("synthesized"):
        from .physics_backend_admission import MYNN_RUC_GSL_COLUMN_ROW
        base = MYNN_RUC_GSL_COLUMN_ROW
        if (not supports_synthesized_profile(profile)
                or row.batch_module != base.batch_module or row.batch_entrypoint != base.batch_entrypoint
                or row.driven_scalar_names != base.driven_scalar_names
                or row.appended_scalar_names != base.appended_scalar_names):
            raise ConfigurationRefusal("physics_backend", row.name,
                                       "the synthesized suite has no Metal admission",
                                       "the supported MYNN/RUC/GSL generation or a named Metal row")
    return profile


class MetalSuitePhysicsBackendV841(MetalTwoPhasePhysicsBackendV841):
    """Shared Hex transaction with suite-owned surface, PBL and drag state."""

    def __init__(self, *, constructor_values, prep_geometry, array_module,
                 gwdo_static=None, rollback=True):
        if gwdo_static is not None:
            raise ValueError("the suite carries drag inside the PBL slot")
        self._constructor = SealedSuiteConstructorV841.from_mapping(constructor_values)
        if self._constructor.physics is not None:
            raise NotImplementedError("the generic registry Metal column seam is not implemented")
        prep_geometry.validate()
        self._values = self._constructor.arwen_kwargs()
        self._values["physics_mode"] = self._constructor.physics_mode
        self._values["physics_correct_skip"] = self._constructor.physics_correct_skip
        self._n_levels = self._constructor.n_levels
        self._n_columns = self._constructor.n_columns
        self._dt = self._constructor.dt
        if prep_geometry.n_cells != self._n_columns:
            raise ValueError("preparation geometry and suite column counts differ")
        self._xp = array_module
        self._factory = run_mpas_column_suite_metal
        self._prep_geometry = prep_geometry
        self._gwdo_static = None
        self._seam = self._new_seam()
        self._rollback = bool(rollback)
        self._phase = "boundary"
        self._snapshot = None
        self._step_start = None
        self._scalar_target = self._scalar_backup = None
        self._pending_gwdo = self.last_gwdo = None
        self._pending_refl10cm = self._committed_refl10cm = None
        self._gf_forcing_seen = False

    def _new_seam(self):
        seam = self._factory(**self._values, array_module=self._xp)
        if seam.selection != self._constructor.selection:
            raise ValueError("the Metal suite selection differs from its sealed constructor")
        if seam.n_soil_levels != self._constructor.n_soil_levels:
            raise ValueError("the Metal suite soil geometry differs from its constructor")
        actual = dict(seam.surface_classification)
        expected = dict(self._constructor.expected_surface_classification())
        if actual != expected:
            raise ValueError(f"suite surface classification differs: {actual} != {expected}")
        self.noahmp_statics_receipt = {"bound": False, "fields": {}}
        return seam

    def begin_step(self, **kwargs):
        raw = super().begin_step(**kwargs)
        try:
            extra = dict(self._seam.last_scalar_rates)
            if set(extra) != set(self._seam.mixed_number_species):
                raise ValueError("the suite number-species rate inventory changed")
            if set(extra) - set(kwargs["scalar_names"]):
                raise ValueError("the suite returned a rate for an untransported species")
            if extra:
                raw = replace(raw, dscalars=MappingProxyType({**raw.dscalars, **extra}))
                raw.validate(n_vert_levels=self._n_levels, n_cells=self._n_columns)
            return raw
        except Exception:
            self._restore_boundary(self._snapshot)
            raise

    @property
    def land_surface_receipt(self):
        land = self._constructor.selection["land_surface_scheme"]
        census = (self._seam.last_ruc_census if land == "ruc"
                  else self._seam.last_noahmp_census)
        return {"land_surface_scheme": land, "census": census,
                "classification": dict(self._seam.surface_classification),
                "engine_drag": self._constructor.selection["drag_scheme"],
                "external_ysu_drag": False}

    def diagnostic_snapshot(self):
        snapshot = super().diagnostic_snapshot()
        for name in ("tslb", "smois"):
            if snapshot["soil"][name].shape != (self._constructor.n_soil_levels,
                                               self._n_columns):
                raise ValueError(f"suite {name} soil geometry changed")
        return snapshot


def build_column_backend(row, *, seam_options=None, **kwargs):
    if seam_options:
        raise ConfigurationRefusal("physics_backend", row.name,
                                   "the suite carries no point-source treatment",
                                   "a source-capable provider with a Metal implementation")
    constructor = SealedSuiteConstructorV841.from_mapping(kwargs["constructor_values"])
    profile = _metal_suite_profile(row)
    if constructor.selection != dict(profile["selection"]):
        raise ValueError("suite constructor selection differs from the provider profile")
    if profile.get("synthesized") and constructor.arwen_kwargs().get("generation", {}) != dict(profile["generation"]):
        raise ValueError("suite constructor generation differs from the provider profile")
    from woof.core.mpas_column_batch import _SPECIES_BY_SCHEME
    if tuple(_SPECIES_BY_SCHEME[constructor.microphysics_scheme]) != row.model_scalar_names():
        raise ValueError("suite transported species differ from the provider row")
    return MetalSuitePhysicsBackendV841(**kwargs)


def additional_memory_bytes(row, n_cells, n_levels):
    """Price the selected suite's scratch, soil and packed RUC allocations.

    Shared Hex residency remains in ``estimate_hex``. This adds a conservative
    full-column scratch inventory and the provider's extra transport copies;
    the MYNN inventory is priced at full width without querying a CUDA card.
    """
    from woof.config import RunConfig
    from woof.core.mpas_column_batch import _MP_PHYSICS_BY_SCHEME
    from woof.core.mpas_column_suite import resolve_suite_selection, resolve_generation
    from woof.core.mynn_pbl_scratch import mynn_pricing_resident_chunk
    from woof.core.preflight import scratch_slot_registry
    from woof.metal.memory_gate import _ruc_table_inventory
    from woof.metal.ruc_metal_workspace import ruc_workspace_bytes
    from .species_row import species_row_for_names

    profile = _metal_suite_profile(row)
    species = species_row_for_names(row.model_scalar_names())
    settings = resolve_suite_selection(**profile["selection"])["config"]
    settings["ra_rrtmg_variant"] = "rrtmg_legacy"
    settings.update(resolve_generation(profile.get("generation"),
                    microphysics_scheme=species.engine_scheme,
                    selection=profile["selection"]))
    if profile["fractional_seaice"] is not None:
        settings["fractional_seaice"] = int(profile["fractional_seaice"])
    cfg = RunConfig(nx=int(n_cells), ny=1, nz=int(n_levels), moist=True,
                    dx=1.0, dy=1.0, ztop=float(n_levels), dt=1.0,
                    run_seconds=0.0,
                    mp_physics=_MP_PHYSICS_BY_SCHEME[species.engine_scheme],
                    **settings)
    with mynn_pricing_resident_chunk(int(n_cells)):
        scratch = sum(8 * math.prod(shape)
                      for shape in scratch_slot_registry(cfg).values())
    from woof.core.ruc_runtime import RUC_PROFILE_BINDING, RUC_STATE_3D
    soil_fields = set(RUC_PROFILE_BINDING) | set(RUC_STATE_3D) | {
        "smois", "tslb", "sh2o", "smcrel"}
    soil = (8 * int(n_cells) * int(profile["n_soil_levels"])
            * len(soil_fields))
    packed = 0
    if int(cfg.sf_surface_physics) == 3:
        table_words, _, _ = _ruc_table_inventory()
        packed = ruc_workspace_bytes(int(n_cells), table_words=table_words)
    extra = max(len(species.names()) - 6, 0)
    transport = 64 * 4 * extra * int(n_cells) * int(n_levels)
    statics = 8 * int(n_cells) * len(profile["orographic_statistics"]["variables"])
    return int(scratch + soil + packed + transport + statics)


__all__ = ["MetalSuitePhysicsBackendV841", "supports_synthesized_profile", "build_column_backend",
           "additional_memory_bytes"]
