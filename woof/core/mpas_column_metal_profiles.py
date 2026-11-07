"""Pure capability checks for bounded generic Metal column plans."""

# This first profile has the batch seam's WSM6/YSU/Noah-MP inventory, with
# WRF's incidence cloud fraction. The generic planner remains authoritative.
_PHYSICS = {
    'mp_physics': 6, 'bl_pbl_physics': 1, 'sf_sfclay_physics': 1,
    'sf_surface_physics': 4, 'gwd_opt': 0, 'ra_physics': 4,
    'ra_rrtmg_variant': 'rrtmg_legacy', 'icloud': 2, 'cu_physics': 0,
}


def supports_physics(physics, *, physics_mode="exact", physics_correct_skip=()):
    """Judge the complete resolved plan without constructing a device."""
    from woof.core.mpas_column_batch import _STATE_RESTART_FIELDS_BY_SCHEME
    from woof.core.mpas_column_suite import column_physics_plan
    try:
        from woof.correct_mode import resolve
        settings = resolve(physics_mode, physics_correct_skip).settings
        selected = dict(physics)
        if any(key in selected and selected[key] != value
               for key, value in settings.items()):
            return False
        # Exact keeps the original plan's sparse spelling and identity.
        if physics_mode == "correct":
            selected.update(settings)
        reference_physics = dict(_PHYSICS)
        if physics_mode == "correct":
            reference_physics.update(settings)
        candidate = column_physics_plan(selected)
        reference = column_physics_plan(reference_physics)
    except (TypeError, ValueError, KeyError):
        return False
    return (dict(candidate.physics) == dict(reference.physics)
            and candidate.species == reference.species
            and candidate.state_fields == reference.state_fields
            and candidate.radii == reference.radii
            and candidate.surface_accumulators == reference.surface_accumulators
            and candidate.surface_receipts == reference.surface_receipts
            and candidate.n_soil_levels == reference.n_soil_levels
            and candidate.static_fields == reference.static_fields
            and candidate.state_fields == tuple(sorted(_STATE_RESTART_FIELDS_BY_SCHEME['wsm6'])))


def supports_generic_profile(profile):
    """Candidate metadata assumes no cumulus; the sealed constructor rechecks it."""
    if (not profile.get('generic') or profile.get('physics') is None
            or profile.get('generation')):
        return False
    physics = dict(profile['physics'])
    if physics.get('cu_physics', 0) != 0:
        return False
    physics.setdefault('cu_physics', 0)
    if not supports_physics(physics):
        return False
    from woof.core.mpas_column_suite import column_physics_plan
    plan = column_physics_plan(physics)
    return (tuple(profile.get('species', ())) == plan.species
            and profile.get('n_soil_levels') == plan.n_soil_levels
            and profile.get('microphysics') == 'mp_wsm6'
            and not profile.get('orographic_statistics')
            and dict(profile.get('selection', {})) == {
                'radiation_scheme': 'rrtmg_legacy', 'surface_layer_scheme': 'revised_mm5',
                'land_surface_scheme': 'noahmp', 'pbl_scheme': 'ysu', 'drag_scheme': 'off'})
