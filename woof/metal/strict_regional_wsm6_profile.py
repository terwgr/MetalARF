"""Eligibility for strict regional WSM6 with optional paired radiation."""
from woof.core.physics_inventory import physics_enabled
from woof.core.hail import enabled as hail_enabled
from woof.metal.strict_regional_kessler_profile import paired_radiation


def supports_extent(nz, dt):
    return 5 <= nz <= 80 and 0 < dt <= 120


def admits(cfg):
    """Select flat periodic or unmapped specified mixed-phase columns.

    Paired radiation requires an unmapped periodic grid. The dycore retains
    its own geometry and live-field checks.
    """
    return (cfg.moist and cfg.moist_cq and cfg.moist_adv_opt == 1
            and cfg.mp_physics == 6 and supports_extent(cfg.nz, cfg.dt)
            and cfg.wsm6_hail_opt in (0, 1) and not cfg.wsm6_conservative_surface
            and (not physics_enabled(cfg) or
                 (not cfg.specified and cfg.map_proj == 0 and int(cfg.use_mp_re) in (0, 1)
                  and paired_radiation(cfg)))
            and not hail_enabled(cfg)
            and cfg.terrain_opt == 0
            and (not cfg.specified or cfg.map_proj == 0)
            and not (cfg.nested or cfg.open_x or cfg.open_y)
            and cfg.km_opt == 1 and cfg.khdif == 0 and cfg.kvdif == 0
            and cfg.zadvect_implicit == 0 and cfg.diff_6th_opt == 0
            and cfg.damp_opt == 0 and cfg.w_damping == 0 and cfg.emdiv == 0
            and not (cfg.output_diagnostics or cfg.nwp_diagnostics
                     or cfg.gsd_diagnostics or cfg.lightning_option)
            and all(int(getattr(cfg, name, 0) or 0) == 0 for name in (
                'sf_urban_physics', 'sf_lake_physics', 'sf_ocean_physics',
                'ifire', 'spp_lsm', 'spp_conv', 'spp_pbl')))
