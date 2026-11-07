"""Eligibility for strict regional Kessler with optional paired radiation."""
from dataclasses import replace

from woof.core.physics_inventory import physics_enabled
from woof.core.hail import enabled as hail_enabled
from woof.config import radiation_scheme_ids


def paired_radiation(cfg):
    from woof.core.physics import _metal_strict_legacy_pins

    return (radiation_scheme_ids(cfg) == (4, 4)
            and _metal_strict_legacy_pins(cfg)
            and not physics_enabled(replace(
                cfg, ra_physics=0, ra_lw_physics=0, ra_sw_physics=0))
            and all(int(getattr(cfg, name, 0) or 0) == 0 for name in (
                "alb_sol", "swint_opt", "slope_rad", "topo_shading", "ra_sw_eclipse")))


def admits(cfg):
    """Select flat periodic or unmapped specified warm rain.

    Paired radiation requires an unmapped periodic grid. The dycore retains
    its own geometry and live-field checks.
    """
    return (cfg.moist and cfg.moist_cq and cfg.moist_adv_opt in (0, 1)
            and cfg.mp_physics == 1 and 2 <= cfg.nz <= 256
            and (not physics_enabled(cfg) or
                 (not cfg.specified and cfg.map_proj == 0 and paired_radiation(cfg)))
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
