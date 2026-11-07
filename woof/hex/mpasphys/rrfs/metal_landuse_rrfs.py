"""RRFS land-use startup around the shared Metal column initializer."""

import numpy as np

from .framework_init_rrfs import FrameworkInitializationRRFS


def initialize_landuse_rrfs(framework, noah):
    """Initialize the fork flags, selected columns and land-use metadata."""
    native, config = framework.native, framework.configs
    classification = native.get_text_field('sfc_input', 'mminlu')
    if classification != noah.resources['landuse']:
        raise ValueError('RRFS land-use resources differ from the native classification')
    if config['config_lsm_scheme'].strip() == 'sf_ruc':
        for name, value in (('spp_lsm_loc', 0), ('mosaic_lu', 1), ('mosaic_soil', 1)):
            if ('diag_physics', name) in native._fields:
                noah._write('diag_physics', name, np.int32(value), count=None)
    if not (config.get('config_do_DAcycling', False) and not config['config_do_restart']):
        noah.initialize_landuse()
    FrameworkInitializationRRFS._landuse_state(framework)
