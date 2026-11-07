"""Construct real native Metal services for the convection-permitting profile.

The caller owns the session, native initial module locations and source-valid
mesh/field inputs. SF_MYNN, cloud fraction and YSU GWDO are registered by the
existing Metal session. Host work here authenticates/decodes input assets;
coefficient reductions and scheme execution remain on Metal. This helper does
not admit a native forecast provider.
"""
from __future__ import annotations
import numpy as np
from woof.metal.array import MetalArrayModule
from woof.hex.mpasphys.framework_resources_v842 import decode_ozone_tables
from woof.hex.mpasphys.metal_noah_v842 import MetalMpasNoah,load_native_resources
from woof.hex.mpasphys.metal_seaice_v842 import MetalMpasSeaice
from woof.hex.mpasphys.metal_bl_mynn_slot_v842 import MetalMpasBlMynn
from woof.hex.mpasphys.metal_thompson_column_v842 import MetalMpasThompson
from woof.hex.mpasphys.metal_thompson_tables_v842 import load_tables as load_thompson
from woof.hex.mpasphys.metal_gf_v842 import MetalMpasGrellFreitas
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import load_tables as load_longwave
from woof.hex.mpasphys.metal_rrtmg_lw_slot_v842 import MetalMpasLongwave
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_tables as load_shortwave
from woof.hex.mpasphys.metal_rrtmg_sw_slot_v842 import MetalMpasShortwave
from woof.hex.mpasphys.driver_v842 import resolve_namelist


def real_services(context, config, *, noah_resources, thompson_resources, ozone_resources,
                  ozone_levels, noah_locations, seaice_locations, lw_resources=None, sw_resources=None):
    """Return the existing session constructor resources for this profile.

    ``noah_resources`` is the native resource descriptor; ``ozone_resources``
    is either a packet with an ``ozone`` row or that raw row. ``ozone_levels``
    is the hydrated Registry level count; no download fallback is used.
    ``thompson_resources`` names the certified native table directory. Optional
    LW/SW descriptors select the same pinned assets as the packaged defaults.
    The four Noah and two sea-ice location words must be supplied by the caller.
    The current slot implementation owns the one-based, single-tile I/J domain.
    """
    active={slot:scheme for slot,scheme in config.selectors.items() if config.active(slot)}
    default=resolve_namelist("&physics\nconfig_physics_suite='convection_permitting'\n/\n")
    expected={slot:scheme for slot,scheme in default.selectors.items() if default.active(slot)}
    if config.suite!='convection_permitting' or active!=expected:
        raise ValueError('real Metal services require the native convection-permitting profile')
    if not config.options['config_o3climatology']:
        raise ValueError('the default profile requires native ozone climatology')
    if (config.options['config_gfconv_closure_deep'],config.options['config_gfconv_closure_shallow'])!=(0,8):
        raise ValueError('native Metal GF requires the default closure selectors')
    noah_words=np.asarray(noah_locations)
    seaice_words=np.asarray(seaice_locations)
    if noah_words.dtype!=np.int32 or noah_words.shape!=(4,):
        raise ValueError('Noah module locations require four int32 words')
    if seaice_words.dtype!=np.int32 or seaice_words.shape!=(2,):
        raise ValueError('sea-ice module locations require two int32 words')
    ozone_row=dict(ozone_resources.get('ozone',ozone_resources))
    if 'nOznLevels' in ozone_row and int(ozone_row['nOznLevels'])!=int(ozone_levels):
        raise ValueError('ozone resources differ from the Registry level count')
    ozone_row['nOznLevels']=int(ozone_levels)
    xp=MetalArrayModule(context)
    noah=load_native_resources(noah_resources,context)
    thompson=load_thompson(thompson_resources,context)
    ozone=decode_ozone_tables(ozone_row,xp)
    lw=load_longwave(context,lw_resources)
    sw=load_shortwave(context,sw_resources)
    gf=MetalMpasGrellFreitas()
    services={('land_surface','sf_noah'):MetalMpasNoah(noah['tables'],xp.asarray(noah_words)),
              ('seaice','sf_noah'):MetalMpasSeaice(context,module_locations=seaice_words,i_start=1,j_index=1),
              ('pbl','bl_mynn'):MetalMpasBlMynn(context),
              ('microphysics','mp_thompson'):MetalMpasThompson(thompson),
              ('convection','cu_grell_freitas'):gf,
              ('radiation_lw','rrtmg_lw'):MetalMpasLongwave(context,lw),
              ('radiation_sw','rrtmg_sw'):MetalMpasShortwave(context,sw)}
    return dict(slot_services=services,noah_resources=noah,ozone_tables=ozone)
