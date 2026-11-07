"""Resolve pinned native resources for the unfinished Metal physics route.

DATA authentication and first-use installation use the existing MPAS resource
manifest and cache. Table transport is host work; coefficient reductions and
scheme initialization use the existing Metal implementations.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from woof.metal.array import MetalArrayModule
from .framework_resources_v842 import decode_ozone_tables, ozone_resources
from .metal_noah_v842 import load_native_resources, packaged_decoder
from .metal_rrtmg_lw_v842 import load_tables as load_longwave
from .metal_rrtmg_sw_v842 import initialize_tables as load_shortwave
from .metal_thompson_tables_v842 import load_tables as load_thompson
from . import resources_build as rb
from . import resources_fetch


def _ozone_directory(directory, levels, device):
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'data' /
        'mpasphys-native' / 'mpas-source-v842.json').read_text())['tables']
    names = ('OZONE_PLEV.TBL', 'OZONE_LAT.TBL', 'OZONE_DAT.TBL')
    return decode_ozone_tables(dict(table_directory=str(directory), nOznLevels=levels,
        table_sha256={name: manifest[name]['sha256'] for name in names}), device)


def load_resources(context, *, landuse, ozone_levels, packet=None, cache=None, microphysics="mp_thompson"):
    """Load native default-profile resources from a packet or installed DATA.

    ``landuse`` is the actual initialized native mminlu classification. An
    explicit packet is resolved by the existing packet loader, including its
    portable paths; its CUDA decoder is replaced by the shipped Metal decoder.
    The no-packet route installs the same pinned release DATA on first use.
    """
    if not isinstance(landuse, str) or not landuse.strip():
        raise ValueError('native Metal resources require the input landuse classification')
    landuse = landuse.strip()
    if microphysics not in ('mp_thompson', 'mp_thompson_aerosols', 'mp_wsm6', 'mp_kessler'):
        raise ValueError("unsupported native Metal microphysics resource selector")
    uses_thompson = microphysics in ('mp_thompson', 'mp_thompson_aerosols')
    if packet is None:
        if 'thompson-tables' not in rb.read_installed(cache)['resources']:
            if os.environ.get('WOOF_HEX_PHYSICS_RESOURCES_FIRST_USE', '') == 'off':
                raise rb.PhysicsResourceError('native Metal DATA is not installed and first-use fetching is off')
            resources_fetch.fetch(cache, pin_sets=('v8.4.2',), data_only=True)
        release = Path(resources_fetch.verify_installed_data(cache)['path'])
        tables = release / 'mpas-data' / rb.MPAS_DATA_COMMIT[:12]
        noah = dict(table_directory=str(tables), landuse=landuse, soil_type=rb.SOIL_TYPE)
        thompson = dict(table_directory=str(release / 'thompson-strict' / 'tables'))
        if microphysics == 'mp_thompson_aerosols':
            thompson['auxiliary'] = str(release / 'thompson-strict' / 'aerosol' / 'auxiliary.bin')
        ozone = _ozone_directory(tables, ozone_levels, MetalArrayModule(context))
        lw = sw = None  # The same pinned RRTMG files ship with the wheel.
    else:
        rows = rb.load_packet(Path(packet))
        noah = dict(rows['sf_noah'], landuse=landuse)
        if microphysics == 'mp_thompson_aerosols' and microphysics not in rows:
            raise rb.PhysicsResourceError('native Metal resource packet lacks Thompson aerosol tables')
        thompson = rows[microphysics] if uses_thompson else None
        ozone_names = ('OZONE_PLEV.TBL', 'OZONE_LAT.TBL', 'OZONE_DAT.TBL')
        table_dir = Path(noah['table_directory'])
        if 'ozone' not in rows and any((table_dir / name).exists() for name in ozone_names):
            ozone = _ozone_directory(table_dir, ozone_levels, MetalArrayModule(context))
        else:
            ozone = ozone_resources(Path(packet), ozone_levels, MetalArrayModule(context))
        lw = rows.get('rrtmg_lw')
        sw_row = rows.get('rrtmg_sw')
        sw = None if sw_row is None else {'table': sw_row.get('table', sw_row.get('data'))}
    load_selected = load_thompson
    if microphysics == 'mp_thompson_aerosols':
        from .metal_aerosol_tables_v842 import load_tables as load_selected
    noah['rust_decoder'] = packaged_decoder()
    return {'noah': load_native_resources(noah, context),
            'thompson': load_selected(thompson, context) if uses_thompson else None,
            'ozone': ozone, 'longwave': load_longwave(context, lw),
            'shortwave': load_shortwave(context, sw)}
