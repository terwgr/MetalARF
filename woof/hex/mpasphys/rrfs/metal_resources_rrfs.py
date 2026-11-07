"""Authenticated resources for the RRFS-44 HRRRv5 Metal session.

The physics-resources packet must include ``metal_tempo.table_witness``, a
path to the portable TEMPO table receipt (relative paths start with ``./``).
"""
from pathlib import Path

from woof.metal.array import MetalArrayModule
from .. import resources_build as rb
from ..framework_resources_v842 import decode_ozone_tables
from ..metal_noah_v842 import load_native_resources, packaged_decoder
from ..metal_rrtmg_lw_v842 import load_tables as load_longwave
from ..metal_rrtmg_sw_v842 import initialize_tables as load_shortwave
from ..metal_rrtmg_lw_slot_v842 import MetalMpasLongwave
from ..metal_rrtmg_sw_slot_v842 import MetalMpasShortwave
from ..native_library_build import source_manifest
from .metal_ruc_slots import MetalRucSlots
from .metal_tempo_slot import MetalMpasTempo
from .tempo_v303 import CONFIG_ARGUMENTS


_TEMPO_REFERENCE = '2cdfb40858266b2ecbf742053f84bcb34dba2df18f693a171dc524c5aec9ae5c'
_OZONE = ('OZONE_PLEV.TBL', 'OZONE_LAT.TBL', 'OZONE_DAT.TBL')
_SHARED = ('LANDUSE.TBL', 'VEGPARM.TBL', 'SOILPARM.TBL', 'GENPARM.TBL',
           *_OZONE, 'RRTMG_LW_DATA', 'RRTMG_SW_DATA')


def load_resources(context, *, config, landuse, ozone_levels, packet):
    """Return the keyword arguments needed by ``MetalPhysicsSession``."""
    if config.pin_set != 'rrfs-44c0849':
        raise rb.PhysicsResourceError('Metal HRRRv5 resources require the RRFS-44 pin set')
    if not isinstance(landuse, str) or not landuse.strip():
        raise rb.PhysicsResourceError('Metal RRFS resources require the input landuse classification')
    if packet is None or not Path(packet).is_file():
        raise rb.PhysicsResourceError('Metal RRFS requires a physics-resources packet')
    rows = rb.load_packet(Path(packet))
    required = (('sf_noah', 'table_directory'), ('sf_noah', 'soil_type'), ('rrtmg_lw', 'table'),
                ('rrtmg_sw', 'data'), ('metal_tempo', 'table_witness'))
    for row, key in required:
        if not isinstance(rows.get(row), dict) or not rows[row].get(key):
            raise rb.PhysicsResourceError(f'Metal RRFS resource packet lacks {row}.{key}')
    witness = Path(rows['metal_tempo']['table_witness'])
    if not witness.is_file():
        raise rb.PhysicsResourceError('Metal RRFS TEMPO table witness is missing')
    for row, key in (('rrtmg_lw', 'table'), ('rrtmg_sw', 'data')):
        descriptor = rows[row][key]
        if not isinstance(descriptor, dict) or not descriptor.get('path') or not descriptor.get('sha256'):
            raise rb.PhysicsResourceError(f'Metal RRFS resource packet lacks {row}.{key} path or sha256')
        if not Path(descriptor['path']).is_file():
            raise rb.PhysicsResourceError(f'Metal RRFS resource packet is missing {row}.{key} data')
    flags = tuple(int(bool(config.execution_options[name])) for name in CONFIG_ARGUMENTS)
    if flags != (1, 1, 0, 0):
        raise rb.PhysicsResourceError('Metal RRFS TEMPO has no pinned witness for the selected flags')

    tables = source_manifest(config.pin_set)['tables']
    shared = source_manifest('v8.4.2')['tables']
    if any(tables[name]['sha256'] != shared[name]['sha256'] for name in _SHARED):
        raise rb.PhysicsResourceError('Metal RRFS shared table pins differ from the v8.4.2 decoders')
    for row, key, name in (('rrtmg_lw', 'table', 'RRTMG_LW_DATA'),
                           ('rrtmg_sw', 'data', 'RRTMG_SW_DATA')):
        if rows[row][key]['sha256'] != tables[name]['sha256']:
            raise rb.PhysicsResourceError(f'Metal RRFS requires the pinned {name}')
    noah = dict(rows['sf_noah'], landuse=landuse.strip(), rust_decoder=packaged_decoder())
    ozone = rows.get('ozone', {})
    ozone_directory = ozone.get('table_directory', noah['table_directory'])
    for name in _SHARED[:4]:
        if not (Path(noah['table_directory']) / name).is_file():
            raise rb.PhysicsResourceError(f'Metal RRFS resource packet is missing {name}')
    for name in _OZONE:
        if not (Path(ozone_directory) / name).is_file():
            raise rb.PhysicsResourceError(f'Metal RRFS resource packet is missing {name}')
    ozone_tables = decode_ozone_tables(dict(table_directory=ozone_directory, nOznLevels=ozone_levels,
        table_sha256={name: tables[name]['sha256'] for name in _OZONE}), MetalArrayModule(context))
    lw = load_longwave(context, {'table': rows['rrtmg_lw']['table']})
    sw = load_shortwave(context, {'table': rows['rrtmg_sw']['data']})
    tempo = MetalMpasTempo.from_table_witness(context, flags, witness,
        pin_set=config.pin_set, reference_sha256=_TEMPO_REFERENCE)
    ruc = MetalRucSlots()
    return dict(resources=Path(packet), noah_resources=load_native_resources(noah, context),
        ozone_tables=ozone_tables, slot_services={
            ('microphysics', 'mp_tempo'): tempo,
            ('land_surface', 'sf_ruc'): ruc,
            ('seaice', 'sf_ruc'): ruc,
            ('radiation_lw', 'rrtmg_lw'): MetalMpasLongwave(context, lw, config=config),
            ('radiation_sw', 'rrtmg_sw'): MetalMpasShortwave(context, sw, config=config),
        })
