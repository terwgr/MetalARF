"""Exercise an ordinary portable native packet against authenticated loaders.

Only seven small native tables are copied into the temporary packet directory.
Thompson and radiation assets remain at their supplied authenticated paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from woof.ingest import rrtmg_coeffs
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.hex.mpasphys.framework_resources_v842 import decode_ozone_tables
from woof.hex.mpasphys.metal_noah_v842 import load_native_resources
from woof.hex.mpasphys.metal_resources_v842 import load_resources
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import load_tables as load_longwave
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_tables as load_shortwave
from woof.hex.mpasphys.metal_thompson_tables_v842 import load_tables as load_thompson


def words(value):
    """All resident words, plus their full shape and dtype."""
    return value.shape, str(value.dtype), value.to_numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noah-packet', type=Path, required=True)
    parser.add_argument('--ozone-packet', type=Path, required=True)
    parser.add_argument('--thompson-directory', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    noah = json.loads(args.noah_packet.read_text())
    ozone = json.loads(args.ozone_packet.read_text())
    with TemporaryDirectory(prefix='mpas-metal-portable-resources-') as tmp:
        root = Path(tmp); tables = root / 'tables'; tables.mkdir()
        packet_dir = root / 'packets'; packet_dir.mkdir()
        for name in ('LANDUSE.TBL', 'VEGPARM.TBL', 'SOILPARM.TBL', 'GENPARM.TBL'):
            shutil.copyfile(Path(noah['table_directory']) / name, tables / name)
        for name in ('OZONE_PLEV.TBL', 'OZONE_LAT.TBL', 'OZONE_DAT.TBL'):
            shutil.copyfile(Path(ozone['table_directory']) / name, tables / name)
        lw = {'table': {'path': str(rrtmg_coeffs.DATA_DIR / 'RRTMG_LW_DATA'),
                        'sha256': rrtmg_coeffs.RRTMG_LW_DATA_SHA256}}
        sw_data = {'path': str(rrtmg_coeffs.DATA_DIR / 'RRTMG_SW_DATA'),
                   'sha256': rrtmg_coeffs.RRTMG_SW_DATA_SHA256}
        portable_noah = dict(noah, table_directory='../tables')
        packet = packet_dir / 'physics-resources.json'
        packet.write_text(json.dumps({'sf_noah': portable_noah,
            'mp_thompson': {'table_directory': str(args.thompson_directory)},
            'rrtmg_lw': lw, 'rrtmg_sw': {'data': sw_data}}))
        with MetalContext(batch_kernel_calls=False) as context:
            noah_ref = load_native_resources(noah, context)
            ozone_ref = decode_ozone_tables(ozone, MetalArrayModule(context))
            thompson_ref = load_thompson({'table_directory': str(args.thompson_directory)}, context)
            lw_ref = load_longwave(context, lw)
            sw_ref = load_shortwave(context, {'table': sw_data})
            routes = [('packet', packet)] + ([('installed', None)] if args.cache else [])
            for route, selected_packet in routes:
                result = load_resources(context, landuse=noah['landuse'],
                    ozone_levels=int(ozone['nOznLevels']), packet=selected_packet, cache=args.cache)
                checked = []
                for name in ('tables', 'landuse_tables'):
                    assert words(result['noah'][name]) == words(noah_ref[name]), name
                    checked.append('noah.' + name)
                assert result['noah']['landuse'] == noah_ref['landuse']
                assert result['noah']['isurban'] == (noah_ref['isurban'] if route == 'packet' else None)
                for name in ('pin', 'lat_ozone', 'ozmixin'):
                    assert words(result['ozone'][name]) == words(ozone_ref[name]), name
                    checked.append('ozone.' + name)
                for name in sorted(k for k in thompson_ref if not k.startswith('_')):
                    assert words(result['thompson'][name]) == words(thompson_ref[name]), name
                    checked.append('thompson.' + name)
                assert result['thompson']['_file_sha256'] == thompson_ref['_file_sha256']
                for name, value in lw_ref.items():
                    if name == 'band':  # Views of the same reduced arrays.
                        continue
                    if name == 'reduced':
                        for key, array in value.items():
                            assert words(result['longwave'][name][key]) == words(array), key
                            checked.append('longwave.reduced.' + key)
                    elif hasattr(value, 'to_numpy'):
                        assert words(result['longwave'][name]) == words(value), name
                        checked.append('longwave.' + name)
                    else:
                        assert np.asarray(result['longwave'][name]).tobytes() == np.asarray(value).tobytes(), name
                        checked.append('longwave.' + name)
                for name in ('packed', 'weights', 'ngb'):
                    assert words(getattr(result['shortwave'], name)) == words(getattr(sw_ref, name)), name
                    checked.append('shortwave.' + name)
                for name in ('offsets', 'static_start', 'exp_start'):
                    assert getattr(result['shortwave'], name) == getattr(sw_ref, name), name
                print(json.dumps({'checked_full_arrays': checked,
                    'route': route, 'packet': 'rebased Noah/O3 directory, no explicit ozone row',
                    'scope': 'authenticated resource loading; no forecast claim'}), flush=True)


if __name__ == '__main__':
    main()
