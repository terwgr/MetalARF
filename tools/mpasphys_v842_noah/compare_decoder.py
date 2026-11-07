"""Compare packaged Noah decoder output and Metal upload with pinned table words."""
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from woof.hex.mpasphys.metal_noah_v842 import packaged_decoder, load_native_resources
from woof.hex.mpasphys.resources_build import DECODED_SHA256
from woof.metal.runtime import MetalContext

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('table_directory', type=Path)
args = parser.parse_args()
decoder = packaged_decoder()
with MetalContext(batch_kernel_calls=False) as context:
    for landuse in ('MODIFIED_IGBP_MODIS_NOAH', 'USGS'):
        expected = subprocess.check_output([decoder['path'], str(args.table_directory), landuse, 'STAS'])
        assert hashlib.sha256(expected).hexdigest() == DECODED_SHA256['sf_noah:' + landuse]
        packet = dict(table_directory=args.table_directory, landuse=landuse, soil_type='STAS')
        default = load_native_resources(packet, context)
        explicit = load_native_resources(dict(packet, rust_decoder=decoder), context)
        for actual in (default, explicit):
            words = b''.join(actual[name].to_numpy().view(np.uint8).tobytes()
                             for name in ('tables', 'landuse_tables'))
            assert words == expected, landuse
        print(landuse, len(expected), 'decoded/uploaded bytes match pinned source certificate; default and explicit paths')
