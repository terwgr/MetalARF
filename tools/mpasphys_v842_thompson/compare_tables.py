"""Compare complete Metal table upload with the upstream decoded-output certificate."""
from pathlib import Path
import argparse,hashlib
import numpy as np
from woof.hex.mpasphys.metal_thompson_tables_v842 import load_tables
from woof.hex.mpasphys.table_resources_v842 import thompson_record_groups,THOMPSON_AUXILIARY_NAMES
from woof.hex.mpasphys import thompson_v842 as native
from woof.hex.mpasphys.resources_build import DECODED_SHA256
from woof.metal.runtime import MetalContext
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('table_directory',type=Path);args=parser.parse_args()
with MetalContext(batch_kernel_calls=False) as context:
    tables=load_tables(dict(table_directory=args.table_directory),context)
    digest=hashlib.sha256();words=0
    names=tuple(name for _,group in thompson_record_groups(native) for name in group)+THOMPSON_AUXILIARY_NAMES
    for name in names:
        array=tables[name].to_numpy();digest.update(name.encode()+b'\0'+array.tobytes());words+=array.size
        assert array.size==np.prod(native.TABLE_SHAPES[name])
    assert digest.hexdigest()==DECODED_SHA256['mp_thompson']
    assert tables['_file_sha256']==native.TABLE_FILE_SHA256
    print('33 native Thompson tables,',words,'uploaded words match the existing decoded-output certificate; no table arithmetic')
