"""Compare native LW initialization against independent Fortran coefficient words."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from woof.hex.mpasphys.metal_radiation_tables_v842 import load_raw_tables
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import load_tables
from woof.hex.mpasphys.rrtmg_lw_v842 import _REDUCTIONS
from woof.ingest.rrtmg_coeffs import DATA_DIR
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
raw=load_raw_tables('lw')
manifest=json.loads((DATA_DIR/'rrtmg_coeffs_oracle_manifest.json').read_text())['arrays']
raw_count=raw_words=0
for module,fields in raw.items():
    for name,value in fields.items():
        reference=manifest[module+'/'+name]
        assert list(value.shape)==reference['shape'] and str(value.dtype)==reference['dtype']
        assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==reference['sha256'],(module,name)
        raw_count+=1;raw_words+=value.size
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*a,**k):
        result=submit(*a,**k);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    tables=load_tables(context);mismatch=[];words=0
    for band,dest,name,points,first,weighted in _REDUCTIONS:
        module=f'rrlw_kg{band:02d}';target=dest.split('/')[1]
        reference=manifest[module+'/'+target]
        got=tables['reduced'][dest].to_numpy().reshape(reference['shape'],order='F')
        digest=hashlib.sha256(np.ascontiguousarray(got).tobytes()).hexdigest()
        words+=got.size
        if digest!=reference['sha256']:mismatch.append(module+'/'+target)
    print('authenticated raw',raw_count,'fields',raw_words,'words; reduced',len(_REDUCTIONS),'fields',words,'words; digest mismatches',mismatch)
    if mismatch:raise SystemExit(1)
