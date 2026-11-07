"""Compare raw native table ingest with the independent Fortran word manifests."""
import hashlib,json
from pathlib import Path
import numpy as np
from woof.hex.mpasphys.metal_radiation_tables_v842 import load_raw_tables
from woof.ingest.rrtmg_coeffs import DATA_DIR

manifest=json.loads((DATA_DIR/"rrtmg_coeffs_oracle_manifest.json").read_text())["arrays"]
for spectrum in ("lw", "sw"):
    raw=load_raw_tables(spectrum);arrays=words=integers=0
    for module, fields in raw.items():
        for name, value in fields.items():
            reference=manifest[module+"/"+name]
            assert list(value.shape)==reference["shape"]
            assert str(value.dtype)==reference["dtype"]
            digest=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            assert digest==reference["sha256"],(module,name)
            arrays+=1;words+=value.size;integers+=int(value.dtype.kind=="i")
    print(spectrum,arrays,"raw fields",words,"words;",integers,
          "integer fields; all shapes/types/byte digests match Fortran authority",flush=True)
