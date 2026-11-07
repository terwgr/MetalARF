"""Compare the split driver with a captured native Noah driver column."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from woof.hex.mpasphys.metal_noah_v842 import run_driver_columns
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

p = argparse.ArgumentParser()
p.add_argument('fixture', type=Path)
a = p.parse_args()
metadata = json.loads((a.fixture / 'fixture.json').read_text())
for name, digest in metadata['files'].items():
    assert hashlib.sha256((a.fixture / name).read_bytes()).hexdigest() == digest
x = np.fromfile(a.fixture / 'input.f32', np.float32).reshape(1,79)
ix = np.fromfile(a.fixture / 'input.i32', np.int32).reshape(1,7)
tables = np.fromfile(a.fixture / 'tables.bin', np.uint32)
with MetalContext(batch_kernel_calls=False) as context:
    submit = context._submit_kernel_batch
    def sync(*args, **kwargs):
        result = submit(*args, **kwargs)
        result.wait()
        return result
    context._submit_kernel_batch = sync
    xp = MetalArrayModule(context)
    dx, di, dt = map(xp.asarray, (x, ix, tables))
    status, routes = run_driver_columns(dx, di, dt, xp.asarray(np.ones((1,2),np.int32)))
    actual = dx.to_numpy()
    actual_i = di.to_numpy()
    st, rt = status.to_numpy(), routes.to_numpy()
expected = np.fromfile(a.fixture / 'native.f32', np.float32).reshape(1,79)
expected_i = np.fromfile(a.fixture / 'native.i32', np.int32).reshape(1,7)
float_mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
int_mismatch = np.flatnonzero(actual_i != expected_i)
print(json.dumps({'float_words':int(actual.size), 'float_mismatches':float_mismatch.tolist(),
                  'int_words':int(actual_i.size), 'int_mismatches':int_mismatch.tolist(),
                  'status':st.tolist(), 'route':rt.tolist()}))
if st.any() or float_mismatch.size or int_mismatch.size:
    raise SystemExit(1)
