"""Compare native Metal Thompson radar with the host-built CUDA source."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys import thompson_v842 as native
from woof.hex.mpasphys.resources_build import thompson_maths, _vendored_maths
from woof.hex.mpasphys.metal_thompson_column_v842 import launch_radar
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

nz, nc = 8, 35
source = f'#define THOMPSON_COLUMN_KMAX {nz}\n' + thompson_maths(_vendored_maths())
source += '\n' + native._SUPPORT + native._COLUMN + native._RADAR_DRIVER
host = HostModule('native_thompson_radar', source).get_function('mpas_thompson_radar')
names = ('qv', 'qc', 'qr', 'nr', 'qs', 'qg', 'th', 'pii', 'p', 'z', 'dz')
fields = {name: np.zeros((nz, nc), np.float32) for name in names}
for name, value in dict(qv=.005, qc=.0002, qr=.0001, nr=2000, qs=.00002, qg=.00001, pii=1, p=90000, dz=500).items():
    fields[name][:] = value
fields['th'][:] = np.array([296, 289, 280, 274, 270, 265, 250, 230], np.float32)[:, None]
fields['z'][:] = np.arange(nz, dtype=np.float32)[:, None] * 500
for col in range(nc):
    if col % 4 == 0:
        for name in ('qc', 'qr', 'qs', 'qg'):
            fields[name][:, col] = 0
    if col % 4 == 1:
        fields['qs'][:, col] = fields['qg'][:, col] = 0
    if col % 4 == 2:
        fields['qr'][:, col] = 0
    for name in ('qc', 'qr', 'qs', 'qg'):
        fields[name][:, col] *= np.float32(1 + col / 10)
targets = ('refl10cm', 'refl10cm_max', 'refl10cm_1km', 'refl10cm_1km_max')
expected = {name: np.full((nz, nc) if name == 'refl10cm' else (nc,), -35., np.float32) for name in targets}
host(((nc + 31) // 32,), (32,), tuple(fields[n] for n in names) + tuple(expected[n] for n in targets) + (np.int32(nz), np.int32(nc)))
with MetalContext(batch_kernel_calls=False) as context:
    submit = context._submit_kernel_batch
    def synchronous(*args, **kwargs):
        result = submit(*args, **kwargs)
        result.wait()
        return result
    context._submit_kernel_batch = synchronous
    xp = MetalArrayModule(context)
    inputs = {n: xp.asarray(v) for n, v in fields.items()}
    output = {n: xp.asarray(np.full_like(v, -35.)) for n, v in expected.items()}
    launch_radar(inputs, output)
    words = differences = 0
    for name, wanted in expected.items():
        actual = output[name].to_numpy()
        bad = np.flatnonzero(actual.view(np.uint32) != wanted.view(np.uint32))
        print(name, 'words', wanted.size, 'differences', len(bad),
              [(int(i), hex(int(actual.view(np.uint32).flat[i])), hex(int(wanted.view(np.uint32).flat[i]))) for i in bad[:5]], flush=True)
        words += wanted.size
        differences += len(bad)
    for name, wanted in fields.items():
        assert inputs[name].to_numpy().tobytes() == wanted.tobytes(), name
    print('native Thompson radar', words, 'words;', differences, 'differences; inputs preserved', flush=True)
    if differences:
        raise SystemExit(1)
