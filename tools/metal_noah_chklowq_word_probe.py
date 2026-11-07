#!/usr/bin/env python3
"""Compare CHKLOWQ with host-compiled source statements from the MPAS Noah driver.

Run device work through the GPU queue. The extracted statements originate in
module_sf_noahdrv.F and retain their float32 order and conditional update.
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.core.noah import noah_chklowq_myj
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    source = (ROOT / 'woof/hex/mpasphys/noah_v842.py').read_text()
    def line(marker):
        return next(row.split('//')[0].strip() for row in source.splitlines()
                    if marker in row)
    q2k = line('module_sf_noahdrv.F:794')
    q2sat = line('module_sf_noahdrv.F:803')
    condition = line('module_sf_noahdrv.F:806')
    assert q2k.startswith('Q2K = ') and q2sat.startswith('Q2SAT = ')
    assert condition.startswith('if (((MYJ && ')
    host_source = '''
#define AD(a,b) ((a)+(b))
#define MU(a,b) ((a)*(b))
#define DV(a,b) ((a)/(b))
extern "C" __global__ void oracle(const float *qv, const float *qgh,
    const float *qz0, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float QV3D[1] = {qv[i]}, QGH = qgh[i], QZ0 = qz0[i];
    float Q2K, Q2SAT, CHKLOWQ = out[i];
    const float TRESH = 0x1.e66666p-1f;
    const bool MYJ = true;
''' + q2k + '\n' + q2sat + '\n' + condition + '''
    CHKLOWQ = 0.0f;
    }
    out[i] = CHKLOWQ;
}
'''
    oracle = HostModule('noah_chklowq_word_probe', host_source).get_function('oracle')
    n = 4096
    rng = np.random.default_rng(42)
    qv = rng.uniform(-0.01, 0.1, n).astype(np.float32)
    qgh = rng.uniform(-0.01, 0.1, n).astype(np.float32)
    qz0 = rng.uniform(-0.01, 0.1, n).astype(np.float32)
    special = np.array([0, 0x80000000, 1, 0x80000001, 0x7f800000,
                        0xff800000, 0x7fc12345, 0xbf800000], np.uint32).view(np.float32)
    qv[:len(special)] = special
    qgh[len(special):2*len(special)] = special
    qz0[2*len(special):3*len(special)] = special
    # Equality and adjacent threshold and upper-bound controls.
    qv[24:30] = np.float32(.01)
    qgh[24:30] = np.float32(.01)
    q2 = np.float32(qv[24] / np.float32(1 + qv[24]))
    qz0[24:27] = [np.nextafter(q2, -np.inf, dtype=np.float32), q2,
                   np.nextafter(q2, np.inf, dtype=np.float32)]
    qz0[27:30] = np.float32(.1)
    one = np.float32(1.)
    threshold = (qgh[27] / (one + qgh[27])) * np.float32(.95)
    lower = threshold / (one - threshold)
    qv[27:30] = [np.nextafter(lower, -np.inf, dtype=np.float32), lower,
                  np.nextafter(lower, np.inf, dtype=np.float32)]
    lower_q2 = qv[27:30] / (one + qv[27:30])
    assert lower_q2[0] < threshold == lower_q2[1] < lower_q2[2]
    initial = np.resize(np.array([0x3f800000, 0x80000000, 1, 0x7fc12345],
                                 np.uint32), n).view(np.float32).copy()
    expected = initial.copy()
    oracle(((n + 63)//64,), (64,), (qv, qgh, qz0, expected, np.int32(n)))
    with MetalContext(batch_kernel_calls=False) as context:
        xp = MetalArrayModule(context)
        dev = {k: xp.asarray(v.reshape(1, n)) for k, v in
               (('qv1', qv), ('qgh', qgh), ('qz0', qz0), ('chklowq', initial))}
        noah_chklowq_myj(dev)
        got = dev['chklowq'].to_numpy().reshape(-1)
    bad = np.flatnonzero(got.view(np.uint32) != expected.view(np.uint32))
    print(f'{n} words, {bad.size} mismatches; indices {bad[:20].tolist()}')
    if bad.size:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
