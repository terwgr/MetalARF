"""Compare the actual cloud-fraction caller with its host source graph."""
import numpy as np
from woof.core.rrtmgp import cal_cldfra1
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED

assert ENABLED
rng = np.random.default_rng(7149)
shape = (7, 37)
qv = rng.uniform(0., .02, shape).astype(np.float32)
qc = rng.uniform(0., 4.e-4, shape).astype(np.float32)
qi = rng.uniform(0., 2.e-4, shape).astype(np.float32)
qs = rng.uniform(0., 1.e-4, shape).astype(np.float32)
t = rng.uniform(215., 310., shape).astype(np.float32)
p = rng.uniform(25000., 100000., shape).astype(np.float32)
small = np.array([0, 1, 0x80000001, 0x007fffff, 0x00800000], np.uint32).view(np.float32)
for value in (qv, qc, qi, qs):
    value.flat[:small.size] = small
qc.flat[5:8] = np.float32(0.0)
qv.flat[8:10] = np.float32(.03)
t.flat[10:12] = np.float32(233.0)
inputs = (qv, qc, qi, qs, t, p)
with MetalContext() as context:
    xp = MetalArrayModule(context)
    device = tuple(xp.asarray(value) for value in inputs)
    count = 0
    for label, fqi, fqs, mp in [('liquid', False, False, None),
                                 ('ice', True, False, None),
                                 ('ice-snow', True, True, None),
                                 ('ferrier', True, False, 5)]:
        expected = cal_cldfra1(*inputs, f_qi=fqi, f_qs=fqs, mp_physics=mp)
        actual = cal_cldfra1(*device, f_qi=fqi, f_qs=fqs, mp_physics=mp).to_numpy()
        a, b = actual.view(np.uint32), expected.view(np.uint32)
        bad = np.flatnonzero(a.ravel() != b.ravel()); count += a.size
        if len(bad):
            i=bad[0]; print(label, len(bad), 'first', i,
                            'host', hex(int(b.ravel()[i])), 'metal', hex(int(a.ravel()[i])))
            raise SystemExit(1)
        print(label, f'0/{a.size} word differences, active={np.count_nonzero(expected)}, fractional={np.count_nonzero((expected > 0) & (expected < 1))}')
print(f'{count} words matched the host cal_cldfra1 source expressions and pinned GFK')
