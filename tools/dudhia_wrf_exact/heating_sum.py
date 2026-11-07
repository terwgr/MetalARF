"""Exercise the production radiation sum with rounded binary32 operands."""
import os
from types import SimpleNamespace
import numpy as np

os.environ.setdefault('GPUWM_WRF_EXACT', '1')
from tools.thompson_real_column_parity import host_backend
host_backend.install()
from woof.core.physics import _sum_radiation_rates
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

original = MetalContext._submit_kernel_batch
def sync(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if result is not None:
        result.wait()
    return result
MetalContext._submit_kernel_batch = sync

rng = np.random.default_rng(42)
words = rng.integers(0, 2**32, size=(2, 4096), dtype=np.uint32)
# Stay finite while covering signed zero, subnormals, cancellation and ties.
words &= np.uint32(0xfeffffff)
words[:, :8] = np.array([[0, 0x80000000, 1, 0x80000001, 0x3f800000, 0x3f800000, 0x00800000, 0x80800000],
                         [0x80000000, 0, 1, 1, 0x33800000, 0xbf800000, 0x807fffff, 0x007fffff]], np.uint32)
lw, sw = words.view(np.float32)
with np.errstate(over='ignore', invalid='ignore'):
    total = np.add(lw, sw, dtype=np.float32)
    residual = np.subtract(total, lw, dtype=np.float32)
with MetalContext() as context:
    xp = MetalArrayModule(context)
    longwave, shortwave = xp.asarray(lw), xp.asarray(sw)
    state = SimpleNamespace(_metal_setup_state=True, _array_module=xp)
    held = _sum_radiation_rates(state, longwave, shortwave)
    np.testing.assert_array_equal(held.to_numpy().view(np.uint32), total.view(np.uint32))
    np.testing.assert_array_equal(shortwave.to_numpy().view(np.uint32), sw.view(np.uint32))
    actual = _sum_radiation_rates(state, longwave, shortwave, publish_residual=True)
    np.testing.assert_array_equal(actual.to_numpy().view(np.uint32), total.view(np.uint32))
    np.testing.assert_array_equal(shortwave.to_numpy().view(np.uint32), residual.view(np.uint32))
print('matched', 2 * lw.size, 'total and residual words; host NumPy binary32')
