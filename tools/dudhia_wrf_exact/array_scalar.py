"""Exercise scalar Metal uploads and copies with their NumPy shapes."""
import numpy as np
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
with MetalContext() as context:
    xp = MetalArrayModule(context)
    for dtype in (np.float32, np.float64, np.int32, np.uint32):
        source = np.asarray(3, dtype=dtype)
        device = xp.asarray(source)
        target = xp.zeros((), dtype=dtype)
        target[...] = device
        assert device.shape == target.shape == ()
        np.testing.assert_array_equal(target.to_numpy(), source)
print('four 0-D dtypes upload and copy with scalar shape')
