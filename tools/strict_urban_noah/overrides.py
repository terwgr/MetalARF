"""Check the Noah-only urban diagnostic copies, including untouched cells."""
import numpy as np
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.urban_ucm_family import submit_ucm_noah_overrides
from woof.wrf_exact import ENABLED

if not ENABLED:
    raise RuntimeError('run with GPUWM_WRF_EXACT=1')
shape = (3, 5)
mask = np.array([0, 1, -1, 0, 1] * 3, np.int32).reshape(shape)
names = ('u10', 'v10', 'psim', 'psih', 'gz1oz0', 'akms')
# Include signed zero, subnormals and finite extremes as raw words.
words = np.array([0, 0x80000000, 1, 0x80000001, 0x7f7fffff,
                  0xff7fffff, 0x3f800000, 0xbf800000, 0x00800000,
                  0x80800000, 0x3eaaaaab, 0x7f000000, 0x3c000001,
                  0x40000000, 0xc0000000], np.uint32).reshape(shape)
with MetalContext() as context:
    xp = MetalArrayModule(context)
    host = {'urban': mask}
    for i, name in enumerate(names):
        host['state_' + name] = np.roll(words, i, axis=1).copy().view(np.float32)
        host['field_' + name] = np.full(shape, i + 20, np.float32)
    host['field_chs'] = np.roll(words, 3, axis=0).copy().view(np.float32)
    host['field_akhs'] = np.full(shape, 29, np.float32)
    device = {name: xp.asarray(value) for name, value in host.items()}
    submit_ucm_noah_overrides(context, device)
    count = 0
    for name in (*names, 'akhs'):
        source = host['field_chs'] if name == 'akhs' else host['state_' + name]
        expected = np.where(mask != 0, source.view(np.uint32),
                            host['field_' + name].view(np.uint32))
        actual = device['field_' + name].to_numpy().view(np.uint32)
        count += expected.size
        if not np.array_equal(expected, actual):
            raise AssertionError(f'{name}: {np.count_nonzero(expected != actual)} word differences')
print(f'Noah urban diagnostic copies: 0/{count} word differences; masked cells and subnormals covered')
