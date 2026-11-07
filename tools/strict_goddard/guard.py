"""Verify the public Metal Goddard adapter refuses strict calls before work."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from woof.core.radiation_composition import ColumnTableRadiation
from woof.wrf_exact import ENABLED

assert ENABLED
adapter = ColumnTableRadiation(datetime(2024, 1, 1), np.zeros((1, 1), np.float32),
                               np.zeros((1, 1), np.float32))
state = SimpleNamespace(_metal_setup_state=True, elapsed_seconds=0.)
fields = {'glw': np.array([[300.]], np.float32)}
before = fields['glw'].copy()
with patch('woof.metal.goddard.MetalArrayModule') as arrays, \
     patch('woof.metal.goddard.interface_temperature') as interface, \
     patch('woof.metal.goddard.solve_chunk') as solve:
    try:
        adapter(atmosphere={}, fields=fields, state=state,
                cfg=SimpleNamespace(icloud=0, mp_physics=1, bl_pbl_physics=0))
    except NotImplementedError as error:
        assert 'strict Metal Goddard radiation' in str(error)
    else:
        raise AssertionError('unsupported strict Goddard call was accepted')
    assert arrays.call_count == interface.call_count == solve.call_count == 0
assert adapter.update_count == 0 and np.array_equal(fields['glw'], before)
print('strict direct Goddard refused before allocation, flux work or publication')
