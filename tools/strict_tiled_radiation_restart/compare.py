from types import SimpleNamespace
from woof.io.restart import _validated_swdd_marker, RestartMismatchError
from tilestream.checkpoint import restart_scalars, driver_restart_header
from tilestream.physics_inventory import carrier_scalars, set_carrier_scalars
for value in (False, True):
    driver = SimpleNamespace(_swdd_from_scheme=value, call_counts={'radiation':1}, ysu_nan_guard_fires=0, microphysics_updates=0, pbl_raw_rates={}, carriers=None)
    state = SimpleNamespace(elapsed_seconds=20., physics=driver)
    header = {'elapsed_seconds':20., 'driver':driver_restart_header(carrier_scalars(state))}
    restored = restart_scalars(header, {'fields/swddir':object()})
    driver._swdd_from_scheme = not value
    set_carrier_scalars(state, restored)
    assert driver._swdd_from_scheme is value
assert _validated_swdd_marker({'call_counts':{'radiation':0}}, {'fields/swddir':None}) is False
assert _validated_swdd_marker({'call_counts':{'radiation':1}}, {}) is False
for header, stored in [({'call_counts':{'radiation':1}}, {'fields/swddir':None}), ({'call_counts':{}}, {'fields/swddir':None}), ({'swdd_from_scheme':None}, {}), ({'swdd_from_scheme':True}, {})]:
    try:
        _validated_swdd_marker(header, stored)
    except RestartMismatchError:
        pass
    else:
        raise AssertionError((header, stored))
print('marker roundtrip and legacy validation passed')
