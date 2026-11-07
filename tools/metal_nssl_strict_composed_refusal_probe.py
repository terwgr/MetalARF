"""Verify unported strict NSSL compositions refuse before device mutation."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from woof.wrf_exact import ENABLED
from woof.core.nssl2_driver_support import NSSL2DriverWorkspace, gather_initialize_and_sediment
from woof.core.nssl2_fused_gs import launch_fused_gs
from woof.core.nssl2_runtime import apply_nssl2_production
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

if not ENABLED:
    raise SystemExit("GPUWM_WRF_EXACT=1 is required")

with MetalContext() as context:
    xp = MetalArrayModule(context)
    shape = (2, 1, 2)
    fields = [xp.asarray(np.full(shape, i + 1, np.float32)) for i in range(19)]
    slab = xp.asarray(np.full((16, *shape), -7, np.float32))
    exports = xp.asarray(np.full((5, *shape[1:]), -8, np.float32))
    ignored = xp.asarray(np.full(shape[1:], -9, np.float32))
    ws = NSSL2DriverWorkspace(slab, exports, shape, ignored)
    w = xp.asarray(np.full((3, 1, 2), 1, np.float32))
    tracked = [*fields, w, slab, exports, ignored]
    before = [x.to_numpy().view(np.uint32).copy() for x in tracked]
    def refused(call):
        try:
            call()
        except NotImplementedError:
            pass
        else:
            raise AssertionError("strict composition did not refuse")
        for original, array in zip(before, tracked):
            assert np.array_equal(original, array.to_numpy().view(np.uint32))
    refused(lambda: gather_initialize_and_sediment(
        fields[0], fields[1], *fields[2:18], 1.0,
        temperature_k=fields[18], workspace=ws))
    refused(lambda: launch_fused_gs(ws, *fields[:4], w, *fields[5:8], 1.0))
    refused(lambda: apply_nssl2_production(SimpleNamespace(p=fields[0]), None, 1.0, None))
    print("three strict composed callers refused before all tracked device writes")
