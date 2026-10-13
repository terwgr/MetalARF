"""Exercise the MP8 post-RK -> next Noah-call seam on native Metal arrays."""
from types import SimpleNamespace
import os
import sys

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(sys.platform != "darwin", reason="requires native Metal")
@pytest.mark.skipif(os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"),
                    reason="local GPU work disabled")
def test_thompson_rain_delivered_once_then_cleared(monkeypatch):
    from woof.core import physics
    from woof.core.microphysics import MicrophysicsDiagnostics
    from woof.core.physics_inventory import microphysics_scratch_slots
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext

    monkeypatch.setattr(physics, "cp", None)
    shape = (2, 3)
    first = np.array([[0, 0.25, 1], [2, 0, 4]], dtype=np.float32)
    second = np.array([[0.5, 0, 2], [0, 3, 0.25]], dtype=np.float32)
    external = np.full(shape, 0.125, dtype=np.float32)
    delivered = []

    def noah_probe(fields, *args, **kwargs):
        # Read at the real Noah entrypoint, after the native forcing kernel.
        delivered.append(fields["rainbl"].to_numpy().copy())
        assert kwargs["frpcpn"] is True

    monkeypatch.setattr(physics, "launch_noah", noah_probe)
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        # Repeat independent cycles to exercise the usual batched runtime.
        for repeat in range(3):
            driver = physics.PhysicsDriver.__new__(physics.PhysicsDriver)
            driver.state = SimpleNamespace(
                _metal_setup_state=True, _array_module=xp,
                mup=xp.zeros(shape, np.float32),
                p=xp.zeros((2, *shape), np.float32))
            driver.mp_physics = 8
            driver._metal_gfs_profile = False
            driver.microphysics_updates = 0
            driver.microphysics = SimpleNamespace(**{
                name: xp.zeros(shape, np.float32)
                for name, _ in microphysics_scratch_slots(8)})
            result = MicrophysicsDiagnostics(**vars(driver.microphysics))
            driver._pending_rainbl = xp.zeros(shape, np.float32)
            driver.fields = {name: xp.zeros(shape, np.float32) for name in
                             ("tsk", "br", "sfcprs", "sfctmp", "qv1",
                              "dz8w1", "rib", "sr", "rainbl")}
            driver.fields["rainbl"][...] = external
            driver.noah_params = None
            driver.bldt_seconds = 30.0
            driver.noah_usemonalb = False
            driver.noah_rdlai2d = False
            driver.noah_opt_thcnd = 1
            driver.urban_coupler = None
            atmosphere = {name: xp.asarray(np.full(dims, value, np.float32))
                          for name, dims, value in (
                              ("p_interface", (3, *shape), 100000),
                              ("temperature", (2, *shape), 280),
                              ("qv", (2, *shape), 0.01),
                              ("dz", (2, *shape), 50))}
            for increment in (first, second):
                result.rainncv[...] = increment
                driver.accept_microphysics(result, dt=15.0)
            np.testing.assert_array_equal(driver._pending_rainbl.to_numpy(),
                                          first + second)
            assert driver.microphysics_updates == 2
            driver._run_noah(atmosphere, SimpleNamespace(), 1)
            np.testing.assert_array_equal(delivered[-1], external + first + second)
            for carrier in (driver.fields["rainbl"], driver._pending_rainbl):
                np.testing.assert_array_equal(carrier.to_numpy(), np.zeros(shape))
            # Clearing carriers must preserve the canonical MP diagnostics.
            np.testing.assert_array_equal(result.rainncv.to_numpy(), second)
            driver._run_noah(atmosphere, SimpleNamespace(), 2)
            np.testing.assert_array_equal(delivered[-1], np.zeros(shape))
            result.rainncv[...] = first
            driver.accept_microphysics(result, dt=15.0)
            driver._run_noah(atmosphere, SimpleNamespace(), 3)
            np.testing.assert_array_equal(delivered[-1], first)
            for carrier in (driver.fields["rainbl"], driver._pending_rainbl):
                np.testing.assert_array_equal(carrier.to_numpy(), np.zeros(shape))
            assert len(delivered) == (repeat + 1) * 3
