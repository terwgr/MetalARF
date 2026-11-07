"""Compare strict radiation carrier and mass coupling caller words."""
import os
from types import SimpleNamespace
import numpy as np

os.environ.setdefault('GPUWM_WRF_EXACT', '1')
from tools.thompson_real_column_parity import host_backend
host_backend.install()
from woof.core.physics import PhysicsDriver, _metal_radiation_mass, couple_column_tendencies
from woof.core import tendency_coupling
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

original = MetalContext._submit_kernel_batch
def sync(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if result is not None:
        result.wait()
    return result
MetalContext._submit_kernel_batch = sync

host = host_backend.HostModule('couple_mass_reference', tendency_coupling._SOURCE).get_function('couple_mass_rates')
rng = np.random.default_rng(17)
nz, ny, nx = 4, 5, 7
shape, plane = (nz, ny, nx), (ny, nx)
mut = rng.uniform(.7, 1.5, plane).astype(np.float32)
c1 = np.array([.4, .6, .8, 1.], np.float32)
c2 = np.array([.2, .3, .5, .7], np.float32)
rate = rng.uniform(-2, 2, shape).astype(np.float32)
rate.flat[:6] = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff], np.uint32).view(np.float32)
msf = np.full(plane, 1.7, np.float32)
mass = np.add(np.multiply(c1[:, None, None], mut[None], dtype=np.float32), c2[:, None, None], dtype=np.float32)
reference = np.empty(shape, np.float32)
dummy = np.zeros(shape, np.float32)
n = rate.size
host(((n+255)//256,), (256,), (c1, c2, mut, msf, rate, dummy, dummy, dummy, dummy, dummy,
      reference, dummy, dummy, dummy, dummy, dummy, np.uint32(1), np.uint32(1), np.int32(1),
      np.int32(0), dummy, dummy, dummy, dummy, np.int32(0), np.int32(0), np.int32(0),
      np.int64(nz), np.int64(ny), np.int64(nx)))
with MetalContext() as context:
    xp = MetalArrayModule(context)
    d = {k: xp.asarray(v) for k, v in dict(p=rate, c1h=c1, c2h=c2, msft=msf, mut=mut, rate=rate).items()}
    state = SimpleNamespace(**d, _metal_setup_state=True, _array_module=xp, has_msf=True)
    state.msfu = xp.asarray(np.full((ny, nx+1), 1.7, np.float32))
    state.msfv = xp.asarray(np.full((ny+1, nx), 1.7, np.float32))
    state.total_mu = lambda: d['mut']
    state.existing_scratch = lambda _: None
    state.scratch = lambda extents, _: xp.zeros(extents, np.float32)
    coupled = couple_column_tendencies(state, SimpleNamespace(specified=False, nested=False), rtheta=d['rate'])
    np.testing.assert_array_equal(coupled.rtheta.to_numpy().view(np.uint32), reference.view(np.uint32))
    multiplied = _metal_radiation_mass(state, d['rate'], d['mut'], divide=False)
    np.testing.assert_array_equal(multiplied.to_numpy().view(np.uint32), np.multiply(rate, mass, dtype=np.float32).view(np.uint32))
    driver = SimpleNamespace(state=state, _wrf_rthraten=multiplied, _wrf_rthraten_coupled=True)
    actual = PhysicsDriver.radiation_theta_forcing(driver)
    expected = np.divide(np.multiply(rate, mass, dtype=np.float32), mass, dtype=np.float32)
    np.testing.assert_array_equal(actual.to_numpy().view(np.uint32), expected.view(np.uint32))
    carried_mut = mut + np.float32(.25)
    state.existing_scratch = lambda _: xp.asarray(carried_mut)
    actual = PhysicsDriver.radiation_theta_forcing(driver)
    carried_mass = np.add(np.multiply(c1[:, None, None], carried_mut[None], dtype=np.float32),
                          c2[:, None, None], dtype=np.float32)
    expected = np.divide(np.multiply(rate, mass, dtype=np.float32), carried_mass, dtype=np.float32)
    np.testing.assert_array_equal(actual.to_numpy().view(np.uint32), expected.view(np.uint32))
print('matched', 4*n, 'mass coupling, carrier multiply and forcing division words; strict mapped grid and differing carried mass; CUDA coupling reference flags', host_backend.CXX_FLAGS)
