"""Compare Metal tendency conversion with host-compiled selected CUDA source."""
from pathlib import Path
import ctypes
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_physics_v841 import _MPAS_V842_SOURCE
from woof.hex.mpasphys.metal_tendencies_v842 import RATE_NAMES, get_tend
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def run():
    module = HostModule('mpas_v842_tendencies', _MPAS_V842_SOURCE)
    cells = module.get_function('mpas_rk_cells_v842_f32')
    edges = module.get_function('mpas_rk_edges_v842_f32')
    rng = np.random.default_rng(2842)
    nz, nc, ne, ns = 4, 7, 9, 5
    names = ('qv', 'qc', 'qi', 'qs', 'ni')
    selector = dict(config_pbl_scheme='bl_mynn', config_convection_scheme='cu_grell_freitas',
                    config_microp_scheme='mp_thompson', config_radt_lw_scheme='rrtmg_lw',
                    config_radt_sw_scheme='rrtmg_sw')
    for subnormal in (False, True):
        a = lambda shape: rng.uniform(-0.2, 0.2, shape).astype(np.float32)
        mass = rng.uniform(.5, 2., (nz, nc)).astype(np.float32)
        mass_edge = rng.uniform(.5, 2., (nz, ne)).astype(np.float32)
        theta = rng.uniform(250, 320, (nz, nc)).astype(np.float32)
        scalars = a((ns, nz, nc)); scalars[0] = np.abs(scalars[0])
        rates = [a((nz, nc)) for _ in RATE_NAMES]
        if subnormal:
            rates = [x * np.float32(2**-127) for x in rates]
        pbl_edge, cu_edge, initial_u = a((nz, ne)), a((nz, ne)), a((nz, ne))
        if subnormal:
            pbl_edge *= np.float32(2**-127)
        pidx = np.array([0, 1, 2, 3, 4, -1, -1, -1], np.int32)
        cidx = np.array([0, 1, 2, -1, -1], np.int32)
        output = [np.zeros_like(mass), np.zeros_like(mass_edge), np.zeros_like(mass),
                  np.zeros_like(scalars)]
        # Host launch receives the original CUDA pointer table, unchanged.
        pointers = (ctypes.c_void_p * len(rates))(*(x.ctypes.data for x in rates))
        args = [nz, nc, nc-1, ns, 1, 1, 1, 1, pidx, cidx,
                np.float32(np.float32(461.6)/np.float32(287.)), mass, theta,
                scalars[0], None, output[2], output[3], output[0]]
        converted = [ctypes.c_void_p(x.ctypes.data) if isinstance(x, np.ndarray)
                     else ctypes.cast(pointers, ctypes.c_void_p) if x is None else x for x in args]
        cells._fn(1, 1, 1, 128, 1, 1, *converted)
        u = initial_u.copy()
        edges((1,), (128,), (nz, ne, ne-1, 1, 0, 1, mass_edge, pbl_edge, cu_edge, output[1], u))
        with MetalContext() as context:
            xp = MetalArrayModule(context)
            upload = lambda x: xp.asarray(x.copy())
            metal_u = upload(initial_u)
            result = get_tend(mass=upload(mass), mass_edge=upload(mass_edge), theta_m=upload(theta),
                scalars=upload(scalars), scalar_names=names,
                tendencies={k: upload(v) for k, v in zip(RATE_NAMES, rates)}, selectors=selector,
                pbl_edge=upload(pbl_edge), convection_edge=upload(cu_edge), tend_u_phys=metal_u,
                n_cells_solve=nc-1, n_edges_solve=ne-1, rk_step=1, dynamics_substep=1)
            actual = [result.rho.to_numpy(), result.rho_u.to_numpy(), result.rho_theta.to_numpy(),
                      result.scalars.to_numpy(), metal_u.to_numpy()]
        for label, expected, got in zip(('rho', 'rho_u', 'rho_theta', 'scalars', 'tend_u'),
                                        [*output, u], actual):
            unequal = np.count_nonzero(expected.view(np.uint32) != got.view(np.uint32))
            if unequal:
                raise AssertionError(f'{label}: {unequal} differing words; subnormal={subnormal}')
        print(f'host-selected CUDA source vs Metal: exact words, subnormal={subnormal}')


if __name__ == '__main__':
    run()
