"""Compare RRFS TEMPO/MYNN cloud carriers to the retained CUDA source."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrfs.radiation_rrfs import CUDA_SOURCE
from woof.hex.mpasphys.rrfs.metal_cloudiness import cloudiness
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

start = CUDA_SOURCE.index('extern "C" __global__ void cloud_from(')
end = CUDA_SOURCE.index('// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F', start)
header = (ROOT / 'woof/core/kernels/glibc_flt32.cuh').read_text()
host = HostModule('rrfs_cloudiness', header + '\n' + CUDA_SOURCE[start:end])
source_from, source_cond = (host.get_function(n) for n in ('cloud_from', 'cloud_condensate'))
nz, nc = 3, 5
shape = (nz, nc)
# Straddle both fraction and condensate thresholds; exercise sequential additions.
qc = np.array([0, 1e-6, 0, 2e-6, 0, 0, 1e-7, 1e-6, 0, 2e-6, 0, 0, 1e-7, 0, 0], np.float32).reshape(shape)
qi = np.array([0, 0, 1e-9, 0, 2e-9] * 3, np.float32).reshape(shape)
profiles = {'qv': np.full(shape, .01, np.float32), 'qc': qc, 'qi': qi,
            'qs': np.array([0, 0, 0, 1e-8, 2e-8] * 3, np.float32).reshape(shape)}
diag = {'cldfrac_bl': np.array([0, .001, .002, .7, .99] * 3, np.float32).reshape(shape),
        'cldfrac_cu': np.array([.5, .002, .9, .5, .001] * 3, np.float32).reshape(shape),
        'qc_bl': np.full(shape, 4e-7, np.float32), 'qi_bl': np.full(shape, 4e-10, np.float32),
        'qc_cu': np.full(shape, 8e-7, np.float32), 'qi_cu': np.full(shape, 8e-10, np.float32)}
density = np.array([.6, .9, 1, 1.5, 2.7], np.float32)
land = np.array([1, 2, 1, 2, 1], np.float32)
local = {n: np.ascontiguousarray(v.T) for n, v in (profiles | diag).items()}
def expected_words(scheme, step, gfl):
    mynn = scheme == 'cld_fraction_mynn'
    carriers = [np.empty((nc, nz), np.float32) for _ in range(5)]
    dx, outland = np.empty(nc, np.float32), np.empty(nc, np.float32)
    source_from((1,), (128,), (nc, nz, np.float32(12000), int(mynn), step, int(gfl),
        density, land, *(local[n] for n in ('qv', 'qc', 'qi', 'qs', 'cldfrac_bl', 'cldfrac_cu')),
        dx, outland, *carriers))
    for suffix, enabled in (('bl', mynn), ('cu', gfl)):
        if enabled:
            source_cond((1,), (128,), (nc*nz, carriers[4], local['qc_'+suffix],
                        local['qi_'+suffix], carriers[1], carriers[2]))
    expected = {n: np.ascontiguousarray(v.T) for n, v in
                zip(('qvrad', 'qcrad', 'qirad', 'qsrad', 'cldfrac'), carriers)}
    expected.update(dx=dx, xland=outland)
    return expected


def main():
    with MetalContext(batch_kernel_calls=False) as context:
        submit = context._submit_kernel_batch

        def sync(*args, **kwargs):
            result = submit(*args, **kwargs)
            result.wait()
            return result

        context._submit_kernel_batch = sync
        xp = MetalArrayModule(context)
        for scheme in ('cld_tempo', 'cld_fraction_mynn'):
            for step in (0, 1):
                for gfl in (False, True):
                    expected = expected_words(scheme, step, gfl)
                    d = {n: xp.asarray(v) for n, v in diag.items()}
                    d['cldfrac'] = xp.asarray(np.full(shape, -1, np.float32))
                    out = cloudiness(scheme=scheme, itimestep=step,
                        convection_scheme='cu_grell_freitas_li' if gfl else 'off',
                        prepared={n: xp.asarray(v) for n, v in profiles.items()},
                        mesh_density=xp.asarray(density), xland=xp.asarray(land),
                        len_disp=12000, diag=d, radshared={})
                    for name, ref in expected.items():
                        np.testing.assert_array_equal(out[name].to_numpy().view(np.uint32),
                            ref.view(np.uint32), err_msg=f'{scheme} step={step} gfl={gfl}: {name}')
                    np.testing.assert_array_equal(d['cldfrac'].to_numpy().view(np.uint32),
                        expected['cldfrac'].view(np.uint32), err_msg='diagnostic')
    print('RRFS cloud carriers match retained source')


if __name__ == '__main__':
    main()
