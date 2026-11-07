"""Compare v842 handoff and number carriers with host-compiled selected source."""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_physics_v841 import _MPAS_V842_SOURCE
from woof.hex.cuda_physics_prep_v841 import _MPAS_NUMBER_CARRIERS_V842_SOURCE
from woof.hex.mpasphys.metal_preparation_v842 import prepare_numbers
from woof.hex.mpasphys.metal_transport_v842 import source_baseline, handoff
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def equal(a, b, label):
    count = np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))
    if count: raise AssertionError(f'{label}: {count} differing words')


def main():
    module = HostModule('mpas_v842_transport', _MPAS_V842_SOURCE)
    number_src = _MPAS_NUMBER_CARRIERS_V842_SOURCE.replace('extern "C" {', '').replace('__global__ void prep_numbers', 'extern "C" __global__ void prep_numbers')
    number_src = number_src.rsplit('}', 1)[0]
    numbers = HostModule('mpas_v842_numbers', number_src).get_function('prep_numbers_v842_u32')
    base = module.get_function('mpas_transport_source_baseline_v842_f32')
    trans = module.get_function('mpas_transport_handoff_v842_f32')
    rng = np.random.default_rng(43)
    nz, nc, ns = 4, 7, 5
    before = rng.uniform(-.1, .1, (ns,nz,nc)).astype(np.float32)
    source = rng.uniform(-.001, .001, before.shape).astype(np.float32)
    rho = rng.uniform(.5, 2, (nz,nc)).astype(np.float32)
    after = rng.uniform(-.1, .1, before.shape).astype(np.float32)
    after.flat[0] = np.float32(-2**-149)
    after.flat[1] = np.float32(-0.)
    theta = rng.uniform(250, 320, rho.shape).astype(np.float32)
    rvord = np.float32(np.float32(461.6)/np.float32(287.))
    for active in (False, True):
        for mono in (False, True):
            expected_base = np.empty_like(before)
            base((2,), (128,), (nz,nc,nc-1,ns,np.float32(2),before,source,rho,expected_base))
            expected = after.copy(); q = np.zeros_like(rho); th = np.ones_like(rho)
            trans((1,), (128,), (nz,nc,nc-1,ns,0,int(active),int(mono),np.float32(2),np.float32(461.6),np.float32(287),rvord,expected_base,expected,theta,q,th))
            with MetalContext() as context:
                xp = MetalArrayModule(context); upload=lambda x: xp.asarray(x.copy())
                m_before, m_source, m_rho = map(upload, (before,source,rho))
                baseline = source_baseline(before=m_before,source=m_source,rho_old=m_rho,dt=2,n_cells_solve=nc-1)
                m_after, m_theta, m_q, m_th = map(upload,(after,theta,np.zeros_like(rho),np.ones_like(rho)))
                handoff(before=baseline,after=m_after,theta_m=m_theta,scalar_names=('qv','qc','qi','qs','ni'),rqvdynten=m_q,rthdynten=m_th,convection_scheme='cu_grell_freitas' if active else 'off',monotonic=mono,dt=2,n_cells_solve=nc-1)
                for label, x, y in [('baseline',expected_base,baseline.to_numpy()),('after',expected,m_after.to_numpy()),('q',q,m_q.to_numpy()),('theta',th,m_th.to_numpy())]: equal(x,y,label)
    names=('nc','ni','nifa','nwfa')
    values={n:rng.uniform(0,100,(nz,nc)).astype(np.float32) for n in names}
    for aerosol in (False,True):
        for mynn in (False,True):
            for ice in (False,True):
                for trio in (False,True):
                    if not (aerosol or mynn): continue
                    outputs=[np.zeros((nz,nc),np.uint32) for _ in names]
                    numbers((1,), (128,), (nz,nc,nc-1,int(aerosol),int(mynn),int(ice),int(trio),*[values[n].view(np.uint32) for n in names],*outputs))
                    with MetalContext() as context:
                        xp=MetalArrayModule(context)
                        result=prepare_numbers(species={'qv':xp.asarray(rho),**{n:xp.asarray(v) for n,v in values.items()}},microphysics='mp_thompson_aerosols' if aerosol else 'mp_thompson',pbl='bl_mynn' if mynn else 'off',scalar_flags={'ni':ice,'nc':trio,'nifa':trio,'nwfa':trio},n_cells_solve=nc-1)
                        for n, expected in zip(names,outputs):
                            if n in result: equal(expected,result[n].to_numpy(),n)
    print('transport and number carriers match host-selected source word for word')

if __name__ == '__main__': main()
