"""Compare real resident held-source callback against its selected source."""
import ctypes
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_physics_v841 import _MPAS_V842_SOURCE
from woof.hex.mpasphys.metal_tendencies_v842 import RATE_NAMES


def exercise(session, result, *, pbl=False):
    p=session.pools; d=session.layout.dimensions
    nz,nc,ne=d['nVertLevels'],d['nCellsSolve'],d['nEdgesSolve']
    names=session.resident_preparation.names; ns=len(names)
    mass=p.field('state','rho_zz',level=2).to_numpy()
    theta=p.field('state','theta_m').to_numpy()
    scalars=p.field('state','scalars').to_numpy()
    edge_mass=p.field('diag','rho_edge').to_numpy()
    rate=[np.zeros((nz,nc),np.float32) for _ in RATE_NAMES]
    needed=('rthblten','rqvblten','rqcblten','rqiblten','rqsblten','rniblten') if pbl else ('rthratensw',)
    for name in needed:
        rate[RATE_NAMES.index(name)]=p.field('tend_physics',name).to_numpy()
    pidx=np.full(8,-1,np.int32); pidx[0]=names.index('qv')
    if pbl:
        for i,name in enumerate(('qv','qc','qi','qs','ni')): pidx[i]=names.index(name)
    cidx=np.full(5,-1,np.int32); cidx[0]=names.index('qv')
    rho=np.zeros_like(mass); theta_out=np.zeros_like(mass); scalar_out=np.zeros_like(scalars)
    u=np.zeros((nz,ne),np.float32); edge=np.zeros_like(u); tend_u=np.zeros_like(u)
    pbl_edge=p.field('tend_physics','rublten_Edge').to_numpy() if pbl else edge
    ptr=(ctypes.c_void_p*len(rate))(*(x.ctypes.data for x in rate))
    args=[nz,nc,nc,ns,int(pbl),0,0,int(not pbl),pidx,cidx,np.float32(np.float32(461.6)/np.float32(287.)),
          mass,theta,scalars[names.index('qv')],None,theta_out,scalar_out,rho]
    values=[ctypes.c_void_p(x.ctypes.data) if isinstance(x,np.ndarray) else ctypes.cast(ptr,ctypes.c_void_p) if x is None else x for x in args]
    host=HostModule('mpas_v842_tendencies',_MPAS_V842_SOURCE)
    host.get_function('mpas_rk_cells_v842_f32')._fn((nz*nc+127)//128,1,1,128,1,1,*values)
    host.get_function('mpas_rk_edges_v842_f32')(((ne*nz+127)//128,),(128,),
        (nz,ne,ne,int(pbl),0,1,edge_mass,pbl_edge,edge,u,tend_u))
    for name,expected,actual in (('rho',rho,result.rho),('rho_u',u,result.rho_u),
                                 ('rho_theta',theta_out,result.rho_theta),('scalars',scalar_out,result.scalars)):
        if not np.array_equal(expected.view(np.uint32),actual.to_numpy().view(np.uint32)):
            raise AssertionError(f'real driver held {name} differs from selected source')
    print(f'resident held {"PBL" if pbl else "radiation"} sources match selected source')
