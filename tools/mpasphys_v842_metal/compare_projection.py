"""Compare unweighted edge projection with the selected host-compiled source."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.cuda_physics_v841 import _CUDA_SOURCE
from woof.hex.mpasphys.metal_projection_v842 import project
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    start=_CUDA_SOURCE.index('extern "C" __global__ void project_edges_v841_f32(')
    end=_CUDA_SOURCE.index('extern "C" __global__ void nonnegative_qv_copy_v841_f32(',start)
    kernel=HostModule('mpas_v842_projection',CUDA_FTZ_HELPERS+_CUDA_SOURCE[start:end]).get_function('project_edges_v841_f32')
    rng=np.random.default_rng(9842); nz,nc,ne=4,7,9
    cells=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[0,6],[2,5],[6,6]],np.int32)
    east=rng.uniform(-1,1,(nc,3)).astype(np.float32); north=rng.uniform(-1,1,(nc,3)).astype(np.float32)
    normal=rng.uniform(-1,1,(ne,3)).astype(np.float32)
    du=rng.uniform(-1,1,(nz,nc)).astype(np.float32); dv=rng.uniform(-1,1,(nz,nc)).astype(np.float32)
    du[0,0]=np.float32(2**-149); du[:,-1]=0; dv[:,-1]=0
    unit=np.ones((nz,ne),np.float32); expected=np.empty_like(unit); invalid=np.zeros(1,np.int32)
    kernel((1,),(128,),(nz,nc,ne,cells,east,north,normal,du,dv,unit,expected,invalid))
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        geo=SimpleNamespace(n_cells=nc,n_edges=ne,cells_on_edge=xp.asarray(cells),east_cell=xp.asarray(east),north_cell=xp.asarray(north),edge_normal_vectors=xp.asarray(normal),validate=lambda:None)
        flag=xp.asarray(np.zeros(1,np.int32))
        actual=project(geometry=geo,zonal=xp.asarray(du),meridional=xp.asarray(dv),invalid=flag).to_numpy()
        if np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32)):
            raise AssertionError('projection differs from selected source')
        if int(flag.to_numpy()[0]) != int(invalid[0]): raise AssertionError('invalid flag differs')
    print('projection matches host-selected source word for word')

if __name__=='__main__': main()
