"""Compare resident RRFS operators with host-built retained CUDA sources."""
import argparse
import importlib
import re
from types import SimpleNamespace
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex import metal_rrfs_launch as dispatch
from woof.hex.metal_rrfs_arguments import ARGUMENTS
from woof.hex.metal_horizontal_rrfs import MetalHorizontalRRFS
from woof.hex.metal_containers_v841 import MetalMesh
from woof.hex.mesh import Mesh
from woof.hex.regional_v841 import derive_regional_masks
from woof.hex.cuda_regional_forecast_v841 import PaddedRegionalHostMesh
from woof.hex.metal_rrfs_launch import Arrays
from woof.verify.metal_adapter import CaseBackend


def run(grid, init):
    host_mesh=Mesh.from_netcdf(grid,init,validate=False)
    padded=PaddedRegionalHostMesh(host_mesh,derive_regional_masks(host_mesh,np.dtype(np.float32)))
    rng=np.random.default_rng(230)
    modules={}; words={}
    original=dispatch.launch
    def checked(area,symbol,count,arguments,**kw):
        module_key = 'scaling' if symbol == 'mixing_scaling_f32' else area
        if module_key not in modules:
            source=importlib.import_module('woof.hex.cuda_'+area+'_rrfs')._CUDA_SOURCE
            if module_key == 'scaling':
                from woof.hex.cuda_horizontal import _CUDA_SOURCE as source
                from woof.hex.native_powf import CUDA_NATIVE_POWF
                source=CUDA_NATIVE_POWF+source.replace('powf(mean_density,','mpas_native_powf_positive(mean_density,')
            modules[module_key]=(source,HostModule('rrfs_'+module_key+'_operators',source))
        source,module=modules[module_key]
        signature=re.search(r'extern "C" __global__ void '+symbol+r'\s*\((.*?)\)\s*\{',source,re.S)[1]
        args=[]; outputs=[]
        for decl,(_,pointer,dtype),value in zip(signature.split(','),ARGUMENTS[area][symbol],arguments,strict=True):
            if pointer:
                v=value.to_numpy();args.append(v)
                if 'const' not in decl:outputs.append((value,v))
            else:args.append((np.int32 if dtype=='int32' else np.float32)(value))
        module.get_function(symbol)(((count+127)//128,kw.get('levels',1),kw.get('species',1)),(128,),tuple(args))
        result=original(area,symbol,count,arguments,**kw)
        for actual,expected in outputs:
            got=actual.to_numpy()
            assert np.array_equal(got.view(np.uint32),expected.view(np.uint32)), (symbol,np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32)))
            words[symbol]=words.get(symbol,0)+got.size
        return result
    import woof.hex.metal_horizontal_rrfs as horizontal
    import woof.hex.metal_acoustic_rrfs as acoustic
    horizontal.launch=acoustic.launch=checked
    with CaseBackend('metal') as backend:
        xp=Arrays(backend.context);mesh=MetalMesh.from_host(backend.context,padded)
        def field(shape,center=0,scale=.01):
            return xp.asarray((center+scale*rng.standard_normal(shape)).astype(np.float32))
        ctx=SimpleNamespace(**{name:xp.asarray((1/np.asarray(padded.arrays[key])).astype(np.float32)) for name,key in
            [('inv_area_cell','areaCell'),('inv_area_triangle','areaTriangle'),('inv_dc_edge','dcEdge'),('inv_dv_edge','dvEdge')]})
        from woof.hex.mixing_rrfs import RRFSMixingConfig
        for nl in (6,59):
            nc,ne,nv=mesh.n_cells,mesh.n_edges,mesh.n_vertices
            if nl == 59:
                mesh.mesh_density=xp.asarray(rng.uniform(.05,8,nc).astype(np.float32))
            h=MetalHorizontalRRFS(mesh,nl,ctx)
            normal,tangent,rho_edge=(field((nl,ne),center) for center in (1,.2,1))
            theta=field((nl,nc),300);vertical=field((nl+1,nc))
            h.compute_dry_mixing_tendencies_v841(normal,tangent,vertical,theta,rho_edge,
                field((nl,nc)),field((nl,nv)),dt=120,config=RRFSMixingConfig(),
                initial_u_tendency=field((nl,ne)))
            checked('dynamics','vector_momentum_rrfs_f32',nl*ne,
                (nl,nc,ne,mesh.max_edges2,normal,rho_edge,field((nl,ne)),field((nl,nc)),
                 field((nl,nc)),mesh.cells_on_edge,mesh.edges_on_edge,mesh.n_edges_on_edge,
                 mesh.weights_on_edge,ctx.inv_dc_edge,field((nl,ne)),1,field((nl,ne))))
            checked('dynamics','euler_w_accumulate_rrfs_f32',(nl-1)*padded.n_cells_solve,
                (nl,nc,padded.n_cells_solve,field((nl,nc)),field((nl,nc)),field((nl,nc),1),
                 field((nl,),.01),field((nl,),.5),field((nl,),.5),field((nl+1,nc))))
            masks=derive_regional_masks(host_mesh,np.dtype(np.float32))
            mask=xp.asarray(np.append(masks.bdy_mask_cell,np.int32(0)).astype(np.int32))
            nearest=xp.asarray(np.append(masks.nearest_relaxation_cell,np.int32(nc-1)).astype(np.int32))
            for zero in (0,1):
                w=vertical.copy()
                checked('regional','w_boundary_rrfs',padded.n_cells_solve,
                    (nl,nc,0,padded.n_cells_solve,5,zero,mask,nearest,w,w))
            h.pressure_gradient_euler_tendency(field((nl,nc)),field((nl,nc)),rho_edge,
                field((nl,nc),1),field((nl,ne)))
            coefficients=acoustic.compute_vertical_implicit_coefficients_metal_rrfs(dts=.7,epssm=.1,
                zz=field((nl,nc),1),cqw=field((nl,nc),1),pressure=field((nl,nc),.9),theta=theta,
                rho_base=field((nl,nc),1),rho_theta_base=field((nl,nc),300),pressure_base=field((nl,nc),.9),
                rho_theta_perturbation=field((nl,nc)),qtot=field((nl,nc),.01),
                rdzw=field((nl,),.01,.001),fzm=field((nl,),.5,.01),fzp=field((nl,),.5,.01),
                rdzu=field((nl,),.01,.001),cell_solve_count=padded.n_cells_solve)
            state=acoustic.MetalAcousticState.zeros(xp,nl,nc,ne)
            forcing=acoustic.MetalAcousticForcing(rho_zz=field((nl,nc),1),theta_m=theta,
                zz=field((nl,nc),1),exner=field((nl,nc),.9),cqu=rho_edge,zxu=field((nl,ne)),
                dss=field((nl,nc),.01,.001),tend_ru=field((nl,ne)),tend_rho=field((nl,nc)),
                tend_rt=field((nl,nc)),tend_rw=field((nl+1,nc)),w=vertical,rw=vertical,rw_save=vertical)
            for step in (1,2,3):
                acoustic.advance_acoustic_step_metal_rrfs(mesh,state,forcing,coefficients,dts=.7,
                    context=ctx,small_step=step,epssm=.1,fzm=field((nl,),.5,.01),fzp=field((nl,),.5,.01),
                    rdzw=field((nl,),.01,.001),cell_solve_count=padded.n_cells_solve)
    print('retained RRFS operator source words',sum(words.values()),words)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('grid');parser.add_argument('init');args=parser.parse_args()
    run(args.grid,args.init)
