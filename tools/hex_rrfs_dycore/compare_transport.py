"""Compare each old RRFS transport launch with the retained CUDA source."""
import re
import sys
import numpy as np
from woof.hex import metal_transport_rrfs as metal
from woof.hex.metal_transport_v841 import MetalAdvectionCoefficients
from woof.hex.metal_rrfs_arguments import ARGUMENTS
from woof.hex.metal_rrfs_launch import Arrays
from woof.hex import cuda_monotonic_rrfs, cuda_transport_rrfs
from woof.verify.metal_adapter import CaseBackend
from tools.thompson_real_column_parity.host_backend import HostModule
from types import SimpleNamespace


def run():
    rng=np.random.default_rng(404)
    def random(shape,low=.01,high=.5): return rng.uniform(low,high,shape).astype(np.float32)
    ns,nl,nc,ne,me,width=2,6,6,6,2,10
    modules={a:HostModule('rrfs_'+a+'_transport_compare',s._CUDA_SOURCE)
             for a,s in [('monotonic',cuda_monotonic_rrfs),('transport',cuda_transport_rrfs)]}
    sources={'monotonic':cuda_monotonic_rrfs._CUDA_SOURCE,'transport':cuda_transport_rrfs._CUDA_SOURCE}
    words=0
    original=metal.launch
    def checked(area,symbol,count,arguments,**kw):
        nonlocal words
        signature=re.search(r'extern "C" __global__ void '+symbol+r'\s*\((.*?)\)\s*\{',sources[area],re.S)[1]
        reference=[]; outputs=[]
        for declaration,(_,pointer,dtype),value in zip(signature.split(','),ARGUMENTS[area][symbol],arguments,strict=True):
            if pointer:
                host=value.to_numpy()
                reference.append(host)
                if 'const' not in declaration: outputs.append((value,host))
            else: reference.append((np.int32 if dtype=='int32' else np.float32)(value))
        modules[area].get_function(symbol)(((count+127)//128,kw.get('levels',1),kw.get('species',1)),(128,),tuple(reference))
        result=original(area,symbol,count,arguments,**kw)
        for actual,expected in outputs:
            got=actual.to_numpy()
            bad=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32))
            assert not bad,(symbol,bad)
            words+=got.size
        return result
    with CaseBackend('metal') as backend:
        xp=Arrays(backend.context)
        put=xp.asarray
        coe=np.array([[e,(e+1)%5] for e in range(5)]+[[5,5]],np.int32)
        edges=np.array([[(c-1)%5,c] for c in range(5)]+[[5,5]],np.int32)
        neighbors=np.array([[(c-1)%5,(c+1)%5] for c in range(5)]+[[5,5]],np.int32)
        counts=np.array([2]*5+[0],np.int32)
        mesh=SimpleNamespace(max_edges=me,n_edges_on_cell=put(counts),edges_on_cell=put(edges),
            cells_on_cell=put(neighbors),cells_on_edge=put(coe),
            edge_sign_on_cell=put(np.tile(np.array([-1,1],np.float32),(nc,1))),dv_edge=put(random((ne,),.6,1.4)))
        context=SimpleNamespace(inv_area_cell=put(random((nc,),.6,1.4)))
        co=MetalAdvectionCoefficients(put(random((ne,width),-.1,.25)),put(random((ne,width),-.025,.025)),
            put(np.array([10,4,4,4,4,0],np.int32)),put(np.array([[(e+j)%5 for j in range(width)] for e in range(ne)],np.int32)),0)
        old,stage,source=(put(random((ns,nl,nc))) for _ in range(3))
        rho0,rho1=(put(random((nl,nc),.9,1.1)) for _ in range(2))
        velocity=random((nl,ne),-.2,.2); velocity[0,:]=np.array([0.,-0.,.1,-.1,0.,-0.],np.float32)
        ww=random((nl+1,nc),-.1,.1);ww[0]=0;ww[-1]=0
        options=dict(coefficients=co,fzm=put(np.full(nl,.5,np.float32)),fzp=put(np.full(nl,.5,np.float32)),
            rdzw=put(random((nl,),.4,.8)),config_time_integration_order=2,config_coef_3rd_order=.25,
            scalar_tendency=source,config_apply_lbcs=True,bdy_mask_cell=put(np.array([0,1,2,3,6,7],np.int32)),
            bdy_mask_edge=put(np.array([0,1,2,4,5,7],np.int32)),cell_solve_count=5)
        metal.launch=checked
        for rk in (1,2,3):
            result=metal.advance_scalars(mesh,context,old,stage,rho0,rho1,put(velocity),put(ww),.125,
                rk_step=rk,config_monotonic=True,**options)
            stage=result.scalars
        # Exercise entry scratch and partial intervals against the complete
        # retained CUDA schedule, independently of the per-launch checks.
        schedule = metal.advance_monotonic_work
        partial=True
        def check_schedule(a, **kw):
            nonlocal words
            if partial:
                a.update(cellstart=1, cellend=5, cellsolvestart=2, cellsolveend=4,
                         edgestart=1, edgeend=4)
            work = ('scalar_old','scalar_new','s_min','s_max','wdtn',
                    'scale_arr','flux_arr','flux_upwind_tmp','flux_tmp')
            for name in work:
                a[name][...] = random(a[name].shape, .2, .8)
            expected = {k: v.to_numpy() if hasattr(v, 'to_numpy') else v for k,v in a.items()}
            sys.modules['cupy'] = np
            cuda_monotonic_rrfs.kernel = lambda source, module, name, cache: modules['monotonic'].get_function(name)
            cuda_monotonic_rrfs.advance_monotonic_work_cuda_rrfs(expected, **kw,
                stencil_layout=tuple(np.ascontiguousarray(expected[n].T) for n in
                    ('advcellsforedge','adv_coefs','adv_coefs_3rd','edgesoncell','edgesoncell_sign')))
            result = schedule(a, **kw)
            for name in (*work, 'scalars_old', 'scalars_new', 'scalar_tend', 'rho_zz_int'):
                got = result[name].to_numpy()
                assert np.array_equal(got.view(np.uint32), expected[name].view(np.uint32)), name
                words += got.size
            return result
        metal.advance_monotonic_work=check_schedule
        metal.advance_scalars(mesh,context,old,stage,rho0,rho1,put(velocity),put(ww),.125,
            rk_step=3,config_monotonic=True,**options)
        partial=False
        def trim_first(v):return put(v.to_numpy()[:-1])
        gm=SimpleNamespace(max_edges=me,**{name:trim_first(getattr(mesh,name)) for name in
            ('n_edges_on_cell','edges_on_cell','cells_on_cell','cells_on_edge','edge_sign_on_cell','dv_edge')})
        gc=SimpleNamespace(inv_area_cell=trim_first(context.inv_area_cell))
        from dataclasses import replace
        gco=replace(co,**{name:trim_first(getattr(co,name)) for name in
            ('adv_coefs','adv_coefs_3rd','n_adv_cells_for_edge','adv_cells_for_edge')})
        global_options={**options,'coefficients':gco,'config_apply_lbcs':False,
            'scalar_tendency':put(source.to_numpy()[...,:-1].copy()),
            'bdy_mask_cell':None,'bdy_mask_edge':None}
        global_old=put(old.to_numpy()[...,:-1].copy())
        metal.advance_scalars(gm,gc,global_old,put(stage.to_numpy()[...,:-1].copy()),
            put(rho0.to_numpy()[...,:-1].copy()),put(rho1.to_numpy()[...,:-1].copy()),
            put(velocity[:,:-1].copy()),put(ww[:,:-1].copy()),.125,
            rk_step=3,config_monotonic=True,**global_options)
        assert np.count_nonzero(global_options['scalar_tendency'].to_numpy()) == 0
        metal.advance_monotonic_work=schedule
        metal.launch=original
    print('old transport source words',words,'mismatches',0)

if __name__=='__main__': run()
