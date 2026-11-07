"""Source-ordered RRFS scalar transport on resident Metal pools."""
import numpy as np
from .metal_rrfs_launch import Arrays, launch, transpose
from .metal_transport_v841 import MetalScalarTransportResult


def advance_monotonic_work(a, *, config_apply_lbcs, stencil_layout=None):
    ns,nl,nc = map(int,a['scalars_old'].shape)
    ne=int(a['uhavg'].shape[1]); me=int(a['edgesoncell'].shape[1]); width=int(a['adv_coefs'].shape[1])
    cl,ch,sl,sh,el,eh=(int(a[n]) for n in ('cellstart','cellend','cellsolvestart','cellsolveend','edgestart','edgeend'))
    dummy,solved=int(a['ncells']),int(a['ncellssolve'])
    if not (0<=cl<=ch<=dummy<nc and 0<=sl<=sh<=dummy and 0<=el<=eh<=int(a['nedges'])<ne):
        raise ValueError('monotonic work intervals exceed native allocation bounds')
    if stencil_layout is None:
        stencil_layout=tuple(transpose(a[name]) for name in
            ('advcellsforedge','adv_coefs','adv_coefs_3rd','edgesoncell','edgesoncell_sign'))
    cells,coef,coef3,edges,sign=stencil_layout
    def call(name,count,*args):
        launch('monotonic',name,count,args,
               levels=1 if name in ('mono_low_vertical_rrfs','mono_finish_rrfs') else nl,
               species=ns if name=='mono_source_rrfs' else 1)
    base=(nl,nc); eb=(*base,ne); cb=(*eb,me); dt=np.float32(a['dt'])
    call('mono_source_rrfs',nc,ns,*base,sl,sh,dt,a['rho_zz_old'],a['scalars_old'],a['scalar_tend'])
    target=a['rho_zz_new']
    if a.get('advance_density',True):
        target=a['rho_zz_int']
        call('mono_density_rrfs',nc,*cb,sl,sh,dt,a['nedgesoncell'],edges,sign,a['dvedge'],a['invareacell'],a['uhavg'],a['wwavg'],a['rdnw'],a['rho_zz_old'],target)
    from .cuda_monotonic_rrfs import _species_per_pass
    work_names = ('scalar_old', 'scalar_new', 's_min', 's_max', 'wdtn',
                  'scale_arr', 'flux_arr', 'flux_upwind_tmp', 'flux_tmp')
    independent = (cl <= sl <= sh <= ch and
        _species_per_pass(ns, sum(a[name].nbytes for name in work_names), None) > 1)
    entry = {name: a[name].copy() for name in work_names} if independent else {}
    for s in range(ns):
        for name, value in entry.items():
            a[name][...] = value
        call('mono_carriers_rrfs',nc,s,*base,cl,ch,dummy,a['scalars_old'],a['scalars_new'],a['scalar_old'],a['scalar_new'])
        call('mono_vertical_rrfs',nc,*base,me,sl,sh,a['coef_3rd_order'],a['nedgesoncell'],a['cellsoncell'],a['scalar_old'],a['scalar_new'],a['wwavg'],a['fnm'],a['fnp'],a['s_min'],a['s_max'],a['wdtn'])
        call('mono_high_edge_rrfs',ne,*eb,width,el,eh,solved,a['cellsonedge'],a['nadvcellsforedge'],cells,coef,coef3,a['uhavg'],a['scalar_new'],a['flux_arr'])
        call('mono_low_vertical_rrfs',nc,*base,sl,sh,dt,a['scalar_old'],a['rho_zz_old'],a['wwavg'],a['rdnw'],a['scalar_new'],a['wdtn'],a['scale_arr'])
        call('mono_low_edge_rrfs',ne,*eb,el,eh,int(config_apply_lbcs),dt,a['cellsonedge'],a['bdymaskedge'],a['dvedge'],a['uhavg'],a['scalar_old'],a['flux_arr'],a['flux_upwind_tmp'],a['flux_tmp'])
        call('mono_scale_rrfs',nc,*cb,sl,sh,a['nedgesoncell'],edges,sign,a['invareacell'],a['flux_upwind_tmp'],a['flux_tmp'],a['s_min'],a['s_max'],target,a['scalar_new'],a['scale_arr'])
        call('mono_limit_edge_rrfs',ne,*eb,el,eh,solved,int(config_apply_lbcs),dt,a['cellsonedge'],a['bdymaskedge'],a['dvedge'],a['uhavg'],a['scalar_old'],a['scale_arr'],a['flux_arr'])
        call('mono_finish_rrfs',nc,s,*cb,sl,sh,cl,ch,a['nedgesoncell'],edges,a['bdymaskcell'],sign,a['invareacell'],a['rdnw'],target,a['scale_arr'],a['flux_arr'],a['wdtn'],a['scalar_new'],a['scalars_new'])
    return a


def _resize_last(value, extent):
    from .metal_v841_launch import launch as dispatch
    xp=Arrays(value.buffer.context)
    shape=(*value.shape[:-1],extent)
    out=xp.zeros(shape,value.dtype)
    rows=int(np.prod(value.shape[:-1]))
    nsolve=min(extent,value.shape[-1]);nfull=max(extent,value.shape[-1])
    dispatch(value.buffer.context,'hexv842_mpas_transport_rows_f32',rows*nsolve,
        dict(src=value.view((value.size,),np.float32),dst=out.view((out.size,),np.float32),
             mapping=xp.asarray(np.array([0],np.int32))),
        dict(nrows=1,nlev=rows,nsolve=nsolve,nfull=nfull,scatter=int(extent>value.shape[-1])))
    return out


def _pad_first(value):
    xp=Arrays(value.buffer.context)
    out=xp.zeros((value.shape[0]+1,*value.shape[1:]),value.dtype)
    out[:value.shape[0]][...]=value
    return out


def advance_scalars(mesh, context, old, stage, rho0, rho1, uh, ww, dt, *,
    coefficients, fzm, fzp, rdzw, rk_step, config_time_integration_order,
    config_coef_3rd_order, scalar_tendency=None, advance_density=True,
    config_apply_lbcs=False, bdy_mask_cell=None, bdy_mask_edge=None,
    cell_solve_count=None, native_work=None, garbage_cell=None, **unused):
    xp=Arrays(old.buffer.context)
    ns,nl,nc=old.shape; ne=uh.shape[1]; me=mesh.max_edges
    solved=nc if cell_solve_count is None else int(cell_solve_count)
    maskc=xp.zeros((nc,),np.int32) if bdy_mask_cell is None else bdy_mask_cell
    maske=xp.zeros((ne,),np.int32) if bdy_mask_edge is None else bdy_mask_edge
    source=xp.zeros_like(old) if scalar_tendency is None else scalar_tendency
    if rk_step==3 and (unused.get('config_monotonic') or unused.get('config_positive_definite')):
        if not config_apply_lbcs and garbage_cell is None:
            from types import SimpleNamespace
            from dataclasses import replace
            padded_mesh=SimpleNamespace(max_edges=me,
                **{name:_pad_first(getattr(mesh,name)) for name in
                   ('cells_on_edge','cells_on_cell','edges_on_cell','edge_sign_on_cell')},
                n_edges_on_cell=_pad_first(mesh.n_edges_on_cell),dv_edge=_pad_first(mesh.dv_edge))
            padded_context=SimpleNamespace(inv_area_cell=_pad_first(context.inv_area_cell))
            padded_coefficients=replace(coefficients,**{name:_pad_first(getattr(coefficients,name))
                for name in ('adv_coefs','adv_coefs_3rd','n_adv_cells_for_edge','adv_cells_for_edge')})
            padded_old,padded_source=_resize_last(old,nc+1),_resize_last(source,nc+1)
            result=advance_scalars(padded_mesh,padded_context,padded_old,_resize_last(stage,nc+1),
                _resize_last(rho0,nc+1),_resize_last(rho1,nc+1),_resize_last(uh,ne+1),_resize_last(ww,nc+1),dt,
                coefficients=padded_coefficients,fzm=fzm,fzp=fzp,rdzw=rdzw,rk_step=rk_step,
                config_time_integration_order=config_time_integration_order,
                config_coef_3rd_order=config_coef_3rd_order,scalar_tendency=padded_source,
                advance_density=advance_density,config_apply_lbcs=False,garbage_cell=nc,
                cell_solve_count=solved,**unused)
            old[...]=_resize_last(padded_old,nc)
            if scalar_tendency is not None:scalar_tendency[...]=_resize_last(padded_source,nc)
            return MetalScalarTransportResult(_resize_last(result.scalars,nc),_resize_last(result.density,nc))
        physical=nc-1 if garbage_cell is None else int(garbage_cell)
        a={} if native_work is None else dict(native_work)
        a.update(ncells=physical,nedges=ne-1,ncellssolve=solved,
            cellstart=0,cellend=physical,cellsolvestart=0,cellsolveend=solved,
            edgestart=0,edgeend=ne-1,dt=dt,coef_3rd_order=config_coef_3rd_order,
            advance_density=advance_density,scalars_old=old,scalars_new=stage.copy(),
            rho_zz_old=rho0,rho_zz_new=rho1,uhavg=uh,wwavg=ww,
            fnm=fzm,fnp=fzp,rdnw=rdzw,invareacell=context.inv_area_cell,
            dvedge=mesh.dv_edge,cellsonedge=mesh.cells_on_edge,cellsoncell=mesh.cells_on_cell,
            edgesoncell=mesh.edges_on_cell,nedgesoncell=mesh.n_edges_on_cell,
            edgesoncell_sign=mesh.edge_sign_on_cell,nadvcellsforedge=coefficients.n_adv_cells_for_edge,
            advcellsforedge=coefficients.adv_cells_for_edge,adv_coefs=coefficients.adv_coefs,
            adv_coefs_3rd=coefficients.adv_coefs_3rd,scalar_tend=source,bdymaskcell=maskc,bdymaskedge=maske)
        for name,shape in dict(scalar_old=(nl,nc),scalar_new=(nl,nc),s_min=(nl,nc),s_max=(nl,nc),
            wdtn=(nl+1,nc),scale_arr=(nl,2,nc),flux_arr=(nl,ne),flux_upwind_tmp=(nl,ne),flux_tmp=(nl,ne),rho_zz_int=(nl,nc)).items():
            if name not in a: a[name]=xp.zeros(shape,np.float32)
        advance_monotonic_work(a,config_apply_lbcs=config_apply_lbcs)
        return MetalScalarTransportResult(a['scalars_new'],a['rho_zz_int'] if advance_density else rho1)
    edge=xp.zeros((ns,nl,ne),np.float32)
    cells,coef,coef3=(transpose(v) for v in (coefficients.adv_cells_for_edge,coefficients.adv_coefs,coefficients.adv_coefs_3rd))
    launch('transport','scalar_edge_rrfs',nl*ne,(ns,nl,nc,ne,coefficients.adv_coefs.shape[1],int(config_apply_lbcs),0,ne,maske,mesh.cells_on_edge,mesh.dv_edge,stage,uh,coefficients.n_adv_cells_for_edge,cells,coef,coef3,edge))
    weight=np.float32(1/3 if advance_density and rk_step==1 and config_time_integration_order==3 else .5 if advance_density and rk_step in (1,2) else 1)
    output,target=xp.empty_like(old),xp.empty_like(rho0)
    launch('transport','scalar_finish_rrfs',nl*nc,(ns,nl,nc,ne,me,solved,maskc,mesh.n_edges_on_cell,transpose(mesh.edges_on_cell),transpose(mesh.edge_sign_on_cell),context.inv_area_cell,uh,edge,ww,fzm,fzp,rdzw,old,stage,rho0,rho1,source,dt,weight,config_coef_3rd_order,output,target))
    return MetalScalarTransportResult(output,target)
