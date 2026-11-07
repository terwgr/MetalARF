"""Source-ordered scalar-offcentred RRFS acoustic launch schedule."""
from __future__ import annotations
from typing import Any
import numpy as np
from .metal_acoustic_v841 import (MetalAcousticState, MetalAcousticForcing,
    MetalVerticalImplicitCoefficients, _float32, _int32, _mesh_value)
from .metal_rrfs_launch import Arrays, launch, transpose


def _kernel(name, cache):
    return name


def _launch_1d(name, count, arguments):
    launch('acoustic', name, count, arguments)


_launch_columns = _launch_1d


def _cell_stencil_layout(edges, signs, ncells):
    return transpose(edges), transpose(signs)


def compute_vertical_implicit_coefficients_metal_rrfs(*, dts: float, cell_solve_count: int | None=None, epssm: float, zz: Any, cqw: Any, pressure: Any, theta: Any, rho_base: Any, rho_theta_base: Any, pressure_base: Any, rho_theta_perturbation: Any, qtot: Any, rdzw: Any, fzm: Any, fzp: Any, rdzu: Any, gravity: float=9.80616, rgas: float=287.0, cp: float=1004.5, kernel_cache: Any | None=None, initial_coefficients: MetalVerticalImplicitCoefficients | None=None) -> MetalVerticalImplicitCoefficients:
    """Fork mpas_atm_time_integration.F:2225-2366 at 1d6e370e: scaled coefficients."""
    cp_module = Arrays(zz.buffer.context)
    zz = _float32(zz, 'zz')
    if zz.ndim != 2:
        raise ValueError('zz must have shape (nVertLevels,nCells)')
    nlev, ncells = map(int, zz.shape)
    nsolve = ncells if cell_solve_count is None else int(cell_solve_count)
    if not 0 <= nsolve <= ncells:
        raise ValueError('cell_solve_count outside resident cells')
    cell_shape = (nlev, ncells)
    inputs: dict[str, Any] = {}
    for name, value in {'cqw': cqw, 'pressure': pressure, 'theta': theta, 'rho_base': rho_base, 'rho_theta_base': rho_theta_base, 'pressure_base': pressure_base, 'rho_theta_perturbation': rho_theta_perturbation, 'qtot': qtot}.items():
        array = _float32(value, name)
        if array.shape != cell_shape:
            raise ValueError(f'{name} shape {array.shape} != {cell_shape}')
        inputs[name] = array
    vertical: dict[str, Any] = {}
    for name, value in {'rdzw': rdzw, 'fzm': fzm, 'fzp': fzp, 'rdzu': rdzu}.items():
        array = _float32(value, name)
        if array.shape != (nlev,):
            raise ValueError(f'{name} shape {array.shape} != ({nlev},)')
        vertical[name] = array

    def zero() -> Any:
        return cp_module.zeros(cell_shape, dtype=cp_module.float32)
    coftz = cp_module.zeros((nlev + 1, ncells), dtype=cp_module.float32)
    cofrz = cp_module.empty((nlev,), dtype=cp_module.float32)
    result = MetalVerticalImplicitCoefficients(cofwr=zero(), cofwz=zero(), coftz=coftz, cofwt=zero(), cofrz=cofrz, a_tri=zero(), b_tri=zero(), c_tri=zero(), alpha_tri=zero(), gamma_tri=zero())
    if initial_coefficients is not None:
        for name in ('cofwr', 'cofwz', 'coftz', 'cofwt', 'a_tri', 'alpha_tri', 'gamma_tri'):
            entry = _float32(getattr(initial_coefficients, name), name)
            target = getattr(result, name)
            if entry.shape != target.shape:
                raise ValueError(f'entry coefficient {name} shape differs from resident cells')
            target[...] = entry
    _launch_1d(_kernel('acoustic_cofrz_rrfs', kernel_cache), nlev, (np.int32(nlev), np.float32(dts), np.float32(epssm), vertical['rdzw'], cofrz))
    coefficient_args = (np.int32(nlev), np.int32(ncells), np.float32(dts), np.float32(epssm), np.float32(gravity), np.float32(rgas), np.float32(cp), zz, inputs['cqw'], inputs['pressure'], inputs['theta'], inputs['rho_base'], inputs['rho_theta_base'], inputs['pressure_base'], inputs['rho_theta_perturbation'], inputs['qtot'], vertical['rdzw'], vertical['fzm'], vertical['fzp'], vertical['rdzu'], np.int32(nsolve), result.cofwr, result.cofwz, result.coftz, result.cofwt, result.cofrz, result.a_tri, result.b_tri, result.c_tri, result.alpha_tri, result.gamma_tri)
    _launch_1d(_kernel('acoustic_coefficients_rrfs', kernel_cache), nlev * ncells, coefficient_args)
    _launch_1d(_kernel('acoustic_matrix_rrfs', kernel_cache), nlev * ncells, coefficient_args)
    _launch_columns(_kernel('acoustic_factor_rrfs', kernel_cache), ncells, coefficient_args)
    return result

def advance_acoustic_step_metal_rrfs(mesh: Any, state: MetalAcousticState, forcing: MetalAcousticForcing, coefficients: MetalVerticalImplicitCoefficients, *, dts: float, context: Any, cell_solve_count: int | None=None, edge_start: int=0, edge_end: int | None=None, cell_start: int=0, cell_end: int | None=None, spec_zone_mask_cell: Any | None=None, spec_zone_mask_edge: Any | None=None, small_step: int, epssm: float, fzm: Any, fzp: Any, rdzw: Any, gravity: float=9.80616, rgas: float=287.0, cp: float=1004.5, in_place: bool=True, kernel_cache: Any | None=None) -> MetalAcousticState:
    """Fork mpas_atm_time_integration.F:2646-2984 at 1d6e370e: global/LAM substep."""
    if small_step < 1:
        raise ValueError('small_step must be at least one')
    out = state if in_place else state.copy()
    nlev, nedges = map(int, out.ru_p.shape)
    ncells = int(out.rho_pp.shape[1])
    edge_end = nedges if edge_end is None else int(edge_end)
    cell_end = ncells if cell_end is None else int(cell_end)
    if not 0 <= cell_start <= cell_end <= ncells:
        raise ValueError('acoustic cell interval outside resident cells')
    if not 0 <= edge_start <= edge_end <= nedges:
        raise ValueError('acoustic edge interval outside resident edges')
    if out.rho_pp.shape != (nlev, ncells) or out.rw_p.shape != (nlev + 1, ncells):
        raise ValueError('acoustic state shapes are inconsistent')
    nsolve = ncells if cell_solve_count is None else int(cell_solve_count)
    if not 0 <= nsolve <= ncells:
        raise ValueError('cell_solve_count outside resident cells')
    cp_module = Arrays(out.ru_p.buffer.context)
    spec_cell = cp_module.zeros(ncells, dtype=cp_module.float32) if spec_zone_mask_cell is None else _float32(spec_zone_mask_cell, 'spec_zone_mask_cell')
    spec_edge = cp_module.zeros(nedges, dtype=cp_module.float32) if spec_zone_mask_edge is None else _float32(spec_zone_mask_edge, 'spec_zone_mask_edge')
    if spec_cell.shape != (ncells,) or spec_edge.shape != (nedges,):
        raise ValueError('specified-zone masks differ from mesh')
    cells_on_edge = _int32(_mesh_value(mesh, 'cells_on_edge'), 'cells_on_edge')
    edges_on_cell = _int32(_mesh_value(mesh, 'edges_on_cell'), 'edges_on_cell')
    counts = _int32(_mesh_value(mesh, 'n_edges_on_cell'), 'n_edges_on_cell')
    signs = _float32(_mesh_value(mesh, 'edge_sign_on_cell'), 'edge_sign_on_cell')
    dv_edge = _float32(_mesh_value(mesh, 'dv_edge'), 'dv_edge')
    inv_dc_edge = _float32(context.inv_dc_edge, 'inv_dc_edge')
    inv_area_cell = _float32(context.inv_area_cell, 'inv_area_cell')
    max_edges = int(edges_on_cell.shape[1])
    edges_on_cell, signs = _cell_stencil_layout(edges_on_cell, signs, ncells)
    fzm = _float32(fzm, 'fzm')
    fzp = _float32(fzp, 'fzp')
    rdzw = _float32(rdzw, 'rdzw')
    _launch_1d(_kernel('acoustic_ru_rrfs', kernel_cache), nlev * nedges, (np.int32(nlev), np.int32(nedges), np.int32(ncells), np.int32(small_step), np.float32(dts), np.int32(edge_start), np.int32(edge_end), np.float32(gravity), np.float32(rgas), np.float32(cp), cells_on_edge, inv_dc_edge, _float32(forcing.zz, 'forcing.zz'), _float32(forcing.exner, 'forcing.exner'), _float32(forcing.cqu, 'forcing.cqu'), _float32(forcing.zxu, 'forcing.zxu'), _float32(forcing.tend_ru, 'forcing.tend_ru'), spec_edge, np.int32(nsolve), out.rho_pp, out.rtheta_pp, out.ru_p, out.ru_avg))
    _launch_1d(_kernel('acoustic_prepare_rrfs', kernel_cache), (nlev + 1) * ncells, (np.int32(nlev), np.int32(ncells), np.int32(small_step), np.int32(nsolve), np.int32(cell_start), np.int32(cell_end), out.rw_p, out.rtheta_pp, out.rtheta_pp_old, out.rho_pp, out.ww_avg))
    cp = Arrays(out.ru_p.buffer.context)
    rs = cp.empty((nlev, ncells), dtype=cp.float32)
    ts = cp.empty_like(rs)
    _launch_1d(_kernel('acoustic_rs_ts_rrfs', kernel_cache), nlev * ncells, (np.int32(nlev), np.int32(ncells), np.int32(nedges), np.int32(max_edges), np.float32(dts), np.float32(epssm), counts, edges_on_cell, cells_on_edge, signs, dv_edge, inv_area_cell, spec_cell, np.int32(nsolve), _float32(forcing.theta_m, 'forcing.theta_m'), rdzw, coefficients.cofrz, coefficients.coftz, out.ru_p, out.rw_p, out.rho_pp, out.rtheta_pp, _float32(forcing.tend_rho, 'forcing.tend_rho'), _float32(forcing.tend_rt, 'forcing.tend_rt'), rs, ts))
    solve_arguments = (np.int32(nlev), np.int32(ncells), np.float32(dts), np.float32(epssm), np.int32(nsolve), spec_cell, _float32(forcing.zz, 'forcing.zz'), _float32(forcing.rho_zz, 'forcing.rho_zz'), fzm, fzp, rdzw, _float32(forcing.dss, 'forcing.dss'), _float32(forcing.w, 'forcing.w'), _float32(forcing.rw, 'forcing.rw'), _float32(forcing.rw_save, 'forcing.rw_save'), _float32(forcing.tend_rw, 'forcing.tend_rw'), rs, ts, coefficients.cofwr, coefficients.cofwz, coefficients.coftz, coefficients.cofwt, coefficients.cofrz, coefficients.a_tri, coefficients.alpha_tri, coefficients.gamma_tri, out.rw_p, out.rho_pp, out.rtheta_pp, out.ww_avg)
    _launch_1d(_kernel('acoustic_column_pre_rrfs', kernel_cache), nlev * ncells, solve_arguments)
    _launch_1d(_kernel('acoustic_column_solve_rrfs', kernel_cache), ncells, solve_arguments)
    _launch_1d(_kernel('acoustic_column_post_w_rrfs', kernel_cache), nlev * ncells, solve_arguments)
    _launch_1d(_kernel('acoustic_column_post_rt_rrfs', kernel_cache), nlev * ncells, solve_arguments)
    return out
