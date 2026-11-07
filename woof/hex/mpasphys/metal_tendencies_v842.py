"""MPAS v8.4.2 held physics source conversion for Metal.

The caller supplies already projected, halo-exchanged edge rates. This is
only the todynamics conversion, not a complete MPAS physics provider.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..metal_dynamics_v841 import _arrays, _context, _i, _require
from ..metal_v841_launch import launch
from ..metal_physics_v841 import _stack

PBL_ORDER = ("qv", "qc", "qi", "qs", "ni", "nc", "nifa", "nwfa")
CU_ORDER = ("qv", "qc", "qi", "qr", "qs")
RATE_NAMES = ("rthblten", "rqvblten", "rqcblten", "rqiblten", "rqsblten",
              "rniblten", "rncblten", "rnifablten", "rnwfablten", "rthcuten",
              "rqvcuten", "rqccuten", "rqicuten", "rqrcuten", "rqscuten",
              "rthratenlw", "rthratensw")


def _launch(name, count, context, arrays, params):
    flat = {k: (v.view((int(v.size),)) if v.ndim != 1 else v) for k, v in arrays.items()}
    launch(context, name, count, flat, params)


@dataclass(frozen=True)
class MetalMpasSourcesV842:
    rho: object
    rho_u: object
    rho_theta: object
    scalars: object
    diabatic_theta: object | None = None


def get_tend(*, mass, mass_edge, theta_m, scalars, scalar_names, tendencies,
             selectors, pbl_edge, convection_edge, tend_u_phys,
             n_cells_solve, n_edges_solve, rk_step, dynamics_substep):
    """Assemble sources using level-two mass and level-one theta/scalars."""
    nlev, ncells = mass.shape
    nedges = mass_edge.shape[1]
    names = tuple(scalar_names)
    if len(set(names)) != len(names) or 'qv' not in names:
        raise ValueError('MPAS scalar names must be unique and include qv')
    if not 0 < n_cells_solve <= ncells or not 0 < n_edges_solve <= nedges:
        raise ValueError('MPAS solve extents must fit resident buffers')
    if rk_step not in (1, 2, 3) or dynamics_substep < 1:
        raise ValueError('invalid MPAS RK stage/split')
    _require('mass', mass, np.float32, (nlev, ncells))
    _require('theta_m', theta_m, np.float32, (nlev, ncells))
    _require('scalars', scalars, np.float32, (len(names), nlev, ncells))
    for name, value in (('mass_edge', mass_edge), ('pbl_edge', pbl_edge),
                        ('convection_edge', convection_edge), ('tend_u_phys', tend_u_phys)):
        _require(name, value, np.float32, (nlev, nedges))
    pbl = selectors['config_pbl_scheme'] != 'off'
    cu_scheme = selectors['config_convection_scheme']
    cu = cu_scheme != 'off'
    pbl_species = ['qv', 'qc', 'qi'] if pbl else []
    if pbl and selectors['config_pbl_scheme'] == 'bl_mynn':
        pbl_species += ['qs', 'ni']
        if selectors['config_microp_scheme'] == 'mp_thompson_aerosols':
            pbl_species += ['nc', 'nifa', 'nwfa']
    cu_species = ['qv', 'qc', 'qi'] if cu else []
    if cu_scheme == 'cu_kain_fritsch':
        cu_species += ['qr', 'qs']
    missing = set(pbl_species + cu_species + ['qv']) - set(names)
    if missing:
        raise ValueError(f'MPAS coupling missing species {sorted(missing)}')
    required = set()
    if pbl:
        required.add('rthblten')
        required.update(RATE_NAMES[1 + PBL_ORDER.index(n)] for n in pbl_species)
    if cu:
        required.add('rthcuten')
        required.update(RATE_NAMES[10 + CU_ORDER.index(n)] for n in cu_species)
    lw = selectors['config_radt_lw_scheme'] != 'off'
    sw = selectors['config_radt_sw_scheme'] != 'off'
    if lw: required.add('rthratenlw')
    if sw: required.add('rthratensw')
    missing = required - tendencies.keys()
    if missing:
        raise ValueError(f'MPAS coupling missing tendencies {sorted(missing)}')
    for name in required:
        _require(name, tendencies[name], np.float32, (nlev, ncells))
    context = _context(mass)
    arrays = _arrays(context)
    def indices(order, used):
        return arrays.asarray(np.array([names.index(n) if n in used or n == 'qv' else -1
                                        for n in order], dtype=np.int32))
    rho = arrays.zeros((nlev, ncells), np.float32)
    rho_theta = arrays.zeros((nlev, ncells), np.float32)
    rho_u = arrays.zeros((nlev, nedges), np.float32)
    scalars_out = arrays.zeros((len(names), nlev, ncells), np.float32)
    rates = _stack(arrays, (nlev, ncells), [tendencies[n] if n in required else None for n in RATE_NAMES])
    _launch('hexv842_mpas_rk_cells_f32', nlev * ncells, context,
            dict(pbl_indices=indices(PBL_ORDER, pbl_species), cu_indices=indices(CU_ORDER, cu_species),
                 mass=mass, theta_m=theta_m, qv=scalars[names.index('qv')], rates=rates,
                 out_theta=rho_theta, out_q=scalars_out, out_rho=rho),
            dict(nlev=_i(nlev), ncells=_i(ncells), nsolve=_i(n_cells_solve),
                 pbl=_i(pbl), cu=_i(cu), lw=_i(lw), sw=_i(sw),
                 rvord=np.float32(np.float32(461.6) / np.float32(287.0))))
    _launch('hexv842_mpas_rk_edges_f32', nlev * nedges, context,
            dict(mass_edge=mass_edge, pbl_edge=pbl_edge, cu_edge=convection_edge,
                 out_u=rho_u, tend_u_phys=tend_u_phys),
            dict(nlev=_i(nlev), nedges=_i(nedges), nsolve=_i(n_edges_solve), pbl=_i(pbl),
                 cu_momentum=_i(cu_scheme in ('cu_tiedtke', 'cu_ntiedtke')),
                 first=_i(rk_step == 1 and dynamics_substep == 1)))
    return MetalMpasSourcesV842(rho, rho_u, rho_theta, scalars_out)
