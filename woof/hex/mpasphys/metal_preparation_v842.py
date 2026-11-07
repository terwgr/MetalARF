"""Selected MPAS v8.4.2 preparation operators for resident Metal pools."""
from __future__ import annotations

import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _i, _require
from .metal_tendencies_v842 import _launch


def prepare_interfaces(*, fields, nlev, ncells, gravity):
    """Update caller-owned native preparation fields, including interpolation weights."""
    names = ('rho_zz', 'zz', 'qv_p', 'pres_p', 't_p', 'rho_p', 'dz_p', 'zgrid', 'w',
             'w_p', 'z_p', 't2_p', 'pres2_p', 'pres_hyd_p', 'pres2_hyd_p', 'pres_hydd_p',
             'pres2_hydd_p', 'psfc_p', 'psfc_hyd_p', 'psfc_hydd_p', 'znu_p', 'znu_hyd_p',
             'plrad', 'invalid', 'pressure_p', 'pressure_base', 'weights_minus', 'weights_plus')
    arrays = {name: fields[name] for name in names}
    for name, value in arrays.items():
        _require(name, value, np.int32 if name == 'invalid' else np.float32)
    context = _context(arrays['rho_zz'])
    _launch('hexv842_mpas_prep_interfaces_f32', ncells, context, arrays,
            dict(nlev=_i(nlev), ncells=_i(ncells), gravity=np.float32(gravity)))


def prepare_numbers(*, species, microphysics, pbl, scalar_flags, n_cells_solve):
    """Select number scratch words exactly as the MPAS interface does."""
    if microphysics not in ('off', 'mp_kessler', 'mp_wsm6', 'mp_thompson', 'mp_thompson_aerosols'):
        raise ValueError(f'unknown MPAS microphysics selector {microphysics!r}')
    if pbl not in ('off', 'bl_ysu', 'bl_mynn'):
        raise ValueError(f'unknown MPAS PBL selector {pbl!r}')
    template = species['qv']
    nlev, ncells = template.shape
    if type(n_cells_solve) is not int or not 0 < n_cells_solve <= ncells:
        raise ValueError('number scratch requires a nonempty native solve extent')
    names = ('nc', 'ni', 'nifa', 'nwfa')
    if any(type(scalar_flags.get(n, False)) is not bool for n in names):
        raise TypeError('native scalar flags require bool')
    aerosol, mynn = microphysics == 'mp_thompson_aerosols', pbl == 'bl_mynn'
    ice = scalar_flags.get('ni', False)
    trio = all(scalar_flags.get(n, False) for n in ('nc', 'nifa', 'nwfa'))
    required = {'nifa', 'nwfa'} if aerosol else set()
    if mynn:
        if ice: required.add('ni')
        if trio: required.update(('nc', 'nifa', 'nwfa'))
    if required - species.keys():
        raise ValueError(f'native number scratch would read absent species: {sorted(required - species.keys())}')
    allocated = names if mynn else ('nifa', 'nwfa') if aerosol else ()
    if not allocated:
        return {}
    context = _context(template)
    arrays = _arrays(context)
    out = {n: arrays.zeros((nlev, ncells), np.float32) for n in allocated}
    values = {n: species.get(n, template) for n in names}
    for n, value in values.items():
        _require(n, value, np.float32, (nlev, ncells))
    bindings = {n: v.view(v.shape, np.uint32) for n, v in values.items()}
    bindings.update({'out_' + n: out.get(n, template).view(template.shape, np.uint32) for n in names})
    _launch('hexv842_mpas_prep_numbers_u32', n_cells_solve, context, bindings,
            dict(nlev=_i(nlev), ncells=_i(ncells), solved=_i(n_cells_solve), aerosol=_i(aerosol),
                 mynn=_i(mynn), ice=_i(ice), trio=_i(trio)))
    return out


def prepare_mass(*, fields, scalar_names, rvord=None):
    """Prepare resident native mass columns using the unchanged v8.4.2 mass operator.

    ``resident_v842.preparation_source`` changes only the interface kernel;
    this uses its identical mass kernel already shipped in Metal.
    """
    names = tuple(scalar_names)
    if not names or names[0] != 'qv' or len(names) != len(set(names)):
        raise ValueError('MPAS preparation requires unique scalars with qv first')
    nlev, ncells = fields['rho_zz'].shape
    mass = (nlev, ncells)
    _require('scalars', fields['scalars'], np.float32, (len(names), *mass))
    for n in ('rho_zz','theta_m','exner','pressure_base','pressure_p','zz'):
        _require(n, fields[n], np.float32, mass)
    _require('zgrid', fields['zgrid'], np.float32, (nlev+1, ncells))
    context = _context(fields['rho_zz'])
    arrays = _arrays(context)
    out = {n: arrays.empty(mass, np.float32) for n in
           ('zz_p','rho_dry','rho_p','th_p','t_p','pi_p','pres_p','zmid_p','dz_p')}
    out['qout'] = arrays.empty((len(names), *mass), np.float32)
    out['invalid'] = arrays.zeros((1,), np.int32)
    _launch('hexv841_prep_mass_v841_f32', ncells, context,
            {**{n: fields[n] for n in ('rho_zz','theta_m','scalars','exner','pressure_base','pressure_p','zgrid','zz')}, **out},
            dict(nlev=_i(nlev),ncells=_i(ncells),nspecies=_i(len(names)),
                 rvord=np.float32(np.float32(461.6)/np.float32(287.) if rvord is None else rvord)))
    return out
