"""Native MPAS lagged heating and cell recovery on resident Metal arrays."""
import numpy as np
from ..metal_dynamics_v841 import _context, _require
from .metal_tendencies_v842 import _launch


def _plane(driver, name, value, context):
    _require(name, value, np.float32, (driver.nlev, driver.ncells))
    if _context(value) is not context:
        raise ValueError('native recovery arrays must share a Metal context')
    return value


def add_lagged(driver, rho, diabatic_theta, tend_rt):
    context = _context(rho)
    arrays = {name: _plane(driver, name, value, context) for name, value in
              (('rho', rho), ('diabatic_theta', diabatic_theta), ('tend_rt', tend_rt))}
    _launch('hexv842_mpas_add_lagged_f32', driver.nlev * driver.ncells, context, arrays,
            dict(nlev=np.int32(driver.nlev), ncells=np.int32(driver.ncells)))


def zero_columns(plane, columns):
    """Write source positive-zero words into selected resident columns."""
    if plane.ndim != 2:
        raise ValueError('zero-column target must be a level-column plane')
    context = _context(plane)
    _require('zero-column target', plane, np.float32)
    _require('zero-column indices', columns, np.int32, (int(columns.size),))
    if _context(columns) is not context:
        raise ValueError('zero-column indices belong to a different Metal context')
    nz, nc = plane.shape
    count = int(columns.size)
    if count:
        _launch('hexv842_mpas_zero_lagged_f32', nz * count, context,
                dict(spec_cells=columns, diabatic_theta=plane),
                dict(nlev=np.int32(nz), ncells=np.int32(nc), nspec=np.int32(count)))


def zero_speczone(driver, diabatic_theta):
    regional = getattr(driver, 'regional_v841', None)
    if regional is None:
        return
    context = _context(diabatic_theta)
    _plane(driver, 'diabatic_theta', diabatic_theta, context)
    zero_columns(diabatic_theta, regional.masks.spec_cells)


def recover_cells(driver, saved_diag, acoustic, stage, diabatic_theta, large_timestep):
    reference = driver.atmosphere.reference
    context = _context(saved_diag.density_perturbation)
    source = dict(rho_p_saved=saved_diag.density_perturbation,
        rtheta_p_saved=saved_diag.rho_theta_perturbation, rho_pp=acoustic.rho_pp,
        rtheta_pp=acoustic.rtheta_pp, rho_base=reference.rho_base,
        rtheta_base=reference.rho_theta_base, exner_base=reference.exner_base,
        zz=driver.atmosphere.vertical.zz, exner_saved=saved_diag.exner,
        pressure_p_saved=saved_diag.pressure_perturbation,
        diabatic_theta=diabatic_theta if diabatic_theta is not None else saved_diag.density_perturbation)
    for name, value in source.items():
        _plane(driver, name, value, context)
    names = ('rho', 'rtheta', 'rho_p', 'rtheta_p', 'theta', 'exner', 'pressure_p')
    outputs = {name: driver.arrays.empty((driver.nlev, driver.ncells), np.float32) for name in names}
    invalid = driver.arrays.zeros((1,), np.int32)
    _launch('hexv842_mpas_recover_cells_f32', driver.nlev * driver.ncells, context,
            dict(source, invalid=invalid, **outputs), dict(nlev=np.int32(driver.nlev), ncells=np.int32(driver.ncells),
            stage=np.int32(stage), rgas=np.float32(287), cp=np.float32(1004.5), p0=np.float32(100000),
            has_diabatic=np.int32(diabatic_theta is not None), dt=np.float32(large_timestep)))
    if stage == 3 and invalid.to_numpy()[0]:
        raise ValueError('native recovery requires a positive finite normal Exner argument; CUDA fallback is unsupported')
    return tuple(outputs[name] for name in names)
