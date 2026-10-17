"""Metal mirror of the v8.4.1 CUDA moist-physics driver kernels.

The six ``*_v841_f32`` kernels of ``CUDA_V841_PHYSICS_DRIVER_SOURCE`` in
``cuda_driver.py`` are mirrored one shader each.  The driver methods that
launch them (``_compute_moist_dynamics_coefficients_v841``,
``_moist_dpdz_v841``, ``_moist_euler_w_v841``) and the two launch sites inside
``_step_device_v841`` are exposed as module-level functions taking the driver
(any object with ``arrays``, ``atmosphere``, ``nlev``, ``ncells``, ``nedges``)
as their first argument, as ``metal_dynamics_v841`` does.

``cqw`` is shaped ``[nlev, ncells]``.  Native MPAS assigns it only for
``k = 1 .. nlev-1`` (zero-based); row 0 is never written and never read, and
the buffer is allocated with ``MetalArrayModule.empty`` exactly as CUDA uses
``cp.empty_like``.  Nothing here zeroes or inspects that row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metal_dynamics_v841 import _arrays, _context, _f, _i, _launch, _require
from .species_row import species_row_for_names


@dataclass(frozen=True, slots=True)
class MetalV841MoistDynamicsCoefficients:
    """Resident native cqu/cqw/qtot held for one full dynamics step."""

    qtot: Any
    cqw: Any
    cqu: Any

    def validate(self, *, n_vert_levels: int, n_cells: int, n_edges: int) -> None:
        for name, shape in (
            ("qtot", (n_vert_levels, n_cells)),
            ("cqw", (n_vert_levels, n_cells)),
            ("cqu", (n_vert_levels, n_edges)),
        ):
            _require(name, getattr(self, name), np.float32, shape)


def compute_moist_dynamics_coefficients_metal_v841(
    driver: Any,
    scalars: Any,
    *,
    scalar_names: tuple[str, ...],
    validation_flag: Any,
) -> MetalV841MoistDynamicsCoefficients:
    """Reproduce atm_compute_moist_coefficients from raw start-state q."""

    species_row = species_row_for_names(scalar_names)
    expected = (len(tuple(scalar_names)), driver.nlev, driver.ncells)
    _require("scalars", scalars, np.float32, expected)
    _require("validation_flag", validation_flag, np.int32, (1,))
    nlev, ncells, nedges = int(driver.nlev), int(driver.ncells), int(driver.nedges)
    mctx = _context(scalars)
    arrays = _arrays(mctx)
    qtot = arrays.empty((nlev, ncells), np.float32)
    # The native routine never assigns the lowest-interface cqw row.  Keep
    # this allocation uninitialized and never inspect/read row zero.
    cqw = arrays.empty((nlev, ncells), np.float32)
    cqu = arrays.empty((nlev, nedges), np.float32)
    _launch(
        "moist_cell_coefficients_v841_f32", ncells, mctx,
        dict(scalars=scalars, qtot=qtot, cqw=cqw, invalid=validation_flag),
        dict(nlev=_i(nlev), ncells=_i(ncells), nmass=_i(species_row.n_mass_loading)),
    )
    _launch(
        "moist_edge_coefficients_v841_f32", nedges, mctx,
        dict(cells_on_edge=driver.atmosphere.mesh.cells_on_edge,
             scalars=scalars, cqu=cqu, invalid=validation_flag),
        dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(nedges),
             nmass=_i(species_row.n_mass_loading)),
    )
    result = MetalV841MoistDynamicsCoefficients(qtot=qtot, cqw=cqw, cqu=cqu)
    result.validate(n_vert_levels=nlev, n_cells=ncells, n_edges=nedges)
    return result


def moist_dpdz_metal_v841(
    driver: Any, density_perturbation: Any, qtot: Any, validation_flag: Any
) -> Any:
    nlev, ncells = int(driver.nlev), int(driver.ncells)
    _require("density_perturbation", density_perturbation, np.float32, (nlev, ncells))
    _require("qtot", qtot, np.float32, (nlev, ncells))
    mctx = _context(density_perturbation)
    result = _arrays(mctx).empty(density_perturbation.shape, np.float32)
    _launch(
        "moist_dpdz_v841_f32", ncells, mctx,
        dict(rho_base=driver.atmosphere.reference.rho_base,
             density_perturbation=density_perturbation, qtot=qtot,
             dpdz=result, invalid=validation_flag),
        dict(nlev=_i(nlev), ncells=_i(ncells), gravity=_f(9.80616)),
    )
    return result


def moist_euler_w_metal_v841(
    driver: Any, pressure_perturbation: Any, dpdz: Any, cqw: Any,
    validation_flag: Any,
) -> Any:
    nlev, ncells = int(driver.nlev), int(driver.ncells)
    mctx = _context(pressure_perturbation)
    result = _arrays(mctx).empty((nlev + 1, ncells), np.float32)
    vertical = driver.atmosphere.vertical
    _launch(
        "euler_w_moist_v841_f32", ncells, mctx,
        dict(pressure_p=pressure_perturbation, dpdz=dpdz, cqw=cqw,
             rdzu=vertical.rdzu, fzm=vertical.fzm, fzp=vertical.fzp,
             result=result, invalid=validation_flag),
        dict(nlev=_i(nlev), ncells=_i(ncells)),
    )
    return result


def modified_theta_dynamics_rate_metal_v841(
    driver: Any, tend_rtheta: Any, tend_rho: Any, rho: Any, theta_m: Any,
    validation_flag: Any,
) -> Any:
    """The stage-3 launch in ``_step_device_v841`` (full physics only)."""

    nlev, ncells = int(driver.nlev), int(driver.ncells)
    mctx = _context(tend_rtheta)
    result = _arrays(mctx).empty(tend_rtheta.shape, np.float32)
    _launch(
        "modified_theta_dynamics_rate_v841_f32", ncells, mctx,
        dict(tend_rtheta=tend_rtheta, tend_rho=tend_rho, rho=rho,
             theta_m=theta_m, result=result, invalid=validation_flag),
        dict(nlev=_i(nlev), ncells=_i(ncells)),
    )
    return result


def gf_dry_theta_dynamics_rate_metal_v841(
    driver: Any, modified_rate: Any, theta_m: Any, qv: Any, validation_flag: Any
) -> tuple[Any, Any]:
    """The end-of-step launch in ``_step_device_v841``; ``qv`` is ``scalars[0]``."""

    nlev, ncells = int(driver.nlev), int(driver.ncells)
    mctx = _context(modified_rate)
    arrays = _arrays(mctx)
    rthdynten = arrays.empty((nlev, ncells), np.float32)
    rqvdynten = arrays.empty((nlev, ncells), np.float32)
    _launch(
        "gf_dry_theta_dynamics_rate_v841_f32", ncells, mctx,
        dict(modified_rate=modified_rate, theta_m=theta_m, qv=qv,
             rthdynten=rthdynten, rqvdynten=rqvdynten, invalid=validation_flag),
        dict(nlev=_i(nlev), ncells=_i(ncells),
             rvord=_f(np.float32(461.6) / np.float32(287.0))),
    )
    return rthdynten, rqvdynten


__all__ = [
    "MetalV841MoistDynamicsCoefficients",
    "compute_moist_dynamics_coefficients_metal_v841",
    "gf_dry_theta_dynamics_rate_metal_v841",
    "modified_theta_dynamics_rate_metal_v841",
    "moist_dpdz_metal_v841",
    "moist_euler_w_metal_v841",
]
