"""Native MPAS diagnostic and cold-EOS components on resident Metal arrays.

These components do not register or admit a complete native forecast provider.
"""
import numpy as np
from ..cuda_physics_prep_v841 import RV_OVER_RD_F32
from ..metal_dynamics_v841 import _arrays,_context,_require
from .metal_tendencies_v842 import _launch


def _run(name,resources,parameters):
    first=next(iter(resources.values()));shape=first.shape;context=_context(first)
    for field,value in resources.items():
        _require(field,value,np.float32,shape)
        if _context(value) is not context:raise ValueError('native diagnostic arrays belong to different Metal contexts')
    count=int(first.size)
    if count:
        _launch('hexv842_mpas_'+name+'_v842_f32',count,context,
            {field:value.view((count,)) for field,value in resources.items()},
            {'count':np.int32(count),**{field:np.float32(value) for field,value in parameters.items()}})


def output_diagnostics(atmosphere,names):
    """Use unclipped prognostic water, preserving native output operand order."""
    state,saved=atmosphere.state,atmosphere.saved
    context=_context(state.rho);xp=_arrays(context)
    theta,rho,pressure=(xp.empty(state.rho.shape,np.float32) for _ in range(3))
    qv = state.scalars[tuple(names).index('qv')] if 'qv' in names else xp.zeros(state.rho.shape, np.float32)
    resources=dict(theta_m=saved.theta_m,qv=qv,
        rho_zz=state.rho,zz=atmosphere.vertical.zz,pressure_base=atmosphere.reference.pressure_base,
        pressure_p=saved.pressure_perturbation,theta=theta,rho=rho,pressure=pressure)
    _run('output_diagnostics',resources,{'rvord':RV_OVER_RD_F32})
    return theta,rho,pressure


def initialize_thermodynamics(atmosphere,*,base_only=False):
    """Apply native cold EOS, or refresh only base words during restart.

    The source's Arm/glibc binary64-FMA power core is evaluated as software
    binary64 words. The caller owns the source-valid thermodynamic state.
    """
    saved,reference=atmosphere.saved,atmosphere.reference
    rgas,cp=np.float32(287),np.float32(1004.5)
    resources=dict(zz=atmosphere.vertical.zz,rtheta_p=saved.rho_theta_perturbation,
        rtheta_base=reference.rho_theta_base,
        exner=saved.exner.copy() if base_only else saved.exner,exner_base=reference.exner_base,
        pressure_p=saved.pressure_perturbation.copy() if base_only else saved.pressure_perturbation,
        pressure_base=reference.pressure_base)
    _run('cold_thermodynamics',resources,dict(rgas=rgas,p0=np.float32(1e5),rcv=np.float32(rgas/np.float32(cp-rgas))))
