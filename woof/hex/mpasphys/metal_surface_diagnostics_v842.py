"""Native post-LSM diagnostic publication on resident Metal pool views."""
import numpy as np
from ..metal_dynamics_v841 import _context, _require
from .metal_tendencies_v842 import _launch
from .sfclay_mm5_v842 import FIELDS, CONSTANTS, _MPAS_NAMES


def surface_diagnostics(packed, constants, *, lsm_scheme, xice, xice_threshold,
                        q2mxy=None, t2mxy=None):
    lsm={'sf_noah':1,'sf_noahmp':2}.get(lsm_scheme,0)
    if not lsm:
        return
    n=int(packed.shape[1])
    _require('surface diagnostics',packed,np.float32,(len(FIELDS),n))
    _require('surface constants',constants,np.float32,(len(CONSTANTS),))
    _require('sea ice',xice,np.float32,(n,))
    if lsm==2 and (q2mxy is None or t2mxy is None):
        raise ValueError('Noah-MP diagnostics require q2mxy and t2mxy')
    q2mxy=packed[0] if q2mxy is None else q2mxy
    t2mxy=packed[0] if t2mxy is None else t2mxy
    context=_context(packed)
    for name,value in (('q2mxy',q2mxy),('t2mxy',t2mxy)):
        _require(name,value,np.float32,(n,))
    if any(_context(a) is not context for a in (constants,xice,q2mxy,t2mxy)):
        raise ValueError('surface diagnostics arrays belong to different contexts')
    if n:
        _launch('hexv842_mpas_surface_diagnostics_f32',n,context,
                dict(a=packed.view((len(FIELDS)*n,)),xice=xice,q2mxy=q2mxy,t2mxy=t2mxy,constants=constants),
                dict(n=np.int32(n),lsm=np.int32(lsm),threshold=np.float32(xice_threshold)))


def run_native_diagnostics(provider, call):
    """Use current surface pressure and publish only q2, t2 and th2."""
    lsm=call.options['config_lsm_scheme']
    if lsm == 'sf_ruc':
        from .rrfs.metal_ruc_boundaries import surface_diagnostics as ruc_diagnostics
        from .rrfs.ruc_lsm_8c9f1a8 import ATM_BINDINGS, DIAG_BINDINGS, _native_scalar

        diag = provider['diag_physics']
        fields = {name: diag[DIAG_BINDINGS.get(name, name)]
                  for name in ('hfx', 'qfx', 'qsfc', 'cqs', 'cqs2', 'chs', 'chs2', 't2', 'th2', 'q2')}
        fields.update((name, provider.prepared(ATM_BINDINGS[name]))
                      for name in ('t3d', 'qv3d', 'dz', 'rho3d', 'p3d'))
        fields.update(tsk=provider.field('sfc_input', 'skintemp'),
                      psfc2d=provider.field('diag', 'surface_pressure'))
        return ruc_diagnostics(fields, solve_cells=int(provider['nCellsSolve']),
                               cp=_native_scalar(provider, 'cp'), rovcp=_native_scalar(provider, 'rovcp'),
                               mynn=call.options['config_sfclayer_scheme'].strip() == 'sf_mynnsfclay')
    if lsm not in ('sf_noah','sf_noahmp'):
        return
    xp=provider.cp
    aliases={'g':'gravity','rovcp':'rcp','r':'R_d','ep1':'ep_1','ep2':'ep_2','p1000mb':'P0'}
    constants=xp.asarray(np.array([provider.native.get_constant(aliases.get(n,n)) for n in CONSTANTS],np.float32))
    diag,sfc=provider['diag_physics'],provider['sfc_input']
    pressure=provider.field('diag','surface_pressure')
    n=int(pressure.size)
    packed=xp.zeros((len(FIELDS),n),np.float32)
    for name in ('hfx','qfx','qsfc','chs2','cqs2'):
        packed[FIELDS.index(name)][...]=diag[_MPAS_NAMES.get(name,name)]
    packed[FIELDS.index('psfcpa')][...]=pressure
    packed[FIELDS.index('tsk')][...]=sfc['skintemp']
    packed[FIELDS.index('xland')][...]=sfc['xland']
    extra=({name:provider.field('output_noahmp',name) for name in ('q2mxy','t2mxy')} if lsm=='sf_noahmp' else {})
    surface_diagnostics(packed,constants,lsm_scheme=lsm,xice=sfc['xice'],
                        xice_threshold=provider['scalar_state']['xice_threshold'],**extra)
    for name in ('q2','t2','th2'):
        diag[_MPAS_NAMES.get(name,name)][...]=packed[FIELDS.index(name)]
