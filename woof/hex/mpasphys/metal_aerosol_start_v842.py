"""Metal initialization for the native aerosol startup transaction."""
import numpy as np
from ..metal_dynamics_v841 import _context, _i
from .metal_tendencies_v842 import _launch


def initialize(startup):
    f = np.float32
    nz, ns = (startup.dimensions[x] for x in ('nVertLevels', 'nCellsSolve'))
    if nz < 2 or ns < 1:
        raise ValueError('native aerosol initializer requires two levels and a solve cell')
    state = startup.native._storage['state', 'scalars'][0]
    indices = {n: int(startup.native._dimension_values['state', 'index_' + n]) - 1 for n in ('nwfa', 'nifa')}
    xp = startup.session.pools.arrays
    z = xp.asarray(np.ascontiguousarray(startup.field('mesh', 'zgrid')[:nz+1, :ns].T, dtype=f))
    zz = xp.asarray(np.ascontiguousarray(startup.field('mesh', 'zz')[0, :ns], dtype=f))
    rho = xp.asarray(np.ascontiguousarray(startup.field('state', 'rho_zz')[0, :ns], dtype=f))
    area = xp.asarray(np.ascontiguousarray(startup.field('mesh', 'areaCell')[:ns], dtype=f))
    prof = {}
    arrays = {}
    for name, (n1,n0) in (('nwfa',(50.e6,300.e6)),('nifa',(.5e6,1.5e6))):
        column = state[indices[name], :nz, :ns]
        on = bool(column.min() == 0 and column.max() == 0)
        log = f(f(-1.0) * f(np.log(np.float64(f(f(n1)/f(n0))))))
        prof[name] = (on, log, f(n0), f(n1))
        arrays[name] = xp.asarray(np.ascontiguousarray(column.T, dtype=f))
    out = xp.zeros((ns,), dtype=f)
    w,i = prof['nwfa'],prof['nifa']
    _launch('hexv842_aerosol_start_f32',ns,_context(z),
        dict(zgrid=z,zz=zz,rho=rho,area=area,nwfa=arrays['nwfa'],nifa=arrays['nifa'],nwfa2d=out),
        dict(nz=_i(nz),ns=_i(ns),w_on=_i(w[0]),w_log=w[1],w0=w[2],w1=w[3],
             i_on=_i(i[0]),i_log=i[1],i0=i[2],i1=i[3]))
    state[indices['nwfa'], :nz, :ns] = arrays['nwfa'].to_numpy().T
    state[indices['nifa'], :nz, :ns] = arrays['nifa'].to_numpy().T
    startup.write('diag_physics','nwfa2d',out.to_numpy(),count=ns)
    startup.write('diag_physics','nifa2d',f(0),count=ns)
