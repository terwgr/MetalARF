"""RRFS cld_tempo and cld_fraction_mynn radiation carrier preparation."""
import math
import numpy as np

from ...metal_dynamics_v841 import _arrays, _context, _i, _require
from ..metal_tendencies_v842 import _launch

CLOUD_SCHEMES = ('cld_tempo', 'cld_fraction_mynn')


def cloudiness(*, scheme, itimestep, convection_scheme, prepared, mesh_density,
               xland, len_disp, diag, radshared, radiation_carriers=None):
    if scheme not in CLOUD_SCHEMES:
        raise NotImplementedError(f'RRFS Metal cloud scheme {scheme!r} is not implemented')
    mynn = scheme == 'cld_fraction_mynn'
    gfl = convection_scheme.strip() == 'cu_grell_freitas_li'
    qv = prepared['qv']
    nlev, ncells = qv.shape
    shape = (nlev, ncells)
    if not nlev or not ncells or not math.isfinite(len_disp) or len_disp <= 0:
        raise ValueError('RRFS cloudiness requires nonempty columns and positive resolved length')
    context = _context(qv)
    if (5 * nlev + 2) * ncells * 4 > context.recommended_working_set_size:
        raise MemoryError('RRFS cloud carriers exceed the Metal working set')
    inputs = {name: prepared[name] for name in ('qv', 'qc', 'qi', 'qs')}
    inputs.update(density=mesh_density, xland=xland)
    for suffix, enabled in (('bl', mynn), ('cu', gfl)):
        for stem in ('cldfrac', 'qc', 'qi'):
            name = stem + '_' + suffix
            inputs[name] = diag[name] if enabled else qv
    for name, value in inputs.items():
        _require(name, value, np.float32, (ncells,) if name in ('density', 'xland') else shape)
        if _context(value) is not context:
            raise ValueError('RRFS cloud inputs belong to different Metal contexts')
    _require('diag_cldfrac', diag['cldfrac'], np.float32, shape)
    if _context(diag['cldfrac']) is not context:
        raise ValueError('RRFS cloud diagnostic belongs to another Metal context')
    arrays = _arrays(context)
    out = {name: arrays.empty(shape, np.float32) for name in
           ('qvrad', 'qcrad', 'qirad', 'qsrad', 'cldfrac')}
    out.update({name: arrays.empty((ncells,), np.float32) for name in ('dx', 'xland')})
    if mynn:
        out.update(qcbl=diag['qc_bl'], qibl=diag['qi_bl'])
    if gfl:
        out.update(qccu=diag['qc_cu'], qicu=diag['qi_cu'])
    targets = radiation_carriers or {}
    for name, target in targets.items():
        _require(name, target, np.float32, out[name].shape)
        if _context(target) is not context:
            raise ValueError('RRFS cloud carrier belongs to another Metal context')
    _launch('hexrrfs_cloudiness', nlev * ncells, context,
            dict(inputs, **{name: out[name] for name in ('qvrad', 'qcrad', 'qirad', 'qsrad', 'cldfrac', 'dx')},
                 land=out['xland'], diag_cldfrac=diag['cldfrac']),
            dict(nlev=_i(nlev), ncells=_i(ncells), mynn=_i(mynn), step=_i(itimestep),
                 gfl=_i(gfl), len_disp=np.float32(len_disp)))
    for name, target in targets.items():
        target[...] = out[name]
    radshared.update(out)
    return out


def cloudiness_slot(call):
    pools = call.pools
    return cloudiness(scheme=call.scheme, itimestep=pools['itimestep'],
        convection_scheme=call.options['config_convection_scheme'], prepared=pools['prepared'],
        mesh_density=pools['mesh']['meshDensity'], xland=pools['sfc_input']['xland'],
        len_disp=pools['len_disp'], diag=pools['diag_physics'], radshared=pools.setdefault('radshared', {}),
        radiation_carriers=pools.get('radiation_carriers'))
