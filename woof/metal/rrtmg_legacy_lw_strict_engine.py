"""Source-ordered strict regional classic RRTMG longwave pipeline."""

import numpy as np

from woof.core import rrtmg_lw as lw
from .array import MetalArrayModule
from .rrtmg_legacy_lw_strict_gas import submit_band
from .rrtmg_legacy_lw_strict_transfer import transfer
from .wrf_exact_rrtmg_lw_prep import prepare


def run_lw_prepared_strict(context, prepared, coefficients):
    state = prepare(context, prepared, coefficients)
    errors = state['err_flag'].to_numpy()
    if np.any(errors):
        raise ValueError(f'rlw_cldprmc Metal abort, code {int(errors[0])}')
    xp = MetalArrayModule(context)
    ncol, nl = int(prepared['ncol']), int(prepared['nlay'])

    def upload(values, dtype=np.float32):
        return xp.asarray(np.ascontiguousarray(np.asarray(values, dtype=dtype).ravel(order='F')))

    def coeff(name):
        return upload(coefficients[name])

    band_tables = {f'kg{band:02d}/{name}': coeff(f'kg{band:02d}/{name}')
                   for band, names in lw.GPU_BAND_TABS.items() for name in names}
    chi = coeff('ref/chi_mls')
    taug = xp.empty((ncol, nl, lw.NGPTLW), np.float32)
    fracs = xp.empty((ncol, nl, lw.NGPTLW), np.float32)
    for band in lw.GPU_BAND_TABS:
        submit_band(band, state['laytrop'], state['fs'], state['isv'], state['wx'],
                    chi, band_tables, taug, fracs, oneminus=coefficients['con/oneminus'])

    fields = {name: state[name] for name in ('pwvcm', 'semiss', 'planklay', 'planklev', 'plankbnd', 'ngb')}
    fields.update(a0=upload(lw.A0), a1=upload(lw.A1), a2=upload(lw.A2),
                  cldfmc=state['cloud']['cldfmcl'], taucmc=state['cloud']['taucmcl'],
                  taug=taug, taua=xp.zeros((ncol, nl, lw.NBNDLW), np.float32), fracs=fracs,
                  pz=state['plev'], ngs=upload(lw.NGS, np.int32), delwave=coeff('wvn/delwave'),
                  **{name: coeff('tbl/'+name) for name in ('tau_tbl', 'exp_tbl', 'tfn_tbl')})
    out = transfer(fields, bpade=coefficients['tbl/bpade'], wtdiff=lw.WTDIFF,
                   fluxfac=coefficients['con/fluxfac'], heatfac=coefficients['con/heatfac'])
    return {'uflx': out['totuflux'], 'dflx': out['totdflux'], 'uflxc': out['totuclfl'],
            'dflxc': out['totdclfl'], 'hr': out['htr'], 'hrc': out['htrc']}
