"""Native MPAS longwave upper gas-band operators, without driver admission."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _context, _require
from .metal_tendencies_v842 import _launch

# Reduced native tables in Fortran-linear order, one dictionary per band.
BAND_TABLE_SIZES = {
    9: {'absa':7020,'absb':2820,'selfref':120,'forref':48,'ka_mn2o':2052,'kb_mn2o':228,'fracrefa':108,'fracrefb':12},
    10: {'absa':390,'absb':1410,'selfref':60,'forref':24,'fracrefa':6,'fracrefb':6},
    11: {'absa':520,'absb':1880,'selfref':80,'forref':32,'ka_mo2':152,'kb_mo2':152,'fracrefa':8,'fracrefb':8},
    12: {'absa':4680,'selfref':80,'forref':32,'fracrefa':72},
    13: {'absa':2340,'selfref':40,'forref':16,'ka_mco2':684,'ka_mco':684,'kb_mo3':76,'fracrefa':36,'fracrefb':4},
    14: {'absa':130,'absb':470,'selfref':20,'forref':8,'fracrefa':2,'fracrefb':2},
    15: {'absa':1170,'selfref':20,'forref':8,'ka_mn2':342,'fracrefa':18},
    16: {'absa':1170,'absb':470,'selfref':20,'forref':8,'fracrefa':18,'fracrefb':2},
}


def band(band_number, laytrop, fs, isv, wx, chi, tables, gas_tau, fracs, *, oneminus):
    """Apply one band to native setcoef state and reduced coefficient tables.

    The caller must supply source-valid indices and nonzero gas-mixture
    denominators where used; the source does not define out-of-range reads.
    """
    if band_number not in BAND_TABLE_SIZES:raise ValueError('unsupported native upper LW band')
    if fs.ndim!=3 or fs.shape[0]!=33:raise ValueError('native LW band needs 33 float state fields')
    _,nc,nl=map(int,fs.shape)
    if nc<1 or nl<1:raise ValueError('native LW band needs columns and layers')
    arrays={'laytrop_v':(laytrop,np.int32,(nc,)),'fs':(fs,np.float32,(33,nc,nl)),
            'isv':(isv,np.int32,(6,nc,nl)),'wx':(wx,np.float32,(nc,4,nl)),
            'chi_mls':(chi,np.float32,(413,)),'gas_tau':(gas_tau,np.float32,(nc,nl,140)),
            'fracs':(fracs,np.float32,(nc,nl,140))}
    arrays.update({name:(tables[name],np.float32,(size,)) for name,size in BAND_TABLE_SIZES[band_number].items()})
    context=_context(fs)
    for name,(value,dtype,shape) in arrays.items():
        _require(name,value,dtype,shape)
        if _context(value) is not context:raise ValueError('native LW band arrays belong to different Metal contexts')
    _launch(f'hexv842_mpas_rrtmg_lw_band{band_number}_f32',nc*nl,context,
            {name:value for name,(value,_,_) in arrays.items()},
            dict(ncol=np.int32(nc),nl=np.int32(nl),oneminus=np.float32(oneminus)))
