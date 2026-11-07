"""Native MPAS v8.4.2 RRTMG longwave Metal building blocks."""
from __future__ import annotations
import re
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch
from . import rrtmg_lw_v842 as native
from woof.core import rrtmg_lw as legacy


def _static_words(name, size, integer):
    """Transport source DATA words without recomputing coefficient values."""
    kind = 'int' if integer else 'float'
    match = re.search(r'extern "C" __device__ const '+kind+r' mpas_static_'+name+
                      r'\['+str(size)+r'\] = \{(.*?)\};',native._INIT_SOURCE,re.S)
    if match is None:
        raise ValueError(f'native RRTMG LW static table {name} is missing')
    words = [word.strip() for word in match.group(1).split(',') if word.strip()]
    if len(words) != size:
        raise ValueError(f'native RRTMG LW static table {name} has the wrong size')
    if integer:
        return np.asarray([int(word) for word in words],dtype=np.int32)
    return np.asarray([float.fromhex(word.removesuffix('f')) for word in words],dtype=np.float32)


def initialize_tables(raw, context):
    """Reduce decoded native table words in source order on Metal.

    ``raw`` is a mapping of rrlw_kgNN modules to contiguous Fortran-order
    float32 device arrays produced from authenticated native table records.
    The returned arrays are immutable across forecast steps and can be
    reconstructed from the same resource packet on restart.
    """
    required=[]
    for band,dest,name,points,first,weighted in native._REDUCTIONS:
        value=raw[f'rrlw_kg{band:02d}'][name]
        _require(f'native RRTMG LW {band}/{name}',value,np.float32,(points*16,))
        if _context(value) is not context:
            raise ValueError('native RRTMG LW tables belong to different Metal contexts')
        required.append((band,dest,points,first,weighted,value))
    static={name:_static_words(name,size,integer) for name,size,integer in native._STATIC_ARRAYS}
    words=256+3*10001+sum(size for _,size,_ in native._STATIC_ARRAYS)
    words+=sum(points*int(legacy.NGC[band-1]) for band,_,points,_,_,_ in required)
    if words*4 > context.recommended_working_set_size:
        raise MemoryError('native RRTMG LW tables exceed Metal recommended working set')
    xp=_arrays(context)
    result={name:xp.asarray(value) for name,value in static.items()}
    result.update({name:np.float32(value) for name,value in native._STATIC_SCALARS.items()})
    weights=xp.empty((256,),np.float32);result['rwgt']=weights
    _launch('hexv842_mpas_rrtmg_lw_rwgt_f32',1,context,dict(rwgt=weights),dict(n=np.int32(1)))
    for name in ('tau_tbl','exp_tbl','tfn_tbl'):
        result[name]=xp.empty((10001,),np.float32)
    _launch('hexv842_mpas_rrtmg_lw_tables_f32',10001,context,
            dict(tau=result['tau_tbl'],exp_table=result['exp_tbl'],tfn=result['tfn_tbl']),dict(n=np.int32(10001)))
    reduced={}
    for band,dest,points,first,weighted,value in required:
        size=points*int(legacy.NGC[band-1]);out=xp.empty((size,),np.float32)
        _launch('hexv842_mpas_rrtmg_lw_cmbgb_f32',size,context,dict(raw=value,rwgt=weights,out=out),
                dict(band=np.int32(band-1),points=np.int32(points),first=np.int32(first),weighted=np.int32(weighted)))
        reduced[dest]=out
        if dest.endswith(('/ka','/kb')):
            reduced[dest.rsplit('/',1)[0]+'/'+('absa' if dest.endswith('/ka') else 'absb')]=out
    result['reduced']=reduced
    result['band']={band:tuple(reduced[f'kg{band:02d}/{name}'] for name in names)
                    for band,names in legacy.GPU_BAND_TABS.items()}
    return result


def inatm(plev, wkl, wx, coldry, wbrodl, pwvcm, *, grav, avogad):
    """Apply native INATM using the full 38-species column stride."""
    if plev.ndim != 2:
        raise ValueError('native RRTMG LW pressure interfaces must be rank two')
    ncol,nlev = map(int,plev.shape)
    nl = nlev-1
    if ncol < 1 or nl < 1:
        raise ValueError('native RRTMG LW needs at least one column and layer')
    arrays={'plev':(plev,(ncol,nl+1)), 'wkl':(wkl,(ncol,native.MXMOL,nl)),
            'wx':(wx,(ncol,native.MAXXSEC,nl)), 'coldry':(coldry,(ncol,nl)),
            'wbrodl':(wbrodl,(ncol,nl)), 'pwvcm_v':(pwvcm,(ncol,))}
    context=_context(plev)
    for name,(value,shape) in arrays.items():
        _require(name,value,np.float32,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW INATM arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_inatm_f32',ncol,context,
            {name:value for name,(value,_) in arrays.items()},
            dict(ncol=np.int32(ncol),nl=np.int32(nl),grav=np.float32(grav),avogad=np.float32(avogad)))


def prepare(model, interfaces, surface, pin, ozone, atmosphere, levels, *,
            icloud, reqc, reqi, reqs, o3input, presence):
    """Prepare one fixed-depth batch of native longwave radiation columns.

    The caller must group columns by the native top-pressure-derived layer
    count before calling this stage.

    Climatological ozone interpolation requires at least two source levels:
    the native one-level lower-neighbor read has no defined source value.
    """
    if model.ndim != 3 or model.shape[1] != 12 or atmosphere.ndim != 3 or atmosphere.shape[1] != 12:
        raise ValueError('native RRTMG LW preparation requires twelve model and radiation fields')
    nc,_,nm=map(int,model.shape);nl=int(atmosphere.shape[2])
    if nc < 1 or nm < 1 or nl <= nm or int(atmosphere.shape[0]) != nc:
        raise ValueError('native RRTMG LW preparation requires extended radiation levels')
    if o3input not in (0,2):
        raise ValueError('native RRTMG LW ozone input selector must be 0 or 2')
    if pin.ndim != 1:
        raise ValueError('native RRTMG LW ozone pressure must be rank one')
    nz=int(pin.size)
    if o3input == 2 and nz < 2:
        raise ValueError('native RRTMG LW ozone interpolation needs two defined source levels')
    expected_ozone=(nc,nz) if o3input == 2 else ozone.shape
    if o3input == 0 and ozone.size < 1:
        raise ValueError('native RRTMG LW requires a valid unused ozone buffer')
    arrays={'m':(model,(nc,12,nm)), 'interfaces':(interfaces,(nc,2,nm+1)),
            'surface':(surface,(nc,6)), 'pin':(pin,(nz,)), 'ozone':(ozone,expected_ozone),
            'a':(atmosphere,(nc,12,nl)), 'lev':(levels,(nc,2,nl+1))}
    context=_context(model)
    for name,(value,shape) in arrays.items():
        _require(name,value,np.float32,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW preparation arrays belong to different Metal contexts')
    if nz < 1:
        raise ValueError('native RRTMG LW requires a valid ozone pressure buffer')
    _launch('hexv842_mpas_rrtmg_lw_prepare_f32',nc,context,
            {name:value for name,(value,_) in arrays.items()},
            dict(nc=np.int32(nc),nm=np.int32(nm),nl=np.int32(nl),icloud=np.int32(icloud),
                 reqc=np.int32(reqc),reqi=np.int32(reqi),reqs=np.int32(reqs),
                 o3input=np.int32(o3input),nz=np.int32(nz),presence=np.int32(presence)))


def load_tables(context, resources=None):
    """Authenticate native LW records, upload raw words, and initialize them.

    The host transport reads the pinned Fortran records without coefficient
    arithmetic; all g-point reduction remains on Metal. If supplied,
    ``resources`` is the native RRTMG-LW table descriptor.
    """
    from .metal_radiation_tables_v842 import load_raw_tables
    raw=load_raw_tables('lw',resources)
    raw_bytes=sum(value.nbytes for row in raw.values() for value in row.values())
    words=256+3*10001+sum(size for _,size,_ in native._STATIC_ARRAYS)
    words+=sum(points*int(legacy.NGC[band-1]) for band,_,_,points,_,_ in native._REDUCTIONS)
    if raw_bytes+words*4 > context.recommended_working_set_size:
        raise MemoryError('native RRTMG LW raw and initialized tables exceed Metal recommended working set')
    xp=_arrays(context)
    device={module:{name:xp.asarray(value.ravel(order='F').copy()) for name,value in row.items()}
            for module,row in raw.items()}
    return initialize_tables(device,context)


def layer_counts(interfaces, counts):
    """Calculate native top-pressure layer counts for a column batch."""
    if interfaces.ndim != 3 or interfaces.shape[1] != 2:
        raise ValueError('native RRTMG LW interfaces require pressure and temperature')
    nc,_,nlev=map(int,interfaces.shape);nm=nlev-1
    if nc < 1 or nm < 1:
        raise ValueError('native RRTMG LW requires at least one column and model layer')
    context=_context(interfaces)
    for name,value,dtype,shape in (('interfaces',interfaces,np.float32,(nc,2,nlev)),('counts',counts,np.int32,(nc,))):
        _require(name,value,dtype,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW count arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_counts_f32',nc,context,dict(interfaces=interfaces,counts=counts),dict(nc=np.int32(nc),nm=np.int32(nm)))


def outputs(model, atmosphere, up, down, up_clear, down_clear, heating, tendency, radii, flux, *, icloud):
    """Publish native longwave tendencies, radii and diagnostics for one depth group."""
    if model.ndim != 3 or model.shape[1] != 12 or atmosphere.ndim != 3 or atmosphere.shape[1] != 12:
        raise ValueError('native RRTMG LW outputs require twelve model and radiation fields')
    nc,_,nm=map(int,model.shape);nl=int(atmosphere.shape[2])
    if nc < 1 or nm < 1 or nl < nm or int(atmosphere.shape[0]) != nc:
        raise ValueError('native RRTMG LW output layer dimensions are inconsistent')
    arrays={'m':(model,(nc,12,nm)),'a':(atmosphere,(nc,12,nl)),
            'u':(up,(nc,nl+1)),'d':(down,(nc,nl+1)),
            'uc':(up_clear,(nc,nl+1)),'dc':(down_clear,(nc,nl+1)),
            'hr':(heating,(nc,nl)),'tendency':(tendency,(nc,nm)),
            'radii':(radii,(nc,3,nm)),'flux':(flux,(nc,11))}
    context=_context(model)
    for name,(value,shape) in arrays.items():
        _require(name,value,np.float32,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW output arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_outputs_f32',nc,context,
            {name:value for name,(value,_) in arrays.items()},
            dict(nc=np.int32(nc),nm=np.int32(nm),nl=np.int32(nl),icloud=np.int32(icloud)))


def mcica(atmosphere, surface, cloud, liquid, ice, snow, *, overlap, idcor, julday):
    """Apply the wrapper's native KISS stream to one fixed-depth group.

    The caller must enforce the native pressure-order precondition and clear
    the output slabs before the overlap-zero early return.
    """
    if atmosphere.ndim != 3 or atmosphere.shape[1] != 12:
        raise ValueError('native RRTMG LW McICA requires twelve atmospheric fields')
    nc,_,nl=map(int,atmosphere.shape)
    if nc < 1 or nl < 4:
        raise ValueError('native RRTMG LW KISS initialization requires four defined layers')
    if not 0 <= overlap <= 5:
        raise ValueError('native RRTMG LW McICA overlap must be between 0 and 5')
    arrays={'a':(atmosphere,(nc,12,nl)),'surface':(surface,(nc,6)),
            'cloud':(cloud,(nc,nl,140)),'liquid':(liquid,(nc,nl,140)),
            'ice':(ice,(nc,nl,140)),'snow':(snow,(nc,nl,140))}
    context=_context(atmosphere)
    for name,(value,shape) in arrays.items():
        _require(name,value,np.float32,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW McICA arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_mcica_f32',nc,context,
            {name:value for name,(value,_) in arrays.items()},
            dict(nc=np.int32(nc),nl=np.int32(nl),overlap=np.int32(overlap),
                 idcor=np.int32(idcor),julday=np.int32(julday)))


def optical_depth(taug, taua, ngb, taut):
    """Combine optical depths; ``ngb`` must be the initialized native band map."""
    if taug.ndim != 3 or taug.shape[2] != 140:
        raise ValueError('native RRTMG LW gas optical depth requires 140 g-points')
    nc,nl,_=map(int,taug.shape)
    if nc < 1 or nl < 1:
        raise ValueError('native RRTMG LW optical depths require columns and layers')
    arrays={'gas_tau':(taug,np.float32,(nc,nl,140)),'taua':(taua,np.float32,(nc,nl,16)),
            'ngb':(ngb,np.int32,(140,)),'taut':(taut,np.float32,(nc,nl,140))}
    context=_context(taug)
    for name,(value,dtype,shape) in arrays.items():
        _require(name,value,dtype,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW optical depth arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_taut_f32',nc*nl*140,context,
            {name:value for name,(value,_,_) in arrays.items()},dict(ncol=np.int32(nc),nl=np.int32(nl)))


def diffusivity(pwvcm, a0, a1, a2, secdiff):
    """Compute the native sixteen-band diffusivity coefficients."""
    if pwvcm.ndim != 1 or pwvcm.size < 1:
        raise ValueError('native RRTMG LW precipitable water requires columns')
    nc=int(pwvcm.size)
    arrays={'pwvcm':(pwvcm,(nc,)),'a0':(a0,(16,)),'a1':(a1,(16,)),
            'a2':(a2,(16,)),'secdiff':(secdiff,(nc,16))}
    context=_context(pwvcm)
    for name,(value,shape) in arrays.items():
        _require(name,value,np.float32,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW diffusivity arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_secdiff_f32',nc,context,
            {name:value for name,(value,_) in arrays.items()},dict(ncol=np.int32(nc)))


def cloud_transfer(cloud, tau, secdiff, ngb, odcld, efclfrac, icldlyr):
    """Prepare cloud transfer; use the initialized native band map.

    The source caller initializes ``icldlyr`` to zero before this stage.
    """
    if cloud.ndim != 3 or cloud.shape[2] != 140:
        raise ValueError('native RRTMG LW cloud transfer requires 140 g-points')
    nc,nl,_=map(int,cloud.shape)
    if nc < 1 or nl < 1:
        raise ValueError('native RRTMG LW cloud transfer requires columns and layers')
    arrays={'cloud':(cloud,np.float32,(nc,nl,140)),'tau':(tau,np.float32,(nc,nl,140)),
            'secdiff':(secdiff,np.float32,(nc,16)),'ngb':(ngb,np.int32,(140,)),
            'odcld':(odcld,np.float32,(nc,nl,140)),
            'efclfrac':(efclfrac,np.float32,(nc,nl,140)),
            'icldlyr':(icldlyr,np.int32,(nc,nl))}
    context=_context(cloud)
    for name,(value,dtype,shape) in arrays.items():
        _require(name,value,dtype,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW cloud transfer arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_prol_f32',nc*nl,context,
            {name:value for name,(value,_,_) in arrays.items()},dict(ncol=np.int32(nc),nl=np.int32(nl)))


def cloud_optics(cloud, ice, liquid, snow, ice_radius, liquid_radius, snow_radius,
                 tables, tau, ncb_flag, err_flag, *, inflag, iceflag, liqflag):
    """Apply native cloud optics; caller clears flags and uses native tables.

    Errors are reported in the supplied shared flag with native status codes.
    """
    if cloud.ndim != 3 or cloud.shape[2] != 140:
        raise ValueError('native RRTMG LW cloud optics requires 140 g-points')
    nc,nl,_=map(int,cloud.shape)
    if nc < 1 or nl < 1:
        raise ValueError('native RRTMG LW cloud optics requires columns and layers')
    fields={'cldfmc':cloud,'ciwpmc':ice,'clwpmc':liquid,'cswpmc':snow,'taucmc':tau}
    arrays={name:(value,np.float32,(nc,nl,140)) for name,value in fields.items()}
    arrays.update({name:(value,np.float32,(nc,nl)) for name,value in
                   (('reicmc',ice_radius),('relqmc',liquid_radius),('resnmc',snow_radius))})
    arrays.update({name:(tables[name],np.float32,(size,)) for name,size in
                   (('absice1',10),('absice2',688),('absice3',736),('absice0',2),('absliq1',928))})
    arrays.update({'ngb':(tables['ngb'],np.int32,(140,)),'ncb_flag':(ncb_flag,np.int32,(nc,)),
                   'err_flag':(err_flag,np.int32,(1,))})
    context=_context(cloud)
    for name,(value,dtype,shape) in arrays.items():
        _require(name,value,dtype,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW cloud optics arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_cldpr_f32',nc*140,context,
            {name:value for name,(value,_,_) in arrays.items()},
            dict(ncol=np.int32(nc),nl=np.int32(nl),inflag=np.int32(inflag),
                 iceflag=np.int32(iceflag),liqflag=np.int32(liqflag),absliq0=np.float32(tables['absliq0'])))


_SETCOEF_FLOAT_OUTPUTS = ('colh2o','colco2','colo3','coln2o','colco','colch4','colo2','colbrd',
    'fac00','fac01','fac10','fac11','rat_h2oco2','rat_h2oco2_1','rat_h2oo3','rat_h2oo3_1',
    'rat_h2on2o','rat_h2on2o_1','rat_h2och4','rat_h2och4_1','rat_n2oco2','rat_n2oco2_1',
    'rat_o3co2','rat_o3co2_1','selffac','selffrac','forfac','forfrac','minorfrac','scaleminor','scaleminorn2')
_SETCOEF_INT_OUTPUTS = ('jp','jt','jt1','indself','indfor','indminor')


def setcoef(inputs, tables, outputs):
    """Calculate native LW interpolation state for the wrapper's band range.

    The source caller zero-initializes output slabs before this stage.
    """
    pavel=inputs['pavel']
    if pavel.ndim != 2:
        raise ValueError('native RRTMG LW interpolation pressure requires two dimensions')
    nc,nl=map(int,pavel.shape)
    if nc < 1 or nl < 1:
        raise ValueError('native RRTMG LW interpolation requires columns and layers')
    arrays={name:(inputs[name],np.float32,shape) for name,shape in
            (('pavel',(nc,nl)),('tavel',(nc,nl)),('tz',(nc,nl+1)),('tbound_v',(nc,)),
             ('semiss',(nc,16)),('coldry',(nc,nl)),('wkl',(nc,38,nl)),('wbroad',(nc,nl)))}
    arrays.update({name:(tables['chi' if name=='chi_mls' else name],np.float32,(size,))
                   for name,size in (('totplnk',2896),('totplk16',181),('preflog',59),('tref',59),('chi_mls',413))})
    arrays.update({name:(outputs[name],np.float32,(nc,nl)) for name in _SETCOEF_FLOAT_OUTPUTS})
    arrays.update({name:(outputs[name],np.int32,(nc,nl)) for name in _SETCOEF_INT_OUTPUTS})
    arrays.update({'laytrop_v':(outputs['laytrop_v'],np.int32,(nc,)),
                   'planklay':(outputs['planklay'],np.float32,(nc,nl,16)),
                   'planklev':(outputs['planklev'],np.float32,(nc,nl+1,16)),
                   'plankbnd':(outputs['plankbnd'],np.float32,(nc,16))})
    context=_context(pavel)
    for name,(value,dtype,shape) in arrays.items():
        _require(name,value,dtype,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW interpolation arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_setcoef_f32',nc,context,
            {name:value for name,(value,_,_) in arrays.items()},
            dict(ncol=np.int32(nc),nlayers=np.int32(nl),istart=np.int32(1)))



_BAND_TABLE_SIZES = {
    1: {'absa':650,'absb':2350,'selfref':100,'forref':40,'ka_mn2':190,'kb_mn2':190,'fracrefa':10,'fracrefb':10},
    2: {'absa':780,'absb':2820,'selfref':120,'forref':48,'fracrefa':12,'fracrefb':12},
    3: {'absa':9360,'absb':18800,'selfref':160,'forref':64,'ka_mn2o':2736,'kb_mn2o':1520,'fracrefa':144,'fracrefb':80},
    4: {'absa':8190,'absb':16450,'selfref':140,'forref':56,'fracrefa':126,'fracrefb':70},
    5: {'absa':9360,'absb':18800,'selfref':160,'forref':64,'ka_mo3':2736,'ccl4':16,'fracrefa':144,'fracrefb':80},
    6: {'absa':520,'selfref':80,'forref':32,'ka_mco2':152,'cfc11adj':8,'cfc12':8,'fracrefa':8},
    7: {'absa':7020,'absb':2820,'selfref':120,'forref':48,'ka_mco2':2052,'kb_mco2':228,'fracrefa':108,'fracrefb':12},
    8: {'absa':520,'absb':1880,'selfref':80,'forref':32,'ka_mco2':152,'kb_mco2':152,'ka_mn2o':152,'kb_mn2o':152,'ka_mo3':152,'cfc12':8,'cfc22adj':8,'fracrefa':8,'fracrefb':8},
}


def band(number, laytrop, fs, isv, wx, chi, tables, gas_tau, fracs, *, oneminus):
    """Compute one implemented native gas band from initialized tables."""
    if 9 <= number <= 16:
        from .metal_rrtmg_lw_upper_v842 import band as upper_band
        return upper_band(number,laytrop,fs,isv,wx,chi,tables,gas_tau,fracs,oneminus=oneminus)
    if number not in _BAND_TABLE_SIZES:
        raise ValueError('native RRTMG LW gas band is not implemented')
    if fs.ndim != 3 or fs.shape[0] != len(native.GPU_FSLOTS):
        raise ValueError('native RRTMG LW band state has incorrect float slots')
    _,nc,nl=map(int,fs.shape)
    if nc < 1 or nl < 1:
        raise ValueError('native RRTMG LW band requires columns and layers')
    arrays={'laytrop_v':(laytrop,np.int32,(nc,)),'fs':(fs,np.float32,(len(native.GPU_FSLOTS),nc,nl)),
            'isv':(isv,np.int32,(len(native.GPU_ISLOTS),nc,nl)),'wx':(wx,np.float32,(nc,4,nl)),
            'chi_mls':(chi,np.float32,(413,)),'gas_tau':(gas_tau,np.float32,(nc,nl,140)),
            'fracs':(fracs,np.float32,(nc,nl,140))}
    arrays.update({name:(tables[name],np.float32,(size,)) for name,size in _BAND_TABLE_SIZES[number].items()})
    context=_context(fs)
    for name,(value,dtype,shape) in arrays.items():
        _require(name,value,dtype,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW band arrays belong to different Metal contexts')
    _launch(f'hexv842_mpas_rrtmg_lw_band{number}_f32',nc*nl,context,
            {name:value for name,(value,_,_) in arrays.items()},
            dict(ncol=np.int32(nc),nl=np.int32(nl),oneminus=np.float32(oneminus)))


def band1(laytrop, fs, isv, wx, chi, tables, gas_tau, fracs, *, oneminus):
    return band(1,laytrop,fs,isv,wx,chi,tables,gas_tau,fracs,oneminus=oneminus)


def band2(laytrop, fs, isv, wx, chi, tables, gas_tau, fracs, *, oneminus):
    return band(2,laytrop,fs,isv,wx,chi,tables,gas_tau,fracs,oneminus=oneminus)


def transfer_march(inputs, tables, outputs, scratch):
    """Run native longwave downward and upward transfer for each g-point.

    Caller supplies initialized tables and source-valid optical depths. Scratch
    persists the four per-layer words owned by each column and g-point.
    """
    cloud=inputs['cldfmc']
    if cloud.ndim != 3 or cloud.shape[2] != 140:
        raise ValueError('native RRTMG LW transfer requires 140 g-points')
    nc,nl,_=map(int,cloud.shape)
    if nc < 1 or nl < 1 or nl > native.MAX_RADIATION_LAYERS:
        raise ValueError('native RRTMG LW transfer depth exceeds the source domain')
    arrays={name:(inputs[name],np.float32,(nc,nl,140)) for name in ('cldfmc','odcld','efclfrac','fracs','taut')}
    arrays.update({name:(inputs[name],np.float32,(nc,16)) for name in ('secdiff','semiss','plankbnd')})
    arrays.update({'planklay':(inputs['planklay'],np.float32,(nc,nl,16)),
                   'planklev':(inputs['planklev'],np.float32,(nc,nl+1,16)),
                   'icldlyr':(inputs['icldlyr'],np.int32,(nc,nl))})
    arrays.update({name:(tables[name],np.float32,(10001,)) for name in ('tau_tbl','exp_tbl','tfn_tbl')})
    arrays['ngb']=(tables['ngb'],np.int32,(140,))
    arrays.update({name:(outputs[name],np.float32,(nc,nl,140)) for name in ('radld_p','radclrd_p','radlu_p','radclru_p')})
    arrays.update({'iclddn_p':(outputs['iclddn_p'],np.int32,(nc,nl,140)),
                   'radlu_sfc':(outputs['radlu_sfc'],np.float32,(nc,140)),
                   'radclru_sfc':(outputs['radclru_sfc'],np.float32,(nc,140)),
                   'scratch':(scratch,np.float32,(nc,140,4,nl))})
    context=_context(cloud)
    for name,(value,dtype,shape) in arrays.items():
        _require(name,value,dtype,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW transfer arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_march_f32',nc*140,context,
            {name:value for name,(value,_,_) in arrays.items()},
            dict(ncol=np.int32(nc),nl=np.int32(nl),bpade=np.float32(tables['bpade'])))


def transfer_accumulate(inputs, tables, outputs):
    """Accumulate native g-point profiles in source band order."""
    profile=inputs['radld_p']
    if profile.ndim != 3 or profile.shape[2] != 140:
        raise ValueError('native RRTMG LW accumulation requires 140 g-points')
    nc,nl,_=map(int,profile.shape)
    if nc < 1 or nl < 1:
        raise ValueError('native RRTMG LW accumulation requires columns and layers')
    arrays={name:(inputs[name],np.float32,(nc,nl,140)) for name in ('radld_p','radclrd_p','radlu_p','radclru_p')}
    arrays['iclddn_p']=(inputs['iclddn_p'],np.int32,(nc,nl,140))
    arrays.update({name:(inputs[name],np.float32,(nc,140)) for name in ('radlu_sfc','radclru_sfc')})
    arrays.update({'ngs':(tables['ngs'],np.int32,(16,)),'delwave':(tables['delwave'],np.float32,(16,))})
    arrays.update({name:(outputs[name],np.float32,(nc,nl+1)) for name in ('totuflux','totdflux','totuclfl','totdclfl')})
    context=_context(profile)
    for name,(value,dtype,shape) in arrays.items():
        _require(name,value,dtype,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW accumulation arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_accum_f32',nc*(nl+1),context,
            {name:value for name,(value,_,_) in arrays.items()},
            dict(ncol=np.int32(nc),nl=np.int32(nl),wtdiff=np.float32(native.WTDIFF)))


def transfer_final(pz, tables, outputs):
    """Convert accumulated native fluxes and compute layer heating."""
    if pz.ndim != 2:
        raise ValueError('native RRTMG LW final pressure requires two dimensions')
    nc,nlev=map(int,pz.shape);nl=nlev-1
    if nc < 1 or nl < 1:
        raise ValueError('native RRTMG LW final stage requires columns and layers')
    arrays={'pz':(pz,(nc,nlev))}
    arrays.update({name:(outputs[name],(nc,nlev)) for name in
                   ('totuflux','totdflux','totuclfl','totdclfl','fnet','fnetc','htr','htrc')})
    context=_context(pz)
    for name,(value,shape) in arrays.items():
        _require(name,value,np.float32,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW final arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_final_f32',nc,context,
            {name:value for name,(value,_) in arrays.items()},
            dict(ncol=np.int32(nc),nl=np.int32(nl),fluxfac=np.float32(tables['fluxfac']),heatfac=np.float32(tables['heatfac'])))


def pack_solver_inputs(atmosphere, levels, surface, outputs):
    """Pack native wrapper fields into the LW solver's contiguous inputs.

    The caller must zero-initialize wkl because unspecified species remain zero.
    """
    if atmosphere.ndim != 3 or atmosphere.shape[1] != 12:
        raise ValueError('native RRTMG LW solver packing requires twelve atmospheric fields')
    nc,_,nl=map(int,atmosphere.shape)
    if nc < 1 or nl < 1:
        raise ValueError('native RRTMG LW solver packing requires columns and layers')
    arrays={'a':(atmosphere,(nc,12,nl)),'lev':(levels,(nc,2,nl+1)),'surface':(surface,(nc,6))}
    arrays.update({name:(outputs[name],(nc,nl)) for name in ('pavel','tavel','reic','relq','resn')})
    arrays.update({name:(outputs[name],(nc,nl+1)) for name in ('plev','tlev')})
    arrays.update({'tsfc':(outputs['tsfc'],(nc,)),'semiss':(outputs['semiss'],(nc,16)),
                   'wkl':(outputs['wkl'],(nc,38,nl)),'wx':(outputs['wx'],(nc,4,nl))})
    context=_context(atmosphere)
    for name,(value,shape) in arrays.items():
        _require(name,value,np.float32,shape)
        if _context(value) is not context:
            raise ValueError('native RRTMG LW solver packing arrays belong to different Metal contexts')
    _launch('hexv842_mpas_rrtmg_lw_pack_f32',nc*nl,context,
            {name:value for name,(value,_) in arrays.items()},dict(ncol=np.int32(nc),nl=np.int32(nl)))



def _solver_workspace_words(nc,nl):
    words=nc*(nl*(33+6+38+4+8)+2*(nl+1)+16+2)
    words+=nc*(nl*140*(4+1+2+1+2+4+1+4)+140*2)
    return words+nc*(nl*32+(nl+1)*16+16+16+(nl+1)*8+nl+2)+1


def solve_group(atmosphere, levels, surface, tables, *, overlap, idcor, julday,
                reqc=0, reqi=0, reqs=0):
    """Run a fixed-depth native LW solver group on Metal.

    Inputs must be source-valid, including the KISS pressure order, positive
    layer pressure differences and valid coefficient indices. This bounded
    component does not provide driver admission or column grouping.
    """
    if atmosphere.ndim != 3 or atmosphere.shape[1] != 12:
        raise ValueError('native RRTMG LW solver requires twelve atmospheric fields')
    nc,_,nl=map(int,atmosphere.shape)
    if nc < 1 or nl < 4 or nl > native.MAX_RADIATION_LAYERS:
        raise ValueError('native RRTMG LW solver needs four to 128 defined layers')
    if not 0 <= overlap <= 5:
        raise ValueError('native RRTMG LW McICA overlap must be between 0 and 5')
    context=_context(atmosphere)
    # Includes every newly allocated slab, including the explicit transfer scratch.
    if _solver_workspace_words(nc,nl)*4 > context.recommended_working_set_size:
        raise MemoryError('native RRTMG LW solver workspace exceeds Metal recommended working set')
    xp=_arrays(context);z=lambda shape,dtype=np.float32:xp.zeros(shape,dtype)
    fs=z((33,nc,nl));isv=z((6,nc,nl),np.int32);wkl=z((nc,38,nl));wx=z((nc,4,nl))
    packed={'pavel':fs[0],'tavel':z((nc,nl)),'plev':z((nc,nl+1)),'tlev':z((nc,nl+1)),
            'tsfc':z((nc,)),'semiss':z((nc,16)),'wkl':wkl,'wx':wx,
            'reic':z((nc,nl)),'relq':z((nc,nl)),'resn':z((nc,nl))}
    pack_solver_inputs(atmosphere,levels,surface,packed)
    coldry=fs[1];wbrodl=z((nc,nl));pwvcm=z((nc,))
    inatm(packed['plev'],wkl,wx,coldry,wbrodl,pwvcm,grav=tables['grav'],avogad=tables['avogad'])
    cloud={name:z((nc,nl,140)) for name in ('cloud','liquid','ice','snow')}
    mcica(atmosphere,surface,cloud['cloud'],cloud['liquid'],cloud['ice'],cloud['snow'],overlap=overlap,idcor=idcor,julday=julday)
    tau=z((nc,nl,140));ncb=z((nc,),np.int32);err=z((1,),np.int32)
    inflag=5 if reqs else 4 if reqi else 3 if reqc else 2
    iceflag=5 if reqs else 4 if reqi else 3
    cloud_optics(cloud['cloud'],cloud['ice'],cloud['liquid'],cloud['snow'],packed['reic'],packed['relq'],packed['resn'],tables,tau,ncb,err,inflag=inflag,iceflag=iceflag,liqflag=1)
    code=int(err.to_numpy()[0])
    if code:
        raise ValueError(f'native RRTMG LW cloud optics returned source error {code}')
    coef={name:fs[native.GPU_FSLOTS.index(name)] for name in _SETCOEF_FLOAT_OUTPUTS}
    coef.update({name:isv[native.GPU_ISLOTS.index(name)] for name in _SETCOEF_INT_OUTPUTS})
    coef.update({'laytrop_v':z((nc,),np.int32),'planklay':z((nc,nl,16)),
                 'planklev':z((nc,nl+1,16)),'plankbnd':z((nc,16))})
    setcoef({'pavel':fs[0],'tavel':packed['tavel'],'tz':packed['tlev'],
             'tbound_v':packed['tsfc'],'semiss':packed['semiss'],'coldry':coldry,'wkl':wkl,'wbroad':wbrodl},tables,coef)
    gas=z((nc,nl,140));fracs=z((nc,nl,140))
    for number in range(1,17):
        btable={name:tables['reduced'][f'kg{number:02d}/{name}'] for name in legacy.GPU_BAND_TABS[number]}
        band(number,coef['laytrop_v'],fs,isv,wx,tables['chi'],btable,gas,fracs,oneminus=tables['oneminus'])
    total=z((nc,nl,140));optical_depth(gas,z((nc,nl,16)),tables['ngb'],total)
    secdiff=z((nc,16));diffusivity(pwvcm,tables['a0'],tables['a1'],tables['a2'],secdiff)
    od=z((nc,nl,140));ef=z((nc,nl,140));icldlyr=z((nc,nl),np.int32)
    cloud_transfer(cloud['cloud'],tau,secdiff,tables['ngb'],od,ef,icldlyr)
    march={name:z((nc,nl,140)) for name in ('radld_p','radclrd_p','radlu_p','radclru_p')}
    march.update({'iclddn_p':z((nc,nl,140),np.int32),'radlu_sfc':z((nc,140)),'radclru_sfc':z((nc,140))})
    transfer_march({'cldfmc':cloud['cloud'],'odcld':od,'efclfrac':ef,'icldlyr':icldlyr,
                    'secdiff':secdiff,'semiss':packed['semiss'],'planklay':coef['planklay'],
                    'planklev':coef['planklev'],'plankbnd':coef['plankbnd'],'fracs':fracs,'taut':total},
                   tables,march,z((nc,140,4,nl)))
    result={name:z((nc,nl+1)) for name in ('totuflux','totdflux','totuclfl','totdclfl','fnet','fnetc','htr','htrc')}
    transfer_accumulate(march,tables,result);transfer_final(packed['plev'],tables,result)
    return result


def run_group(model, interfaces, surface, pin, ozone, tables, *, icloud, overlap,
              idcor, julday, reqc=0, reqi=0, reqs=0, o3input=0, presence=15):
    """Run one native wrapper group whose columns share a layer count.

    Counts are copied to the host as launch metadata; weather fields remain
    resident. The caller supplies source-valid pressure and thermodynamic data.
    """
    if model.ndim != 3 or model.shape[1] != 12:
        raise ValueError('native RRTMG LW wrapper requires twelve model fields')
    nc,_,nm=map(int,model.shape)
    if nc < 1 or nm < 1:
        raise ValueError('native RRTMG LW wrapper requires columns and layers')
    context=_context(model);xp=_arrays(context)
    counts=xp.empty((nc,),np.int32);layer_counts(interfaces,counts)
    host_counts=counts.to_numpy();nl=int(host_counts[0])
    if np.any(host_counts != nl):
        raise ValueError('native RRTMG LW wrapper group must share its layer count')
    if nl < 4 or nl > native.MAX_RADIATION_LAYERS:
        raise ValueError('native RRTMG LW wrapper needs four to 128 defined layers')
    group_words=nc*(12*nl+2*(nl+1)+nl+4*nm+13)
    if (_solver_workspace_words(nc,nl)+group_words)*4 > context.recommended_working_set_size:
        raise MemoryError('native RRTMG LW wrapper workspace exceeds Metal recommended working set')
    atmosphere=xp.empty((nc,12,nl),np.float32);levels=xp.empty((nc,2,nl+1),np.float32)
    prepare(model,interfaces,surface,pin,ozone,atmosphere,levels,icloud=icloud,
            reqc=reqc,reqi=reqi,reqs=reqs,o3input=o3input,presence=presence)
    if overlap:
        status=xp.empty((nc,),np.int32)
        _launch('hexv842_mpas_rrtmg_lw_pressure_f32',nc,context,dict(a=atmosphere,status=status),
                dict(nc=np.int32(nc),nl=np.int32(nl)))
        if np.any(status.to_numpy()):
            raise ValueError('native RRTMG LW KISS seed requires pressure ordered from bottom layers')
    result=solve_group(atmosphere,levels,surface,tables,overlap=overlap,idcor=idcor,
                       julday=julday,reqc=reqc,reqi=reqi,reqs=reqs)
    heating=xp.empty((nc,nl),np.float32)
    _launch('hexv842_mpas_rrtmg_lw_heat_f32',nc*nl,context,dict(htr=result['htr'],heating=heating),
            dict(nc=np.int32(nc),nl=np.int32(nl)))
    tendency=xp.empty((nc,nm),np.float32);radii=xp.empty((nc,3,nm),np.float32);flux=xp.empty((nc,11),np.float32)
    outputs(model,atmosphere,result['totuflux'],result['totdflux'],result['totuclfl'],
            result['totdclfl'],heating,tendency,radii,flux,icloud=icloud)
    return {'tendency':tendency,'radii':radii,'flux':flux,'counts':counts}
