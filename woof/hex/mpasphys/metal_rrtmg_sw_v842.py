"""Native MPAS v8.4.2 shortwave coefficient initialization on Metal.

This is an initialization component, not a complete radiation slot or provider.
The caller supplies decoded raw coefficient buffers in Fortran linear order.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from woof.core.rrtmg_sw import _CUDA_TABLE_ARRAYS
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch
from .rrtmg_sw_v842 import NGC, NGB, STATIC_LAYOUT

_RAW_NAMES={'absa':'kao','absb':'kbo','selfref':'selfrefo','forref':'forrefo',
            'sfluxref':'sfluxrefo','absch4':'absch4o','rayl':'raylo',
            'abso3a':'abso3ao','abso3b':'abso3bo','rayla':'raylao',
            'raylb':'raylbo','absh2o':'absh2oo','absco2':'absco2o'}

@dataclass(frozen=True)
class MetalMpasShortwaveTables:
    packed: object
    weights: object
    ngb: object
    offsets: dict
    static_start: int
    exp_start: int


def initialize_raw_coefficients(context, raw, shapes):
    """Reduce the native 16-g-point inputs and create immutable solver tables.

    ``raw`` maps native keys to contiguous device float32 vectors; ``shapes``
    gives their native Fortran dimensions. Authenticity and source decoding are
    the caller's responsibility. No atmospheric fields are copied to host.
    """
    plans=[]; offsets={}; offset=0
    for define,band,var in _CUDA_TABLE_ARRAYS:
        key=f'kg{band}/{_RAW_NAMES[var]}'
        shape=tuple(shapes[key]); value=raw[key]
        first=var in ('sfluxref','rayla')
        if not shape or shape[0 if first else -1]!=16:
            raise ValueError(f'{key}: native g-point dimension must be 16')
        size=int(np.prod(shape)); _require(key,value,np.float32,(size,))
        if _context(value) is not context: raise ValueError(f'{key} belongs to another context')
        rows=size//16; count=rows*NGC[band-16]
        offsets[define]=offset
        plans.append((value,band-16,rows,first,var=='sfluxref',offset,count))
        offset+=count
    static_start=offset
    for name,count in STATIC_LAYOUT:
        offsets[name]=offset;offset+=count
    exp_start=offset;offset+=10001
    if sum(count for _,count in STATIC_LAYOUT)!=6936:
        raise ValueError('native SW static table layout changed')
    if 4*(offset+224+len(NGB))>context.recommended_working_set_size:
        raise MemoryError('MPAS shortwave initialization exceeds Metal working set')
    xp=_arrays(context);packed=xp.empty((offset,),np.float32);weights=xp.empty((224,),np.float32)
    _launch('hexv842_sw_weights_f32',14,context,dict(rwgt=weights),dict(reserved=np.int32(0)))
    for value,band,rows,first,unweighted,start,count in plans:
        _launch('hexv842_sw_reduce_f32',count,context,
            dict(raw=value,rwgt=weights,reduced=packed[start:start+count]),
            {k:np.int32(v) for k,v in dict(band=band,rows=rows,first_axis=first,unweighted=unweighted,count=count).items()})
    _launch('hexv842_sw_static_f32',6936,context,dict(out=packed[static_start:exp_start]),dict(reserved=np.int32(0)))
    _launch('hexv842_sw_exp_f32',10001,context,dict(out=packed[exp_start:]),dict(reserved=np.int32(0)))
    return MetalMpasShortwaveTables(packed,weights,xp.asarray(np.asarray(NGB,np.int32)),offsets,static_start,exp_start)


def convert_outputs(*, cosine, exner, up, down, upc, downc, direct, heat, scalars, tendency):
    """Convert solver fluxes to native SW diagnostics and theta tendency.

    Profiles are column-major, with ``nz+2`` flux interfaces and ``nz+1``
    heating entries. Night columns retain the caller's existing values except
    for cosine, exactly as the selected source does.
    """
    nc,nz=exner.shape
    if nc<1 or nz<1: raise ValueError('MPAS SW output conversion requires nonempty columns and levels')
    context=_context(exner)
    values=dict(cosine=cosine,exner=exner,up=up,down=down,upc=upc,downc=downc,direct=direct,heat=heat,scalars=scalars,tendency=tendency)
    shapes={**{name:(nc,nz+2) for name in ('up','down','upc','downc','direct')},
            'cosine':(nc,), 'exner':(nc,nz), 'heat':(nc,nz+1), 'scalars':(14,nc), 'tendency':(nc,nz)}
    for name,value in values.items():
        _require(name,value,np.float32,shapes[name])
        if _context(value) is not context: raise ValueError(f'{name} belongs to another context')
    _launch('hexv842_sw_outputs_f32',nc,context,values,dict(nc=np.int32(nc),nz=np.int32(nz)))


def radiation_controls(context, *, seconds, interval, julian, degrad, dpd):
    """Compute native SW time, interval, declination and solar constant."""
    values=np.asarray([seconds,interval,julian,degrad,dpd],np.float32)
    if not np.all(np.isfinite(values)) or interval<=0:
        raise ValueError('MPAS SW controls need finite inputs and positive interval')
    out=_arrays(context).empty((4,),np.float32)
    _launch('hexv842_sw_controls_f32',1,context,dict(out=out),dict(zip(('seconds','interval','julian','degrad','dpd'),values)))
    return out


def prepare_sounding(*, fields, surface, pressure_interfaces, temperature_interfaces,
                     pin, ozone, controls, gmt, degrad, icloud, o3input, has_reqc, has_reqi, has_reqs):
    """Build native column-major SW solver carriers from device-resident input.

    The inputs use the native wrapper's packed FIELD_NAMES and SURFACE_NAMES
    orders. Ozone interpolation requires at least two levels; the upstream
    single-level equality case reads outside the native allocation.
    """
    if len(fields.shape)!=3 or fields.shape[0]!=12:
        raise ValueError('MPAS SW needs 12 packed sounding fields')
    _,nc,nz=fields.shape;nozn=int(pin.size)
    if nc<1 or nz<1 or nozn<2 or o3input not in (0,2):
        raise ValueError('MPAS SW sounding requires nonempty columns, at least two ozone levels and o3input 0 or 2')
    if not np.isfinite(gmt) or not np.isfinite(degrad):
        raise ValueError('MPAS SW sounding time and angle controls must be finite')
    context=_context(fields)
    values={'fields':fields,'surf':surface,'pw':pressure_interfaces,'tw':temperature_interfaces,'pin':pin,'oz':ozone,'controls':controls}
    shapes={'fields':(12,nc,nz),'surf':(7,nc),'pw':(nc,nz+1),'tw':(nc,nz+1),'pin':(nozn,),'oz':(nc,nozn),'controls':(4,)}
    for name,value in values.items():
        _require(name,value,np.float32,shapes[name])
        if _context(value) is not context: raise ValueError(f'{name} belongs to another context')
    words=15*nc*(nz+1)+2*nc*(nz+2)+nc*(nz+1)+nc
    if 4*words>context.recommended_working_set_size:
        raise MemoryError('MPAS SW sounding workspace exceeds Metal working set')
    xp=_arrays(context)
    output=dict(layers=xp.empty((15,nc,nz+1),np.float32),interfaces=xp.empty((2,nc,nz+2),np.float32),hgt=xp.empty((nc,nz+1),np.float32),coszr=xp.empty((nc,),np.float32))
    params={k:np.int32(v) for k,v in dict(nc=nc,nz=nz,icloud=icloud,o3input=o3input,reqc=has_reqc,reqi=has_reqi,reqs=has_reqs,nozn=nozn).items()}
    params.update(gmt=np.float32(gmt),degrad=np.float32(degrad))
    _launch('hexv842_sw_prep_f32',nc,context,{**values,**output},params)
    return output


def initialize_tables(context, resources=None):
    """Read pinned native SW records on the host and initialize them on Metal."""
    from .metal_radiation_tables_v842 import load_raw_tables
    from .rrtmg_sw_v842 import SCALAR_BITS, LAYREF
    from woof.core.rrtmg_sw import _CUDA_SCALARS
    decoded=load_raw_tables('sw',resources)
    for _,band,var in _CUDA_SCALARS:
        value=decoded[f'rrsw_kg{band}'][var]
        if value.shape!=() or int(value.view(np.uint32))!=SCALAR_BITS[f'kg{band}/{var}']:
            raise ValueError(f'kg{band}/{var}: native SW scalar differs from the pinned table')
    for band,expected in LAYREF.items():
        value=decoded[f'rrsw_kg{band}']['layreffr']
        if value.shape!=() or int(value)!=expected:
            raise ValueError(f'kg{band}/layreffr: native SW reference layer differs from the pinned table')
    xp=_arrays(context); raw={};shapes={}
    for _,band,var in _CUDA_TABLE_ARRAYS:
        key=f'kg{band}/{_RAW_NAMES[var]}'
        value=decoded[f'rrsw_kg{band}'][_RAW_NAMES[var]]
        shapes[key]=value.shape
        raw[key]=xp.asarray(np.ascontiguousarray(value.ravel(order='F')))
    return initialize_raw_coefficients(context,raw,shapes)


def mcica_subcolumns(*, layers, height, latitude, overlap, decorrelation, julday):
    """Generate native KISS McICA subcolumns with the wrapper's fixed seed mode.

    The four pressure-derived KISS seeds require at least four solver layers.
    The eight slabs include the source's initially clear optical properties.
    """
    if len(layers.shape)!=3 or layers.shape[0]!=15:
        raise ValueError('MPAS SW McICA needs 15 sounding fields')
    _,nc,nl=layers.shape
    if nc<1 or nl<4 or overlap not in range(6) or decorrelation not in (0,1):
        raise ValueError('MPAS SW McICA needs at least four levels and native overlap/decorrelation selectors')
    context=_context(layers)
    for name,value,shape in [('layers',layers,(15,nc,nl)),('height',height,(nc,nl)),('latitude',latitude,(nc,))]:
        _require(name,value,np.float32,shape)
        if _context(value) is not context: raise ValueError(f'{name} belongs to another context')
    words=8*nc*nl*112
    if words*4>context.recommended_working_set_size:
        raise MemoryError('MPAS SW McICA workspace exceeds Metal working set')
    xp=_arrays(context);out=xp.zeros((8,nc,nl,112),np.float32);out[5].fill(1.)
    if overlap:
        arrays=dict(play=layers[0],cf=layers[8],iw=layers[9],lw=layers[10],sw=layers[11],hgt=height,latitude=latitude,uniform=layers[0],out=out)
        params={k:np.int32(v) for k,v in dict(nc=nc,nl=nl,icld=overlap,idcor=decorrelation,julday=julday).items()}
        _launch('hexv842_sw_mcica_f32',nc,context,arrays,params)
    return out


def run_shortwave(*, tables, fields, surface, pressure_interfaces, temperature_interfaces,
                  pin, ozone, controls, gmt, degrad, icloud, o3input, has_reqc, has_reqi,
                  has_reqs, overlap, decorrelation, julday, aerosol_inputs=None,
                  equation_of_time=None):
    """Run native SW preparation, day-column McICA, solver and output stages.

    Weather inputs and outputs stay on Metal. The host reads the day count
    and device validation statuses to allocate the batch and reject failures.
    Thompson-aerosol inputs are optional; provider registration is separate.
    """
    from .metal_rrtmg_sw_solver_v842 import solve_sunlit
    _,nc,nz=fields.shape; nl=nz+1;context=_context(fields);xp=_arrays(context)
    if aerosol_inputs is not None and (aerosol_inputs['pressure'].shape!=(nc,nz) or _context(aerosol_inputs['pressure']) is not context):
        raise ValueError('MPAS SW aerosol inputs must match the sounding extent and context')
    aerosol=thompson_aerosols(**aerosol_inputs) if aerosol_inputs is not None else None
    prep=prepare_sounding(fields=fields,surface=surface,pressure_interfaces=pressure_interfaces,
        temperature_interfaces=temperature_interfaces,pin=pin,ozone=ozone,controls=controls,
        gmt=gmt,degrad=degrad,icloud=icloud,o3input=o3input,has_reqc=has_reqc,has_reqi=has_reqi,has_reqs=has_reqs)
    if equation_of_time is not None:
        day, day_offset, second_sine = np.asarray(equation_of_time, dtype=np.float32)
        if not np.isfinite([day, day_offset, second_sine]).all():
            raise ValueError('RRFS SW equation of time requires finite inputs')
        context.submit_kernel('hexrrfs_sw_solar',
            dict(surface=surface, controls=controls, coszr=prep['coszr']),
            dict(nc=nc, julian=day, gmt=np.float32(gmt), degrad=np.float32(degrad),
                 day_offset=day_offset, second_sine=second_sine),
            exact_policy=False, grid_size=nc)
    idx=xp.empty((nc,),np.int32);count=xp.zeros((1,),np.int32)
    _launch('hexv842_sw_day_select_i32',1,context,dict(cosine=prep['coszr'],indices=idx,count=count),dict(nc=np.int32(nc)))
    nd=int(count.to_numpy()[0])
    scalars=xp.zeros((14,nc),np.float32);scalars[0][...]=prep['coszr']
    tendency=xp.zeros((nc,nz),np.float32);profiles=xp.zeros((4,nc,nz+2),np.float32)
    if not nd:
        result=dict(scalars=scalars,tendency=tendency,profiles=profiles)
        if aerosol is not None: result.update(taod5503d=aerosol['taod5503d'],taod5502d=aerosol['taod5502d'])
        return result
    if nl<4:raise ValueError('MPAS SW McICA requires four layers for pressure-derived seeds')
    if 4*nd*(55*nl*112+200*nl+200)>context.recommended_working_set_size:
        raise MemoryError('MPAS SW batch exceeds Metal working set')
    layer=xp.empty((15,nd,nl),np.float32);interfaces=xp.empty((2,nd,nl+1),np.float32)
    height=xp.empty((nd,nl),np.float32);exner=xp.empty((nd,nz),np.float32)
    cosine=xp.empty((nd,),np.float32);albedo=xp.empty((nd,),np.float32);latitude=xp.empty((nd,),np.float32)
    params={k:np.int32(v) for k,v in dict(nc=nc,nd=nd,nz=nz).items()}
    _launch('hexv842_sw_day_gather_f32',nd*(nl+1),context,dict(indices=idx,layers_in=prep['layers'],interfaces_in=prep['interfaces'],height_in=prep['hgt'],fields_in=fields,cosine_in=prep['coszr'],surface_in=surface,layers_out=layer,interfaces_out=interfaces,height_out=height,exner_out=exner,cosine_out=cosine,albedo_out=albedo,latitude_out=latitude),params)
    if overlap:
        invalid=xp.zeros((nd,),np.int32)
        _launch('hexv842_sw_kiss_guard_i32',nd,context,dict(play=layer[0],invalid=invalid),dict(nc=np.int32(nd),nl=np.int32(nl)))
        if np.any(invalid.to_numpy()):
            raise ValueError('module_ra_rrtmg_sw.F:1742-1744 KISS requires bottom-to-top layer pressure')
    cloud=mcica_subcolumns(layers=layer,height=height,latitude=latitude,overlap=overlap,decorrelation=decorrelation,julday=julday)
    cloudflag=5 if has_reqs else 4 if has_reqi else 3 if has_reqc else 2
    iceflag=5 if has_reqs else 4 if has_reqi else 3
    day_aerosol=None
    if aerosol is not None:
        tau=xp.empty((nd,14,nl),np.float32);ssa=xp.empty((nd,14,nl),np.float32);asy=xp.empty((nd,14,nl),np.float32)
        _launch('hexv842_sw_aerosol_gather_f32',nd*14*nl,context,dict(indices=idx,tau_in=aerosol['tauaer'],ssa_in=aerosol['ssaaer'],asy_in=aerosol['asyaer'],tau_out=tau,ssa_out=ssa,asy_out=asy),dict(nd=np.int32(nd),nl=np.int32(nl)))
        day_aerosol=(tau,asy,ssa)
    flux=solve_sunlit(tables=tables,layers=layer,interfaces=interfaces,cosine=cosine,albedo=albedo,controls=controls,mcica=cloud,cloudflag=cloudflag,iceflag=iceflag,aerosol=day_aerosol)
    day_scalars=xp.zeros((14,nd),np.float32);day_tendency=xp.zeros((nd,nz),np.float32)
    convert_outputs(cosine=cosine,exner=exner,up=flux['swuflx'],down=flux['swdflx'],upc=flux['swuflxc'],downc=flux['swdflxc'],direct=flux['swdkdir'],heat=flux['swhr'],scalars=day_scalars,tendency=day_tendency)
    _launch('hexv842_sw_day_scatter_f32',nd*(nz+2),context,dict(indices=idx,scalars_in=day_scalars,tendency_in=day_tendency,up=flux['swuflx'],upc=flux['swuflxc'],down=flux['swdflx'],downc=flux['swdflxc'],scalars_out=scalars,tendency_out=tendency,profiles_out=profiles),params)
    result=dict(scalars=scalars,tendency=tendency,profiles=profiles)
    if aerosol is not None: result.update(taod5503d=aerosol['taod5503d'],taod5502d=aerosol['taod5502d'])
    return result


def thompson_aerosols(*, pressure, temperature, qv, dz, nwfa, nifa, xland):
    """Native live Thompson aerosol driver: land/ocean classes and 3-D optics."""
    if len(pressure.shape)!=2:raise ValueError('MPAS SW aerosols need column-major profiles')
    nc,nz=pressure.shape;nl=nz+1
    if nc<1 or nz<1:raise ValueError('MPAS SW aerosols need nonempty profiles')
    context=_context(pressure);xp=_arrays(context)
    values=dict(pressure=pressure,temperature=temperature,qv=qv,dz=dz,nwfa=nwfa,nifa=nifa,xland=xland)
    for name,value in values.items():
        _require(name,value,np.float32,(nc,) if name=='xland' else (nc,nz))
        if _context(value) is not context:raise ValueError(f'{name} belongs to another context')
    if 4*nc*(42*nl+nz+3)>context.recommended_working_set_size:
        raise MemoryError('MPAS SW aerosol workspace exceeds Metal working set')
    aod3=xp.empty((nc,nz),np.float32);aod2=xp.empty((nc,),np.float32)
    typ=xp.empty((nc,),np.int32);invalid=xp.zeros((nc,),np.int32)
    _launch('hexv842_sw_gt_aod_f32',nc,context,{**values,'aod3':aod3,'aod2':aod2,'type':typ,'invalid':invalid},dict(nc=np.int32(nc),nz=np.int32(nz)))
    if np.any(invalid.to_numpy()):
        raise ValueError('native Thompson aerosol type is undefined when xland is not 1 or 2')
    tau=xp.empty((nc,14,nl),np.float32);ssa=xp.empty((nc,14,nl),np.float32);asy=xp.empty((nc,14,nl),np.float32)
    _launch('hexv842_sw_aerosol_f32',nc,context,dict(pressure=pressure,temperature=temperature,qv=qv,aod3=aod3,type=typ,tau=tau,ssa=ssa,asy=asy),dict(nc=np.int32(nc),nz=np.int32(nz),nl=np.int32(nl)))
    return dict(tauaer=tau,ssaaer=ssa,asyaer=asy,taod5503d=aod3,taod5502d=aod2,taer_type=typ)
