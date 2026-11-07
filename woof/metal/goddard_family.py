"""Typed resident Goddard column execution and device-budgeted scratch."""
from __future__ import annotations
import numpy as np
from .array import MetalArray,MetalArrayModule
from .goddard_workspace import workspace_words


def _submit(name,resources,parameters,count):
    values=tuple(resources.values())
    if not values or any(not isinstance(a,MetalArray) for a in values):
        raise TypeError('Goddard resources must be MetalArrays')
    context=values[0].buffer.context
    for field,a in resources.items():
        dtype=np.dtype(np.int32 if field=='flags' else np.uint64 if field=='workspace' else np.float32)
        if a.dtype!=dtype or not a.flags.c_contiguous:
            raise TypeError(f'Goddard {field} requires contiguous {dtype}')
        if a.buffer.context is not context:
            raise ValueError('Goddard resources must share one context')
    if count<=0 or count>0x7fffffff:
        raise ValueError('Goddard launch size must fit positive int32')
    from woof.wrf_exact import ENABLED as WRF_EXACT
    entry = 'wrf_exact_' + name if WRF_EXACT else name
    return context.submit_kernel(entry,{k:a.view((a.size,)) for k,a in resources.items()},
                                 parameters,exact_policy=WRF_EXACT,grid_size=count,
                                 threads_per_group=32 if name in ('goddard_bands_columns','goddard_tiers_columns') else 128)


def column_chunk(context,levels,columns,requested):
    from woof.wrf_exact import ENABLED as WRF_EXACT
    from woof.core.radiation_composition import ColumnTableRadiation
    if WRF_EXACT and int(levels)>ColumnTableRadiation.metal_strict_max_levels:
        raise NotImplementedError(f'strict Metal Goddard above {ColumnTableRadiation.metal_strict_max_levels} levels requires further split band execution')
    words=workspace_words(levels)+int(WRF_EXACT and levels>49)
    # Live raw/prepared/cloud/flux/result words plus the binary64 arena.
    bytes_per_column=4*(109*levels+50)+8*words
    budget=min(int(context.recommended_working_set_size)//16,
               int(context.max_buffer_length))
    limit=min(budget//bytes_per_column,0x7fffffff//max(words,63*levels))
    if limit<1:
        raise MemoryError('Goddard cannot fit one column in the device working-set budget')
    return min(int(columns),int(requested),limit,32 if WRF_EXACT else int(columns))


def interface_temperature(t,z,fnm,fnp,*,levels,columns):
    xp=MetalArrayModule(t.buffer.context)
    tw=xp.empty((levels+1,columns),np.float32)
    expected={'t':levels*columns,'z':(levels+1)*columns,'fnm':levels,'fnp':levels}
    inputs={'t':t,'z':z,'fnm':fnm,'fnp':fnp}
    if levels<2 or any(a.size!=expected[n] for n,a in inputs.items()):
        raise ValueError('Goddard interface-temperature inputs have inconsistent dimensions')
    _submit('goddard_interface_temperature',dict(inputs,tw=tw),
            dict(nz=levels,columns=columns),tw.size)
    return tw


def solve_chunk(fields,table,*,levels,full_columns,first,columns,julday,solcon,
                direct_radii,heat_lw,heat_sw,surface):
    xp=MetalArrayModule(heat_lw.buffer.context)
    np_=int(levels);nc=int(columns)
    if not (2<=np_ and 0<=first<full_columns and 0<nc<=full_columns-first):
        raise ValueError('Goddard chunk dimensions are invalid')
    if max((20*np_+7)*nc,(2*np_+15)*full_columns)>0x7fffffff:
        raise ValueError('Goddard packed indexing must fit int32')
    raw=xp.empty(((20*np_+7)*nc,),np.float32)
    common=dict(np=np_,full_columns=int(full_columns),first=int(first),columns=nc)
    _submit('goddard_pack_grid',dict(fields,raw=raw),
            dict(common,direct_radii=int(bool(direct_radii))),raw.size)
    band=xp.empty(((5*np_+15)*nc,),np.float32)
    flags=xp.empty((3*nc,),np.int32)
    dz=xp.empty((np_*nc,),np.float32)
    q=xp.empty((6*np_*nc,),np.float32);re=xp.empty(q.shape,np.float32)
    _submit('goddard_prepare_columns',dict(raw=raw,band=band,flags=flags,dz=dz,q=q,re=re),
            dict(np=np_,columns=nc,julday=int(julday)),nc)
    from woof.wrf_exact import ENABLED as WRF_EXACT
    if WRF_EXACT:
        prepared=flags.to_numpy().reshape(3,nc)
        if np.any((prepared[2]!=0) & ((prepared[0]<2) | (prepared[1]<2)
                                      | (prepared[0]>=prepared[1]))):
            raise ValueError('strict Goddard shortwave requires ordered cloud indices below the top layer')
    optics=xp.empty((63*np_*nc,),np.float32)
    _submit('goddard_cloud_columns',dict(dz=dz,q=q,re=re,lut=table,output=optics),
            dict(np=np_,columns=nc),np_*nc)
    flux=xp.empty(((6*(np_+1)+4)*nc,),np.float32)
    tiered=WRF_EXACT and np_>49
    words=workspace_words(np_)+int(tiered)
    workspace=xp.empty((words*nc,),np.uint64)
    resources=dict(input=band,flags=flags,optics=optics,output=flux,workspace=workspace)
    params=dict(np=np_,columns=nc,workspace_words=words)
    if tiered:
        for mode,band_number in ((0,0),*((1,i) for i in range(1,11)),(2,0)):
            _submit('goddard_tiers_columns',resources,dict(params,tier_mode=mode,tier_band=band_number),nc)
    else:
        _submit('goddard_bands_columns',resources,params,nc)
    output=xp.empty(((2*np_+15)*nc,),np.float32)
    _submit('goddard_publish_columns',dict(band=band,flux=flux,raw=raw,output=output),
            dict(np=np_,columns=nc,solcon=float(np.float32(solcon))),nc)
    _submit('goddard_scatter_grid',dict(input=output,heat_lw=heat_lw,heat_sw=heat_sw,surface=surface),common,output.size)
