"""Private strict classic LW transfer; consumes supplied gas and aerosol optics."""
import math
import numpy as np
from woof.metal.array import MetalArray, MetalArrayModule


def transfer(inputs, *, bpade, wtdiff, fluxfac, heatfac):
    first=inputs['cldfmc']
    if not isinstance(first,MetalArray):raise TypeError('cldfmc must be MetalArray')
    context=first.buffer.context
    if any(not isinstance(v,MetalArray) or v.buffer.context is not context for v in inputs.values()):
        raise TypeError('transfer inputs must be MetalArrays on one context')
    if first.ndim!=3:raise ValueError('cldfmc must be rank three')
    ncol,ng,nl=first.shape
    if ncol<1 or ng!=140 or not 0<nl<=128 or ncol*ng*nl>np.iinfo(np.int32).max:
        raise ValueError('transfer extent is outside the supported index range')
    f=np.float32; i=np.int32
    expected={
        **{k:((ncol,),f) for k in ('pwvcm',)},
        **{k:((16,),f) for k in ('a0','a1','a2','delwave')},
        'cldfmc':((ncol,ng,nl),f),'taucmc':((ncol,ng,nl),f),
        'taug':((ncol,nl,ng),f),'taua':((ncol,nl,16),f),
        'ngb':((ng,),i),'ngs':((16,),i),
        'semiss':((ncol,16),f),'plankbnd':((ncol,16),f),
        'planklay':((ncol,nl,16),f),'planklev':((ncol,nl+1,16),f),
        'fracs':((ncol,nl,ng),f),'pz':((ncol,nl+1),f),
        **{k:((10001,),f) for k in ('tau_tbl','exp_tbl','tfn_tbl')}}
    if set(inputs)!=set(expected):raise ValueError('transfer input inventory differs')
    for name,(shape,dtype) in expected.items():
        value=inputs[name]
        if value.shape!=shape or value.dtype!=np.dtype(dtype) or not value.flags.c_contiguous:
            raise ValueError(f'transfer input {name} shape, dtype, or layout differs')
    outputs=(
        ('taut',(ncol,nl,ng),f),('secdiff',(ncol,16),f),('odcld',(ncol,ng,nl),f),('efclfrac',(ncol,ng,nl),f),('icldlyr',(ncol,nl),i),
        *[(k,(ncol,ng,nl),f) for k in ('radld_p','radclrd_p','radlu_p','radclru_p')],('iclddn_p',(ncol,ng,nl),i),
        ('radlu_sfc',(ncol,ng),f),('radclru_sfc',(ncol,ng),f),
        *[(k,(ncol,nl+1),f) for k in ('totuflux','totdflux','totuclfl','totdclfl','fnet','fnetc','htr','htrc')])
    sizes=[math.prod(shape)*np.dtype(dtype).itemsize for _,shape,dtype in outputs]
    if any(size>int(context.max_buffer_length) for size in sizes):
        raise MemoryError('one transfer output exceeds the device buffer limit')
    unique={id(v.buffer):v.buffer for v in inputs.values()}
    if sum(b.size for b in unique.values())+sum(sizes)>int(context.recommended_working_set_size):
        raise MemoryError('transfer inputs and outputs exceed the recommended working set')
    xp=MetalArrayModule(context);v=dict(inputs)
    for name,shape,dtype in outputs:v[name]=xp.empty(shape,dtype=dtype)
    base={'ncol':ncol,'nl':nl}
    stages=(
     ('taut','taug taua ngb taut',dict(base,ngpt=ng,nbnd=16),ncol*nl*ng),
     ('secdiff','pwvcm a0 a1 a2 secdiff',{'ncol':ncol},ncol),
     ('prol','cldfmc taucmc secdiff ngb odcld efclfrac icldlyr',base,ncol*nl),
     ('march','cldfmc odcld efclfrac icldlyr secdiff semiss planklay planklev plankbnd fracs taut tau_tbl exp_tbl tfn_tbl ngb radld_p radclrd_p radlu_p radclru_p iclddn_p radlu_sfc radclru_sfc',dict(base,bpade=f(bpade)),ncol*ng),
     ('accum','radld_p radclrd_p radlu_p radclru_p iclddn_p radlu_sfc radclru_sfc ngs delwave totuflux totdflux totuclfl totdclfl',dict(base,wtdiff=f(wtdiff)),ncol*(nl+1)),
     ('final','pz totuflux totdflux totuclfl totdclfl fnet fnetc htr htrc',dict(base,fluxfac=f(fluxfac),heatfac=f(heatfac)),ncol))
    for suffix,names,params,grid in stages:
        context.submit_kernel('wrf_exact_rrtmg_lw_'+suffix,{k:v[k] for k in names.split()},params,exact_policy=True,grid_size=grid,threads_per_group=128)
    return v
