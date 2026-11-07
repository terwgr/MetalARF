"""Private strict preparation stages for classic regional RRTMG longwave."""
import numpy as np
from woof.core import rrtmg_lw as lw
from woof.metal.array import MetalArrayModule


def prepare(context, prepared, coefficients):
    """Submit validated diagnostic- or supplied-radius cloud preparation.

    Returns the CUDA-layout 38-species workspace, zero-initialized FS/IS,
    Planck interpolation, cloud optics and source status flags. The caller
    must check err_flag before consuming the result.
    """
    xp = MetalArrayModule(context)
    n, nl = int(prepared['ncol']), int(prepared['nlay'])
    if n < 1 or nl < 1 or nl > 128:
        raise ValueError('invalid classic LW column dimensions')
    max_words = max(n * 140 * nl, n * 38 * nl, n * 16 * (nl + 1), 2896)
    budget = int(context.recommended_working_set_size)
    if n * 140 > np.iinfo(np.int32).max or max_words > np.iinfo(np.int32).max:
        raise ValueError('strict LW prep dimensions exceed int32 limits')
    if budget <= 0 or max_words * 4 > budget:
        raise ValueError('strict LW prep buffer exceeds the device working set')
    flags = tuple(int(prepared[key]) for key in ('inflglw', 'iceflglw', 'liqflglw'))
    if flags not in ((2, 3, 1), (5, 5, 1)):
        raise NotImplementedError('strict LW prep supports only cloud flags 2/3/1 and 5/5/1')
    inflag, iceflag, liqflag = flags
    def host(value, shape):
        value = np.ascontiguousarray(value, dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('invalid strict LW preparation input')
        return value
    def upload(value):
        return xp.asarray(np.ascontiguousarray(value))
    def coeff(key, *, dtype=np.float32):
        return upload(np.asarray(coefficients[key], dtype=dtype).ravel(order='F'))
    plev_h=host(prepared['plev'],(n,nl+1)); play_h=host(prepared['play'],(n,nl))
    if (np.any(plev_h[:,:-1]<=0) or np.any(plev_h[:,-1]<0)
            or np.any(plev_h[:,:-1]<=plev_h[:,1:]) or np.any(play_h<=0)):
        raise ValueError('strict LW prep requires positive decreasing pressures')
    tlay_h=host(prepared['tlay'],(n,nl)); tlev_h=host(prepared['tlev'],(n,nl+1))
    if np.any(tlay_h<=0): raise ValueError('strict LW prep requires positive layer temperatures')
    tsfc_h=host(prepared['tsfc'],(n,)); emis_h=host(prepared['emis'],(n,16))
    wkl_h=np.zeros((n,38,nl),np.float32)
    for slot,key in ((0,'h2ovmr'),(1,'co2vmr'),(2,'o3vmr'),(3,'n2ovmr'),(5,'ch4vmr'),(6,'o2vmr')):
        wkl_h[:,slot,:]=host(prepared[key],(n,nl))
    if np.any(wkl_h<0): raise ValueError('strict LW prep requires nonnegative gas concentrations')
    wx_h=np.zeros((n,4,nl),np.float32)
    for slot,key in enumerate(('ccl4vmr','cfc11vmr','cfc12vmr','cfc22vmr')):
        wx_h[:,slot,:]=host(prepared[key],(n,nl))
    if np.any(wx_h<0): raise ValueError('strict LW prep requires nonnegative cross sections')
    clouds={}
    for key in ('cldfmcl','ciwpmcl','clwpmcl','cswpmcl','taucmcl'):
        a=np.asarray(prepared[key],np.float32)
        if a.shape==(140,n,nl): a=a.transpose(1,0,2)
        elif a.shape==(n,nl,140): a=a.transpose(0,2,1)
        clouds[key]=host(a,(n,140,nl))
        if np.any(clouds[key]<0): raise ValueError('strict LW prep requires nonnegative cloud input')
    reic_h=host(prepared['reicmcl'],(n,nl)); relq_h=host(prepared['relqmcl'],(n,nl)); resn_h=host(prepared['resnmcl'],(n,nl))
    active=(clouds['cldfmcl']>=np.float32(1.e-20)) & ((clouds['ciwpmcl']+clouds['clwpmcl']+clouds['cswpmcl']>=np.float32(1.e-20)) | (clouds['taucmcl']>=np.float32(1.e-20)))
    ice=np.any(active & ((clouds['ciwpmcl']+clouds['cswpmcl'])!=0),axis=1)
    liquid=np.any(active & (clouds['clwpmcl']!=0),axis=1)
    invalid_snow = iceflag == 5 and np.any(
        np.any(active & (clouds['cswpmcl'] > 0), axis=1)
        & ((resn_h < 5) | (resn_h > 140)))
    if (np.any(ice & ((reic_h < 5) | (reic_h > 140)))
            or np.any(liquid & ((relq_h < 2.5) | (relq_h > 60)))
            or invalid_snow):
        raise ValueError('strict LW prep rejects invalid active cloud radii')
    bands=np.asarray(coefficients['wvn/ngb'],np.int32)
    if bands.shape!=(140,) or np.any((bands<1)|(bands>16)):
        raise ValueError('strict LW prep requires valid g-point bands')
    fs=xp.zeros((len(lw.GPU_FSLOTS),n,nl),np.float32)
    isv=xp.zeros((len(lw.GPU_ISLOTS),n,nl),np.int32)
    play=upload(play_h); fs[0][:]=play
    coldry=fs[1]; plev=upload(plev_h); wkl=upload(wkl_h); wx=upload(wx_h)
    wbrodl=xp.empty((n,nl),np.float32); pwvcm=xp.empty((n,),np.float32)
    submit=context.submit_kernel
    submit('wrf_exact_rrtmg_lw_inatm',dict(plev=plev,wkl=wkl,wx=wx,coldry=coldry,wbrodl=wbrodl,pwvcm=pwvcm),
           dict(ncol=n,nl=nl,grav=np.float32(coefficients['con/grav']),avogad=np.float32(coefficients['con/avogad'])),
           exact_policy=True,grid_size=n,threads_per_group=128)
    params=dict(ncol=n,nlayers=nl,istart=1)
    arrays=dict(pavel=play,tavel=upload(tlay_h),tz=upload(tlev_h),tbound_v=upload(tsfc_h),semiss=upload(emis_h),
                coldry=coldry,wkl=wkl,wbroad=wbrodl,totplnk=coeff('wvn/totplnk'),totplk16=coeff('wvn/totplk16'),
                preflog=coeff('ref/preflog'),tref=coeff('ref/tref'),chi_mls=coeff('ref/chi_mls'))
    for i,key in enumerate(lw.GPU_FSLOTS):
        if key not in ('pavel','coldry'): arrays[key]=fs[i]
    for i,key in enumerate(lw.GPU_ISLOTS): arrays[key]=isv[i]
    arrays.update(laytrop_v=xp.zeros((n,),np.int32),planklay=xp.zeros((n,nl,16),np.float32),
                  planklev=xp.zeros((n,nl+1,16),np.float32),plankbnd=xp.zeros((n,16),np.float32))
    submit('wrf_exact_rrtmg_lw_setcoef',arrays,params,exact_policy=True,grid_size=n,threads_per_group=128)
    cloud={name:upload(value) for name,value in clouds.items()}
    cloudargs=dict(cldfmc=cloud['cldfmcl'],ciwpmc=cloud['ciwpmcl'],clwpmc=cloud['clwpmcl'],cswpmc=cloud['cswpmcl'],
                   reicmc=upload(reic_h),relqmc=upload(relq_h),resnmc=upload(resn_h),ngb=upload(bands),taucmc=cloud['taucmcl'],
                   ncb_flag=xp.zeros((n,),np.int32),err_flag=xp.zeros((1,),np.int32))
    for name in ('absice1','absice2','absice3','absice0','absliq1'):cloudargs[name]=coeff('cld/'+name)
    submit('wrf_exact_rrtmg_lw_cldprmc',cloudargs,
           dict(ncol=n,nl=nl,inflag=inflag,iceflag=iceflag,liqflag=liqflag,
                absliq0=np.float32(coefficients['cld/absliq0'])),
           exact_policy=True,grid_size=n*140,threads_per_group=128)
    return dict(plev=plev,semiss=arrays['semiss'],wkl=wkl,wx=wx,fs=fs,isv=isv,coldry=coldry,wbrodl=wbrodl,pwvcm=pwvcm,
                laytrop=arrays['laytrop_v'],planklay=arrays['planklay'],planklev=arrays['planklev'],
                plankbnd=arrays['plankbnd'],cloud=cloud,ngb=cloudargs['ngb'],
                ncb_flag=cloudargs['ncb_flag'],err_flag=cloudargs['err_flag'])
