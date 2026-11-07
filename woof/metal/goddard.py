"""Goddard adapter on resident Metal model columns."""
from __future__ import annotations
import numpy as np
from .array import MetalArray,MetalArrayModule
from .goddard_family import column_chunk,interface_temperature,solve_chunk


def run(adapter,*,atmosphere,fields,state,cfg):
    from woof.wrf_exact import ENABLED as WRF_EXACT
    if WRF_EXACT and (int(getattr(cfg, 'mp_physics', 0)) != 0
                      or int(getattr(cfg, 'bl_pbl_physics', 0)) != 0
                      or int(getattr(cfg, 'icloud', 1)) not in (0, 1)):
        raise NotImplementedError(
            'strict Metal Goddard radiation supports mp=0, PBL=0 and icloud=0 or 1')
    if WRF_EXACT and atmosphere['pressure'].shape[0]>adapter.metal_strict_max_levels:
        raise NotImplementedError(f'strict Metal Goddard above {adapter.metal_strict_max_levels} levels requires further split band execution')
    def host(value):
        if isinstance(value,MetalArray):return value.to_numpy()
        if hasattr(value,'get'):return value.get()
        return value
    if WRF_EXACT:
        inputs = {name: atmosphere[name] for name in
                  ('temperature','pressure','p_interface','z_interface','dz','rho','exner')}
        inputs.update({name: fields[name] for name in ('tsk','emiss','albedo','glw')})
        inputs.update({name: getattr(state,name,None) for name in
                       ('qv','qc','qr','qi','qs','qg','qh')})
        inputs.update(fnm=state.fnm,fnp=state.fnp,latitude=adapter.latitude_deg,
                      longitude=adapter.longitude_deg,elapsed=state.elapsed_seconds,
                      radt=cfg.radt if cfg.radt>0 else cfg.radt_minutes)
        for name,value in inputs.items():
            if value is not None and not np.isfinite(host(value)).all():
                raise ValueError(f'strict Metal Goddard requires finite {name}')
    from woof.core.radiation_tables import radiation_table_host_words
    from woof.core.radiation_composition import radiation_radius_provider_fields
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags,radconst
    from woof.core.rrtmgp import cal_cldfra1,cal_cldfra2
    from woof.core.mynn_radiation import merge_mynn_bl_clouds,mynn_bl_cloud_active,wrf_itimestep
    from woof.core.swint import calendar_scalars
    pressure=atmosphere['pressure'];context=pressure.buffer.context
    xp=MetalArrayModule(context)
    nz,ny,nx=pressure.shape;ncol=ny*nx
    if nz<2:raise ValueError('Goddard interface-temperature extrapolation needs two layers')
    def grid(value):
        value=xp.asarray(value,dtype=np.float32)
        return value.view((value.size//ncol,ncol))
    zeros=xp.zeros((nz,ncol),np.float32)
    moist={name:grid(getattr(state,name)) if getattr(state,name,None) is not None else zeros
           for name in ('qv','qc','qr','qi','qs','qg','qh')}
    t,p=grid(atmosphere['temperature']),grid(pressure)
    fqi,fqs=((True,True) if int(cfg.mp_physics)==7 else legacy_cloud_fraction_flags(int(cfg.mp_physics)))
    mp=int(cfg.mp_physics)
    cf=(cal_cldfra2(moist['qc'],moist['qi'],f_qc=mp!=0,f_qi=fqi) if cfg.icloud==2 else
        cal_cldfra1(moist['qv'],moist['qc'],moist['qi'],moist['qs'],t,p,
                   f_qc=True,f_qi=fqi,f_qs=fqs,mp_physics=mp) if cfg.icloud==1 else zeros)
    active=cfg.icloud==1 and mynn_bl_cloud_active(getattr(cfg,'bl_pbl_physics',0),getattr(cfg,'icloud_bl',0))
    if active:moist['qc']=moist['qc'].copy();moist['qi']=moist['qi'].copy()
    moist['qc'],moist['qi'],cf=merge_mynn_bl_clouds(moist['qc'],moist['qi'],cf,
        qc_bl=grid(fields['qc_bl']) if active else None,
        qi_bl=grid(fields['qi_bl']) if active else None,
        cldfra_bl=grid(fields['cldfra_bl']) if active else None,
        bl_pbl_physics=getattr(cfg,'bl_pbl_physics',0),icloud_bl=getattr(cfg,'icloud_bl',0) if active else 0,
        itimestep=wrf_itimestep(state.elapsed_seconds,cfg.dt) if active else 1,
        bl_mynn_version=getattr(cfg,'bl_mynn_version','wrf_461'),temperature=t if active else None)
    values,direct=radiation_radius_provider_fields(state,int(cfg.mp_physics),getattr(cfg,'use_mp_re',1))
    radii=[zeros if value is None else grid(value) for value in values]
    julian,xtime,gmt=calendar_scalars(adapter.start_time,state.elapsed_seconds)
    declin,solcon=radconst(julian)
    minutes=np.float32(cfg.radt if cfg.radt>0 else cfg.radt_minutes)
    latitude=np.asarray(host(adapter.latitude_deg),np.float32)
    longitude=np.asarray(host(adapter.longitude_deg),np.float32)
    from woof.core.solar import solar_geometry_device
    lat_device=xp.asarray(latitude).view((ncol,))
    lon_device=xp.asarray(longitude).view((ncol,))
    _ds,cosine,_ha=solar_geometry_device(julian,xtime+minutes*np.float32(.5),gmt,
                                       lat_device,lon_device,declination=declin)
    tw=interface_temperature(t,grid(atmosphere['z_interface']),
        xp.asarray(state.fnm,dtype=np.float32),xp.asarray(state.fnp,dtype=np.float32),
        levels=nz,columns=ncol)
    cached=getattr(adapter,'_metal_cloud_table',None)
    if cached is None or cached.buffer.context is not context:
        cached=xp.asarray(radiation_table_host_words('goddard-cloud'))
        adapter._metal_cloud_table=cached
    heat_lw=xp.empty((nz,ny,nx),np.float32);heat_sw=xp.empty(heat_lw.shape,np.float32)
    surface=xp.empty((15,ny,nx),np.float32)
    inputs=dict(pw=grid(atmosphere['p_interface']),temperature=t,pressure=p,cldfra=cf,
                dz=grid(atmosphere['dz']),rho=grid(atmosphere['rho']),
                latitude=lat_device,cosine=cosine,tw=tw,
                tsk=fields['tsk'],emiss=fields['emiss'],albedo=fields['albedo'],
                exner=atmosphere['exner'],**moist)
    inputs.update(zip(('re_ice','re_cloud','re_rain','re_snow','re_graupel','re_hail'),radii))
    chunk=column_chunk(context,nz,ncol,adapter.column_chunk)
    for first in range(0,ncol,chunk):
        solve_chunk(inputs,cached,levels=nz,full_columns=ncol,first=first,
                    columns=min(chunk,ncol-first),julday=int(julian)+1,solcon=solcon,
                    direct_radii=direct,heat_lw=heat_lw,heat_sw=heat_sw,surface=surface)
    if WRF_EXACT:
        for name,value in (('longwave heating',heat_lw),('shortwave heating',heat_sw),
                           ('surface flux',surface),('cosine',cosine),('cloud fraction',cf)):
            if not np.isfinite(host(value)).all():
                raise ValueError(f'strict Metal Goddard produced nonfinite {name}')
    adapter.update_count+=1
    return adapter._result(xp, heat_lw=heat_lw, heat_sw=heat_sw, surface=surface,
                           cosine=cosine.view((ny,nx)), cldfra=cf.view((nz,ny,nx)), fields=fields)
