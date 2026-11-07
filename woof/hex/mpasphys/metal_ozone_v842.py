"""Selected native ozone initialization and monthly update on Metal.

The native exact-final-latitude overrun is explicitly refused.
"""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _context, _i, _require
from .metal_tendencies_v842 import _launch
from .radshared_v842 import OzoneCalendarState, _monthly_date_weights


def initialize_monthly(*, latitude, lat_ozone, table, monthly):
    """Initialize full-cell monthly ozone without the native undefined endpoint."""
    ncells=latitude.size
    if table.ndim != 3 or table.shape[:2] != (12,64):
        raise ValueError('ozone table must have shape [12,64,nOznLevels]')
    nlev=table.shape[2]
    _require('latitude',latitude,np.float32,(ncells,))
    _require('lat_ozone',lat_ozone,np.float32,(64,))
    _require('table',table,np.float32,(12,64,nlev))
    _require('monthly',monthly,np.float32,(12,nlev,ncells))
    # Native mpas_atmphys_o3climatology.F:164-167 reads beyond the last
    # latitude at equality. Refuse before writing any output; no replacement
    # value can be certified at that source-undefined boundary.
    degrees=latitude.to_numpy()/np.float32(np.float32(3.1415926)/np.float32(180.))
    if np.any(degrees == lat_ozone.to_numpy()[-1]):
        raise ValueError('MPAS ozone initialization is undefined at its final latitude')
    _launch('hexv842_mpas_ozone_latitude_f32',12*nlev*ncells,_context(latitude),
            dict(lat=latitude,lat_ozone=lat_ozone,table=table,monthly=monthly),
            dict(ncells=_i(ncells),nlev=_i(nlev)))


class MetalOzoneMonthlyUpdate:
    def __init__(self):
        self.calendar=OzoneCalendarState()

    def update(self, *, current_date, monthly, out, n_cells_solve):
        """Apply the native date update to solve columns of the ozone diagnostic."""
        if monthly.ndim != 3 or monthly.shape[0] != 12:
            raise ValueError('monthly ozone must have shape [12,level,cell]')
        _,nlev,ncells=monthly.shape
        if not 0 < n_cells_solve <= ncells:
            raise ValueError('ozone solve extent must fit resident fields')
        _require('monthly',monthly,np.float32,(12,nlev,ncells))
        _require('out',out,np.float32,(nlev,ncells))
        month1,month2,d1,d2=_monthly_date_weights(current_date,self.calendar)
        _launch('hexv842_mpas_ozone_update_f32',nlev*ncells,_context(monthly),
                dict(monthly=monthly,out=out),
                dict(ncells=_i(ncells),nlev=_i(nlev),nsolve=_i(n_cells_solve),
                     month1=_i(month1),month2=_i(month2),d1=_i(d1),d2=_i(d2)))

    def restart_state(self):
        return {'february_days':self.calendar.february_days}

    def restore_restart_state(self,state):
        if state['february_days'] not in (28,29):
            raise ValueError('invalid saved MPAS ozone February length')
        self.calendar=OzoneCalendarState(state['february_days'])


def interpolate_cam_time(*, julian, monthly, out, n_cells_solve):
    """CAM cyclic time interpolation into solve columns of the ozone pool."""
    if monthly.ndim != 3 or monthly.shape[0] != 12:
        raise ValueError('monthly ozone must have shape [12,level,cell]')
    _,nlev,ncells=monthly.shape
    if not 0 < n_cells_solve <= ncells or not np.isfinite(julian) or not 0 <= julian < 366:
        raise ValueError('invalid ozone solve extent or CAM Julian day')
    _require('monthly',monthly,np.float32,(12,nlev,ncells))
    _require('out',out,np.float32,(nlev,ncells))
    _launch('hexv842_mpas_ozone_time_f32',nlev*ncells,_context(monthly),
            dict(monthly=monthly,out=out),
            dict(ncells=_i(ncells),nlev=_i(nlev),nsolve=_i(n_cells_solve),julian=np.float32(julian)))


def interpolate_vertical(*, pmid, pin, ozone, out, n_cells_solve, pressure_in_pa=False):
    """RRTMG ozone interpolation with the source's cross-column search bound."""
    from woof.metal.array import MetalArrayModule
    if pmid.ndim != 2 or ozone.ndim != 2 or pin.ndim != 1:
        raise ValueError('ozone and pressure require level-by-cell profiles')
    nz,ncells=pmid.shape; nlev=pin.size
    if nlev < 2 or not 0 < n_cells_solve <= ncells:
        raise ValueError('invalid MPAS ozone levels or solve extent')
    _require('pmid',pmid,np.float32,(nz,ncells))
    _require('pin',pin,np.float32,(nlev,))
    _require('ozone',ozone,np.float32,(nlev,ncells))
    _require('out',out,np.float32,(nz,ncells))
    context=_context(pmid); xp=MetalArrayModule(context)
    upper=xp.zeros((n_cells_solve,),dtype=np.int32)
    start=xp.zeros((1,),dtype=np.int32)
    for k in range(nz-1,-1,-1):
        _launch('hexv842_mpas_ozone_min_upper_f32',1,context,
                dict(upper=upper,start=start),dict(nsolve=_i(n_cells_solve),nlev=_i(nlev)))
        _launch('hexv842_mpas_ozone_vertical_f32',n_cells_solve,context,
                dict(pmid=pmid,pin=pin,ozone=ozone,start=start,upper=upper,out=out),
                dict(ncells=_i(ncells),nsolve=_i(n_cells_solve),nlev=_i(nlev),
                     k=_i(k),pascal=_i(bool(pressure_in_pa))))


class MetalOzoneService:
    """Native-pool ozone initialization and CAM driver update.

    The caller provides authenticated decoded tables and invokes initialize at
    the native physics initialization boundary, before any ozone pool views.
    """
    def __init__(self, session, tables):
        self.session, self.tables = session, tables
        self.initialized = False

    def initialize(self):
        p=self.session.pools; native=self.session.native
        d=self.session.layout.dimensions; nc=d['nCells']; nl=d['nOznLevels']
        for pool,name in (('mesh','latCell'),('atm_input','ozmixm'),('atm_input','pin')):
            row=p.rows[pool,name]
            if (pool,row['pool_name'],1) in p.buffers:
                raise RuntimeError('ozone initialization requires uncached native pool fields')
        xp=p.arrays; tables={name:xp.asarray(self.tables[name]) for name in ('lat_ozone','ozmixin','pin')}
        _require('pin',tables['pin'],np.float32,(nl,))
        latitude=xp.asarray(np.ascontiguousarray(native.get_field('mesh','latCell')[:nc]))
        monthly=xp.zeros((12,nl,nc),dtype=np.float32)
        initialize_monthly(latitude=latitude,lat_ozone=tables['lat_ozone'],table=tables['ozmixin'],monthly=monthly)
        native_monthly=native.get_field('atm_input','ozmixm')
        native_monthly[:,:,:nc]=monthly.to_numpy()
        native.put_field('atm_input','ozmixm',native_monthly,level=1,nlevels=1)
        native_pin=native.get_field('atm_input','pin')
        native_pin[...] = tables['pin'].to_numpy()
        native.put_field('atm_input','pin',native_pin,level=1,nlevels=1)
        self.initialized=True

    def update(self, call):
        if not self.initialized:
            raise RuntimeError('native ozone initialization must precede the CAM update')
        p=self.session.pools
        interpolate_cam_time(julian=self.session.arguments.calendar['julian'],
            monthly=p.field('atm_input','ozmixm'),out=p.field('diag_physics','o3clim'),
            n_cells_solve=self.session.layout.dimensions['nCellsSolve'])

    def vertical(self, *, pressure, out, pressure_in_pa=False):
        if not self.initialized:
            raise RuntimeError('native ozone initialization must precede vertical interpolation')
        p=self.session.pools
        interpolate_vertical(pmid=pressure,pin=p.field('atm_input','pin'),
            ozone=p.field('diag_physics','o3clim'),out=out,
            n_cells_solve=self.session.layout.dimensions['nCellsSolve'],pressure_in_pa=pressure_in_pa)

    def restart_state(self):
        return {'initialized': self.initialized}

    def restore_restart_state(self, state):
        if type(state.get('initialized')) is not bool:
            raise ValueError('invalid Metal ozone initialization state')
        self.initialized=state['initialized']
