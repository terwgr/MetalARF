"""Selected MPAS v8.4.2 surface update services for resident Metal pools."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _context, _i, _require
from .metal_tendencies_v842 import _launch
from .surface_update_v842 import MonthlyCalendar


class MetalMonthlySurface:
    """MPAS monthly vegetation/albedo update and saved February length."""
    def __init__(self):
        self.calendar=MonthlyCalendar()

    def update(self, *, current_date, config_sfc_albedo, fields, n_cells_solve):
        land=fields['landmask']; ncells=land.size
        if not 0 < n_cells_solve <= ncells:
            raise ValueError('surface solve extent must fit resident pools')
        _require('landmask',land,np.int32,(ncells,))
        for name in ('greenfrac','albedo12m'):
            _require(name,fields[name],np.float32,(12,ncells))
        for name in ('vegfra','shdmin','shdmax','sfc_albbck'):
            _require(name,fields[name],np.float32,(ncells,))
        m1,m2,w1,w2,span=self.calendar.weights(current_date)
        _launch('hexv842_mpas_monthly_surface_f32',n_cells_solve,_context(land),
                dict(green=fields['greenfrac'],alb=fields['albedo12m'],land=land,
                     veg=fields['vegfra'],lo=fields['shdmin'],hi=fields['shdmax'],bg=fields['sfc_albbck']),
                dict(nsolve=_i(n_cells_solve),ncells=_i(ncells),m1=_i(m1),m2=_i(m2),
                     w1=_i(w1),w2=_i(w2),span=_i(span),use_alb=_i(config_sfc_albedo)))

    def restart_state(self):
        return {'february':self.calendar.february}

    def restore_restart_state(self,state):
        if state['february'] not in (28,29):
            raise ValueError('invalid native saved February length')
        self.calendar=MonthlyCalendar(february=state['february'])


def accumulate(*, rate, total, count, dt, bucket, ringing, its, ite, n_cells_solve):
    """Accumulate one native radiation or precipitation bucket over owned cells."""
    n=rate.size
    if not 1 <= its <= ite <= n_cells_solve <= n:
        raise ValueError('its/ite must be MPAS inclusive owned-cell bounds')
    _require('rate',rate,np.float32,(n,))
    _require('total',total,np.float32,(n,))
    _require('count',count,np.int32,(n,))
    _launch('hexv842_mpas_bucket_f32',n,_context(rate),
            dict(rate=rate,total=total,count=count),
            dict(n=_i(n),begin=_i(its-1),end=_i(ite),dt=np.float32(dt),
                 bucket=np.float32(bucket),ringing=_i(ringing)))


def update_radiation_diagnostics(*, configs, diag, its, ite, n_cells_solve, dt_dyn, ringing):
    from .surface_update_v842 import RADIATION_FIELDS
    for name in RADIATION_FIELDS:
        accumulate(rate=diag[name],total=diag['ac'+name],count=diag['i_ac'+name],
                   dt=dt_dyn,bucket=configs['config_bucket_radt'],ringing=ringing,
                   its=its,ite=ite,n_cells_solve=n_cells_solve)


def update_precipitation_buckets(*, configs, diag, its, ite, n_cells_solve, dt_dyn, ringing,
                                 convection=True, microphysics=True):
    for enabled,suffix,rate,dt in ((convection,'c','cuprec',dt_dyn),
                                   (microphysics,'nc','rainncv',1.0)):
        if enabled:
            accumulate(rate=diag[rate],total=diag['rain'+suffix],count=diag['i_rain'+suffix],
                       dt=dt,bucket=configs['config_bucket_rain'+suffix],ringing=ringing,
                       its=its,ite=ite,n_cells_solve=n_cells_solve)


class MetalSurfaceService:
    """Implemented native driver callbacks; missing services remain unbound."""
    OPERATIONS=('physics_update_surface','update_radiation_diagnostics','update_convection_step2')

    def __init__(self, session):
        self.session=session
        self.monthly=MetalMonthlySurface()
        self.executions={name:0 for name in self.OPERATIONS}

    def callbacks(self):
        return {name:self.invoke for name in self.OPERATIONS}

    def invoke(self, call):
        operation=call.slot
        if operation not in self.OPERATIONS:
            raise ValueError(f'unsupported MPAS Metal surface operation {operation}')
        views=self.session.pools
        solve=self.session.layout.dimensions['nCellsSolve']
        if operation=='physics_update_surface':
            names=('landmask','albedo12m','greenfrac','vegfra','shdmin','shdmax','sfc_albbck')
            fields={n:views.field('sfc_input',n) for n in names}
            self.monthly.update(current_date=self.session.arguments.calendar['current_date'],
                                config_sfc_albedo=call.options['config_sfc_albedo'],
                                fields=fields,n_cells_solve=solve)
        elif operation=='update_radiation_diagnostics':
            from .surface_update_v842 import RADIATION_FIELDS
            names=(*RADIATION_FIELDS, *('ac'+n for n in RADIATION_FIELDS),
                   *('i_ac'+n for n in RADIATION_FIELDS))
            fields={n:views.field('diag_physics',n) for n in names}
            update_radiation_diagnostics(configs=call.options,diag=fields,its=self.session.arguments.its,
                ite=self.session.arguments.ite,n_cells_solve=solve,dt_dyn=call.dt,ringing=call.flags.radiation_bucket)
        else:
            fields={n:views.field('diag_physics',n) for n in ('cuprec','rainc','i_rainc')}
            update_precipitation_buckets(configs=call.options,diag=fields,its=self.session.arguments.its,
                ite=self.session.arguments.ite,n_cells_solve=solve,dt_dyn=call.dt,
                ringing=call.flags.rain_bucket,convection=True,microphysics=False)
        self.executions[operation]+=1

    def restart_state(self):
        return {'february':self.monthly.calendar.february,'executions':dict(self.executions)}

    def restore_restart_state(self, state):
        if set(state['executions']) != set(self.OPERATIONS) or any(type(v) is not int or v<0 for v in state['executions'].values()):
            raise ValueError('invalid MPAS surface operation counters')
        self.monthly.restore_restart_state({'february':state['february']})
        self.executions=dict(state['executions'])


class MetalSurfaceServiceRRFS(MetalSurfaceService):
    """RRFS monthly surface update; other surface services are unchanged."""

    def invoke(self, call):
        if call.slot != 'physics_update_surface':
            return super().invoke(call)
        from .rrfs.driver_rrfs import package_selection
        from .rrfs.surface_rrfs import DUST_MONTHLY

        views=self.session.pools
        options=call.options
        solve=self.session.layout.dimensions['nCellsSolve']
        land=views.field('sfc_input','landmask')
        ncells=land.size
        if not 0 < solve <= ncells:
            raise ValueError('surface solve extent must fit resident pools')
        _require('landmask',land,np.int32,(ncells,))
        m1,m2,w1,w2,span=self.monthly.calendar.weights(self.session.arguments.calendar['current_date'])

        def update(pool, monthly, out_pool, out, mode=0, lo=None, hi=None):
            source=views.field(pool,monthly)
            target=views.field(out_pool,out)
            _require(monthly,source,np.float32,(12,ncells))
            _require(out,target,np.float32,(ncells,))
            low=views.field('sfc_input',lo) if lo else target
            high=views.field('sfc_input',hi) if hi else target
            _require(lo or out,low,np.float32,(ncells,))
            _require(hi or out,high,np.float32,(ncells,))
            _launch('hexrrfs_mpas_monthly_surface_f32',solve,_context(land),
                    dict(monthly=source,land=land,out=target,lo=low,hi=high),
                    dict(nsolve=_i(solve),ncells=_i(ncells),m1=_i(m1),m2=_i(m2),
                         w1=_i(w1),w2=_i(w2),span=_i(span),mode=_i(mode)))

        if options['config_sfc_albedo']:
            update('sfc_input','albedo12m','sfc_input','sfc_albbck',mode=1)
        if str(options.get('config_dust_scheme','off')).strip() != 'off':
            pairs=tuple(pair for pair in DUST_MONTHLY
                        if all(('diag_physics',name) in self.session.native._fields for name in pair))
            if not pairs:
                raise KeyError('dust is on but the route Registry declares no FENGSHA monthly drag input')
            for monthly,out in pairs:
                update('diag_physics',monthly,'diag_physics',out)
        if options.get('config_gvf_update',True):
            update('sfc_input','greenfrac','sfc_input','vegfra',mode=2,lo='shdmin',hi='shdmax')
        if package_selection(self.session.config).get('lsm_ruc_in',False):
            update('sfc_input','lai12m','diag_physics','lai')
        self.executions['physics_update_surface']+=1
