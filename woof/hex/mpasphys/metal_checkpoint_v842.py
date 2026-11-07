"""Metal session state in the existing native MPAS checkpoint carrier.

This same-mesh adapter preserves the native carrier schema and delegates its
sealed I/O, schedule and identity checks to CheckpointRoute/MovingRoute. It is
not a public forecast admission path or a native MPAS restart file.
"""
from __future__ import annotations
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import json
from pathlib import Path
import time
import numpy as np

from .checkpoint_v842 import CLOCK_START_WORDS, CheckpointRoute
from .moving_v842 import (SCHEMA, TIME_FORMAT, STATE_FIELDS, SAVED_FIELDS,
                          CATEGORY_FIELDS, _key, _write_sealed)


_METAL_SCHEMA='mpas-metal-checkpoint/v1'
# Values assigned by FrameworkInitialization.physics_run_init/physics_init
# belong to the new segment, including its timestep-derived substep counts.
_RUN_INIT={'dt_cu','dt_dyn','dt_microp','dt_pbl','dt_radtlw','dt_radtsw','n_cu','n_microp',
           'ims','ime','jms','jme','kms','kme','ids','ide','jds','jde','kds','kde',
           'its','ite','jts','jte','kts','kte','num_months','num_soils','num_oznlevels',
           'l_conv','l_radtlw','l_radtsw','l_diags','l_camlw','l_acrain','l_acradt',
           'xice_threshold','sf_surface_physics','has_reqc','has_reqi','has_reqs',
           'f_qc','f_qr','f_qi','f_qs','f_qg','f_nc','f_ni','f_nifa','f_nwfa','f_qoz','f_nbca'}


def _host_words(value):
    if isinstance(value,np.ndarray):return value.copy()
    if hasattr(value,'to_numpy'):return value.to_numpy()
    raise TypeError('Metal checkpoint state requires resident Metal or NumPy arrays')


class MetalCheckpointRoute(CheckpointRoute):
    """Same-mesh Metal checkpoint with native carrier compatibility."""

    @classmethod
    def from_route(cls, route):
        if not isinstance(route,CheckpointRoute):raise TypeError('Metal checkpoints require a resolved CheckpointRoute')
        return cls(**{item.name:getattr(route,item.name) for item in fields(CheckpointRoute)})

    def export(self, path, *, session, dycore, mesh, vertical, names, layout, step, dt):
        """Seal a boundary whose session owns every active scheme's persistent state."""
        if self.checkpoint_identity is None or self.checkpoint_physics is None or self.checkpoint_dt is None:
            raise ValueError('native checkpoint identity/schedule must bind before export')
        if self.origin_time is None:raise ValueError('native checkpoint origin is not bound')
        if getattr(dycore,'stochastic',None) is not None or getattr(dycore,'chemistry',None) is not None:
            raise ValueError('Metal checkpoint does not yet carry stochastic or chemistry state')
        if Fraction(str(dt))!=Fraction(self.checkpoint_dt):raise ValueError('checkpoint timestep differs from its bound schedule')
        elapsed=Fraction(str(step))*Fraction(str(dt));clock=session.native.clock_state
        if clock.start_time.calendar!='gregorian':raise ValueError('native carrier requires the Gregorian calendar')
        if elapsed!=clock.elapsed:raise ValueError('checkpoint step and native segment clock differ')
        seconds=Fraction(str(self.offset_seconds))+elapsed
        if seconds.denominator!=1:raise ValueError('native checkpoint requires an integral-second boundary')
        valid=(datetime.strptime(self.origin_time,TIME_FORMAT)+timedelta(seconds=int(seconds))).strftime(TIME_FORMAT)
        if clock.current_time.stamp()!=valid:raise ValueError('native checkpoint clock and origin differ')
        started=datetime.now(timezone.utc).isoformat();began=time.perf_counter()
        snapshot=session.transaction.checkpoint()
        if not snapshot['driver']['initialized']:raise ValueError('native checkpoint requires initialized physics')
        atmosphere=dycore.atmosphere
        state={n:_host_words(getattr(atmosphere.state,n)) for n in STATE_FIELDS}
        state.update({n:_host_words(getattr(atmosphere.saved,n)) for n in SAVED_FIELDS})
        rows=self.carried_rows(layout);all_pools=snapshot['pools']
        pools={_key(row,level):all_pools[f"{row['pool']}.{row['pool_name']}.{level}"]
               for row in rows for level in range(1,row['nlevels']+1)}
        services=snapshot['services'];window=getattr(session,'precipitation_window',None)
        variables={name:(value.copy(order='F'),lower)
                   for name,(value,lower) in session.native._variables.items()
                   if value.ndim<2 and name not in _RUN_INIT}
        if (window is None)!=('precipitation_window' not in services):
            raise ValueError('Metal precipitation window is not bound to transaction state')
        physics=dict(suite=getattr(session.config,'suite',None),pools=pools,
            fields={f"{r['pool']}.{r['name']}":dict(dimensions=r['dimensions'],nlevels=r['nlevels'],kind=r['kind']) for r in rows},
            categories={n:self._pool_cells(session.native,layout,'sfc_input',n) for n in CATEGORY_FIELDS},
            isice=int(np.asarray(session.native.get_field('sfc_input',self._pool_name(layout,'sfc_input','isice_lu'))).reshape(-1)[0]),
            window=None if window is None else services['precipitation_window'],
            surface_service=services.get('surface'),manager=snapshot['driver']['manager'],
            manager_elapsed=str(Fraction(str(self.physics_clock_offset(dt)[1])) + elapsed),
            metal=dict(schema=_METAL_SCHEMA,pools=all_pools,services=services,
                       variables=variables,
                       driver_fields=snapshot['driver']['fields'],initialized=snapshot['driver']['initialized']))
        continuation = self.continuation_state(session)
        expected_clock = set(CLOCK_START_WORDS) if session.config.moist_physics else set()
        if set(continuation['continuation']['clock_start']) != expected_clock:
            raise ValueError('Metal checkpoint cannot carry the physics start hour')
        physics.update(continuation)
        payload=dict(schema=SCHEMA,route='mpas-suite',time_seconds=float(seconds),origin_time=self.origin_time,
                     valid_time=valid,state=state,mesh=mesh,vertical=vertical,scalar_names=list(names),
                     physics=physics,stochastic=None,chemistry=None)
        receipt=_write_sealed(Path(path),payload,dict(physics_fields=len(rows),
            physics_words=int(sum(np.asarray(v).size for v in pools.values())),state={k:list(v.shape) for k,v in state.items()}))
        receipt['checkpoint']=dict(schema='mpas-native-checkpoint/v1',dt=self.checkpoint_dt,
                                   identity=self.checkpoint_identity,physics=self.checkpoint_physics)
        receipt['checkpoint_export']=dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
                                          seconds=time.perf_counter()-began,
                                          parallelism='host carrier serialization and SHA256; native state already published')
        Path(path).with_suffix(Path(path).suffix+'.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
        return receipt

    def after_step(self, step, **values):
        if self.operator or self.control is not None or self.shock:
            raise ValueError('Metal checkpoint does not implement moving-region diagnostics')
        self.save_due_checkpoint(step, **values)
        return False

    def restore_physics(self, session, layout, weights, mesh):
        """Restore on a fresh initialized segment, retaining its configured timestep variables."""
        if weights is not None:raise ValueError('Metal checkpoint does not yet support moving-mesh remap')
        if self.carrier is None:raise ValueError('Metal checkpoint carrier is not loaded')
        if self.checkpoint_identity is None or self.checkpoint_physics is None or self.restart_metadata is None:
            raise ValueError('native checkpoint identity must bind before restore')
        if self.checkpoint_identity!=self.restart_metadata['identity']:
            raise ValueError('native checkpoint mesh, vertical grid, base state or species differ')
        physics=self.carrier['physics'];saved=physics.get('metal',{})
        if saved.get('schema')!=_METAL_SCHEMA:raise ValueError('carrier lacks Metal checkpoint state')
        if physics['suite']!=getattr(session.config,'suite',None):raise ValueError('checkpoint physics suite differs')
        clock=session.native.clock_state;offset=Fraction(str(self.carrier['time_seconds']))
        if clock.start_time.calendar!='gregorian':raise ValueError('native carrier requires the Gregorian calendar')
        if clock.elapsed!=0 or clock.current_time.stamp()!=self.carrier['valid_time']:
            raise ValueError('Metal resume needs a fresh clock at the checkpoint valid time')
        if clock.dt<=0 or offset%clock.dt:raise ValueError('native checkpoint must lie on an outer-step boundary')
        tx=session.transaction
        if tx.phase!='boundary':raise RuntimeError('Metal checkpoint restore requires a step boundary')
        states=saved['services']
        if set(states)!=set(tx.stateful_services):raise ValueError('Metal checkpoint service set differs')
        if not saved['initialized']:raise ValueError('Metal checkpoint physics was not initialized')
        variables={name:(value,lower) for name,(value,lower) in session.native._variables.items()
                   if value.ndim<2 and name not in _RUN_INIT}
        if set(variables)!=set(saved['variables']):raise ValueError('native scalar and vector variable set differs')
        for name,(value,lower) in saved['variables'].items():
            target,bounds=variables[name]
            if value.shape!=target.shape or value.dtype!=target.dtype or lower!=bounds:
                raise ValueError(f'native variable checkpoint layout differs at {name}')
        text=states['clock']['text'];native=session.native
        expected={f'{pool}.{name}.{level}' for (pool,name),levels in native._text_fields.items() for level in range(1,levels+1)}
        if set(text)!=expected:raise ValueError('native checkpoint text field set differs')
        for key,value in text.items():
            if not isinstance(value,str) or len(value.encode('ascii'))>512 or '\0' in value:
                raise ValueError(f'invalid MPAS text field {key}')
        tx.phase='failed'
        tx.pools.restore(saved['pools'])
        for name,service in tx.stateful_services.items():
            if name!='clock':service.restore_restart_state(states[name])
        for name,(value,_) in saved['variables'].items():variables[name][0][...]=value
        tx.driver.hooks.restore_restart_state(saved['driver_fields']);tx.driver._initialized=True
        for key,value in text.items():
            pool,name,level=key.rsplit('.',2)
            native.put_text_field(pool,name,value,level=int(level),nlevels=native._text_fields[pool,name])
        phase=self.restore_alarm_phase(session,session.config)
        tx.phase='boundary'
        return dict(fields=len(saved['pools']),methods={'same mesh':len(saved['pools'])},alarm_phase=phase)
