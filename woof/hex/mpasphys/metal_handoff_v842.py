"""MPAS post-transport callback for the Metal dycore's candidate tuple."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _i, _require
from .metal_tendencies_v842 import _launch
from .metal_transport_v842 import source_baseline, handoff


class MetalResidentTransport:
    """Bind a dycore candidate and native-pool synchronization callback.

    The caller must roll back its active physics transaction if preparation or
    synchronization fails; the candidate remains unpublished until commit.
    """
    def __init__(self, session):
        self.session=session
        self.pending=None

    def bind(self, *, current, carried, outer, rate, held, scalar_advection,
             monotonic, positive_definite, convection, scalar_names, synchronize,
             boundary_columns=None):
        if self.pending is not None:
            raise RuntimeError('MPAS Metal transport candidate is already bound')
        if not callable(synchronize):
            raise TypeError('MPAS Metal transport requires native-pool synchronization')
        if convection in ('cu_grell_freitas','cu_ntiedtke') and rate is None:
            raise RuntimeError('MPAS convection requires the final RK theta rate')
        names=tuple(scalar_names)
        if len(names)!=len(set(names)) or any(n not in names for n in self.session.resident_preparation.names):
            raise ValueError('dycore scalar rows do not cover the native Registry names')
        self.pending=(current,carried,outer,rate,held,bool(scalar_advection),
                      bool(monotonic),bool(positive_definite),convection,names,synchronize,
                      boundary_columns)

    def prepare(self, call):
        if self.pending is None:
            raise RuntimeError('MPAS post-transport callback has no bound dycore candidate')
        current,carried,outer,rate,held,advect,monotonic,positive,convection,candidate_names,synchronize,boundary_columns=self.pending
        if call is not None and call.options['config_convection_scheme'] != convection:
            raise ValueError('bound transport convection differs from the native driver selection')
        p=self.session.pools; d=self.session.layout.dimensions
        nz,nc=d['nVertLevels'],d['nCellsSolve']
        if d['nCells'] != nc:
            raise ValueError('MPAS Metal transport requires halo synchronization for unsolved cells')
        names=self.session.resident_preparation.names
        if outer.scalars.ndim!=3 or outer.scalars.shape[:2] != (len(candidate_names),nz) or current.scalars.shape != outer.scalars.shape:
            raise ValueError('MPAS Metal transport candidate scalar layout differs')
        full=outer.scalars.shape[2]
        if full not in (nc,nc+1) or outer.rho.shape != (nz,full) or carried.theta_m.shape != (nz,full) or held.scalars.shape != outer.scalars.shape:
            raise ValueError('MPAS Metal transport candidate extent differs')
        for name,value in (('outer scalars',outer.scalars),('current scalars',current.scalars),('held scalars',held.scalars)):
            _require(name,value,np.float32,(len(candidate_names),nz,full))
        for name,value in (('outer rho',outer.rho),('theta',carried.theta_m)):
            _require(name,value,np.float32,(nz,full))
        if rate is not None:
            _require('modified theta rate',rate,np.float32,(nz,full))
        xp=_arrays(_context(outer.scalars))
        mapping=xp.asarray(np.array([candidate_names.index(name) for name in names],np.int32))
        identity=xp.asarray(np.array([0],np.int32))
        native_indices=xp.asarray(np.arange(len(names),dtype=np.int32))
        def gather(value, indices, rows, width=full):
            output=xp.empty((rows,nz,width),np.float32)
            _launch('hexv842_mpas_transport_rows_f32',rows*nz*width,_context(value),
                    dict(src=value,dst=output,mapping=indices),dict(nrows=_i(rows),nlev=_i(nz),
                         nsolve=_i(width),nfull=_i(full),scatter=_i(0)))
            return output
        def scalar(value):
            return gather(value,identity,1)[0]
        before=gather(outer.scalars,mapping,len(names))
        after=gather(current.scalars,mapping,len(names))
        source=gather(held.scalars,mapping,len(names))
        rho=scalar(outer.rho); theta=scalar(carried.theta_m)
        rth=xp.zeros((nz,full),np.float32) if rate is None else scalar(rate)
        rqv=xp.empty((nz,full),dtype=outer.rho.dtype)
        if advect and (monotonic or positive):
            before=source_baseline(before=before,source=source,rho_old=rho,
                                   dt=self.session.dt,n_cells_solve=nc)
            p.field('state','scalars',level=1)[...] = gather(before,native_indices,len(names),nc)
        handoff(before=before,after=after,theta_m=theta,scalar_names=names,
                rqvdynten=rqv,rthdynten=rth,convection_scheme=convection,monotonic=monotonic,
                dt=self.session.dt,n_cells_solve=nc)
        if boundary_columns is not None:
            # LB1 removes boundary relaxation from convection's resolved
            # forcing, after handoff and before native field publication.
            from .metal_lagged_v842 import zero_columns
            zero_columns(rth, boundary_columns)
            zero_columns(rqv, boundary_columns)
        _launch('hexv842_mpas_transport_rows_f32',len(names)*nz*full,_context(after),
                dict(src=after,dst=current.scalars,mapping=mapping),dict(nrows=_i(len(names)),nlev=_i(nz),
                     nsolve=_i(full),nfull=_i(full),scatter=_i(1)))
        synchronize(current,carried)
        if convection in ('cu_grell_freitas','cu_ntiedtke') and ('tend_physics','rqvdynten') in self.session.native._fields:
            p.field('tend_physics','rqvdynten')[...] = gather(rqv,identity,1,nc)[0]
        p.field('tend_physics','rthdynten')[...] = gather(rth,identity,1,nc)[0]
        p.publish()
        self.pending=None

    def restart_state(self):
        if self.pending is not None:
            raise RuntimeError('MPAS transport cannot checkpoint an unconsumed candidate')
        return {}

    def restore_restart_state(self, state):
        if state != {}:
            raise ValueError('invalid MPAS transport checkpoint')
        self.pending=None
