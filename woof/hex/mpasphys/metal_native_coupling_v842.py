"""Resident MPAS state publication from an existing Metal dycore candidate.

Only the single-rank, fully solved mesh is supported. The supplied wind
callback must implement the native reconstruction and its compiler-bound GNU
trig graph. MetalMpasWinds supplies that path; generic precise::cos/sin differs.
The caller owns transaction rollback and any later host publication.
"""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _i, _require
from .metal_tendencies_v842 import _launch


class MetalNativeCoupling:
    def __init__(self, session, scalar_names, *, reconstruct_winds):
        if not callable(reconstruct_winds):
            raise ValueError('native coupling requires a source-faithful wind callback')
        self.session=session
        self.names=tuple(scalar_names)
        if len(self.names)!=len(set(self.names)) or (session.config.moist_physics and 'qv' not in self.names):
            raise ValueError('dycore scalar names must be unique and include qv')
        self.native_names=tuple(next(row['components'] for row in session.layout.fields
            if row['pool']=='state' and row['name']=='scalars'))
        if not set(self.native_names)<=set(self.names):
            raise ValueError('dycore scalar names do not cover the native Registry')
        self.reconstruct_winds=reconstruct_winds

    def synchronize(self, atmosphere, *, level=1, prepared_columns=False):
        """Write source-ordered carriers and diagnostics into live Metal pool views.

        The callback returns an object with native ``u_p`` and ``v_p`` device
        arrays. It must preserve the regional dummy-column donor convention.
        Unsupported halos and missing or malformed fields are refused before
        writing pool words. No host staging or host pool publication occurs.
        """
        if prepared_columns:
            raise ValueError('precomputed MPAS column coupling is not implemented')
        d=self.session.layout.dimensions
        nc,ne,nl=d['nCellsSolve'],d['nEdgesSolve'],d['nVertLevels']
        if nc!=d['nCells'] or ne!=d['nEdges']:
            raise ValueError('native Metal coupling requires single-rank fully solved cells and edges')
        s,a=atmosphere.state,atmosphere.saved
        context=_context(s.rho);xp=_arrays(context)
        fields=[('state','rho_zz',level,s.rho),('state','theta_m',level,a.theta_m),
                ('state','u',level,a.normal_velocity),('state','w',level,a.vertical_velocity),
                ('state','scalars',level,s.scalars),('diag','rho_p',1,a.density_perturbation),
                ('diag','rtheta_p',1,a.rho_theta_perturbation),
                ('diag','pressure_p',1,a.pressure_perturbation),('diag','exner',1,a.exner)]
        if s.rho.shape not in ((nl,nc),(nl,nc+1)):
            raise ValueError('native Metal coupling cell extent differs')
        if s.scalars.shape!=(len(self.names),nl,s.rho.shape[-1]):
            raise ValueError('native Metal coupling scalar layout differs')
        if a.normal_velocity.shape not in ((nl,ne),(nl,ne+1)):
            raise ValueError('native Metal coupling edge extent differs')
        wind=self.reconstruct_winds(atmosphere)
        from .metal_diagnostics_v842 import output_diagnostics
        theta,rho,pressure=output_diagnostics(atmosphere,self.names)
        fields.extend([('diag','uReconstructZonal',1,wind.u_p),
                       ('diag','uReconstructMeridional',1,wind.v_p),
                       ('diag','rho',1,rho),('diag','theta',1,theta),('diag','pressure',1,pressure)])
        copies=[]
        for pool,name,index,value in fields:
            target=self.session.pools.field(pool,name,level=index)
            _require(f'{pool}.{name}',value,np.float32)
            if _context(value) is not context or _context(target) is not context:
                raise ValueError('native Metal coupling fields belong to different contexts')
            if value.ndim!=target.ndim or value.shape[-2]!=target.shape[-2] or value.shape[-1] not in (target.shape[-1],target.shape[-1]+1):
                raise ValueError(f'native Metal coupling layout differs for {pool}.{name}')
            if name=='scalars':
                if target.shape[0]!=len(self.native_names):raise ValueError('native scalar row count differs')
                indices=[self.names.index(n) for n in self.native_names]
            else:
                if value.ndim!=2:raise ValueError(f'native Metal coupling rank differs for {pool}.{name}')
                indices=[0]
            copies.append((value,target,xp.asarray(np.asarray(indices,np.int32))))
        for source,target,mapping in copies:
            rows=target.shape[0] if target.ndim==3 else 1
            levels,width=target.shape[-2:]
            if not rows:
                continue
            _launch('hexv842_mpas_transport_rows_f32',rows*levels*width,context,
                    dict(src=source,dst=target,mapping=mapping),
                    dict(nrows=_i(rows),nlev=_i(levels),nsolve=_i(width),
                         nfull=_i(source.shape[-1]),scatter=_i(0)))
        return wind
