"""Partial native MPAS held-source assembly for Metal."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _i
from ..metal_physics_v841 import MetalPhysicsGeometryV841
from .metal_projection_v842 import project
from .metal_tendencies_v842 import MetalMpasSourcesV842, get_tend, _launch


class MetalResidentCoupling:
    """Assemble native held sources for a single-rank edge mesh.

    Mesh geometry is packed on the host once. Unsolved endpoints use the
    native zero garbage cell; multi-rank halo exchange is not bound.
    """
    def __init__(self, session):
        self.session=session
        self.geometry=None
        self.convection_edge=None
        self.invalid=None

    def _geometry(self, context):
        if self.geometry is not None:
            return self.geometry
        p=self.session.pools; d=self.session.layout.dimensions
        nc,ne=d['nCellsSolve'],d['nEdgesSolve']
        raw=p.field('mesh','cellsOnEdge').to_numpy().T.astype(np.int32)
        total=int(d['nCells'])
        if np.any((raw<0)|(raw>total+1)):
            raise ValueError('MPAS Metal coupling encountered invalid edge connectivity')
        if np.any((raw>nc)&(raw<=total)):
            raise ValueError('MPAS Metal coupling requires halo exchange for unsolved edge endpoints')
        cells=raw-np.int32(1)
        dummy=bool(np.any((raw==0)|(raw==total+1)))
        if dummy: cells=np.where((raw==0)|(raw==total+1),nc,cells).astype(np.int32)
        east=p.field('mesh','east').to_numpy().T
        north=p.field('mesh','north').to_numpy().T
        if dummy:
            east=np.pad(east,((0,1),(0,0)))
            north=np.pad(north,((0,1),(0,0)))
        mesh={'cellsOnEdge':cells,'east':east,'north':north,
              'edgeNormalVectors':p.field('mesh','edgeNormalVectors').to_numpy().T}
        if dummy: mesh['garbage_cell']=nc
        self.geometry=MetalPhysicsGeometryV841.from_host(context,mesh)
        return self.geometry

    def restart_state(self):
        # Edge workspaces are derived on the next first RK stage.
        return {}

    def restore_restart_state(self, state):
        if state != {}:
            raise ValueError('invalid resident coupling checkpoint')
        self.geometry=None
        self.convection_edge=None
        self.invalid=None

    def get_tend(self, call):
        p=self.session.pools; d=self.session.layout.dimensions
        nz,nc,ne=d['nVertLevels'],d['nCellsSolve'],d['nEdgesSolve']
        options=call.options
        select=getattr(getattr(self.session, 'driver_api', None), 'coupling_selectors', None)
        if select is not None:
            if (call.pools['rk_step'], call.pools['dynamics_substep']) != (1, 1):
                raise RuntimeError('resident RRFS sources must be assembled before the first RK stage')
            if options.get('do_sppt', False):
                raise NotImplementedError('RRFS Metal coupling does not implement SPPT tendency hooks')
            options=select(options)
        mass=p.field('state','rho_zz',level=2); context=_context(mass); xp=_arrays(context)
        if not self.session.config.moist_physics:
            p.field('diag', 'tend_u_phys')[...] = xp.zeros((nz, ne), np.float32)
            return MetalMpasSourcesV842(xp.zeros((nz, nc), np.float32),
                xp.zeros((nz, ne), np.float32), xp.zeros((nz, nc), np.float32),
                xp.zeros((len(self.session.resident_preparation.names), nz, nc), np.float32))
        pbl=options['config_pbl_scheme']!='off'
        cu=options['config_convection_scheme'] in ('cu_tiedtke','cu_ntiedtke')
        if select is not None and (pbl or cu) and d['nEdgesSolve'] != d['nEdges']:
            raise NotImplementedError('RRFS Metal coupling requires all physical edges for tendency publication')
        if self.invalid is None:
            self.invalid=xp.zeros((1,),np.int32)
            self.convection_edge=xp.zeros((nz,ne),np.float32)
        pbl_edge=p.field('tend_physics','rublten_Edge') if pbl else xp.zeros((nz,ne),np.float32)
        rates={row['pool_name']:p.field('tend_physics',row['pool_name'])
               for row in self.session.layout.fields if row['pool']=='tend_physics'
               and (row['pool_name'].endswith('ten') or row['pool_name'] in ('rthratenlw','rthratensw'))}
        if (pbl or cu) and (call.pools['rk_step'],call.pools['dynamics_substep']) == (1,1):
            geometry=self._geometry(context)
            for enabled,u,v,target in ((pbl,'rublten','rvblten',pbl_edge),
                                       (cu,'rucuten','rvcuten',self.convection_edge)):
                if enabled:
                    du,dv=rates[u],rates[v]
                    if geometry.regional_garbage_cell is not None:
                        padded=[]
                        for value in (du,dv):
                            target_rate=xp.empty((nz,nc+1),np.float32)
                            _launch('hexv842_mpas_pad_rates_f32',nz*(nc+1),context,
                                    dict(rates=value,out=target_rate),dict(nlev=_i(nz),ncells=_i(nc)))
                            padded.append(target_rate)
                        du,dv=padded
                    target[...] = project(geometry=geometry,zonal=du,meridional=dv,invalid=self.invalid)
        out=get_tend(mass=mass,mass_edge=p.field('diag','rho_edge'),
            theta_m=p.field('state','theta_m'),scalars=p.field('state','scalars'),
            scalar_names=self.session.resident_preparation.names,tendencies=rates,selectors=options,
            pbl_edge=pbl_edge,convection_edge=self.convection_edge,tend_u_phys=p.field('diag','tend_u_phys'),
            n_cells_solve=nc,n_edges_solve=ne,rk_step=call.pools['rk_step'],
            dynamics_substep=call.pools['dynamics_substep'])
        p.field('tend','scalars_tend')[...] = out.scalars
        p.publish()
        return out
