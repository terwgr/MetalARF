"""Metal turbine kernels with the CUDA driver's setup and call ordering."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .array import MetalArray, MetalArrayModule
from woof.core.wind_farm import WindFarm
from woof.wind_table import read_table, validate_config


def host(value):
    return value.to_numpy() if isinstance(value, MetalArray) else np.asarray(value)


@lru_cache(None)
def signature(name):
    return next(k for k in json.loads(Path(__file__).with_name('kernel_signatures.json').read_text())['kernels']
                if k['name'] == name)


class Kernel:
    def __init__(self, name, context):
        self.name, self.context = name, context

    def __call__(self, grid, block, args):
        spec = signature(self.name)
        arrays = [v for v in args if isinstance(v, (MetalArray, np.ndarray))]
        scalars = [v for v in args if not isinstance(v, (MetalArray, np.ndarray))]
        xp = MetalArrayModule(self.context)
        resources = {r[0]: xp.asarray(v) for r, v in zip(spec['resources'], arrays)}
        declared = [p for p in spec['parameters'] if p[0] != 'n' or self.name != 'wind_fitch_sparse']
        parameters = {k: int(v) if t == 'int32' else float(v)
                      for (k, t), v in zip(declared, scalars)}
        count = int(np.prod(grid)) * int(np.prod(block))
        if not count:
            return
        if self.name == 'wind_fitch_sparse':
            parameters['n'] = count
        if len(arrays) != len(spec['resources']) or len(scalars) != len(declared):
            raise ValueError(f'{self.name}: turbine kernel argument count differs')
        if any(v.buffer.context is not self.context or not v.flags.c_contiguous for v in resources.values()):
            raise ValueError('turbine arrays must be contiguous on one Metal context')
        self.context.submit_kernel(self.name, {k: v.view((v.size,)) for k, v in resources.items()},
                                   parameters, exact_policy=False, grid_size=count,
                                   threads_per_group=128)
        # Only setup mapping uses NumPy pointers. Forecast calls stay resident.
        for r, v in zip(spec['resources'], arrays):
            if r[2] != 'read' and isinstance(v, np.ndarray):
                v[...] = resources[r[0]].to_numpy()


class _Cost:
    def __init__(self):
        self.calls = 0

    def begin(self):
        self.calls += 1

    def end(self):
        pass


class MetalWindFarm(WindFarm):
    def __init__(self, cfg, shape, *, context, **kwargs):
        self.context = context
        self.arrays = MetalArrayModule(context)
        loader = lambda module, name: Kernel(name, context)
        super().__init__(cfg, shape, _array_module=np, _kernel_loader=loader, _cost=_Cost(),
                         **{k: host(v) if v is not None and k in ('latitude', 'longitude', 'msf', 'z_interface') else v
                            for k, v in kwargs.items()})
        uploaded = {}
        for name, value in vars(self).copy().items():
            if isinstance(value, np.ndarray):
                uploaded[id(value)] = self.arrays.asarray(value)
                setattr(self, name, uploaded[id(value)])
        if self.sparse:
            self.fused_prefix = tuple(uploaded.get(id(v), v) for v in self.fused_prefix)
        # The CUDA replay is optional instrumentation, not a physics carrier.
        self.replay = {}

    def apply_gad(self, atmosphere, state, cfg, tendencies):
        from woof.core.physics import couple_column_tendencies, couple_sase_w_tendency
        self.sample(atmosphere, {})
        o = self.options
        self.launch('wind_gad', self.nc*self.shape[0], (
            np.int32(self.nc), np.int32(self.shape[0]), np.int32(self.shape[2]), self.dx, self.dy,
            np.float32(o['gad_sigma_grid']*self.dx), self.ptr, self.members, self.turbines,
            self.meta, self.curves, self.blades, self.polars, self.pm, self.hub,
            atmosphere['z_interface'], atmosphere['u'], atmosphere['v'],
            np.float32(o['gad_induction_tolerance']), np.int32(o['gad_induction_iterations']),
            self.du, self.dv, self.dw, self.failures))
        force = couple_column_tendencies(state, cfg, ru=self.du, rv=self.dv)
        vertical = couple_sase_w_tendency(state, cfg, self.dw)
        for name, target in (('ru', force.ru), ('rv', force.rv), ('rw', vertical)):
            previous = getattr(tendencies, name)
            if previous is not None:
                self.context.submit_kernel('hexv841_add_inplace_f32',
                    {'increment': previous.view((previous.size,)), 'target': target.view((target.size,))},
                    {'nlevels': 1, 'nowners': target.size}, exact_policy=False, grid_size=target.size)
            setattr(tendencies, name, target)

    def history_fields(self):
        # Reductions belong to the history boundary, as on CUDA; no step downloads.
        hub, turbines, col, farm = (host(v) for v in (self.hub, self.turbines, self.col, self.farm_index))
        power, wind = hub[:, 7], hub[:, 2]
        nf = len(self.farm_ids)
        fields = dict(POWER=np.bincount(col, weights=power, minlength=self.nc).astype(np.float32).reshape(self.shape[1:]),
                      WIND_TKE_SOURCE=host(self.tke_rate), WIND_TKE_SOURCE_MAX=host(self.tke_rate).max(axis=0),
                      TURBINE_POWER=power, TURBINE_HUB_WIND=wind, TURBINE_FARM_INDEX=(farm+1).astype(np.float32),
                      TURBINE_HUB_HEIGHT=turbines[:, 2],
                      FARM_POWER=np.bincount(farm, weights=power, minlength=nf).astype(np.float32),
                      FARM_HUB_WIND=(np.bincount(farm, weights=wind, minlength=nf)/np.bincount(farm, minlength=nf)).astype(np.float32))
        return {k: self.arrays.asarray(np.ascontiguousarray(v)) for k, v in fields.items()}

    def receipt(self):
        return dict(schema='gpuwm-wind-receipt.v1', scheme=self.scheme, turbine_count=self.nt,
                    table_sha256=self.table_sha256, execution='fused_sparse_columns' if self.sparse else 'legacy',
                    occupied_columns=self.active_columns.size, timing_source='not instrumented', cost=None,
                    options=self.options, turbine_ids=self.ids, farm_ids=self.farm_ids,
                    input_provenance=self.data.get('provenance', {}),
                    gad_unconverged_cell_calls=int(host(self.failures).sum(dtype=np.int64)))

    def write_replay(self, path):
        return None


def attach_wind_farm(cfg, shape, *, context, **kwargs):
    validate_config(cfg)
    data, digest = read_table(cfg.wind_farm_table, cfg.wind_farm_scheme)
    reason = None
    if cfg.wind_farm_scheme == 'auto':
        if not 1000 <= min(cfg.dx, cfg.dy) <= max(cfg.dx, cfg.dy) <= cfg.wind_farm_max_dx_m:
            reason = 'outside evaluated candidate spacing range'
        elif cfg.bl_pbl_physics != 5 or not cfg.bl_mynn_tkeadvect or cfg.bldt != 0:
            reason = 'requires every-step MYNN with advected TKE'
    rows = data['turbines']
    if not reason and rows and not any('grid_x' in t for t in rows):
        lat, lon = kwargs.get('latitude'), kwargs.get('longitude')
        if lat is None or lon is None:
            raise ValueError('geographic turbines require domain latitude/longitude')
        ll = np.asarray([[t['latitude'], t['longitude']] for t in rows], np.float32)
        xy = np.empty((len(rows), 2), np.float32)
        nz, ny, nx = shape
        Kernel('wind_locate', context)(((len(rows)+127)//128,), (128,),
            (np.int32(len(rows)), np.int32(nx), np.int32(ny), ll,
             np.ascontiguousarray(host(lat), dtype=np.float32), np.ascontiguousarray(host(lon), dtype=np.float32), xy))
        rows = [dict(t, grid_x=float(x), grid_y=float(y)) for t, (x, y) in zip(rows, xy)
                if np.isfinite(x) and np.isfinite(y) and 1 <= x < nx-2 and 1 <= y < ny-2]
        data = dict(data, turbines=rows)
    if not rows:
        reason = reason or 'no turbines in the modeled interior'
    receipt = dict(requested=cfg.wind_farm_scheme, enabled=reason is None, table_sha256=digest,
                   candidate_max_dx_m=cfg.wind_farm_max_dx_m, turbine_count=len(rows),
                   reason=reason or 'turbines in domain and compatible grid and physics')
    return (None if reason else MetalWindFarm(cfg, shape, context=context, prepared=(data, digest), **kwargs)), receipt


def attach_voronoi_wind_farm(cfg, *, context, latitude, longitude, area, z_interface):
    from types import SimpleNamespace
    data, digest = read_table(cfg.wind_farm_table, 'fitch')
    rows = data['turbines']
    if not rows:
        return None
    if any('grid_x' in t or 'grid_y' in t for t in rows):
        raise ValueError('Voronoi turbines require geographic coordinates')
    lat, lon, area = (np.ascontiguousarray(host(v), dtype=np.float32).ravel() for v in (latitude, longitude, area))
    if lon.size != lat.size or area.size != lat.size or not np.all(np.isfinite(area) & (area > 0)):
        raise ValueError('Voronoi cell areas must be finite and positive for turbine density')
    ll = np.asarray([[t['latitude'], t['longitude']] for t in rows], np.float32)
    indices = np.empty(len(rows), np.int32)
    Kernel('wind_locate_cells', context)(((len(rows)+127)//128,), (128,),
        (np.int32(len(rows)), np.int32(lat.size), ll, lat, lon, indices))
    mapped = [dict(t, grid_x=int(i), grid_y=0) for t, i in zip(rows, indices)]
    local = SimpleNamespace(wind_farm_scheme='fitch', wind_farm_table=cfg.wind_farm_table,
        windfarm_tke_factor=getattr(cfg, 'windfarm_tke_factor', .25), dx=1., dy=1., dt=cfg.dt)
    return MetalWindFarm(local, (z_interface.shape[0]-1, 1, lat.size), context=context,
        prepared=(dict(data, turbines=mapped), digest), msf=np.reciprocal(np.sqrt(area)).reshape(1, -1),
        z_interface=z_interface, cell_layout=True)
