"""MPAS output diagnostics on Metal, with shared host stream orchestration.

Native pools carry the padded allocation words; translated CUDA kernels own
all field arithmetic. Host work only packs, transports and publishes words.
"""
import numpy as np

from woof.metal.array import MetalArrayModule
from .metal_tendencies_v842 import _launch
from .resident_output_diagnostics_v842 import ResidentOutputDiagnostics, PRIMARY, INITIALIZED
from ..mpasdiag.isobaric import FIELDS as ISOBARIC, IsobaricDiagnostics
from ..mpasdiag import convective_rrfs
from ..mpasdiag.utils import DiagnosticsUtils, Direction


_ARGUMENTS = {
    'fixed_pressure': 'nc ni no p f po out upper',
    'layer_mean': 'nc nl p1 p2 f p out',
    'slp_columns': 'nc nl p temp h q out failed',
    'slp_finish': 'nc out failed scale',
    'prepare_mass': 'nc nl stride thermo pp pb theta exner q p temp dew',
    'prepare_interfaces': 'nc nl hs p h pi',
    'prepare_vertices': 'nc nv nl degree cov kite area p pv',
    'compute_diag_cols': 'c0 n nl cap mask rvord height sz u su v sv theta st qv sq exner se pb sb pp sp '
                         'outs work',
    'vertex_helicity_gather_pool': 'nc nl degree offsets entries kite vort sv uh',
    'split_helicity_pool': 'nc nl w sw area positive negative',
    'smooth_helicity': 'ns nl maxedges bdy nedges edges cells dv dc positive negative spos sneg',
    'update_diag_pool': 'ns nl mask steps convection z sz w sw u su v sv pb sb pp spr refl sr u10 v10 t2 '
                        'rainc rainnc snow uh neg tail outs work',
    'reset_diag_pool': 'nc mask outs',
}


class DiagnosticKernels:
    array_module = np

    def __init__(self, context):
        self.context = context
        self.arrays = MetalArrayModule(context)
        self.work_bytes = context.recommended_working_set_size // 16

    def launch(self, name, count, args):
        if not count:
            return
        values = dict(zip(_ARGUMENTS[name].split(), args, strict=True))
        if 'outs' in values:
            values['no'] = values['outs'].shape[1]
        if 'mask' in values:
            mask = int(values.pop('mask'))
            values.update(mask_lo=np.asarray(mask & 0xffffffff, np.uint32).view(np.int32),
                          mask_hi=np.asarray(mask >> 32, np.uint32).view(np.int32))
        if name == 'update_diag_pool' and not isinstance(values['tail'], np.ndarray):
            values['tail'] = np.zeros(int(values['ns']), np.float32)
        resources, parameters, targets = {}, {}, {}
        for key, value in values.items():
            if isinstance(value, np.ndarray) and value.ndim:
                packed = np.ascontiguousarray(value)
                resources[key] = self.arrays.asarray(packed.ravel())
                targets[key] = value
            else:
                parameters[key] = np.float32(value) if isinstance(value, (float, np.floating)) else np.int32(value)
        if name != 'fixed_pressure':
            parameters['dispatch_count'] = np.int32(count)
        kernel = 'hexv842_deck_' + name
        _launch(kernel, count, self.context, resources, parameters)
        from woof.metal.generated import kernel_specs
        for key, dtype, access in getattr(kernel_specs, kernel.upper() + '_RESOURCES'):
            if access == 'read_write':
                targets[key][...] = resources[key].to_numpy().reshape(targets[key].shape)

    def scale(self, value, factor):
        source = self.arrays.asarray(np.ascontiguousarray(value).ravel())
        result = self.arrays.empty(source.shape, np.float32)
        _launch('hexv842_deck_scale', source.size, self.context,
                dict(src=source, dst=result), dict(n=np.int32(source.size), factor=np.float32(factor)))
        return result.to_numpy().reshape(value.shape)

    @staticmethod
    def helicity_plan(cov, nc, vertex_order):
        sequence = np.arange(cov.size, dtype=np.int64) if vertex_order is None else (
            np.asarray(vertex_order)[:, None] * cov.shape[1] + np.arange(cov.shape[1])).ravel()
        cells = cov.ravel()[sequence].astype(np.int64) - 1
        order = np.argsort(cells * cells.size + np.arange(cells.size))
        entries = sequence[order].astype(np.int32)
        offsets = np.zeros(nc + 2, np.int32)
        offsets[1:] = np.cumsum(np.bincount(cells, minlength=nc + 1))
        return offsets, entries


class MetalOutputDiagnostics(ResidentOutputDiagnostics):
    produced = (PRIMARY | INITIALIZED | frozenset(ISOBARIC) | frozenset(convective_rrfs.FIELDS)
                | {'t_iso_levels', 'z_iso_levels'})
    convective_maxima = convective_rrfs.ACCUMULATORS

    def __init__(self, session, dycore):
        self.session, self.dycore = session, dycore
        self.cp = np
        self.buffers = {}
        self._fields, self._fields_state = {}, None
        self._event_views = None
        self._resume = None
        self.when = None
        self.updates = 0
        self.last_step = None
        self.kernels = DiagnosticKernels(session.pools.arrays.context)

    def bind(self, writer):
        self.writer = writer
        self.when = writer.current_time(step=0, dt=0)
        self.utils = DiagnosticsUtils()
        self.utils.init(self.views)
        self.pools = {}
        fields = {
            'mesh': {'zgrid', 'cellsOnVertex', 'kiteAreasOnVertex', 'areaTriangle', 'areaCell',
                     'bdyMaskCell', 'nEdgesOnCell', 'edgesOnCell', 'cellsOnCell', 'dvEdge', 'dcEdge'},
            'state': {'theta_m', 'scalars', 'w'},
            'diag': self.produced | {'pressure_base', 'pressure_p', 'exner', 'relhum',
                                     'uReconstructZonal', 'uReconstructMeridional'},
            'diag_physics': {'do_diag_dbz', 'refl10cm', 'u10', 'v10', 't2m', 'raincv',
                             'rainncv', 'snowncv', 'wind_z_agl_tail'},
        }
        for pool, names in fields.items():
            self.pools[pool] = {r['pool_name']: np.array(self.session.native.get_field(pool, r['pool_name']), copy=True, order='C')
                                for r in self.session.layout.fields
                                if r['pool'] == pool and r['pool_name'] in names}
        nv = self.session.layout.dimensions['nVertices']
        for name in ('cellsOnVertex', 'kiteAreasOnVertex'):
            self.pools['mesh'][name] = self.pools['mesh'][name][:, :nv].copy()
        self.pools['mesh']['areaTriangle'] = self.pools['mesh']['areaTriangle'][:nv].copy()
        self.pools['mesh'].update(self.session.layout.dimensions)
        self.pools['state']['index_qv'] = self.session.layout.indices['state', 'index_qv']
        self.isobaric = IsobaricDiagnostics(exchange_pressure_halo=lambda p: None,
            backend=(np, self.kernels.launch, self.kernels.scale))
        self.isobaric.setup(self.pools, None, self.utils)
        self.convective = convective_rrfs.ConvectiveDiagnosticsModule(backend=self.kernels)
        self.convective.setup(self.pools, None, self.utils)
        self.buffers = {('diag', name): self.pools['diag'][name] for name in self.produced}
        self.accumulates = any(self.utils.stream_inclusion_count(name, Direction.OUTPUT)
                               for name in self.convective_maxima)
        if self.session.layout.configs.get('config_apply_lbcs', False):
            # The shared ordering helper uses only topology; its final upload is NumPy here.
            from .rrfs.output_diagnostics_rrfs import native_rank_vertex_order
            from types import SimpleNamespace
            proxy = SimpleNamespace(native=self.session.native, layout=self.session.layout,
                                    device_pools=SimpleNamespace(cp=np))
            self.convective.driver.vertex_order = native_rank_vertex_order(proxy)

    def _refresh(self):
        self.session.pools.publish()
        for pool in ('state', 'diag', 'diag_physics'):
            values = self.pools[pool]
            for name, target in values.items():
                if isinstance(target, np.ndarray) and (pool != 'diag' or name not in self.produced):
                    value = self.session.native.get_field(pool, name)
                    target[...] = value[tuple(slice(0, size) for size in target.shape)]

    def _solve_primary(self):
        a, driver = self.dycore.atmosphere, self.dycore
        solved = driver.horizontal.solve_diagnostics(a.state.rho, a.state.rho_u,
            dt=driver.config.config_dt, apvm_upwinding=driver.config.config_apvm_upwinding,
            normal_velocity=a.saved.normal_velocity, rk_step=3)
        for name, value in (('vorticity', solved.vorticity), ('divergence', solved.divergence), ('ke', solved.kinetic_energy)):
            target = self.pools['diag'][name]
            words = value.to_numpy()
            target[..., :words.shape[-1]] = words

    def _update_convective(self):
        from .rrfs.output_diagnostics_rrfs import _StreamIntervals
        self.convective.update(self.session.layout.configs, _StreamIntervals(self.writer),
                               self.dycore.config.config_dt)

    def before_write(self, *, step, when):
        self._refresh()
        self.when = when
        self._begin_event()
        resume, self._resume = (self._resume, None) if step == 0 else (None, self._resume)
        if resume is not None:
            for name, value in resume.items():
                self.pools['diag'][name][...] = value
        elif step == 0:
            self.convective.reset()
        if self.accumulates or any(v.output_alarm for v in self.views()):
            self._solve_primary()
        if resume is None:
            self._update_convective()
        self.isobaric.compute()
        self.convective.compute()
        for name in self.produced:
            self.session.native.put_field('diag', name, self.pools['diag'][name])
        self.updates += 1
        self.last_step = step

    def field(self, pool, name):
        return None

    def accumulator_state(self):
        return {name: self.pools['diag'][name].copy() for name in self.convective_maxima}

    def after_snapshot(self):
        self.convective.reset()
        self._end_event()

    def close(self):
        self.buffers.clear()
        self.pools.clear()

    def receipt(self):
        return dict(updates=self.updates, last_step=self.last_step, produced=sorted(self.produced),
                    arithmetic='translated CUDA kernels on Metal; shared stream scheduling')
