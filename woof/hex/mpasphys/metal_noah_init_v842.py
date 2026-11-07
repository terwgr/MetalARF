"""Native MPAS land-use and Noah initialization on Metal."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _arrays
from . import noah_v842
from .metal_noah_v842 import initialize_landuse, initialize_soil


class MetalNoahInitialization:
    """Initialize all local cells using authenticated native Noah resources."""

    def __init__(self, framework, resources, context):
        self.framework, self.resources, self.context = framework, resources, context

    def _write(self, pool, name, value, count):
        framework = self.framework
        framework.write(pool, name, value, count=count)
        views = framework.session.pools
        row = views.rows[pool, name]
        key = (pool, row['pool_name'], 1)
        if key in views.buffers:
            _, device, selection = views.buffers[key]
            current = framework.native.get_field(pool, row['pool_name'], level=1)
            device[...] = current[selection]

    def initialize_landuse(self):
        f, resources = self.framework, self.resources
        native, config, nc = f.native, f.configs, f.dimensions['nCells']
        classification = native.get_text_field('sfc_input', 'mminlu')
        if classification != resources['landuse']:
            raise ValueError('landuse resource classification differs from mminlu')
        native.landuse_state = dict(mminlu=classification,
            isurban=noah_v842.LANDUSE_URBAN_INDEX.get(classification, 0),
            iswater=int(f.field('sfc_input', 'iswater')),
            isice=int(f.field('sfc_input', 'isice')))
        if nc * 96 > self.context.recommended_working_set_size:
            raise MemoryError('MPAS landuse initialization workspace exceeds Metal recommended working set')
        floats, integers = np.zeros((nc,14),np.float32), np.zeros((nc,9),np.int32)
        fields = (('mesh','latCell'),('sfc_input','snoalb'),('sfc_input','snowc'),
                  ('sfc_input','xice'),('sfc_input','sfc_albbck'),('diag_physics','z0'),
                  ('diag_physics','znt'),('diag_physics','sfc_emibck'),
                  ('diag_physics','sfc_albedo'),('diag_physics','sfc_emiss'),
                  ('diag_physics','thc'),('diag_physics','mavail'),('diag_physics','ust'))
        for i,key in enumerate(fields):
            if key in native._fields:
                floats[:,i] = f.field(*key)[:nc]
        floats[:,13] = native.get_variable('xice_threshold')[0]
        integers[:,0] = f.field('sfc_input','ivgtyp')[:nc]
        integers[:,1:] = [native.clock_state.current_time.day_of_year,
            native.landuse_state['iswater'], native.landuse_state['isice'],
            int(config['config_do_restart']), int(config['config_sfc_albedo']),
            int(config['config_frac_seaice']), int(fields[11] in native._fields),
            int(fields[12] in native._fields)]
        xp = _arrays(self.context)
        packed = xp.asarray(floats)
        status = initialize_landuse(packed,xp.asarray(integers),resources['landuse_tables']).to_numpy()
        if np.any(status):
            raise RuntimeError('MPAS landuse initialization failed: '+str(status.tolist()))
        result = packed.to_numpy()
        for i,key in enumerate(fields):
            if i >= 4 and key in native._fields:
                self._write(*key,result[:,i],count=nc)

    def initialize_noah(self):
        f, native, config = self.framework, self.framework.native, self.framework.configs
        nc = f.dimensions['nCells']
        if nc * 80 > self.context.recommended_working_set_size:
            raise MemoryError('MPAS Noah initialization workspace exceeds Metal recommended working set')
        names = ('tslb','smois','sh2o','snoalb','snow','snowh')
        values = np.concatenate([f.field('sfc_input',name)[...,:nc].reshape(-1,nc).T
                                 for name in names],axis=1)
        integers = np.empty((nc,4),np.int32)
        integers[:,0] = f.field('sfc_input','ivgtyp')[:nc]
        integers[:,1] = f.field('sfc_input','isltyp')[:nc]
        integers[:,2:] = int(config['config_do_restart']),int(config['config_sfc_snowalbedo'])
        xp = _arrays(self.context)
        packed = xp.asarray(np.ascontiguousarray(values))
        status = initialize_soil(packed,xp.asarray(integers),self.resources['tables']).to_numpy()
        if np.any(status):
            raise RuntimeError('MPAS Noah initialization failed: '+str(status.tolist()))
        result = packed.to_numpy();offset = 0
        for name in names:
            width = 4 if name in ('tslb','smois','sh2o') else 1
            value = result[:,offset:offset+width].T
            self._write('sfc_input',name,value if width>1 else value[0],count=nc)
            offset += width
