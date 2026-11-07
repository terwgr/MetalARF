"""Resident RRFS MYNN-EDMF PBL slot for the supported source revisions."""

import numpy as np

from .. import metal_bl_mynn_v842 as shared
from . import bl_mynnedmf_c6d08709, bl_mynnedmf_c942418
from .driver_rrfs import (
    _PBL_COMMON_TEND, _PBL_DIAG_TEND, _PBL_NUMBER_TEND, _PBL_VOLUME,
)
from woof.metal.array import MetalArray

from .metal_mynnedmf_columns import launch_columns


_BOUNDS = ('ids', 'ide', 'jds', 'jde', 'kds', 'kde', 'ims', 'ime', 'jms', 'jme',
           'kms', 'kme', 'jts', 'jte', 'kts', 'kte')
_SURFACE_INPUTS = ('hfx', 'hpbl', 'qfx', 'ust', 'wspd', 'znt', 'ch', 'qsfc')
_ZERO_SURFACE = ('uoce_p', 'voce_p', 'excessh_p', 'excessq_p', 'maxwidthbl_p',
                 'maxmfbl_p', 'zbl_plume_p')
_SURFACE_OUTPUTS = dict(hpbl='hpbl_p', kpbl='kpbl_p', maxmf='maxmfbl_p',
                       maxwidth='maxwidthbl_p', ztop_plume='zbl_plume_p',
                       excess_h='excessh_p', excess_q='excessq_p')
SOURCE_KERNELS = {
    'rrfs.bl_mynnedmf_c6d08709': (bl_mynnedmf_c6d08709, 'hexrrfs_mynnedmf_columns'),
    'rrfs.bl_mynnedmf_c942418': (bl_mynnedmf_c942418, 'hexrrfs_mynnedmf_old_columns'),
}


class MetalMpasMynnEdmf:
    """Use the original PBL prepare, column and publication order on Metal."""

    def __init__(self, module='rrfs.bl_mynnedmf_c6d08709'):
        self.source, self.entrypoint = SOURCE_KERNELS[module]
        self.flags = tuple(name for name in self.source.ARGUMENTS if name.startswith('f_'))
        self.extra_surface = {name: name + '_p'
                              for name in getattr(self.source, 'EXTRA_SURFACE_FIELDS', ())}

    def restart_state(self):
        return {}

    def restore_restart_state(self, state):
        if state != {}:
            raise ValueError('RRFS MYNN-EDMF state belongs to native pools')

    def launch_slot(self, call):
        if (call.slot, call.scheme) != ('pbl', 'bl_mynnedmf'):
            raise ValueError('RRFS MYNN-EDMF requires its PBL slot')
        pools = call.pools
        source = self.source
        if pools.level != 1:
            raise ValueError('RRFS MYNN-EDMF requires native state time level one')
        native, context = pools.native, pools.cp.context
        options = dict(pools['configs'])
        options.update(call.options)
        scalar = pools['scalar_state']
        metadata = {name: scalar[name] for name in (*_BOUNDS, *self.flags, 'dt_pbl', 'do_chemistry', 'num_chem')}
        metadata.update(its=pools.session.arguments.its, ite=pools.session.arguments.ite)
        nc, nz = int(pools['nCellsSolve']), int(pools['nVertLevels'])
        if (nc < 1 or nz < 3 or nz * nc > np.iinfo(np.int32).max
                or any(metadata[name] != 1 for name in ('ims', 'jms', 'kms', 'jts', 'jte', 'its', 'kts'))
                or metadata['ite'] != nc or metadata['kte'] != nz
                or metadata['ime'] < nc or metadata['kme'] < nz + 1):
            raise ValueError('RRFS MYNN-EDMF requires one native solve row and the upper interface level')
        if (metadata['do_chemistry'] or metadata['f_nbca'] or metadata['f_qoz']
                or options.get('config_mynn_enh_mix') or options.get('config_spp_pbl')
                or options['config_do_DAcycling']):
            raise NotImplementedError('Metal RRFS MYNN-EDMF requires chemistry, black carbon, ozone, fire mixing, SPP and DA cycling disabled')
        dt = np.float32(metadata['dt_pbl'])
        length = np.float32(options['config_len_disp'])
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(length) or length <= 0:
            raise ValueError('RRFS MYNN-EDMF requires positive finite native timestep and grid length')

        prep, diag, tend, surface, mesh = (pools[name] for name in
            ('prepared', 'diag_physics', 'tend_physics', 'sfc_input', 'mesh'))

        def require(name, array, dtype, shape):
            if (not isinstance(array, MetalArray) or array.buffer.context is not context
                    or array.dtype != np.dtype(dtype) or array.shape != shape):
                raise ValueError(f'RRFS MYNN-EDMF {name} requires a resident {np.dtype(dtype)} {shape}')
            return array

        prepared = {}
        for key, name in source.NATIVE_ARRAY_BINDINGS.items():
            spec = source.ARGUMENTS[key]
            storage = native.variable_storage(name)
            if storage is None:
                if not spec['optional']:
                    raise ValueError(f'RRFS MYNN-EDMF requires native allocation {name}')
                continue
            if len(spec['dimensions']) == 4 or key in ('vd3d', 'frp_mean', 'emis_ant_no'):
                continue
            shape = (int(metadata['kme']), nc) if len(spec['dimensions']) == 3 else (nc,)
            dtype = np.int32 if spec['kind'] == 'int' else np.float32
            prepared[name] = require(name, prep[name], dtype, shape)
        args = source.native_driver_arguments(prepared, metadata, options, itimestep=call.itimestep)
        for name in ('qc', 'qi', 'qs', 'nc', 'ni', 'nifa', 'nwfa'):
            if metadata['f_' + name] and args[name] is None:
                raise ValueError(f'RRFS MYNN-EDMF enabled scalar {name} is not allocated')

        initialize = []
        publish = []
        for field in _SURFACE_INPUTS:
            initialize.append((require(field, diag[field], np.float32, (nc,)), prepared[field + '_p']))
        for field, native_name in (('xland', 'xland_p'), ('skintemp', 'tsk_p')):
            initialize.append((require(field, surface[field], np.float32, (nc,)), prepared[native_name]))
        for name, field in _PBL_VOLUME.items():
            array = require(field, diag[field][:nz], np.float32, (nz, nc))
            initialize.append((array, prepared[name][:nz]))
            publish.append((prepared[name][:nz], array))
        for field, name in {**_SURFACE_OUTPUTS, **self.extra_surface}.items():
            dtype = np.int32 if field == 'kpbl' else np.float32
            publish.append((prepared[name], require(field, diag[field], dtype, (nc,))))
        for field, name in (('kzh', 'kzh_p'), ('kzm', 'kzm_p'), ('kzq', 'kzh_p')):
            publish.append((prepared[name][:nz], require(field, diag[field][:nz], np.float32, (nz, nc))))
        for field in _PBL_DIAG_TEND:
            publish.append((prepared[field + '_p'][:nz], require(field, diag[field][:nz], np.float32, (nz, nc))))
        tendency_fields = (*_PBL_COMMON_TEND, 'rqsblten')
        if metadata['f_ni']:
            tendency_fields += ('rniblten',)
        if metadata['f_nc'] and metadata['f_nifa'] and metadata['f_nwfa']:
            tendency_fields += ('rncblten', 'rnifablten', 'rnwfablten')
        for field in tendency_fields:
            publish.append((prepared[field + '_p'][:nz], require(field, tend[field][:nz], np.float32, (nz, nc))))
        zero_volumes = (*_PBL_COMMON_TEND, *_PBL_DIAG_TEND, *_PBL_NUMBER_TEND, 'kzh', 'kzm')
        zero = [prepared[field + '_p'][:nz] for field in zero_volumes]
        zero += [prepared[name] for name in (*_ZERO_SURFACE, *self.extra_surface.values())]
        zero += [prepared['pattern_spp_pbl'][:nz], require('kzq_p', prep['kzq_p'][:nz], np.float32, (nz, nc))]
        lw, sw = (require(name, tend[name][:nz], np.float32, (nz, nc))
                  for name in ('rthratenlw', 'rthratensw'))
        density = require('meshDensity', mesh['meshDensity'], np.float32, (nc,))

        # Changes to prepared module state are owned by the enclosing transaction.
        context.submit_bookkeeping(initialize)
        prepared['kpbl_p'].fill(1)
        context.submit_bookkeeping(((None, array) for array in zero), zero=True)
        shared.radiation(lw, sw, prepared['rthraten_p'][:nz])
        shared.grid_spacing(density, length, prepared['dx_p'])
        result = launch_columns(args, source=source, entrypoint=self.entrypoint)
        context.submit_bookkeeping(publish)
        return result
