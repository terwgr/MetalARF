"""Native MPAS v8.4.2 Noah building blocks for resident Metal pools."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch
from . import noah_v842


def load_native_resources(resources, context):
    """Authenticate and upload the native decoder's actual table bytes."""
    xp = _arrays(context)
    class WordUpload:
        @staticmethod
        def asarray(value):
            data = np.asarray(value, dtype=np.uint8)
            if data.size % 4:
                raise ValueError('native Noah table bytes must be word aligned')
            return xp.asarray(data.view(np.uint32))
    if 'rust_decoder' not in resources:
        resources = dict(resources, rust_decoder=packaged_decoder())
    result = noah_v842.load_native_resources(resources, WordUpload)
    result['landuse_tables'] = result['landuse_tables'].view((7*12*100,), dtype=np.float32)
    return result



def packaged_decoder():
    """Resolve the shipped Darwin decoder and verify its source/artifact binding."""
    import hashlib
    import json
    import platform
    from pathlib import Path
    if (platform.system(), platform.machine()) != ('Darwin', 'arm64'):
        raise RuntimeError('packaged native Noah decoder requires Apple Silicon Darwin')
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'libexec/noah-table-decode.json').read_text())
    if manifest['schema'] != 'mpas-noah-table-decoder/v1' or manifest['target'] != 'aarch64-apple-darwin':
        raise ValueError('packaged native Noah decoder manifest differs')
    for name in ('table_decode.rs', 'table_schema.rs'):
        if hashlib.sha256((root / 'native' / name).read_bytes()).hexdigest() != manifest['sources'][name]:
            raise ValueError(f'packaged native Noah decoder source differs: {name}')
    path = root / 'libexec/noah-table-decode'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('packaged native Noah decoder artifact differs')
    return dict(path=str(path), sha256=manifest['sha256'])


def initialize_landuse(floats, integers, tables):
    """Run native land-use initialization on packed solve cells.

    A status of -1 denotes a table index outside the native table's allocated
    category range; the source's corresponding read is undefined.
    """
    n = int(floats.shape[0])
    _require('landuse floats', floats, np.float32, (n, 14))
    _require('landuse integers', integers, np.int32, (n, 9))
    _require('landuse tables', tables, np.float32, (7*12*100,))
    context = _context(floats)
    if _context(integers) is not context or _context(tables) is not context:
        raise ValueError('landuse arrays belong to different Metal contexts')
    status = _arrays(context).zeros((n,), np.int32)
    if n:
        _launch('hexv842_mpas_noah_landuse_f32', n, context,
                dict(floats=floats.view((14*n,)), integers=integers.view((9*n,)),
                     tables=tables, status=status), dict(n=np.int32(n)))
    return status


def initialize_soil(floats, integers, tables):
    """Apply native four-layer cold-start initialization to packed cells.

    Status 140 is the native lower soil-category refusal; -1 marks an
    upper/category table index whose corresponding source read is undefined.
    """
    n = int(floats.shape[0])
    _require('soil floats', floats, np.float32, (n, 15))
    _require('soil integers', integers, np.int32, (n, 4))
    _require('Noah tables', tables, np.uint32, (noah_v842.table_dtype().itemsize//4,))
    context = _context(floats)
    if _context(integers) is not context or _context(tables) is not context:
        raise ValueError('Noah initialization arrays belong to different Metal contexts')
    status = _arrays(context).zeros((n,), np.int32)
    if n:
        _launch('hexv842_mpas_noah_init_f32', n, context,
                dict(floats=floats.view((15*n,)), integers=integers.view((4*n,)),
                     tables=tables, status=status), dict(n=np.int32(n)))
    return status


def run_glacier_columns(floats, integers, tables):
    """Run the native four-layer glacial SFLX on packed columns."""
    n = int(floats.shape[0])
    _require('glacier floats', floats, np.float32, (n, 49))
    _require('glacier integers', integers, np.int32, (n, 4))
    _require('Noah tables', tables, np.uint32, (noah_v842.table_dtype().itemsize//4,))
    context = _context(floats)
    if _context(integers) is not context or _context(tables) is not context:
        raise ValueError('Noah glacier arrays belong to different Metal contexts')
    status = _arrays(context).zeros((n,), np.int32)
    if n:
        _launch('hexv842_mpas_noah_glacier_f32', n, context,
                dict(floats=floats.view((49*n,)), integers=integers.view((4*n,)),
                     tables=tables, status=status), dict(n=np.int32(n)))
    return status


def run_land_columns(floats, integers, tables):
    """Run native four-layer Noah SFLX on packed columns.

    Status 426 is the native fixed-layer guard; -1 identifies a source table
    index outside its allocated category range.
    """
    n = int(floats.shape[0])
    _require('land floats', floats, np.float32, (n, 110))
    _require('land integers', integers, np.int32, (n, 14))
    _require('Noah tables', tables, np.uint32, (noah_v842.table_dtype().itemsize//4,))
    context = _context(floats)
    if _context(integers) is not context or _context(tables) is not context:
        raise ValueError('Noah land arrays belong to different Metal contexts')
    status = _arrays(context).zeros((n,), np.int32)
    if n:
        _launch('hexv842_mpas_noah_land_f32', n, context,
                dict(floats=floats.view((110*n,)), integers=integers.view((14*n,)),
                     tables=tables, status=status), dict(n=np.int32(n)))
    return status


def run_driver_columns(floats, integers, tables, locations):
    """Execute the split native Noah driver on packed four-layer columns.

    Glacier errors stop the continuation because the source's error path
    subsequently reads values that the glacial solver did not initialize.
    Locations contain the caller's native I/J pair for each column.
    """
    n = int(floats.shape[0])
    _require('driver floats', floats, np.float32, (n, 79))
    _require('driver integers', integers, np.int32, (n, 7))
    _require('Noah tables', tables, np.uint32, (noah_v842.table_dtype().itemsize//4,))
    _require('driver locations', locations, np.int32, (n, 2))
    context = _context(floats)
    if any(_context(a) is not context for a in (integers, tables, locations)):
        raise ValueError('Noah driver arrays belong to different Metal contexts')
    # Frame, routes, leaf arrays, and status/placeholder arrays.
    if n * 948 > context.recommended_working_set_size:
        raise MemoryError('Noah driver workspace exceeds Metal recommended working set')
    xp = _arrays(context)
    frames = xp.zeros((n*55,), np.uint32)
    routes = xp.zeros((n,), np.int32)
    land = xp.zeros((n, 110), np.float32)
    landi = xp.zeros((n, 14), np.int32)
    glacier = xp.zeros((n, 49), np.float32)
    glacieri = xp.zeros((n, 4), np.int32)
    status = xp.zeros((n,), np.int32)
    if n:
        placeholder = xp.zeros((n,), np.int32)
        args = dict(floats=floats, integers=integers, tables=tables, frames=frames,
                    routes=routes, land=land, landi=landi, glacier=glacier,
                    glacieri=glacieri, land_status=placeholder,
                    glacier_status=placeholder, locations=locations, status=status)
        _launch('hexv842_mpas_noah_frame_f32', n, context, args,
                dict(n=np.int32(n), mode=np.int32(0)))
        args['land_status'] = run_land_columns(land, landi, tables)
        args['glacier_status'] = run_glacier_columns(glacier, glacieri, tables)
        _launch('hexv842_mpas_noah_frame_f32', n, context, args,
                dict(n=np.int32(n), mode=np.int32(1)))
    return status, routes


def run_mpas_surface(fields, tables, module_locations, *, dt_pbl, itimestep,
                     isurban, isice, config_sfc_albedo, xice_threshold,
                     microphysics_enabled, convection_enabled):
    """Run the native Noah driver on resident level-first MPAS pool views.

    The caller supplies the four-word module-location state and is responsible
    for including it in its transaction and restart state.
    """
    n = int(fields['skintemp'].shape[0])
    nlev = int(fields['t'].shape[0])
    npres = int(fields['pres2_hyd'].shape[0])
    if nlev < 2 or npres < 2:
        raise ValueError('Noah requires two atmospheric and pressure levels')
    context = _context(fields['skintemp'])
    _require('Noah tables', tables, np.uint32, (noah_v842.table_dtype().itemsize//4,))
    _require('Noah module locations', module_locations, np.int32, (4,))
    if _context(tables) is not context or _context(module_locations) is not context:
        raise ValueError('Noah tables and location state must share the pool context')
    values = {}
    for name, kind in noah_v842._MPAS_INPUT_FIELDS.items():
        enabled = (microphysics_enabled if name in ('rainncv','sr')
                   else convection_enabled)
        value = fields['skintemp'] if kind == 'optional_surface' and not enabled else fields[name]
        shape = ((4,n) if kind == 'soil' else (nlev,n) if kind == 'atmosphere'
                 else (npres,n) if kind == 'pressure' else (n,))
        dtype = np.int32 if kind == 'integer_surface' else np.float32
        _require(name, value, dtype, shape)
        if _context(value) is not context:
            raise ValueError('Noah pool fields must share one Metal context')
        values[name] = value
    # Packed driver data and locations plus the driver frame/solver workspace.
    if n * 1300 > context.recommended_working_set_size:
        raise MemoryError('Noah surface workspace exceeds Metal recommended working set')
    xp = _arrays(context)
    floats = xp.empty((n,79), np.float32)
    integers = xp.empty((n,7), np.int32)
    locations = xp.empty((n,2), np.int32)
    if n:
        _launch('hexv842_mpas_noah_from_f32',n,context,
                dict(floats=floats,integers=integers,locations=locations,**values),
                dict(n=np.int32(n),dt=np.float32(dt_pbl),xice_threshold=np.float32(xice_threshold),
                     micro=np.int32(microphysics_enabled),convection=np.int32(convection_enabled),
                     isurban=np.int32(isurban),isice=np.int32(isice),step=np.int32(itimestep),
                     usemonalb=np.int32(config_sfc_albedo)))
    status, routes = run_driver_columns(floats,integers,tables,locations)
    if n:
        _launch('hexv842_mpas_noah_locations_i32',1,context,
                dict(locations=locations,routes=routes,state=module_locations),dict(n=np.int32(n)))
        _launch('hexv842_mpas_noah_to_f32',n,context,
                dict(floats=floats,status=status,**{k:values[k] for k in noah_v842._MPAS_OUTPUT_FIELDS}),
                dict(n=np.int32(n)))
    return status


class MetalMpasNoah:
    """Native Noah slot with caller-supplied tables and module-location state."""

    bind_pools = staticmethod(noah_v842.bind_native_pools)

    def __init__(self, tables, module_locations):
        _require('Noah tables', tables, np.uint32, (noah_v842.table_dtype().itemsize//4,))
        _require('Noah module locations', module_locations, np.int32, (4,))
        if _context(tables) is not _context(module_locations):
            raise ValueError('Noah tables and location state must share one context')
        self.tables, self.module_locations = tables, module_locations

    def restart_state(self):
        return {'module_locations': self.module_locations.to_numpy()}

    def restore_restart_state(self, state):
        locations = np.asarray(state['module_locations'])
        if locations.shape != (4,) or locations.dtype != np.int32:
            raise ValueError('Noah restart module locations must be four int32 words')
        self.module_locations[...] = locations

    def launch_slot(self, call):
        if (call.slot, call.scheme) != ('land_surface', 'sf_noah'):
            raise ValueError('MPAS Metal land slot requires sf_noah')
        arguments = dict(call.pools)
        bound_tables = arguments.pop('tables', self.tables)
        if bound_tables is not self.tables:
            raise ValueError('Noah slot tables differ from the session tables')
        if not arguments.pop('level_first', True) or arguments.pop('start', 0) != 0:
            raise ValueError('Noah Metal pools require the full level-first solve extent')
        count = arguments.pop('count', arguments['fields']['skintemp'].size)
        if count != arguments['fields']['skintemp'].size:
            raise ValueError('Noah Metal pool count differs from solve extent')
        arguments.update(itimestep=call.itimestep,
                         config_sfc_albedo=call.options['config_sfc_albedo'],
                         microphysics_enabled=call.options['config_microp_scheme'] != 'off',
                         convection_enabled=call.options['config_convection_scheme'] != 'off')
        status = run_mpas_surface(tables=self.tables, module_locations=self.module_locations,
                                  **arguments)
        errors = status.to_numpy()
        if np.any(errors):
            error = int(errors[errors != 0][0])
            if error == -1:
                raise RuntimeError('MPAS Noah table index is undefined in the native source')
            raise RuntimeError(f'MPAS Noah native fatal condition at source line {error}')
        return status
