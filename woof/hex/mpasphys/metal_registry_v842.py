"""Private Metal dispatch for the source-backed v8.4.2 slots implemented so far."""
from .driver_v842 import Backend, SlotRegistry, CORRECTIONS_V842, resolve_namelist
from .route_table import DEFAULT_ROW, route_row
from ..physics_suites import load_table


class MetalSlotRegistry(SlotRegistry):
    def declared(self, slot, scheme, backend):
        if backend == 'metal':
            return self._corrections.get((slot, scheme, backend), frozenset())
        return super().declared(slot, scheme, backend)

    def register_metal(self, slot, scheme, callback, *, corrections=()):
        spec = load_table()['mpasphys_v842']['slots']
        if slot not in spec or scheme not in spec[slot]['schemes'] or scheme == 'off':
            raise ValueError(f'unknown MPAS slot/scheme {slot}/{scheme}')
        if not callable(callback):
            raise ValueError('Metal slot callback must be callable')
        key = slot, scheme, 'metal'
        if key in self._callbacks:
            raise ValueError(f'physics slot already registered: {key}')
        corrections = frozenset(corrections)
        unknown = sorted(corrections - set(CORRECTIONS_V842))
        if unknown:
            raise ValueError(f'slot {key} declares unknown corrections {unknown}')
        self._callbacks[key] = callback
        self._corrections[key] = corrections


def metal_slot_definitions(slot_services, config=None):
    """One inventory for Metal registration and correction settlement."""
    native_v842 = getattr(config, 'table_key', DEFAULT_ROW) == DEFAULT_ROW
    definitions = {}
    if native_v842:
        from .sf_mynn_v842 import CORRECT_MODE_SOURCE as surface_corrections
        definitions.update({
            ('cloudiness', 'cld_fraction'): ('cloudiness', frozenset()),
            ('surface_layer', 'sf_mynn'): ('surface_layer', frozenset(surface_corrections)),
            ('gwdo', 'bl_ysu_gwdo'): ('gwdo', frozenset()),
        })
    spec = route_row(config)['slots']
    if not native_v842:
        from .rrfs.metal_cloudiness import CLOUD_SCHEMES
        descriptors = spec.get('cloudiness', {}).get('cuda_entries', {})
        definitions.update({('cloudiness', scheme): ('cloudiness', frozenset())
                            for scheme in CLOUD_SCHEMES
                            if descriptors.get(scheme, {}).get('module') == 'rrfs.radiation_rrfs'})
    for key, service in slot_services.items():
        slot, scheme = key
        row = spec.get(slot, {})
        schemes = row.get('schemes', ())
        if not native_v842:
            schemes = (*schemes, *row.get('internal_schemes', ()))
        if scheme == 'off' or scheme not in schemes:
            raise ValueError(f'unknown MPAS slot/scheme {slot}/{scheme}')
        if key in definitions:
            raise ValueError(f'physics slot already registered: {key}')
        if not callable(getattr(service, 'launch_slot', None)):
            raise ValueError(f'Metal slot has no executable service: {key}')
        corrections = frozenset(getattr(service, 'corrections', ()))
        unknown = sorted(corrections - (set(CORRECTIONS_V842) if native_v842 else set()))
        if unknown:
            raise ValueError(f'slot {key} declares unknown corrections {unknown}')
        definitions[key] = (service, corrections)
    return definitions


def require_metal_native_profile(config):
    if getattr(config, 'table_key', DEFAULT_ROW) != DEFAULT_ROW:
        from .rrfs.driver_rrfs import resolve_namelist as resolve_rrfs

        if config.table_key not in ('mpasphys_rrfs', 'mpasphys_rrfs_44c0849') or config.suite != 'hrrrv5':
            raise ValueError('native Metal RRFS requires the supported HRRRv5 source row')
        expected = resolve_rrfs("&physics\nconfig_physics_suite='hrrrv5'\n/\n",
                                table_key=config.table_key)
        if config.selectors != expected.selectors:
            raise ValueError('native Metal RRFS requires the HRRRv5 suite selectors')
        unsupported = ('config_do_DAcycling', 'config_ngw_scheme', 'config_mynn_enh_mix',
                       'config_spp_lsm', 'config_sppt', 'config_sst_update',
                       'config_sstdiurn_update', 'config_deepsoiltemp_update', 'config_lakemodel')
        options = config.execution_options
        enabled = [key for key in unsupported if options.get(key)]
        if enabled:
            raise ValueError('native Metal RRFS does not support ' + ', '.join(enabled))
        return
    if config.suite == 'none':
        expected = resolve_namelist("&physics\nconfig_physics_suite='none'\n/\n")
        if config.selectors != expected.selectors:
            raise ValueError('native Metal none requires the default suite selectors')
        return
    if config.suite == 'mesoscale_reference':
        expected = resolve_namelist("&physics\nconfig_physics_suite='mesoscale_reference'\n/\n")
        if config.selectors != expected.selectors:
            raise ValueError('native Metal mesoscale reference requires the default suite selectors')
        return
    default = resolve_namelist("&physics\nconfig_physics_suite='convection_permitting'\n/\n")
    expected = dict(default.selectors)
    ysu = config.selectors['pbl'] == 'bl_ysu'
    if ysu:
        from .sfclay_mm5_v842 import SCHEMES

        surface = config.selectors['surface_layer']
        if surface not in SCHEMES:
            raise ValueError('native Metal YSU requires an MM5 surface layer')
        expected.update(pbl='bl_ysu', surface_layer=surface)
    if not ysu and config.selectors['convection'] == 'off':
        expected['convection'] = 'off'
    if not ysu and config.selectors['gwdo'] == 'off':
        expected['gwdo'] = 'off'
    microphysics = config.selectors['microphysics']
    if not ysu and microphysics in ('mp_thompson_aerosols', 'mp_wsm6', 'mp_kessler'):
        expected['microphysics'] = microphysics
    if config.suite != default.suite or config.selectors != expected:
        raise ValueError('native Metal requires a supported convection-permitting profile')


def resolve_metal_namelist(text, *, slot_services, table_key=DEFAULT_ROW, **kwargs):
    """Settle the default suite against the actual Metal slot declarations."""
    if any(name in kwargs for name in ('backends', 'default_backend', 'declarations', '_allow_metal')):
        raise ValueError('Metal slot backends and declarations come from the supplied services')
    if kwargs.get('correct_skip') is not None:
        kwargs['correct_skip'] = tuple(kwargs['correct_skip'])
    if table_key != DEFAULT_ROW:
        from dataclasses import replace
        from types import MappingProxyType
        from .driver_v842 import parse_fortran_namelist
        from .rrfs.driver_rrfs import resolve_namelist as resolve_rrfs

        preliminary = resolve_rrfs(text, table_key=table_key, **kwargs)
        require_metal_native_profile(preliminary)
        row = route_row(preliminary)
        group = parse_fortran_namelist(text).get(row['backend_namelist_group'], {})
        if any(service['backend_selector'] in group for service in row.get('services', {}).values()):
            raise ValueError('native Metal RRFS chooses the driver service backend')
        active = {slot: scheme for slot, scheme in preliminary.selectors.items()
                  if preliminary.active(slot) and not preliminary.native_embedded(slot)}
        definitions = metal_slot_definitions(slot_services, preliminary)
        missing = sorted(active.items() - definitions.keys())
        if missing:
            raise ValueError(f'Metal slot implementations are missing: {missing}')
        result = resolve_rrfs(text, table_key=table_key, default_backend=Backend.NATIVE_CPU,
                              backends=dict.fromkeys(active, Backend.METAL), **kwargs)
        return replace(result, service_backends=MappingProxyType({'surface_update': Backend.METAL}))
    preliminary = resolve_namelist(text, _allow_metal=True, **kwargs)
    require_metal_native_profile(preliminary)
    if any(preliminary.backends[slot] == Backend.OFF and scheme != 'off'
           for slot, scheme in preliminary.selectors.items()):
        raise ValueError('active Metal slots cannot be disabled by a backend override')
    definitions = metal_slot_definitions(slot_services)
    active = {slot: scheme for slot, scheme in preliminary.selectors.items() if scheme != 'off'}
    missing = sorted((slot, scheme) for slot, scheme in active.items() if (slot, scheme) not in definitions)
    if missing:
        raise ValueError(f'Metal slot implementations are missing: {missing}')
    def declared(slot, scheme, backend):
        return definitions.get((slot, scheme), (None, frozenset()))[1] if backend == Backend.METAL else frozenset()
    return resolve_namelist(text, backends={slot: Backend.METAL for slot in active},
                            declarations=declared, _allow_metal=True, **kwargs)
