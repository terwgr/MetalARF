"""Exercise real resolver admission and the constructor's generation guard."""
from woof.hex.physics_suites import resolve_physics
from woof.hex.physics_backend_admission import resolve_backend
from woof.hex.metal_generic_physics_v841 import MetalGenericPhysicsBackendV841

base={'icloud':2,'config_gwdo_scheme':'off','config_convection_scheme':'off'}
def adapter(overrides):
    resolved=resolve_physics('arwen_mpas_column',overrides,nominal_dx_m=12000)
    return resolve_backend(resolved.backend_row).metal_adapter_module
assert adapter(base)=='woof.hex.metal_generic_physics_v841'
assert adapter({**base,'rrtmg_cloud_optics_form':'noaa_wrf39'}) is None
try:
    MetalGenericPhysicsBackendV841(constructor_values={'generation':{'rrtmg_cloud_optics_form':'noaa_wrf39'}},
                                   prep_geometry=None,array_module=None)
except NotImplementedError:
    pass
else:
    raise AssertionError('constructor dropped a generation override')
print('real resolver: supported baseline admitted, generation refused; constructor refuses before context')
