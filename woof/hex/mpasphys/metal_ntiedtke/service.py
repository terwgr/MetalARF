"""Native New Tiedtke convection service for resident Metal physics."""

from woof.hex.mpasphys.ntiedtke_v842 import SCHEME, bind_native_pools
from woof.metal.array import MetalArrayModule
from .executor import launch

def update_convection_step1(call):
    # MPAS's per-step branch is empty for New Tiedtke (it is active for KF).
    if call.options['config_convection_scheme'] != SCHEME:
        raise ValueError('New Tiedtke convection hook requires its native scheme')


class MetalMpasNewTiedtke:
    """The executor owns scratch and the native pools own held state."""
    bind_pools = staticmethod(bind_native_pools)

    def launch_slot(self, call):
        if (call.slot, call.scheme, str(call.backend)) != ('convection', SCHEME, 'metal'):
            raise ValueError('New Tiedtke requires its Metal convection slot')
        bound = call.pools['ntiedtke_v842']
        b = bound['binding']
        module = MetalArrayModule(b.fields['t3d'].buffer.context)
        result = launch(module, b, bound['constants'], itimestep=call.itimestep)
        return result

    def callbacks(self):
        return {'update_convection_step1': update_convection_step1}

    def restart_state(self):
        return {}

    def restore_restart_state(self, state):
        if state:
            raise ValueError('New Tiedtke has no scheme-private restart state')
