"""Form the held start frame's native microphysics reflectivity on Metal."""
import numpy as np

from .thompson_refl_start_v842 import StartFrameReflectivity, SCHEMES, RGAS, CP_DRY, P0
from .wsm6_refl_start_v842 import StartFrameReflectivity as Wsm6StartFrame


class MetalStartFrameReflectivity(StartFrameReflectivity):
    def __init__(self, snapshot, *, scheme, context):
        self.context = context
        super().__init__(snapshot, scheme=scheme, device=0)

    def _run(self):
        from woof.metal.array import MetalArrayModule
        from .metal_thompson_column_v842 import launch_radar
        state, zgrid, _ = self._args
        try:
            xp = MetalArrayModule(self.context)
            nz, nc = state['qv'].shape
            z = (np.asarray(zgrid, np.float32) if zgrid is not None else
                 np.broadcast_to(np.arange(nz + 1, dtype=np.float32)[:, None] * np.float32(500), (nz + 1, nc)))
            fields = {name: xp.asarray(np.ascontiguousarray(state[name], dtype=np.float32))
                      for name in self.NEED if name != 'theta'}
            fields.update(th=xp.asarray(np.ascontiguousarray(state['theta'], dtype=np.float32)),
                          p=xp.asarray(np.ascontiguousarray(state['p'], dtype=np.float32)),
                          pii=xp.asarray(np.ascontiguousarray((state['p'].astype(np.float64) / P0) ** (RGAS / CP_DRY), dtype=np.float32)))
            out = {name: xp.empty((nz, nc) if name == 'refl10cm' else (nc,), np.float32)
                   for name in ('refl10cm', 'refl10cm_max', 'refl10cm_1km')}
            out['refl10cm_1km_max'] = xp.asarray(np.full((nc,), -35., np.float32))
            if hasattr(self, 'constants'):
                from .metal_wsm6_radar_v842 import launch_radar as wsm6_radar
                fields['zgrid'] = xp.asarray(np.ascontiguousarray(z))
                wsm6_radar(fields, out, xp.asarray(self.constants))
            else:
                fields.update(z=xp.asarray(np.ascontiguousarray(z[:-1])),
                              dz=xp.asarray(np.ascontiguousarray(z[1:] - z[:-1])))
                launch_radar(fields, out)
            self.result = {name: value.to_numpy() for name, value in out.items()}
        except BaseException as error:
            self.error = error


class MetalWsm6StartFrameReflectivity(Wsm6StartFrame):
    def __init__(self, snapshot, *, scheme, context, constants):
        self.context = context
        super().__init__(snapshot, scheme=scheme, device=0, constants=constants)

    _run = MetalStartFrameReflectivity._run


def start_frame_reflectivity(configs, snapshot, *, context, constants=None):
    scheme = str(configs.get('config_microp_scheme', '')).strip()
    if scheme in SCHEMES:
        return MetalStartFrameReflectivity(snapshot, scheme=scheme, context=context)
    if scheme == 'mp_wsm6':
        from .wsm6_refl_start_v842 import host_constants
        return MetalWsm6StartFrameReflectivity(snapshot, scheme=scheme, context=context,
                                               constants=host_constants(constants))
    return None
