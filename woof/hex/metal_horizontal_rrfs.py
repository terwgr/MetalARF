"""Retained RRFS horizontal operators on resident Metal arrays."""
import numpy as np
from .cuda_horizontal_rrfs import CudaHorizontalRRFS
from .metal_horizontal_v841 import MetalHorizontalV841
from .metal_rrfs_arguments import ARGUMENTS
from .metal_rrfs_launch import Arrays, launch


class MetalHorizontalRRFS(MetalHorizontalV841):
    def __init__(self, *args):
        super().__init__(*args)
        self.cp = Arrays(self.metal)

    def _launch(self, kernel, total, arrays, params=None):
        if kernel in ARGUMENTS['horizontal']:
            if params is None:
                return self._launch_v841(kernel,total,arrays)
            values = {**arrays, **params}
            arguments = tuple(values[name] for name, _, _ in ARGUMENTS['horizontal'][kernel])
            if kernel == 'pressure_gradient_v841_f32':
                total = self.nlev * self.nedges
            return self._launch_v841(kernel, total, arguments)
        if kernel == 'mixing_scaling_f32' and params is None:
            cells, density, nedges, enabled, del2, del4 = arrays
            arrays = dict(cells_on_edge=cells, mesh_density=density, del2=del2, del4=del4)
            params = dict(nedges=int(nedges), scale_with_mesh=int(enabled))
        return super()._launch(kernel, total, arrays, params)

    def _launch_v841(self, symbol, count, arguments):
        launch('horizontal', symbol, count, arguments)

    # This schedule contains only allocation, scalar staging and kernel
    # dispatch. The overrides above keep every field on Metal.
    compute_dry_mixing_tendencies_v841 = CudaHorizontalRRFS.compute_dry_mixing_tendencies_v841
