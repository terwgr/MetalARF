"""Metal eclipse obscuration and shortwave dimming (woof.eclipse_windows)."""
from __future__ import annotations

import numpy as np

from .array import MetalArray, MetalArrayModule


def _flat(a):
    return a.view((a.size,))


def _submit(name, resources, parameters, count):
    context = next(iter(resources.values())).buffer.context
    if any(not isinstance(a, MetalArray) or a.buffer.context is not context
           or not a.flags.c_contiguous for a in resources.values()):
        raise TypeError("eclipse resources must be contiguous MetalArrays on one context")
    if count:
        context.submit_kernel(name, {k: _flat(a) for k, a in resources.items()},
                              parameters, exact_policy=False, grid_size=count,
                              threads_per_group=256)


class MetalEclipseGeometry:
    """The CUDA ``eclipse_obscuration`` field on resident Metal arrays."""

    def __init__(self, latitude_deg, longitude_deg, context):
        xp = MetalArrayModule(context)
        host = lambda v: v.to_numpy() if isinstance(v, MetalArray) else np.asarray(v)
        self._xp = xp
        self.latitude = xp.asarray(np.ascontiguousarray(host(latitude_deg), np.float32))
        self.longitude = xp.asarray(np.ascontiguousarray(host(longitude_deg), np.float32))
        if self.latitude.shape != self.longitude.shape:
            raise ValueError("eclipse latitude and longitude shapes must match")
        self.obscuration = xp.zeros(self.latitude.shape, np.float32)
        self.mask = xp.zeros(self.latitude.shape, np.int32)

    def at(self, valid_time):
        from woof.eclipse_windows import besselian_row_at
        row, hour = besselian_row_at(valid_time)
        if row is None:
            self.obscuration.fill(0)
            self.mask.fill(0)
            return self.obscuration
        words = np.asarray(tuple(row) + (hour,), np.float64)
        _submit("eclipse_obscuration",
                {"lat": self.latitude, "lon": self.longitude,
                 "besselian": self._xp.asarray(words), "obscur": self.obscuration,
                 "mask": self.mask},
                {"n": self.latitude.size}, self.latitude.size)
        return self.obscuration


def scale(src, obscuration, dst):
    """dst = src * (1 - obscuration), one plane or broadcast over levels."""
    _submit("eclipse_scale", {"src": src, "obscur": obscuration, "dst": dst},
            {"n": dst.size, "plane": obscuration.size}, dst.size)


def dim_heating(base, obscuration, heating, total, changed):
    """Dim the shortwave heating and add its change to the total heating."""
    _submit("eclipse_heating", {"base": base, "obscur": obscuration,
                                "heating": heating, "total": total,
                                "changed": changed},
            {"n": heating.size, "plane": obscuration.size}, heating.size)


def select(changed, src, dst):
    """dst = src where the shortwave changed."""
    _submit("eclipse_select", {"changed": changed, "src": src, "dst": dst},
            {"n": dst.size}, dst.size)
