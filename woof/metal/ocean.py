"""Metal surface-ocean mixed layer (woof.core.surface_ocean, option 1)."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def _submit(name, resources, parameters, count):
    context = next(iter(resources.values())).buffer.context
    if any(not isinstance(a, MetalArray) or a.buffer.context is not context
           or not a.flags.c_contiguous or a.dtype != np.float32
           for a in resources.values()):
        raise TypeError("surface ocean resources must be contiguous float32 MetalArrays on one context")
    if count:
        context.submit_kernel(name, {k: a.view((a.size,)) for k, a in resources.items()},
                              parameters, exact_policy=False, grid_size=count,
                              threads_per_group=128)


def oml_init(state, tsk, hml0, start):
    _submit("ocean_oml_init", {**state, "tsk": tsk},
            {"hml0": float(hml0), "start": int(bool(start)), "n": tsk.size}, tsk.size)


def oml_step(state, tsk, forcing, *, coupled_ice, dt, gamma, relaxation_time,
             rhowater, cwater, stbolt, gravity):
    _submit("ocean_oml_step", {**state, "tsk": tsk, **forcing},
            {"coupled_ice": int(coupled_ice), "dt": float(dt), "gam": float(gamma),
             "relaxation": float(relaxation_time), "rhowater": float(rhowater),
             "cwater": float(cwater), "stbolt": float(stbolt), "gravity": float(gravity),
             "n": tsk.size}, tsk.size)


def sst_diagnostics(t, t0, land, lake, sst, change):
    _submit("ocean_sst_diagnostics",
            {"t": t, "t0": t0, "land": land, "lake": lake, "sst": sst, "change": change},
            {"n": t.size}, t.size)
