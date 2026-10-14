"""Backend ownership and explicit readback for the supplied case runners.

Metal runs use the original DomainState and native core dispatch. CPU work
is restricted to initial-condition construction, case metrics and output.
"""
from __future__ import annotations

from types import SimpleNamespace
import os
import numpy as np


def is_metal(state):
    return bool(getattr(state, "_metal_setup_state", False))


def as_host(value):
    if isinstance(value, np.ndarray) or np.isscalar(value):
        return np.asarray(value)
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    import cupy
    return cupy.asnumpy(value)


class _ReadbackModule:
    """NumPy metric operations, never an array module for model integration."""
    asnumpy = staticmethod(as_host)

    def __getattr__(self, name):
        return getattr(np, name)


def metric_module(state):
    if is_metal(state) or getattr(state, "_case_host_readback", False):
        return _ReadbackModule()
    import cupy
    return cupy


class HostCaseState(SimpleNamespace):
    def existing_scratch(self, name):
        return self._scratch.get(name)

    def height_half(self):
        return self._height_half

    def total_theta(self):
        base = self.thb if self.thb.ndim == 3 else self.thb[:, None, None]
        return base + self.thp


def host_state(state, names):
    """Copy only metric operands; keep the live forecast state on Metal."""
    values = {name: as_host(getattr(state, name)) for name in names
              if getattr(state, name, None) is not None}
    for name in ("thb", "phb", "mub2d", "dnw", "fnm", "fnp", "c1h", "c2h"):
        if name not in values:
            values[name] = as_host(getattr(state, name))
    for name in ("tke", "e_sgs"):
        value = getattr(state, name, None)
        values[name] = None if value is None else as_host(value)
    scratch = {}
    for name in ("smag_khv", "mp_rainnc"):
        value = state.existing_scratch(name)
        if value is not None:
            scratch[name] = as_host(value)
    return HostCaseState(**values, cfg=state._cfg if hasattr(state, "_cfg") else None,
                         _case_host_readback=True, _scratch=scratch,
                         _height_half=state.height_half())


class CaseBackend:
    """One explicitly selected backend context for a complete case run."""
    def __init__(self, backend):
        if backend not in ("cuda", "metal"):
            raise ValueError(f"Unknown case backend {backend!r}")
        self.backend = backend
        self.arrays = self.context = None

    def __enter__(self):
        if self.backend == "metal":
            from woof.metal.array import MetalArrayModule
            from woof.metal.runtime import MetalContext
            self.context = MetalContext()
            if os.environ.get("WOOF_METAL_VERIFY_SYNC") == "1":
                # Verification can fence every typed-kernel batch instead of
                # repeating full cases to detect asynchronous ordering errors.
                self.context.batch_kernel_calls = False
                submit = self.context._submit_kernel_batch
                def synchronous_submit(*args, **kwargs):
                    result = submit(*args, **kwargs)
                    result.wait()
                    return result
                self.context._submit_kernel_batch = synchronous_submit
            self.arrays = MetalArrayModule(self.context)
        return self

    def __exit__(self, *_):
        if self.context is not None:
            self.context.close()
            self.context = self.arrays = None
