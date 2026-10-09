"""Loader for the semi-Lagrangian CUDA translation unit.

Deliberately separate from :mod:`woof.globe.core.kernels`.  That loader prepends a
physics preamble and ``common.cuh`` to every source it reads and its assembled
strings are digested into the kernel manifest that pins the physics core; a
new file there would sit inside a hash surface this lane has no business
touching.  This module compiles one self-contained source with the same
explicit UTF-8 read and the same one-module-per-process caching, and nothing
else in the tree changes shape because of it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SOURCE = Path(__file__).with_name("kernels.cu")

#: Explicit, because a locale-decoded read of a source is how the same
#: checkout produced two different compiled strings on two machines.
_ENCODING = "utf-8"

_OPTIONS = ("-std=c++17",)


def module_source() -> str:
    """The exact string handed to nvrtc."""
    return _SOURCE.read_text(encoding=_ENCODING)


@lru_cache(maxsize=1)
def load_module():
    import cupy as cp

    module = cp.RawModule(code=module_source(), options=_OPTIONS)
    module.compile()
    return module


@lru_cache(maxsize=None)
def get_kernel(name: str):
    """One stable CuPy function wrapper per entry point."""
    return load_module().get_function(name)


def kernel_attributes(name: str) -> dict:
    """Register and local-memory footprint of one entry point.

    The measurement that decided the kernel's shape: the fully unrolled
    variant used 168 registers with no spill and read 1.75x slow, and the
    un-unrolled variant spilled.  Reporting the numbers beside a timing is
    what keeps a future edit from silently walking back into either.
    """
    fn = get_kernel(name)
    attrs = dict(fn.attributes)
    return {
        "num_regs": int(attrs.get("num_regs", -1)),
        "local_size_bytes": int(attrs.get("local_size_bytes", -1)),
        "shared_size_bytes": int(attrs.get("shared_size_bytes", -1)),
        "max_threads_per_block": int(attrs.get("max_threads_per_block", -1)),
    }
