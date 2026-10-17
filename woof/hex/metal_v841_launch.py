"""Launch one Hex v8.4.1 Metal kernel on resident arrays.

The ``hexv841_*`` shaders mirror the CUDA v8.4.1 kernels one for one.  Each
launch covers ``count`` threads, as the CUDA launch does, and is skipped when
there is nothing to cover.
"""

from __future__ import annotations

from typing import Any, Mapping


def launch(
    context: Any,
    entrypoint: str,
    count: int,
    arrays: Mapping[str, Any],
    params: Mapping[str, Any],
) -> None:
    if count < 1:
        return
    from woof.metal.generated import kernel_specs

    context.submit_kernel(
        entrypoint, dict(arrays), dict(params), exact_policy=False,
        grid_size=int(count),
        threads_per_group=getattr(
            kernel_specs, f"{entrypoint.upper()}_THREADS_PER_GROUP", 128),
    )


__all__ = ["launch"]
