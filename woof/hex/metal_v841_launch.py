"""Launch one Hex v8.4.1 Metal kernel on resident arrays.

The ``hexv841_*`` shaders mirror the CUDA v8.4.1 kernels one for one.  Each
launch covers ``count`` threads, as the CUDA launch does, and is skipped when
there is nothing to cover.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Mapping

#: Called as ``POST_LAUNCH(entrypoint, arrays)`` after every launch while a
#: regional run is attached, as the CUDA kernel cache's ``post_launch``: the
#: regional garbage discipline restores the pool value in each padded array.
POST_LAUNCH: Callable[[str, Iterable[Any]], None] | None = None
_in_hook = False


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
    global _in_hook
    hook = POST_LAUNCH
    if hook is not None and not _in_hook:
        _in_hook = True
        try:
            hook(entrypoint, arrays.values())
        finally:
            _in_hook = False


@contextmanager
def suspended() -> Iterator[None]:
    """Run the enclosed launches without the post-launch hook.

    The regional garbage discipline belongs to the dycore's arrays.  The
    column physics runs its padded garbage column on duplicated, well-posed
    values (``pad_regional_physics_column``, the preparation's lend), which a
    pool-value restore would undo.
    """

    global POST_LAUNCH
    hook, POST_LAUNCH = POST_LAUNCH, None
    try:
        yield
    finally:
        POST_LAUNCH = hook


__all__ = ["launch", "suspended"]
