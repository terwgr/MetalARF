"""Public YSU family adapter.

The ordinary non-BEP, non-topographic entrypoint is implemented by
``woof.metal.ysu`` and registered in the shared typed Metal ABI. BEP/BEM and
topographic variants remain explicitly unavailable until their additional
source branches and math dependencies are implemented.
"""

from __future__ import annotations

from .ysu import submit_ysu_column


def _unsupported(entrypoint: str) -> None:
    raise NotImplementedError(
        f"Metal {entrypoint} is unavailable; only ordinary non-BEP, "
        "non-topographic YSU is currently implemented. No CPU fallback is "
        "provided.")


def submit_ysu_column_bep(*args, **kwargs):
    """Fail closed for the BEP/BEM column variant."""
    _unsupported("ysu_column_bep")


def submit_ysu_column_topo(*args, **kwargs):
    """Fail closed for the topo-wind column variant."""
    _unsupported("ysu_column_topo")
