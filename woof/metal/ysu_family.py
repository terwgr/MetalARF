"""Public YSU family adapter.

``woof.metal.ysu`` implements the ordinary, topo_wind and BEP/BEM column
variants through the one registered ``ysu_column`` entry point.
"""

from __future__ import annotations

from .ysu import submit_ysu_column

__all__ = ["submit_ysu_column"]
