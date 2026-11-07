"""Case readers retained for :mod:`tilestream.find_storm`.

Weather-field rendering is provided by :mod:`tilestream.render_case_rust`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _read(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


