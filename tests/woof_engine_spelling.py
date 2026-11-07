"""The engine's spelling of an emitter's comment header, for byte controls pinned at engine commits.

Recast WOOF's config emitters (``woof domain``, ``woof import-namelist``) name WOOF in the comment header
they write ("# woof experiment translated from WRF namelists by `woof import-namelist`", "Emitted by
`woof domain`", "woof/experiment.py derives them"), where the historical controls these tests pin were
emitted by the engine and say gpuwm (the command, the module path, the home folder ``~/.gpuwm``).  Only comment
lines move; every setting line is still compared byte for byte.
"""
from __future__ import annotations

import re

_COMMENT = re.compile(r"(?m)^#.*$")
_PRODUCT = re.compile(r"(?<![\w-])woof\b")


def engine_spelling(text: str) -> str:
    return _COMMENT.sub(lambda line: _PRODUCT.sub("gpuwm", line.group(0)), text)
