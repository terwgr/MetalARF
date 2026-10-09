"""The distribution's identity, read back rather than restated.

Kept in its own module so that asking what version is installed does not
import the model.  ``woof/globe/__init__.py`` pulls in the dynamics, the
regional bridge and the semi-implicit solvers at module scope, all of which
need the engine; a packaging gate that only wants the number must not need a
GPU stack to get it.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

__all__ = ["DISTRIBUTION_NAME", "IMPORT_NAME", "CONSOLE_SCRIPT", "__version__"]

#: The name this package is published under.
DISTRIBUTION_NAME = "recast-woof"

#: The name a user types in an import line.  It is the model's own name and
#: not the engine's: putting the engine's name in every user import line
#: would name the wrong owner for code the engine does not carry.
IMPORT_NAME = "woof.globe"

#: The one console script, and the prog name every parser and every refusal
#: sentence in this package uses.
CONSOLE_SCRIPT = "woof global"

# Read out of installed metadata.  pyproject's [project].version is the single
# place the number is written; a second literal is a promise to update two
# files at every cut, and the engine already broke that promise once.
try:
    __version__ = _distribution_version(DISTRIBUTION_NAME)
except PackageNotFoundError:  # pragma: no cover - an uninstalled source tree
    # Says so rather than inventing a number.
    __version__ = "0+unknown"
