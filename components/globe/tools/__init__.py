"""The driver scripts, outside the wheel.

They are instruments rather than library code, so they stay in the repository
the way the neighbouring distribution keeps its own drivers, and are reached
from the suite through `tests/conftest.py`.

This file exists for one reason: the engine also ships a package named
`tools`, and without a regular package here Python resolves `from tools import
...` to the engine's directory of that name no matter where this repository
sits on the path. A namespace portion never wins against a regular package.
"""
