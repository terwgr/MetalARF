"""The component doors: ``woof hex`` and ``woof global``.

The regional engine is the core of this distribution.  Two components ride
with it as subpackages: the hex model (``woof.hex``, an MPAS-style
unstructured-mesh port, preview) and the global model (``woof.globe``,
preview; ``global`` is a Python keyword, so the package cannot carry that
name, but the command word can).  Each keeps its own complete command-line
parser, so the front door hands the rest of the command line to it
unchanged instead of re-declaring its options here.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys

#: word -> (entry module, entry function, label, one-line help)
COMPONENTS = {
    "hex": ("woof.hex.cli", "main", "hex model (preview)",
            "the hex (MPAS-style mesh) model, preview: woof hex --help"),
    "global": ("woof.globe.cli", "main", "global model (preview)",
               "the global model, preview: woof global --help"),
}


def component_available(word: str) -> bool:
    module = COMPONENTS[word][0]
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def register_component_doors(sub) -> None:
    """List each component in the front door's help; dispatch happens earlier."""
    for word, (_module, _func, _label, text) in COMPONENTS.items():
        sub.add_parser(word, help=text, add_help=False)


def dispatch_component(tokens: list[str]) -> int | None:
    """Run ``tokens`` through a component when its first word names one.

    Returns the component's exit code, or None when the words are not a
    component command and the regional parser should take them.
    """
    if not tokens or tokens[0] not in COMPONENTS:
        return None
    word = tokens[0]
    module_name, func_name, label, _text = COMPONENTS[word]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and module_name.startswith(error.name):
            print(f"woof {word}: the {label} is not part of this install "
                  f"({module_name} is missing). Reinstall recast-woof.",
                  file=sys.stderr)
            return 2
        raise
    entry = getattr(module, func_name)
    print(f"woof {word}: the {label}; its config and checkpoint formats may "
          "change in a minor release.", file=sys.stderr)
    prog = f"woof {word}"
    saved = sys.argv[0]
    sys.argv[0] = prog
    try:
        result = entry(tokens[1:])
    except SystemExit as stop:
        code = stop.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv[0] = saved
    return 0 if result is None else int(result)
