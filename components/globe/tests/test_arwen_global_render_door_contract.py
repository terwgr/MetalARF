"""The render door calls the engine's renderer with keywords it declares.

THE BREAKAGE THIS PREVENTS, measured 2026-09-07.  ``render_door`` passed
``series=False`` to :func:`woof.render.render_wrfouts_rust` from the day the
door was written, and no engine has ever declared that parameter: not the
2.7.0 release candidate, not the tree the model is developed on.  So every
draw died at the call with a ``TypeError`` AFTER the tapes had been exported
and the export receipt written, which is the most expensive place to fail.
Nothing caught it because the demo pictures were drawn through the engine's
own ``woof render``, so the package's headline picture door never drew a
single picture in anger.

The check reads the CALL, not a copy of it: the keyword set is parsed out of
``render_door.py`` and compared with the installed engine's signature.  A
keyword added to the door and not to the renderer fails here, in a second,
instead of after an export.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

import woof.globe.render_door as render_door


def _call_keywords(function_name: str) -> set[str]:
    """The keyword names ``render_door`` passes to ``function_name``."""

    source = pathlib.Path(render_door.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: set[str] = set()
    calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if name != function_name:
            continue
        calls += 1
        for keyword in node.keywords:
            if keyword.arg is not None:
                found.add(keyword.arg)
    assert calls == 1, (
        f"expected exactly one call to {function_name} in render_door.py, found {calls}")
    return found


def test_render_door_passes_only_keywords_the_engine_declares():
    keywords = _call_keywords("render_wrfouts_rust")
    assert keywords, "the door passes no keywords at all, which is not the call it makes"

    try:
        from woof.render import render_wrfouts_rust
    except ImportError:  # pragma: no cover - only without the engine installed
        pytest.skip("the installed woof does not carry woof.render")

    declared = set(inspect.signature(render_wrfouts_rust).parameters)
    unknown = sorted(keywords - declared)
    assert not unknown, (
        "render_door passes keyword(s) the installed engine's renderer does not "
        f"declare: {', '.join(unknown)}. Every draw dies at the call with a "
        "TypeError, after the tapes have been exported."
    )
