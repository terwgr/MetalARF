"""Skip a case whose input the Recast WOOF tree leaves out, naming what left it out.

THE BREAKAGE THIS PREVENTS
--------------------------
Recast WOOF is assembled from the engine's public export minus what WOOF does
not carry: the desktop and terminal front ends, the release-cut machinery,
case-named configurations and their verification case module, and campaign
records.  The list is ``_woof_left_out.txt`` beside this file, written by the
assembly from its own exclusion list.  The shipped tests are the engine's, and
a test that reads one of those pieces failed on the WOOF tree although the
code it covers ships unchanged: on recast-woof 1.1.1 about 150 Stage 1 cases
stopped on a missing ``configs/real74_*.toml`` fixture, the terminal-UI module
or the release tooling.  It is the same rule ``tests/_release_export.py``
applies to the engine's own export, for the inputs WOOF leaves out on top.

WHEN IT SKIPS
-------------
Only when the input a case failed to read is absent from this tree AND a rule
in ``_woof_left_out.txt`` names it.  A present input runs.  An absent input no
rule names fails as before, naming its path: that absence is a defect, not the
assembly working.  A tree without ``_woof_left_out.txt`` (the engine's own
checkout) skips nothing.  The skip reason names the input and the rule.
"""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = Path(__file__).resolve().parent / "_woof_left_out.txt"

_CANNOT_IMPORT = re.compile(r"cannot import name '(\w+)' from '([\w.]+)'")
_DOES_NOT_EXIST = re.compile(r"(\S+) does not exist;")


def rules() -> list[str]:
    if not RULES_FILE.is_file():
        return []
    return [line for raw in RULES_FILE.read_text(encoding="utf-8").splitlines()
            if (line := raw.strip()) and not line.startswith("#")]


def rule_for(rel: str, active: list[str]) -> str | None:
    for rule in active:
        if rule.endswith("/**"):
            base = rule[:-3]
            if rel == base or rel.startswith(base + "/"):
                return rule
        elif rel == rule:
            return rule
    return None


def _relative(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _module_paths(module: str) -> list[str]:
    stem = module.replace(".", "/")
    return [stem + ".py", stem]


def _inputs(error: BaseException) -> list[str]:
    """The repository-relative paths an error says were missing."""
    found: list[str] = []
    if isinstance(error, ImportError):
        match = _CANNOT_IMPORT.search(str(error))
        if match is not None:
            found += _module_paths(f"{match.group(2)}.{match.group(1)}")
        elif error.name:
            found += _module_paths(error.name)
    filename = getattr(error, "filename", None)
    if isinstance(error, OSError) and filename:
        rel = _relative(filename)
        if rel is not None:
            found.append(rel)
    for match in _DOES_NOT_EXIST.finditer(str(error)):
        rel = _relative(match.group(1))
        if rel is not None:
            found.append(rel)
    return found


def left_out(rel: str) -> bool:
    """Is ``rel`` absent from this tree because a rule leaves it out?"""
    return not (ROOT / rel).exists() and rule_for(rel, rules()) is not None


#: Native artifacts the engine declares whose workspace a tree may leave out:
#: artifact -> its workspace.  The engine's tests name them among the
#: artifacts every release carries.
WORKSPACE_ARTIFACTS = {"arwen-tui": "tools/arwen-tui"}


def without_left_out_artifacts(names) -> set[str]:
    """``names`` minus the artifacts whose workspace this tree leaves out."""
    active = rules()
    return {name for name in names
            if not (name in WORKSPACE_ARTIFACTS and not (ROOT / WORKSPACE_ARTIFACTS[name]).exists()
                    and rule_for(WORKSPACE_ARTIFACTS[name], active) is not None)}


#: The workflows the public Recast WOOF repository carries in place of the
#: engine's (which the tree leaves out, ``.github/**``): each publishes a
#: release's own files to PyPI and is told apart by its first comment.
OWN_WORKFLOWS = {
    ".github/workflows/publish.yml": "# Publishes recast-woof to PyPI",
    ".github/workflows/publish-data.yml": "# Publishes recast-woof-data to PyPI",
}


def engine_workflow_reason(rel: str = ".github/workflows/publish.yml") -> str | None:
    """Why a case reading the engine's workflow ``rel`` cannot run in this tree, or ``None``.

    The engine's release workflow is not in this tree: either the path is
    absent and left out, or it holds Recast WOOF's own publish workflow, a
    different file under the same name.
    """
    path = ROOT / rel
    marker = OWN_WORKFLOWS.get(rel)
    if marker is not None and path.is_file() and marker in path.read_text(encoding="utf-8"):
        return (f"{rel} is Recast WOOF's own publish workflow here; the engine's release workflow these "
                "assertions read is left out of this tree; this case runs in the engine's tree")
    active = rules()
    if active and not path.exists() and rule_for(rel, active) is not None:
        return (f"reads {rel}, which the Recast WOOF tree leaves out ({RULES_FILE.name}: "
                f"{rule_for(rel, active)}); this case runs in the engine's tree")
    return None


def skip_reason(error: BaseException) -> str | None:
    """Why a case that raised ``error`` cannot run in this tree, or ``None``."""
    active = rules()
    if not active:
        return None
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        for rel in _inputs(error):
            if (ROOT / rel).exists():
                continue
            rule = rule_for(rel, active)
            if rule is not None:
                return (f"reads {rel}, which the Recast WOOF tree leaves out "
                        f"({RULES_FILE.name}: {rule}); this case runs in the engine's tree")
        error = error.__cause__ or error.__context__
    return None
