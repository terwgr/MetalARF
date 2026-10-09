"""Which engine callables this package uses have a DIFFERENT SIGNATURE in another engine tree.

`tools/measure_boundary.py` answers whether a name resolves.  That is not the
whole boundary.  A name can resolve and still be a different callable: the
engine tree this model was developed on carries a keyword-only argument that a
release-candidate tree does not, the install reports success, and the run dies
at the call several minutes in, inside the physics.  Four such divergences were
found this way in one afternoon, each of which an existence check called
IDENTICAL.

Run it with the two engine trees to compare::

    python tools/measure_engine_signatures.py <tree-a> <tree-b>

Each tree root is a directory containing a ``gpuwm/`` package.  The first is
the reference (the tree whose behaviour this package expects); the second is
the candidate.  Rows marked LACKS are the ones that break a run: an argument
this package passes that the candidate does not accept.

Nothing is imported.  The comparison is on the parsed source, so it runs
against a checkout, an unpacked wheel or a site-packages directory, with no
CUDA runtime and no engine install.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "arwen_global"


def _params(node: ast.arguments) -> tuple:
    return (tuple(p.arg for p in node.posonlyargs),
            tuple(p.arg for p in node.args),
            node.vararg.arg if node.vararg else None,
            tuple(p.arg for p in node.kwonlyargs),
            node.kwarg.arg if node.kwarg else None)


def signatures(root: pathlib.Path) -> dict[str, dict[str, tuple]]:
    """Every module-level function, method and annotated class field, by file."""
    out: dict[str, dict[str, tuple]] = {}
    package_root = root / "woof"
    if not package_root.is_dir():
        raise SystemExit(f"{root} contains no gpuwm/ package")
    for path in package_root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        table: dict[str, tuple] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                table[node.name] = _params(node.args)
            elif isinstance(node, ast.ClassDef):
                fields = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        table[f"{node.name}.{item.name}"] = _params(item.args)
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        fields.append(item.target.id)
                if fields:
                    # A dataclass's field list is its constructor signature.
                    table[f"{node.name}.__fields__"] = ((), tuple(fields), None, (), None)
        out[rel] = table
    return out


def names_this_package_uses(package: pathlib.Path) -> set[str]:
    """Every attribute, bare name and imported name this package calls or imports.

    Deliberately broad: a name that is merely spelled here costs one row of
    output, while a name that is missed costs a run.  The spectral core is
    excluded because it is this package's own code, not the engine's.
    """
    used: set[str] = set()
    for path in package.rglob("*.py"):
        if "/spectral/" in path.as_posix():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    used.add(func.id)
                elif isinstance(func, ast.Attribute):
                    used.add(func.attr)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    used.add(alias.name)
    return used


def compare(reference: pathlib.Path, candidate: pathlib.Path,
            used: set[str]) -> list[tuple[str, str, list[str], list[str]]]:
    left, right = signatures(reference), signatures(candidate)
    rows = []
    for rel, table in left.items():
        other = right.get(rel)
        if other is None:
            continue
        for name, sig in table.items():
            head, leaf = name.split(".")[0], name.split(".")[-1]
            if head.startswith("_") or (head not in used and leaf not in used):
                continue
            theirs = other.get(name)
            if theirs is None or theirs == sig:
                continue
            ours_kw = set(sig[1]) | set(sig[3])
            theirs_kw = set(theirs[1]) | set(theirs[3])
            lacks = sorted(ours_kw - theirs_kw)
            extra = sorted(theirs_kw - ours_kw)
            if lacks or extra:
                rows.append((rel, name, lacks, extra))
    rows.sort()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=pathlib.Path,
                        help="the engine tree whose behaviour this package expects")
    parser.add_argument("candidate", type=pathlib.Path,
                        help="the engine tree to check against it")
    parser.add_argument("--fail-on-lacks", action="store_true",
                        help="exit 1 when the candidate does not accept an argument this package passes")
    args = parser.parse_args(argv)

    used = names_this_package_uses(PACKAGE)
    rows = compare(args.reference, args.candidate, used)
    lacks = [row for row in rows if row[2]]

    print(f"{len(rows)} callables differ; {len(lacks)} do not accept an argument this package passes")
    print()
    for rel, name, missing, extra in rows:
        detail = []
        if missing:
            detail.append("candidate LACKS " + ", ".join(missing))
        if extra:
            detail.append("candidate has extra " + ", ".join(extra))
        print(f"{rel}::{name}  --  {'; '.join(detail)}")

    if args.fail_on_lacks and lacks:
        print(file=sys.stderr)
        print(f"{len(lacks)} arguments this package passes are not accepted by "
              f"{args.candidate}; a run dies at the call, not at the install",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
