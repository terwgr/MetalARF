"""Measure this package's boundary against an INSTALLED woof engine.

The boundary between this distribution and the engine it runs on is a list of
modules and symbols, and it is the one thing about this package that can go
wrong without anybody typing a wrong character: the engine moves, the symbol
list does not, and an install that reported success dies at the first import
of a forecast.

So the boundary is MEASURED, at every engine bump, rather than transcribed
into a document that then ages.  This script is the instrument.

    python tools/measure_boundary.py                 # human table
    python tools/measure_boundary.py --json          # machine
    python tools/measure_boundary.py --fail-on-gap   # exit 1 if anything is missing

It reads the `from woof...` and `import woof...` statements out of the
package's own source with `ast` -- never by importing the package, which is
the point: the package may be exactly what cannot be imported -- and then
resolves each module and each symbol against the engine that is installed in
the interpreter running this script.

WHAT A GAP MEANS.  A missing module or symbol is a command of this package
that will fail, and the table says so per row.  It is not always a defect in
this package: the carve deliberately left several symbols on the engine's
side as a patch series the engine has to carry, and until it does, this table
is the list of what is waiting.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "arwen_global"


def collect(root: Path) -> dict[str, dict[str, list[str]]]:
    """{module: {symbol: [files that import it]}} for every woof target."""

    found: dict[str, dict[str, list[str]]] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        where = path.relative_to(root.parent.parent).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not _is_engine(module):
                    continue
                for alias in node.names:
                    found.setdefault(module, {}).setdefault(
                        alias.name, []).append(where)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_engine(alias.name):
                        found.setdefault(alias.name, {}).setdefault(
                            "<module>", []).append(where)
    return found


#: Import failures that mean "this machine has no card", not "the engine
#: moved".  Listed rather than sniffed from the message shape, so a genuinely
#: missing engine module can never hide behind the same sentence.
_OPTIONAL_BACKENDS = ("cupy", "cupyx")


def _is_optional_backend(verdict: str) -> bool:
    return any(f"'{name}'" in verdict for name in _OPTIONAL_BACKENDS)


def _is_engine(name: str) -> bool:
    return name == "woof" or name.startswith("woof.")


def resolve(found: dict) -> list[dict]:
    rows: list[dict] = []
    for module in sorted(found):
        try:
            loaded = importlib.import_module(module)
            module_verdict = "present"
        except Exception as exc:
            loaded = None
            module_verdict = f"ABSENT ({type(exc).__name__}: {exc})"
        optional = (module_verdict.startswith("ABSENT")
                    and _is_optional_backend(module_verdict))
        for symbol in sorted(found[module]):
            if loaded is None:
                # A module that fails to import because an OPTIONAL backend is
                # missing is not a boundary gap: the symbol is there and the
                # machine simply has no card.  Calling it absent would put a
                # permanent row in this table on every CPU host, which is how
                # a real gap comes to look like background noise.
                verdict = ("optional backend absent" if optional
                           else "module absent")
            elif symbol == "<module>":
                verdict = "present"
            elif hasattr(loaded, symbol):
                verdict = "present"
            else:
                # `from gpuwm.x import y` also succeeds when y is a SUBMODULE
                # that has not been imported yet, so a missing attribute is
                # not yet a gap.
                try:
                    importlib.import_module(f"{module}.{symbol}")
                    verdict = "present (submodule)"
                except Exception as exc:
                    # THE SAME RULE THE MODULE ROW USES, and it was missing
                    # here.  A submodule that fails to import because the
                    # machine has no CUDA is not a boundary gap either: the
                    # symbol is there.  Measured 2026-09-09 on the desktop,
                    # after the carve made `woof.core.microphysics` reachable
                    # from a carried file: the tool reported one gap on a
                    # CPU-only host and named a module the engine ships.  A
                    # false gap in this table is worse than no table, because
                    # the next reader learns to skip the section.
                    detail = f"{type(exc).__name__}: {exc}"
                    verdict = ("optional backend absent"
                               if _is_optional_backend(detail) else "ABSENT")
            rows.append({
                "module": module,
                "module_verdict": module_verdict,
                "symbol": symbol,
                "verdict": verdict,
                "private": symbol.startswith("_") and symbol != "<module>",
                "importers": sorted(set(found[module][symbol])),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_boundary",
        description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-on-gap", action="store_true")
    parser.add_argument(
        "--root", type=Path, default=SRC,
        help="the package source to read imports out of (default: this repo)")
    args = parser.parse_args(argv)

    rows = resolve(collect(args.root))
    gaps = [row for row in rows
            if row["verdict"] in ("ABSENT", "module absent")]
    optional = [row for row in rows
                if row["verdict"] == "optional backend absent"]
    private = [row for row in rows if row["private"]]

    if args.json:
        print(json.dumps({
            "engine": _engine_version(),
            "symbols": len(rows),
            "gaps": len(gaps),
            "optional_backend": len(optional),
            "optional_backend_modules": sorted(
                {row['module'] for row in optional}),
            "private": len(private),
            "rows": rows,
        }, indent=2, sort_keys=True))
    else:
        print(f"engine: woof {_engine_version()}")
        print(f"{len(rows)} symbols across "
              f"{len({row['module'] for row in rows})} modules; "
              f"{len(gaps)} gap(s), {len(optional)} needing an optional "
              f"backend, {len(private)} private import(s)\n")
        if gaps:
            print("GAPS -- each one is a command of this package that fails:")
            for row in gaps:
                print(f"  {row['module']}.{row['symbol']}")
                print(f"      {row['verdict']}")
                for importer in row["importers"]:
                    print(f"      imported by {importer}")
            print()
        if private:
            print("PRIVATE names crossing the distribution boundary "
                  "(a contract that does not exist):")
            for row in private:
                print(f"  {row['module']}.{row['symbol']}  "
                      f"[{row['verdict']}]")
            print()
        # WHAT THIS HOST COULD NOT CHECK IS PART OF THE ANSWER.  A module
        # that imports cupy at module scope does not load on a CPU host,
        # so every symbol in it is stamped "optional backend absent"
        # without hasattr being evaluated at all: a symbol genuinely
        # missing from one of those modules reads as clean here.
        # Measured from the built sdist against a venv created from
        # scratch with woof 2.7.0 (desktop, 2026-09-10): 199 symbols,
        # 0 gaps, 40 needing an optional backend, and the sentence
        # underneath said every symbol resolved.
        if not gaps and not optional:
            print("No gaps: every symbol this package names resolves against "
                  "the installed engine.")
        elif not gaps:
            blind = sorted({row['module'] for row in optional})
            print(f"No gaps among the {len(rows) - len(optional)} symbols "
                  "this host could resolve.  "
                  f"{len(optional)} symbol(s) were NOT checked: "
                  f"{', '.join(blind)} need an optional backend this "
                  "host does not have, so every symbol in them was "
                  "excused without being looked for.  Run this on a "
                  "card host to close them.")
    return 1 if (args.fail_on_gap and gaps) else 0


def _engine_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("woof")
    except PackageNotFoundError:
        return "not installed"


if __name__ == "__main__":
    sys.exit(main())
