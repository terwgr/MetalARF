"""Report every machine identity left in the tree, with file and line.

`tests/test_no_provenance.py` ASSERTS the leaks that must never ship: a
home-directory path, a private network address, a personal name, the name
of a tool a file was written with, and since 0.1.2 the hostnames of the
machines figures were taken on as well (the published 0.1.1 wheel carried
six in carried physics comments).  This script still REPORTS that class
over the whole tree, including files that do not ship, one line per hit:

    <file>:<line>: <the sentence, with the machine's name in it>

A hostname in a measurement docstring is not dangerous the way a private
subnet is, but it is not useful to a reader either: they cannot reach the
machine, and the fact that matters (which card, which operating system,
which interpreter) is usually in the same sentence.

    python tools/provenance_sweep.py            # report
    python tools/provenance_sweep.py --count    # one number

Exit status is 0 whether or not it finds anything: this is an instrument,
not a gate.  The gate is the test file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "build", "dist",
              ".venv", "venv", "node_modules"}

#: Hostnames of the machines this project's numbers were measured on, and
#: run directories underneath them.
MACHINE_NAME = re.compile(r"\b(weather-node-\d+|node-[1-9])\b")


def candidates() -> list[Path]:
    found = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".md", ".toml",
                                                     ".yml", ".cu", ".json"}:
            continue
        if set(path.relative_to(ROOT).parts) & SKIP_PARTS:
            continue
        if path.name in {"provenance_sweep.py", "test_no_provenance.py"}:
            continue  # they state the patterns as literals
        found.append(path)
    return found


def sweep() -> list[str]:
    rows: list[str] = []
    for path in candidates():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.splitlines()
        for match in MACHINE_NAME.finditer(text):
            number = text[:match.start()].count("\n") + 1
            body = lines[number - 1].strip() if number <= len(lines) else ""
            rows.append(
                f"{path.relative_to(ROOT).as_posix()}:{number}: {body[:150]}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", action="store_true",
                        help="print only the number of mentions")
    args = parser.parse_args()

    rows = sweep()
    if args.count:
        print(len(rows))
        return 0
    for row in rows:
        print(row)
    files = len({row.split(":", 1)[0] for row in rows})
    print(f"\n{len(rows)} mentions across {files} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
