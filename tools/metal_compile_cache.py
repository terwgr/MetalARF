"""Reuse AIR objects when compiler, flags and transitive local headers match."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile


def cached_air(args: list[str], root: Path, run) -> str:
    if ("-c" not in args or "-o" not in args
            or Path(args[0]).name != "metal"):
        return run(args)
    source = Path(args[args.index("-c") + 1])
    target = Path(args[args.index("-o") + 1])
    if source.suffix != ".metal" or target.suffix != ".air":
        return run(args)
    include_dirs = [Path(args[i + 1]) for i, arg in enumerate(args[:-1]) if arg == "-I"]
    digest = hashlib.sha256()
    tool = Path(args[0]).resolve()
    stat = tool.stat()
    digest.update(f"{tool}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    skip = {args.index("-o") + 1}
    for i, arg in enumerate(args):
        if i not in skip and not arg.startswith("-fmodules-cache-path="):
            digest.update(arg.encode() + b"\0")
    visited = set()

    def consume(path):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        data = path.read_bytes()
        digest.update(str(path).encode() + b"\0" + data)
        for header in re.findall(rb'^\s*#\s*include\s*"([^"\n]+)"', data, re.M):
            name = header.decode()
            found = next((directory / name for directory in (path.parent, *include_dirs)
                          if (directory / name).is_file()), None)
            if found is None:
                raise FileNotFoundError(f"{path}: include {name}")
            consume(found)

    consume(source)
    cache = root / "build" / ".metal-air-cache"
    cache.mkdir(parents=True, exist_ok=True)
    entry = cache / (digest.hexdigest() + ".air")
    if entry.is_file():
        shutil.copyfile(entry, target)
        return ""
    output = run(args)
    # Atomic publication lets simultaneous family builds share the cache.
    fd, temporary = tempfile.mkstemp(dir=cache, suffix=".air")
    os.close(fd)
    try:
        shutil.copyfile(target, temporary)
        os.replace(temporary, entry)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output
