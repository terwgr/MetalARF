"""Durable CLI worker for a detached terminal UI job.

``python -P -m woof.globe.tui_worker --job-dir DIR [--windows-job NAME] --
<woof global args>`` is this distribution's half of the handshake the engine
publishes as ``gpuwm.tui_worker``: a launcher spawns it, the worker publishes
``process.json`` and a ``ready`` marker, WAITS for the launcher to write a
``start`` marker, and only then imports and runs the CLI, leaving
``result.json`` behind whatever happens.

WHY THE HANDSHAKE HAS TO EXIST HERE TOO.  The start marker is published only
after the launcher owns the process group (POSIX) or the JobObject (Windows).
A worker that ran the CLI before that seam would already be integrating on a
card the launcher cannot yet stop, which on this model means a forecast that
survives the workspace that started it -- the exact failure the marker
prevents.  The bound below is on the pre-launch handshake ALONE and is not a
limit on preparation or on a forecast once the launcher releases it: a global
day is hours.

WHAT IS IMPORTED AND WHAT IS NOT.  ``_now``, ``_write_result`` and
``_join_windows_job`` are the ENGINE's, imported rather than copied: they are
the ``result.json`` shape a launcher parses, the exclusive-create-and-replace
that makes it durable, and the Windows JobObject dance with its venv
redirector caveat, and a second copy of any of them would be a second thing to
keep in step.  :func:`main` is carried because the engine's ends with ``from
woof.cli import main``: a worker that ran the ENGINE's CLI from this module
would answer ``woof global`` arguments with ``woof``'s parser, which is a
job that starts, refuses at exit 2 and looks to the launcher exactly like a
successful spawn of the wrong command.

A launcher therefore changes ONE token to own a job of this package instead of
one of the engine's: the module it spawns.  The job directory's contract, the
marker names, the result schema and the exit codes are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

# The engine's own three, by import.  A ``result.json`` a launcher can parse
# is defined by the engine's writer, not by a copy of it here.
#
# THEY ARE PRIVATE NAMES, which is a contract that does not exist, so the
# import is guarded: an engine that renames one would otherwise fail here as a
# bare ImportError naming an underscore, seconds after a launcher spawned a
# job it now cannot tell the end of.  ``tools/measure_boundary.py`` lists all
# three, and ``tests/test_arwen_global_tui_worker.py`` asserts they are the
# engine's objects rather than copies, so a rename fails the suite before it
# reaches a launcher.
try:
    from woof.tui_worker import _join_windows_job, _now, _write_result
except ImportError as _error:  # pragma: no cover - an engine that renamed one
    raise ImportError(
        "the installed engine's gpuwm.tui_worker does not carry "
        f"{_error.name or 'one of _now, _write_result, _join_windows_job'}, "
        "which this worker imports so that its result.json is byte-shaped by "
        "the engine's own writer and a launcher needs one parser rather than "
        "two.  This stops every detached woof global job: the handshake "
        "cannot publish a result a launcher can read.  The remedy is an "
        "engine whose tui_worker carries those three, or a public spelling "
        "of them to import instead."
    ) from _error

__all__ = ["RESULT_SCHEMA", "PROCESS_SCHEMA", "HANDSHAKE_TIMEOUT_S", "main"]

#: The schemas a launcher pins.  Spelled from the engine's own documents so a
#: launcher written for ``gpuwm.tui_worker`` needs no second branch.
RESULT_SCHEMA = "gpuwm-tui-result-v1"
PROCESS_SCHEMA = "gpuwm-tui-process-v1"

#: How long an ABANDONED pre-launch handshake is waited on.  The engine's own
#: bound, and for the engine's own reason: it covers the seam between "the
#: worker is ready" and "the launcher has taken ownership", never the work.
HANDSHAKE_TIMEOUT_S = 60.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--windows-job")
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cli_args = args.cli_args
    if cli_args[:1] == ["--"]:
        cli_args = cli_args[1:]
    directory = args.job_dir.resolve(strict=True)
    record = {"schema": RESULT_SCHEMA, "pid": os.getpid(),
              "cli_args": cli_args, "started_at": _now(), "exit_code": None,
              # WHICH product owns this job.  A launcher that can spawn both
              # this module and the engine's sees two identical result
              # documents otherwise, and a stale job directory would be read
              # as the wrong product's.
              "producer": "woof global"}
    code = 1
    try:
        if args.windows_job:
            _join_windows_job(args.windows_job)
        with (directory / "process.json").open("x", encoding="utf-8") as process:
            json.dump({"schema": PROCESS_SCHEMA, "pid": os.getpid(),
                       "parent_pid": os.getppid(), "started_at": record["started_at"],
                       "cwd": str(Path.cwd()), "cli_args": cli_args,
                       "process_group": os.getpgrp() if os.name != "nt" else None,
                       "windows_job": args.windows_job,
                       "producer": "woof global"}, process, indent=2)
            process.write("\n")
            process.flush()
            os.fsync(process.fileno())
        with (directory / "ready").open("x") as ready:
            ready.flush()
            os.fsync(ready.fileno())
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
        marker = directory / "start"
        while not marker.is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "the launcher did not release the startup handshake")
            time.sleep(0.02)
        if not cli_args:
            raise ValueError("a job requires a woof global command")
        from woof.globe.cli import main as cli_main

        result = cli_main(cli_args)
        code = 0 if result is None else int(result)
    except SystemExit as error:
        if error.code is None:
            code = 0
        elif isinstance(error.code, int):
            code = error.code
        else:
            print(error.code, file=sys.stderr, flush=True)
            code = 1
    except KeyboardInterrupt:
        print("woof global: interrupted; partial output has no completion "
              "receipt.", file=sys.stderr, flush=True)
        code = 130
    except Exception as error:  # noqa: BLE001 - result.json is the report
        traceback.print_exc()
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        code = 1
    finally:
        record.update(ended_at=_now(), exit_code=code,
                      status=("completed" if code == 0 else
                              "interrupted" if code == 130 else "failed"))
        _write_result(directory, record)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
