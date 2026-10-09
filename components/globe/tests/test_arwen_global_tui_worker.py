"""The durable worker's handshake, which is what makes a job ownable.

A terminal workspace or a desktop application does not run a forecast in its
own process: it spawns a worker, takes ownership of the worker's process group
(POSIX) or JobObject (Windows), and only then releases it.  Everything below
holds the three properties that make that safe, and each one is a real failure
that has happened to a launcher rather than a hypothesis:

* the worker publishes ``process.json`` and ``ready`` and then WAITS.  A
  worker that ran the CLI before the launcher released it would already be
  integrating on a card the launcher cannot stop, and on this model that is a
  forecast that outlives the workspace that started it.
* ``result.json`` is left behind whatever happens -- success, refusal,
  interrupt -- because a job directory with no result is indistinguishable
  from a worker that is still running.
* the CLI it runs is THIS package's.  The engine's worker ends with ``from
  woof.cli import main``, so a launcher that spawned the engine's module for
  a ``woof global`` job would get a process that starts, refuses at exit 2,
  and looks from the outside exactly like a successful spawn.

The tests run the worker as a real subprocess.  A handshake tested in-process
is not a handshake.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from woof.globe import tui_worker

#: How long a test waits for a file the worker is about to write.  Generous
#: because the first import of the package on a cold filesystem is seconds,
#: and a flaky test is worse than a slow one.
WAIT_S = 300.0


def _spawn(job_dir: Path, *cli_args: str) -> subprocess.Popen:
    job_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, "-P", "-m", "woof.globe.tui_worker",
         "--job-dir", str(job_dir), "--", *cli_args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8")


def _wait_for(path: Path, timeout: float = WAIT_S) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"{path} did not appear within {timeout:g} s")
        time.sleep(0.05)


def test_the_engines_helpers_are_imported_and_not_copied():
    """``result.json``'s shape is the engine's, so its writer must be too."""

    import woof.tui_worker as engine

    assert tui_worker._write_result is engine._write_result
    assert tui_worker._now is engine._now
    assert tui_worker._join_windows_job is engine._join_windows_job
    assert tui_worker.RESULT_SCHEMA == "gpuwm-tui-result-v1"
    assert tui_worker.PROCESS_SCHEMA == "gpuwm-tui-process-v1"


def test_the_worker_waits_for_the_start_marker_then_runs_this_packages_cli(tmp_path):
    job = tmp_path / "job"
    process = _spawn(job, "run-plan", "--probe", "--no-readiness")
    try:
        _wait_for(job / "ready")
        record = json.loads((job / "process.json").read_text("utf-8"))
        assert record["schema"] == tui_worker.PROCESS_SCHEMA
        assert record["producer"] == "woof global"
        assert record["cli_args"] == ["run-plan", "--probe", "--no-readiness"]
        if os.name != "nt":
            # The launcher's handle on POSIX: without a process group there is
            # nothing to signal but one pid, and a forecast's children survive.
            assert record["pid"] == process.pid
            assert record["process_group"] == os.getpgid(process.pid)
        else:
            # MEASURED on Windows 2026-09-09: a venv's `Scripts\\python.exe` is
            # a REDIRECTOR, so the pid the launcher spawned is not the pid
            # running the forecast -- here 50684 spawned, 19732 integrating.
            # This is why the worker publishes its own pid and why
            # `--windows-job` exists: a launcher that killed the pid it
            # spawned would leave the real process running on the card.  The
            # document is the authority, never the spawn.
            assert isinstance(record["pid"], int) and record["pid"] > 0
            assert record["process_group"] is None

        # NOTHING HAS RUN YET.  The worker is holding at the seam; a result
        # here would mean the CLI ran before the launcher owned the process.
        assert not (job / "result.json").exists()
        assert process.poll() is None

        (job / "start").write_text("", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=WAIT_S)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    result = json.loads((job / "result.json").read_text("utf-8"))
    assert result["schema"] == tui_worker.RESULT_SCHEMA
    assert result["producer"] == "woof global"
    assert result["exit_code"] == 0
    assert result["status"] == "completed"
    assert result["ended_at"] >= result["started_at"]
    assert process.returncode == 0
    # The CLI that ran is this package's: an engine worker would have refused
    # `run-plan --probe` at exit 2 with an unknown-command message.
    document = json.loads(stdout)
    assert document["schema"] == "gpuwm.run-plan.probe.v1"
    assert document["producer"]["distribution"] == "woof global"
    assert stderr is not None


def test_a_refusal_still_leaves_a_result(tmp_path):
    """A job directory with no result is a job nobody can tell the end of."""

    job = tmp_path / "job"
    process = _spawn(job, "run-plan", str(tmp_path / "no-such-plan.json"))
    try:
        _wait_for(job / "ready")
        (job / "start").write_text("", encoding="utf-8")
        _stdout, stderr = process.communicate(timeout=WAIT_S)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    result = json.loads((job / "result.json").read_text("utf-8"))
    assert result["status"] == "failed"
    assert result["exit_code"] != 0
    assert process.returncode == result["exit_code"]
    assert "no-such-plan.json" in stderr


def test_an_abandoned_handshake_ends_the_worker_rather_than_hanging(tmp_path):
    """A launcher that dies before releasing must not leave a worker forever.

    The bound is on the PRE-LAUNCH seam only, so it is shortened here rather
    than waited out: a test that took the shipped minute would be a minute of
    every suite, and what is being held is that the timeout exists and that
    the worker reports the reason it fired.
    """

    job = tmp_path / "job"
    job.mkdir()
    script = (
        "import woof.globe.tui_worker as worker;"
        "worker.HANDSHAKE_TIMEOUT_S = 1.0;"
        "import sys; sys.exit(worker.main("
        f"['--job-dir', {str(job)!r}, '--', 'run-plan', '--probe']))")
    process = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True,
                             encoding="utf-8", timeout=WAIT_S)
    assert process.returncode == 1
    result = json.loads((job / "result.json").read_text("utf-8"))
    assert result["status"] == "failed"
    assert "handshake" in result["error"]["message"]


def test_a_job_directory_is_used_once(tmp_path):
    """``process.json`` is created exclusively, so a reused directory refuses.

    Two workers in one job directory would leave one result document
    describing whichever finished last, and a launcher reading it would
    attribute one run's outcome to the other.
    """

    job = tmp_path / "job"
    first = _spawn(job, "run-plan", "--probe", "--no-readiness")
    try:
        _wait_for(job / "ready")
        second = _spawn(job, "run-plan", "--probe", "--no-readiness")
        second.communicate(timeout=WAIT_S)
        assert second.returncode != 0
        (job / "start").write_text("", encoding="utf-8")
        first.communicate(timeout=WAIT_S)
    finally:
        for process in (first,):
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert first.returncode == 0
