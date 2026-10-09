"""The machine surface a workspace reads while a long command runs.

Two files per command, both under the command's own output directory:

``status.json``
    the current state, rewritten ATOMICALLY on every transition.  A reader
    that opens it at any instant sees a complete document, never a truncated
    one; that is the whole reason for the temp-file-and-replace below rather
    than an open-and-write in place.

``<command>.log``
    append-only, one line per event, human text, and its path is printed on
    the first line of stdout so a reader who is not driving a workspace can
    still find it.

The schema is the contract in ``docs/tui-contract.md``.  It is deliberately
small: a workspace polling a file should be able to draw a progress bar, a
stage name and an outcome without parsing prose, and everything richer stays
in the receipt the command writes at the end.

Nothing here knows what a terminal workspace is, and no code in this package
draws one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import json
import os
from pathlib import Path

__all__ = ["STATUS_NAME", "SCHEMA", "StatusWriter", "utc_now"]

#: The file a workspace polls.
STATUS_NAME = "status.json"

#: The document's schema id.  A workspace pins it; a change of shape changes
#: this string, so a reader can refuse a document it does not understand
#: rather than misread it.
SCHEMA = "gpuwm-global-status-v1"

#: The four states a command can be in.  `refused` is separate from `failed`
#: on purpose: a refusal is this package declining to do something and naming
#: why, and a workspace should show it as an answer rather than as a crash.
STATES = ("running", "done", "failed", "refused")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class StatusWriter:
    """Writes ``status.json`` and the command log for one command run."""

    outdir: Path
    command: str
    stages: tuple[str, ...] = ()
    #: Set once the writer is live; None means the outdir could not be made
    #: and the command runs without a machine surface rather than dying for
    #: want of one.
    _path: Path | None = field(default=None, init=False)
    _log: Path | None = field(default=None, init=False)
    _stage_index: int = field(default=0, init=False)
    _step: int = field(default=0, init=False)
    _step_count: int = field(default=0, init=False)
    _started: str = field(default_factory=utc_now, init=False)

    def __post_init__(self) -> None:
        self.outdir = Path(self.outdir)
        try:
            self.outdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        self._path = self.outdir / STATUS_NAME
        self._log = self.outdir / f"{self.command.replace(' ', '-')}.log"
        self._write("running")

    # -- the surface a command drives -----------------------------------

    @property
    def log_path(self) -> Path | None:
        return self._log

    def stage(self, name: str, *, step_count: int = 0) -> None:
        """Enter a named stage.  ``step_count`` sizes its progress bar."""

        if name in self.stages:
            self._stage_index = self.stages.index(name)
        else:
            self.stages = (*self.stages, name)
            self._stage_index = len(self.stages) - 1
        self._step = 0
        self._step_count = step_count
        self.note(f"stage {name}")
        self._write("running", stage=name)

    def step(self, index: int, *, note: str | None = None) -> None:
        self._step = index
        if note:
            self.note(note)
        self._write("running")

    def note(self, line: str) -> None:
        """One human line in the log.  Never in status.json."""

        if self._log is None:
            return
        try:
            with self._log.open("a", encoding="utf-8") as stream:
                stream.write(f"{utc_now()} {line}\n")
        except OSError:
            pass

    def done(self, note: str | None = None) -> None:
        if note:
            self.note(note)
        self._write("done")

    def failed(self, reason: str) -> None:
        self.note(f"failed: {reason}")
        self._write("failed", reason=reason)

    def refused(self, reason: str) -> None:
        self.note(f"refused: {reason}")
        self._write("refused", reason=reason)

    # -- the atomic write ------------------------------------------------

    def _write(self, state: str, *, stage: str | None = None,
               reason: str | None = None) -> None:
        if self._path is None:
            return
        if stage is None:
            stage = (self.stages[self._stage_index]
                     if self.stages and self._stage_index < len(self.stages)
                     else self.command)
        payload = {
            "schema": SCHEMA,
            "command": self.command,
            "stage": stage,
            "stage_index": self._stage_index,
            "stage_count": len(self.stages),
            "step": self._step,
            "step_count": self._step_count,
            "started_utc": self._started,
            "updated_utc": utc_now(),
            "eta_s": None,
            "state": state,
        }
        if reason is not None:
            payload["reason"] = reason
        if self._log is not None:
            payload["log"] = str(self._log)
        temp = self._path.with_suffix(".json.tmp")
        try:
            temp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(temp, self._path)
        except OSError:
            pass
