"""No cache sidecar, report or table this package writes names the machine.

THE BREAKAGE THIS PREVENTS: through 0.1.1 the statics cache sidecar recorded
`socket.gethostname()`, and two probe tools recorded `platform.node()`.  A
geog cache is built once and copied between machines, attached to support
requests and published beside runs, so every copy named the machine it was
built on.  What a reader can act on is the kind of machine, which is what
these files now record.
"""

from __future__ import annotations

import platform
import re
import socket
from pathlib import Path

from woof.globe import statics

REPO = Path(__file__).resolve().parents[1]


def test_the_sidecar_records_the_kind_of_machine_not_its_name():
    kind = statics.machine_class()
    assert re.fullmatch(r"[a-z0-9_.]+-[a-z0-9_.]+", kind), kind
    assert kind == f"{platform.system().lower()}-{platform.machine().lower()}"
    assert kind != socket.gethostname().lower()


def test_no_writer_this_package_ships_reads_the_hostname():
    offenders = []
    for root in ("src", "tools"):
        for path in (REPO / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in ("gethostname(", "platform.node(", "getfqdn(",
                          "os.uname("):
                if token in text:
                    offenders.append(f"{path.relative_to(REPO)}: {token}")
    assert not offenders, offenders
