"""The global model's demo leg: one documented route, and the config it ships.

`woof global` gained a front door, and a door nobody can find their way to
is the same as no door -- so this file holds the ROUTE together rather than
any one command in it.  Four things have to agree, and each of them used to
be free to move on its own:

* ``configs/arwen_global_t255_quickstart.toml`` -- the experiment,
* ``docs/ARWEN_GLOBAL_QUICKSTART.md`` -- the page that runs it,
* :func:`woof.globe.analysis_fetch.gdas_fetch_epilogue` -- what the
  acquisition door says comes next, which is the only place a reader meets
  the route without having found the page first,
* and the three model doors themselves.

The agreement that carries the route is the FILE NAME.  The fetch door lands
``gdas.tHHz.pgrb2.0p25.f000``; the shipped config reads a path; the page
tells a reader to type both.  Nothing but this file connects the three, and
a config pointing at a file the documented fetch never writes is a quickstart
that refuses on its first command.

The end-to-end test at the bottom runs the real doors as subprocesses on the
shipped smoke config -- run, export, render, including the real Rust renderer
binary -- because "these three commands compose" is the entire claim of a
demo leg, and it is not a claim any amount of unit coverage makes.
"""

from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from datetime import datetime
from pathlib import Path
import os
import re
import shlex
import subprocess
import sys
import tomllib

import pytest

from woof import cli, fetch
from woof.globe import analysis_fetch
from woof.globe.analysis_initial import resolve_analysis_mapping
from woof.globe.config import load_config

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_CONFIG = _shipped_configs() / "arwen_global_t255_quickstart.toml"
# The page lives at the top of `docs/` in this distribution.  It was
# `docs/public/` in the tree this file was carved from, where `docs/`
# also held working records that were not for a reader; this repository
# publishes nothing else, so the extra level named nothing.
QUICKSTART_DOC = ROOT / "docs" / "ARWEN_GLOBAL_QUICKSTART.md"
SMOKE_CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")

#: The cycle the shipped config is written against, and the one every
#: documented command in the page spells.
CYCLE = datetime(2026, 8, 30, 18)


def _fenced_commands(path: Path) -> list[str]:
    """Every shell command in a page's fenced blocks, continuations joined."""

    commands: list[str] = []
    pending: list[str] = []
    inside = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            inside = line.strip() == "```bash"
            continue
        if not inside or not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.endswith("\\"):
            pending.append(stripped[:-1].strip())
            continue
        pending.append(stripped)
        commands.append(" ".join(pending))
        pending = []
    assert not pending, f"{path.name} ends inside a line continuation"
    return commands


def test_the_quickstart_config_is_a_whole_t255_day() -> None:
    """The arithmetic the header claims is the arithmetic the loader reads."""

    cfg = load_config(QUICKSTART_CONFIG)
    assert cfg.truncation == 255
    assert cfg.vertical_coordinate == "surface_stretched"
    assert cfg.vertical.nlev == 40
    assert (cfg.backend, cfg.precision) == ("cupy", "float32")
    assert cfg.duration_s == 86400.0
    # Whole steps, and an output cadence that is a whole number of them --
    # otherwise the nine checkpoints the page promises do not land on
    # output times and the last one is not hour 24.
    # 300 s on the semi-Lagrangian core, the shipped default (2026-09-06)
    steps = cfg.duration_s / cfg.dt_s
    cadence = cfg.output_interval_s / cfg.dt_s
    assert steps == int(steps) == 288
    assert cadence == int(cadence) == 36
    assert steps % cadence == 0
    assert int(steps // cadence) + 1 == 9


def test_the_quickstart_config_runs_from_wherever_it_was_installed() -> None:
    """No path only this machine has, and no case name anywhere.

    A config that ships in the wheel and names ``C:/Users/<somebody>`` is a
    config every reader but its author gets a FileNotFoundError from.  The
    only paths admitted here are relative ones.
    """

    from tools.check_case_token_leakage import CASE_TOKENS

    text = QUICKSTART_CONFIG.read_text(encoding="utf-8")
    raw = tomllib.loads(text)

    def strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for value in node.values():
                yield from strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from strings(value)

    for value in strings(raw):
        assert not re.match(r"^[A-Za-z]:[\\/]", value), \
            f"{value!r} is an absolute path on one particular machine"
        assert not value.startswith(("/", "~", "\\\\")), \
            f"{value!r} is not a relative path"
        assert "\\" not in value, \
            f"{value!r} spells a separator TOML reads as an escape"

    lowered = text.lower()
    for token in CASE_TOKENS:
        assert token not in lowered, \
            f"case token {token!r} reached a shipped generic config"


def test_the_quickstart_mapping_is_a_bare_id_that_ships_in_the_wheel() -> None:
    """The one spelling that resolves from an install, not just a checkout.

    ``woof/authorities/rw-wps-...json`` is a CHECKOUT-relative path and
    resolves only when the reader's working directory is the repository root;
    the sibling configs under ``configs/verify/`` still spell it that way and
    are run from there.  A quickstart is typed by someone who ran `pip
    install`, so it names the bare id and the resolver globs the packaged
    authorities directory for it.
    """

    cfg = load_config(QUICKSTART_CONFIG)
    assert cfg.analysis_mapping == analysis_fetch.analysis_mapping_id()
    assert "/" not in cfg.analysis_mapping

    resolved = resolve_analysis_mapping(cfg.analysis_mapping)
    assert resolved.name == "rw-wps-gdas-global-analysis-grib2.mapping.json"
    # It ships: package-data claims `data/**/*`, and the carried copies are
    # the directory the resolver globs when the engine's table has no row,
    # so an installed wheel answers the same.
    assert resolved.parent.name == "authorities"

    # And the family id is still ambiguous, which is the reason the config
    # has to be specific rather than a stylistic preference.
    with pytest.raises(ValueError, match="exactly one"):
        resolve_analysis_mapping("gdas")


def test_an_analysis_config_that_names_no_mapping_still_resolves(
        tmp_path) -> None:
    """Fixed means default: the fallback has to be a mapping that EXISTS.

    ``initial.analysis_mapping`` defaulted to ``"gdas"``, which was unique
    when it was written and now globs three authority mappings -- the global
    analysis, the regional pgrb2 profile and its donor.  Every config that
    left the key out therefore refused before step zero with `analysis
    mapping id 'gdas' must match exactly one authority mapping, found 3`.
    A default that cannot resolve is not a default.
    """

    text = QUICKSTART_CONFIG.read_text(encoding="utf-8")
    without = "\n".join(
        line for line in text.splitlines()
        if not line.strip().startswith("analysis_mapping"))
    path = tmp_path / "no-mapping.toml"
    path.write_text(without + "\n", encoding="utf-8")

    cfg = load_config(path)
    assert cfg.analysis_mapping == analysis_fetch.analysis_mapping_id()
    assert resolve_analysis_mapping(cfg.analysis_mapping).exists()


def test_the_fetch_hint_and_the_shipped_config_name_the_same_file() -> None:
    """The route's one seam, held from both sides.

    The acquisition door prints an ``[initial]`` block a reader pastes.  If
    it names a different file, or a different mapping, than the config this
    repository ships, then following the printed hint and following the page
    produce two different runs and one of them refuses.

    The hint is this package's own (`woof.globe.analysis_fetch`) because a
    published engine carries neither name: on `pip install recast-woof
    woof global` the engine's GDAS fetch ended in a full stop about a
    regional route and said nothing about the model the reader installed.
    The engine's version still wins the day it exists, and this holds
    whichever one answered.
    """

    cfg = load_config(QUICKSTART_CONFIG)
    out = Path(cfg.analysis_grib).parent

    lines = analysis_fetch.gdas_fetch_epilogue(
        out=out, cycle=CYCLE, whole_globe_analysis=True)
    body = "\n".join(lines)

    # The hint's TOML is TOML: it is pasted into a config file, so a
    # Windows separator in it would be an invalid escape rather than a path.
    pasted = tomllib.loads("\n".join(
        line.split("#", 1)[1].strip()
        for line in body.splitlines() if line.strip().startswith("#   ")))
    assert pasted["initial"] == {
        "mode": "analysis",
        "analysis_grib": cfg.analysis_grib,
        "analysis_mapping": cfg.analysis_mapping,
    }

    # The command it prints is this door, and the config it names is this
    # file -- both spelled from the same constants the hint is built from.
    # The config is named by the SHIPPED NAME, which resolves from an install
    # as well as from a checkout; a repository-relative path would be typed
    # by someone who has the repository.
    assert "woof global run" in body
    assert analysis_fetch.GDAS_GLOBAL_QUICKSTART_CONFIG in body
    assert analysis_fetch.GDAS_GLOBAL_QUICKSTART_DOC in body
    assert (_shipped_configs()
            / f"{analysis_fetch.GDAS_GLOBAL_QUICKSTART_CONFIG}.toml") \
        == QUICKSTART_CONFIG
    assert (ROOT / analysis_fetch.GDAS_GLOBAL_QUICKSTART_DOC) == QUICKSTART_DOC


def test_the_hint_names_the_file_the_full_file_transport_really_writes(
) -> None:
    """Not a guess about the name: the transport's own format string.

    ``_fetch_gfs_fullfile_locked`` names each object
    ``f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}"`` with the prefix the
    container table gives the source.  The hint, the config and that
    expression have to produce one string.
    """

    cfg = load_config(QUICKSTART_CONFIG)
    prefix = fetch.GFS_CONTAINER_PREFIX["gdas"]
    expected = (Path(cfg.analysis_grib).parent
                / f"{prefix}.t{CYCLE:%H}z.pgrb2.0p25.f{0:03d}").as_posix()
    assert cfg.analysis_grib == expected


def test_a_cropped_fetch_does_not_print_a_next_step_that_refuses() -> None:
    """The stop for a crop is not retired, it is narrowed.

    A default GDAS fetch is a NOMADS area crop, and a crop cannot initialize
    a global model -- ``analysis_initial`` refuses a longitude ring that does
    not close.  Printing a `woof global run` next step there would send a
    reader to a refusal, which is the exact thing the original full stop
    existed to avoid, and the crop's own sentence says why by name.
    """

    cropped = "\n".join(analysis_fetch.gdas_fetch_epilogue(
        out=Path("data/gdas"), cycle=CYCLE, whole_globe_analysis=False))
    whole = "\n".join(analysis_fetch.gdas_fetch_epilogue(
        out=Path("data/gdas"), cycle=CYCLE, whole_globe_analysis=True))

    assert "woof global run" not in cropped
    assert "woof global run" in whole
    assert "longitude ring" in cropped


def test_the_documented_route_is_four_real_doors() -> None:
    """Every command on the page resolves, and every flag on it exists."""

    sys.path.insert(0, str(ROOT / "tests"))
    from doc_command_parity import door_options, doors, resolve_door

    known = doors()
    seen: list[str] = []
    for command in _fenced_commands(QUICKSTART_DOC):
        resolved = resolve_door(command, known)
        if resolved is None:
            continue
        door, body = resolved
        assert door in known, \
            f"{QUICKSTART_DOC.name} runs {door!r}, which no parser defines"
        options = door_options(known[door])
        for flag in re.findall(r"(?<![\w-])(--[A-Za-z][\w-]*)", body):
            assert flag in options, \
                f"{door} has no {flag}; the page prints `{body}`"
        seen.append(door)

    # The steps the page is FOR, in the order it prints them.  The first is
    # this package's own acquisition door: a published engine's `woof fetch`
    # ends a whole-globe GDAS fetch with a full stop about a REGIONAL route
    # and says nothing about the model the reader installed, so the route's
    # first step is a door of this package that binds the two non-default
    # flags and prints what comes next.  The engine's spelling stays on the
    # page right under it, and is still graded.
    assert seen[:5] == [
        "woof global fetch-analysis", "woof fetch", "woof global run",
        "woof global export", "woof render",
    ]
    # The sizing step is PROSE, not a fenced command, and that is deliberate.
    # It used to print `woof check configs/<name>.toml`, which an installed
    # reader cannot run twice over: the path is a directory in a checkout they
    # do not have, and `woof check` is the ENGINE's door, which does not
    # resolve this package's shipped experiment names either, so rewriting the
    # path as a bare name does not save the line (measured 2026-09-07 from the
    # installed wheel).  `run` prices the card itself and refuses before it
    # allocates, so the step is a paragraph saying that, and `woof check` is
    # named as taking a file.  A fenced `woof check` on this page would be a
    # line the reader cannot run, which is what this test exists to stop.
    assert "woof check" not in seen, (
        "the quickstart prints a fenced `woof check` line again; from an "
        "installed wheel that door resolves neither a checkout path nor a "
        "shipped experiment name")
    body = QUICKSTART_DOC.read_text(encoding="utf-8")
    assert "`woof check`" in body, (
        "the sizing paragraph no longer mentions the engine's check door")


def test_the_documented_fetch_command_is_the_one_the_config_needs() -> None:
    """The acquisition line parses, and its two non-default flags are there.

    ``--mode full-file`` and ``--hours 0`` are the whole reason this fetch
    differs from every other GDAS fetch on the docs site, and dropping either
    one produces bytes the global initializer refuses.
    """

    parser = cli.build_parser()
    fetch_line = next(
        command for command in _fenced_commands(QUICKSTART_DOC)
        if command.startswith("woof fetch "))
    args = parser.parse_args(shlex.split(fetch_line)[1:])

    assert args.source == "gdas"
    assert args.hours == 0
    assert args.mode == "full-file"
    assert args.area is None and args.point is None

    cfg = load_config(QUICKSTART_CONFIG)
    assert Path(args.out) == Path(cfg.analysis_grib).parent
    assert fetch.parse_cycle(args.cycle, "gdas") == CYCLE


def test_the_three_doors_compose_from_run_to_rendered_image(tmp_path) -> None:
    """The demo leg's only real claim, run as three real subprocesses.

    Smoke truncation, numpy backend, the real ``rw_wrfbatch``: this is about
    whether ``run`` -> ``export`` -> ``render`` compose at all, which no
    mock can answer.  A four-step T3 atmosphere has nothing in it worth
    looking at, and that is fine -- the image is evidence that the tape is
    readable and the frame is global, not evidence of a forecast.
    """

    from woof import render

    engine, why = render.drawable_engine()
    if engine is None:
        pytest.skip(f"no chainable render engine on this box: {why}")

    env = {**os.environ, "GPUWM_NO_LOCAL_GPU": "1"}
    outdir = tmp_path / "run"
    tapes = tmp_path / "tapes"
    images = tmp_path / "png"

    def door(*argv: str) -> subprocess.CompletedProcess:
        done = subprocess.run(
            [sys.executable, "-m", "arwen_global", *argv],
            cwd=ROOT, capture_output=True, text=True, env=env, timeout=900)
        assert done.returncode == 0, done.stdout + done.stderr
        return done

    door("run", SMOKE_CONFIG, "--outdir", str(outdir))
    checkpoints = sorted(outdir.glob("arwen_global_step*.npz"))
    assert len(checkpoints) == 3, [path.name for path in checkpoints]
    assert (outdir / "arwen-global-receipt.json").is_file()

    door("export", SMOKE_CONFIG, *[str(p) for p in checkpoints],
         "--outdir", str(tapes), "--nlat", "90", "--nlon", "180",
         "--start-date", "2026-08-30_18:00:00")
    written = sorted(tapes.glob("wrfout_d01_*"))
    assert len(written) == len(checkpoints), [p.name for p in written]

    # `--outdir`, not `--out`, and the valid time is required: the render
    # door offsets every checkpoint from it, and a picture whose valid time
    # was guessed is a picture nobody can verify against an observation.
    done = door("render", str(written[0]), "--products", "2m_temperature",
                "--outdir", str(images),
                "--start-date", "2026-08-30_18:00:00")
    drawn = sorted(images.rglob("*.png"))
    assert drawn, done.stdout + done.stderr
    # The layout law, and the door's own --help sentence for it:
    # <outdir>/<domain>/<product>/<valid-day>/<picture>.  `--outdir` IS the
    # case folder in this distribution -- the reader names it on the command
    # line rather than having one invented inside it -- so the domain is the
    # first level below it.  A flat drop would mean the global tape took a
    # different path through the renderer than every other wrfout.
    relative = drawn[0].relative_to(images).parts
    assert len(relative) == 4, relative
    assert relative[0].startswith("d01"), relative
    assert relative[1] == "2m_temperature", relative
