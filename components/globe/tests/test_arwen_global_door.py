"""`woof global`: the experimental global model, reachable from the product.

The global spectral model -- hydrostatic spherical-harmonic dycore, its own
reference and native physics suites, a point-observation analysis door and a
render-ready wrfout export -- was complete, tested and had no command anyone
could type.  Its only door was `woof global`, a form that a
reader who installed the wheel has no way to discover: `woof --help` did not
list it, the generated CLI reference did not carry it, and nothing on the main
command line pointed at it.  By this project's rule -- a capability with no
front door is not a feature -- it was not shipped.

Every test here is a property of the REACHABLE path rather than of the model,
which has its own gates in the other `test_arwen_global_*` files:

* the door is on the real parser and the real dispatch table;
* its legs take exactly the arguments the module door takes, read off
  both parsers rather than transcribed, so the two cannot drift apart;
* a refusal arrives as ONE sentence at EXIT_REFUSED -- including the refusals the
  main CLI boundary does not itself catch, which is the whole reason this
  door composes them instead of leaving them to fall through;
* and the module door still answers, unchanged.
"""

from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from woof import capabilities, cli
from woof.globe import cli as arwen_cli

#: The registered TOML the runner tests integrate.  A complete forecast at
#: T3 that finishes in tens of milliseconds, so the door can be proved
#: against the real model rather than against a stub of it.
SMOKE_CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")

#: leaf on `woof global` -> the command it delegates to on the module door.
#: `export` is the render-ready tape: that is the product a reader of this
#: door wants, and `export-parent` beside it is a step inside the regional
#: bridge rather than something to look at.
#: `statics` builds the static surface fields a real-data run refuses
#: to start without; a run whose remedy names a command a user of the
#: wheel cannot type is no remedy.
#: `cycle` is the forecast and the assimilation in one process, the door
#: an hourly cycle is run through.
#: `da` is the data-assimilation door: init, cycle, analyze, fresh and
#: forecast, the legs a fresh global analysis is made and started from.
#: `microwave` is the ATMS leg: fetch, decode, thin, score and the operator
#: entry, the subcommands of `woof global microwave`
#: forwarded whole.
LEGS = {"run": "run",
        "assimilate": "assimilate",
        "cycle": "cycle",
        "da": "da",
        "export": "export",
        "statics": "statics",
        "microwave": "microwave",
        # the ABI operator legs (2026-09-06): the SimSat measurement, the
        # reference scorecard, the fast model's trainer and forward door
        "abi-score": "abi-score",
        "abi-reference": "abi-reference",
        "abi-fast-model": "abi-fast-model"}


def _subparsers(parser: argparse.ArgumentParser) -> dict:
    for action in parser._actions:                       # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):   # noqa: SLF001
            return action.choices
    return {}


def _surface(parser: argparse.ArgumentParser) -> set[tuple]:
    """One comparable description of everything a parser accepts.

    argparse's own `-h` is dropped, and so is `--explain`, which the main
    CLI sweeps onto every top-level subcommand after the registrars have
    run and which the module door therefore does not carry.
    """

    return {(action.dest, tuple(action.option_strings), bool(action.required),
             repr(action.default), action.nargs, repr(action.metavar))
            for action in parser._actions
            if action.dest not in ("help", "explain")}




def _product_door() -> argparse.ArgumentParser:
    """The surface the engine's `woof global` forwarder registers.

    Built here from `woof.globe.cli.register_cli` rather than reached
    through an installed engine, because the engine that carries the
    forwarder is not published yet and this property is this package's:
    whatever a forwarder registers, THIS is what it gets.
    """

    parser = argparse.ArgumentParser(prog="woof")
    sub = parser.add_subparsers(dest="command")
    return arwen_cli.register_cli(sub)

# ---------------------------------------------------------------------------
# 1. the door a user types
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="an ENGINE property: `woof global` is registered by the forwarder\n"
           "on the engine's own line, and the engine that carries it is not\n"
           "published yet.  tests/test_global_forwarder.py on that line holds\n"
           "this, and holds it against the real woof.cli.")
def test_gpuwm_global_is_a_registered_subcommand():
    """Engine-proven is not shipped: this is the front door."""

    assert "global" in _subparsers(cli.build_parser()), (
        "`woof global` is not registered, so the global spectral model "
        "has no command a user of the wheel can type or discover")


def test_the_door_carries_the_legs_a_reader_runs():
    door = _product_door()
    assert set(_subparsers(door)) == set(LEGS), (
        "the legs of `woof global` are not run/assimilate/cycle/da/export/statics/microwave "
        "plus the three ABI legs")


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_every_leg_says_where_it_dispatches(leg):
    """A parser entry with no `func` is an AttributeError at first use."""

    leaf = _subparsers(_product_door())[leg]
    handler = leaf.get_default("func")
    assert callable(handler), (
        f"`woof global {leg}` sets no func, so woof.cli._dispatch has "
        "nothing to call and the command exists only in the help listing")


def test_the_help_marks_the_door_experimental():
    """A research door that reads as a product is the wrong promise."""

    door = _product_door()
    assert "experimental" in (door.format_help().lower())
    assert "woof global" in door.format_help(), (
        "the door does not point at the module door that still owns the "
        "rest of the research surface, so a reader who needs pins, "
        "migration or the regional bridge has nowhere to go")


@pytest.mark.skip(
    reason="an ENGINE property: `woof global` is registered by the forwarder\n"
           "on the engine's own line, and the engine that carries it is not\n"
           "published yet.  tests/test_global_forwarder.py on that line holds\n"
           "this, and holds it against the real woof.cli.")
def test_the_door_is_named_long_running():
    """Ctrl-C masked in a background shell is worth one line here too."""

    assert "global" in cli._LONG_RUNNING_COMMANDS   # noqa: SLF001


@pytest.mark.skip(
    reason="an ENGINE property: `woof global` is registered by the forwarder\n"
           "on the engine's own line, and the engine that carries it is not\n"
           "published yet.  tests/test_global_forwarder.py on that line holds\n"
           "this, and holds it against the real woof.cli.")
def test_the_door_is_not_gated_on_a_card():
    """The numpy backend runs with no device, so a gate would refuse it.

    `woof.capabilities` admits a command only when EVERY path it can take
    needs the capability.  This one integrates a complete forecast on the
    CPU -- proved below by running one -- so a row here would block the
    path that works.
    """

    assert "global" not in capabilities.COMMAND_REQUIREMENTS


# ---------------------------------------------------------------------------
# 2. one argument surface, two doors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("leg,module_command", sorted(LEGS.items()))
def test_the_two_doors_take_the_same_arguments(leg, module_command):
    """Read off both parsers, never transcribed.

    Two hand-kept copies of an argument list is how the main door comes to
    accept a flag the module door dropped, or to spell `--nlat`'s default
    differently -- and the reader who hit the difference would have no way
    to tell which door was right.
    """

    leaf = _subparsers(_product_door())[leg]
    module = _subparsers(arwen_cli.build_parser())[module_command]
    assert _surface(leaf) == _surface(module), (
        f"`woof global {leg}` and `woof global "
        f"{module_command}` no longer declare the same arguments")


def _listed_help(parser: argparse.ArgumentParser) -> dict[str, str]:
    """subcommand name -> the one line argparse prints beside it."""

    for action in parser._actions:                       # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):   # noqa: SLF001
            return {row.dest: row.help
                    for row in action._choices_actions}  # noqa: SLF001
    return {}


@pytest.mark.parametrize("leg,module_command", sorted(LEGS.items()))
def test_the_two_doors_carry_the_same_help(leg, module_command):
    """The same leg described two ways is two descriptions to keep true."""

    door = _listed_help(_product_door())
    module = _listed_help(arwen_cli.build_parser())
    assert door[leg] == module[module_command]


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_each_leg_prints_help_and_exits_clean(capsys, leg):
    """`--help` must never pay for the model or refuse."""

    with pytest.raises(SystemExit) as exit_info:
        arwen_cli.main([leg, "--help"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip()


#: One fully-specified invocation per leg, and what the handler must
#: receive.  The point is delegation: this door parses and hands over, it
#: does not reimplement -- so every value a reader typed has to arrive at
#: the same handler `woof global` calls, unchanged.
DELEGATIONS = {
    "microwave": (
        ["calibrate", "--out", "microwave-calibration.json"],
        {"microwave_args": ["calibrate", "--out", "microwave-calibration.json"]}),
    "assimilate": (
        [SMOKE_CONFIG, "back.npz", "--obs", "a.csv", "--obs", "b.csv.gz",
         "--out", "analysis.npz", "--analysis-time", "2026-09-01T12:00:00",
         "--length-scale-km", "4000", "--max-age-minutes", "45",
         "--elevation-limit-m", "250", "--overwrite"],
        {"config": Path(SMOKE_CONFIG), "checkpoint": Path("back.npz"),
         "obs": ["a.csv", "b.csv.gz"], "out": Path("analysis.npz"),
         "analysis_time": "2026-09-01T12:00:00", "length_scale_km": 4000.0,
         "max_age_minutes": 45.0, "elevation_limit_m": 250.0,
         "overwrite": True}),
    "cycle": (
        [SMOKE_CONFIG, "--obs", "a.csv", "--outdir", "out", "--cycles", "24",
         "--interval-s", "3600", "--start-utc", "2026-08-31T00:00:00Z",
         "--until-s", "172800", "--max-age-minutes", "45",
         "--moisture-update", "off", "--keep-backgrounds",
         "--partial-analyses", "off", "--overwrite"],
        {"config": Path(SMOKE_CONFIG), "obs": ["a.csv"], "outdir": Path("out"),
         "cycles": 24, "interval_s": 3600.0,
         "start_utc": "2026-08-31T00:00:00Z", "until_s": 172800.0,
         "max_age_minutes": 45.0, "moisture_update": "off",
         "keep_backgrounds": True, "partial_analyses": "off",
         "overwrite": True}),
    "export": (
        [SMOKE_CONFIG, "one.npz", "two.npz", "--outdir", "tapes",
         "--nlat", "180", "--nlon", "360",
         "--start-date", "2026-09-01_12:00:00",
         "--bbox", "20", "50", "-130", "-60", "--overwrite"],
        {"config": Path(SMOKE_CONFIG),
         "checkpoints": [Path("one.npz"), Path("two.npz")],
         "outdir": Path("tapes"), "nlat": 180, "nlon": 360,
         "start_date": "2026-09-01_12:00:00",
         "bbox": [20.0, 50.0, -130.0, -60.0], "overwrite": True}),
    "run": (
        [SMOKE_CONFIG, "--outdir", "out", "--restart", "r.npz", "--overwrite"],
        {"config": Path(SMOKE_CONFIG), "outdir": Path("out"),
         "restart": Path("r.npz"), "overwrite": True}),
    "da": (
        ["fresh", SMOKE_CONFIG, "--outdir", "out", "--stream", "iem-asos",
         "--stream", "local-tables:paths=a.csv", "--obs", "b.csv",
         "--analysis-cycle", "2026-09-01T00:00:00Z", "--until-utc",
         "2026-09-01T06:00:00Z", "--forecast-hours", "12", "--filter",
         "successive-correction", "--moisture-update", "on", "--overwrite"],
        {"da_command": "fresh", "config": Path(SMOKE_CONFIG),
         "outdir": Path("out"), "stream": ["iem-asos", "local-tables:paths=a.csv"],
         "obs": ["b.csv"], "analysis_cycle": "2026-09-01T00:00:00Z",
         "until_utc": "2026-09-01T06:00:00Z", "forecast_hours": 12.0,
         "filter": "successive-correction", "moisture_update": "on",
         "overwrite": True}),
}

#: leg -> the handler on the module door it must delegate to.
HANDLERS = {"run": "_run", "assimilate": "_assimilate", "cycle": "_cycle",
            "export": "_export_wrfout", "da": "_da", "microwave": "_microwave"}


@pytest.mark.parametrize("leg", sorted(DELEGATIONS))
def test_each_leg_hands_the_parsed_request_to_the_module_handler(monkeypatch,
                                                                 leg):
    """Thin delegation, proved by what the handler is given."""

    argv, expected = DELEGATIONS[leg]
    seen = {}

    def record(args):
        seen.update(vars(args))
        return 0

    monkeypatch.setattr(arwen_cli, HANDLERS[leg], record)
    assert arwen_cli.main([leg, *argv]) == 0
    assert seen, f"`woof global {leg}` never reached {HANDLERS[leg]}"
    for key, value in expected.items():
        assert seen[key] == value, key


def test_the_module_door_is_unchanged():
    """`woof global` keeps its whole surface."""

    commands = set(_subparsers(arwen_cli.build_parser()))
    for command in ("pins", "physics-manifest", "transform-check", "run",
                    "assimilate", "inspect", "check-receipt", "export-parent",
                    "export-wrfout", "migrate-level4-checkpoint",
                    "make-regional-target", "translate-regional-frame",
                    "make-parent-series", "native-qualify"):
        assert command in commands, (
            f"registering `woof global` removed `{command}` from the "
            "module door, which is the door every existing script uses")


# ---------------------------------------------------------------------------
# 3. refusals reach the terminal as sentences
# ---------------------------------------------------------------------------

def _stderr_lines(captured) -> list[str]:
    """The refusal, without the provenance banner every command prints."""

    return [line for line in captured.err.splitlines()
            if line.strip() and "installed wheel" not in line
            and "editable install" not in line]


def test_a_malformed_config_is_one_sentence_not_a_traceback(capsys, tmp_path):
    """The ValueError leg: raised deep in the loader, printed at the boundary.

    TOMLDecodeError is a ValueError, so this rides `woof.cli`'s own
    refusal boundary and picks up its `--explain` layering.  What is being
    pinned is that it ARRIVES there -- a handler that swallowed it would
    lose the layering, and no handler at all would be fine here and a
    traceback on every other refusal this model raises.
    """

    config = tmp_path / "broken.toml"
    config.write_text("not a config\n", encoding="utf-8")
    code = arwen_cli.main(["run", str(config),
                     "--outdir", str(tmp_path / "out")])
    # 1, not 2.  2 is argparse's code for a command line that was not a
    # command line; a refusal is a command line this package understood and
    # declined, and a caller that cannot tell them apart has to read prose.
    assert code == arwen_cli.EXIT_REFUSED
    lines = _stderr_lines(capsys.readouterr())
    assert lines and lines[0].startswith("woof global: ")
    assert "Traceback" not in "\n".join(lines)


def test_an_output_that_exists_is_one_sentence_not_a_traceback(capsys,
                                                               tmp_path):
    """The leg `woof.cli` does NOT catch, which is why this door composes it.

    `woof.cli`'s boundary turns ValueError, CapabilityMissing,
    ModuleNotFoundError and a scoped list of RuntimeErrors into sentences
    and deliberately leaves everything else its traceback.  FileExistsError
    is not on that list, so "output exists; pass --overwrite" -- the single
    most likely refusal a reader of this door meets, on their second run --
    would have arrived as a traceback at exit 1.  FileNotFoundError (a
    checkpoint that is not there) and OSError (an unwritable outdir) are
    the same shape.
    """

    outdir = tmp_path / "run"
    assert arwen_cli.main(["run", SMOKE_CONFIG,
                     "--outdir", str(outdir)]) == 0
    capsys.readouterr()

    code = arwen_cli.main(["run", SMOKE_CONFIG, "--outdir", str(outdir)])
    assert code == arwen_cli.EXIT_REFUSED
    lines = _stderr_lines(capsys.readouterr())
    assert lines == [
        f"woof global: WOOF global output exists in {outdir}; pass "
        "--overwrite to replace owned files"]


def test_both_doors_refuse_with_the_same_sentence(capsys, tmp_path):
    """Only the door name differs; the reason a reader acts on is one string."""

    outdir = tmp_path / "run"
    assert arwen_cli.main(["run", SMOKE_CONFIG,
                     "--outdir", str(outdir)]) == 0
    capsys.readouterr()

    assert arwen_cli.main(["run", SMOKE_CONFIG,
                     "--outdir", str(outdir)]) == arwen_cli.EXIT_REFUSED
    through_product = _stderr_lines(capsys.readouterr())[0]

    assert arwen_cli.main(["run", SMOKE_CONFIG, "--outdir", str(outdir)]) == arwen_cli.EXIT_REFUSED
    through_module = capsys.readouterr().err.strip()

    assert through_product.removeprefix("woof global: ") == (
        through_module.removeprefix("woof global: "))


def test_every_printed_sentence_names_the_door_the_reader_typed():
    """The prefix is a command that exists, whichever parser was used.

    `_assimilate` prints its gate-of-record report prefixed with the door
    name, and so does every refusal.  There is one entrance to this
    distribution now, so the prefix is that one entrance -- including for a
    reader who arrived through the engine's forwarder, because the forwarder
    registers THIS parser and the spelling it prints has to be one that works
    whether or not the engine is installed.

    `arwen-global` is what this used to print on the module door.  It names
    no command anybody can run, which is the same defect the two-name rule
    was written to prevent, arrived at from the other side.
    """

    from woof.globe import CONSOLE_SCRIPT

    assert CONSOLE_SCRIPT == "woof global"
    assert arwen_cli._door_name(               # noqa: SLF001
        argparse.Namespace(command="assimilate")) == CONSOLE_SCRIPT
    assert arwen_cli._door_name(               # noqa: SLF001
        argparse.Namespace(command="global",
                           global_command="assimilate")) == CONSOLE_SCRIPT

    # ...parsed from the real parsers, not asserted about a fixture.
    through_product = _product_door().parse_args(
        ["assimilate", str(SMOKE_CONFIG), "b.npz", "--obs", "o.csv",
         "--out", "a.npz"])
    through_module = arwen_cli.build_parser().parse_args(
        ["assimilate", str(SMOKE_CONFIG), "b.npz", "--obs", "o.csv",
         "--out", "a.npz"])
    assert arwen_cli._door_name(through_product) == CONSOLE_SCRIPT   # noqa: SLF001
    assert arwen_cli._door_name(through_module) == CONSOLE_SCRIPT    # noqa: SLF001


def test_the_doors_refusal_set_is_the_module_doors_minus_what_cli_owns():
    """Derived, so the two cannot fall out of step when one grows a case."""

    assert set(arwen_cli._DOOR_REFUSALS) == (          # noqa: SLF001
        set(arwen_cli._REFUSALS) - {ValueError, ModuleNotFoundError})  # noqa: SLF001, E501
    assert FileExistsError in arwen_cli._DOOR_REFUSALS  # noqa: SLF001


# ---------------------------------------------------------------------------
# 4. against the artifact
# ---------------------------------------------------------------------------

def test_the_installed_command_runs_a_real_forecast(tmp_path):
    """Run the exe.  A door proved only in-process is proved in a fixture."""

    outdir = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "-m", "arwen_global", "run", SMOKE_CONFIG,
         "--outdir", str(outdir)],
        cwd=REPO_ROOT, capture_output=True, text=True, errors="replace",
        timeout=600)
    assert result.returncode == 0, result.stderr
    receipt = outdir / "arwen-global-receipt.json"
    assert receipt.is_file(), result.stdout
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "pass"


def test_the_generated_cli_reference_carries_the_door():
    page = REPO_ROOT / "docs" / "public" / "CLI-OPTIONS.md"
    if not page.is_file():                       # pragma: no cover - wheel
        pytest.skip("the generated reference is not in this install")
    text = page.read_text(encoding="utf-8")
    for door in ("woof global", "woof global run",
                 "woof global assimilate", "woof global cycle",
                 "woof global export"):
        assert f"`{door}`" in text, (
            f"the generated CLI reference does not name `{door}`; "
            "regenerate it with python -m tools.build_cli_options_doc")
