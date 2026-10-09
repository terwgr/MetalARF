"""``woof update`` prints the upgrade and executes nothing.

CPU-only.  The distribution lookup is stubbed both ways -- installed and
absent -- so the two branches are exercised without depending on how the
test machine happens to have woof on its path.
"""

import subprocess
import sys
from types import SimpleNamespace

import woof.cli as cli
import woof.update_cli as update_cli


def _installed(name="woof"):
    return SimpleNamespace(metadata={"Name": name}, version="1.5.0")


def test_update_is_a_registered_subcommand():
    """The word the field report typed reaches a handler, not a usage
    error.  `invalid choice: 'update'` on the shipped 1.5.0 wheel is what
    this command exists to answer."""

    choices = next(action.choices
                   for action in cli.build_parser()._actions
                   if getattr(action, "choices", None)
                   and "doctor" in action.choices)
    assert "update" in choices


def test_update_names_the_resolved_distribution_and_this_interpreter(
        monkeypatch, capsys):
    monkeypatch.setattr(
        "woof.runtime_manifest.installed_distribution",
        lambda package="woof": _installed())
    assert cli.main(["update"]) == 0
    out = capsys.readouterr().out
    assert "pip install --upgrade recast-woof" in out
    # This interpreter, not a bare `pip`: a machine with several
    # environments has several pips and only one of them owns this one.
    assert sys.executable in out
    assert "-m pip install --upgrade recast-woof" in out
    assert update_cli.upgrade_command("woof") in out


def test_update_upgrades_the_distribution_that_actually_provides_gpuwm(
        monkeypatch, capsys):
    """Two distributions publish this package; printing the public name
    to somebody who installed the other one is an instruction that adds a
    second copy instead of upgrading theirs."""

    monkeypatch.setattr(
        "woof.runtime_manifest.installed_distribution",
        lambda package="woof": _installed("rw-wps"))
    assert cli.main(["update"]) == 0
    out = capsys.readouterr().out
    assert "pip install --upgrade rw-wps" in out
    assert "--upgrade woof" not in out


def test_update_says_so_when_there_is_no_distribution_to_upgrade(
        monkeypatch, capsys):
    monkeypatch.setattr(
        "woof.runtime_manifest.installed_distribution",
        lambda package="woof": None)
    assert cli.main(["update"]) == 0
    out = capsys.readouterr().out
    assert "no installed distribution provides" in out.lower() \
        or "which no installed distribution provides" in out
    assert "pip install --upgrade recast-woof" not in out


def test_update_reports_the_asset_directories_an_upgrade_preserves(
        monkeypatch, capsys):
    """The field report confirmed staged downloads survived 1.4.1 ->
    1.5.0; a reader who does not know that pays for the download again."""

    from woof.bridges import default_bridge_dir
    from woof.physics_compat import user_thompson_table_root

    monkeypatch.setattr(
        "woof.runtime_manifest.installed_distribution",
        lambda package="woof": _installed())
    assert cli.main(["update"]) == 0
    out = capsys.readouterr().out
    assert str(default_bridge_dir()) in out
    assert str(user_thompson_table_root()) in out


def test_update_runs_no_subprocess_and_writes_nothing(monkeypatch, tmp_path):
    """PRINT-ONLY is the contract: pip is never launched from inside the
    process whose import path pip would be rewriting."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "woof.runtime_manifest.installed_distribution",
        lambda package="woof": _installed())

    def forbidden(*args, **kwargs):
        raise AssertionError("woof update executed a subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    assert cli.main(["update"]) == 0
    assert list(tmp_path.iterdir()) == []
