"""Package data is checked against the INSTALLED wheel, not the source tree.

`tests/test_packaging_declaration.py` checks that every data file under
`src/` is selected by a package-data glob.  That is the declaration.  This
file checks the other half, which is the half that actually bites: that the
files are THERE after an install, reachable through the same accessors the
doors use, from a working directory that is not the repository.

A source tree can be right and the wheel still wrong -- a glob that matches
in `setup.py`'s walk and not in the wheel writer's, a directory that is not a
package and so is never visited, a data file shadowed by an entry in
`.gitignore`.  Every one of those produces an install that imports and then
cannot find its own configs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("arwen_global", reason="the package under test")

from woof.globe import configs_dir  # noqa: E402
from woof.globe.doors import companion_pins_path  # noqa: E402


#: Where the experiments live in the checkout.  The comparison below needs a
#: referent OUTSIDE the install; without one it compares the installed tree
#: against itself and both sides move together.
_SOURCE_CONFIGS = (Path(__file__).resolve().parents[1]
                   / "src" / "arwen_global" / "configs")


def test_the_configs_are_reachable_from_the_installed_package():
    """The wheel carries every experiment the checkout has.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-10 on a Linux CPU host: a
    wheel built at this tip, installed into a from-scratch venv, then stripped
    of 53 of its 55 shipped TOMLs.  A comparison whose two sides both read
    ``config_root()`` passed that install in 0.66 s, because removing a file
    removes it from both sides at once.  Only a referent outside the install
    -- the checkout's own ``src/arwen_global/configs`` -- can see a file that
    left.

    A hard count is not the answer either: this assertion said 54 while 55
    shipped, so for one recut it failed on every install of a correct wheel.
    The check is a comparison against the source tree, and the count is
    whatever both sides agree on.
    """

    root = configs_dir.config_root()
    assert root.is_dir(), (
        f"the shipped experiments are missing from this install ({root}); the "
        "wheel did not carry its own package data")
    names = sorted(configs_dir.list_configs())
    on_disk = sorted(path.stem for path in root.glob("*.toml"))
    assert names == on_disk, (
        f"{len(names)} experiments resolve but {len(on_disk)} .toml files are "
        "installed beside them")
    assert names, "the installed package ships no experiments at all"

    if not _SOURCE_CONFIGS.is_dir():
        pytest.skip(
            "no checkout beside this test (a wheel-only run of the suite); "
            "the installed tree agrees with itself, which is all that can be "
            f"asked without {_SOURCE_CONFIGS}")
    expected = sorted(path.stem for path in _SOURCE_CONFIGS.glob("*.toml"))
    assert expected, f"the checkout ships no experiments at all ({_SOURCE_CONFIGS})"
    missing = sorted(set(expected) - set(names))
    extra = sorted(set(names) - set(expected))
    assert not missing and not extra, (
        f"the install at {root} carries {len(names)} experiments and the "
        f"checkout at {_SOURCE_CONFIGS} has {len(expected)}; missing from the "
        f"install: {missing or 'none'}; present only in the install: "
        f"{extra or 'none'}")


def test_a_bare_experiment_name_resolves_without_a_checkout(tmp_path, monkeypatch):
    """The reachability claim, tested from somewhere that is not the repo."""

    monkeypatch.chdir(tmp_path)
    name = "arwen_global_gdas_t255_native_24h"
    resolved = configs_dir.resolve_config(name)
    assert resolved.is_file()
    assert resolved.name == f"{name}.toml"
    assert resolved.parent == configs_dir.config_root()


def test_a_file_on_disk_wins_over_a_shipped_name_of_the_same_spelling(tmp_path, monkeypatch):
    """A reader who edits a copy runs their copy.  Always."""

    monkeypatch.chdir(tmp_path)
    mine = tmp_path / "arwen_global_gdas_t255_native_24h.toml"
    mine.write_text("# my edit\n", encoding="utf-8")
    resolved = configs_dir.resolve_config(mine.name)
    # The path is handed back as the reader spelled it -- relative stays
    # relative -- so the comparison resolves both sides.
    assert resolved.resolve() == mine.resolve()
    assert resolved.parent.resolve() != configs_dir.config_root()


def test_a_missing_name_names_the_shipped_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        configs_dir.resolve_config("no_such_experiment")
    message = str(caught.value)
    assert "woof global configs" in message
    assert f"{len(configs_dir.list_configs())} experiments" in message


def test_the_kernel_source_is_reachable_from_the_installed_package():
    import woof.globe

    kernels = Path(woof.globe.__file__).parent / "semilag" / "kernels.cu"
    assert kernels.is_file(), (
        "the semi-Lagrangian kernel source is the only non-Python file inside "
        "this package's modules; without it the semi-Lagrangian core -- the "
        "DEFAULT core -- has nothing to compile")
    assert kernels.stat().st_size > 20_000


def test_the_door_pins_are_reachable_from_the_installed_package():
    pins = companion_pins_path()
    assert pins.is_file(), (
        f"the companion bundle's pins are missing from this install ({pins}); "
        "fetch-doors would have nothing to verify a download against")
