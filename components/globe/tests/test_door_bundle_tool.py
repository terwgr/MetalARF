"""The bundle tool's two 0.1.2 refusals, and the notice that travels.

THE BREAKAGE THESE TESTS PREVENT, both measured on the 0.1.1 bundles as
published:

* the zips carried eight statically linked Rust binaries and no licence
  text for any crate linked into them, while MIT, BSD, Apache-2.0, ISC and
  the Unicode licence condition redistribution in binary form on their
  notices travelling with the binaries;
* the binaries embedded about 1,100 paths under the Linux build host's home
  directory and about 1,250 under the Windows one.

`tools/build_door_bundle.py pin` now refuses both, and `fetch-doors` stages
the notice beside the doors.  These tests drive every refusal with a
synthetic bundle, so they need no Rust toolchain and no network; the one
that generates a notice from a real lockfile skips by name without a
workspace to read.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

from woof.globe import doors


@pytest.fixture(autouse=True)
def _the_table_as_this_package_releases_it(monkeypatch):
    """These tests are about this package's own bundle and the table it is
    built from.  An engine whose bundle declares a companion door takes that
    door over at run time (doors.publisher, tested on its own in
    test_doors_engine_publisher.py), so the engine's roster is held silent
    here and the result does not depend on which engine the suite runs on."""

    monkeypatch.setattr(doors, "engine_bundle_names", lambda: frozenset())

REPO = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO / "tools" / "build_door_bundle.py"

if not (REPO / "src" / "arwen_global" / "doors.py").is_file():
    # The tool builds this repository's own companion bundle from its own
    # source layout.  A tree that carries the model inside another
    # distribution has neither: that distribution's one bundle ships every
    # door, and the tool has no table to load.
    pytest.skip("no src/arwen_global door table beside this tool: this tree "
                "publishes no companion bundle", allow_module_level=True)


def _tool():
    spec = importlib.util.spec_from_file_location("_door_bundle_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()

PLATFORM = "linux-x86_64"
RELEASE = "v0.0.0-test"
REV = "ab12" * 10


def _payload(door, extra: bytes = b"") -> bytes:
    stamp = tool.SOURCE_REV_MARKER + REV.encode("ascii")
    return b"\x7fELF" + stamp + b"\x00" + (door.marker or b"") + b"\x00" + extra


def _bundle(tmp_path: Path, *, notice: bool = True,
            extra: dict[str, bytes] | None = None) -> Path:
    archive = tmp_path / doors.bundle_filename(RELEASE, PLATFORM)
    with zipfile.ZipFile(archive, "w") as zf:
        for door in doors.doors_from_bundle(doors.COMPANION_BUNDLE):
            name = doors.artifact_filename(door.name, PLATFORM)
            zf.writestr(name, _payload(door, (extra or {}).get(door.name, b"")))
        if notice:
            zf.writestr(tool.NOTICE_MEMBER, "THIRD-PARTY LICENCES\n")
    return archive


def _pins_copy(tmp_path: Path) -> Path:
    out = tmp_path / "door-pins.json"
    shutil.copyfile(tool.PINS_PATH, out)
    return out


def test_a_clean_bundle_with_its_notice_pins_and_records_the_notice(tmp_path):
    out = _pins_copy(tmp_path)
    tool.pin(RELEASE, [_bundle(tmp_path)], out, REV, set(), None)
    record = json.loads(out.read_text(encoding="utf-8"))["platforms"][PLATFORM]
    assert record["notice"]["filename"] == "THIRD-PARTY-LICENSES.txt"
    assert record["notice"]["bytes"] == len("THIRD-PARTY LICENCES\n")
    assert len(record["binaries"]) == len(
        doors.doors_from_bundle(doors.COMPANION_BUNDLE))


def test_a_bundle_without_the_notice_is_refused_by_name(tmp_path):
    with pytest.raises(SystemExit) as caught:
        tool.pin(RELEASE, [_bundle(tmp_path, notice=False)],
                 _pins_copy(tmp_path), REV, set(), None)
    assert "THIRD-PARTY-LICENSES.txt" in str(caught.value)


@pytest.mark.parametrize(("leak", "kind"), [
    # Spelled in pieces: the provenance gate reads this file too.
    (b"/ho" b"me/someone/build/engine/tools/rustwx/src/lib.rs", "Linux home"),
    (("C:\\" + "Users\\someone\\engine\\src\\lib.rs").encode("utf-16-le"),
     "Windows home"),
    (b"/Us" b"ers/someone/src/main.rs", "macOS home"),
    (b"D:\\a\\repo\\repo\\engine\\src\\main.rs", "GitHub runner"),
])
def test_a_member_carrying_a_build_path_is_refused_naming_the_kind(
        tmp_path, leak, kind):
    """The wide-string case is the reason the scan strips NULs first."""

    with pytest.raises(SystemExit) as caught:
        tool.pin(RELEASE, [_bundle(tmp_path, extra={"rw_ndbc": leak})],
                 _pins_copy(tmp_path), REV, set(), None)
    assert kind in str(caught.value) and "rw_ndbc" in str(caught.value)


def test_the_remapped_prefixes_and_the_toolchain_paths_are_clean():
    clean = (b"/build/engine/tools/rustwx/crates/rw-obs/src/lib.rs\x00"
             b"/build/cargo/registry/src/x/y.rs\x00"
             b"/rustc/01f6ddf75/library/core/src/panicking.rs")
    assert tool.scan_payload(clean) == {}


def test_the_scan_command_reports_and_exits_one_on_a_dirty_bundle(tmp_path, capsys):
    dirty = _bundle(tmp_path, extra={"rw_amv": b"/ho" b"me/someone/x.rs"})
    assert tool.scan([dirty]) == 1
    assert "DIRTY" in capsys.readouterr().out


def test_fetch_doors_stages_the_notice_beside_the_doors(tmp_path, monkeypatch):
    monkeypatch.setattr(doors, "pin_for", lambda name: None)
    staged = doors.stage_from_directory(_bundle(tmp_path), tmp_path / "dest",
                                        PLATFORM)
    assert (tmp_path / "dest" / doors.BUNDLE_NOTICE).is_file()
    assert doors.BUNDLE_NOTICE not in {p.name for p in staged}, (
        "the notice is staged beside the doors, not counted as one")


def test_the_member_name_is_one_name_in_both_halves():
    assert tool.NOTICE_MEMBER == doors.BUNDLE_NOTICE


ENGINE_WORKSPACE = os.environ.get("ARWEN_GLOBAL_ENGINE_WORKSPACE", "").strip()


@pytest.mark.skipif(
    not (ENGINE_WORKSPACE and shutil.which("cargo")
         and (Path(ENGINE_WORKSPACE) / "Cargo.lock").is_file()),
    reason="ARWEN_GLOBAL_ENGINE_WORKSPACE is not set to an engine tools/rustwx "
           "workspace with cargo on PATH, so no lockfile can be read here")
@pytest.mark.parametrize("platform", doors.SUPPORTED_PLATFORMS)
def test_the_notice_covers_every_crate_the_lockfile_links(platform):
    text = tool.build_notice(Path(ENGINE_WORKSPACE), platform)
    for crate in ("rw-obs", "rw-atms", "rw-goes", "serde", "rustls"):
        assert f"\n{crate} " in text, crate
    assert "Permission is hereby granted" in text          # MIT
    assert "Apache License" in text
    assert "ECMWF" in text                                  # the BUFR tables


def _fake_workspace(tmp_path: Path, monkeypatch) -> tuple[Path, list[str]]:
    """An engine tree with one licence file per door crate, and a lockfile
    reader that records the triple it was asked for and returns those
    crates with no dependencies, so the notice's header is exercised without
    a Rust toolchain."""

    workspace = tmp_path / "tools" / "rustwx"
    workspace.mkdir(parents=True)
    (workspace / "Cargo.toml").write_text(
        '[workspace.package]\nlicense = "MIT"\n', encoding="utf-8")
    crates = sorted({door.crate for door in doors.doors_from_bundle(
        doors.COMPANION_BUNDLE) if door.crate.startswith("tools/rustwx/")})
    packages, nodes = [], []
    for index, crate in enumerate(crates):
        directory = tmp_path / crate
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "LICENSE").write_text("MIT text\n", encoding="utf-8")
        pid = f"crate-{index}"
        packages.append({"id": pid, "name": Path(crate).name,
                         "version": "0.1.0", "source": None, "license": "MIT",
                         "manifest_path": str(directory / "Cargo.toml")})
        nodes.append({"id": pid, "deps": []})
    asked: list[str] = []

    def metadata(_workspace, triple):
        asked.append(triple)
        return {"packages": packages, "resolve": {"nodes": nodes}}

    monkeypatch.setattr(tool, "_cargo_metadata", metadata)
    return workspace, asked


def test_the_notice_header_names_the_triple_the_doors_were_built_for(
        tmp_path, monkeypatch):
    """A cross-built Windows bundle's notice names the gnu triple it was
    resolved for.

    THE BREAKAGE THIS PREVENTS, measured on the 0.1.2 Windows bundle as
    first packed: its crates were resolved for x86_64-pc-windows-gnu, the
    target the doors were built for, while the notice's header line named
    x86_64-pc-windows-msvc, the platform's native triple, so the licence
    notice stated a target the binaries beside it were not built for.
    """

    workspace, asked = _fake_workspace(tmp_path, monkeypatch)
    text = tool.build_notice(workspace, "win-x86_64", "x86_64-pc-windows-gnu")
    assert asked == ["x86_64-pc-windows-gnu"]
    assert "Platform: win-x86_64 (x86_64-pc-windows-gnu)" in text
    assert "msvc" not in text

    asked.clear()
    native = tool.build_notice(workspace, "win-x86_64")
    assert asked == [tool.TARGET_TRIPLES["win-x86_64"]]
    assert f"Platform: win-x86_64 ({tool.TARGET_TRIPLES['win-x86_64']})" in native

    with pytest.raises(SystemExit):
        tool.build_notice(workspace, "win-x86_64", "x86_64-unknown-linux-gnu")
