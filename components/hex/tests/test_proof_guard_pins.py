"""The proof guard: the engine checkout guard and masked-content authority
digests.

Leg 1: ``verify_arwen_checkout_git`` names the engine tree a run executes.
It hashes the sixteen seam files (``woof.hex.engine_identity.SEAM_PATHS``)
and records them with the checkout's HEAD/tree/dirty-state, or an install's
version and RECORD digest, as receipt provenance.  The seam bytes are not
compared with a pinned engine any more: the engine ships with this port.
What still refuses: a missing seam file, a seam file that is dirty in a git
tree (its bytes could not be named by commit), a tree that is neither a git
tree nor the install, and a provider row's own manifest that moved.

Leg 2: the three regenerated native authorities are pinned by masked-content
digest (the random 10-char ``file_id`` global attribute MPAS stamps into every
output is masked, value bytes only, located via the netCDF header), so a
bit-exact rerun satisfies the pin while a single flipped data byte refuses.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from _layout import RUNNER_PATH


ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> object:
    name = "_test_proof_guard_pins_runner"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _git(repo: Path, *arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def _digests(repo: Path) -> dict[str, str]:
    from woof.hex import engine_identity

    return {
        relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        for relative in engine_identity.SEAM_PATHS
    }


@pytest.fixture()
def engine_checkout(tmp_path: Path) -> tuple[object, Path]:
    """A throwaway git tree carrying every seam path, with made-up bytes.

    The guard no longer compares the bytes with anything, so any bytes do;
    each file gets distinct content so a digest mix-up cannot pass.
    """

    from woof.hex import engine_identity

    runner = _load_runner()
    repo = tmp_path / "engine-checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "proof-guard@test")
    _git(repo, "config", "user.name", "proof-guard test")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "commit.gpgsign", "false")
    for relative in engine_identity.SEAM_PATHS:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"seam file {relative}\n".encode("ascii"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed: sixteen seam files")
    return runner, repo


def test_a_clean_checkout_is_named_by_commit_and_measured_digests(
    engine_checkout: tuple[object, Path],
) -> None:
    runner, repo = engine_checkout

    record = runner.verify_arwen_checkout_git(repo)

    assert record["kind"] == "git"
    assert record["head"] == _git(repo, "rev-parse", "HEAD")
    assert record["clean"] is True
    assert record["dirty_paths"] == []
    assert record["manifest"]["files"] == _digests(repo)


def test_a_committed_seam_edit_is_recorded_not_refused(
    engine_checkout: tuple[object, Path],
) -> None:
    """The retired pin: a seam byte that moved in a committed engine tree is
    the engine this run executes, and the receipt names its new digest."""

    runner, repo = engine_checkout
    before = runner.verify_arwen_checkout_git(repo)
    target = repo / "woof" / "core" / "physics.py"
    target.write_bytes(target.read_bytes() + b"# the engine moved\n")
    _git(repo, "add", "woof/core/physics.py")
    _git(repo, "commit", "-q", "-m", "engine: a seam file moved")

    after = runner.verify_arwen_checkout_git(repo)

    assert after["head"] != before["head"]
    moved = after["manifest"]["files"]["woof/core/physics.py"]
    assert moved == hashlib.sha256(target.read_bytes()).hexdigest()
    assert moved != before["manifest"]["files"]["woof/core/physics.py"]


def test_a_missing_seam_file_refuses_by_name(
    engine_checkout: tuple[object, Path],
) -> None:
    runner, repo = engine_checkout
    (repo / "docs" / "mpas-seam.md").unlink()

    with pytest.raises(RuntimeError) as error:
        runner.verify_arwen_checkout_git(repo)

    text = str(error.value)
    assert "docs/mpas-seam.md" in text
    assert "cannot name the engine bytes" in text


def test_a_dirty_seam_file_refuses_because_no_commit_names_it(
    engine_checkout: tuple[object, Path],
) -> None:
    runner, repo = engine_checkout
    target = repo / "woof" / "core" / "gf.py"
    target.write_bytes(target.read_bytes() + b"# uncommitted\n")

    with pytest.raises(RuntimeError) as error:
        runner.verify_arwen_checkout_git(repo)

    text = str(error.value)
    assert "woof/core/gf.py" in text
    assert "dirty" in text


def test_dirty_unrelated_file_is_recorded_loudly_and_execution_proceeds(
    engine_checkout: tuple[object, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner, repo = engine_checkout
    (repo / "scratch.log").write_text("an unrelated untracked file\n")

    record = runner.verify_arwen_checkout_git(repo)

    assert record["clean"] is False
    assert "scratch.log" in record["dirty_paths"]
    assert record["manifest"]["files"] == _digests(repo)
    loud = capsys.readouterr().out
    assert "scratch.log" in loud


def test_a_provider_rows_own_manifest_still_gates_its_bytes(
    engine_checkout: tuple[object, Path],
) -> None:
    """A provider row binds files the engine does not ship, so the manifest
    it hands is still compared, and a moved byte refuses by name."""

    runner, repo = engine_checkout
    extra = repo / "provider" / "batch.py"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"provider batch\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "provider file")
    wanted = {"provider/batch.py": "0" * 64}

    with pytest.raises(RuntimeError) as error:
        runner.verify_arwen_checkout_git(repo, manifest=wanted)

    text = str(error.value)
    assert "provider/batch.py" in text
    assert "does not match the proven manifest" in text

    good = {"provider/batch.py": hashlib.sha256(extra.read_bytes()).hexdigest()}
    record = runner.verify_arwen_checkout_git(repo, manifest=good)
    assert record["manifest"]["files"] == good


def test_a_path_that_is_not_a_git_tree_refuses_by_name(tmp_path: Path) -> None:
    """A directory that is neither a git working tree nor the installed
    engine -- an unpacked tarball, a copied site-packages -- names neither a
    commit nor a record, so it is refused, by name, with the remedy."""

    runner = _load_runner()
    plain = tmp_path / "not-a-repository"
    plain.mkdir()

    with pytest.raises(RuntimeError) as error:
        runner.verify_arwen_checkout_git(plain)

    text = str(error.value)
    assert "neither a woof git working tree nor the woof" in text
    assert "RECORD digest" in text, "what an install is named by instead"
    assert "--gpuwm-checkout" in text, "the remedy"


def test_the_installed_engine_is_admitted_and_named_by_its_record(
    tmp_path: Path,
) -> None:
    """The installed engine is recorded by version and RECORD digest, with
    the measured digests of its seam files."""

    from importlib.metadata import PackageNotFoundError, distribution

    from woof.hex import engine_identity

    runner = _load_runner()
    installed = engine_identity.installed_root()
    if installed is None:
        pytest.skip("no woof installed in this interpreter")
    try:
        dist = distribution(engine_identity.DISTRIBUTION)
    except PackageNotFoundError:  # pragma: no cover
        pytest.skip("woof is importable but carries no distribution metadata")
    probe = subprocess.run(
        ["git", "-C", str(installed), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        pytest.skip("woof is installed from a git tree here; the git route applies")
    if engine_identity.inspect_seam(Path(installed)).absent:
        pytest.skip("this woof install does not carry every seam file")

    record = runner.verify_arwen_checkout_git(Path(installed))
    assert record["kind"] == "installed"
    assert record["head"] is None and record["tree"] is None
    named = record["installed_distribution"]
    assert named["version"] == dist.version
    assert len(named["record_sha256"]) == 64
    assert set(record["manifest"]["files"]) == set(engine_identity.SEAM_PATHS)
    assert runner.verify_arwen_checkout_git(Path(installed)) == record, (
        "two reads of one install name it identically"
    )

    # Only the root this interpreter imports woof from is an install.
    elsewhere = tmp_path / "copied-site-packages"
    elsewhere.mkdir()
    assert runner.installed_engine_provenance(elsewhere) is None


def test_head_and_tree_are_provenance_not_gates() -> None:
    runner = _load_runner()
    import inspect

    source = inspect.getsource(runner.verify_arwen_checkout_git)
    assert "WOOF checkout HEAD changed" not in source
    assert "WOOF checkout tree changed" not in source
    assert "6b896c3dd5ef2fb94507210af49766f78f831d57" not in source


# --- leg 2: masked-content authority digests -------------------------------


REGENERATED_MASKED_DIGESTS = {
    "native_f000": "38575bfcbbe581c25ceffeec25d22061b6f22cea2308f639e8fcce093d58da17",
    "native_f030": "1cf267557cf394f0209fbce6a69350e386221d4e791c2ceefc72200d3a45da47",
    "native_f001": "2b867a3352d7580280c01120b5db7fb4e6979be528317770417dff04b9f58b4c",
}
RETIRED_WHOLE_FILE_DIGESTS = (
    "3c2917677726eac4b1514052e298f22418ad2b1ba63541016f42aaa632bcaf29",
    "e75ccb83b654a382a96b8ccb79b9232353f1119d19ee26efbe1b46158ca7ea16",
    "f6b1ec4aa0aac5c556147efac6806c094fd2e1945ee8c1a6038d2ea604604f01",
)

NETCDF3_FORMATS = ("NETCDF3_64BIT_DATA", "NETCDF3_64BIT_OFFSET", "NETCDF3_CLASSIC")


def _write_synthetic_history(path: Path, fmt: str, file_id: str) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path, "w", format=fmt) as dataset:
        dataset.setncattr("on_a_sphere", "YES")
        dataset.setncattr("sphere_radius", 6371229.0)
        dataset.setncattr("file_id", file_id)
        dataset.setncattr("model_name", "mpas")
        dataset.createDimension("nCells", 8)
        dataset.createDimension("nVertLevels", 3)
        variable = dataset.createVariable("theta", "f8", ("nCells", "nVertLevels"))
        variable[:] = np.arange(24, dtype=np.float64).reshape(8, 3) + 250.0


@pytest.mark.parametrize("fmt", NETCDF3_FORMATS)
def test_file_id_value_span_is_located_via_the_netcdf_header(
    tmp_path: Path, fmt: str
) -> None:
    runner = _load_runner()
    path = tmp_path / f"history-{fmt}.nc"
    _write_synthetic_history(path, fmt, "abcdefghij")

    offset, length = runner.netcdf_file_id_value_span(path)

    data = path.read_bytes()
    assert length == 10
    assert data[offset : offset + length] == b"abcdefghij"


def test_rewritten_file_id_alone_preserves_the_masked_digest(tmp_path: Path) -> None:
    runner = _load_runner()
    original = tmp_path / "history-a.nc"
    _write_synthetic_history(original, "NETCDF3_64BIT_DATA", "aaaaaaaaaa")
    offset, length = runner.netcdf_file_id_value_span(original)

    rerun = tmp_path / "history-b.nc"
    data = bytearray(original.read_bytes())
    data[offset : offset + length] = b"zzzzzzzzzz"
    rerun.write_bytes(bytes(data))

    first = runner.netcdf_masked_digests(original)
    second = runner.netcdf_masked_digests(rerun)
    assert first["masked_sha256"] == second["masked_sha256"]
    assert first["sha256"] != second["sha256"]
    assert first["file_id"] == "aaaaaaaaaa"
    assert second["file_id"] == "zzzzzzzzzz"


def test_single_flipped_data_byte_changes_the_masked_digest(tmp_path: Path) -> None:
    runner = _load_runner()
    original = tmp_path / "history-a.nc"
    _write_synthetic_history(original, "NETCDF3_64BIT_DATA", "aaaaaaaaaa")

    corrupted = tmp_path / "history-c.nc"
    data = bytearray(original.read_bytes())
    data[-1] ^= 0x01
    corrupted.write_bytes(bytes(data))

    assert (
        runner.netcdf_masked_digests(original)["masked_sha256"]
        != runner.netcdf_masked_digests(corrupted)["masked_sha256"]
    )


def test_file_record_verifies_masked_digest_and_records_both(tmp_path: Path) -> None:
    runner = _load_runner()
    original = tmp_path / "history-a.nc"
    _write_synthetic_history(original, "NETCDF3_64BIT_DATA", "aaaaaaaaaa")
    offset, length = runner.netcdf_file_id_value_span(original)
    digests = runner.netcdf_masked_digests(original)
    pin = {"bytes": original.stat().st_size, "masked_sha256": digests["masked_sha256"]}

    rerun = tmp_path / "history-b.nc"
    data = bytearray(original.read_bytes())
    data[offset : offset + length] = b"zzzzzzzzzz"
    rerun.write_bytes(bytes(data))

    record = runner._file_record("native_f000", rerun, pin)
    assert record["masked_sha256"] == digests["masked_sha256"]
    assert record["sha256"] == hashlib.sha256(bytes(data)).hexdigest()
    assert record["file_id"] == "zzzzzzzzzz"
    assert record["bytes"] == original.stat().st_size

    corrupted = tmp_path / "history-c.nc"
    flipped = bytearray(original.read_bytes())
    flipped[-1] ^= 0x01
    corrupted.write_bytes(bytes(flipped))
    with pytest.raises(RuntimeError) as error:
        runner._file_record("native_f000", corrupted, pin)
    text = str(error.value)
    assert "native_f000" in text
    assert "masked content digest" in text


def test_native_authority_pins_are_the_regenerated_masked_digests() -> None:
    runner = _load_runner()
    for role, masked in REGENERATED_MASKED_DIGESTS.items():
        pin = runner.AUTHORITY_PINS[role]
        assert pin["masked_sha256"] == masked
        assert "sha256" not in pin, (
            f"{role} still carries a whole-file sha256 pin; a bit-exact rerun "
            "could never satisfy it"
        )
        assert pin["bytes"] == 1_584_808_024
    for role in (
        "grid",
        "static",
        "init",
        "native_validation_receipt",
        "native_launch_receipt",
        "native_closure",
    ):
        pin = runner.AUTHORITY_PINS[role]
        assert "sha256" in pin and "masked_sha256" not in pin


def test_retired_whole_file_digests_are_gone_from_the_execution_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for digest in RETIRED_WHOLE_FILE_DIGESTS:
        assert digest not in source
