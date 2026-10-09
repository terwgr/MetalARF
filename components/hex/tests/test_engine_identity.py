"""The engine a run executes is measured and named, not pinned.

``woof.hex.engine_identity`` replaced the exact engine pin (``engine_pin``,
its two verdict tables and ``tools/measure_engine_verdicts.py``) when the
port and the engine moved into one distribution.  These tests hold what is
left: the seam files are hashed and reported, a tree missing one is refused
by name at the door and in ``doctor``, and a moved byte is recorded rather
than refused.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import subprocess

import pytest

from woof.hex import doctor
from woof.hex import engine_identity
from woof.hex import forecast_door as door

#: Folded into recast-woof the engine and this package are one distribution
#: (``woof.hex``); standalone this is woof hex beside a separately installed
#: woof.
FOLDED = not engine_identity.__name__.startswith("hexcore.")


def _seam_tree(tmp_path: Path, version: str | None = "2.8.0") -> Path:
    root = tmp_path / "engine"
    for relative in engine_identity.SEAM_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"# {relative}\n".encode("ascii"))
    if version is not None:
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "woof"\nversion = "{version}"\n', encoding="utf-8"
        )
    return root


def test_every_seam_file_is_hashed_and_the_version_read_from_the_tree(tmp_path):
    root = _seam_tree(tmp_path)

    inspection = engine_identity.inspect_seam(root)

    assert inspection.accepted and inspection.absent == ()
    assert inspection.checked == len(engine_identity.SEAM_PATHS) == 16
    assert inspection.declared == "2.8.0"
    for relative, digest in inspection.digests:
        assert digest == sha256((root / relative).read_bytes()).hexdigest()


def test_a_moved_byte_is_measured_not_refused(tmp_path):
    root = _seam_tree(tmp_path)
    before = engine_identity.inspect_seam(root).manifest
    (root / "woof/core/physics.py").write_bytes(b"# moved\n")

    after = engine_identity.inspect_seam(root)

    assert after.accepted
    assert after.manifest["woof/core/physics.py"] != before["woof/core/physics.py"]
    assert door.seam_source_problem(root) is None


def test_a_missing_seam_file_is_named_at_the_door(tmp_path):
    root = _seam_tree(tmp_path)
    (root / "docs/mpas-seam.md").unlink()

    inspection = engine_identity.inspect_seam(root)
    assert not inspection.accepted
    assert inspection.absent == ("docs/mpas-seam.md",)

    problem = door.seam_source_problem(root)
    assert problem is not None
    assert "docs/mpas-seam.md" in problem
    assert "--gpuwm-checkout" in problem


def test_the_contract_surface_is_the_column_batch_then_the_document(tmp_path):
    root = _seam_tree(tmp_path)
    expected = sha256()
    expected.update((root / "woof/core/mpas_column_batch.py").read_bytes())
    expected.update((root / "docs/mpas-seam.md").read_bytes())

    assert engine_identity.contract_surface_sha256(root) == expected.hexdigest()


def test_a_git_tree_is_named_by_its_head_and_a_plain_tree_by_nothing(tmp_path):
    root = _seam_tree(tmp_path)
    assert engine_identity.git_head(root) is None

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "engine-identity@test")
    git("config", "user.name", "engine identity test")
    git("config", "commit.gpgsign", "false")
    git("add", "-A")
    git("commit", "-q", "-m", "seed")

    assert engine_identity.git_head(root) == git("rev-parse", "HEAD")
    # A subdirectory of a git tree is not the tree's root, so it names nothing.
    assert engine_identity.git_head(root / "woof") is None


def test_the_remedy_names_the_declared_engine_range():
    assert engine_identity.ENGINE_REQUIREMENT in engine_identity.remedy()
    if FOLDED:
        # One distribution carries both, so the remedy names it and no range.
        assert engine_identity.ENGINE_REQUIREMENT == engine_identity.DISTRIBUTION
    else:
        assert engine_identity.ENGINE_REQUIREMENT.startswith("woof>=2.8.0,")


def _seam_finding(findings):
    return next(f for f in findings if f.subject == doctor._SEAM_SUBJECT)


def test_doctor_verifies_an_install_that_carries_every_seam_file(tmp_path, monkeypatch):
    root = _seam_tree(tmp_path, version=None)
    monkeypatch.setattr(engine_identity, "installed_root", lambda: root)
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "2.8.0")

    findings = doctor.check_physics_seam()
    seam = _seam_finding(findings)

    assert seam.status == doctor.VERIFIED
    assert "2.8.0" in seam.detail
    assert doctor.blocking_gaps(findings) == []


def test_doctor_refuses_an_install_missing_a_seam_file(tmp_path, monkeypatch):
    root = _seam_tree(tmp_path, version=None)
    (root / "woof/io/restart.py").unlink()
    monkeypatch.setattr(engine_identity, "installed_root", lambda: root)
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "2.8.0")

    findings = doctor.check_physics_seam()
    seam = _seam_finding(findings)

    assert seam.status == doctor.MISSING and seam.required
    assert "woof/io/restart.py" in seam.detail
    assert engine_identity.ENGINE_REQUIREMENT in seam.remedy
    assert [f.subject for f in doctor.blocking_gaps(findings)] == [seam.subject]


def test_doctor_says_so_rather_than_crashing_when_gpuwm_cannot_be_located(monkeypatch):
    monkeypatch.setattr(engine_identity, "installed_root", lambda: None)
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "2.8.0")

    seam = _seam_finding(doctor.check_physics_seam())

    assert seam.status == doctor.INFO


def test_doctor_refuses_when_no_engine_is_installed(monkeypatch):
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: None)

    findings = doctor.check_physics_seam()

    assert len(findings) == 1
    assert findings[0].status == doctor.MISSING and findings[0].required
    assert "pip install" in findings[0].remedy


def test_the_retired_pin_leaves_no_module_behind():
    import importlib.util

    assert importlib.util.find_spec(f"{engine_identity.__package__}.engine_pin") is None
    tools = Path(__file__).resolve().parents[1] / "tools"
    assert not (tools / "measure_engine_verdicts.py").exists()


def test_the_adapter_names_the_engine_it_measured(tmp_path, monkeypatch):
    """The adapter's identity is the tree's own bytes, version and commit."""

    pytest.importorskip("numpy")
    try:
        from woof.hex import cuda_arwen_physics_v841 as adapter
    except ImportError as error:  # pragma: no cover - needs the CUDA estate
        pytest.skip(f"the adapter module does not import here: {error}")

    root = _seam_tree(tmp_path)
    monkeypatch.setattr(adapter, "_glacier_composed_tu_sha256", lambda: "g" * 64)

    identity = adapter._measure_engine(root)

    assert identity.version == "2.8.0"
    assert identity.build_commit is None
    assert identity.manifest == engine_identity.inspect_seam(root).manifest
    assert identity.glacier_composed_tu_sha256 == "g" * 64

    missing = Path("woof/core/gf.py")
    (root / missing).unlink()
    # The refusal prints the path the platform spells, so match that spelling.
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        adapter._measure_engine(root)
