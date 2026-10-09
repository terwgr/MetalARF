"""Which woof engine this port is running on, measured rather than pinned.

The port owns no physics.  Every column runs through the engine's
``run_mpas_column_batch`` seam, so a run has to be able to say which engine
bytes it executed.  Through 0.3.1 that was answered by a PIN: sixteen engine
files held to SHA-256 digests in the adapter, one exact published engine
admitted (``woof>=2.7.4,<2.7.5``), a table of measured verdicts for every
published engine and a second table for private builds.  That pin existed
for one breakage: pip resolving a separately published engine whose bytes
this port was never run with.

In WOOF the port and the engine ship in one distribution from one commit,
so that breakage cannot happen in an install, and the pin retired with it.
What stays is PROVENANCE: this module hashes the same sixteen seam files of
the engine a run actually imports and hands the digests, the engine version
and, for a git tree, its commit to the receipts.  Nothing here refuses a
run because a byte moved.

The seam CONTRACT is still held, at development time instead of at launch:
``tests/test_engine_seam_contract.py`` hashes the engine's
``mpas_column_batch.py`` and ``docs/mpas-seam.md`` and compares them with
:data:`woof.hex.cuda_arwen_physics_v841.MPAS_SEAM_CONTRACT_SURFACE_SHA256`.
A change to the seam's published surface fails that test in the same tree
that made it, which is where the change can be measured.

This module imports nothing heavy at module scope on purpose: ``doctor``
imports it, and a report about a broken estate must not need the estate to
work.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import subprocess

#: The engine distribution, spelled once.
DISTRIBUTION = "recast-woof"

#: The engine range the standalone ``woof hex`` distribution declares.  The
#: seam contract test and the CPU suite were run against the 2.8 engine; a
#: 2.9 engine has not been.  Inside the one WOOF distribution this line has
#: no role, because the engine and the port are the same install.
ENGINE_REQUIREMENT = "recast-woof"

#: The engine files the seam executes, relative to the engine's root (the
#: directory ``gpuwm/`` and ``docs/`` hang off).  Hashed into every receipt.
SEAM_PATHS: tuple[str, ...] = (
    "woof/core/mpas_column_batch.py",
    "woof/core/physics.py",
    "woof/core/microphysics.py",
    "woof/core/gf.py",
    "woof/core/kernels/gf.cu",
    "woof/core/rrtmg_legacy.py",
    "woof/core/noahmp_runtime.py",
    "woof/core/noahmp_glacier.py",
    "woof/core/noahmp_glacier_gpu.py",
    "woof/core/noahmp_kernel_sources.py",
    "woof/core/kernels/__init__.py",
    "woof/core/kernels/noahmp_leaves.cu",
    "woof/core/kernels/noahmp_glacier.cu",
    "woof/config.py",
    "woof/io/restart.py",
    "docs/mpas-seam.md",
)

#: The two files whose bytes ARE the seam's published contract: the column
#: batch itself and the document that states it.
CONTRACT_SURFACE_PATHS: tuple[str, ...] = (
    "woof/core/mpas_column_batch.py",
    "docs/mpas-seam.md",
)


@dataclass(frozen=True)
class SeamInspection:
    """What one engine tree's seam files are, byte for byte."""

    root: Path
    #: ``(path, sha256)`` for every seam path the tree carries, sorted.
    digests: tuple[tuple[str, str], ...]
    #: Seam paths the tree does not carry at all.
    absent: tuple[str, ...]
    #: The version the tree declares for itself, or ``None``.
    declared: str | None = None

    @property
    def checked(self) -> int:
        return len(self.digests)

    @property
    def manifest(self) -> dict[str, str]:
        return dict(self.digests)

    @property
    def accepted(self) -> bool:
        """Every seam path is present, so the executed bytes can be named."""

        return not self.absent


@dataclass(frozen=True)
class EngineIdentity:
    """The engine a run executed, as receipts and restart identities name it."""

    #: The version the tree declares, or ``None`` when it states none.
    version: str | None
    #: HEAD of a git working tree, or ``None`` for an install.
    build_commit: str | None
    #: The measured digest of every seam path.
    manifest: dict[str, str]
    #: ``sha256(translation_unit_source('noahmp_glacier'))`` as this engine
    #: composes it.
    glacier_composed_tu_sha256: str


def inspect_seam(root: Path) -> SeamInspection:
    """Hash every seam path under ``root``; report what is absent separately."""

    root = Path(root)
    found: dict[str, str] = {}
    absent: list[str] = []
    for relative in SEAM_PATHS:
        try:
            payload = (root / relative).read_bytes()
        except OSError:
            absent.append(relative)
            continue
        found[relative] = sha256(payload).hexdigest()
    return SeamInspection(
        root=root,
        digests=tuple(sorted(found.items())),
        absent=tuple(sorted(absent)),
        declared=declared_version(root),
    )


def contract_surface_sha256(root: Path) -> str:
    """The digest of the seam's published contract in the tree at ``root``."""

    surface = sha256()
    for relative in CONTRACT_SURFACE_PATHS:
        surface.update((Path(root) / relative).read_bytes())
    return surface.hexdigest()


def git_head(root: Path) -> str | None:
    """HEAD of ``root`` when ``root`` is the top of a git working tree."""

    root = Path(root).resolve()
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root),
             "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    lines = completed.stdout.split()
    if len(lines) != 2 or Path(lines[0]).resolve() != root:
        return None
    return lines[1]


def installed_version() -> str | None:
    """The woof distribution version pip has installed, or ``None``."""

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as distribution_version

    try:
        return distribution_version(DISTRIBUTION)
    except PackageNotFoundError:
        return None


def installed_root() -> Path | None:
    """The directory the seam's ``gpuwm/...`` paths hang off, installed.

    ``find_spec`` rather than an import: importing woof pulls its whole
    physics estate, and a report about that estate must not depend on it
    loading.
    """

    import importlib.util

    try:
        spec = importlib.util.find_spec("woof")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent.parent


def checkout_version(root: Path) -> str | None:
    """The version a woof SOURCE CHECKOUT declares, read without importing.

    ``pyproject.toml`` states it statically; an install root carries none
    and answers ``None`` here (``installed_version`` reads that case).
    """

    declaration = Path(root) / "pyproject.toml"
    try:
        text = declaration.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import tomllib

        parsed = tomllib.loads(text)
    except Exception:
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
        return match.group(1) if match else None
    project = parsed.get("project")
    if isinstance(project, dict):
        version = project.get("version")
        if isinstance(version, str):
            return version
    return None


def declared_version(root: Path) -> str | None:
    """What a tree says it is: its own ``pyproject.toml``, else pip's metadata
    when ``root`` is the installed package's root, else ``None``."""

    found = checkout_version(root)
    if found is not None:
        return found
    installed = installed_root()
    if installed is not None and installed == Path(root).resolve():
        return installed_version()
    return None


def remedy() -> str:
    """The command that puts an engine on this machine."""

    return f'  pip install "{ENGINE_REQUIREMENT}"'


__all__ = [
    "CONTRACT_SURFACE_PATHS",
    "DISTRIBUTION",
    "ENGINE_REQUIREMENT",
    "EngineIdentity",
    "SEAM_PATHS",
    "SeamInspection",
    "checkout_version",
    "contract_surface_sha256",
    "declared_version",
    "git_head",
    "inspect_seam",
    "installed_root",
    "installed_version",
    "remedy",
]
