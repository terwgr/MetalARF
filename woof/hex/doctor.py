"""``woof hex doctor``: what this install can actually reach, and what to run.

A wheel for this project is deliberately partial, and saying so is the
point.  The Python driver, the front doors and the tables ship in it.
The things that do the work do not:

* the Rust engines (``rw_mpas_init``, ``rw_mpas_convert``,
  ``rw_wrfbatch``) are built from the woof ``tools/rustwx`` workspace
  and staged onto a machine -- ``woof fetch-bridges`` is the one
  command that does it;
* the CUDA lane needs the CuPy wheel for CUDA 13 -- the major every GPU
  door here refuses below -- and no pip extra can check what the box's
  driver actually serves, so this module reads it and says;
* the card itself: every architecture this install's NVRTC compiles for
  runs, anchored or not, and this module names the card, its compute
  capability and its anchor status, or the one refusal a card can meet;
* the meshes and their static fields are external assets with no fetch
  path in this distribution;
* the forecast runs on the installed woof and records the digests of its
  sixteen seam files, which every engine from 2.5.8 carries in
  site-packages.

Every one of those is a place a fresh install meets a wall.  This module
exists so the wall has a sign on it: each gap prints THE command that
closes it, on this platform, spelled as it is typed.  A bare
``ImportError`` three commands later is the failure mode being replaced.

Statuses.  ``verified`` means the deep check ran and passed -- the module
imported in a short-lived subprocess, the binary was found on a named
rung.  ``present`` is for what can only be checked by existence.
``missing`` is a gap, and a gap always carries a remedy.  ``info`` is
context that is never a gap.

Exit status is 1 when any REQUIRED finding is missing, 0 otherwise.
Required means: a door this distribution advertises cannot open, or the
install itself is wrong.  The three Python dependencies and the three
engines are required in the first sense; an installed woof that lacks a
seam file is required in the second.  CuPy, the card's architecture and
the mesh assets are reported and do not fail the process, because a user
who only wants the
render door should not be told the install is broken by the absence of a
mesh.

Through 0.3.1 this report also compared the engine's seam bytes with a
pinned engine, because a separately published engine could differ from the
one this port was run with.  The engine now ships with this port and that
comparison retired with the pin.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from . import DISTRIBUTION_NAME
from . import engines


#: How long a single import probe may take before it is called wedged.
_PROBE_TIMEOUT_S = 90

VERIFIED = "verified"
PRESENT = "present"
MISSING = "missing"
INFO = "info"


@dataclass
class Finding:
    """One checked thing: what it is, what was found, what to run."""

    subject: str
    status: str
    detail: str
    #: Commands that close the gap.  Every line is a command as typed or a
    #: ``#`` comment -- never prose fused onto a command.
    remedy: str = ""
    #: Whether a gap here closes every front door.
    required: bool = False
    evidence: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "status": self.status,
            "detail": self.detail,
            "remedy": self.remedy,
            "required": self.required,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def _import_probe(module: str) -> tuple[bool, str]:
    """Import ``module`` in a short-lived subprocess and read its version.

    A subprocess rather than an import here for two reasons that have
    both been measured on this stack: a broken native dependency (a
    netCDF4 or CuPy whose shared libraries do not load) can take the
    interpreter down rather than raise, and a partially initialised
    module poisons every later check in the same process.  The report
    must survive the thing it is reporting on.
    """

    code = (
        "import importlib, json, sys\n"
        f"m = importlib.import_module({module!r})\n"
        "print(json.dumps({'version': getattr(m, '__version__', None),"
        " 'file': getattr(m, '__file__', None)}))\n"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"the import probe could not run: {error}"
    if probe.returncode != 0:
        tail = (probe.stderr or "").strip().splitlines()
        return False, tail[-1] if tail else f"import exited {probe.returncode}"
    try:
        facts = json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return True, "imported (version not reported)"
    version = facts.get("version") or "version not reported"
    return True, f"imported, {version}"


def _distribution_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as distribution_version

    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def check_interpreter() -> list[Finding]:
    from . import __version__

    return [
        Finding(
            subject="distribution",
            status=INFO,
            detail=f"{DISTRIBUTION_NAME} {__version__}",
            evidence={
                "import_package": __package__ or "hexcore",
                "location": str(Path(__file__).resolve().parent),
            },
        ),
        Finding(
            subject="interpreter",
            status=INFO,
            detail=(
                f"Python {platform.python_version()} on "
                f"{platform.system()} {platform.machine()}"
            ),
            evidence={"executable": sys.executable},
        ),
    ]


#: Module -> the distribution that installs it, where they differ.
_REQUIRED_MODULES = (
    ("numpy", "numpy", "arrays; 48 modules import it at line one"),
    ("netCDF4", "netCDF4", "reads and writes every mesh, static and history file"),
    ("scipy", "scipy", "the regridder's spatial index"),
)


def check_python_dependencies() -> list[Finding]:
    findings: list[Finding] = []
    for module, distribution, why in _REQUIRED_MODULES:
        ok, detail = _import_probe(module)
        findings.append(
            Finding(
                subject=f"{module} ({why})",
                status=VERIFIED if ok else MISSING,
                detail=detail,
                remedy="" if ok else f"  pip install {distribution}",
                required=True,
            )
        )
    return findings


#: CUDA major -> the pip extra that installs its CuPy wheel.  The same table
#: ``woof.doctor`` carries, and deliberately the same: a user who runs both
#: doctors on one box must not be told two different things.
_GPU_EXTRA_BY_MAJOR = {12: "gpu-cu12", 13: "gpu-cu13"}


def cuda_runtime_floor() -> int | None:
    """The CUDA runtime version every GPU door here requires, READ not restated.

    Taken off :func:`woof.hex.cuda_backend.runtime.require_cuda`'s own
    default, because that function is the thing that refuses and a second
    copy of the number in this file is a second thing to keep true.  The
    number this reads is the one the refusal quotes.

    ``None`` when the port's own CUDA module cannot be imported at all,
    which is a broken install rather than a missing card; the caller says so
    rather than guessing a floor.
    """

    try:
        from inspect import signature

        from .cuda_backend.runtime import require_cuda

        default = signature(require_cuda).parameters["min_runtime_version"].default
        return int(default)
    except Exception:  # pragma: no cover - a broken install
        return None


def _driver_library_names() -> tuple[str, ...]:
    """The CUDA driver library this platform would load, by name."""

    if sys.platform == "win32":
        return ("nvcuda.dll",)
    if sys.platform == "darwin":
        return ()
    return ("libcuda.so.1", "libcuda.so")


def _no_local_gpu() -> bool:
    for variable in ("WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(variable, "") not in ("", "0"):
            return True
    return False


def _driver_cuda_major() -> int | None:
    """The CUDA major this box's DRIVER serves, or ``None`` if unknown.

    Read with ``ctypes`` straight off the driver library rather than through
    CuPy, because the case that needs the answer most is the box that has NO
    CuPy yet -- which is exactly where every CuPy-based probe is by
    definition unavailable.  ``cuDriverGetVersion`` is the one entry point
    that answers without ``cuInit``: no context, no device opened, nothing
    that could disturb a card another process is holding.  It is still
    driver contact, so the no-local-GPU declaration suppresses it.

    Ported from ``woof.doctor._driver_cuda_major``, which had already
    solved this on the same boxes.  Every failure is a ``None``: a machine
    with no NVIDIA driver is the ordinary case here, not an error.
    """

    if _no_local_gpu():
        return None
    for name in _driver_library_names():
        try:
            library = ctypes.CDLL(name)
            version = ctypes.c_int(0)
            if library.cuDriverGetVersion(ctypes.byref(version)) != 0:
                continue
        except (OSError, AttributeError, ValueError):
            continue
        if version.value > 0:
            return version.value // 1000
    return None


def _cupy_runtime_probe() -> dict[str, object] | None:
    """CuPy's CUDA RUNTIME version and wheel name, in a short-lived process.

    Separate from :func:`_import_probe` because importing cleanly is the
    thing that misled everybody: a CuPy built for the wrong CUDA major
    imports, reports a version, allocates, and runs cuBLAS.  What tells the
    two apart is ``runtimeGetVersion``, and nothing was reading it.

    A subprocess for the same reason every probe here uses one -- a CuPy
    whose shared libraries do not load takes the interpreter down rather
    than raising.
    """

    code = (
        "import json, sys\n"
        "out = {}\n"
        "try:\n"
        "    import cupy\n"
        "    out['cupy_version'] = getattr(cupy, '__version__', None)\n"
        "    out['runtime_version'] = int(cupy.cuda.runtime.runtimeGetVersion())\n"
        "except Exception as error:\n"
        "    out['error'] = f'{type(error).__name__}: {error}'\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        document = json.loads((probe.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


def _installed_cupy_wheels() -> list[str]:
    """Every installed CuPy distribution, by the name pip would uninstall."""

    from importlib import metadata

    names: list[str] = []
    try:
        distributions = list(metadata.distributions())
    except Exception:  # pragma: no cover - depends on the environment
        return names
    for distribution in distributions:
        try:
            name = distribution.metadata["Name"]
        except Exception:  # pragma: no cover - a malformed dist-info
            continue
        if name and str(name).lower().startswith("cupy"):
            names.append(str(name))
    return sorted(set(names))


def check_gpu_runtime() -> list[Finding]:
    """CuPy, and the CUDA major this distribution's own runtime floor requires.

    THE DEFECT THIS REPLACES, and it is worth stating because it cost a
    measured cross-machine walk (``evidence/xmachine-20260827`` section 5,
    ``evidence/userwalk-20260827`` section 4.2).  The old check asked one
    question -- does ``import cupy`` work -- and printed a remedy naming both
    extras with a comment telling the user to match their driver's CUDA
    major.  ``render``'s compact view, which is what ``doctor`` prints by
    default, filters comment lines out and keeps the FIRST command, so the
    one line a user actually saw was ``pip install "recast-woof[gpu-cu12]"``.
    On a CUDA-13 box that installs a CuPy every GPU door in this port
    refuses: ``cuda.runtime_version=12090 < required 13000``.  Doctor then
    reported ``verified`` on that same CuPy, because it imported.

    So this check reads three things instead of one: the runtime floor
    ``require_cuda`` will enforce, the CUDA major the DRIVER serves, and --
    when a CuPy is installed -- the CUDA runtime that CuPy actually carries.
    A CuPy the floor refuses is reported as a GAP with the exact refusal the
    forecast door would raise, before a card is ever opened, which is the
    difference between failing early by name and failing at the first real
    device call.
    """

    floor = cuda_runtime_floor()
    floor_major = None if floor is None else floor // 1000
    box_major = _driver_cuda_major()
    matching = _GPU_EXTRA_BY_MAJOR.get(floor_major) if floor_major else None
    evidence: dict[str, object] = {
        "required_runtime_version": "unreadable" if floor is None else floor,
        "driver_cuda_major": "unknown" if box_major is None else box_major,
    }

    ok, detail = _import_probe("cupy")
    if not ok:
        if floor is None:
            return [
                Finding(
                    subject="cupy (the CUDA lane)",
                    status=MISSING,
                    detail=detail,
                    remedy=(
                        "  # this install could not import its own CUDA runtime\n"
                        "  # module, so the required CUDA major is unreadable\n"
                        f'  pip install --force-reinstall "{DISTRIBUTION_NAME}"'
                    ),
                    evidence=evidence,
                )
            ]
        if box_major is not None and box_major < floor_major:
            return [
                Finding(
                    subject="cupy (the CUDA lane)",
                    status=MISSING,
                    detail=(
                        f"this box's driver serves CUDA {box_major} and every "
                        f"GPU door here requires runtime {floor}: no CuPy "
                        f"wheel closes that. "
                    )
                    + detail,
                    remedy=(
                        f"  # NOT a pip problem.  cupy-cuda{floor_major}x needs a "
                        f"driver serving\n"
                        f"  # CUDA {floor_major}; this one serves CUDA {box_major}.  "
                        f"Update the NVIDIA driver,\n"
                        f"  # or run the CUDA lane on a machine that has one.  "
                        f"Installing\n"
                        f"  # cupy-cuda{box_major}x instead would import, probe "
                        f"clean, and then be\n"
                        f"  # refused by name: cuda.runtime_version < required "
                        f"{floor}."
                    ),
                    evidence=evidence,
                )
            ]
        install = f'  pip install "{DISTRIBUTION_NAME}[{matching}]"'
        why = (
            f"  # this port's CUDA doors require runtime {floor}, so the CUDA-"
            f"{floor_major}\n  # wheel is the only one they admit"
        )
        if box_major is not None:
            why += f"; this box's driver serves CUDA {box_major}\n"
        else:
            why += "\n"
        return [
            Finding(
                subject="cupy (the CUDA lane)",
                status=MISSING,
                detail=detail,
                remedy=why + install,
                evidence=evidence,
            )
        ]

    probe = _cupy_runtime_probe() or {}
    runtime = probe.get("runtime_version")
    evidence["cupy_runtime_version"] = (
        "unreadable" if not isinstance(runtime, int) else runtime
    )
    if floor is None or not isinstance(runtime, int):
        return [
            Finding(
                subject="cupy (the CUDA lane)",
                status=PRESENT,
                detail=detail,
                evidence=evidence,
            )
        ]
    if runtime < floor:
        wheels = _installed_cupy_wheels()
        evidence["installed_cupy_distributions"] = ", ".join(wheels) or "unknown"
        removal = (
            f"  pip uninstall -y {' '.join(wheels)}"
            if wheels
            else f"  pip uninstall -y cupy-cuda{runtime // 1000}x"
        )
        return [
            Finding(
                subject="cupy (the CUDA lane)",
                status=MISSING,
                detail=(
                    f"cupy imports but carries CUDA runtime {runtime}; every "
                    f"GPU door here refuses below {floor}"
                ),
                remedy=(
                    f"  # the forecast door will raise, by name:\n"
                    f"  #   CudaRefusal: cuda.runtime_version={runtime} < "
                    f"required {floor}\n"
                    f"  # remove the wrong-major wheel FIRST -- pip will leave "
                    f"both\n"
                    f"  # installed and import order, not intent, picks the "
                    f"winner\n"
                    f"{removal}\n"
                    f'  pip install "{DISTRIBUTION_NAME}[{matching}]"'
                ),
                evidence=evidence,
            )
        ]
    return [
        Finding(
            subject="cupy (the CUDA lane)",
            status=VERIFIED,
            detail=f"{detail}; CUDA runtime {runtime}, at or above {floor}",
            evidence=evidence,
        )
    ]


_ARCHITECTURE_SUBJECT = "GPU architecture (the card and NVRTC)"


def _card_probe() -> dict[str, object] | None:
    """The card's name and compute capability, and NVRTC's target list.

    Read in a short-lived process, for the reason every probe here uses
    one.  It reads device 0's properties and asks NVRTC which targets it
    generates code for, through the same function the forecast door asks
    (``arch_admission.nvrtc_target_archs``).  It compiles nothing and
    launches nothing.  ``None`` when the probe itself could not run.
    """

    code = (
        "import json, sys\n"
        "out = {}\n"
        "try:\n"
        "    import cupy\n"
        "    out['device_count'] = int(cupy.cuda.runtime.getDeviceCount())\n"
        "    if out['device_count'] > 0:\n"
        "        p = cupy.cuda.runtime.getDeviceProperties(0)\n"
        "        name = p['name']\n"
        "        if isinstance(name, bytes):\n"
        "            name = name.decode('utf-8', 'replace')\n"
        "        out['name'] = str(name)\n"
        "        out['compute'] = [int(p['major']), int(p['minor'])]\n"
        "except Exception as error:\n"
        "    out['error'] = f'{type(error).__name__}: {error}'\n"
        "if 'compute' in out:\n"
        "    try:\n"
        f"        from {__package__}.cuda_backend.arch_admission import nvrtc_target_archs\n"
        "        out['nvrtc_targets'] = list(nvrtc_target_archs())\n"
        "    except Exception as error:\n"
        "        out['nvrtc_error'] = f'{type(error).__name__}: {error}'\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        document = json.loads((probe.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


def check_gpu_architecture() -> list[Finding]:
    """The card: its name, compute capability and anchor status.

    Every architecture NVRTC can compile the kernels for runs this port by
    default (:mod:`woof.hex.cuda_backend.arch_admission`).  So an unanchored
    card is never a gap here: it is reported with its status, as the
    forecast door prints it and every receipt records it.  The one gap is
    the forecast door's own NVRTC refusal, a card whose target this
    install's NVRTC does not list (or an NVRTC that cannot be asked at all),
    because on it no kernel could be built.  A card that cannot be read at
    all is context, not a gap: the CuPy finding above says why.
    """

    from .cuda_backend.arch_admission import architecture_status, nvrtc_refusal

    if _no_local_gpu():
        return [
            Finding(
                subject=_ARCHITECTURE_SUBJECT,
                status=INFO,
                detail=(
                    "not read: this box declares that no GPU work happens on "
                    "it (WOOF_HEX_NO_LOCAL_GPU / GPUWM_NO_LOCAL_GPU)"
                ),
            )
        ]
    probe = _card_probe()
    if probe is None or "compute" not in probe:
        if probe is None:
            why = "the card probe could not run"
        elif probe.get("error"):
            why = str(probe["error"])
        else:
            why = "no CUDA device is visible to this process"
        return [
            Finding(
                subject=_ARCHITECTURE_SUBJECT,
                status=INFO,
                detail=f"no card read: {why}",
                evidence={} if probe is None else dict(probe),
            )
        ]
    compute = (int(probe["compute"][0]), int(probe["compute"][1]))
    name = str(probe.get("name") or "unnamed device")
    status = architecture_status(compute)
    evidence: dict[str, object] = dict(status.as_dict())
    evidence["name"] = name
    card = (
        f"{name}, compute capability {compute[0]}.{compute[1]} ({status.sm})"
    )
    floor = cuda_runtime_floor()
    floor_major = None if floor is None else floor // 1000
    if "nvrtc_targets" not in probe:
        error = probe.get("nvrtc_error", "no answer")
        evidence["nvrtc_error"] = error
        extra = _GPU_EXTRA_BY_MAJOR.get(floor_major) if floor_major else None
        remedy = (
            f"  # NVRTC could not be asked which targets it compiles for, and\n"
            f"  # every kernel is compiled by NVRTC when the run starts; the\n"
            f"  # CUDA extra installs CuPy with the NVRTC it loads\n"
        )
        if extra:
            remedy += f'  pip install "{DISTRIBUTION_NAME}[{extra}]"'
        return [
            Finding(
                subject=_ARCHITECTURE_SUBJECT,
                status=MISSING,
                detail=(
                    f"{card}: NVRTC could not be asked which targets it "
                    f"compiles for ({error}), so no kernel could be built on "
                    f"this install"
                ),
                remedy=remedy.rstrip("\n"),
                evidence=evidence,
            )
        ]
    targets = tuple(int(value) for value in probe["nvrtc_targets"])
    evidence["nvrtc_targets"] = [f"compute_{value}" for value in targets]
    target = compute[0] * 10 + compute[1]
    if target not in targets:
        if targets and target > max(targets):
            newer = f"CUDA {floor_major} " if floor_major else ""
            remedy = (
                f"  # this NVRTC generates code up to compute_{max(targets)}, "
                f"older than the card;\n"
                f"  # a newer {newer}NVRTC lists compute_{target}\n"
                f"  pip install --upgrade nvidia-cuda-nvrtc"
            )
        else:
            remedy = (
                f"  # NOT a pip problem.  The CUDA toolkit this port requires "
                f"generates no\n"
                f"  # code for compute_{target}; run the CUDA lane on a card "
                f"it lists"
            )
        return [
            Finding(
                subject=_ARCHITECTURE_SUBJECT,
                status=MISSING,
                detail=f"{name}: {nvrtc_refusal(compute, targets)}",
                remedy=remedy,
                evidence=evidence,
            )
        ]
    detail = f"{card}: admitted, {status.describe()}"
    if not status.anchored:
        detail += (
            "; the forecast door measures the numeric route on this card "
            "before a run"
        )
    return [
        Finding(
            subject=_ARCHITECTURE_SUBJECT,
            status=VERIFIED,
            detail=detail,
            evidence=evidence,
        )
    ]


#: The engine range the standalone distribution declares, spelled once in
#: ``engine_identity``.
def _requirement() -> str:
    from . import engine_identity

    return engine_identity.ENGINE_REQUIREMENT


_SEAM_SUBJECT = "woof seam files (the sixteen the forecast records)"


def _check_seam_bytes(installed: str) -> Finding:
    """Does the installed engine carry every seam file the forecast names?

    The engine ships with this port, so its seam bytes are recorded, not
    compared with a pin.  What still stops a forecast is a missing seam
    file: the run hashes each one into its receipts and refuses a tree that
    does not carry them all.
    """

    from . import engine_identity

    root = engine_identity.installed_root()
    if root is None:
        return Finding(
            subject=_SEAM_SUBJECT,
            status=INFO,
            detail=(
                "woof is installed but its files could not be located on "
                "disk, so the seam files were not read.  The forecast reads "
                "them again at launch and refuses by name."
            ),
        )
    inspection = engine_identity.inspect_seam(root)
    total = len(engine_identity.SEAM_PATHS)
    if inspection.accepted:
        return Finding(
            subject=_SEAM_SUBJECT,
            status=VERIFIED,
            detail=(
                f"woof {installed}: all {total} seam files are in this "
                "install; the forecast records their digests"
            ),
            evidence={"root": str(root), "seam_files": inspection.checked},
        )
    return Finding(
        subject=_SEAM_SUBJECT,
        status=MISSING,
        detail=(
            f"woof {installed} is installed and {len(inspection.absent)} of "
            f"its {total} seam files are not in the install: "
            + ", ".join(inspection.absent)
            + ".  The forecast refuses at launch; the two front doors that "
            "drive Rust binaries are unaffected."
        ),
        remedy=engine_identity.remedy(),
        required=True,
        evidence={
            "installed": installed,
            "declared": _requirement(),
            "absent": list(inspection.absent),
            "root": str(root),
        },
    )


def check_physics_seam() -> list[Finding]:
    """woof: the installed distribution and its seam files.

    Two findings, because they are two different facts with two different
    remedies: whether an engine is installed at all, and whether it carries
    every seam file the forecast names in its receipts.  Hashing them costs
    one read of sixteen small files and needs no card, no network and no
    checkout.
    """

    from . import engine_identity

    findings: list[Finding] = []
    installed = _distribution_version("recast-woof")
    if installed is None:
        findings.append(
            Finding(
                subject="woof (the physics seam)",
                status=MISSING,
                detail="no woof distribution is installed in this interpreter",
                remedy=engine_identity.remedy(),
                required=True,
            )
        )
    else:
        findings.append(
            Finding(
                subject="woof (the physics seam)",
                status=PRESENT,
                detail=f"woof {installed} is installed",
                evidence={"version": installed, "declared": _requirement()},
            )
        )
        findings.append(_check_seam_bytes(installed))
    return findings


def check_engines() -> list[Finding]:
    """The three Rust binaries, and the one command that stages all of them."""

    findings: list[Finding] = []
    # Asked once, of the INSTALLED woof, and reported plainly: whether the
    # one-command staging route can supply these engines at all on this box.
    # woof's published 2.5.2 bundles rw_wrfbatch and none of the MPAS
    # binaries, so "run woof fetch-bridges" is a complete answer for one of
    # the three engines here and no answer at all for the other two.  A
    # report that does not distinguish those sends a user round a loop.
    supplied = {spec.name: engines.gpuwm_bundles(spec) for spec in engines.ENGINES}
    unsupplied = sorted(name for name, ok in supplied.items() if ok is False)
    if unsupplied:
        findings.append(
            Finding(
                subject="woof fetch-bridges coverage",
                status=INFO,
                detail=(
                    "the woof installed here bundles no "
                    f"{', '.join(unsupplied)}, so `{engines.FETCH_COMMAND}` "
                    "cannot supply "
                    f"{'them' if len(unsupplied) > 1 else 'it'}.  A later "
                    "woof release adds these artifacts; until then they come "
                    "from a source build."
                ),
                evidence={"not_bundled": unsupplied},
            )
        )
    for spec in engines.ENGINES:
        path, source = engines.locate(spec)
        if path is not None:
            findings.append(
                Finding(
                    subject=f"{spec.name} ({spec.subject})",
                    status=VERIFIED,
                    detail=f"found via {source}: {path}",
                    required=True,
                    evidence={"path": str(path), "resolved_from": source},
                )
            )
            continue
        findings.append(
            Finding(
                subject=f"{spec.name} ({spec.subject})",
                status=MISSING,
                detail=(
                    f"{source}.  Without it {spec.what_breaks}.  "
                    f"Looked at: {engines.resolution_order(spec)}"
                ),
                remedy=engines.chmod_remedy(source) or engines.remedy(spec),
                required=True,
                evidence={"resolution_order": engines.resolution_order(spec)},
            )
        )
    return findings


def check_assets() -> list[Finding]:
    """The mesh pair, which this distribution ships and fetches neither."""

    return [
        Finding(
            subject="mesh grid + static pair",
            status=INFO,
            detail=(
                "external assets.  This distribution carries no mesh and has "
                "no fetch path for one, so every door that needs a mesh takes "
                "--grid and --static explicitly and refuses rather than guess "
                "a default.  Two routes exist: generate a mesh of any "
                "resolution with the staged rw_mpas_mesh and rw_mpas_static "
                "binaries, or supply an existing grid/static pair."
            ),
            evidence={"bridge_directory": str(engines.USER_BRIDGE_DIR)},
        )
    ]


def collect() -> list[Finding]:
    """Every finding, in the order a reader should meet them."""

    findings: list[Finding] = []
    findings.extend(check_interpreter())
    findings.extend(check_python_dependencies())
    findings.extend(check_physics_seam())
    findings.extend(check_gpu_runtime())
    findings.extend(check_gpu_architecture())
    findings.extend(check_engines())
    findings.extend(check_assets())
    return findings


def blocking_gaps(findings: list[Finding]) -> list[Finding]:
    """The findings that make the exit status 1.  The rule, stated once."""

    return [f for f in findings if f.status == MISSING and f.required]


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
_LABEL = {
    VERIFIED: "OK      ",
    PRESENT: "PRESENT ",
    MISSING: "MISSING ",
    INFO: "INFO    ",
}


def render(findings: list[Finding], *, explain: bool) -> str:
    """One line per finding, or the whole evidence and remedy block."""

    lines: list[str] = []
    for finding in findings:
        label = _LABEL.get(finding.status, finding.status)
        if explain:
            lines.append(f"{label}{finding.subject}")
            lines.append(f"          {finding.detail}")
            if finding.evidence:
                for key, value in sorted(finding.evidence.items()):
                    lines.append(f"          {key}: {value}")
            if finding.remedy:
                lines.append("")
                lines.extend(finding.remedy.splitlines())
            lines.append("")
            continue
        headline = finding.detail.splitlines()[0] if finding.detail else ""
        if len(headline) > 96:
            headline = headline[:93] + "..."
        lines.append(f"{label}{finding.subject}: {headline}")
        if finding.remedy:
            first = [
                line
                for line in finding.remedy.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if first:
                lines.append(f"          {first[0].strip()}")
    return "\n".join(lines)


def summarise(findings: list[Finding]) -> str:
    blocking = blocking_gaps(findings)
    if not blocking:
        gaps = [f for f in findings if f.status == MISSING]
        if gaps:
            return (
                f"\nEvery required check passed.  {len(gaps)} optional gap(s) "
                f"remain; each prints its own command above."
            )
        return "\nEvery check passed."
    names = ", ".join(f.subject.split(" (")[0] for f in blocking)
    return (
        f"\n{len(blocking)} required item(s) missing: {names}.\n"
        f"Run `woof hex doctor --explain` for the full remedy for each."
    )


def add_doctor_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "doctor",
        help="report what this install can reach, and the command for each gap",
        description=(
            "Check the runtime estate this distribution needs and cannot "
            "carry: the Python dependencies, the physics seam, the CUDA "
            "lane, and the Rust engines the front doors drive.  Every gap "
            "prints the command that closes it."
        ),
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print the evidence and the whole pasteable remedy for each finding",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the findings as JSON instead of the text report",
    )
    parser.set_defaults(handler=run_doctor)


def run_doctor(arguments: argparse.Namespace) -> int:
    findings = collect()
    if getattr(arguments, "json", False):
        print(
            json.dumps(
                {
                    "distribution": DISTRIBUTION_NAME,
                    "findings": [finding.as_dict() for finding in findings],
                    "blocking": [
                        finding.subject for finding in blocking_gaps(findings)
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render(findings, explain=getattr(arguments, "explain", False)))
        print(summarise(findings))
    return 1 if blocking_gaps(findings) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="woof hex doctor")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--json", action="store_true")
    return run_doctor(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Finding",
    "INFO",
    "MISSING",
    "PRESENT",
    "VERIFIED",
    "add_doctor_parser",
    "blocking_gaps",
    "collect",
    "main",
    "render",
    "run_doctor",
    "summarise",
]
