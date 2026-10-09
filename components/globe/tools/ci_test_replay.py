"""Replay this repository's CI test job locally, against the built wheel.

The job replayed is ``.github/workflows/test.yml``'s ``test`` job: build
the distribution, install it into a fresh environment, and run

    python -m pytest -q -m "<marker filter>" tests

THE MARKER FILTER AND THE TEST PATHS ARE PARSED OUT OF THE WORKFLOW,
never restated here.  A hand-maintained copy of a CI selection is exactly
the drift this script exists to rule out: the hex line has a memory of a
gate that restated a number in four places and watched the copies walk
apart.

WHY THIS SCRIPT EXISTS AT ALL, and it is not convenience.  The release
memory of this family requires a replay that is green BEFORE a tag is
created, and it requires one leg of that replay to run on Linux.  Both
requirements were bought:

* A tag is spent the moment it exists (forward commits only), so a test
  job that fails after the tag has cost a version number.  The engine's
  ledger records eight tags stranded that way.
* A publish job once died inside pytest's own reporter on Linux, on a
  release every local gate had passed, because every local gate ran on
  Windows.  A replay that can only run on the machine the code was
  written on is blind in exactly the direction that has already failed.

THE ENGINE IS NOT ON PyPI YET, and that is why ``--engine-dist`` exists.
The CI job installs the wheel and lets pip resolve ``woof`` from PyPI,
which is the motion a user performs and the motion that must eventually
be green.  Until the engine's floor version publishes, that resolution
cannot succeed for anybody, so this replay accepts a directory of locally
built engine distributions and installs those first.  It says which route
it took, in the transcript, every time: a proof against bytes nobody can
download is a different proof from a proof against PyPI, and the two must
never be confused for one another in a report.

Usage:
    python tools/ci_test_replay.py [--engine-dist DIR] [--workdir DIR]
        [--keep] [--python EXE]

Exit status is pytest's, so 0 means the job's test leg would have passed.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"

#: Seeded into the replay venv before anything else, because the CI
#: interpreter already carries them and a fresh venv on 3.12+ does not.
#: ``ensurepip`` stopped seeding setuptools at 3.12 (PEP 632), and this
#: repository's packaging gates drive setuptools directly to measure what
#: the wheel and the sdist would contain.  Without this seed a replay run
#: from a 3.12+ interpreter goes red on its own environment while the job
#: it replays is fine, and the verdict then describes the harness rather
#: than the tree.
VENV_SEEDS = ("build", "setuptools>=77", "wheel", "pytest>=8.0")


def venv_python(venv_dir: Path) -> Path:
    """The interpreter inside a venv, on either platform."""

    windows = os.name == "nt"
    return venv_dir / ("Scripts" if windows else "bin") / (
        "python.exe" if windows else "python")


def parse_test_job(workflow_text: str) -> tuple[str, list[str]]:
    """The ``-m`` marker expression and test paths of the ``test`` job.

    Parsed from the one pytest invocation in the workflow rather than
    copied, so this script cannot drift behind the job it replays.
    """

    match = re.search(
        # The path list is bounded to the SAME LINE.  `\s` matches a
        # newline, and with it this pattern ran on into the next YAML
        # key and reported `uses` as one of the job's test paths.
        r"python -m pytest[^\n]*? -m \"([^\"]+)\"((?:[^\S\n]+[\w./-]+)+)",
        workflow_text)
    if not match:
        raise SystemExit(
            f"cannot find the test job's pytest command in {WORKFLOW}; "
            "the workflow's shape changed, so update this parser rather "
            "than restating the selection here")
    marker = match.group(1)
    paths = [token for token in match.group(2).split()
             if not token.startswith("-")]
    if not paths:
        raise SystemExit(
            f"parsed no test paths out of {WORKFLOW}; update this parser")
    return marker, paths


def parse_test_job_flags(workflow_text: str) -> list[str]:
    """The job's OTHER pytest flags, in the order the job writes them.

    THE BREAKAGE THIS PREVENTS, and it was live in this script for one
    revision.  The job runs ``-rs``, and ``-rs`` is not decoration in this
    suite: a dozen test modules skip themselves by name while the installed
    engine is behind, and each of those skips carries the missing symbol and
    the patch item that supplies it.  A replay that dropped ``-rs`` printed
    ``s`` and no sentence, so the one thing a reader needs in order to tell
    "the engine is behind" from "this package is broken" was missing from
    exactly the transcript that has to be read before a tag is created.

    ``-m`` and its expression are excluded; they come back from
    :func:`parse_test_job`.
    """

    marker_expression, _paths = parse_test_job(workflow_text)
    match = re.search(r"python -m pytest([^\n]*)", workflow_text)
    if not match:  # pragma: no cover - parse_test_job refuses first
        raise SystemExit(
            f"cannot find the test job's pytest command in {WORKFLOW}")
    # shlex, not split(): the marker expression is one quoted word and
    # naive splitting hands `-m` the token `"not` and leaves the rest of
    # the expression looking like five separate flags.
    tokens = shlex.split(match.group(1))
    # The trailing test paths come off first, then `-m` and its expression.
    # WHAT IS LEFT IS PASSED THROUGH UNTOUCHED, in order, because a flag can
    # carry a value in the next token: filtering on `startswith("-")` alone
    # dropped `no:cacheprovider` off the back of `-p` and the replay then ran
    # with a cache the job does not have.
    while tokens and tokens[-1] in _paths:
        tokens.pop()
    flags: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "-m":
            skip_next = True
            continue
        flags.append(token)
    if "-m" in flags or marker_expression in " ".join(flags):
        raise SystemExit(
            "the selection leaked into the reporting flags; the parser and "
            f"the job in {WORKFLOW} disagree about the pytest command")
    return flags


def run(argv: list[str], *, cwd: Path, env: dict[str, str], label: str) -> None:
    print(f"\n=== {label}\n    $ {' '.join(argv)}\n    (cwd {cwd})", flush=True)
    completed = subprocess.run(argv, cwd=str(cwd), env=env)
    if completed.returncode != 0:
        raise SystemExit(
            f"replay step failed (exit {completed.returncode}): {label}")


def build_env(workdir: Path) -> dict[str, str]:
    """The environment block the whole replay runs under.

    Separated from :func:`main` so the two properties that changed a
    verdict when they were missing are reachable from a test without
    replaying a whole job.
    """

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # The job sets this and it is not cosmetic: a developer box with a card
    # in it takes a different route through device admission than a runner
    # without one, and the replay answers for the runner.
    env["GPUWM_NO_LOCAL_GPU"] = "1"

    # The runner's home carries no ~/.woof; a developer box does, and a
    # stale staged binary there has already turned a doctor probe from a
    # skip into a failure that named the machine rather than the tree.
    home = workdir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)

    # ...but pip keeps the real cache.  With HOME redirected pip loses its
    # usual anchor, and re-downloading the engine's dependency set can fail
    # a replay that had the bytes already.  Cache reuse cannot change a
    # verdict; a network blip can.
    if "PIP_CACHE_DIR" not in env:
        if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
            env["PIP_CACHE_DIR"] = str(
                Path(os.environ["LOCALAPPDATA"]) / "pip" / "Cache")
        else:
            env["PIP_CACHE_DIR"] = os.path.expanduser("~/.cache/pip")
    return env


def make_venv(venv_dir: Path, *, interpreter: str) -> None:
    """A virtual environment with pip in it, whatever the interpreter is.

    `python -m venv` bootstraps pip through `ensurepip`, and an
    interpreter can perfectly well exist without it: the standalone
    CPython builds that version managers hand out ship no `ensurepip`
    payload, and Debian splits it into a separate package.  The failure
    is a `CalledProcessError` out of `venv/__init__.py` that names
    ensurepip and nothing else, which reads like a broken interpreter
    rather than a missing optional component.

    THE BREAKAGE THIS PREVENTS: this replay has to run on whatever
    interpreter the box has, because the whole point of the Linux leg is
    that it runs somewhere other than the machine the code was written
    on.  Refusing there would make the interpreter the CI matrix actually
    pins the one interpreter the replay cannot use.

    So `uv venv --seed` is the fallback when it is available, and when it
    is not the original failure is re-raised with what to install.
    """

    try:
        if interpreter == sys.executable:
            venv.create(venv_dir, with_pip=True)
        else:
            subprocess.run([interpreter, "-m", "venv", str(venv_dir)],
                           check=True)
        return
    except (subprocess.CalledProcessError, OSError) as failure:
        uv = shutil.which("uv")
        if uv is None:
            raise SystemExit(
                f"cannot create a virtual environment from {interpreter} "
                f"({failure}).  That interpreter has no ensurepip payload; "
                "install `uv` (pip install uv) and this replay will seed "
                "the environment with it instead") from failure
        print("")
        print("=== " + interpreter + " has no ensurepip; "
              "seeding the environment with uv", flush=True)
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        subprocess.run([uv, "venv", "--seed", "--python", interpreter,
                        str(venv_dir)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine-dist", type=Path, default=None,
        help="directory of locally built woof distributions to install "
             "before the package, for use while the declared engine floor "
             "is not yet on PyPI")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="where the venv and the built wheel go "
                             "(default: a fresh temporary directory)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the workdir instead of removing it")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter to build the venv from "
                             "(CI runs 3.11 and 3.12; default: this one)")
    args = parser.parse_args()

    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    marker, test_paths = parse_test_job(workflow_text)
    pytest_flags = parse_test_job_flags(workflow_text)
    print(f"replaying {WORKFLOW.name}'s test job against {REPO_ROOT}")
    print(f"  marker: -m {marker!r}")
    print(f"  flags:  {pytest_flags}")
    print(f"  paths:  {test_paths}")

    workdir = args.workdir.resolve() if args.workdir else Path(
        tempfile.mkdtemp(prefix="gpuwm-global-replay-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"  work:   {workdir}")

    env = build_env(workdir)

    venv_dir = workdir / "venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    make_venv(venv_dir, interpreter=args.python)
    python = venv_python(venv_dir)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=workdir, env=env, label="upgrade pip in the replay venv")
    run([str(python), "-m", "pip", "install", *VENV_SEEDS],
        cwd=workdir, env=env,
        label="seed the venv with what the CI interpreter already has "
              f"({', '.join(VENV_SEEDS)})")

    dist = workdir / "dist"
    run([str(python), "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=REPO_ROOT, env=env, label="build the wheel the job installs")
    wheels = sorted(glob.glob(str(dist / "*.whl")))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, got {wheels}")

    if args.engine_dist is not None:
        engine = args.engine_dist.resolve()
        engine_wheels = sorted(glob.glob(str(engine / "*.whl")))
        if not engine_wheels:
            raise SystemExit(f"no engine wheels under {engine}")
        print("\n*** ENGINE ROUTE: locally built distributions, NOT PyPI.")
        print("*** Every result below is a proof against these bytes:")
        for path in engine_wheels:
            print(f"***   {path}")
        run([str(python), "-m", "pip", "install", *engine_wheels],
            cwd=workdir, env=env,
            label="install the locally built engine distributions")
    else:
        print("\n*** ENGINE ROUTE: PyPI, exactly as the CI job resolves it.")

    run([str(python), "-m", "pip", "install", wheels[0]],
        cwd=workdir, env=env, label="install the built wheel")

    # An entry point that resolves to nothing is a packaging defect a run
    # against the source tree cannot see, so the job checks it before the
    # suite and so does this.
    run([str(python), "-c",
         "import woof.globe; print(woof.globe.__version__); "
         "print(woof.globe.__file__)"],
        cwd=workdir, env=env, label="the installed distribution answers")

    print(f"\n=== pytest {' '.join(pytest_flags)} -m {marker!r} "
          f"{' '.join(test_paths)}", flush=True)
    completed = subprocess.run(
        [str(python), "-m", "pytest", *pytest_flags, "-m", marker,
         *test_paths],
        cwd=str(REPO_ROOT), env=env)

    if not args.keep and args.workdir is None and completed.returncode == 0:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"removed {workdir}")
    else:
        print(f"kept {workdir}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
