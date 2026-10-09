"""The workflows are a contract, and the publisher is switched off.

Three things about `.github/workflows/` are asserted here rather than
trusted, because each one fails silently and each one has a cost:

1.  **The publisher cannot fire by accident.**  Every job in
    `publish.yml` is gated on a repository variable that only a deliberate
    act in the repository settings sets.  A missing variable is the empty
    string, so every job is skipped and the run completes having done
    nothing.  If that guard is ever dropped from one job, the file
    goes from "reviewable" to "armed" without anything else changing, and
    nothing in a diff review reliably catches a deleted `if:` line.
2.  **The selections agree.**  `test.yml` and `publish.yml` run the same
    pytest selection, and `tools/ci_test_replay.py` parses it out of the
    workflow rather than restating it.  A publish that trusts a green from
    another workflow is trusting a run against another ref, so the
    selection is written twice on purpose and checked here.
3.  **The privileges are separated.**  The cell with the PyPI identity has
    no repository write authority, the cell that promotes the release has
    no PyPI identity, and `build.yml` has neither.

The workflows are parsed with a small purpose-built reader rather than a
YAML library.  The CI environment installs the built wheel, pytest and
setuptools and nothing else; a gate that needs a fourth package would
either add one to the runtime or skip itself, and a provenance-shaped gate
that skips is a gate that reports green while proving nothing.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

#: The repository variable every publish job is gated on.  It does not
#: exist, and creating it is the deliberate act that arms the publisher.
PUBLISH_SWITCH = "GPUWM_GLOBAL_PUBLISH_AUTHORIZED"

#: The exact pytest selection the CI test job runs.
SELECTION = "not gpu and not slow and not network"


def workflow(name: str) -> str:
    path = WORKFLOWS / name
    assert path.is_file(), f"{path} is missing"
    return path.read_text(encoding="utf-8")


def directives(text: str) -> str:
    """The workflow with its comments removed.

    THE BREAKAGE THIS AVOIDS, and it is a false positive rather than a
    false negative, which is the kind that erodes a gate until somebody
    weakens it: the privilege checks below look for `id-token: write` and
    `contents: write`, and both strings appear in the header comments that
    EXPLAIN the privilege separation.  A comment describing a permission
    is not a permission, and a gate that cannot tell them apart teaches
    the next author to stop writing the comment.
    """

    return chr(10).join(line.split("#", 1)[0] for line in text.splitlines())


def jobs(text: str) -> dict[str, str]:
    """Job name to job body, for a workflow written at two-space indent.

    Deliberately small and deliberately strict: it finds the `jobs:` key at
    column zero and then every key at exactly two spaces of indent under
    it.  If a workflow is ever reindented this returns nothing, and
    :func:`test_the_reader_sees_the_jobs_that_are_there` fails rather than
    every gate below passing on an empty dictionary.
    """

    start = re.search(r"^jobs:\s*$", text, re.MULTILINE)
    if not start:
        return {}
    body = text[start.end():]
    names = list(re.finditer(r"^  ([A-Za-z_][A-Za-z0-9_-]*):\s*$", body,
                             re.MULTILINE))
    found: dict[str, str] = {}
    for index, match in enumerate(names):
        end = names[index + 1].start() if index + 1 < len(names) else len(body)
        found[match.group(1)] = body[match.start():end]
    return found


# ---------------------------------------------------------------------------
# the reader has to be able to see
# ---------------------------------------------------------------------------
def test_the_reader_sees_the_jobs_that_are_there() -> None:
    """Validate the instrument before believing anything it reports."""

    publish = jobs(workflow("publish.yml"))
    assert set(publish) >= {"test", "cut", "prepare", "doors", "publish",
                            "release"}, sorted(publish)
    assert set(jobs(workflow("test.yml"))) >= {"engine_availability", "test"}
    assert set(jobs(workflow("build.yml"))) >= {"build"}
    assert set(jobs(workflow("bridges.yml"))), "bridges.yml declares no jobs"


# ---------------------------------------------------------------------------
# 1.  the publisher cannot fire
# ---------------------------------------------------------------------------
def test_every_publish_job_is_gated_on_the_switch() -> None:
    text = workflow("publish.yml")
    guard = re.compile(
        r"if:\s*vars\." + PUBLISH_SWITCH + r"\s*==\s*'1'")
    ungated = [name for name, body in jobs(text).items()
               if not guard.search(body)]
    assert ungated == [], (
        f"these publish jobs carry no `if: vars.{PUBLISH_SWITCH} == '1'` "
        f"guard, so they would run: {ungated}.  The whole file is disabled "
        "by construction, and it is disabled one job at a time")


def test_the_switch_is_named_in_the_file_that_uses_it() -> None:
    """A guard nobody can find is a guard nobody can audit.

    The variable name appears in the header comment as well as in every
    `if:`, so a reader who opens the file learns what the switch is and
    what throwing it would do, without reading six job definitions.
    """

    text = workflow("publish.yml")
    header = text.split("name: publish", 1)[0]
    assert PUBLISH_SWITCH in header, (
        "the header comment does not name the repository variable the file "
        "is gated on")
    assert text.count(PUBLISH_SWITCH) >= 7, (
        "the switch should appear once in the header and once per job")


def test_publish_fires_on_no_push_and_no_bare_tag() -> None:
    """A workflow that publishes on a push is a standing authorisation.

    Publishing a GitHub release is a deliberate act on a specific tag.  A
    `push:` trigger here, even filtered to tags, would mean that creating a
    tag uploads to PyPI, and a tag is created long before anybody has
    decided to publish one.
    """

    text = workflow("publish.yml")
    trigger = text.split("\npermissions:", 1)[0].split("\non:", 1)[1]
    assert "release:" in trigger and "types: [published]" in trigger
    assert "workflow_dispatch:" in trigger
    assert not re.search(r"^\s{2}push:", trigger, re.MULTILINE), (
        "publish.yml has a push trigger:\n" + trigger)
    assert "pull_request_target" not in text, (
        "pull_request_target runs with the base repository's secrets "
        "against a fork's code")


def test_no_workflow_reads_a_publishing_secret() -> None:
    """Trusted Publishing means there is no credential to read.

    OIDC has the runner prove its identity to PyPI directly.  A token in
    this repository would be a long-lived credential to leak, to rotate,
    and to print by accident, and its presence would mean the OIDC path is
    not the one actually in use.
    """

    banned = re.compile(
        r"(secrets\.[A-Z_]*(PYPI|TWINE|TOKEN|PASSWORD)"
        r"|TWINE_PASSWORD|TWINE_USERNAME|password:)", re.IGNORECASE)
    offenders = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for match in banned.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: {match.group(0)}")
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# 2.  the selections agree
# ---------------------------------------------------------------------------
def _selections(text: str) -> list[str]:
    return re.findall(r'python -m pytest[^\n]*? -m "([^"]+)"', text)


def test_the_test_job_runs_the_declared_selection() -> None:
    found = _selections(workflow("test.yml"))
    assert found == [SELECTION], found


def test_the_publish_job_runs_the_same_selection() -> None:
    """Written twice on purpose, and checked so the two cannot drift.

    A publish that trusted test.yml's green would be trusting a run against
    another ref.  It runs the suite itself, and if the two expressions ever
    differ the release is proving something other than what CI proves.
    """

    assert _selections(workflow("publish.yml")) == [SELECTION]


def test_the_local_replay_parses_the_selection_out_of_the_workflow() -> None:
    """The replay must read the job, not a copy of the job.

    THE BREAKAGE THIS PREVENTS: a hand-maintained copy of a CI selection in
    a local harness drifts, and then a green replay before a tag means
    nothing about the job that runs after it.  The release memory of this
    family requires that replay to be green BEFORE a tag exists, because a
    tag is spent the moment it is created.
    """

    import importlib.util

    path = ROOT / "tools" / "ci_test_replay.py"
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location("_ci_test_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    marker, paths = module.parse_test_job(workflow("test.yml"))
    assert marker == SELECTION
    assert paths == ["tests"], paths

    source = path.read_text(encoding="utf-8")
    assert SELECTION not in source, (
        "the replay restates the CI selection as a literal instead of "
        "parsing it; that copy is exactly what drifts")


def test_the_local_replay_carries_the_jobs_reporting_flags() -> None:
    """The replay runs the job's pytest command, not a shorter one.

    THE BREAKAGE THIS PREVENTS, measured on this tree: the replay ran
    `-q -p no:cacheprovider` while the job runs `-q -rs --durations=15 -p
    no:cacheprovider`.  Dozens of modules in this suite skip themselves by
    name while the installed engine is behind, and each skip names the
    missing symbol and the patch item that supplies it.  Without `-rs` the
    replay transcript printed `s` characters and no sentences, so the
    transcript that must be read before a tag could not distinguish an
    engine that is behind from a package that is broken.
    """

    import importlib.util

    path = ROOT / "tools" / "ci_test_replay.py"
    spec = importlib.util.spec_from_file_location("_ci_test_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    flags = module.parse_test_job_flags(workflow("test.yml"))
    reporting = [f for f in flags if f.startswith("-r")]
    assert reporting, flags
    # Skip reasons must be in the report, and so must the failure list: `-r`
    # REPLACES pytest's default `fE` rather than adding to it, so `-rs` alone
    # produces a log naming no failed test.
    assert all("s" in f or "a" in f or "A" in f for f in reporting), reporting
    assert any("a" in f or "A" in f or ("f" in f and "E" in f)
               for f in reporting), reporting
    # A flag that takes a value keeps it: filtering on a leading dash alone
    # dropped `no:cacheprovider` off the back of `-p`.
    assert flags[flags.index("-p") + 1] == "no:cacheprovider", flags
    # The selection does not leak into the flag list; it has its own parser.
    assert "-m" not in flags and SELECTION not in " ".join(flags), flags


# ---------------------------------------------------------------------------
# 3.  privileges are separated
# ---------------------------------------------------------------------------
def test_only_the_pypi_cell_can_mint_an_oidc_token() -> None:
    publish = jobs(directives(workflow("publish.yml")))
    with_token = [name for name, body in publish.items()
                  if "id-token: write" in body]
    assert with_token == ["publish"], (
        "the OIDC identity belongs to the upload cell and nowhere else: "
        f"{with_token}")
    assert "environment: pypi" in publish["publish"], (
        "the trusted publisher registration binds the environment name; "
        "without it PyPI rejects the upload as an authentication failure")
    assert "actions/checkout" not in publish["publish"], (
        "the cell holding the PyPI identity has no checkout, so it cannot "
        "build or alter what it uploads")
    assert "contents: write" not in publish["publish"]


def test_only_the_release_cell_can_write_to_the_repository() -> None:
    publish = jobs(directives(workflow("publish.yml")))
    with_write = [name for name, body in publish.items()
                  if "contents: write" in body]
    assert with_write == ["release"], with_write
    assert "id-token" not in publish["release"], (
        "the cell that promotes the release has no PyPI identity")


def test_the_build_workflow_can_publish_nothing() -> None:
    """build.yml runs on every push, including tag pushes.

    It exists to prove the artefact, which means it runs in the least
    deliberate circumstances of any workflow here.  It must therefore hold
    no identity of any kind.
    """

    text = directives(workflow("build.yml"))
    assert "id-token" not in text
    assert "environment:" not in text
    assert "gh-action-pypi-publish" not in text
    assert "contents: write" not in text
    assert "permissions:\n  contents: read" in text


def test_the_publisher_action_is_the_registered_one() -> None:
    text = directives(workflow("publish.yml"))
    uses = re.findall(r"uses: (pypa/gh-action-pypi-publish@[0-9a-f]{40})",
                      text)
    assert len(uses) == 1, (
        f"exactly one upload step, and it is the pinned action: {uses}")


def test_every_action_is_pinned_to_a_commit() -> None:
    """A moving tag on a third-party action is somebody else's write access.

    `@v4` resolves to whatever that tag points at on the day the job runs,
    inside a job that in this repository can hold an upload identity.
    """

    floating = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for reference in re.findall(r"^\s*(?:- )?uses: (\S+)", text,
                                    re.MULTILINE):
            if reference.startswith("./"):
                continue  # a reusable workflow in this repository
            if not re.search(r"@[0-9a-f]{40}$", reference):
                floating.append(f"{path.name}: {reference}")
    assert floating == [], floating


# ---------------------------------------------------------------------------
# the seam between this lane and the doors lane
# ---------------------------------------------------------------------------
def test_the_release_job_and_the_door_bundles_agree_on_their_contract() -> None:
    """publish.yml still calls bridges.yml, and the release page is checked
    against the wheel's own pins rather than against that build.

    THE BREAKAGE THIS PREVENTS, in two halves.  bridges.yml builds the
    doors from the engine's public tag on a runner whose toolchain is not
    the release build's, so its bytes are never the released ones, and
    before that tag carried the crates it failed by name; a publish whose
    upload waited on that job would gate PyPI on a build that proves only
    that the source compiles, and before the tag could never upload.  So
    the upload waits on the tests and the distributions only, and the release
    job asks the release page for one bundle per platform the pins declare
    and re-hashes each against `door-pins.json` inside the checkout.  A
    release page carrying a bundle for one platform and none for the other,
    or a bundle whose bytes are not the pinned ones, is refused before the
    distributions are attached.  bridges.yml stays called on every cut so
    a public tag that stops building the doors is visible as a red job here.
    """

    bridges = workflow("bridges.yml")
    publish = workflow("publish.yml")

    assert "workflow_call:" in bridges, (
        "bridges.yml has to be callable for publish.yml to call it")
    assert "uses: ./.github/workflows/bridges.yml" in publish, (
        "publish.yml must call the doors workflow rather than carry a "
        "second copy of the build recipe for the same binaries")

    publish_jobs = jobs(directives(publish))
    for name in ("publish", "release"):
        needs = re.search(r"needs:\s*\[([^\]]*)\]", publish_jobs[name])
        assert needs, f"the {name} job declares no needs list"
        assert "doors" not in needs.group(1), (
            f"the {name} job waits on the doors build, whose bytes are not "
            "the released ones; the bundles come from the pins")

    release = publish_jobs["release"]
    for needle in ("door-pins.json", "gh release view", "gh release download",
                   "sha256sum"):
        assert needle in release, (
            f"the release job no longer {needle!r}: the release page's "
            "bundles must be checked by name and by bytes against the pins")



# ---------------------------------------------------------------------------
# the suite job can run the gates that read history
# ---------------------------------------------------------------------------
def test_the_suite_job_checks_out_the_history_its_gates_read() -> None:
    """The suite job fetches full history, so the line-ending gate runs.

    THE BREAKAGE THIS PREVENTS: two tests in `tests/test_line_endings.py`
    compare this branch's stored bytes against `main` and refuse a commit
    that rewrote a file's line endings.  `actions/checkout` fetches one
    ref by default, that checkout has neither `main` nor `origin/main`,
    and both tests skip themselves by name.  The gate then reports green
    in the one place a reviewer reads it, having measured nothing, which
    is exactly the failure it was written for: ten test files flipped
    whole and nobody saw it.
    """

    body = directives(jobs(workflow("test.yml")).get("test", ""))
    assert body.strip(), "test.yml declares no `test` job"

    lines = body.splitlines()
    checkout = [index for index, line in enumerate(lines)
                if re.search(r"uses: actions/checkout@[0-9a-f]{40}", line)]
    assert checkout, "the suite job does not check the repository out"

    step = []
    for line in lines[checkout[0] + 1:]:
        if line.strip().startswith("- "):
            break
        step.append(line)
    assert any(re.match(r"^\s+fetch-depth:\s*0\s*$", line) for line in step), (
        "the suite job's checkout does not set `fetch-depth: 0`, so the "
        "line-ending gate has no `main` to compare against and skips")


@pytest.mark.parametrize("name", ["test.yml", "build.yml", "publish.yml"])
def test_every_workflow_this_lane_owns_is_present(name: str) -> None:
    assert (WORKFLOWS / name).is_file()
