"""The six source mappings this model names, and the resolver that finds them.

A source is metadata by rule, so a mapping belongs in the ENGINE's authority
table.  No published engine has these six: woof 2.7.0 ships sixty rows under
`woof/authorities/` and not one of them is the GDAS global analysis, the GFS
cloud-cover reference, the ATMS column product, the ECMWF open-data forecast
or either GFS surface product.  Against that engine every shipped GDAS
experiment refused at its first door with `found 0 (none)`, which is not a
package anyone can run.

So the bytes ride in the package as well, and this file holds the two halves
of that decision:

* THE BYTES ARE THE SOURCE TREE'S.  Each carried file is byte-identical
  to the engine checkout this model's numbers were measured in.  A mapping that drifted would
  decode a different set of records under the same file name, and every
  receipt would still record that name.
* THE ENGINE IS ASKED FIRST.  The resolver reads the engine's table before
  its own copies, so the day a published engine takes a row that row is the
  one every command opens and the carried copy is never read again.  When
  both carry a spec with DIFFERENT bytes the resolver refuses by name rather
  than choosing, because choosing silently is how a run gets initialized
  from a source table nobody looked at.

The byte check needs the engine checkout the carve reads from, which exists
only on the machine the carve is run from; it skips by name elsewhere rather
than passing vacuously.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from woof.globe.analysis_initial import (
    PACKAGE_AUTHORITIES_DIR,
    resolve_analysis_mapping,
    resolve_analysis_mapping_row,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

#: The engine checkout the carve takes these bytes from, named by the same
#: environment variable `tools/resync_from_owner.py` takes its source tree
#: from.  A path is never written down here: a path that resolves on one
#: machine is not an instruction to anybody else, and the byte check is only
#: meaningful where the source tree is.
SOURCE_TREE_ENV = "WOOF_GLOBAL_SOURCE_WORKTREE"


def _source_authorities() -> Path | None:
    root = os.environ.get(SOURCE_TREE_ENV)
    if not root:
        return None
    directory = Path(root) / "woof" / "authorities"
    return directory if directory.is_dir() else None


def _carried_names() -> tuple[str, ...]:
    """The file names the re-cut carries, read off the re-cut itself.

    Listed in one place only.  A test that restated the six would keep
    passing on the day the re-cut stopped carrying one of them.
    """

    from resync_from_owner import AUTHORITY_MAPPINGS

    return tuple(AUTHORITY_MAPPINGS)


def test_every_carried_mapping_is_in_the_package_and_in_the_recut() -> None:
    names = _carried_names()
    assert len(names) == 6
    on_disk = sorted(path.name for path in
                     PACKAGE_AUTHORITIES_DIR.glob("*.mapping.json"))
    assert on_disk == sorted(names)


@pytest.mark.parametrize("name", _carried_names())
def test_a_carried_mapping_is_valid_json_the_decoder_can_read(name) -> None:
    """Not a byte blob: the decoder opens these and reads two keys first."""

    document = json.loads(
        (PACKAGE_AUTHORITIES_DIR / name).read_text(encoding="utf-8"))
    assert document["schema"] == "rw-wps.mapping.v1"
    assert document["format"] == "grib2"
    # Self-contained, which is why the carve is six files and not eighteen:
    # `decode_mapped_source` opens the mapping and the inputs, and nothing
    # else.  A composed source keeps a `.composition.json` beside it and
    # these six are not composed.
    assert "composition" not in document


@pytest.mark.parametrize("name", _carried_names())
def test_a_carried_mapping_is_the_source_trees_bytes(name) -> None:
    """The graded bytes, not a transcription of them.

    Runs where the source tree is, which is the machine the carve is run
    from; elsewhere it skips by name rather than passing vacuously.
    """

    authorities = _source_authorities()
    if authorities is None:
        pytest.skip(
            f"{SOURCE_TREE_ENV} does not name an engine checkout with an "
            "authorities directory; the byte check runs where the carve is "
            "run")
    source = authorities / name
    assert source.is_file(), f"{name} is not in {authorities}"
    carried = PACKAGE_AUTHORITIES_DIR / name
    assert hashlib.sha256(carried.read_bytes()).hexdigest() == \
        hashlib.sha256(source.read_bytes()).hexdigest(), name


@pytest.mark.parametrize("name", _carried_names())
def test_the_recut_does_not_rewrite_a_carried_mapping(name) -> None:
    """The rewiring is exempted here, and the exemption has to bite.

    Two of the six name `gpuwm.arwen_global.surface_seeding` in their notes,
    which is exactly the string the re-cut rewrites everywhere else.  If the
    exemption were dropped, the carried copy would differ from the engine's
    by that prose and the resolver would refuse every command the day the
    engine published the row.
    """

    from resync_from_owner import VERBATIM_PREFIXES

    assert any(f"data/authorities/{name}".startswith(prefix)
               for prefix in VERBATIM_PREFIXES)


# ------------------------------------------------------------- the resolver

def test_every_named_source_resolves_on_this_engine() -> None:
    """The whole point: no command refuses for want of a source row."""

    for spec in ("gdas-global", "gfs-surface-state", "gfs-surface-flux",
                 "ecmwf-open-data-global-forecast"):
        row = resolve_analysis_mapping_row(spec)
        assert row.path.is_file(), spec
        assert row.origin in ("engine", "package"), spec
    for name in ("rw-wps-gfs-pgrb2-0p25-cloud-cover.mapping.json",
                 "rw-wps-gdas-pgrb2-0p25-microwave-columns.mapping.json"):
        assert resolve_analysis_mapping(name).name == name


def test_an_ambiguous_family_id_is_still_refused() -> None:
    """`gdas` names three products; a config has to be specific."""

    with pytest.raises(ValueError, match="exactly one"):
        resolve_analysis_mapping("gdas")


def test_a_spec_no_table_answers_names_both_tables() -> None:
    with pytest.raises(ValueError) as caught:
        resolve_analysis_mapping("no-such-source")
    text = str(caught.value)
    assert "found 0 (none)" in text
    assert str(PACKAGE_AUTHORITIES_DIR) in text


def test_a_named_path_that_is_not_there_is_not_silently_substituted(
        tmp_path) -> None:
    """A caller who typed a path meant that file."""

    with pytest.raises(FileNotFoundError):
        resolve_analysis_mapping(
            str(tmp_path / "authorities"
                / "rw-wps-gdas-global-analysis-grib2.mapping.json"))


def test_a_named_path_that_exists_wins(tmp_path) -> None:
    carried = PACKAGE_AUTHORITIES_DIR / "rw-wps-gdas-global-analysis-grib2.mapping.json"
    copy = tmp_path / carried.name
    copy.write_bytes(carried.read_bytes())
    row = resolve_analysis_mapping_row(str(copy))
    assert row.origin == "path"
    assert row.path == copy.resolve()


#: The two mapping specs this package names in its own source as bare FILE
#: NAMES, with the module constant each is read from.
_BARE_FILE_NAME_SPECS = (
    ("rw-wps-gdas-pgrb2-0p25-microwave-columns.mapping.json",
     "woof.globe.microwave.columns", "mapping_path"),
    ("rw-wps-gfs-pgrb2-0p25-cloud-cover.mapping.json",
     "woof.globe.radiation_scorecard", "reference_mapping_path"),
)


@pytest.mark.parametrize("spec, module, reader", _BARE_FILE_NAME_SPECS)
def test_a_file_of_that_name_in_the_working_directory_does_not_win(
        spec, module, reader, tmp_path, monkeypatch) -> None:
    """A bare mapping name is a TABLE KEY, never a working-directory file.

    The ATMS operator and the radiation scorecard name their mapping by
    file name, in this package's own source.  While any existing file of
    that name won on the working directory alone, `woof global microwave
    columns` run from a directory a fetch had populated could decode
    through a document neither authority table publishes, the receipt would
    record a bare relative name, and the refusal that catches a substituted
    source table could not fire because only one table was ever asked.
    """

    import importlib

    shadow = tmp_path / spec
    shadow.write_text(json.dumps({"bogus": True}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    row = resolve_analysis_mapping_row(spec)
    assert row.origin in ("engine", "package")
    assert row.path.is_absolute()
    assert row.path.resolve() != shadow.resolve()
    assert getattr(importlib.import_module(module), reader)() == row.path


def test_a_bare_id_is_not_shadowed_by_a_directory_of_that_name(
        tmp_path, monkeypatch) -> None:
    """The id spelling has the same rule, and a directory can hold it."""

    (tmp_path / "gdas-global").mkdir()
    monkeypatch.chdir(tmp_path)
    row = resolve_analysis_mapping_row("gdas-global")
    assert row.origin in ("engine", "package")
    assert row.path.is_file()


@pytest.mark.parametrize("spelling", ("./{name}", ".{sep}{name}", "{absolute}"))
def test_a_caller_who_typed_a_path_still_gets_that_file(
        spelling, tmp_path, monkeypatch) -> None:
    """The other half: `--mapping` naming a local document keeps working.

    A bare name falling through to the tables must not take the typed-path
    door with it, or a caller pointing at a document they wrote has no
    spelling that reaches it.
    """

    carried = PACKAGE_AUTHORITIES_DIR / NAME
    local = tmp_path / NAME
    local.write_bytes(carried.read_bytes())
    monkeypatch.chdir(tmp_path)
    spec = spelling.format(name=NAME, sep=os.sep, absolute=str(local))
    row = resolve_analysis_mapping_row(spec)
    assert row.origin == "path", spec
    assert Path(row.path).resolve() == local.resolve(), spec
    # ABSOLUTE, whichever spelling was typed.  This path is what the run
    # receipt records, and `./name` recorded as `name` names a different
    # file on every machine that reads it.
    assert row.path.is_absolute(), spec
    assert row.path == local.resolve(), spec


def _fake_engine(tmp_path, monkeypatch, files: dict[str, bytes]) -> Path:
    """An engine authority table with exactly ``files`` in it."""

    directory = tmp_path / "engine-authorities"
    directory.mkdir()
    for name, payload in files.items():
        (directory / name).write_bytes(payload)
    monkeypatch.setattr(
        "woof.globe.analysis_initial._engine_authorities_dir",
        lambda: directory)
    return directory


NAME = "rw-wps-gdas-global-analysis-grib2.mapping.json"


def test_the_engine_row_wins_the_day_the_engine_has_it(
        tmp_path, monkeypatch) -> None:
    """The carried copy retires itself; it is not preferred, ever."""

    carried = PACKAGE_AUTHORITIES_DIR / NAME
    engine = _fake_engine(tmp_path, monkeypatch, {NAME: carried.read_bytes()})
    row = resolve_analysis_mapping_row("gdas-global")
    assert row.origin == "engine"
    assert row.path == engine / NAME
    assert row.package_path == carried


def test_an_engine_row_with_different_bytes_is_refused_by_name(
        tmp_path, monkeypatch) -> None:
    """A row that MOVED is the thing this refusal exists to catch.

    Identical bytes are the row landing.  Different bytes mean the engine's
    source table decodes a different set of records under the same file
    name, and a forecast initialized through it is not the forecast this
    model was graded as.  Picking either side silently is the failure.
    """

    carried = PACKAGE_AUTHORITIES_DIR / NAME
    moved = carried.read_bytes().replace(b'"format": "grib2"',
                                         b'"format": "grib2" ')
    assert moved != carried.read_bytes()
    _fake_engine(tmp_path, monkeypatch, {NAME: moved})
    with pytest.raises(ValueError) as caught:
        resolve_analysis_mapping("gdas-global")
    text = str(caught.value)
    assert "carried twice with different bytes" in text
    assert NAME in text
    # BOTH DIGESTS, because "they differ" alone leaves a reader unable to
    # tell whether the engine's row is one they have already seen; the
    # receipts record the same digest for the row that decoded a run.
    import hashlib

    assert hashlib.sha256(moved).hexdigest() in text
    assert hashlib.sha256(carried.read_bytes()).hexdigest() in text


def test_an_engine_that_does_not_import_is_not_a_missing_mapping(
        monkeypatch) -> None:
    """The engine's own absence has its own sentence, in the doctor.

    Letting an ImportError here turn into "this mapping does not exist"
    sends a reader to look for a source table when the problem is the
    install.
    """

    def explode():
        raise ImportError("woof is not installed")

    monkeypatch.setattr(
        "woof.globe.analysis_initial._engine_authorities_dir", explode)
    assert resolve_analysis_mapping_row("gdas-global").origin == "package"


def test_the_doctor_says_which_table_answered() -> None:
    """A second copy of a source on one machine is a printed line."""

    from woof.globe.doctor import build_report

    rows = dict(build_report().sections)["source mappings"]
    findings = {row.label: row.finding for row in rows}
    assert "gdas-global" in findings
    assert any(word in findings["gdas-global"]
               for word in ("carried by this package", "from the engine"))
    assert not [row for row in rows if row.verdict == "gap"]


def test_every_doctor_mapping_row_prints_the_digest_of_the_copy_that_answered(
) -> None:
    """The instrument reads; it does not assert.

    The summary row used to say the carried copies are "byte-identical to
    the tree this model was graded in" while doctor never opens that tree:
    the byte comparison lives in this file and skips unless
    WOOF_GLOBAL_SOURCE_WORKTREE names an engine checkout, so on a machine
    without one the claim was printed and measured by nothing.  What doctor
    CAN measure is the SHA-256 of the file that answered each row, which is
    the digest a run receipt records as mapping_sha256 and the number a
    reader compares.
    """

    from woof.globe.doctor import build_report

    rows = dict(build_report().sections)["source mappings"]
    printed = 0
    for row in rows:
        if row.label == "carried copies":
            joined = " ".join(row.detail)
            assert "byte-identical" not in joined
            assert "graded in" not in joined
            continue
        if row.verdict == "gap":
            continue
        answered = resolve_analysis_mapping_row(row.label).path
        digest = hashlib.sha256(answered.read_bytes()).hexdigest()
        assert f"sha256 {digest}" in row.detail, row.label
        printed += 1
    assert printed >= len(_carried_names())


def test_the_doctor_calls_a_disagreement_a_disagreement(
        tmp_path, monkeypatch) -> None:
    """Two tables answering differently is not "nobody answers".

    The row used to read `no authority table answers it` for a spec BOTH
    tables carry, which sends a reader to look for a file that is missing
    when the file is present twice and the two copies decode different
    records under one name.
    """

    from woof.globe.doctor import build_report

    carried = PACKAGE_AUTHORITIES_DIR / NAME
    moved = carried.read_bytes().replace(b'"format": "grib2"',
                                         b'"format": "grib2" ')
    assert moved != carried.read_bytes()
    _fake_engine(tmp_path, monkeypatch, {NAME: moved})

    rows = dict(build_report().sections)["source mappings"]
    row = next(row for row in rows if row.label == "gdas-global")
    assert row.verdict == "gap"
    assert row.finding == "the two tables disagree"
    assert any("carried twice with different bytes" in line
               for line in row.detail)
