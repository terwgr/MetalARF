"""The machine seam: the documents a client reads and the run it drives.

The claim this file holds is not "the code runs".  It is that a client
already written against the ENGINE's integration contract drives THIS package
by changing the module it spawns -- so the tests below use the shipped
Integration Kit's own ``client.py`` where it is on disk, and reimplement
nothing where it is not: the schema check, the manifest digest pin, the
torn-tail rule and the monotonic-sequence rule are the kit's, applied to a
real short run of this model.

Four things have to agree and each was free to move on its own before this
file existed:

* the query documents' schema ids, which a client pins,
* the plan document's shape, which a client writes,
* the three durable documents a run leaves, which a client reattaches to,
* and the registries the documents are VIEWS of, which grow by rows.

The end-to-end test at the bottom executes a real plan through the real
doors -- T21 on the numpy backend, twelve model hours, the Rust renderer --
because "a client can drive this" is not a claim any amount of unit coverage
makes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import pytest

from woof.globe import runplan
from woof.globe.configs_dir import config_root, list_configs

SHORT_EXPERIMENT = "arwen_global_t21_baroclinic_ten_day"

#: Twelve model hours of the T21 case: 120 steps at dt = 360 s, two output
#: intervals.  MEASURED on the Windows desktop 2026-09-09 at 10.2 s of
#: integration; the render stage is what makes the end-to-end case a minute.
SHORT_UNTIL_S = 43200.0

#: Twenty steps, which is the cold state plus the final checkpoint and about
#: two seconds of wall.  Enough to write all four durable documents, which is
#: what the CI-runnable end-to-end case needs; the forecast itself is proven at
#: length by every other file in this suite.
TINY_UNTIL_S = 7200.0

#: A shipped experiment whose ``[initial]`` table names a real analysis file,
#: used where a test needs a DECLARED INPUT it can point at nothing.  Read and
#: edited rather than authored here: a config written in a test would drift
#: from the loader's own rules and stop testing them.
ANALYSIS_EXPERIMENT = "arwen_global_t255_quickstart"


def _config_missing_its_analysis() -> str:
    text = (config_root() / f"{ANALYSIS_EXPERIMENT}.toml").read_text("utf-8")
    edited = re.sub(r"^analysis_grib = .*$",
                    'analysis_grib = "no-such-analysis.grib2"',
                    text, count=1, flags=re.MULTILINE)
    assert "no-such-analysis.grib2" in edited
    return edited


def _config_with_output_every(steps: int) -> str:
    """The T21 case with its output cadence rewritten to ``steps`` steps.

    Edited from the shipped TOML rather than authored here, for the same
    reason as the config above: a config written in a test would drift from
    the loader's own rules and stop testing them.  A tight cadence is what
    makes a restart's checkpoint count a number that MOVES with the restart
    point in a few seconds of wall.
    """

    text = (config_root() / f"{SHORT_EXPERIMENT}.toml").read_text("utf-8")
    dt = float(re.search(r"^dt_s = (.+)$", text, re.MULTILINE).group(1))
    edited = re.sub(r"^output_interval_s = .*$",
                    f"output_interval_s = {dt * steps}",
                    text, count=1, flags=re.MULTILINE)
    assert f"output_interval_s = {dt * steps}" in edited
    return edited


def _plan(tmp_path: Path, **overrides) -> Path:
    document = {
        "schema": runplan.PLAN_SCHEMA,
        "name": "a short T21 forecast",
        "route": "experiment",
        "config": {"path": SHORT_EXPERIMENT},
        "output_root": str(tmp_path / "run"),
        "run_options": {"until_s": SHORT_UNTIL_S, "overwrite": True},
    }
    document.update(overrides)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _cli(*arguments: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "arwen_global", *arguments],
        cwd=None if cwd is None else str(cwd),
        capture_output=True, text=True, encoding="utf-8", timeout=1800)


# --------------------------------------------------------------- the schemas


def test_every_schema_id_is_the_engines():
    """A client pins schema ids, so a typo here is a client that stops reading.

    The ids are IMPORTED from ``woof.runplan`` rather than spelled, and this
    holds that import to the engine's own values so a re-spelling in either
    tree fails a test instead of a client.
    """

    import woof.runplan as engine

    for name in ("PLAN_SCHEMA", "EVENT_SCHEMA", "MANIFEST_SCHEMA",
                 "RESOLVE_SCHEMA", "ESTIMATE_SCHEMA", "PROBE_SCHEMA",
                 "CATALOG_SCHEMA", "SOURCES_SCHEMA",
                 "PHYSICS_PROFILES_SCHEMA", "EVENTS_FILENAME",
                 "MANIFEST_FILENAME"):
        assert getattr(runplan, name) == getattr(engine, name), name
    assert runplan.EVENT_TAGS == engine.EVENT_TAGS


def test_the_manifest_carries_every_key_the_engines_does(tmp_path):
    """A reader written for the engine's manifest finds nothing missing.

    The engine's own ``write_manifest`` is run against a synthetic engine plan
    and the key sets are compared, so a field the engine adds is a failure
    here rather than a client that reattaches and cannot find the stream it
    was told to read.
    """

    import woof.runplan as engine

    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    engine_plan = engine.RunPlan(
        name="x", route="experiment", config_path=None,
        config_inline="[x]\n", config_intent=None,
        config_base_dir=engine_dir, fetch_arguments=None,
        output_root=engine_dir, run_options={}, sha256="0" * 64,
        source="synthetic", automatic_resolutions=())
    engine_manifest = json.loads(engine.write_manifest(
        engine_plan, run_dir=engine_dir, events_path=engine_dir / "events.jsonl",
        run_id="engine-run", started_at_utc="2026-09-09T00:00:00+00:00"
    ).read_text(encoding="utf-8"))

    package_dir = tmp_path / "package"
    package_dir.mkdir()
    plan = runplan.load_plan(_plan(tmp_path))
    package_manifest = json.loads(runplan.write_manifest(
        plan, run_dir=package_dir, run_id="package-run",
        started_at_utc="2026-09-09T00:00:00+00:00"
    ).read_text(encoding="utf-8"))

    missing = sorted(set(engine_manifest) - set(package_manifest))
    assert not missing, f"the engine's manifest carries {missing} and this one does not"
    assert package_manifest["schema"] == engine_manifest["schema"]
    assert package_manifest["producer"]["distribution"] == "woof global"


def test_the_heartbeat_is_the_engines_document(tmp_path):
    """``run-progress.json`` is built by the engine's dataclass and writer."""

    from woof.supervisor import HEARTBEAT_SCHEMA

    beat = runplan._Heartbeat(tmp_path / "run-progress.json", run_id="r1",
                              config_sha256="a" * 64,
                              started_at_utc="2026-09-09T00:00:00+00:00")
    beat.preparing("plan accepted")
    document = json.loads((tmp_path / "run-progress.json").read_text("utf-8"))
    assert document["schema"] == HEARTBEAT_SCHEMA
    assert document["status"] == "preparing:plan-accepted"
    beat.integrating(model_elapsed_seconds=60.0, outer_step=1)
    beat.complete()
    document = json.loads((tmp_path / "run-progress.json").read_text("utf-8"))
    assert document["status"] == "complete"
    assert document["outer_step"] == 1
    assert document["run_id"] == "r1"


# ----------------------------------------------------------- the plan shape


def test_a_plan_naming_a_shipped_experiment_resolves_to_the_wheels_copy(tmp_path):
    plan = runplan.load_plan(_plan(tmp_path))
    assert plan.config_path == (config_root() / f"{SHORT_EXPERIMENT}.toml")
    assert plan.route == "experiment"
    assert plan.run_dir == (tmp_path / "run").resolve()


def test_a_path_that_exists_beats_a_shipped_name(tmp_path):
    """A reader who copies a config next to their plan runs THEIR copy."""

    copy = tmp_path / f"{SHORT_EXPERIMENT}.toml"
    copy.write_bytes((config_root() / f"{SHORT_EXPERIMENT}.toml").read_bytes())
    plan = runplan.load_plan(_plan(
        tmp_path, config={"path": f"{SHORT_EXPERIMENT}.toml"}))
    assert plan.config_path == copy.resolve()


def test_the_prepared_route_is_refused_by_name(tmp_path):
    """The kit's own example plan names it; a silent remap would run something else."""

    path = _plan(tmp_path, route="prepared")
    with pytest.raises(runplan.PlanError) as refusal:
        runplan.load_plan(path)
    message = str(refusal.value)
    assert "prepared" in message
    assert "prepared-cache root" in message
    assert "route 'go'" in message


def test_an_unsupported_option_is_refused_and_never_dropped(tmp_path):
    """The kit says so in as many words: never silently discard an option."""

    path = _plan(tmp_path, run_options={"render_products": "2m_temperature"})
    with pytest.raises(runplan.PlanError) as refusal:
        runplan.load_plan(path)
    assert "render_products" in str(refusal.value)
    assert "go" in str(refusal.value)


def test_a_fetch_block_and_an_intent_are_refused_by_name(tmp_path):
    with pytest.raises(runplan.PlanError) as fetch_refusal:
        runplan.load_plan(_plan(tmp_path, fetch={"args": ["gdas"]}))
    assert "no fetch stage" in str(fetch_refusal.value)
    with pytest.raises(runplan.PlanError) as intent_refusal:
        runplan.load_plan(_plan(tmp_path, config={"intent": {"hours": 6}}))
    assert "no wizard" in str(intent_refusal.value)


def test_a_foreign_schema_is_refused(tmp_path):
    with pytest.raises(runplan.PlanError) as refusal:
        runplan.load_plan(_plan(tmp_path, schema="gpuwm.run-plan.v99"))
    assert "gpuwm.run-plan.v1" in str(refusal.value)


def test_every_route_option_has_a_default_and_a_validator(tmp_path):
    """A route may not name an option the defaults table cannot answer for."""

    for route in runplan.ROUTES.values():
        for key in route.run_options:
            assert key in runplan._RUN_OPTION_DEFAULTS, key
            # And the validator knows it: an option accepted by the parser and
            # unknown to _run_option would raise inside a run instead of at
            # the door.
            runplan._run_option(key, runplan._RUN_OPTION_DEFAULTS[key],
                                tmp_path)


# ------------------------------------------------------------- the registries


def test_the_source_registry_is_a_view_of_its_tables(monkeypatch):
    """A row appended to the table appears in the document with no edit here."""

    from woof.globe import obs_streams, sources

    grafted = dict(obs_streams.ANCHOR_SOURCES)
    grafted["grafted-analysis"] = obs_streams.AnchorSource(
        name="grafted-analysis",
        subject="a source appended to the table by this test",
        url_template="https://example.invalid/{yyyymmdd}/{hh}",
        mapping="grafted-mapping-id",
        cadence_s=6 * 3600,
        availability_basis="a test",
        notes="a test")
    monkeypatch.setattr(obs_streams, "ANCHOR_SOURCES", grafted)
    document = sources.source_inventory()
    ids = [row["source_id"] for row in document["sources"]]
    assert "grafted-analysis" in ids
    row = next(r for r in document["sources"] if r["source_id"] == "grafted-analysis")
    assert row["mapping"]["id"] == "grafted-mapping-id"
    # The mapping is MEASURED against the installed engine, so a graft that
    # names a mapping nothing carries reports absent rather than runnable.
    assert row["mapping"]["present"] is False
    assert row["maturity"]["runnable"] is False


def test_the_source_document_names_the_engine_fields_it_does_not_answer():
    document = runplan.source_inventory()
    assert document["schema"] == runplan.SOURCES_SCHEMA
    assert document["producer"]["distribution"] == "woof global"
    assert document["engine_fields_not_carried"]
    for row in document["sources"]:
        # Nothing is invented: every engine field this package cannot answer
        # is named with a reason rather than filled with a value.
        assert set(row["not_applicable"]) == set(
            document["engine_fields_not_carried"])
        assert row["role"] in document["roles"]


def test_a_registry_id_is_never_a_file_name():
    """``source_id`` and ``field_mapping`` are ids, on every row, always.

    Two of the four verification specs were spelled as the mapping file,
    which is how their readers open them, so the registry held an id on four
    rows and a file name on two of the SAME field.  Measured 2026-09-10
    against the installed 2.7.0's own ``woof sources --json``: all 32 of its
    rows carry an identifier in ``maturity.field_mapping`` and none carries a
    file name.  The file spelling is an alias here, and still resolves.
    """

    from woof.globe import sources

    document = sources.source_inventory()
    for row in document["sources"]:
        assert not row["source_id"].endswith(sources.MAPPING_SUFFIX), row
        spec = row["mapping"]["id"]
        assert not spec.endswith(sources.MAPPING_SUFFIX), row["source_id"]
        assert row["maturity"]["field_mapping"] == spec
        assert not row["maturity"]["field_mapping"].endswith(
            sources.MAPPING_SUFFIX), row["source_id"]
        for alias in row["aliases"]:
            # Every alias answers.  An alias nothing resolves is a spelling
            # published to a picker that then cannot use it.
            assert sources._resolve(document, alias) is row, alias
        if spec != row["source_id"]:
            assert spec in row["aliases"], row["source_id"]


def test_a_derived_label_is_never_dressed_as_a_registry_id():
    """``cadence_mapping`` names a policy table this registry does not have.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-10 against the installed
    2.7.0: every one of the engine's 32 rows carries a cadence_mapping drawn
    from a real set (`uniform-hourly-analysis-series-v1`,
    `uniform-gfs-forecast-series-v1`, `pending`, ...).  This package emitted
    `uniform-6-hour-cycle-series`, formatted from the cycle interval, in the
    same field: a string shaped exactly like an id that resolves nowhere and
    that no row or rule said was derived.  Null is an answer this document
    already uses, and the interval itself is carried in seconds.
    """

    from woof.globe import sources

    document = sources.source_inventory()
    assert "maturity.cadence_mapping" in document["engine_fields_not_carried"]
    for row in document["sources"]:
        assert row["maturity"]["cadence_mapping"] is None, row["source_id"]
    anchors = [row for row in document["sources"]
               if row["forcing_interval_seconds"] is not None]
    assert anchors, "no row carries the cycle interval the label was built from"
    for row in anchors:
        assert row["forcing_interval_seconds"] > 0


def test_a_row_holds_every_role_that_is_true_of_it(monkeypatch):
    """A role is a fact about the row, not a shadow of another field.

    THE BREAKAGE THIS PREVENTS: the background anchor's role was derived from
    the ABSENCE of a shipped experiment naming it, so the IFS row would have
    stopped reporting the assimilation role it still holds the moment one
    config initialized from it, with no other change anywhere.
    """

    from woof.globe import doctor, sources

    plain = {row["source_id"]: row for row in sources.source_rows()}
    assert set(plain["ifs-open-data"]["roles"]) == {"background_anchor"}
    assert plain["ifs-open-data"]["used_by"] == []

    monkeypatch.setattr(
        doctor, "_config_mapping_ids",
        lambda: {"ecmwf-open-data-global-forecast": ["a_grafted_experiment"]})
    grafted = {row["source_id"]: row for row in sources.source_rows()}
    row = grafted["ifs-open-data"]
    # BOTH, because both are true, and each says on what basis.
    assert set(row["roles"]) == {"background_anchor", "initialization"}
    assert all(row["roles"].values())
    assert row["role"] == "initialization"
    # ...and `used_by` is still the separate fact it always was.
    assert row["used_by"] == ["a_grafted_experiment"]


def test_every_shipped_experiment_lands_in_exactly_one_physics_profile():
    """A config the grouping cannot place would vanish from the menu."""

    document = runplan.physics_profile_menu()
    assert document["schema"] == runplan.PHYSICS_PROFILES_SCHEMA
    assert document["experiments_unplaced"] == []
    named = [name for profile in document["profiles"]
             for name in profile["experiments"]]
    assert sorted(named) == sorted(list_configs())
    assert len(named) == len(set(named)), "an experiment is in two profiles"


def test_the_physics_menu_measures_the_installed_engine():
    """The native suite's admissibility is a measurement, not an assertion."""

    from woof.globe.engine_compat import engine_signature_gaps

    document = runplan.physics_profile_menu()
    verdict = document["engine_verdict"]
    measured_gaps = engine_signature_gaps()
    native = [profile for profile in document["profiles"]
              if profile["mode"] == "arwen-native"]
    assert native, "the native suite is not in the menu"
    # REGISTERED is the other half of admissible, and it always was: an
    # adapter no install registers is closed whatever the engine boundary
    # says, and this comparison passed only while the boundary happened to
    # be closed too.  The carve opened it, and the unregistered row then
    # read as a contradiction rather than as the second reason it is.
    registered = [profile for profile in native if profile["registered"]]
    assert registered, "no native adapter is registered in this install"
    if measured_gaps:
        assert verdict["native_suite_runnable_here"] is False
        for profile in native:
            assert profile["admissible"] is False
            assert profile["why_not"]
        # The statement names the calls, so a reader is not sent hunting.
        for gap, _missing in measured_gaps:
            assert gap.name in verdict["statement"]
    else:
        for profile in registered:
            assert profile["admissible"] is verdict["native_suite_runnable_here"]
    for profile in native:
        if not profile["registered"]:
            assert profile["admissible"] is False
            assert "registered" in profile["why_not"]


def test_a_symbol_gap_that_stops_no_forecast_leaves_the_native_suite_admissible():
    """A gap stops what its row names, and this document says so twice.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10
    against published woof 2.7.2 on the merged tree.  `native_suite_runnable_
    here` was `not symbol_gaps`, which was the same sentence as "no native
    column symbol is missing" for exactly as long as every row in `GAPS` was
    a native column symbol.  The carve took that physics into this package
    and left two rows -- a card-pricing check no subcommand is wired to, and
    the surface-energy scorecard's regrid -- neither of which a forecast
    enters.  `run-plan --physics-profiles` then reported the native profile
    `admissible: false`, with a `why_not` naming those two, inside a document
    whose own statement read "each one stops what its row names and nothing
    else".  A picker reading that menu would not offer the physics this
    package exists to run.

    Both rows carry `stops_a_native_forecast=False` today, so this asserts
    the live document; the arm below asserts the other direction, because a
    verdict that can only answer one way is not a measurement.
    """

    from woof.globe.engine_compat import GAPS, engine_gaps

    standing = engine_gaps()
    document = runplan.physics_profile_menu()
    verdict = document["engine_verdict"]
    blocking = [gap for gap in standing if gap.stops_a_native_forecast]
    assert verdict["symbol_gaps_stopping_a_native_forecast"] == [
        row for row in verdict["symbol_gaps"]
        if row["stops_a_native_forecast"]]
    assert len(verdict["symbol_gaps_stopping_a_native_forecast"]) == len(blocking)
    if standing and not blocking:
        assert verdict["native_suite_runnable_here"] is True
        # Reported, not hidden: a gap that stops something else is still in
        # the document and still named with what it stops.
        assert verdict["symbol_gaps"], "a standing gap vanished from the menu"
        for gap in standing:
            assert gap.symbol in verdict["statement"]
        for profile in document["profiles"]:
            if profile["mode"] == "arwen-native" and profile["registered"]:
                assert profile["admissible"] is True, profile["why_not"]
    # Every row must have decided, which is what the dataclass field with no
    # default buys: a row added later cannot inherit an unasked answer.
    for gap in GAPS:
        assert isinstance(gap.stops_a_native_forecast, bool)


def test_a_symbol_gap_that_does_stop_a_forecast_closes_the_native_suite(
        monkeypatch):
    """The other direction of the same instrument.

    A verdict built from a table can be wrong by always saying yes.  This
    puts one blocking row in front of it and requires the menu to close the
    native profile and name that row, and only that row.
    """

    from woof.globe import engine_compat

    blocker = engine_compat.EngineGap(
        module="woof.core.preflight",
        symbol="a_symbol_no_engine_carries_" + "for_this_arm",
        stops="the native column forecast, in this test only",
        handling="refused",
        stops_a_native_forecast=True,
        # A forecast is `run` and `go`, which are documented commands.
        stops_a_documented_command=True)
    monkeypatch.setattr(engine_compat, "GAPS",
                        engine_compat.GAPS + (blocker,))
    document = runplan.physics_profile_menu()
    verdict = document["engine_verdict"]
    assert verdict["native_suite_runnable_here"] is False
    named = [row["symbol"] for row
             in verdict["symbol_gaps_stopping_a_native_forecast"]]
    assert named == [blocker.symbol], named
    assert blocker.symbol in verdict["statement"]
    native = [profile for profile in document["profiles"]
              if profile["mode"] == "arwen-native" and profile["registered"]]
    assert native, "the native suite is not in the menu"
    for profile in native:
        assert profile["admissible"] is False
        assert blocker.symbol in profile["why_not"]


def test_the_physics_menu_names_the_engine_fields_it_does_not_answer():
    """Carried with an answer, or NAMED with a reason.  Never simply absent.

    THE BREAKAGE THIS PREVENTS: the schema id on this document is the
    ENGINE's, so a picker built against the engine's own document keys the
    engine's row fields and reads the engine's top-level ones.  A field that
    is silently absent hands it a KeyError with nothing in the document, and
    nothing on the page, to explain the shape it got.  The source registry
    was already held to this; the menu was not.
    """

    document = runplan.physics_profile_menu()
    assert document["gpuwm_version"] == (
        document["producer"]["engine"]["version"])
    assert document["engine_fields_not_carried"]
    for row in document["profiles"]:
        assert set(row["not_applicable"]) == set(
            runplan.PROFILE_FIELDS_NOT_CARRIED), row["profile_id"]

    # MEASURED against the INSTALLED engine's own document rather than a
    # field list copied here: an engine that grows a row field fails this
    # until the field is either answered or named with its reason.
    import contextlib
    import io

    from woof.runplan import physics_profile_menu as engine_menu

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            engine = engine_menu()
    except Exception as error:  # noqa: BLE001 - an instrument says what it is
        pytest.skip("the installed engine's own physics menu could not be "
                    f"built here, so its field set is unmeasured: {error}")
    engine_row_fields = set().union(
        *(set(row) for row in engine["profiles"])) if engine["profiles"] else set()
    assert engine_row_fields, "the engine's menu carries no profile row"
    for row in document["profiles"]:
        unanswered = engine_row_fields - set(row) - set(row["not_applicable"])
        assert not unanswered, (row["profile_id"], sorted(unanswered))
    top = (set(engine) - set(document)
           - set(document["engine_fields_not_carried"]))
    assert not top, sorted(top)


@pytest.mark.doors
def test_the_catalog_never_claims_a_slug_it_has_not_measured():
    """``unmeasured`` is a third answer, and it is not ``renderable``."""

    document = runplan.render_catalog()
    assert document["schema"] == runplan.CATALOG_SCHEMA
    if document.get("products") is None:
        pytest.skip("no rw_wrfbatch is staged here; the catalog has no slugs "
                    "to report and says so in its error field")
    statuses = {row["global_tape_status"] for row in document["products"]}
    assert statuses <= {"renderable", "missing-fields", "excluded",
                        "unmeasured"}
    counts = document["global_tape_counts"]
    assert counts["listed"] == len(document["products"])
    assert (counts["renderable"] + counts["unavailable"]
            + counts["unmeasured"]) == counts["listed"]
    # The headline number says WHICH arm it counted, and the other arm's
    # totals are beside it rather than left for a picker to assume.
    assert "SINGLE-FRAME" in counts["basis"]
    # ...and WHERE it was measured, in the same block.  THE BREAKAGE THIS
    # PREVENTS, measured 2026-09-10 on a Linux host reading a table measured
    # on Windows: the availability half of these counts is a stored
    # measurement from one host, and a client reading the headline alone got
    # the other platform's number with nothing in the block to say so.  Every
    # stamp `global_tape` carries, this carries.
    stamp = document["global_tape"]
    for key in ("host", "platform", "measured_utc",
                "renderer_digest_matches_installed"):
        assert key in counts, key
        assert counts[key] == stamp[key], key
    series = counts["series"]
    assert (series["renderable"] + series["unavailable"]
            + series["unmeasured"]) == counts["listed"]
    assert series["renderable"] == sum(
        1 for row in document["products"]
        if row["global_tape_series_status"] == "renderable")
    assert counts["renderable"] == sum(
        1 for row in document["products"]
        if row["global_tape_status"] == "renderable")
    # Every unavailable slug carries the renderer's OWN reason.
    for row in document["products"]:
        if row["global_tape_status"] in ("missing-fields", "excluded"):
            assert row["global_tape_detail"]
    assert document["global_tape"]["measured_utc"]
    assert document["global_tape"]["renderer"]["sha256"]


@pytest.mark.doors
def test_the_catalogs_default_products_are_all_renderable():
    """The door's default must draw something, on this package's own tapes."""

    from woof.globe.render_door import DEFAULT_PRODUCTS

    document = runplan.render_catalog()
    if document.get("products") is None:
        pytest.skip("no rw_wrfbatch is staged here")
    status = {row["name"]: row["global_tape_status"]
              for row in document["products"]}
    for slug in DEFAULT_PRODUCTS.split(","):
        assert status.get(slug) == "renderable", slug


def test_the_probe_reports_readiness_without_claiming_a_verdict_it_lacks():
    document = runplan.probe_environment(readiness=False)
    assert document["schema"] == runplan.PROBE_SCHEMA
    # At the engine's own spelling for this schema id, measured rather than
    # declared: a client reading both probes reads one field.
    assert document["gpuwm_version"] == (
        document["producer"]["engine"]["version"])
    assert document["readiness"]["collected"] is False
    assert document["readiness"]["ready"] is None
    assert set(document["schemas"]) >= {
        "plan", "event", "manifest", "resolve", "estimate", "probe",
        "catalog", "sources", "physics_profiles"}
    assert "prepared" in document["routes"]
    assert document["routes"]["prepared"].startswith("REFUSED")


def test_the_probe_names_the_engine_fields_it_does_not_answer():
    """Carried with an answer, or NAMED with a reason.  Never simply absent.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-10 against the installed
    2.7.0 on this machine: this document reuses the ENGINE's schema id, and
    the engine's own probe carries a top-level `provenance` block saying which
    engine tree would execute.  This one dropped it without naming it, so a
    client that reads `provenance` from a `gpuwm.run-plan.probe.v1` document
    got a KeyError with nothing in the document to explain the shape.  The
    source registry and the physics menu were already held to this rule; the
    probe was not.
    """

    document = runplan.probe_environment(readiness=False)
    assert isinstance(document.get("provenance"), dict), (
        "the engine's probe carries a top-level provenance receipt and this "
        "one must either carry it or name it as not carried")
    assert document["provenance"].get("package_path")

    # MEASURED against the INSTALLED engine's own probe rather than a field
    # list copied here: an engine that grows a top-level probe field fails
    # this until the field is either answered or named with its reason.
    import contextlib
    import io

    from woof.runplan import probe_environment as engine_probe

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            engine = engine_probe(readiness=False)
    except Exception as error:  # noqa: BLE001 - an instrument says what it is
        pytest.skip("the installed engine's own probe could not be built "
                    f"here, so its field set is unmeasured: {error}")
    unanswered = (set(engine) - set(document)
                  - set(document.get("engine_fields_not_carried") or {}))
    assert not unanswered, sorted(unanswered)


# ------------------------------------------------------- resolve and estimate


def test_resolve_lists_the_automatic_resolutions_a_reviewer_needs(tmp_path):
    plan = runplan.load_plan(_plan(tmp_path))
    document, cfg, _path = runplan.resolve_plan(plan, require_inputs=False)
    assert document["schema"] == runplan.RESOLVE_SCHEMA
    scopes = {(entry["scope"], entry["key"])
              for entry in document["automatic_resolutions"]}
    assert ("memory", "latitude_bands") in scopes
    assert ("memory", "host_spill_slices") in scopes
    assert ("grid", "quadrature") in scopes
    assert document["configuration"]["config_hash"] == cfg.config_hash
    assert document["inputs_present"] is True


def test_the_resolved_grid_states_the_model_top_the_config_asks_for(tmp_path):
    """The review document's model top, against the TOML that set it.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10:
    `_config_snapshot` read `cfg.a_half_pa[-1]`, the SURFACE end of the
    half-level ladder, where a = p_top * (1 - b) and b = 1, so every
    `gpuwm.run-plan.resolved.v1` document reported a model top of 0.0 Pa --
    for all 55 shipped experiments, in the one document the desktop
    application shows a reviewer before a run is started.

    Held against the configuration's own `vertical.p_top_pa` and against the
    model's own vertical summary, never against a remembered number: an
    experiment free to change its ladder must not be able to make this pass
    by moving the value the test was written around.
    """

    import tomllib

    from woof.globe.config import load_config

    plan = runplan.load_plan(_plan(tmp_path))
    document, cfg, path = runplan.resolve_plan(plan, require_inputs=False)
    stated = document["configuration"]["grid"]["p_top_pa"]
    asked = tomllib.loads(
        Path(path).read_text(encoding="utf-8"))["vertical"]["p_top_pa"]
    assert stated == pytest.approx(asked)
    assert stated == pytest.approx(cfg.vertical.describe()["p_top_pa"])
    assert stated > 0.0

    # ...and it is the top for every experiment that loads, not only this one:
    # the two ends of the ladder differ on all of them, so an index that reads
    # the wrong end is wrong everywhere rather than in one config.
    # ...and the set that does not load is NAMED, not swallowed: the menu
    # measures which shipped experiments the plan route can execute, so a
    # config that stops loading for a new reason fails here instead of
    # quietly leaving the loop.
    doors = runplan.physics_profile_menu()["experiment_doors"]
    refused = set()
    for name in list_configs():
        try:
            other = load_config(config_root() / f"{name}.toml")
        except Exception:  # noqa: BLE001 - named against the menu below
            refused.add(name)
            continue
        assert other.a_half_pa[0] != other.a_half_pa[-1], name
        assert other.a_half_pa[-1] == pytest.approx(0.0), name
    assert refused == {name for name, row in doors.items()
                       if row["run_plan_route"] is None}


def test_the_frame_count_is_the_runners_own_arithmetic(tmp_path):
    """The count a client is shown is the count the run writes.

    Not asserted against a transcription: the plan is RUN in the end-to-end
    test below and the checkpoints on disk are counted there.  Here the
    arithmetic is held to the runner's rule directly.
    """

    from woof.globe.config import load_config

    cfg = load_config(config_root() / f"{SHORT_EXPERIMENT}.toml")
    schedule = runplan._frame_schedule(cfg, SHORT_UNTIL_S)
    assert schedule["total_steps"] == int(round(SHORT_UNTIL_S / cfg.dt_s))
    assert schedule["output_every_steps"] == int(
        round(cfg.output_interval_s / cfg.dt_s))
    assert schedule["checkpoints"] == 3
    # A duration that does not land on the output cadence still writes a final
    # checkpoint, so the count is one higher than the cadence alone gives.
    ragged = runplan._frame_schedule(cfg, cfg.output_interval_s * 1.5)
    assert ragged["checkpoints"] == 3
    assert schedule["restart_step"] is None
    assert schedule["exact"] is True


def test_a_restart_plan_counts_only_the_checkpoints_it_will_write(tmp_path):
    """The restart-aware count, against the files three real runs leave.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10:
    `_frame_schedule` counted every cadence step from 0 and always added the
    cold state, while the runner writes the cold state only on a cold start
    and begins stepping at the restart checkpoint's step.  A T21 restart from
    step 60 and one from step 180 both reported 5 checkpoints, in the
    resolved document, in the estimate and in the forecast `stage_started`
    event's `expected_checkpoints`, and wrote 3 and 1.  The figure did not
    move with the restart point at all.

    Held against RUNS, not against itself: every arm counts the
    `arwen_global_step*.npz` files the run left and the `output_committed`
    events it emitted, and the two restart points are asserted to give
    DIFFERENT answers, which is the property the defect lacked.
    """

    cadence = 10
    inline = _config_with_output_every(cadence)
    until = 20 * 360.0

    def run(name: str, **options) -> tuple[dict, Path]:
        run_dir = tmp_path / name
        plan = _plan(tmp_path, config={"inline": inline},
                     output_root=str(run_dir),
                     run_options={"until_s": until, "overwrite": True,
                                  **options})
        plan = plan.rename(tmp_path / f"{name}.json")
        resolved = json.loads(_cli("run-plan", str(plan), "--resolve",
                                   cwd=tmp_path).stdout)
        estimate = json.loads(_cli("run-plan", str(plan), "--estimate",
                                   cwd=tmp_path).stdout)
        result = _cli("run-plan", str(plan), cwd=tmp_path)
        assert result.returncode == 0, result.stderr[-4000:]
        events = runplan.read_events(run_dir / "events.jsonl")
        return {
            "schedule": resolved["output_schedule"],
            "estimate": estimate["disk"]["checkpoints"],
            "expected": next(
                event["expected_checkpoints"] for event in events
                if event["event"] == "stage_started"
                and event["stage"] == "forecast"),
            "committed": len([
                event for event in events
                if event["event"] == "output_committed"
                and event["kind"] == "checkpoint"]),
            "on_disk": sorted(
                path.name for path in
                run_dir.glob("arwen_global_step*.npz")),
        }, run_dir

    cold, cold_dir = run("cold")
    # 20 steps at a 10-step cadence: the cold state, step 10 and step 20.
    assert cold["on_disk"] == ["arwen_global_step00000000.npz",
                               "arwen_global_step00000010.npz",
                               "arwen_global_step00000020.npz"]
    assert cold["schedule"]["restart_step"] is None

    answers = {}
    for step in (0, cadence):
        arm, _dir = run(
            f"restart{step}",
            restart=str(cold_dir / f"arwen_global_step{step:08d}.npz"))
        written = len(arm["on_disk"])
        assert arm["schedule"]["restart_step"] == step
        assert arm["schedule"]["exact"] is True
        # Every surface that states the count states the one the run wrote.
        assert arm["schedule"]["checkpoints"] == written
        assert arm["schedule"]["checkpoints_from_restart"] == written
        assert arm["estimate"] == written
        assert arm["expected"] == written
        assert arm["committed"] == written
        # ...and no cold state was written into a restarted lineage.
        assert f"arwen_global_step{step:08d}.npz" not in arm["on_disk"]
        answers[step] = written

    # THE NUMBER MOVES WITH THE RESTART POINT.  Equal counts here is exactly
    # the defect: a restart from the cold state writes both cadence frames
    # and one from step 10 writes only the last.
    assert answers[0] == 2 and answers[cadence] == 1, answers

    # A restart file that is not a checkpoint gets NO count rather than one
    # taken from step 0, and the basis says why.
    not_a_checkpoint = _plan(
        tmp_path, config={"inline": inline},
        output_root=str(tmp_path / "unreadable"),
        run_options={"until_s": until, "overwrite": True,
                     "restart": str(tmp_path / "cold.json")})
    document, _cfg, _path = runplan.resolve_plan(
        runplan.load_plan(not_a_checkpoint),
        generate_into=tmp_path / "scratch-unreadable", require_inputs=False)
    schedule = document["output_schedule"]
    assert schedule["checkpoints"] is None
    assert schedule["exact"] is False
    assert "could not be read" in schedule["basis"]

    # ...and a restart file that is not there at all is a declared input
    # with a hole in it, refused before anything is started.
    missing = _plan(
        tmp_path, config={"inline": inline},
        output_root=str(tmp_path / "missing"),
        run_options={"until_s": until, "overwrite": True,
                     "restart": str(tmp_path / "no-such-checkpoint.npz")})
    with pytest.raises(runplan.PlanError) as refusal:
        runplan.resolve_plan(runplan.load_plan(missing),
                             generate_into=tmp_path / "scratch-missing",
                             require_inputs=True)
    assert "no-such-checkpoint.npz" in str(refusal.value)


def _go_plan(tmp_path: Path, name: str, **options) -> Path:
    """A `go` plan with the render options a client actually sends."""

    plan = _plan(tmp_path, route="go",
                 output_root=str(tmp_path / name),
                 run_options={"until_s": SHORT_UNTIL_S, "overwrite": True,
                              **options})
    return plan.rename(tmp_path / f"{name}.json")


@pytest.mark.doors
def test_a_render_product_the_renderer_does_not_carry_is_refused(tmp_path):
    """A slug that cannot draw is answered at review, not after a forecast.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10:
    a `go` plan carrying `render_products: "not_a_product"` resolved at exit
    0 with an empty `warnings` list, the run then spent the statics and the
    whole forecast, and the render stage died at the end with "the render
    stage did not finish (exit 1); its own output says why" -- which names
    neither the slug nor anything a client could act on.  On the T21 case
    that cost 28 s; on `arwen_global_gdas_t255_native_sl_si_24h` it is the
    whole forecast day.

    The list is the renderer's own, so a machine with no staged `rw_wrfbatch`
    cannot answer this at all and says so rather than reporting a pass.
    """

    from woof.globe.doors import find_door

    if find_door("rw_wrfbatch") is None:
        pytest.skip("no rw_wrfbatch is staged here, so the renderer's own "
                    "slug list cannot be asked; `woof fetch-bridges` "
                    "stages it")

    bad = runplan.load_plan(_go_plan(
        tmp_path, "bad-slug", start_date="2026-09-01_00:00:00",
        render_products="not_a_product"))

    # The query mode REPORTS it, with the whole document around the hole.
    document, _cfg, _path = runplan.resolve_plan(
        bad, generate_into=tmp_path / "scratch-bad", require_inputs=False)
    assert document["render"]["unknown_products"] == ["not_a_product"]
    assert document["render"]["products_check_error"] is None
    assert any("not_a_product" in warning["message"]
               and warning["scope"] == "render"
               for warning in document["warnings"]), document["warnings"]

    # ...and the route that starts work REFUSES, naming the slug, what
    # measured it, and the door that lists the alternatives.
    with pytest.raises(runplan.PlanError) as refusal:
        runplan.resolve_plan(bad, generate_into=tmp_path / "scratch-bad-run",
                             require_inputs=True)
    assert "not_a_product" in str(refusal.value)
    assert "--catalog" in str(refusal.value)
    assert "Nothing was started" in str(refusal.value)

    # A slug the catalog carries passes with no render warning at all...
    good = runplan.load_plan(_go_plan(
        tmp_path, "good-slug", start_date="2026-09-01_00:00:00",
        render_products="2m_temperature"))
    clean, _cfg, _path = runplan.resolve_plan(
        good, generate_into=tmp_path / "scratch-good", require_inputs=True)
    assert clean["render"]["unknown_products"] == []
    assert [warning for warning in clean["warnings"]
            if warning["scope"] == "render"] == []

    # ...and a parameterized form is UNCHECKED, never unknown: the renderer
    # resolves `var:<name>` against the tape's own stored variables, so
    # refusing it here would refuse a request the renderer accepts.
    generic = runplan.load_plan(_go_plan(
        tmp_path, "generic-slug", start_date="2026-09-01_00:00:00",
        render_products="2m_temperature,var:T2"))
    document, _cfg, _path = runplan.resolve_plan(
        generic, generate_into=tmp_path / "scratch-generic",
        require_inputs=True)
    assert document["render"]["unknown_products"] == []
    assert document["render"]["unchecked_products"] == ["var:T2"]


def test_a_render_stage_that_will_skip_itself_says_so_at_review(tmp_path):
    """The stage list a reviewer reads is the stage list the run performs.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10:
    a `go` plan with no `start_date` resolved with `render` in its stage list
    and an empty `warnings` list, then completed at exit 0 with the render
    stage skipped and no pictures at all.  A reviewer who approved that plan
    expecting imagery got a green run and an empty gallery.

    Both conditions the stage itself checks are answered here, from the plan
    document alone, so this needs no renderer.
    """

    no_date = runplan.load_plan(_go_plan(tmp_path, "no-date"))
    document, _cfg, _path = runplan.resolve_plan(
        no_date, generate_into=tmp_path / "scratch-no-date",
        require_inputs=False)
    assert "render" in document["plan"]["stages"]
    assert document["render"]["will_run"] is False
    assert "start_date" in document["render"]["skipped_reason"]
    assert any("no pictures" in warning["message"]
               and warning["scope"] == "render"
               for warning in document["warnings"]), document["warnings"]

    off = runplan.load_plan(_go_plan(
        tmp_path, "render-off", start_date="2026-09-01_00:00:00",
        render=False))
    document, _cfg, _path = runplan.resolve_plan(
        off, generate_into=tmp_path / "scratch-off", require_inputs=False)
    assert document["render"]["will_run"] is False
    assert document["render"]["skipped_reason"] == "run_options.render is false"

    # And the sentence is the STAGE's own: a skip reason that drifted from
    # the one the run emits would be a second answer to one question.
    on = runplan.load_plan(_go_plan(
        tmp_path, "render-on", start_date="2026-09-01_00:00:00"))
    document, _cfg, _path = runplan.resolve_plan(
        on, generate_into=tmp_path / "scratch-on", require_inputs=False)
    assert document["render"]["will_run"] is True
    assert document["render"]["skipped_reason"] is None


def test_a_run_with_no_card_reports_absent_and_never_zero(tmp_path):
    """A device figure for a run with no device is not a small figure."""

    plan = runplan.load_plan(_plan(tmp_path))
    document = runplan.estimate_plan(plan)
    assert document["schema"] == runplan.ESTIMATE_SCHEMA
    vram = document["vram"]
    assert vram["backend"] == "numpy"
    assert vram["device_peak_bytes"] is None
    assert vram["card_required_bytes"] is None
    assert vram["host_peak_bytes"] > 0
    assert "null" in vram["device_basis"]
    assert document["wall_time"]["seconds"] is None
    assert document["disk"]["bytes"] is None


def test_resolve_refuses_a_missing_declared_input(tmp_path):
    """The engine's contract: an unavailable input stays a clear refusal."""

    plan = runplan.load_plan(_plan(
        tmp_path, config={"inline": _config_missing_its_analysis()}))
    scratch = tmp_path / "scratch"
    with pytest.raises(runplan.PlanError) as refusal:
        runplan.resolve_plan(plan, generate_into=scratch, require_inputs=True)
    assert "no-such-analysis.grib2" in str(refusal.value)
    assert "Nothing was started" in str(refusal.value)
    # ...and the query mode REPORTS it instead of raising, so a reviewer sees
    # the whole document with the hole in it.
    document, _cfg, _path = runplan.resolve_plan(
        plan, generate_into=scratch, require_inputs=False)
    assert document["inputs_present"] is False
    assert any("no-such-analysis.grib2" in warning["message"]
               for warning in document["warnings"])


# ------------------------------------------------------------- the query CLI


@pytest.mark.parametrize("arguments,schema", [
    (["sources", "--json"], "gpuwm.run-plan.sources.v1"),
    (["run-plan", "--sources"], "gpuwm.run-plan.sources.v1"),
    (["run-plan", "--catalog"], "gpuwm.run-plan.catalog.v1"),
    (["run-plan", "--physics-profiles"], "gpuwm.run-plan.physics-profiles.v1"),
    (["run-plan", "--probe", "--no-readiness"], "gpuwm.run-plan.probe.v1"),
])
def test_every_query_mode_returns_one_json_document(arguments, schema):
    """stdout is the machine channel and carries the document and nothing else.

    Run as a SUBPROCESS on purpose: the registries and the renderer print for
    a person, and the redirect that keeps those lines off stdout is only real
    across a process boundary.
    """

    result = _cli(*arguments)
    assert result.returncode == 0, result.stderr[-2000:]
    document = json.loads(result.stdout)
    assert document["schema"] == schema
    assert document["producer"]["distribution"] == "woof global"
    assert document["producer"]["engine"]["distribution"] == "woof"


def test_the_human_sources_view_is_the_json_documents_view():
    """Two spellings of one document, so the pair cannot drift."""

    listing = _cli("sources")
    assert listing.returncode == 0, listing.stderr[-2000:]
    document = json.loads(_cli("sources", "--json").stdout)
    for row in document["sources"]:
        assert row["source_id"] in listing.stdout
    one = _cli("sources", document["sources"][0]["source_id"], "--json")
    narrowed = json.loads(one.stdout)
    assert len(narrowed["sources"]) == 1
    # The registry's own count survives the narrowing, so a one-row reply is
    # never read as a one-row registry.
    assert narrowed["source_count"] == document["source_count"]
    assert narrowed["requested_source"] == document["sources"][0]["source_id"]
    miss = _cli("sources", "not-a-source")
    assert miss.returncode != 0
    assert "not a registered source id or alias" in miss.stderr


# ------------------------------------------------------------ the kit client


def kit_examples() -> Path:
    """The Integration Kit's examples, INSIDE the installed engine wheel.

    The kit is not vendored here and no environment variable points at it:
    ``woof`` ships it at ``docs/integration-kit/`` beside its own package, so
    the client these tests drive is the one a real integrator was handed by
    the same install this package depends on.  A copy in this repository would
    prove only that the copy agrees with itself.
    """

    import woof

    root = (Path(woof.__file__).resolve().parent.parent
            / "docs" / "integration-kit" / "examples")
    if not (root / "client.py").is_file():
        raise AssertionError(
            f"the installed engine does not carry its integration kit at "
            f"{root}; the client contract these tests hold cannot be checked "
            "against anything but the kit itself")
    return root


@pytest.fixture(scope="session")
def kit_client():
    examples = kit_examples()
    sys.path.insert(0, str(examples))
    try:
        import client as module
    finally:
        sys.path.remove(str(examples))
    return module


@pytest.mark.parametrize("mode,arguments,schema", [
    ("sources", ["sources", "--json"], "gpuwm.run-plan.sources.v1"),
    ("products", ["run-plan", "--catalog"], "gpuwm.run-plan.catalog.v1"),
    ("physics", ["run-plan", "--physics-profiles"],
     "gpuwm.run-plan.physics-profiles.v1"),
    ("inventory", ["run-plan", "--probe", "--no-readiness"],
     "gpuwm.run-plan.probe.v1"),
])
def test_the_shipped_kit_client_queries_this_package(kit_client, mode,
                                                     arguments, schema):
    """The kit's own ``client.query`` against ``python -m arwen_global``.

    Its ``query`` spawns ``[python, "-m", "woof.cli", *arguments]``; the ONE
    token a client changes to drive this package is that module.  So the call
    below is the kit's function, monkeypatched at exactly that token and
    nowhere else -- its schema check, its exit-code check, its
    one-JSON-document rule and its ``ok``/``error`` check all run unaltered.
    """

    import subprocess as kit_subprocess

    real_run = kit_subprocess.run
    seen = {}

    def run(command, **options):
        assert command[1:3] == ["-m", "woof.cli"], command
        seen["command"] = ["-m", "arwen_global", *command[3:]]
        return real_run([command[0], "-m", "arwen_global", *command[3:]],
                        **options)

    kit_client.subprocess.run = run
    try:
        document = kit_client.query(sys.executable, arguments, schema)
    finally:
        kit_client.subprocess.run = real_run
    assert seen["command"][:2] == ["-m", "arwen_global"]
    assert document["schema"] == schema
    assert document["producer"]["distribution"] == "woof global"


def test_the_kit_client_refuses_a_document_of_the_wrong_schema(kit_client):
    """The kit's own guard still fires: this package cannot loosen it."""

    with pytest.raises(kit_client.InterfaceError):
        kit_client.query(sys.executable,
                         ["run-plan", "--probe", "--no-readiness"],
                         "gpuwm.run-plan.catalog.v1")


# ------------------------------------------------------------- the whole run


def test_a_real_plan_run_writes_the_three_durable_documents(tmp_path, kit_client):
    """One real forecast, driven as a client drives it, read back as one does.

    The T21 case on the numpy backend, twenty steps, through the ``experiment``
    route: no renderer, no card, seconds of wall, so this runs wherever the
    package is installed rather than only where the Rust doors are staged.  The
    ``go`` route's five stages and its pictures are held by the test below.

    Everything checked afterwards is checked the way the kit's ``RunReader``
    checks it: the manifest's schema and run identity, the heartbeat bound to
    that run, and an event stream whose sequence is dense and monotonic and
    whose last record is the completion.
    """

    run_dir = tmp_path / "run"
    plan = _plan(
        tmp_path, route="experiment", output_root=str(run_dir),
        run_options={"until_s": TINY_UNTIL_S, "overwrite": True})
    result = _cli("run-plan", str(plan), cwd=tmp_path)
    assert result.returncode == 0, result.stderr[-4000:]

    # stdout is the machine channel: every line is one event and nothing else.
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    mirrored = [json.loads(line) for line in lines]
    assert mirrored, "the run mirrored no events to stdout"

    manifest = json.loads((run_dir / "run-manifest.json").read_text("utf-8"))
    assert manifest["schema"] == "gpuwm.run-manifest.v1"
    assert manifest["route"] == "experiment"
    run_id = manifest["run_id"]

    progress = json.loads(Path(manifest["progress_path"]).read_text("utf-8"))
    assert progress["schema"] == "gpuwm.run-progress/v1"
    assert progress["run_id"] == run_id
    assert progress["status"] == "complete"

    # THE NEWEST CHECKPOINT, not the one before it.  The final checkpoint is
    # submitted to the asynchronous writer inside the last progress callback
    # and lands on disk only when the runner joins that writer, so the
    # heartbeat named the second-to-last one at `status: complete` on every
    # run.  This file is the current-state authority a client resumes from.
    on_disk = sorted(run_dir.glob("arwen_global_step*.npz"))
    assert on_disk, sorted(path.name for path in run_dir.iterdir())
    assert progress["last_checkpoint"] is not None
    assert Path(progress["last_checkpoint"]).resolve() == on_disk[-1].resolve()

    # The fourth document: the small file the other long doors already write,
    # at the path this manifest names, so a workspace polling `go` polls this.
    status = json.loads(Path(manifest["status_path"]).read_text("utf-8"))
    assert status["schema"] == manifest["status_schema"]
    assert status["command"] == "run-plan"
    assert status["state"] == "done"
    assert status["stage_count"] == len(runplan.ROUTES["experiment"].stages)
    assert Path(status["log"]).is_file()

    # ...and NOTHING ELSE writes one here.  A probe that answered a question
    # by creating a second status file and a second log raced this one and
    # left `run-plan-statics.log` in a finished run directory.
    strays = sorted(path.name for path in run_dir.glob("*status*")
                    if path.name != "status.json")
    assert strays == [], strays
    assert not list(run_dir.glob("run-plan-statics*"))

    events = runplan.read_events(manifest["events_path"])
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1))
    assert all(event["schema_version"] == "gpuwm.run-plan.event.v1"
               for event in events)
    assert all(isinstance(event["emitted_unix_ms"], int)
               and not isinstance(event["emitted_unix_ms"], bool)
               for event in events)
    assert all(event["event"] in runplan.EVENT_TAGS for event in events)
    assert events[0]["event"] == "plan_accepted"
    assert events[-1]["event"] == "completed"
    assert [event["sequence"] for event in mirrored] == [
        event["sequence"] for event in events]

    stages = [event["stage"] for event in events
              if event["event"] == "stage_started"]
    assert stages == list(runplan.ROUTES["experiment"].stages)

    # The forecast's progress reached the status surface with a DENOMINATOR.
    # `step 47 of 0` is what a workspace polls and cannot draw.
    log = Path(status["log"]).read_text("utf-8")
    total = next(event["total_steps"] for event in events
                 if event["event"] == "stage_started"
                 and event["stage"] == "forecast")
    assert re.search(r"step \d+ of %d," % total, log), log[-800:]

    # Every committed output is a file that EXISTS: that is what the tag has
    # to mean for a reader that is about to open one.
    committed = [event for event in events
                 if event["event"] == "output_committed"]
    for event in committed:
        assert Path(event["path"]).is_file(), event["path"]
    checkpoints = [event for event in committed
                   if event["kind"] == "checkpoint"]
    # ...and the count is the one --estimate promised before the run.
    estimate = json.loads(_cli("run-plan", str(plan), "--estimate",
                               cwd=tmp_path).stdout)
    assert len(checkpoints) == estimate["disk"]["checkpoints"]
    assert not [event for event in committed if event["kind"] == "picture"], (
        "the experiment route draws nothing and must commit no picture")

    # The capsule the manifest names exists EXACTLY when a run failed, which
    # is what makes its presence an answer.  This run completed.
    assert not Path(manifest["failure_capsule_path"]).exists()

    # AND THE KIT'S OWN READER, unaltered, against this run's three files.
    # Its RunReader is where every rule above actually lives for an
    # integrator: the manifest schema and run identity, the heartbeat bound to
    # that run, the torn-tail rule and the monotonic sequence.  It is run here
    # rather than reimplemented, because a reimplementation would prove only
    # that this file agrees with itself.
    reader = kit_client.RunReader(run_dir / "run-manifest.json")
    snapshot = reader.snapshot()
    assert snapshot["run_id"] == run_id
    assert snapshot["progress"]["status"] == "complete"
    assert [event["sequence"] for event in snapshot["new_events"]] == [
        event["sequence"] for event in events]
    assert snapshot["new_events"][-1]["event"] == "completed"
    # A second snapshot returns nothing new and does not re-read the stream.
    assert reader.snapshot()["new_events"] == []
    assert snapshot["manifest_sha256"] == hashlib.sha256(
        (run_dir / "run-manifest.json").read_bytes()).hexdigest()
    # And its refusal still fires: a manifest that changed under a pinned
    # reader is a reconnect, never a silent continuation.
    (run_dir / "run-manifest.json").write_text(
        (run_dir / "run-manifest.json").read_text("utf-8").replace(
            run_id, "another-run"), encoding="utf-8")
    with pytest.raises(kit_client.InterfaceError):
        reader.snapshot()


def test_a_second_run_into_one_directory_is_a_stream_the_kit_can_read(tmp_path,
                                                                      kit_client):
    """Two runs, one directory, and the kit's own reader on the second.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-09.
    Every durable document but one is replaced whole by its own writer at the
    start of a run; the event stream is opened in APPEND mode and continues
    the sequence it finds.  So a second run into a directory it owns used to
    leave one file carrying two run ids under one climbing sequence, and
    ``RunReader.snapshot()`` -- the kit's, unmodified, which is the client
    this door exists for -- raised ``InterfaceError("Event belongs to another
    run")`` on its FIRST poll.  The client could not attach to that run at
    all, through the package's own documented ``overwrite`` option.
    """

    run_dir = tmp_path / "run"
    plan = _plan(tmp_path, output_root=str(run_dir),
                 run_options={"until_s": TINY_UNTIL_S, "overwrite": True})

    first = _cli("run-plan", str(plan), cwd=tmp_path)
    assert first.returncode == 0, first.stderr[-4000:]
    first_id = json.loads(
        (run_dir / "run-manifest.json").read_text("utf-8"))["run_id"]
    first_events = runplan.read_events(run_dir / "events.jsonl")

    second = _cli("run-plan", str(plan), cwd=tmp_path)
    assert second.returncode == 0, second.stderr[-4000:]
    manifest = json.loads((run_dir / "run-manifest.json").read_text("utf-8"))
    second_id = manifest["run_id"]
    assert second_id != first_id

    # The live stream is the second run's alone, dense from one.
    events = runplan.read_events(manifest["events_path"])
    assert {event["run_id"] for event in events
            if "run_id" in event} == {second_id}
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1))

    # The first run's history was rotated aside, not destroyed, and the
    # manifest names where it went.
    rotated = Path(manifest["superseded_events_path"])
    assert rotated.name == f"events-{first_id}.jsonl"
    assert [event["sequence"] for event in runplan.read_events(rotated)] == [
        event["sequence"] for event in first_events]
    assert any(event["event"] == "warning"
               and str(rotated) in event["message"] for event in events)

    # AND THE KIT'S OWN READER, which is the claim: a client attaching to the
    # second run reads it, where it used to be refused on the first poll.
    reader = kit_client.RunReader(run_dir / "run-manifest.json")
    snapshot = reader.snapshot()
    assert snapshot["run_id"] == second_id
    assert snapshot["progress"]["status"] == "complete"
    assert [event["sequence"] for event in snapshot["new_events"]] == [
        event["sequence"] for event in events]
    assert snapshot["new_events"][-1]["event"] == "completed"


def test_a_run_that_does_not_own_the_directory_is_refused_by_name(tmp_path):
    """Without ``overwrite`` the second run is a refusal, and writes nothing.

    The other half of the rule above: appending to a stream this run does not
    own is the one answer that leaves a client nothing readable, so it is
    refused rather than done quietly, and the previous run's bytes are still
    exactly as they were.
    """

    run_dir = tmp_path / "run"
    owned = _plan(tmp_path, output_root=str(run_dir),
                  run_options={"until_s": TINY_UNTIL_S, "overwrite": True})
    assert _cli("run-plan", str(owned), cwd=tmp_path).returncode == 0

    events_path = run_dir / "events.jsonl"
    before = hashlib.sha256(events_path.read_bytes()).hexdigest()
    manifest_before = (run_dir / "run-manifest.json").read_bytes()
    run_id = json.loads(manifest_before.decode("utf-8"))["run_id"]

    borrowed = _plan(tmp_path, output_root=str(run_dir),
                     run_options={"until_s": TINY_UNTIL_S})
    refused = _cli("run-plan", str(borrowed), cwd=tmp_path)
    assert refused.returncode == 1, refused.stdout[-2000:]
    # The refusal names the breakage, the run whose stream it found, and both
    # remedies.
    assert "Event belongs to another run" in refused.stderr
    assert run_id in refused.stderr
    assert "output_root" in refused.stderr
    assert "run_options.overwrite" in refused.stderr
    # Nothing was started and nothing was touched.
    assert refused.stdout.strip() == ""
    assert hashlib.sha256(events_path.read_bytes()).hexdigest() == before
    assert (run_dir / "run-manifest.json").read_bytes() == manifest_before
    assert not list(run_dir.glob("events-*.jsonl"))


def test_a_failed_plan_ends_in_a_failed_event_and_a_failed_heartbeat(tmp_path):
    """A refusal is reported through the stream, not as a traceback."""

    run_dir = tmp_path / "run"
    plan = _plan(tmp_path, config={"inline": _config_missing_its_analysis()},
                 output_root=str(run_dir),
                 run_options={"overwrite": True})
    result = _cli("run-plan", str(plan), cwd=tmp_path)
    assert result.returncode == 1, result.stdout[-2000:]
    events = runplan.read_events(run_dir / "events.jsonl")
    assert events[-1]["event"] == "failed"
    assert events[-1]["error"]["type"] == "PlanError"
    assert events[-1]["remedy"]
    progress = json.loads((run_dir / "run-progress.json").read_text("utf-8"))
    assert progress["status"] == "failed"
    # A refusal is not a crash: the small file says `refused`, which is the
    # state a workspace shows as an answer rather than as a failure.
    status = json.loads((run_dir / "status.json").read_text("utf-8"))
    assert status["state"] == "refused"
    assert status["reason"]

    # AND THE CAPSULE THE MANIFEST NAMES.  run-manifest.json declares
    # failure_capsule_path and failure_capsule_schema from the first
    # millisecond of a run, so a client branches on them; a field naming a
    # document nothing writes is worse than an absent field.  Nothing in this
    # package wrote one until this test existed.
    manifest = json.loads((run_dir / "run-manifest.json").read_text("utf-8"))
    capsule_path = Path(manifest["failure_capsule_path"])
    assert capsule_path.is_file(), manifest["failure_capsule_path"]
    capsule = json.loads(capsule_path.read_text("utf-8"))
    assert capsule["schema"] == manifest["failure_capsule_schema"]
    assert capsule["run_id"] == manifest["run_id"]
    assert capsule["exception"]["type"] == "PlanError"
    assert capsule["exception"]["message"]
    # The config the run would have used travels with the report, so the
    # support round trip that asks for it is already answered.
    assert "no-such-analysis.grib2" in capsule["config_text"]["text"]
    # No card is claimed by a run that selected none.
    assert capsule["gpu"]["uuid"] == "none"
    # ...and the event points at it, which is how a client finds it without
    # re-reading the manifest.
    assert events[-1]["failure_capsule"] == str(capsule_path)
    assert events[-1]["failure_capsule_error"] is None


def test_the_pages_that_count_the_shipped_experiments_count_them_right():
    """A page that states a number its own package disagrees with.

    THE BREAKAGE THIS PREVENTS, found 2026-09-09 while building the physics
    menu: the README and ``configs_dir``'s own docstring both said fifty-four
    while fifty-five experiments shipped, because the recut that added the
    fifty-fifth moved neither sentence.  A reader who counts
    ``woof global configs`` and gets a different answer has no way to tell
    which of the two is stale, and the menu this file builds is read the same
    way.
    """

    words = {52: "fifty-two", 53: "fifty-three", 54: "fifty-four",
             55: "fifty-five", 56: "fifty-six", 57: "fifty-seven"}
    total = len(list_configs())
    arwen = sum(1 for name in list_configs() if name.startswith("arwen_global_"))
    assert total in words and arwen in words, (
        "this gate's number-word table stops at fifty-seven; extend it")
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert f"{words[total].capitalize()} experiments ship inside" in readme

    from woof.globe import configs_dir

    docstring = configs_dir.__doc__ or ""
    assert f"{words[total].capitalize()} configurations travel" in docstring
    assert f"the {words[arwen]}" in docstring


@pytest.mark.slow
@pytest.mark.doors
def test_the_go_route_runs_every_stage_and_commits_real_pictures(tmp_path):
    """The staged door end to end, including the real Rust renderer.

    Separate from the case above because it needs a staged `rw_wrfbatch` and
    about a minute of wall: CI installs the wheel and stages no Rust door, so a
    single test carrying both claims would either fail there or be excluded
    from CI along with the durable-document claim, which is the half that
    should run everywhere.

    Weather fields are drawn by the Rust renderer and by nothing else, so a
    machine without it cannot answer this question at all and says so rather
    than reporting a pass.
    """

    from woof.globe.doors import find_door

    if find_door("rw_wrfbatch") is None:
        pytest.skip("rw_wrfbatch is not staged here, so the render stage has "
                    "nothing that can draw a weather field; `woof "
                    "fetch-bridges` stages it")

    run_dir = tmp_path / "run"
    plan = _plan(
        tmp_path, route="go", output_root=str(run_dir),
        run_options={
            "until_s": SHORT_UNTIL_S, "overwrite": True,
            "start_date": "2026-09-01_00:00:00",
            "render_products": "2m_temperature",
        })
    result = _cli("run-plan", str(plan), cwd=tmp_path)
    assert result.returncode == 0, result.stderr[-4000:]

    events = runplan.read_events(run_dir / "events.jsonl")
    stages = [event["stage"] for event in events
              if event["event"] == "stage_started"]
    assert stages == list(runplan.ROUTES["go"].stages)
    # The statics stage had nothing to do on a synthetic planet, and said so
    # rather than passing silently.
    skipped = {event["stage"] for event in events
               if event["event"] == "stage_finished" and event.get("skipped")}
    assert "statics" in skipped

    committed = [event for event in events
                 if event["event"] == "output_committed"]
    pictures = [event for event in committed if event["kind"] == "picture"]
    assert pictures, "the render stage committed no pictures"
    for event in committed:
        assert Path(event["path"]).is_file(), event["path"]
    assert any(event["event"] == "first_products_ready" for event in events)
    # The render law's own layout, which the plan route must not invert:
    # <outdir>/<domain>/<product>/<valid-day>/<file>.png
    for event in pictures:
        parts = Path(event["path"]).relative_to(run_dir / "pictures").parts
        assert len(parts) == 4, parts
        assert parts[1] == "2m_temperature"
        assert parts[3].endswith(".png")
    assert events[-1]["event"] == "completed"


@pytest.mark.doors
def test_a_failed_render_names_what_did_not_draw(tmp_path):
    """The failure a client is handed names the cause, not the exit code.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10:
    the render stage raised "the render stage did not finish (exit 1); its
    own output says why", and that sentence is what the terminal `failed`
    event carried.  The renderer's own line -- which names the slug and lists
    every alternative -- reached stderr and the render log only, and those
    are the human output this door exists so a client need not parse.

    The stage is entered directly, with real checkpoints from a real run and
    the real renderer, because `resolve_plan` now refuses an unknown slug
    before a run starts: this holds the OTHER half, that a render failure
    the plan could not foresee still names itself.  `execute_plan` puts
    `str(error)` in the `failed` event, which the failed-plan test above
    holds.
    """

    import hashlib as _hashlib

    from woof.runplan import EventStream
    from woof.supervisor import HEARTBEAT_NAME

    from woof.globe.doors import find_door
    from woof.globe.status import StatusWriter

    if find_door("rw_wrfbatch") is None:
        pytest.skip("rw_wrfbatch is not staged here, so the render stage has "
                    "nothing that can draw a weather field")

    run_dir = tmp_path / "run"
    cold = _plan(tmp_path, route="experiment", output_root=str(run_dir),
                 run_options={"until_s": TINY_UNTIL_S, "overwrite": True})
    assert _cli("run-plan", str(cold), cwd=tmp_path).returncode == 0
    assert sorted(run_dir.glob("arwen_global_step*.npz"))

    config = config_root() / f"{SHORT_EXPERIMENT}.toml"
    document = _plan(
        tmp_path, route="go", output_root=str(run_dir),
        run_options={"until_s": TINY_UNTIL_S, "overwrite": True,
                     "start_date": "2026-09-01_00:00:00",
                     "render_products": "not_a_product"})
    plan = runplan.load_plan(document.rename(tmp_path / "render-failure.json"))
    status = StatusWriter(run_dir, "render-probe",
                          stages=runplan.ROUTES["go"].stages)
    heartbeat = runplan._Heartbeat(
        run_dir / HEARTBEAT_NAME, run_id="render-failure",
        config_sha256=_hashlib.sha256(plan.config_bytes()).hexdigest(),
        started_at_utc=runplan._now_utc())
    with EventStream(tmp_path / "render-events.jsonl") as events:
        runner = runplan._StageRunner(
            plan, events=events, heartbeat=heartbeat, run_dir=run_dir,
            status=status)
        with pytest.raises(RuntimeError) as failure:
            runplan._render_stage(plan, config, runner=runner)
    message = str(failure.value)
    assert "not_a_product" in message, message
    assert "the renderer reported" in message, message


def test_the_physics_menu_says_which_experiments_run_plan_can_execute():
    """A menu offering a plan route that refuses is a menu with dead choices.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10:
    the menu placed the three shipped `global_spectral_*` configs in
    `reference-suite-v1` and asserted every one of the 55 shipped experiments
    was placed, while neither `woof global run-plan` nor `woof global run`
    can execute any of them -- both refuse with "unknown top-level tables:
    global_spectral, ...".  A desktop application building its menu from
    `gpuwm.run-plan.physics-profiles.v1` offered three dead choices.

    Held at the DOOR: every experiment the document says a plan cannot
    execute is put through `run-plan --resolve` and has to refuse there, and
    one it says a plan can execute has to resolve.
    """

    document = runplan.physics_profile_menu()
    doors = document["experiment_doors"]
    assert sorted(doors) == sorted(list_configs())
    # Every profile's dead choices are the same set, read the same way.
    for profile in document["profiles"]:
        assert profile["experiments_not_plannable"] == sorted(
            name for name in profile["experiments"]
            if doors[name]["run_plan_route"] is None)

    refused = sorted(name for name, row in doors.items()
                     if row["run_plan_route"] is None)
    assert refused, "no experiment was measured as unplannable at all"

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        for name in refused:
            plan = root / f"{name}.json"
            plan.write_text(json.dumps({
                "schema": runplan.PLAN_SCHEMA, "name": name,
                "route": "experiment", "config": {"path": name},
                "output_root": str(root / name)}), encoding="utf-8")
            result = _cli("run-plan", str(plan), "--resolve", cwd=root)
            assert result.returncode != 0, (
                f"{name} is offered as unplannable and resolved anyway")
            row = doors[name]
            assert row["reason"], name
            if row["door"] is not None:
                # It runs SOMEWHERE, and the row names where.
                assert "woof.globe.spectral" in row["door"], row

        # ...and the plannable side is not vacuous.
        plan = root / "plannable.json"
        plan.write_text(json.dumps({
            "schema": runplan.PLAN_SCHEMA, "name": SHORT_EXPERIMENT,
            "route": "experiment", "config": {"path": SHORT_EXPERIMENT},
            "output_root": str(root / "plannable")}), encoding="utf-8")
        assert doors[SHORT_EXPERIMENT]["run_plan_route"] == "experiment"
        assert _cli("run-plan", str(plan), "--resolve",
                    cwd=root).returncode == 0


def test_an_inline_config_needs_a_directory_rather_than_leaving_one_behind(tmp_path):
    """A scratch directory belongs to whoever can tell when the caller is done.

    An inline config has to be written before the loader can read it, and
    ``resolve_plan`` cannot know how long its caller will hold the path it
    returns.  It therefore refuses rather than creating a directory in the
    system temp that nothing removes; a run writes into its own run directory,
    where the file is the run's provenance, and a query mode into a scratch
    directory it removes itself.
    """

    plan = runplan.load_plan(_plan(
        tmp_path, config={"inline": _config_missing_its_analysis()}))
    with pytest.raises(runplan.PlanError) as refusal:
        runplan.resolve_plan(plan, require_inputs=False)
    assert "needs a directory" in str(refusal.value)

    # The query mode supplies one and leaves nothing: the document comes back
    # and the config it names is gone with the scratch directory.
    before = set(Path(tempfile.gettempdir()).glob("arwen-global-plan-*"))
    result = _cli("run-plan", str(_plan(
        tmp_path, config={"inline": _config_missing_its_analysis()})),
        "--resolve", cwd=tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout)["schema"] == runplan.RESOLVE_SCHEMA
    assert set(Path(tempfile.gettempdir()).glob(
        "arwen-global-plan-*")) == before


def test_the_mapping_resolver_reports_which_table_answered_when_it_can():
    """``origin`` is a third answer, and its absence is not "the engine's".

    This package will carry copies of the source mappings a published engine
    has not taken, and the resolver that says WHICH copy answered is the one a
    reader needs then: a receipt that records a file name cannot say which of
    two tables opened it.  Where that resolver is in the install, every row
    reports the table; where it is not, `origin` is null, which means UNKNOWN.
    """

    from woof.globe import analysis_initial, sources

    resolver = getattr(analysis_initial, "resolve_analysis_mapping_row", None)
    document = sources.source_inventory()
    for row in document["sources"]:
        assert "origin" in row["mapping"], row["source_id"]
        if row["mapping"]["present"]:
            assert row["mapping"]["resolved"]
            if resolver is None:
                assert row["mapping"]["origin"] is None
            else:
                assert row["mapping"]["origin"] in ("engine", "package", "path")
        else:
            assert row["mapping"]["refusal"]
            assert row["mapping"]["origin"] is None


def test_one_mapping_is_one_row_however_many_tables_name_it():
    """The anchor table and the verification table overlap by design.

    The IFS open-data profile is both this model's background anchor and the
    observation scorecard's reference columns, so a spec an anchor already
    answers for is not printed again as a verification source: one mapping
    under two source ids tells a picker there are two sets of bytes to fetch.
    """

    from woof.globe import sources

    document = sources.source_inventory()
    mappings = [row["mapping"]["id"] for row in document["sources"]]
    assert len(mappings) == len(set(mappings)), mappings
    identifiers = [row["source_id"] for row in document["sources"]]
    assert len(identifiers) == len(set(identifiers)), identifiers


def test_a_stage_that_announces_its_step_count_sizes_the_status_progress_bar():
    """`step 47 of 0` is a numerator with no denominator.

    The forecast stage already tells the event stream how many steps it has.
    The status file is the surface a terminal workspace polls, and it takes
    the same number through `StatusWriter.stage(step_count=...)`, so the two
    are read off one value rather than passed twice and allowed to disagree.
    """

    class _Recorder:
        def __init__(self):
            self.stages = []

        def stage(self, name, *, step_count=0):
            self.stages.append((name, step_count))

        def note(self, line):
            pass

    class _Quiet:
        def preparing(self, stage):
            pass

    class _Silent:
        def emit(self, tag, **fields):
            pass

    recorder = _Recorder()
    runner = runplan._StageRunner(
        plan=None, events=_Silent(), heartbeat=_Quiet(),
        run_dir=Path("."), status=recorder)
    runner.begin("forecast", total_steps=120, dt_s=360.0)
    runner.begin("finalize")
    assert recorder.stages == [("forecast", 120), ("finalize", 0)]
