"""The sources this model can be initialized from and scored against.

``woof global sources`` and ``woof global run-plan --sources`` are the two
spellings of one document, exactly as the engine's pair are: the human table
is a VIEW of the JSON, so the two cannot drift.  The document's schema id is
the engine's ``gpuwm.run-plan.sources.v1``, because a client that already
reads the engine's picker reads this one with no new parser.

WHERE THE ROWS COME FROM, and none of them is written here.

``woof.globe.obs_streams.ANCHOR_SOURCES``
    the two global products this model reads as gridded state: the GDAS
    0.25-degree analysis every shipped experiment initializes from, and the
    ECMWF open-data IFS step-0 object.  Each row already declares its URL
    template, its authority mapping id, its cadence and how its availability
    is measured.

``woof.globe.doctor._NAMED_MAPPINGS``
    the products this package reads for VERIFICATION rather than
    initialization: the GFS cloud-cover field the radiation scorecard grades
    against, the GFS surface state and flux fields the surface-energy
    scorecard reads, and the GDAS microwave columns the ATMS forward operator
    needs.  They are sources in the sense a picker cares about -- bytes from
    a named upstream through a named mapping -- and leaving them out of the
    picker is how a user comes to believe a scorecard needs no data.

``woof.globe.configs``
    which shipped experiments name which mapping, read out of the TOMLs.

A row appended to either table appears in both spellings with no edit here.
That is the arbitrary acceptance test, and
``tests/test_arwen_global_run_plan.py`` grafts a row to prove it.

FIELDS THE ENGINE'S ROWS CARRY AND THESE DO NOT.  The engine's registry
answers a regional preparation chain: a Lambert coverage window, a WRF eta
level mapping, a stock-WRF certification gate, a prepared-cache runner.  None
of those is a question about a global spectral model, and inventing a value
for one would be worse than leaving it out -- a picker would colour a row by
it.  So each row carries ``not_applicable``, naming the engine field and the
one sentence why this package does not answer it, and the document carries
the same list at the top.  Unknown fields are not invented; they are named.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "MAPPING_SUFFIX",
    "NOT_APPLICABLE",
    "REGISTRY_SCHEMA",
    "ROLES",
    "mapping_facts",
    "source_id_for",
    "source_inventory",
    "source_rows",
]

#: What a mapping SPEC may end in and a registry id never does.  Two of the
#: four verification specs are spelled as the mapping file itself, because
#: that is how ``doctor._NAMED_MAPPINGS`` names them for the resolver;
#: ``source_id`` is the field a picker keys and displays, so a row would
#: otherwise render a file name beside five slugs.
MAPPING_SUFFIX = ".mapping.json"

#: This package's own registry id.  Not the engine's
#: ``gpuwm-native-source-adapters-v1``: these rows are a different table with
#: different fields, and reusing that id would tell a reader the engine's
#: adapter contract applies to them.
REGISTRY_SCHEMA = "arwen-global-source-registry-v1"

#: What a row is FOR, which is the field a picker actually branches on here.
#: A row's ``role`` is the PRIMARY one, which is what a table column shows;
#: its ``roles`` names every one that is true of it, because these are not
#: exclusive: the assimilation's background anchor stays the assimilation's
#: background anchor on the day a shipped experiment also initializes from it.
ROLES = {
    "initialization": "gridded state this model can be cold-started from "
                      "(the config's [initial] mode = 'analysis')",
    "background_anchor": "a weak low-pass background constraint on the "
                         "largest scales in the assimilation, never "
                         "pseudo-observations",
    "verification": "a published field this package's scorecards grade "
                    "against; never an input to a forecast",
}

#: Engine registry fields this package does not answer, and why.  Named
#: rather than filled: a picker that read an invented value would draw it.
NOT_APPLICABLE = {
    "coverage": "the model is global; there is no coverage window to "
                "declare, and null here means global rather than unknown",
    "composition_requirement": "there is no prepared-cache composition on "
                               "this line; a config names its analysis file "
                               "directly in [initial]",
    "packaged_profile": "the engine owns the packaged decode profiles; this "
                        "package names an authority mapping id and nothing "
                        "else",
    "runner": "there is no per-source runner: every source reaches the model "
              "through one mapped GRIB2 read",
    "upstream_ingest": "the same reason: one read path, so the field would "
                       "carry one constant on every row",
    "wizard_planable": "this distribution ships no domain wizard; a global "
                       "run has no domain to fit",
    "maturity.level_mapping": "levels are the config's own hybrid A/B ladder, "
                              "not a WRF eta mapping",
    "maturity.cadence_mapping": "the engine's rows put a cadence-policy "
                                "REGISTRY ID here; this registry publishes no "
                                "cadence policy table, so there is no id to "
                                "name and a string built from the cycle "
                                "interval would resolve nowhere.  The fact is "
                                "carried plainly instead, in seconds, by "
                                "forcing_interval_seconds",
    "maturity.stock_wrf_gate": "stock WRF is not a referee for a global "
                               "spectral model; the verification of record "
                               "is observation skill",
    "max_forecast_hour": "how far a run goes is the config's duration_s, not "
                         "a property of the source: the analysis is one "
                         "instant and the forecast length is chosen",
    "default_product": "a row here is keyed on ONE object, spelled out by "
                       "fetch.url_template, so there is no product family to "
                       "choose between and a default would name the only "
                       "member of a set of one",
    "required_products": "the same reason: nothing is composed from several "
                         "product families, so the list would restate "
                         "fetch.url_template on every row",
    "upstream_model_id": "a row here is keyed on an AUTHORITY MAPPING, not "
                         "on an upstream model: two rows read the same GFS "
                         "and a third shares the GDAS the initialization row "
                         "reads, so the field would either duplicate "
                         "source_id or need a per-model table, which is the "
                         "per-model branch this package does not carry",
    "certification_rule": "that rule is the stock-WRF acceptance gate; "
                          "stock WRF is not a referee for a global spectral "
                          "model, and readiness_rule below is the rule these "
                          "rows are actually graded by",
}


def source_id_for(spec: str) -> str:
    """The registry id for one mapping spec: the spec without its suffix.

    DERIVED, never listed: a row appended to either table has to appear with
    no edit here, and an id table would be the per-row edit this registry
    exists without.  The full spec stays where a file name belongs -- in
    ``mapping.id``, in ``maturity.field_mapping`` and in ``aliases``, so
    ``woof global sources <the full spelling>`` still answers.
    """

    return spec[:-len(MAPPING_SUFFIX)] if spec.endswith(MAPPING_SUFFIX) else spec


def mapping_facts(spec: str) -> dict[str, Any]:
    """Which authority table answers one mapping spec, measured right now.

    MEASURED against the install at the moment the document is built, never
    cached and never assumed: the same package on two engines gives two
    answers, and that difference is the single most useful fact in this
    document today.

    ONE RESOLVER, and it says WHICH TABLE answered.
    ``analysis_initial.resolve_analysis_mapping_row`` returns a row carrying
    ``origin``, which is the fact a reader needs now that this package
    carries copies of the rows a published engine has not taken: a receipt
    that records only a file name cannot say which of two copies opened it.
    A refusal is reported rather than raised, because this is a query mode.
    """

    from .analysis_initial import resolve_analysis_mapping_row

    try:
        row = resolve_analysis_mapping_row(spec)
    except Exception as error:  # noqa: BLE001 - a query mode reports
        return {"id": spec, "present": False, "resolved": None,
                "origin": None,
                "refusal": str(error).split(";")[0].strip()}
    return {"id": spec, "present": True, "resolved": row.path.name,
            "origin": row.origin, "refusal": None}


def _users_of(spec: str, config_ids: dict[str, list[str]]) -> list[str]:
    return sorted(config_ids.get(spec, ()))


def _anchor_rows(config_ids: dict[str, list[str]]) -> list[dict[str, Any]]:
    from .obs_streams import ANCHOR_SOURCES

    rows = []
    for anchor in ANCHOR_SOURCES.values():
        users = _users_of(anchor.mapping, config_ids)
        mapping = mapping_facts(anchor.mapping)
        # EVERY ROLE THAT IS TRUE, and the primary one for a table column.
        # Being in this table IS the background-anchor role -- that is what
        # the table is -- and initialization is the separate, measured fact
        # that a shipped experiment's [initial] names this mapping.  Deriving
        # one from the absence of the other made a row stop reporting the
        # assimilation role it still held the moment a config named it.
        roles = {
            "background_anchor":
                "every row in obs_streams.ANCHOR_SOURCES is a weak low-pass "
                "background constraint on the largest scales, which is what "
                "that table is",
        }
        if users:
            roles["initialization"] = (
                "{} shipped experiment(s) name this mapping in their "
                "[initial] table".format(len(users)))
        role = "initialization" if users else "background_anchor"
        rows.append({
            "source_id": anchor.name,
            "aliases": [anchor.mapping],
            "title": anchor.subject,
            "display_name": anchor.subject,
            "source_kind": "deterministic_state",
            "role": role,
            "role_summary": ROLES[role],
            "roles": roles,
            "file_family": "GRIB2",
            "decoder": "the engine's mapped GRIB2 read "
                       "(woof.mapped_source) driven by the authority "
                       "mapping named below",
            "forcing_interval_seconds": float(anchor.cadence_s),
            # The engine's own null for a deterministic source.  Carried
            # rather than named, because null here is an ANSWER: this
            # distribution reads no ensemble product, so a client that
            # branches on the field gets the same verdict it would from the
            # engine's deterministic rows.
            "member_set": None,
            "credentials": {
                "required": False,
                "items": [],
                "summary": "the upstream object is public; this row declares "
                           "no credential requirement",
            },
            "fetch": {
                "kind": "http_object",
                "url_template": anchor.url_template,
                "availability_basis": anchor.availability_basis,
                "route_id": None,
                "refusal": None,
                "door": "woof global obs anchors",
            },
            "mapping": mapping,
            "run_plan": {
                "routes": ["experiment", "go"],
                "intent_supported": False,
                "intent_refusal": "this distribution ships no domain wizard; "
                                  "a plan names an experiment TOML through "
                                  "config.path or config.inline",
            },
            "used_by": users,
            "maturity": {
                "status": ("runnable_mapping" if mapping["present"]
                           else "mapping_absent_from_installed_engine"),
                "runnable": bool(mapping["present"]),
                "field_mapping": anchor.mapping,
                # NULL IS THE ANSWER, and the reason is named in
                # engine_fields_not_carried.  The engine's rows put a REGISTRY
                # ID here, drawn from its own cadence policy table; this
                # registry publishes no such table, so a string formatted
                # from the cycle interval would look like an id and resolve
                # nowhere.  The fact itself is carried plainly, in seconds, by
                # this row's forcing_interval_seconds.
                "cadence_mapping": None,
            },
            "notes": anchor.notes,
            "not_applicable": dict(NOT_APPLICABLE),
        })
    return rows


def _verification_rows(config_ids: dict[str, list[str]]) -> list[dict[str, Any]]:
    from .doctor import _NAMED_MAPPINGS

    from .obs_streams import ANCHOR_SOURCES

    # A mapping an anchor row already answers for is not also a verification
    # row.  The two tables overlap by design -- the IFS open-data profile is
    # both this model's background anchor and the obs scorecard's reference
    # columns -- and printing one mapping twice under two source ids tells a
    # picker there are two sources to fetch.
    claimed = {anchor.mapping for anchor in ANCHOR_SOURCES.values()}
    rows = []
    for spec, used_for in _NAMED_MAPPINGS:
        if spec in claimed:
            continue
        mapping = mapping_facts(spec)
        identifier = source_id_for(spec)
        # The file-name spelling stays reachable, because two of these rows
        # are opened as files by their own readers and a reader who saw that
        # name types it.  It is an ALIAS and never the id: the resolver
        # answers both spellings (`analysis_initial._authority_matches` asks
        # for the file whose name ends at the id and then for the family
        # glob), so the registry can key one kind of thing.
        aliases = sorted({spec, identifier + MAPPING_SUFFIX} - {identifier})
        rows.append({
            "source_id": identifier,
            "aliases": aliases,
            "title": used_for,
            "display_name": used_for,
            "source_kind": "published_field",
            "role": "verification",
            "role_summary": ROLES["verification"],
            "roles": {"verification": "read for " + used_for},
            "file_family": "GRIB2",
            "decoder": "the engine's mapped GRIB2 read "
                       "(woof.mapped_source) driven by this mapping",
            "forcing_interval_seconds": None,
            "member_set": None,
            "credentials": {
                "required": False,
                "items": [],
                "summary": "the upstream object is public; this row declares "
                           "no credential requirement",
            },
            "fetch": {
                "kind": "operator_supplied",
                "url_template": None,
                "availability_basis": None,
                "route_id": None,
                "refusal": "this package publishes no transport for a "
                           "verification field; the scorecard door reads the "
                           "file it is pointed at",
                "door": None,
            },
            "mapping": mapping,
            "run_plan": {
                "routes": [],
                "intent_supported": False,
                "intent_refusal": "a verification field is never an input to "
                                  "a forecast, so no run-plan route reads it",
            },
            "used_by": _users_of(spec, config_ids),
            "maturity": {
                "status": ("runnable_mapping" if mapping["present"]
                           else "mapping_absent_from_installed_engine"),
                "runnable": bool(mapping["present"]),
                # The MAPPING id, the same kind of thing every other row puts
                # here, measured against the engine's own document: its 32
                # rows carry an identifier in this field and none carries a
                # file name.
                "field_mapping": identifier,
                "cadence_mapping": None,
            },
            "notes": "read for " + used_for,
            "not_applicable": dict(NOT_APPLICABLE),
        })
    return rows


def source_rows() -> list[dict[str, Any]]:
    """Every registered source row, initialization first."""

    from .doctor import _config_mapping_ids

    config_ids = _config_mapping_ids()
    rows = _anchor_rows(config_ids) + _verification_rows(config_ids)
    return sorted(rows, key=lambda row: (row["role"] != "initialization",
                                         row["source_id"]))


def source_inventory() -> dict[str, Any]:
    """The ``gpuwm.run-plan.sources.v1`` document for this package."""

    from .runplan import SOURCES_SCHEMA, producer_block, route_summaries

    rows = source_rows()
    producer = producer_block()
    return {
        "schema": SOURCES_SCHEMA,
        "producer": producer,
        # The engine version, at the key the engine's own document puts it,
        # so a picker written for that document reads it here without
        # digging.  MEASURED from the installed distribution through the
        # provenance receipt rather than declared, for the same reason the
        # producer block is: the physics a run integrates is the INSTALLED
        # engine's, and a document that named a version from a table in this
        # package would name the wrong one on the machine that matters.
        "gpuwm_version": producer["engine"]["version"],
        "registry_schema": REGISTRY_SCHEMA,
        "source_count": len(rows),
        "runnable_source_count": sum(
            1 for row in rows if row["maturity"]["runnable"]),
        "readiness_rule": "runnable means the authority mapping this row "
                          "names resolves against the INSTALLED engine; the "
                          "mapping is measured when this document is built, "
                          "so the same package answers differently on two "
                          "engines",
        "roles": dict(ROLES),
        "routes": route_summaries(),
        "sources": rows,
        "engine_fields_not_carried": dict(NOT_APPLICABLE),
    }


# --------------------------------------------------------------------- CLI

#: Registry FIELD names -- never source names -- and the heading each one
#: gets.  A row that grows a field grows a column by being named here; a
#: source appended to a table appears with no edit at all.
_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("id", ("source_id",)),
    ("role", ("role",)),
    ("kind", ("source_kind",)),
    ("family", ("file_family",)),
    ("mapping", ("mapping", "id")),
    ("mapped", ("mapping", "present")),
    ("fetch", ("fetch", "kind")),
    ("experiments", ("used_by",)),
)


def _walk(row, path):
    value = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return str(len(value)) if value else "-"
    return str(value) or "-"


def _resolve(document, wanted: str):
    rows = {str(row["source_id"]): row for row in document["sources"]}
    row = rows.get(wanted)
    if row is not None:
        return row
    for candidate in rows.values():
        if wanted in {str(alias) for alias in candidate.get("aliases", ())}:
            return candidate
    raise ValueError(
        f"{wanted!r} is not a registered source id or alias.\n"
        f"  why: the registry carries {len(rows)} rows and this is not one of "
        "them, so nothing in this package could read, initialize from or "
        "score against it.\n"
        "  remedy: run `woof global sources` with no argument for the whole "
        "list, then name a row's id.")


def _format_listing(document) -> str:
    rows = list(document["sources"])
    cells = [[_cell(_walk(row, path)) for _heading, path in _COLUMNS]
             for row in rows]
    headings = [heading for heading, _path in _COLUMNS]
    widths = [len(heading) for heading in headings]
    for line in cells:
        widths = [max(width, len(value)) for width, value in zip(widths, line)]

    def render(values) -> str:
        return "  " + "  ".join(value.ljust(width) for value, width
                                in zip(values, widths)).rstrip()

    producer = document["producer"]
    lines = [
        "woof global sources: {} registered, {} whose authority mapping the "
        "installed engine carries (registry {}, {} {}, woof {})".format(
            document["source_count"], document["runnable_source_count"],
            document["registry_schema"], producer["distribution"],
            producer["version"], producer["engine"]["version"]),
        render(headings),
    ]
    lines.extend(render(line) for line in cells)
    lines.append("readiness: " + document["readiness_rule"])
    lines.append(
        "one row in full: `woof global sources ID`; the same facts as JSON: "
        "`woof global sources --json` (identical to `woof global run-plan "
        "--sources`)")
    return "\n".join(lines)


def _format_value(value, indent: str) -> list[str]:
    """One registry value, printed by SHAPE and never by field name."""

    if isinstance(value, dict):
        lines = []
        for key, nested in value.items():
            rendered = _format_value(nested, indent + "  ")
            if len(rendered) == 1:
                lines.append(f"{indent}{key}: {rendered[0].strip()}")
            else:
                lines.append(f"{indent}{key}:")
                lines.extend(rendered)
        return lines or [f"{indent}(none)"]
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{indent}(none)"]
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return [f"{indent}{', '.join(str(item) for item in value)}"]
        lines = []
        for item in value:
            lines.extend(_format_value(item, indent + "  "))
        return lines
    parts = str(value).splitlines() or ["-"]
    return [f"{indent}{parts[0]}"] + [f"{indent}  {part.strip()}"
                                      for part in parts[1:]]


def _format_row(row, document) -> str:
    lines = ["woof global sources: {} (registry {}, {} {})".format(
        row["source_id"], document["registry_schema"],
        document["producer"]["distribution"],
        document["producer"]["version"])]
    for key, value in row.items():
        rendered = _format_value(value, "    ")
        if len(rendered) == 1:
            lines.append(f"  {key}: {rendered[0].strip()}")
        else:
            lines.append(f"  {key}:")
            lines.extend(rendered)
    return "\n".join(lines)


def add_sources_arguments(parser) -> None:
    parser.add_argument(
        "source", nargs="?", default=None, metavar="ID",
        help="print ONE row in full, named by its registry id or any alias it "
             "declares (omit for the listing)")
    parser.add_argument(
        "--json", action="store_true",
        help="emit the registry document instead of the table -- the "
             "gpuwm.run-plan.sources.v1 schema, narrowed to the one row when "
             "ID is given")


def sources_main(args) -> int:
    """``woof global sources [ID] [--json]``.

    A VIEW, never a second source of truth: every fact printed here is copied
    out of the document :func:`source_inventory` builds, so the human and
    machine spellings cannot drift.
    """

    import contextlib
    import json
    import sys

    with contextlib.redirect_stdout(sys.stderr):
        document = source_inventory()

    wanted = getattr(args, "source", None)
    if getattr(args, "json", False):
        if wanted is not None:
            # Refuse BEFORE printing: a machine consumer that asked for one
            # row must not receive the whole registry and mistake it for an
            # answer.
            row = _resolve(document, wanted)
            document = dict(document)
            document["sources"] = [row]
            # ``source_count`` stays the REGISTRY's count, because that is
            # what it means; this says the rows were narrowed.
            document["requested_source"] = wanted
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    if wanted is None:
        print(_format_listing(document))
        return 0
    print(_format_row(_resolve(document, wanted), document))
    return 0
