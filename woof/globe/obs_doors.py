"""Observation front-door rows the published engine does not carry yet.

An observation front door is a TABLE ROW, not a code path: a binary name, the
environment variable that overrides it, what it is a door onto, and the exact
``--abi`` line the Python side was written against.  The engine owns the
mechanism (:class:`woof.obs.frontdoor.FrontDoor` -- the resolution ladder,
the probe, the JSON record contract) and this module owns nothing but five
rows of data.

WHY THE ROWS ARE HERE, and it is two different reasons.

FIVE ROWS THE ENGINE HAS NO BINARY FOR.  IGRA2 radiosondes, atmospheric
motion vectors, marine platforms, GNSS radio occultation and the WIS2 feed
have binaries the engine's 2.7 line does not build, and therefore rows its
``FRONT_DOORS`` table does not carry.  Without them, ``da cycle --stream
igra2`` against a published engine dies on a ``KeyError`` naming nothing,
seconds after the user asked for a stream this package documents.

TWO ROWS THE ENGINE HAS AND THIS PACKAGE MUST NOT USE.  ``rw_asos`` and
``rw_goes`` are in the engine's table, and taking the engine's row for either
is how this package would break in the worst available way.  Measured
2026-09-07 against woof 2.7.0: the engine's ``rw_asos`` offers ``stations``,
``fetch``, ``decode`` and ``verify``, and the surface stream here calls
``networks``, ``table`` and ``awc``; the engine's ``rw_goes`` offers
``list``, ``fetch``, ``cwp``, ``cloud-top`` and ``verify``, and the ABI leg
here calls ``bt``, ``colocate``, ``quicklook`` and ``forward``.  The engine's
row also carries the engine's ``--abi`` line, which is a proper prefix of the
one this package was written against, so taking it would mean a probe that
passes and a run that dies several minutes later on an unknown subcommand.

So the rule is not "engine first" but "each door's publisher first", and
:data:`woof.globe.doors.DOORS` is where that is written down.  A door the
ENGINE publishes takes the engine's row, always and unconditionally.  A door
THIS PACKAGE publishes takes the row here, resolved from the directory
``woof global fetch-doors`` staged it into.  The day the engine takes those
crates onto its own line -- patch item 06 of the carve, which ships the
binaries with them -- one column in ``doors.py`` moves each row back to the
engine and this table stops being consulted for it.
``tests/test_obs_doors_fallback.py`` is what holds the two tables together.

The markers are the engine's own, copied verbatim from the tree this package
was carved from, so a binary that satisfies the engine satisfies this table
and the reverse.  They are not version numbers: each one names the record
contract, so a rebuilt-but-unchanged binary still matches and a binary whose
records changed shape does not.
"""
from __future__ import annotations

__all__ = ["FALLBACK_ROWS", "front_door", "front_doors", "shadowed_rows"]

#: Binary name -> (env var, subject, --abi line).  Tabs inside the marker are
#: the record contract's own separators and are written as escapes so the
#: table survives an editor that trims whitespace.
_ROWS: dict[str, tuple[str, str, str]] = {
    'rw_igra2': (
        'WOOF_RW_IGRA2',
        'the radiosonde front door',
        'gpuwm-obs.igra2-fetch.v1\t'
        'gpuwm-obs.igra2-table.v1\tgpuwm-obs.table.v2\t'
        'surface_pressure_pa\ttemperature_k\tdewpoint_k\t'
        'wind_u_m_s\twind_v_m_s\tlevel_times'
    ),
    'rw_amv': (
        'WOOF_RW_AMV',
        'the atmospheric-motion-vector front door',
        'gpuwm-obs.amv-fetch.v1\tgpuwm-obs.amv-table.v1\t'
        'gpuwm-obs.table.v2\twind_u_m_s\twind_v_m_s\tdqf\t'
        'zenith\tthin'
    ),
    'rw_ndbc': (
        'WOOF_RW_NDBC',
        'the ocean-platform front door',
        'gpuwm-obs.ndbc-fetch.v1\t'
        'gpuwm-obs.ndbc-table.v1\tgpuwm-obs.table.v2\t'
        'surface_pressure_pa\ttemperature_k\tdewpoint_k\t'
        'wind_u_m_s\twind_v_m_s\tanemometer_reduction'
    ),
    'rw_gnssro': (
        'WOOF_RW_GNSSRO',
        'the radio-occultation front door',
        'gpuwm-obs.gnssro-fetch.v1\t'
        'gpuwm-obs.gnssro-table.v1\tgpuwm-obs.table.v2\t'
        'refractivity_n\theight_anchored\tdry_pressure'
    ),
    'rw_wis2': (
        'WOOF_RW_WIS2',
        'the WIS2 subscriber',
        'gpuwm-obs.wis2-subscribe.v1\t'
        'gpuwm-obs.wis2-probe.v1\tmqtt-3.1.1\tmessages\t'
        'payloads\tcoverage\tlatency'
    ),
    # The two the engine also has a row for.  From 2.8.0 on each marker here
    # is the engine's own line, byte for byte: the engine took this
    # package's `networks`, `table` and `awc` (rw_asos) and `bt`,
    # `colocate`, `superobs`, `quicklook` and `forward` (rw_goes) onto its
    # line, so one binary built from the engine's commit answers both
    # tables.  The rw_asos line moved to the engine's v2 surface record,
    # which adds `observation_time` and the one-minute route; the engine's
    # observation readers accept v1 and v2 records alike.  A binary built
    # before that (the 0.1.1 bundle's) prints the v1 line and is refused by
    # the probe here rather than failing ten minutes into a cycle.
    'rw_asos': (
        'WOOF_RW_ASOS',
        'the surface-station front door',
        'gpuwm-obs.asos-surface.v2\tstations\treports\tprovenance\t'
        'observation_time\ttemperature_2m\tdewpoint_2m\twind_speed_10m\t'
        'mslp\tK\tm s-1\tPa\t'
        'gpuwm-obs.asos-table.v1\tgpuwm-obs.table.v2\t'
        'iem-asos-1min'
    ),
    'rw_goes': (
        'WOOF_RW_GOES',
        'the GOES-R ABI front door and clear-sky forward operator',
        'gpuwm-obs.goes-fetch.v1\tgpuwm-obs.goes-cwp.v2\t'
        'gpuwm-obs.goes-cloudtop.v2\tgpuwm-obs.goes-bt.v1\t'
        'gpuwm-da.abi-forward.v1'
    ),
}

#: The row names this module can supply, in the order the streams document.
FALLBACK_ROWS: tuple[str, ...] = tuple(_ROWS)


def _engine_table() -> dict[str, object]:
    """The engine's own front doors, keyed by BINARY name."""

    from woof.obs import frontdoor

    return {door.name: door for door in frontdoor.FRONT_DOORS.values()}


def _published_here() -> frozenset[str]:
    """The binary names THIS package publishes, from the one table that says.

    Read from :data:`woof.globe.doors.DOORS` rather than repeated, so the
    day a door moves to the engine's bundle it moves here in the same edit.
    """

    from .doors import COMPANION_BUNDLE, doors_from_bundle

    return frozenset(door.name for door in doors_from_bundle(COMPANION_BUNDLE))


def shadowed_rows() -> tuple[str, ...]:
    """Rows here that the installed engine carries and this package does not
    publish.

    Should always be empty.  A non-empty answer means this table has started
    to shadow a door whose publisher is the engine, which is the fork this
    module exists to avoid, and the packaging test fails on it.  The two rows
    that deliberately override an engine row are excluded, because for those
    the override IS the contract: their entry in ``doors.DOORS`` names this
    package as the publisher, and the doctor prints both copies.
    """

    engine = _engine_table()
    published_here = _published_here()
    return tuple(name for name in FALLBACK_ROWS
                 if name in engine and name not in published_here)


def front_door(name: str):
    """The front door for a binary name, from whoever publishes that door.

    A door the engine publishes takes the engine's row unconditionally.  A
    door this package publishes takes the row here, because the engine's row
    for the same name names an older record contract and an older binary, and
    a probe that passes against the wrong contract is worse than no probe.

    A name in neither table is a refusal naming every door that does exist,
    because a stream nobody can decode should say so rather than raise a bare
    ``KeyError``.
    """

    from .doors import bind_companion_doors

    engine = _engine_table()
    published_here = _published_here()
    if name in _ROWS and name in published_here:
        # Staged by `fetch-doors` into this package's own directory; the
        # engine's ladder reads the environment variable first, so pointing
        # it there is all the redirection needed and it is a printed one.
        bind_companion_doors()
        from woof.obs.frontdoor import FrontDoor

        env_var, subject, marker = _ROWS[name]
        return FrontDoor(
            name=name, env_var=env_var, subject=subject, abi_marker=marker)
    if name in engine:
        return engine[name]
    if name in _ROWS:
        from woof.obs.frontdoor import FrontDoor

        env_var, subject, marker = _ROWS[name]
        return FrontDoor(
            name=name, env_var=env_var, subject=subject, abi_marker=marker)
    known = ", ".join(sorted(set(engine) | set(_ROWS)))
    raise KeyError(
        f"{name} is not an observation front door on this install; "
        f"the doors are: {known}")


def front_doors() -> dict[str, object]:
    """Every front door reachable here, each row from its own publisher."""

    engine = _engine_table()
    names = sorted(set(engine) | set(FALLBACK_ROWS))
    return {name: front_door(name) for name in names}
