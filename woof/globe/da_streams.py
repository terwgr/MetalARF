"""Observation streams for the DA door: fetchers with manifests.

A STREAM is a public observation source the door can fetch for a time
window on its own: the fetch is recorded (URL or path, bytes, SHA-256,
the wall it took and how far behind real time the window's end sat when
the bytes arrived), and what it lands is a table the neutral observation
layer decodes (:mod:`woof.globe.obs_table`: a new stream is a
decoder table entry there, and a :data:`STREAM_TABLE` entry here).

The decoders are the tree's: ``iem-asos`` is fetched by ``rw_asos``, the
Rust surface front door (``tools/rustwx/crates/rw-obs``), and its CSV is
read by the ``iem-asos-csv`` table entry; ``local-tables`` is the door's
oldest route, files or URLs already in the obs-table vocabulary (the IGRA2
level table ``tools/arwen_global_igra2_levels_csv.py`` writes, a METAR
cache, an AMV table), recorded with their digests.  Python here fetches
and books; it decodes nothing (the Python boundary).

Interface for the observations lane: register a stream by adding a
:data:`STREAM_TABLE` entry whose factory takes the option dictionary of a
``--stream NAME:key=value;key=value`` spelling and returns an object with
``name``, ``description`` and ``fetch(window_start, window_end, out_dir)
-> list[FetchRecord]``; the files a fetch lands must decode through an
obs-table entry.  Nothing else in the door changes for a new stream.

Every stream of the observation streams module with a decoding door
(:func:`woof.globe.obs_streams.fetch_stream`: ``iem-metar``,
``awc-metar``, ``igra2``, ``ndbc``, ``goes-dmw``, ``cdaac-ro``,
``gnss-ro``) is in the table under its own name through
:class:`ObsStreamsStream`, so ``--stream cdaac-ro`` or ``--stream
igra2:refresh_s=86400`` fetches the stream per window through its Rust
door and lands its ``gpuwm-obs.table.v2`` table.  A stream whose source
is a daily archive (the IGRA2 year-to-date zips, the CDAAC daily
tarballs, the NDBC 45-day files) is not re-fetched every hour:
``refresh_s`` (a per-stream default) keeps the last fetch's table for
that long and the door's own caches make a repeated fetch a digest check
rather than a download.

The analysis fetch for ``fresh`` rides here too
(:func:`fetch_analysis`): the newest published GDAS cycle through the
tree's own fetch door (``woof fetch --source gdas --mode full-file``,
``rw_fetch`` underneath) with its ``fetch-manifest.json``, or a GRIB
already on disk, digested.

Causal bookkeeping (design amendment F, 2026-09-06): every fetch record
carries the window it was taken for (the measurement times it covers),
the instant the bytes were first on disk here (``first_receipt_utc``),
the publication time where the source states one (``publication_utc``,
None otherwise) and a latency CLASS read from the measured latency behind
real time (:func:`classify_latency`, the observation streams module's
vocabulary): ``fast`` inside one hour, ``replay`` inside a day,
``retrospective`` beyond, and ``unverified`` for a table already on disk
whose arrival time nobody measured (a historical case).  The rows
themselves carry ``received_time`` in the ``gpuwm-obs.table.v2`` files
the Rust doors write, and the cycle applies the information cutoff row
by row (:func:`woof.globe.obs_streams.apply_information_cutoff`).  ``fresh`` states its information
cutoff and the cycle records every window's records so a later reader can
tell a fast analysis from a delayed replay.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .obs_table import DECODER_TABLES, fetch_obs, match_decoder

FETCH_MANIFEST_SCHEMA = "gpuwm.arwen-global-da-fetch-manifest/v1"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _utc(moment: dt.datetime) -> dt.datetime:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def _stamp(moment: dt.datetime) -> str:
    return _utc(moment).strftime(TIME_FORMAT)


@dataclass
class FetchRecord:
    """One fetched object of one stream for one window."""

    stream: str
    location: str
    path: str
    bytes: int
    sha256: str
    window_start_utc: str
    window_end_utc: str
    fetched_utc: str
    wall_s: float
    #: Seconds between the window's end and the moment the bytes were on
    #: disk: how far behind real time this stream is reachable.  None for a
    #: table already on disk.
    latency_behind_real_time_s: float | None
    decoder: str | None = None
    extra: dict = field(default_factory=dict)
    #: The source's stated publication instant, when it states one.
    publication_utc: str | None = None

    @property
    def first_receipt_utc(self) -> str:
        """The instant the bytes were first on disk here (the fetch)."""
        return self.fetched_utc

    @property
    def latency_class(self) -> str:
        return classify_latency(self.latency_behind_real_time_s)

    def to_json(self) -> dict[str, object]:
        return {
            "stream": self.stream, "location": self.location, "path": self.path,
            "bytes": int(self.bytes), "sha256": self.sha256,
            "window_start_utc": self.window_start_utc,
            "window_end_utc": self.window_end_utc,
            "fetched_utc": self.fetched_utc, "wall_s": float(self.wall_s),
            "latency_behind_real_time_s": self.latency_behind_real_time_s,
            "first_receipt_utc": self.first_receipt_utc,
            "publication_utc": self.publication_utc,
            "latency_class": self.latency_class,
            "decoder": self.decoder, **({"extra": self.extra} if self.extra else {}),
        }


#: The latency classes of amendments F and H, by measured latency behind
#: real time: the observation streams module's names (its streams carry a
#: class per source; a fetched object carries one per fetch).
LATENCY_CLASSES = ("fast", "replay", "retrospective", "unverified")


def classify_latency(latency_s: float | None) -> str:
    """``fast`` within 3600 s of the window's end, ``replay`` within 86400 s,
    ``retrospective`` beyond; ``unverified`` when nobody measured it (a
    table on disk)."""
    if latency_s is None:
        return "unverified"
    latency_s = float(latency_s)
    if latency_s <= 3600.0:
        return "fast"
    if latency_s <= 86400.0:
        return "replay"
    return "retrospective"


class ObservationStream(Protocol):
    name: str
    description: str

    def fetch(
        self, window_start: dt.datetime, window_end: dt.datetime, out_dir: Path,
    ) -> list[FetchRecord]: ...


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _decoder_of(path: Path) -> str | None:
    """The obs-table entry whose header the file carries, or None."""
    try:
        text, _ = fetch_obs(str(path))
    except (OSError, ValueError):
        return None
    for line in text.splitlines()[:10]:
        found = match_decoder(line.split(","), DECODER_TABLES)
        if found is not None:
            return found.source
    return None


def _url_file_name(location: str) -> str:
    """A file name for a table fetched from a URL: the last path segment
    with its query and fragment dropped and every character outside
    letters, digits, dot, dash and underscore replaced, ``table.csv`` when
    nothing is left.  A query string is part of the request, not of the
    file: the IEM archive's CSV service answers at ``asos.py?station=...``
    and a name carrying ``?``, ``&`` and ``%`` is refused by Windows and
    is no name for a table on any host (found by the 2026-09-06 refutation,
    the first fetch of a public table through this route)."""
    from urllib.parse import urlsplit

    stem = Path(urlsplit(location).path).name
    stem = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in stem).strip("._-")
    return stem or "table.csv"


@dataclass
class LocalTableStream:
    """Observation tables already on disk or at a URL, in the obs-table
    vocabulary; recorded with their digests, fetched from a URL once into
    ``out_dir``.  The ``--obs`` route of the cycle door, as a stream."""

    locations: tuple[str, ...]
    name: str = "local-tables"
    description: str = "observation tables already decoded into the obs-table vocabulary"

    def fetch(self, window_start, window_end, out_dir: Path) -> list[FetchRecord]:
        if not self.locations:
            raise ValueError("local-tables needs at least one table path or URL")
        records = []
        for location in self.locations:
            start = time.perf_counter()
            if "://" in location:
                text, provenance = fetch_obs(location)
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"{provenance['sha256'][:16]}-{_url_file_name(location)}"
                target.write_text(text, encoding="utf-8", newline="\n")
                path = target
                latency = (
                    dt.datetime.now(dt.timezone.utc) - _utc(window_end)
                ).total_seconds()
            else:
                path = Path(location)
                if not path.is_file():
                    raise FileNotFoundError(f"observation table {path} does not exist")
                latency = None
            size, sha = _digest(path)
            records.append(FetchRecord(
                stream=self.name, location=str(location), path=str(path),
                bytes=size, sha256=sha,
                window_start_utc=_stamp(window_start), window_end_utc=_stamp(window_end),
                fetched_utc=_stamp(dt.datetime.now(dt.timezone.utc)),
                wall_s=time.perf_counter() - start,
                latency_behind_real_time_s=latency,
                decoder=_decoder_of(path),
            ))
        return records


@dataclass
class AsosStream:
    """Worldwide surface stations through ``rw_asos`` (the Rust front door
    over the Iowa Environmental Mesonet ASOS archive).

    The station table is frozen once per ``out_dir`` and networks list
    (``rw_asos stations``, digest recorded); each window is fetched with
    ``slack_minutes`` either side (a report is matched to a valid time
    within minutes of it) into one CSV the ``iem-asos-csv`` table entry
    decodes: temperature and dewpoint (Fahrenheit in the file), wind,
    altimeter and elevation.  ``networks`` defaults to every network the
    tree's surface-network table lists; ``bbox`` is W,S,E,N.
    """

    networks: tuple[str, ...] | None = None
    bbox: tuple[float, float, float, float] | None = None
    slack_minutes: int = 60
    name: str = "iem-asos"
    description: str = "ASOS/METAR surface reports worldwide, IEM archive through rw_asos"

    def _networks(self) -> tuple[str, ...]:
        if self.networks:
            return tuple(self.networks)
        from woof.obs.surface_networks import load_table

        table = load_table()
        names = tuple(sorted(network.id for network in table.networks))
        if not names:
            raise ValueError(
                "iem-asos: no networks named and the surface-network table "
                "lists none; pass --stream iem-asos:networks=IA_ASOS,IL_ASOS"
            )
        return names

    def fetch(self, window_start, window_end, out_dir: Path) -> list[FetchRecord]:
        from woof.obs import frontdoor

        door = frontdoor.ASOS
        out_dir.mkdir(parents=True, exist_ok=True)
        networks = self._networks()
        key = hashlib.sha256(
            ("|".join(networks) + "|" + (",".join(map(str, self.bbox)) if self.bbox else "")).encode()
        ).hexdigest()[:12]
        stations = out_dir / f"asos-stations-{key}.json"
        start = time.perf_counter()
        if not stations.is_file():
            freeze = ["--networks", ",".join(networks), "--out", str(stations)]
            if self.bbox:
                freeze += ["--bbox", ",".join(f"{v:g}" for v in self.bbox)]
            door.run("stations", freeze, schema="gpuwm-obs.asos-stations.v1")
        slack = dt.timedelta(minutes=int(self.slack_minutes))
        csv = out_dir / f"asos-{_utc(window_start):%Y%m%dT%H%M}-{_utc(window_end):%Y%m%dT%H%M}.csv"
        fetched = door.run(
            "fetch",
            ["--stations", str(stations),
             "--start", _stamp(_utc(window_start) - slack),
             "--end", _stamp(_utc(window_end) + slack),
             "--out", str(csv)],
            schema="gpuwm-obs.asos-fetch.v1",
        )
        now = dt.datetime.now(dt.timezone.utc)
        size, sha = _digest(csv)
        return [FetchRecord(
            stream=self.name, location=str(fetched.get("url") or fetched.get("archive") or "IEM ASOS archive"),
            path=str(csv), bytes=size, sha256=sha,
            window_start_utc=_stamp(window_start), window_end_utc=_stamp(window_end),
            fetched_utc=_stamp(now), wall_s=time.perf_counter() - start,
            latency_behind_real_time_s=(now - _utc(window_end)).total_seconds(),
            decoder="iem-asos-csv",
            extra={
                "rows": fetched.get("rows"), "stations_table": str(stations),
                "networks": len(networks), "fetch_record_sha256": fetched.get("sha256"),
            },
        )]


#: How long a stream's fetched table serves later windows before the
#: stream is fetched again (seconds); the cadence of the source, not of
#: the cycle.  0 fetches every window.
STREAM_REFRESH_S: dict[str, float] = {
    "iem-metar": 0.0,
    "awc-metar": 0.0,
    "goes-dmw": 0.0,
    "ndbc": 3600.0,
    "igra2": 6 * 3600.0,
    "cdaac-ro": 3600.0,
    "gnss-ro": 24 * 3600.0,
    "wis2": 0.0,
}


@dataclass
class ObsStreamsStream:
    """One stream of :mod:`woof.globe.obs_streams` fetched per
    window through :func:`~woof.globe.obs_streams.fetch_stream`.

    The window is fetched with ``slack_s`` either side (a report inside
    the window is matched at its own time; the slack keeps the boundary
    reports of a replay archive), the table lands under
    ``<out_dir>/<name>/`` with the door's fetch and table records and the
    stream manifest, and the fetch record carries the manifest's measured
    latency behind real time and its class.  Within ``refresh_s`` of the
    last fetch the same table is offered again (the digest unchanged, so
    the cycle decodes it once).  A stream that holds nothing for the
    window (:class:`~woof.globe.obs_streams.StreamEmpty`) is
    recorded as such: an empty fetch with the reason, never a fabricated
    table; a source that will not answer is a failed fetch record with the
    reason (``extra.failed``), the window's other streams analysed.
    """

    stream: str
    options: dict = field(default_factory=dict)
    slack_s: float = 1800.0
    refresh_s: float | None = None
    name: str = ""
    description: str = ""
    _last: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        from .obs_streams import STREAMS

        spec = STREAMS[self.stream]
        self.name = self.name or spec.name
        self.description = self.description or spec.subject
        if self.refresh_s is None:
            self.refresh_s = float(STREAM_REFRESH_S.get(self.stream, 0.0))

    def fetch(self, window_start, window_end, out_dir: Path) -> list[FetchRecord]:
        from .obs_streams import STREAMS, StreamEmpty, fetch_stream

        spec = STREAMS[self.stream]
        out_dir = Path(out_dir)
        now = dt.datetime.now(dt.timezone.utc)
        start = time.perf_counter()
        slack = dt.timedelta(seconds=float(self.slack_s))
        held = self._last
        if held and (now - held["fetched"]).total_seconds() < float(self.refresh_s) \
                and held["start"] <= _utc(window_start) and held["end"] >= _utc(window_end):
            record = held["record"]
            return [FetchRecord(**{**record.__dict__, "window_start_utc": _stamp(window_start),
                                   "window_end_utc": _stamp(window_end), "wall_s": 0.0,
                                   "extra": {**record.extra, "reused_within_refresh_s": float(self.refresh_s)}})]
        fetch_start = _utc(window_start) - slack
        fetch_end = _utc(window_end) + slack
        try:
            fetched = fetch_stream(
                self.stream, _stamp(fetch_start), _stamp(fetch_end), out_dir,
                networks=self.options.get("networks"), stations=self.options.get("stations"),
                satellites=self.options.get("satellites"), missions=self.options.get("missions"),
                seconds=int(self.options["seconds"]) if "seconds" in self.options else None,
                archive=self.options.get("archive"),
            )
        except StreamEmpty as empty:
            return [FetchRecord(
                stream=self.name, location=", ".join(spec.sources), path="", bytes=0, sha256="",
                window_start_utc=_stamp(window_start), window_end_utc=_stamp(window_end),
                fetched_utc=_stamp(now), wall_s=time.perf_counter() - start,
                latency_behind_real_time_s=None, decoder=None,
                extra={"empty": True, "reason": str(empty), "latency_class_of_stream": spec.latency_class},
            )]
        except (RuntimeError, OSError, ValueError) as failure:
            # A source that will not answer (the broker down, the archive
            # host refusing, a door that died) is one stream's failure, not
            # the window's: the record names it and the other streams are
            # analysed.  The receipt's causal record and stream roster carry
            # the record, so the gap is stated, never silent.
            return [FetchRecord(
                stream=self.name, location=", ".join(spec.sources), path="", bytes=0, sha256="",
                window_start_utc=_stamp(window_start), window_end_utc=_stamp(window_end),
                fetched_utc=_stamp(now), wall_s=time.perf_counter() - start,
                latency_behind_real_time_s=None, decoder=None,
                extra={"failed": True, "reason": str(failure)[:2000], "latency_class_of_stream": spec.latency_class},
            )]
        size, sha = _digest(fetched.table)
        latency = fetched.latency_behind_real_time_s
        if latency is None:
            latency = (now - _utc(window_end)).total_seconds()
        record = FetchRecord(
            stream=self.name, location=", ".join(spec.sources), path=str(fetched.table),
            bytes=size, sha256=sha,
            window_start_utc=_stamp(window_start), window_end_utc=_stamp(window_end),
            fetched_utc=_stamp(now), wall_s=time.perf_counter() - start,
            latency_behind_real_time_s=float(latency), decoder="gpuwm-obs.table.v2",
            extra={
                "manifest": str(out_dir / spec.name / "manifest.json"),
                "latency_class_of_stream": spec.latency_class,
                "latency_basis_of_stream": spec.latency_basis,
                "rows": (fetched.manifest.get("table") or {}).get("rows"),
                "counters": fetched.manifest.get("counters"),
                "fetch_window_utc": [_stamp(fetch_start), _stamp(fetch_end)],
            },
        )
        self._last = {"fetched": now, "start": fetch_start, "end": fetch_end, "record": record}
        return [record]


def _obs_streams_factory(stream: str):
    def factory(options: dict[str, str]) -> ObsStreamsStream:
        refresh = options.get("refresh_s")
        slack = options.get("slack_s")
        return ObsStreamsStream(
            stream=stream,
            options={k: v for k, v in options.items() if k not in ("refresh_s", "slack_s")},
            refresh_s=None if refresh is None else float(refresh),
            slack_s=1800.0 if slack is None else float(slack),
        )
    return factory


def _parse_options(text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for part in filter(None, text.split(";")):
        if "=" not in part:
            raise ValueError(
                f"stream option {part!r} is not key=value (spelling: "
                "NAME:key=value;key=value)"
            )
        key, value = part.split("=", 1)
        options[key.strip()] = value.strip()
    return options


def _asos_factory(options: dict[str, str]) -> AsosStream:
    networks = tuple(filter(None, options.get("networks", "").split(","))) or None
    bbox = None
    if "bbox" in options:
        parts = [float(v) for v in options["bbox"].split(",")]
        if len(parts) != 4:
            raise ValueError("iem-asos bbox must be W,S,E,N")
        bbox = (parts[0], parts[1], parts[2], parts[3])
    return AsosStream(
        networks=networks, bbox=bbox,
        slack_minutes=int(options.get("slack_minutes", 60)),
    )


def _local_factory(options: dict[str, str]) -> LocalTableStream:
    paths = tuple(filter(None, options.get("paths", "").split(",")))
    return LocalTableStream(locations=paths)


#: Stream name -> factory of an :class:`ObservationStream` from the
#: option dictionary.  The observations lane adds its streams here (the
#: radiosonde real-time feed, GOES AMVs through rw-sat, GNSS RO, NDBC,
#: the external-analysis anchor); the door needs nothing else.
def _atms_factory(options: dict[str, str]):
    from .radiance_streams import atms_factory

    return atms_factory(options)


def _abi_factory(options: dict[str, str]):
    from .radiance_streams import abi_factory

    return abi_factory(options)


STREAM_TABLE: dict[str, Callable[[dict[str, str]], ObservationStream]] = {
    "iem-asos": _asos_factory,
    "local-tables": _local_factory,
    # The radiance streams (woof.globe.radiance_streams): a fetch
    # that lands and decodes the window's granules through rw_atms and
    # rw_goes, and ``batches(context)`` that hands the ensemble filter the
    # screened PointObs batches with their own operators.
    "atms": _atms_factory,
    "goes-abi": _abi_factory,
}


def radiance_streams(streams) -> list:
    """The streams of a list that hand the filter their own batches
    (``batches(context)``): the radiance streams."""
    return [stream for stream in streams if callable(getattr(stream, "batches", None))]


def _register_obs_streams() -> None:
    """Every stream of the observation streams module with a decoding door
    joins the table under its own name."""
    from .obs_streams import STREAMS

    for name, spec in STREAMS.items():
        if spec.door is not None and spec.decoder_built and not spec.account_gated and name not in STREAM_TABLE:
            STREAM_TABLE[name] = _obs_streams_factory(name)


_register_obs_streams()


def resolve_stream(spec: str) -> ObservationStream:
    """``NAME`` or ``NAME:key=value;key=value`` -> the stream object;
    refused by name for a stream the table does not carry."""
    name, _, options_text = spec.partition(":")
    name = name.strip()
    factory = STREAM_TABLE.get(name)
    if factory is None:
        raise ValueError(
            f"unknown observation stream {name!r}; the table carries "
            f"{sorted(STREAM_TABLE)}.  A new stream is a STREAM_TABLE entry "
            "with a fetcher and an obs-table decoder entry, never a code path"
        )
    return factory(_parse_options(options_text))


def fetch_streams(
    streams, window_start: dt.datetime, window_end: dt.datetime, out_dir: Path,
) -> tuple[list[FetchRecord], Path]:
    """Fetch every stream for the window into ``out_dir`` and write the
    window's manifest beside the files.  Returns ``(records, manifest)``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[FetchRecord] = []
    for stream in streams:
        records.extend(stream.fetch(window_start, window_end, out_dir))
    manifest = out_dir / f"da-fetch-{_utc(window_end):%Y%m%dT%H%M%SZ}.json"
    write_fetch_manifest(manifest, records, window_start, window_end)
    return records, manifest


def write_fetch_manifest(
    path: Path, records: list[FetchRecord], window_start, window_end,
) -> dict[str, object]:
    classes: dict[str, int] = {}
    for record in records:
        classes[record.latency_class] = classes.get(record.latency_class, 0) + 1
    payload = {
        "schema": FETCH_MANIFEST_SCHEMA,
        "window_start_utc": _stamp(window_start),
        "window_end_utc": _stamp(window_end),
        "written_utc": _stamp(dt.datetime.now(dt.timezone.utc)),
        "records": [record.to_json() for record in records],
        "bytes": int(sum(record.bytes for record in records)),
        "latency_classes": classes,
    }
    payload["self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


# ---------------------------------------------------------------------------
# The analysis for ``fresh``
# ---------------------------------------------------------------------------

@dataclass
class AnalysisFetch:
    source: str
    cycle_utc: str
    path: str
    bytes: int
    sha256: str
    manifest: str | None
    location: str | None
    wall_s: float
    fetched_utc: str
    mapping: str


#: The source id the cold start decodes this product through.  Read through
#: `analysis_fetch`, which asks the ENGINE for it first: a second literal
#: here would let the two disagree the day the engine publishes the row, and
#: the fetch receipt would then name a source the run did not use.
from .analysis_fetch import analysis_mapping_id as _analysis_mapping_id

GDAS_ANALYSIS_MAPPING = _analysis_mapping_id()
GDAS_CYCLE_HOURS = (0, 6, 12, 18)


def fetch_analysis(
    out_dir: Path, *, source: str = "gdas", cycle: dt.datetime | None = None,
    grib: str | Path | None = None, engine: str = "auto", progress=print,
) -> AnalysisFetch:
    """The initial analysis: ``grib`` already on disk (digested, its
    cycle stated by the caller), or the newest published GDAS cycle
    through the tree's fetch door (whole-globe f000, the object the cold
    start needs; ``fetch-manifest.json`` beside it carries URL, bytes and
    SHA-256 per file)."""
    out_dir = Path(out_dir)
    start = time.perf_counter()
    if grib is not None:
        path = Path(grib)
        if not path.is_file():
            raise FileNotFoundError(f"analysis GRIB {path} does not exist")
        if cycle is None:
            raise ValueError(
                "an analysis GRIB given by path needs its cycle instant "
                "(--analysis-cycle), because the file name does not state the date"
            )
        size, sha = _digest(path)
        return AnalysisFetch(
            source=source, cycle_utc=_stamp(cycle), path=str(path), bytes=size,
            sha256=sha, manifest=None, location=None,
            wall_s=time.perf_counter() - start,
            fetched_utc=_stamp(dt.datetime.now(dt.timezone.utc)),
            mapping=GDAS_ANALYSIS_MAPPING if source == "gdas" else source,
        )
    if source != "gdas":
        raise ValueError(
            f"fetch_analysis fetches the GDAS analysis; {source!r} is not a "
            "source this door fetches (pass --analysis-grib for a file on disk)"
        )
    from woof import fetch as fetch_door

    if cycle is None:
        cycle = fetch_door.resolve_latest_cycle("gdas", 0)
    cycle_naive = _utc(cycle).replace(tzinfo=None)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_door.fetch_gfs_fullfile(
        cycle=cycle_naive, hours=(0,), area=None, out=out_dir, progress=progress,
        source="gdas", engine=engine, all_levels=True,
    )
    path = out_dir / f"gdas.t{cycle_naive:%H}z.pgrb2.0p25.f000"
    if not path.is_file():
        raise FileNotFoundError(
            f"the GDAS fetch completed but {path} is not there; the manifest is {manifest}"
        )
    size, sha = _digest(path)
    location = None
    try:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        for entry in payload.get("files", []):
            if isinstance(entry, dict) and str(entry.get("name", "")).endswith(path.name):
                location = entry.get("url") or entry.get("source_url")
    except (OSError, ValueError):
        pass
    return AnalysisFetch(
        source="gdas", cycle_utc=_stamp(cycle), path=str(path), bytes=size, sha256=sha,
        manifest=str(manifest), location=location, wall_s=time.perf_counter() - start,
        fetched_utc=_stamp(dt.datetime.now(dt.timezone.utc)), mapping=GDAS_ANALYSIS_MAPPING,
    )


__all__ = [
    "AnalysisFetch",
    "AsosStream",
    "ObsStreamsStream",
    "STREAM_REFRESH_S",
    "FETCH_MANIFEST_SCHEMA",
    "FetchRecord",
    "LATENCY_CLASSES",
    "classify_latency",
    "GDAS_ANALYSIS_MAPPING",
    "LocalTableStream",
    "ObservationStream",
    "STREAM_TABLE",
    "fetch_analysis",
    "fetch_streams",
    "radiance_streams",
    "resolve_stream",
    "write_fetch_manifest",
]
