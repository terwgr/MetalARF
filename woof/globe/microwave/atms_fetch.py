"""Fetch one day of ATMS SDR granules from the NOAA JPSS open-data buckets.

The Advanced Technology Microwave Sounder on NOAA-20 and NOAA-21 publishes
its calibrated brightness temperatures as IDPS Sensor Data Records: one
HDF5 file per 32 s granule (12 scans of 96 fields of view, 22 channels)
under ``ATMS-SDR/`` and one geolocation file per granule under
``ATMS-SDR-GEO/`` in the public buckets ``noaa-nesdis-n20-pds`` and
``noaa-nesdis-n21-pds`` (AWS Open Data, anonymous HTTPS).  This module is
the fetcher: it lists a day, downloads every granule pair, hashes every
byte it wrote, and records the manifest the design demands (URL, size,
SHA-256, per-hour volume, publication latency behind real time).

Orchestration only.  Nothing here reads the inside of a file: the HDF5
decode is the ``rw_atms`` bridge (``tools/rustwx/crates/rw-atms``), and
the fetcher's manifest is what the decoder is handed.

The granule name carries its own time stamps, so the latency of the feed
is measured from the file itself:

    SATMS_j02_d20260901_t0000031_e0000347_b19730_c20260901000902460000_oebc_ops.h5
          |    |         |        |        |      |
          |    |         |        |        |      creation instant (UTC, microseconds)
          |    |         |        |        orbit number
          |    |         |        granule end HHMMSS.s
          |    |         granule start HHMMSS.s
          |    day of the granule start
          spacecraft (j01 NOAA-20, j02 NOAA-21)

``created - end`` is the processing latency of the ground segment;
``s3_last_modified - end`` is the latency a consumer of the bucket sees.
Both are recorded per granule and summarised per hour.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import statistics
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

MANIFEST_SCHEMA = "gpuwm-atms-fetch-manifest-v1"

#: Public bucket per spacecraft.  Suomi NPP's ATMS is in a third bucket
#: (``noaa-nesdis-snpp-pds``) but the instrument has been in a degraded
#: mode since 2022; it is deliberately not listed until someone measures
#: its channel health.
BUCKETS: dict[str, str] = {
    "noaa-20": "noaa-nesdis-n20-pds",
    "noaa-21": "noaa-nesdis-n21-pds",
}

SPACECRAFT_CODES: dict[str, str] = {"j01": "noaa-20", "j02": "noaa-21"}

#: Product prefix per role.  The SDR carries the brightness temperatures,
#: the GEO the per-beam geolocation and viewing geometry.
PRODUCTS: dict[str, str] = {"sdr": "ATMS-SDR", "geo": "ATMS-SDR-GEO"}

FILE_PREFIXES: dict[str, str] = {"sdr": "SATMS", "geo": "GATMO"}

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

_GRANULE_RE = re.compile(
    r"^(?P<product>[A-Z]{5})_(?P<spacecraft>j0[12])_d(?P<day>\d{8})"
    r"_t(?P<start>\d{7})_e(?P<end>\d{7})_b(?P<orbit>\d{5})"
    r"_c(?P<created>\d{20})_(?P<origin>[a-z]{4})_(?P<domain>[a-z]{3})\.h5$"
)


class AtmsFetchError(RuntimeError):
    """A listing or download that did not produce what it claimed."""


@dataclass(frozen=True)
class GranuleName:
    product: str
    spacecraft: str
    start: dt.datetime
    end: dt.datetime
    orbit: int
    created: dt.datetime

    @property
    def pair_key(self) -> tuple[str, str, str, int]:
        """What an SDR file and its GEO file share: spacecraft, start,
        end and orbit.  The creation stamps differ between the two."""
        return (
            self.spacecraft,
            self.start.isoformat(timespec="milliseconds"),
            self.end.isoformat(timespec="milliseconds"),
            self.orbit,
        )


def parse_granule_name(filename: str) -> GranuleName:
    match = _GRANULE_RE.match(filename)
    if match is None:
        raise AtmsFetchError(f"not an IDPS ATMS granule name: {filename!r}")
    day = dt.datetime.strptime(match["day"], "%Y%m%d").replace(tzinfo=dt.timezone.utc)

    def stamp(text: str) -> dt.datetime:
        hh, mm, ss, tenth = int(text[0:2]), int(text[2:4]), int(text[4:6]), int(text[6])
        return day + dt.timedelta(hours=hh, minutes=mm, seconds=ss, milliseconds=100 * tenth)

    start = stamp(match["start"])
    end = stamp(match["end"])
    if end < start:
        # The granule crossed midnight; ``d`` is the start day.
        end += dt.timedelta(days=1)
    created = dt.datetime.strptime(match["created"][:14], "%Y%m%d%H%M%S").replace(
        tzinfo=dt.timezone.utc
    ) + dt.timedelta(microseconds=int(match["created"][14:20]))
    return GranuleName(
        product=match["product"],
        spacecraft=match["spacecraft"],
        start=start,
        end=end,
        orbit=int(match["orbit"]),
        created=created,
    )


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    last_modified: dt.datetime
    etag: str


def _open(url: str, timeout_s: float):
    request = urllib.request.Request(url, headers={"User-Agent": "gpuwm-atms-fetch/1"})
    return urllib.request.urlopen(request, timeout=timeout_s)


def list_objects(bucket: str, prefix: str, *, timeout_s: float = 60.0, attempts: int = 4) -> list[S3Object]:
    """Every object under ``prefix`` through ListObjectsV2, following the
    continuation token until the bucket says it is done.  A page that the
    bucket resets is asked for again (``attempts`` times, a growing pause
    between): the download below already retries, and a cycle an hour in
    must not die on a listing transient the next request would not see."""
    objects: list[S3Object] = []
    token: str | None = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = f"https://{bucket}.s3.amazonaws.com/?{urllib.parse.urlencode(query)}"
        last_error: Exception | None = None
        for attempt in range(int(attempts)):
            try:
                with _open(url, timeout_s) as response:
                    body = response.read()
                break
            except Exception as error:  # noqa: BLE001 - retried, then raised with the cause
                last_error = error
                time.sleep(2.0 * (attempt + 1))
        else:
            raise AtmsFetchError(f"listing {prefix} in {bucket} failed after {attempts} attempts: {last_error}")
        root = ET.fromstring(body)
        for contents in root.iter(f"{_S3_NS}Contents"):
            key = contents.findtext(f"{_S3_NS}Key")
            size = contents.findtext(f"{_S3_NS}Size")
            modified = contents.findtext(f"{_S3_NS}LastModified")
            etag = contents.findtext(f"{_S3_NS}ETag") or ""
            if key is None or size is None or modified is None:
                raise AtmsFetchError(f"listing of {prefix} returned an incomplete record")
            objects.append(
                S3Object(
                    key=key,
                    size=int(size),
                    last_modified=dt.datetime.fromisoformat(modified.replace("Z", "+00:00")),
                    etag=etag.strip('"'),
                )
            )
        truncated = root.findtext(f"{_S3_NS}IsTruncated") == "true"
        token = root.findtext(f"{_S3_NS}NextContinuationToken") if truncated else None
        if not token:
            break
    return objects


def day_prefix(product: str, day: dt.date) -> str:
    return f"{PRODUCTS[product]}/{day:%Y/%m/%d}/"


def object_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_size: int, *, timeout_s: float = 120.0,
             attempts: int = 4) -> tuple[str, float]:
    """Stream ``url`` to ``destination`` and return ``(sha256, seconds)``.

    A file already present with the expected size is hashed and kept (a
    resumed day does not re-download what it has).  A short read is a
    refusal, not a truncated file left behind."""
    if destination.exists() and destination.stat().st_size == expected_size:
        return sha256_of(destination), 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(attempts):
        digest = hashlib.sha256()
        started = time.monotonic()
        try:
            with _open(url, timeout_s) as response, partial.open("wb") as handle:
                for block in iter(lambda: response.read(1 << 18), b""):
                    digest.update(block)
                    handle.write(block)
            written = partial.stat().st_size
            if written != expected_size:
                raise AtmsFetchError(
                    f"{url} delivered {written} bytes, the listing said {expected_size}"
                )
            partial.replace(destination)
            return digest.hexdigest(), time.monotonic() - started
        except Exception as error:  # noqa: BLE001 - retried, then raised with the cause
            last_error = error
            time.sleep(2.0 * (attempt + 1))
    raise AtmsFetchError(f"{url} failed after {attempts} attempts: {last_error}")


def _hour_key(instant: dt.datetime) -> str:
    return instant.strftime("%Y-%m-%dT%HZ")


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "count": len(ordered),
    }


def fetch_day(
    satellite: str,
    day: dt.date,
    out_dir: str | Path,
    *,
    workers: int = 8,
    products: tuple[str, ...] = ("sdr", "geo"),
    limit: int | None = None,
    progress=None,
) -> dict[str, object]:
    """Fetch every SDR and GEO granule of ``day`` and write the manifest.

    Returns the manifest (also written to ``out_dir/fetch-manifest.json``
    with ``SHA256SUMS`` beside it).  ``limit`` caps the number of granule
    pairs (a probe of the door, never a day of record) and is recorded.
    """
    if satellite not in BUCKETS:
        raise AtmsFetchError(f"unknown satellite {satellite!r}; known: {sorted(BUCKETS)}")
    bucket = BUCKETS[satellite]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wall_started = time.monotonic()
    listed_at = dt.datetime.now(dt.timezone.utc)

    listings: dict[str, list[S3Object]] = {}
    for product in products:
        listings[product] = list_objects(bucket, day_prefix(product, day))
    listing_seconds = time.monotonic() - wall_started

    # Pair SDR and GEO by the granule identity; an unpaired file is
    # recorded and skipped, because a brightness temperature without its
    # geolocation cannot be placed and a geolocation without its
    # radiances says nothing.
    by_key: dict[tuple, dict[str, S3Object]] = {}
    names: dict[str, GranuleName] = {}
    for product, objects in listings.items():
        for obj in objects:
            name = parse_granule_name(obj.key.rsplit("/", 1)[-1])
            names[obj.key] = name
            by_key.setdefault(name.pair_key, {})[product] = obj
    complete = {key: pair for key, pair in by_key.items() if set(pair) == set(products)}
    unpaired = {
        "|".join(str(part) for part in key): sorted(pair)
        for key, pair in by_key.items() if set(pair) != set(products)
    }
    ordered_keys = sorted(complete)
    if limit is not None:
        ordered_keys = ordered_keys[:limit]

    jobs: list[tuple[str, S3Object]] = []
    for key in ordered_keys:
        for product in products:
            jobs.append((product, complete[key][product]))

    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    def run(job: tuple[str, S3Object]) -> dict[str, object]:
        product, obj = job
        filename = obj.key.rsplit("/", 1)[-1]
        destination = out / product / filename
        url = object_url(bucket, obj.key)
        digest, seconds = download(url, destination, obj.size)
        name = names[obj.key]
        return {
            "product": product,
            "key": obj.key,
            "url": url,
            "path": str(destination.relative_to(out)),
            "size": obj.size,
            "sha256": digest,
            "etag": obj.etag,
            "s3_last_modified": obj.last_modified.isoformat(),
            "granule_start": name.start.isoformat(timespec="milliseconds"),
            "granule_end": name.end.isoformat(timespec="milliseconds"),
            "orbit": name.orbit,
            "created": name.created.isoformat(timespec="microseconds"),
            "processing_latency_s": (name.created - name.end).total_seconds(),
            "publication_latency_s": (obj.last_modified - name.end).total_seconds(),
            "download_seconds": seconds,
        }

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            product, obj = futures[future]
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001 - recorded, the day continues
                failures.append({"key": obj.key, "error": str(error)})
            done += 1
            if progress is not None and (done % 200 == 0 or done == len(jobs)):
                progress(done, len(jobs))

    records.sort(key=lambda record: (record["granule_start"], record["product"]))

    hours: dict[str, dict[str, object]] = {}
    for record in records:
        hour = _hour_key(dt.datetime.fromisoformat(record["granule_start"]))
        slot = hours.setdefault(
            hour,
            {"granules": 0, "bytes": 0, "bytes_by_product": {p: 0 for p in products},
             "_processing": [], "_publication": []},
        )
        slot["bytes"] += record["size"]
        slot["bytes_by_product"][record["product"]] += record["size"]
        if record["product"] == products[0]:
            slot["granules"] += 1
            slot["_processing"].append(record["processing_latency_s"])
            slot["_publication"].append(record["publication_latency_s"])
    for slot in hours.values():
        slot["processing_latency_s"] = _summary(slot.pop("_processing"))
        slot["publication_latency_s"] = _summary(slot.pop("_publication"))

    total_bytes = sum(int(record["size"]) for record in records)
    wall = time.monotonic() - wall_started
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "satellite": satellite,
        "bucket": bucket,
        "day": day.isoformat(),
        "products": {product: PRODUCTS[product] for product in products},
        "listed_at": listed_at.isoformat(timespec="seconds"),
        "listing_seconds": listing_seconds,
        "listed_objects": {product: len(objects) for product, objects in listings.items()},
        "granule_pairs_listed": len(complete),
        "granule_pairs_fetched": len(ordered_keys),
        "limit": limit,
        "unpaired": unpaired,
        "files": records,
        "failures": failures,
        "hours": dict(sorted(hours.items())),
        "total_bytes": total_bytes,
        "wall_seconds": wall,
        "effective_mb_per_s": (total_bytes / 1.0e6 / wall) if wall > 0 else None,
        "workers": workers,
    }
    (out / "fetch-manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{record['sha256']}  {record['path']}\n")
    return manifest


WINDOW_MANIFEST_SCHEMA = "gpuwm-atms-window-manifest-v1"


def fetch_window(
    satellite: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
    cache_dir: str | Path,
    *,
    workers: int = 8,
    products: tuple[str, ...] = ("sdr", "geo"),
    slack_s: float = 60.0,
) -> dict[str, object]:
    """The granule pairs whose scan overlaps ``(window_start - slack,
    window_end + slack]``, listed from the day prefixes the window spans,
    downloaded into ``cache_dir/<day>/<product>/`` (the day layout of
    :func:`fetch_day`, so a day already fetched is found and not fetched
    again), and the window's manifest written beside them
    (``cache_dir/window-<end>.json``: every file with URL, size, SHA-256,
    the bucket's last-modified instant, the granule's own start, end and
    creation stamps, the publication latency, the pairs by path).  A
    listing that finds no pair is a refusal by name, not an empty window."""
    if satellite not in BUCKETS:
        raise AtmsFetchError(f"unknown satellite {satellite!r}; known: {sorted(BUCKETS)}")
    bucket = BUCKETS[satellite]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    start = window_start if window_start.tzinfo else window_start.replace(tzinfo=dt.timezone.utc)
    end = window_end if window_end.tzinfo else window_end.replace(tzinfo=dt.timezone.utc)
    lo = start - dt.timedelta(seconds=float(slack_s))
    hi = end + dt.timedelta(seconds=float(slack_s))
    days = []
    day = lo.date()
    while day <= hi.date():
        days.append(day)
        day += dt.timedelta(days=1)
    wall_started = time.monotonic()
    listings: dict[str, list[S3Object]] = {product: [] for product in products}
    prefixes = []
    for day in days:
        for product in products:
            prefix = day_prefix(product, day)
            prefixes.append(prefix)
            listings[product].extend(list_objects(bucket, prefix))
    listing_seconds = time.monotonic() - wall_started
    by_key: dict[tuple, dict[str, S3Object]] = {}
    names: dict[str, GranuleName] = {}
    for product, objects in listings.items():
        for obj in objects:
            name = parse_granule_name(obj.key.rsplit("/", 1)[-1])
            if name.end < lo or name.start > hi:
                continue
            names[obj.key] = name
            by_key.setdefault(name.pair_key, {})[product] = obj
    complete = {key: pair for key, pair in by_key.items() if set(pair) == set(products)}
    if not complete:
        raise AtmsFetchError(
            f"no complete {satellite} granule pair overlaps {start.isoformat()} to {end.isoformat()} "
            f"in {bucket} ({', '.join(prefixes)}); the window has no ATMS pass to read"
        )
    jobs = [(product, complete[key][product]) for key in sorted(complete) for product in products]
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    def run(job):
        product, obj = job
        filename = obj.key.rsplit("/", 1)[-1]
        name = names[obj.key]
        destination = cache / name.start.strftime("%Y-%m-%d") / product / filename
        digest, seconds = download(object_url(bucket, obj.key), destination, obj.size)
        return {
            "product": product, "key": obj.key, "url": object_url(bucket, obj.key),
            "path": str(destination), "size": obj.size, "sha256": digest, "etag": obj.etag,
            "s3_last_modified": obj.last_modified.isoformat(),
            "granule_start": name.start.isoformat(timespec="milliseconds"),
            "granule_end": name.end.isoformat(timespec="milliseconds"),
            "orbit": name.orbit, "created": name.created.isoformat(timespec="microseconds"),
            "processing_latency_s": (name.created - name.end).total_seconds(),
            "publication_latency_s": (obj.last_modified - name.end).total_seconds(),
            "download_seconds": seconds,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            product, obj = futures[future]
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001 - recorded, then refused below
                failures.append({"key": obj.key, "error": str(error)})
    if failures:
        raise AtmsFetchError(f"{len(failures)} of {len(jobs)} granule files failed: {failures[:3]}")
    records.sort(key=lambda record: (record["granule_start"], record["product"]))
    by_start: dict[tuple, dict[str, str]] = {}
    for record in records:
        key = (record["granule_start"], record["granule_end"], record["orbit"])
        by_start.setdefault(key, {})[record["product"]] = record["path"]
    pairs = [{"start": key[0], "end": key[1], "orbit": key[2], **paths}
             for key, paths in sorted(by_start.items()) if set(paths) == set(products)]
    publication = [r["publication_latency_s"] for r in records if r["product"] == products[0]]
    manifest = {
        "schema": WINDOW_MANIFEST_SCHEMA,
        "satellite": satellite, "bucket": bucket, "prefixes": prefixes,
        "window_start": start.isoformat(timespec="seconds"), "window_end": end.isoformat(timespec="seconds"),
        "slack_s": float(slack_s), "listed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "listing_seconds": listing_seconds,
        "granule_pairs": len(pairs), "files": records, "pairs": pairs,
        "publication_latency_s": _summary(publication),
        "total_bytes": int(sum(int(r["size"]) for r in records)),
        "downloaded_now": int(sum(1 for r in records if r["download_seconds"] > 0.0)),
        "wall_seconds": time.monotonic() - wall_started,
        "workers": workers,
    }
    manifest_path = cache / f"window-{end.strftime('%Y%m%dT%H%M%SZ')}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def granule_pairs(manifest: dict[str, object], root: str | Path) -> list[tuple[Path, Path]]:
    """``(sdr_path, geo_path)`` for every complete pair in a manifest."""
    root = Path(root)
    by_start: dict[tuple, dict[str, Path]] = {}
    for record in manifest["files"]:  # type: ignore[index]
        key = (record["granule_start"], record["granule_end"], record["orbit"])
        by_start.setdefault(key, {})[record["product"]] = root / record["path"]
    pairs = []
    for key in sorted(by_start):
        pair = by_start[key]
        if "sdr" in pair and "geo" in pair:
            pairs.append((pair["sdr"], pair["geo"]))
    return pairs


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="woof global microwave.atms_fetch",
        description="fetch one day of ATMS SDR granule pairs from the NOAA JPSS open-data bucket",
    )
    parser.add_argument("--satellite", choices=sorted(BUCKETS), required=True)
    parser.add_argument("--day", required=True, help="UTC day, YYYY-MM-DD")
    parser.add_argument("--out", required=True, help="output directory (sdr/ and geo/ beneath it)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="fetch only the first N granule pairs (a probe, recorded in the manifest)")
    args = parser.parse_args(argv)
    day = dt.date.fromisoformat(args.day)

    def progress(done: int, total: int) -> None:
        print(f"fetched {done}/{total} files", file=sys.stderr, flush=True)

    manifest = fetch_day(
        args.satellite, day, args.out, workers=args.workers, limit=args.limit, progress=progress
    )
    print(json.dumps({
        "granule_pairs_fetched": manifest["granule_pairs_fetched"],
        "total_bytes": manifest["total_bytes"],
        "wall_seconds": round(float(manifest["wall_seconds"]), 1),
        "failures": len(manifest["failures"]),
        "manifest": str(Path(args.out) / "fetch-manifest.json"),
    }))
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":  # pragma: no cover - the door
    raise SystemExit(_main())
